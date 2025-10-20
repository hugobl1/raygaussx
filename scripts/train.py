import logging,random
import torch
import numpy as np
import matplotlib.pyplot as plt
import optix as ox
import cupy as cp
import time

from tqdm import tqdm
from classes import point_cloud,scene

from optix_raycasting import optix_utils as u_ox
from optix_raycasting import render_optix as r_ox

from scripts import test
from utils import general_utils as utilities
from utils.optim_utils import define_optimizer_manager
from utils.loss_utils import l1_loss, ssim
from fused_ssim import fused_ssim
from utils.logger import log_phase

from lpipsPyTorch import lpips

from torch.utils.tensorboard import SummaryWriter

logger = logging.getLogger(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train(config, quiet=True):
  writer = SummaryWriter(log_dir=config.save.tensorboard_logs)
  mempool = cp.get_default_memory_pool()

  learnable_point_cloud=point_cloud.PointCloud(data_type=config.pointcloud.data_type,device=device)
  opt_scene=scene.Scene(config=config,pointcloud=learnable_point_cloud,train_resolution_scales=config.scene.train_resolution_scales,test_resolution_scales=config.scene.test_resolution_scales)

  opt_scene.pointcloud.check_contiguous()

  mean_psnr_train = []
  mean_psnr_test = []
  
  supersampling=(config.training.supersampling_x,config.training.supersampling_y)
  buffer_size=supersampling[0]*supersampling[1]*opt_scene.getTrainCameras()[0].image_height*opt_scene.getTrainCameras()[0].image_width*config.training.max_prim_slice
  hit_prim_idx=cp.zeros((buffer_size),dtype=cp.int32)

  config.training.optimization.position.init_lr = float(opt_scene.cameras_extent)*config.training.optimization.position.init_lr
  config.training.optimization.position.final_lr = float(opt_scene.cameras_extent)*config.training.optimization.position.final_lr
  [optim_manag_positions,optim_manag_rgb,optim_manag_spherical_harmonics,optim_manag_densities,optim_manag_scales,optim_manag_quaternions,optim_manag_sph_gauss_features,optim_manag_bandwidth_sharpness,optim_manag_lobe_axis]=define_optimizer_manager(config.training.optimization,opt_scene.pointcloud,[],[],[],[],[],[],[],[],[])
  opt_scene.pointcloud.setup_optimizers({"xyz":optim_manag_positions,"rgb":optim_manag_rgb,"sh":optim_manag_spherical_harmonics,"density":optim_manag_densities,
                                        "scales":optim_manag_scales,"quaternions":optim_manag_quaternions,
                                        "sph_gauss_features":optim_manag_sph_gauss_features,
                                        "bandwidth_sharpness":optim_manag_bandwidth_sharpness,
                                        "lobe_axis":optim_manag_lobe_axis})

  cfg_train=config.training

  viewpoint_stack = None

  first_iter=0
  if cfg_train.checkpoint>0:
    logger.info("Restoring model from checkpoint: %s", cfg_train.checkpoint_folder)
    first_iter=opt_scene.pointcloud.restore_model(config.training.checkpoint,config.training.checkpoint_folder,config.training.optimization)

  #Compute min and max scales of gaussians for initialization
  opt_scene.pointcloud.compute_3D_filter(cameras=opt_scene.getTrainCameras())
  with torch.no_grad():
    for camera in opt_scene.getTrainCameras():
      dists_cam_gauss = torch.norm(opt_scene.pointcloud.positions-camera.camera_center[None,:],dim=1)
      max_scale = 0.05*dists_cam_gauss.flatten()
      log_max_scale = torch.log(max_scale).repeat(3,1).permute(1,0)
      opt_scene.pointcloud.scales[opt_scene.pointcloud.scales>log_max_scale]=log_max_scale[opt_scene.pointcloud.scales>log_max_scale]
    mask = opt_scene.pointcloud.scales<opt_scene.pointcloud.filter_3D.repeat(1,3)
    opt_scene.pointcloud.scales[mask] = opt_scene.pointcloud.filter_3D.repeat(1,3)[mask]

  start_time = time.time()

  accum_steps = 20
  accum_start = 20000
  accum_counter = 0

  log_phase(logger, "Starting training")
  logger.info("Number of iterations: %d", config.training.n_iters)
  for iter in tqdm(range(first_iter,config.training.n_iters)):
    use_accum = (iter >= accum_start) and (accum_steps > 1)
    if (not use_accum) or (accum_counter == 0):
      opt_scene.pointcloud.optim_managers.zero_grad()
      
    # Pick a random Camera
    if not viewpoint_stack:
      viewpoint_stack = opt_scene.getTrainCameras(config.scene.train_resolution_scales).copy()

    viewpoint_cam = viewpoint_stack.pop(random.randint(0, len(viewpoint_stack)-1))

    gt_image = viewpoint_cam.original_image.cuda()
    if iter==first_iter:
      ################# OPTIX #################
      ctx=u_ox.create_context(log=logger)
      pipeline_options = ox.PipelineCompileOptions(traversable_graph_flags=ox.TraversableGraphFlags.ALLOW_SINGLE_GAS,
                                                num_payload_values=1,
                                                num_attribute_values=0,
                                                exception_flags=ox.ExceptionFlags.NONE,
                                                pipeline_launch_params_variable_name="params")
      module_fwd = u_ox.create_module(ctx, pipeline_options,stage="forward")
      program_grps_fwd = u_ox.create_program_groups(ctx, module_fwd)
      pipeline_fwd = u_ox.create_pipeline(ctx, program_grps_fwd, pipeline_options)
      module_bwd = u_ox.create_module(ctx, pipeline_options,stage="backward")
      program_grps_bwd = u_ox.create_program_groups(ctx, module_bwd)
      pipeline_bwd = u_ox.create_pipeline(ctx, program_grps_bwd, pipeline_options)
      with torch.no_grad():
        cp_positions,cp_scales,cp_quaternion,cp_densities=utilities.torch2cupy(opt_scene.pointcloud.positions,
                                                                              opt_scene.pointcloud.get_scale(),opt_scene.pointcloud.get_normalized_quaternion(),
                                                                              opt_scene.pointcloud.densities)
      L1,L2,L3=u_ox.quaternion_to_rotation(cp_quaternion)
      bboxes = u_ox.compute_ellipsoids_bbox(cp_positions,cp_scales,L1,L2,L3,cp_densities)
      gas = u_ox.create_acceleration_structure(ctx, bboxes)

    update = False
    
    settings=r_ox.RenderOptixSettings(ctx,update, gas, program_grps_fwd, pipeline_fwd,
                  program_grps_bwd, pipeline_bwd, config.scene.dt,config.scene.dynamic_sampling,viewpoint_cam,
                  config.training.max_prim_slice,opt_scene.pointcloud.num_sph_gauss, iteration=iter, jitter=config.training.jitter, rnd_sample=config.training.rnd_sample,supersampling=(config.training.supersampling_x,config.training.supersampling_y),white_background=config.scene.white_background,
                  hit_prim_idx=hit_prim_idx)
    positions,scales,normalized_quaternions,densities,color_features,sph_gauss_features,bandwidth_sharpness,lobe_axis=opt_scene.pointcloud.get_data(normalized_quaternion=not(cfg_train.normalize_quaternion),normalized_lobe_axis=not(cfg_train.normalize_lobe_axis))

    image=r_ox.RenderOptixFunction.apply(positions,scales,normalized_quaternions,densities,color_features,
                                         sph_gauss_features,bandwidth_sharpness,lobe_axis,
                                         settings)

    image_mean=utilities.reduce_supersampling(viewpoint_cam.image_width,viewpoint_cam.image_height,image,supersampling)
    image_mean=image_mean.permute(2,0,1)

    Ll1 = l1_loss(image_mean, gt_image)
    lambda_ssim = config.training.lambda_ssim
    #Fused SSIM for faster computation
    loss_ssim = fused_ssim(image_mean.unsqueeze(0),gt_image.unsqueeze(0))
    # loss_ssim=ssim(image_mean,gt_image)
    train_loss = (1.0 - lambda_ssim) * Ll1 + lambda_ssim * (1.0 - loss_ssim)
    
    ratio_scales_r =(2/(3.14159265359*(3**0.5)))*((scales[:,0]**2+scales[:,1]**2+scales[:,2]**2)**(3/2))/(scales[:,0]*scales[:,1]*scales[:,2])
    ratio_scales_r = torch.clamp(ratio_scales_r,min=10) - 10
    loss_iso_r = ratio_scales_r.mean()
    lambda_iso = config.training.lambda_iso
    train_loss += lambda_iso*loss_iso_r

    if use_accum:
      train_loss = train_loss / float(accum_steps)
    
    train_loss.backward()
    torch.cuda.synchronize()

    if opt_scene.pointcloud.positions.grad is not None:
      opt_scene.pointcloud.accumulate_gradient(opt_scene.pointcloud.positions.grad,viewpoint_cam)
      opt_scene.pointcloud.accumulate_gradient_gaussians_not_visible(opt_scene.pointcloud.positions.grad)

    stepped_this_iter = False
    if use_accum:
      accum_counter += 1
      if accum_counter >= accum_steps:
        opt_scene.pointcloud.optim_managers.step()
        opt_scene.pointcloud.optim_managers.zero_grad()
        accum_counter = 0
        stepped_this_iter = True
    else:
      opt_scene.pointcloud.optim_managers.step()
      opt_scene.pointcloud.optim_managers.zero_grad()

    if stepped_this_iter and iter % 1000 == 0 and iter > 0:
      opt_scene.pointcloud.check_required_grad()

    if iter%100==0:
      writer.add_scalar("Loss/train", train_loss.item(), iter)
      writer.add_scalar("NumPrim", len(opt_scene.pointcloud.positions), iter)
      lrs = opt_scene.pointcloud.optim_managers.get_lrs()
      for name, lr in lrs.items():
        writer.add_scalar(f"LearningRate/{name}", lr, iter)
      
    if (iter>0) and (iter%500==0):
      # print("Number gaussians not visible deleted : ", len(opt_scene.pointcloud.densities)-torch.sum(opt_scene.pointcloud.num_accum_gnv>0).item())
      logger.info(f"[ITER]: {iter} Number gaussians not visible deleted : {len(opt_scene.pointcloud.densities)-torch.sum(opt_scene.pointcloud.num_accum_gnv>0).item()}")
      #if (iter==config.training.n_iters-1):
      #  opt_scene.pointcloud.delete_gaussians_not_seen()
      opt_scene.pointcloud.reset_gradient_accum_gaussians_not_visible()
      # opt_scene.pointcloud.save_model(iter,config.save.models)
    
    # Z-order curve reordering
    if iter%1000==0 and iter>0:
      opt_scene.pointcloud.reorder_zordercurve()
      #Check that tensors require grad
      opt_scene.pointcloud.check_required_grad()
      


    # Unlock color features
    if cfg_train.unlock_color_features:     
      unlock_freq=cfg_train.unlock_freq
      if (iter>0) and (iter%unlock_freq==0)and iter<=(config.training.limit_degree_tot*unlock_freq):
        degree_unlock=iter//unlock_freq
        if degree_unlock<=config.training.limit_degree_sh:
          opt_scene.pointcloud.unlock_spherical_harmonics((degree_unlock+1)**2)
      
      if iter%unlock_freq==0 and iter<=(config.training.limit_degree_tot*unlock_freq):
        degree_unlock=iter//unlock_freq
        if degree_unlock>config.training.limit_degree_sh:
          opt_scene.pointcloud.unlock_spherical_gaussians(2*degree_unlock+1)

      if config.training.limit_degree_sh==-1:
        with torch.no_grad():
          #Ahnilate the effect of rgb colors
          opt_scene.pointcloud.rgb*=0.0
    
    ############################################################################################################
    ################################ Saving and testing part ##################################################
    ############################################################################################################
    #if (iter>0 and iter%10000==0) or (iter==config.training.n_iters-1):
    #if (iter%10000==0) or (iter==config.training.n_iters-1):
    if (iter==config.training.n_iters-1):

      if (iter==config.training.n_iters-1):
        end_time=time.time()
        training_time=end_time-start_time
        opt_scene.pointcloud.save_model(iter,config.save.models)
      tqdm.write(f"[ITER]: {iter} Number points : {len(opt_scene.pointcloud.positions)}")

      ## Train
      PSNR_list_train_scales=[]
      for scale in config.scene.train_resolution_scales:
        PSNR_list_train,gt_images_list,images_list,_=test.inference(hit_prim_idx,opt_scene.pointcloud,opt_scene.getTrainCameras([scale]),config.training.max_prim_slice,config.scene.dt,rnd_sample=config.training.rnd_sample,supersampling=(config.training.supersampling_x,config.training.supersampling_y), white_background=config.scene.white_background,
                                                                    dynamic_sampling=config.scene.dynamic_sampling)
        PSNR_list_train_scales.append(np.mean(PSNR_list_train))
        tqdm.write(f"[ITER]: {iter} \tPSNR Train scale {1.0/scale} (mean): {np.mean(PSNR_list_train):.3f}")
        if not quiet:
          tqdm.write(f"[ITER]: {iter} \tPSNR Train scale {1.0/scale} (min): {np.min(PSNR_list_train):.3f} \tIndex image {np.argmin(PSNR_list_train)}")
          tqdm.write(f"[ITER]: {iter} \tPSNR Train scale {1.0/scale} (max): {np.max(PSNR_list_train):.3f} \tIndex image {np.argmax(PSNR_list_train)}")
        for i in range(len(images_list)):
          plt.imsave(config.save.screenshots+"/train/"+"iter"+str(iter)+"scale"+str(1.0/scale)+"_"+str(i)+"_pred.png",images_list[i])
          plt.imsave(config.save.screenshots+"/train/"+"iter"+str(iter)+"scale"+str(1.0/scale)+"_"+str(i)+"_xgt.png",gt_images_list[i])
      
      ## Test
      if config.scene.eval:
        PSNR_list_test_scales=[]
        SSIM_list_test_scales=[]
        LPIPS_list_test_scales=[]
        for scale in config.scene.test_resolution_scales:
          PSNR_list_test,gt_images_list,images_list,mean_time=test.inference(hit_prim_idx,opt_scene.pointcloud,opt_scene.getTestCameras([scale]),config.training.max_prim_slice,config.scene.dt,rnd_sample=config.training.rnd_sample,supersampling=(config.training.supersampling_x,config.training.supersampling_y), white_background=config.scene.white_background,
                                                                             dynamic_sampling=config.scene.dynamic_sampling)
          PSNR_list_test_scales.append(np.mean(PSNR_list_test))
          ##
          SSIM_test,LPIPS_test=0,0
          for i in range(len(images_list)):
            permute_images=torch.tensor(images_list[i].transpose(2,0,1),dtype=torch.float32,device=device)[None,...]
            permute_gt_images=torch.tensor(gt_images_list[i].transpose(2,0,1),dtype=torch.float32,device=device)[None,...]
            SSIM_test+=ssim(permute_images, permute_gt_images).item()
            LPIPS_test+=lpips(permute_images, permute_gt_images, net_type='vgg').item()
          SSIM_list_test_scales.append(SSIM_test/len(images_list))
          LPIPS_list_test_scales.append(LPIPS_test/len(images_list))
          tqdm.write(f"[ITER]: {iter} \tPSNR Test scale {1.0/scale} (mean): {np.mean(PSNR_list_test):.3f}")
          tqdm.write(f"[ITER]: {iter} \tMean rendering time {1.0/scale} (mean): {mean_time:.2f}")
          if not quiet:
            tqdm.write(f"[ITER]: {iter} \tPSNR Test scale {1.0/scale} (min): {np.min(PSNR_list_test):.3f} \tIndex image {np.argmin(PSNR_list_test)}")
            tqdm.write(f"[ITER]: {iter} \tPSNR Test scale {1.0/scale} (max): {np.max(PSNR_list_test):.3f} \tIndex image {np.argmax(PSNR_list_test)}")
          for i in range(len(images_list)):
            plt.imsave(config.save.screenshots+"/test/"+"iter"+str(iter)+"scale"+str(1.0/scale)+"_"+str(i)+"_pred.png",images_list[i])
            plt.imsave(config.save.screenshots+"/test/"+"iter"+str(iter)+"scale"+str(1.0/scale)+"_"+str(i)+"_xgt.png",gt_images_list[i])

      #Save PSNR and SSIM in a file
      with open(config.save.metrics+'/PSNR_SSIM.txt', 'a') as f:
        f.write(config.scene.source_path+"\n")
        if (iter==config.training.n_iters-1):
          f.write("Training time: "+str(training_time)+"\n")
        f.write(" Iteration: "+str(iter)+"\n")
        f.write("Number points: "+ str(len(opt_scene.pointcloud.positions))+"\n")
        ##Train
        for i,scale in enumerate(config.scene.train_resolution_scales):
          f.write("PSNR Train scale "+str(1.0/scale)+" (mean): "+str(PSNR_list_train_scales[i])+"\n")
        mean_psnr_train.append(np.mean(PSNR_list_train_scales))
        ##Test
        if config.scene.eval:
          for i,scale in enumerate(config.scene.test_resolution_scales):
            f.write("PSNR Test scale "+str(1.0/scale)+" (mean): "+str(PSNR_list_test_scales[i])+"\n")
          ## SSIM and LPIPS
          for i,scale in enumerate(config.scene.test_resolution_scales):
            f.write("SSIM Test scale "+str(1.0/scale)+" (mean): "+str(SSIM_list_test_scales[i])+"\n")
            f.write("LPIPS Test scale "+str(1.0/scale)+" (mean): "+str(LPIPS_list_test_scales[i])+"\n")
          ##
          if len(config.scene.train_resolution_scales)>1:
            f.write("PSNR Train multiscale(mean): "+str(np.mean(PSNR_list_train_scales))+"\n")
          if len(config.scene.test_resolution_scales)>1:
            f.write("PSNR Test multiscale(mean): "+str(np.mean(PSNR_list_test_scales))+"\n")
          f.write("Mean rendering time: "+str(mean_time)+"\n")
          f.write("\n")
          mean_psnr_test.append(np.mean(PSNR_list_test_scales))
  ############################################################################################################
    # Densification
    if iter < cfg_train.densify_until_iter and iter>0:
      if iter > cfg_train.densify_from_iter and iter % cfg_train.densification_interval == 0:
        opt_scene.pointcloud.densify_and_prune(cfg_train.densify_grad_threshold, u_ox.SIGMA_THRESHOLD, opt_scene.cameras_extent, cameras=opt_scene.getTrainCameras())
        mempool.free_all_blocks()      
    
    if (iter%100==0):
      opt_scene.pointcloud.compute_3D_filter(cameras=opt_scene.getTrainCameras())

    with torch.no_grad():
      # opt_scene.pointcloud.densities[opt_scene.pointcloud.densities<0]=0
      opt_scene.pointcloud.clamp_density()
      if cfg_train.normalize_quaternion:
        opt_scene.pointcloud.normalize_quaternion()
      if opt_scene.pointcloud.num_sph_gauss>0 and cfg_train.normalize_lobe_axis:
        opt_scene.pointcloud.normalize_lobe_axis()
      if not quiet:
        if iter%1000==0:
          print("Min minimum size gaussian: ", torch.min(opt_scene.pointcloud.filter_3D))
          print("Max minimum size gaussian: ", torch.max(opt_scene.pointcloud.filter_3D))
          print("Min gaussian scale: ", torch.min(opt_scene.pointcloud.scales))
          print("Max gaussian scale: ", torch.max(opt_scene.pointcloud.scales))
      mask = opt_scene.pointcloud.scales<opt_scene.pointcloud.filter_3D.repeat(1,3)
      opt_scene.pointcloud.scales[mask] = opt_scene.pointcloud.filter_3D.repeat(1,3)[mask]
      #opt_scene.pointcloud.scales[opt_scene.pointcloud.scales<-6.5]=-6.5
      #opt_scene.pointcloud.scales[opt_scene.pointcloud.scales>cfg_train.max_scale]=cfg_train.max_scale
    
  config.training.optimization.position.init_lr = config.training.optimization.position.init_lr/float(opt_scene.cameras_extent)
  config.training.optimization.position.final_lr = config.training.optimization.position.final_lr/float(opt_scene.cameras_extent)
  writer.close()
  if config.scene.eval:
    return mean_psnr_test[-1],np.mean(SSIM_list_test_scales),np.mean(LPIPS_list_test_scales),mean_time
  else:
    return mean_psnr_train[-1],None,None,None
