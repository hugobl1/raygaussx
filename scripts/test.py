import torch
import numpy as np
import optix as ox
import cupy as cp
import logging
from tqdm import tqdm

from utils import general_utils as utilities
from utils.metrics_utils import PSNR
from skimage.metrics import structural_similarity as SSIM

from optix_raycasting import optix_utils as u_ox

from torch.utils.dlpack import to_dlpack

log = logging.getLogger(__name__)

with open('optix_raycasting/cuda_train/forward/vec_math.h') as f:
    code = f.read()
with open('utils/sh_utils.cu') as f:
    code += f.read()
cupy_module = cp.RawModule(code=code)
cuda_forward_sh = cupy_module.get_function('forward_sh')  

def compute_cupy_rgb(camera_center,cupy_positions,cupy_color_features,
                          cp_sph_gauss_features,cp_bandwidth_sharpness,cp_lobe_axis,num_sph_gauss,
                          degree_sh):
  with torch.no_grad():
    num_points=len(cupy_positions)
    block_size=128
    num_blocks=(num_points+block_size-1)//block_size
    cupy_camera_center=cp.from_dlpack(to_dlpack(camera_center.contiguous()))
    cp_colors_rgb=cp.zeros((num_points,3),dtype=cp.float32)
    cuda_forward_sh((num_blocks,),(block_size,),
                      (cupy_camera_center,cupy_positions,cupy_color_features, cp_sph_gauss_features,cp_bandwidth_sharpness,cp_lobe_axis,
                      num_sph_gauss,
                      degree_sh,num_points,cp_colors_rgb))
    return cp_colors_rgb
    
def inference(hit_prim_idx,pointcloud,cam_list,max_prim_slice,dt_step,rnd_sample,supersampling,white_background,dynamic_sampling,culling=False):
    #buffer_size=supersampling[0]*supersampling[1]*cam_list[0].image_height*cam_list[0].image_width*max_prim_slice
    #hit_prim_idx=cp.zeros((buffer_size),dtype=cp.int32)
    with torch.no_grad():
      PSNR_list=[]
      #SSIM_list=[]
      gt_images_list=[]
      images_list=[]

      ################# OPTIX #################
      ctx=u_ox.create_context(log=log)
      pipeline_options = ox.PipelineCompileOptions(traversable_graph_flags=ox.TraversableGraphFlags.ALLOW_SINGLE_GAS,
                                              num_payload_values=1,
                                              num_attribute_values=0,
                                              exception_flags=ox.ExceptionFlags.NONE,
                                              pipeline_launch_params_variable_name="params")
      module = u_ox.create_module(ctx, pipeline_options,stage="test")
      program_grps = u_ox.create_program_groups(ctx, module)
      pipeline = u_ox.create_pipeline(ctx, program_grps, pipeline_options)
      
      cp_positions,cp_scales, cp_quaternions,cp_densities, cp_color_features,cp_sph_gauss_features,cp_bandwidth_sharpness,cp_lobe_axis = utilities.torch2cupy(pointcloud.positions,
                                                                 pointcloud.get_scale(),pointcloud.get_normalized_quaternion(), 
                                                                 pointcloud.get_density(), pointcloud.get_color_features().reshape(-1),
                                                                 pointcloud.sph_gauss_features.reshape(-1),pointcloud.get_bandwidth_sharpness().reshape(-1),
                                                                 pointcloud.get_normalized_lobe_axis().reshape(-1))
      L1,L2,L3=u_ox.quaternion_to_rotation(cp_quaternions)
      bboxes = u_ox.compute_ellipsoids_bbox(cp_positions,cp_scales,L1,L2,L3,cp_densities)
      
      bb_min=bboxes[:,:3].min(axis=0)
      bb_max=bboxes[:,3:].max(axis=0)

      gas = u_ox.create_acceleration_structure(ctx, bboxes)
      sbt = u_ox.create_sbt(program_grps)

      order_sh=int(np.sqrt(pointcloud.spherical_harmonics.shape[2]+1).item()-1)

      #for cam in tqdm(cam_list):
      mean_time=[]
      for cam in cam_list:
        if culling:
          ################### Start culling ###################
          projected_points=pointcloud.project_points(cam)
          depth=projected_points[:,2]+1e-6
          projected_points[:,0]=projected_points[:,0]/depth
          projected_points[:,1]=projected_points[:,1]/depth
          mask=depth>0
          projected_points[:,2]=projected_points[:,2]/depth

          view_points_x=projected_points[:,0]/torch.tan(torch.tensor(cam.FoVx / 2))
          view_points_y=projected_points[:,1]/torch.tan(torch.tensor(cam.FoVy / 2))
          mask =  mask&(view_points_x >= -1.3) & (view_points_x <= 1.3) & (view_points_y >= -1.3) & (view_points_y <= 1.3)
          xyz,color_features,densities,scales,quaternions,sph_gauss_features,bandwidth_sharpness,lobe_axis=pointcloud.select_inside_mask(mask)
          cp_positions,cp_scales,cp_quaternions, cp_densities, cp_color_features,cp_sph_gauss_features,cp_bandwidth_sharpness,cp_lobe_axis = utilities.torch2cupy(xyz,scales,quaternions,densities,color_features.reshape(-1),
                                                                                                    sph_gauss_features.reshape(-1),bandwidth_sharpness.reshape(-1),lobe_axis.reshape(-1))

          ################### End culling ################### 
          ################ Update new scene ################
          L1,L2,L3=u_ox.quaternion_to_rotation(cp_quaternions)
          bboxes = u_ox.compute_ellipsoids_bbox(cp_positions,cp_scales,L1,L2,L3,cp_densities)  
          bb_min=bboxes[:,:3].min(axis=0)
          bb_max=bboxes[:,3:].max(axis=0)

          gas = u_ox.create_acceleration_structure(ctx, bboxes)
          sbt = u_ox.create_sbt(program_grps)

          #Check memory is contiguous
          pointcloud.check_contiguous()
          ###############################################

        start=torch.cuda.Event(enable_timing=True)
        end=torch.cuda.Event(enable_timing=True)
        start.record()
        cp_color_features_rgb=compute_cupy_rgb(cam.camera_center,cp_positions,cp_color_features,
                        cp_sph_gauss_features,cp_bandwidth_sharpness,cp_lobe_axis,pointcloud.num_sph_gauss,
                        order_sh)


        ray_colors= u_ox.launch_pipeline_test(pipeline, sbt, gas,bb_min,bb_max,dt_step,dynamic_sampling,cam,
                                          cp_densities,cp_color_features_rgb,cp_positions,
                                          cp_scales, cp_quaternions,
                                          max_prim_slice=max_prim_slice,
                                          rnd_sample=rnd_sample,supersampling=supersampling,white_background=white_background,hit_prim_idx=hit_prim_idx)

        end.record()
        torch.cuda.synchronize()
        time_render=start.elapsed_time(end)
        mean_time.append(time_render)

        ray_colors=torch.from_dlpack(ray_colors)
        ray_colors_mean=utilities.reduce_supersampling(cam.image_width,cam.image_height,ray_colors,supersampling)

        ray_colors_numpy = ray_colors_mean.detach().cpu().numpy().clip(0,1)
        gt_numpy = cam.original_image.permute(1,2,0).detach().cpu().numpy().clip(0,1)
        psnr=PSNR(ray_colors_numpy,gt_numpy)
        
        #ssim=SSIM(ray_colors_mean).detach().cpu().numpy(), cam.original_image.permute(1,2,0).detach().cpu().numpy(),channel_axis=2,data_range=1)
        PSNR_list.append(psnr)
        #SSIM_list.append(ssim)
        
        gt_images_list.append(gt_numpy)
        images_list.append(ray_colors_numpy)
      return PSNR_list, gt_images_list, images_list, np.mean(mean_time)
    
def render(pointcloud,cam_list,max_prim_slice,dt_step,rnd_sample,supersampling,white_background,dynamic_sampling):
    buffer_size=supersampling[0]*supersampling[1]*cam_list[0].image_height*cam_list[0].image_width*max_prim_slice
    hit_prim_idx=cp.zeros((buffer_size),dtype=cp.int32)
    with torch.no_grad():
      images_list=[]

      ################# OPTIX #################
      ctx=u_ox.create_context(log=log)    
      pipeline_options = ox.PipelineCompileOptions(traversable_graph_flags=ox.TraversableGraphFlags.ALLOW_SINGLE_GAS,
                                              num_payload_values=1,
                                              num_attribute_values=0,
                                              exception_flags=ox.ExceptionFlags.NONE,
                                              pipeline_launch_params_variable_name="params")
      module = u_ox.create_module(ctx, pipeline_options,stage="test")
      program_grps = u_ox.create_program_groups(ctx, module)
      pipeline = u_ox.create_pipeline(ctx, program_grps, pipeline_options)
      cp_positions,cp_scales, cp_quaternions,cp_densities, cp_color_features,cp_sph_gauss_features,cp_bandwidth_sharpness,cp_lobe_axis = utilities.torch2cupy(pointcloud.positions,
                                                                 pointcloud.get_scale(),pointcloud.get_normalized_quaternion(), 
                                                                 pointcloud.get_density(), pointcloud.get_color_features().reshape(-1),
                                                                 pointcloud.sph_gauss_features.reshape(-1),pointcloud.get_bandwidth_sharpness().reshape(-1),
                                                                 pointcloud.get_normalized_lobe_axis().reshape(-1))

      L1,L2,L3=u_ox.quaternion_to_rotation(cp_quaternions)
      bboxes = u_ox.compute_ellipsoids_bbox(cp_positions,cp_scales,L1,L2,L3,cp_densities)
      
      bb_min=bboxes[:,:3].min(axis=0)
      bb_max=bboxes[:,3:].max(axis=0)

      gas = u_ox.create_acceleration_structure(ctx, bboxes)
      sbt = u_ox.create_sbt(program_grps)

      order_sh=int(np.sqrt(pointcloud.spherical_harmonics.shape[2]+1).item()-1)
      #Check memory is contiguous
      pointcloud.check_contiguous()

      mean_time=[]
      # mean_time_next=[]
      for cam in tqdm(cam_list):
          start=torch.cuda.Event(enable_timing=True)
          end=torch.cuda.Event(enable_timing=True)
          start.record()
          cp_color_features_rgb=compute_cupy_rgb(cam.camera_center,cp_positions,cp_color_features,
                          cp_sph_gauss_features,cp_bandwidth_sharpness,cp_lobe_axis,pointcloud.num_sph_gauss,
                          order_sh)
          ray_colors= u_ox.launch_pipeline_test(pipeline, sbt, gas,bb_min,bb_max,dt_step,dynamic_sampling,cam,
                                           cp_densities,cp_color_features_rgb,cp_positions,
                                           cp_scales, cp_quaternions,
                                           max_prim_slice=max_prim_slice,
                                            rnd_sample=rnd_sample,supersampling=supersampling,white_background=white_background
                                            ,hit_prim_idx=hit_prim_idx)
          end.record()
          torch.cuda.synchronize()
          time_render=start.elapsed_time(end)
          mean_time.append(time_render)
          ray_colors=torch.from_dlpack(ray_colors)
          ray_colors_mean=utilities.reduce_supersampling(cam.image_width,cam.image_height,ray_colors,supersampling)

          ray_colors_numpy = ray_colors_mean.detach().cpu().numpy().clip(0,1)

          images_list.append(ray_colors_numpy)
      print("Mean time",np.mean(mean_time))
      return images_list