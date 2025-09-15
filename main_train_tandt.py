import logging
# logging.basicConfig(
#     filename='log_output.txt',  # Chemin du fichier
#     level=logging.INFO,        # Niveau minimal de journalisation
#     format='%(asctime)s - %(levelname)s - %(message)s'  # Format des messages
# )
import argparse,os
from datetime import datetime
from omegaconf import OmegaConf
from scripts.train import train
import time
import numpy as np
import cupy as cp
import gc
import torch

parser = argparse.ArgumentParser(description="Ray Gauss Training.")
#parser.add_argument("-config", type=str, default="configs/nerf_synthetic.yml", help="Path to config file")
parser.add_argument("-config", type=str, default="configs/base.yml", help="Path to config file")
parser.add_argument("--arg_names", type=str, nargs='+', help="Names of arguments")
parser.add_argument("--arg_values", type=str, nargs='+', help="Values of arguments")
parser.add_argument("--save_dir", type=str, default=None, help="Path to save directory")
args = parser.parse_args()

############################################################################################################
now = datetime.now()
timestamp = now.strftime("%Y-%m-%d_%H-%M-%S")
if args.save_dir is not None:
  timestamp=args.save_dir
timestamp = timestamp+'_tandt'
timestamped_dir=os.path.join('output',timestamp)
os.makedirs(timestamped_dir,exist_ok=True)

config=OmegaConf.load(args.config)

############################################################################################################

if args.arg_names is not None and args.arg_values is not None:
  for arg_name,arg_value in zip(args.arg_names,args.arg_values):
    tree_arg_name=arg_name.split(".") #Split a in a string list separating
    current_config=config
    compteur=0
    for arg in tree_arg_name:
      if arg in current_config:
        if compteur==len(tree_arg_name)-1:
          #Convert arg_value to the right type
          if isinstance(current_config[arg],bool):
            if arg_value=="True":
              arg_value=True
            elif arg_value=="False":
              arg_value=False
            else:
              raise ValueError("The argument value should be True or False")
          elif isinstance(current_config[arg],int):
            arg_value=int(arg_value)
          elif isinstance(current_config[arg],float):
            arg_value=float(arg_value)
          elif isinstance(current_config[arg],str):
            arg_value=str(arg_value)
          else:
            raise ValueError("The argument type is not supported")
          current_config[arg]=arg_value
        else:        
          current_config=current_config[arg]
          compteur+=1
      else:
        raise ValueError("The argument name doesn't exist in the config file")


config_save_models = config.save.models
config_save_screenshots = config.save.screenshots
config_save_metrics = config.save.metrics
config_save_logs=config.save.logs
config_save_tensorboard_logs= config.save.tensorboard_logs


if __name__ == "__main__":
  print("#"*80)
  scenes=["tandt/truck","tandt/train"]
  mean_psnr_over_iter = []
  mean_ssim_over_iter = []
  mean_lpips_over_iter = []
  mean_fps_over_iter = []
  start_time = time.time()
  for scene_to_train in scenes:
    print("#"*80)
    print(scene_to_train)
    mempool = cp.get_default_memory_pool()
    config.scene.source_path = "./dataset/tandt_db/" + scene_to_train
    config.scene.train_resolution_scales = [1.0]
    config.scene.test_resolution_scales = [1.0]
    timestamped_dir_scene_to_train = timestamped_dir + "/" + scene_to_train
    config.save.models=os.path.join(timestamped_dir_scene_to_train, config_save_models)
    config.save.screenshots=os.path.join(timestamped_dir_scene_to_train, config_save_screenshots)
    config.save.metrics=os.path.join(timestamped_dir_scene_to_train, config_save_metrics)
    config.save.logs=os.path.join(timestamped_dir_scene_to_train,config_save_logs)
    config.save.tensorboard_logs=os.path.join(timestamped_dir_scene_to_train,config_save_tensorboard_logs)
    os.makedirs(config.save.models,exist_ok=True)
    os.makedirs(os.path.join(config.save.screenshots,"train"),exist_ok=True)
    os.makedirs(os.path.join(config.save.screenshots,"test"),exist_ok=True)
    os.makedirs(config.save.metrics,exist_ok=True)
    os.makedirs(config.save.tensorboard_logs,exist_ok=True)
    os.makedirs(os.path.join(timestamped_dir_scene_to_train,"config"),exist_ok=True)
    #Save the modified config file
    OmegaConf.save(config,os.path.join(timestamped_dir_scene_to_train,"config","config.yml"))

    file_handler = logging.FileHandler(config.save.logs, encoding="utf-8")
    file_handler.setFormatter(logging.Formatter(
        fmt="%(asctime)s | %(levelname)-7s | %(name)-21s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    ))

    logging.basicConfig(
        level=logging.INFO,
        handlers=[
            file_handler
        ],
        force=True
    )
    psnr_test,ssim_test,lpips_test,fps_test=train(config,quiet=True)
    mean_psnr_over_iter.append(psnr_test)
    mean_ssim_over_iter.append(ssim_test)
    mean_lpips_over_iter.append(lpips_test)
    mean_fps_over_iter.append(fps_test)
    gc.collect()
    torch.cuda.empty_cache()
    mempool.free_all_blocks()
  print("Mean PSNR : ", np.mean(mean_psnr_over_iter))
  print("Mean SSIM : ", np.mean(mean_ssim_over_iter))
  print("Mean LPIPS : ", np.mean(mean_lpips_over_iter))
  print("Mean Train time : ", ((time.time()-start_time)/60)/len(scenes))
  print("Mean FPS : ", 1.0/(np.mean(mean_fps_over_iter)/1000.0))
  print()
  print()
  

  #psnr_test,opt_scene=train(config,quiet=True)
  #print("PSNR : ",psnr_test)
  #Save model at the end of the training
  #opt_scene.pointcloud.save_model(config.training.n_iters ,config.save.models)

