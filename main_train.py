import logging
import argparse,os
from datetime import datetime
from omegaconf import OmegaConf
import omegaconf as om
from scripts.train import train
from utils.logger import log_nvidia_smi

parser = argparse.ArgumentParser(description="Ray Gauss Training.")
parser.add_argument("-config", type=str, default="configs/nerf_synthetic.yml", help="Path to config file")
# Add argument_names and argument_values that is a list of strings and a list of floats for changing config parameters
parser.add_argument("--arg_names", type=str, nargs='+', help="Names of arguments")
parser.add_argument("--arg_values", type=str, nargs='+', help="Values of arguments")
parser.add_argument("--save_dir", type=str, default=None, help="Path to save directory")
args = parser.parse_args()

############################################################################################################
now = datetime.now()
timestamp = now.strftime("%Y-%m-%d_%H-%M-%S")
if args.save_dir is not None:
  timestamp=args.save_dir
timestamped_dir=os.path.join('output',timestamp)
os.makedirs(timestamped_dir,exist_ok=True)
#In this folder, save model, config file
os.makedirs(os.path.join(timestamped_dir,"config"),exist_ok=True)

config=OmegaConf.load(args.config)

config.save.models=os.path.join(timestamped_dir, config.save.models)
config.save.screenshots=os.path.join(timestamped_dir, config.save.screenshots)
config.save.metrics=os.path.join(timestamped_dir, config.save.metrics)
config.save.logs=os.path.join(timestamped_dir,config.save.logs)
config.save.tensorboard_logs=os.path.join(timestamped_dir,config.save.tensorboard_logs)

os.makedirs(config.save.models,exist_ok=True)
os.makedirs(os.path.join(config.save.screenshots,"train"),exist_ok=True)
os.makedirs(os.path.join(config.save.screenshots,"test"),exist_ok=True)
os.makedirs(config.save.metrics,exist_ok=True)
os.makedirs(config.save.tensorboard_logs,exist_ok=True)

file_handler = logging.FileHandler(config.save.logs, encoding="utf-8")
file_handler.setFormatter(logging.Formatter(
    fmt="%(asctime)s | %(levelname)-7s | %(name)-21s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
))

logging.basicConfig(
    level=logging.INFO,
    handlers=[
        file_handler
    ]
)

# logging.basicConfig(
#     filename=config.save.logs,
#     level=logging.INFO,
#     format="%(asctime)s | %(levelname)-7s | %(name)-20s | %(message)s",
#     datefmt="%Y-%m-%d %H:%M:%S"
# )
logger = logging.getLogger(__name__)
log_nvidia_smi(logger)

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
          #List case
          elif isinstance(current_config[arg],om.listconfig.ListConfig):
            arg_value=[float(arg_value)]
          else:
            raise ValueError("The argument type is not supported, the type of the argument is ",type(current_config[arg]))
          current_config[arg]=arg_value
        else:        
          current_config=current_config[arg]
          compteur+=1
      else:
        raise ValueError("The argument name doesn't exist in the config file")

#Save the modified config file
OmegaConf.save(config,os.path.join(timestamped_dir,"config","config.yml"))

if __name__ == "__main__":
  psnr_test,_,_,_=train(config,quiet=True)
  print("PSNR : ",psnr_test)
  #Save model at the end of the training
  #opt_scene.pointcloud.save_model(config.training.n_iters ,config.save.models)

