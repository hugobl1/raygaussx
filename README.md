<p align="center">

  <h1 align="center">RayGaussX: Accelerating Gaussian-Based Ray Marching for Real-Time and High-Quality Novel View Synthesis</h1>
  <p align="center">
    <a href="https://www.linkedin.com/in/hugo-blanc-a2b46016a/">Hugo Blanc</a>
    ·
    <a href="https://scholar.google.com/citations?user=zR1n_4QAAAAJ&hl=fr">Jean-Emmanuel Deschaud</a>
    ·
    <a href="https://scholar.google.fr/citations?user=3eO15d0AAAAJ&hl=fr">Alexis Paljic</a>

  </p>
  <h2 align="center">ICCV 2025</h2>

  <h3 align="center"><a href="https://drive.google.com/file/d/1-look4HeGlXI_SnkEXY9_cgyMQ77JPDj/view?usp=sharing">Paper</a> | <a href="https://arxiv.org/abs/2509.07782">arXiv</a> | <a href="https://raygaussx.github.io/">Project Page</a>  
  <div align="center"></div>
</p>


<p align="center">
  <a href="">
    <img src="./media/Video_Beryl.gif" alt="Logo" width="95%">
  </a>
</p>

<p align="center">
We present an enhanced differentiable ray-casting algorithm for rendering Gaussians with scene features, enabling efficient 3D scene learning and real-time rendering.
</p>
<br>

## Update
  - 20/10/2025: Improved results with gradient accumulation over 20 images

PSNR results on different datasets:
|                 |NeRF synthetic|NSVF synthetic|Mip-NeRF360|Tanks&Temple|Deep Blending|
|-----------------|-------------|-------------|-------------|-------------|-------------|
|RayGaussX (Paper)|    34.54    |    38.75    |    28.43    |    23.76    |    30.32    |
|RayGaussX (Code) |    34.65    |    38.82    |    28.57    |    23.78    |    30.64    |

## Hardware Requirements
  - CUDA-ready GPU
  - 24 GB VRAM (to train to paper evaluation quality)

## Software Requirements

The following software components are required to ensure compatibility and optimal performance:

- **Ubuntu or Windows**
- **NVIDIA Drivers**: Install NVIDIA drivers, tested with version 575.64.03.
- **CUDA Toolkit**: Tested with version 12.9. You can dowload it from the [CUDA Toolkit 12.9 Downloads page](https://developer.nvidia.com/cuda-12-9-0-download-archive)
- **NVIDIA OptiX 7.6**: NVIDIA’s OptiX ray tracing engine, version 7.6, is required for graphics rendering and computational tasks. You can download it from the [NVIDIA OptiX Legacy Downloads page](https://developer.nvidia.com/designworks/optix/downloads/legacy).
- **Anaconda**: Install [Anaconda](https://anaconda.com/download), a distribution that includes Conda, for managing packages and environments efficiently.


## Installation (Ubuntu, Windows)

There are two steps in the installation process:
1. **OS-specific setup** (Ubuntu or Windows)  
2. **Common setup** (applies to both Ubuntu & Windows)

---

### 1. OS-specific setup

### Ubuntu:
Follow the steps below to set up the project:

   ```bash
  #Python-Optix requirements
  export OPTIX_PATH=/path/to/optix
  #For example, if OptiX is in your home folder: export OPTIX_PATH=~/NVIDIA-OptiX-SDK-7.6.0-linux64-x86_64/
  export CUDA_PATH=/path/to/cuda_toolkit
  #For example, the CUDA Toolkit is installed by default in /usr/local/: export CUDA_PATH=/usr/local/cuda-12.9
  export OPTIX_EMBED_HEADERS=1 # embed the optix headers into the package

  
  git clone https://github.com/hugobl1/raygaussx.git
  cd raygaussx
  conda env create --file environment.yml
  conda activate raygaussx
  ```

### Windows:
You will need a C++ compiler like Visual Studio Community: [link](https://visualstudio.microsoft.com/fr/vs/community/) \
Follow the steps below to set up the project:

   ```bash
  #Python-Optix requirements
  set OPTIX_PATH=\path\to\optix
  #For example, the repo is by default on C disk: set OPTIX_PATH=C:\ProgramData\NVIDIA Corporation\OptiX SDK 7.6.0
  set OPTIX_EMBED_HEADERS=1 # embed the optix headers into the package

  git clone https://github.com/hugobl1/raygaussx.git
  cd raygaussx
  conda env create --file environment_win.yml
  conda activate raygaussx
  conda install git
  ```

---

### 2. Common setup (Ubuntu & Windows)
Then install [Pytorch](https://pytorch.org/get-started/locally/) (choose the version appropriate for your installed CUDA Toolkit), the simple-knn submodule, and [fused-ssim](https://github.com/rahul-goel/fused-ssim), for example for CUDA Toolkit 12.9:

   ```bash
  pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu129
  pip3 install submodules/simple-knn/
  pip3 install git+https://github.com/rahul-goel/fused-ssim/ --no-build-isolation

  # Optional — for the GUI (pick version that match your CUDA Toolkit)
  pip3 install "cuda-python==12.9.0" "cuda-bindings==12.9.0"
  ```

# Datasets

Please download and unzip the following datasets, then place them in the `dataset` folder.  

| Dataset                        | Download Link   |
|--------------------------------|-----------------|
| Synthetic-NeRF                 | [download(.zip)](https://drive.google.com/file/d/1a3l9OL2lRA3z490QFNoDdZuUxTWrbdtD/view?usp=sharing) |
| Synthetic-NSVF                 | [download(.zip)](https://drive.google.com/file/d/1calWbNNuWgZJyBqJnkj8K9CK_Hvh0ccE/view?usp=sharing) |
| Mip-NeRF 360                   | [download](https://jonbarron.info/mipnerf360/) |
| Deep Blending + Tanks&Temples  | [download(.zip)](https://drive.google.com/file/d/1snnKl8fcksEPY24V_0YNCYWtCqdd0Elc/view?usp=sharing) |

Datasets used for videos:
| Dataset                        | Download Link   |
|--------------------------------|-----------------|
| Beryl                 | [download(.zip)](https://cloud.minesparis.psl.eu/index.php/s/HK547COP6rcoqOe) |
| Globe                 | [download(.zip)](https://cloud.minesparis.psl.eu/index.php/s/Xs4ILsDPApsa8hP) |


<!-- #### Trained Models

If you would like to directly visualize a model trained by RayGaussX, we provide the trained point clouds for each scene in Mip-NeRF 360. In this case, you can skip the training of the scene and evaluate or visualize it directly: [Download Link](https://drive.google.com/file/d/1E0_Tg2QeMx2kyohPhfRtfV656oQFQ2Kv/view?usp=sharing). -->
### Pretrained Models

If you want to **skip training** and run evaluation or visualization directly, we provide pretrained checkpoints for every scene in each dataset:
| Dataset                        | Download Link   |
|--------------------------------|-----------------|
| Synthetic-NeRF                 | [download](https://cloud.minesparis.psl.eu/index.php/s/6OEwQxuxjKjc961) |
| Synthetic-NSVF                 | [download](https://cloud.minesparis.psl.eu/index.php/s/GzEP3G9GviSNe7l) |
| Mip-NeRF 360                   | [download](https://cloud.minesparis.psl.eu/index.php/s/2x0F0LfekdebfTM) |
| Tanks&Temples                  | [download](https://cloud.minesparis.psl.eu/index.php/s/iAOW9UI9B8bBfaf) |
| Deep Blending                  | [download](https://cloud.minesparis.psl.eu/index.php/s/ztvCmU8ondtGhA7) |

Datasets used for videos:
| Dataset                        | Download Link   |
|--------------------------------|-----------------|
| Beryl                 | [download](https://cloud.minesparis.psl.eu/index.php/s/S6pF7LBiyZdioza) |
| Globe                 | [download](https://cloud.minesparis.psl.eu/index.php/s/WhqkME7y6vgXs2f) |

# Training and Evaluation
To reproduce the results on entire datasets, follow the instructions below:

---

### NeRF-Synthetic Dataset
1. **Prepare the Dataset**: Ensure the NeRF-Synthetic dataset is downloaded and placed in the `dataset` directory.

2. **Run Training Script**: Execute the following command:

   ```bash
   python main_train_blender.py
    ```

This will start the training and evaluation on the NeRF-Synthetic dataset with the configuration parameter in `nerf_synthetic.yml`.

<details>
<summary><b><u> 📊 RayGaussX Results Produced on RTX 5090</u></b></summary>

| NeRF-Synthetic    | PSNR  | SSIM  | LPIPS | Train (s) | FPS |
|------------|-------|-------|-------|---------------|-----|
| Chair      | 37,21 | 0,990 | 0,009 | 736,6         | 141 |
| Drums      | 27,13 | 0,960 | 0,030 | 851,4         | 105 |
| Ficus      | 35,09 | 0,988 | 0,011 | 501,5         | 208 |
| Hotdog     | 38,48 | 0,988 | 0,015 | 610,5         | 182 |
| Lego       | 37,02 | 0,986 | 0,012 | 585,8         | 202 |
| Materials  | 31,33 | 0,969 | 0,027 | 614,8         | 169 |
| Mic        | 38,02 | 0,995 | 0,004 | 645,8         | 139 |
| Ship       | 32,15 | 0,914 | 0,088 | 1027,4        | 81  |
| **Average**   | **34,55** | **0,974** | **0,024** | **696,7** | **140** |

</details>

---

### Synthetic-NSVF Dataset
1. **Prepare the Dataset**: Ensure the Synthetic-NSVF dataset is downloaded and placed in the `dataset` directory.

2. **Run Training Script**: Execute the following command:

   ```bash
   python main_train_synthetic-nsvf.py
    ```

This will start the training and evaluation on the Synthetic-NSVF dataset with the configuration parameter in `nerf_synthetic.yml`.

<details>
  <summary><b><u> 📊 RayGaussX Results Produced on RTX 5090</u></b></summary>

| Scene      | PSNR  | SSIM  | LPIPS | Training Time | FPS |
|------------|-------|-------|-------|---------------|-----|
| Bike       | 41,39 | 0,996 | 0,003 | 864,8         | 107 |
| Lifestyle  | 36,33 | 0,984 | 0,019 | 1188,8        | 77  |
| Palace     | 40,41 | 0,989 | 0,009 | 1278,6        | 81  |
| Robot      | 40,13 | 0,996 | 0,006 | 1096,9        | 77  |
| Spaceship  | 40,12 | 0,995 | 0,006 | 824,7         | 115 |
| Steamtrain | 39,10 | 0,995 | 0,006 | 1135,8        | 79  |
| Toad       | 38,40 | 0,989 | 0,011 | 2115,1        | 42  |
| Wineholder | 34,02 | 0,981 | 0,017 | 1697,4        | 50  |
| **Average**| **38,74** | **0,990** | **0,010** | **1275,3** | **78** |

</details>

---

### Mip-NeRF 360 Dataset
To reproduce results on the **Mip-NeRF 360** dataset:

1. **Prepare the Dataset**: Download and place the [Mip-NeRF 360](https://jonbarron.info/mipnerf360/) dataset in the `dataset` directory.

2. **Run Training Script**: Execute the following command:

   ```bash
   python main_train_mipnerf360.py
    ```

<details>
  <summary><b><u> 📊 RayGaussX Results Produced on RTX 5090</u></b></summary>

| Scene    | PSNR  | SSIM  | LPIPS | Training Time | FPS |
|----------|-------|-------|-------|---------------|-----|
| Bicycle  | 25,79 | 0,798 | 0,174 | 3276,0        | 32  |
| Bonsai   | 34,07 | 0,958 | 0,154 | 2036,9        | 45  |
| Counter  | 30,68 | 0,932 | 0,156 | 2287,0        | 42  |
| Flowers  | 22,18 | 0,652 | 0,288 | 3040,9        | 34  |
| Garden   | 28,26 | 0,886 | 0,089 | 2614,6        | 45  |
| Kitchen  | 32,84 | 0,941 | 0,103 | 2754,1        | 34  |
| Room     | 31,97 | 0,937 | 0,173 | 1740,2        | 54  |
| Stump    | 26,58 | 0,785 | 0,201 | 2572,6        | 40  |
| Treehill | 23,08 | 0,675 | 0,261 | 2839,7        | 37  |
| **Average** | **28,38** | **0,840** | **0,178** | **2573,5** | **40** |

</details>


---

### Deep Blending Dataset
To reproduce results on the **Deep Blending** dataset:

1. **Prepare the Dataset**: Download and unzip [Deep Blending](https://drive.google.com/file/d/1snnKl8fcksEPY24V_0YNCYWtCqdd0Elc/view?usp=sharing) into the `dataset` directory.

2. **Run Training Script**: Execute the following command:
   ```bash
   python main_train_db.py

<details>
  <summary><b><u> 📊 RayGaussX Results Produced on RTX 5090</u></b></summary>

| Scene      | PSNR  | SSIM  | LPIPS | Training Time | FPS |
|------------|-------|-------|-------|---------------|-----|
| Drjohnson  | 29,85 | 0,909 | 0,244 | 1781,7        | 55  |
| Playroom   | 30,78 | 0,910 | 0,245 | 1218,5        | 87  |
| **Average**| **30,32** | **0,909** | **0,245** | **1500,1** | **71** |

</details>


---

### Tanks&Temples Dataset
To reproduce results on the **Tanks&Temples** dataset:

1. **Prepare the Dataset**: Download and unzip [Tanks&Temples](https://drive.google.com/file/d/1snnKl8fcksEPY24V_0YNCYWtCqdd0Elc/view?usp=sharing) into the `dataset` directory.

2. **Run Training Script**: Execute the following command:
   ```bash
   python main_train_tandt.py

<details>
  <summary><b><u> 📊 RayGaussX Results Produced on RTX 5090</u></b></summary>

| Scene   | PSNR  | SSIM  | LPIPS | Training Time | FPS |
|---------|-------|-------|-------|---------------|-----|
| Train   | 22,28 | 0,833 | 0,182 | 1665,1        | 54  |
| Truck   | 25,07 | 0,891 | 0,119 | 1492,2        | 72  |
| **Average** | **23,67** | **0,862** | **0,150** | **1578,7** | **62** |

</details>

---

### All Datasets
To reproduce results on all datasets:

1. **Prepare the Datasets**: Download and unzip each dataset into the `dataset` directory.

2. **Run Training Scripts**: Execute the following command:
   ```bash
    bash train_all_datasets.sh

---
3. **Results**: The results for each scene can be found in the `output` folder after training is complete.

### Single Scene
To train and test a single scene, simply use the following commands:

   ```bash
    python main_train.py -config "path_to_config_file" --save_dir "name_save_dir" --arg_names scene.source_path --arg_values "scene_path"
    python main_test.py -output "./output/name_save_dir" -iter save_iter
    # For example, to train and evaluate the hotdog scene from NeRF Synthetic:
    # python main_train.py -config "./configs/nerf_synthetic.yml" --save_dir "hotdog" --arg_names scene.source_path --arg_values "./dataset/nerf_synthetic/hotdog"
    # python main_test.py -output "./output/hotdog" -iter 29999
```


        
By default, only the last iteration is saved (29999 in the base config files).

# PLY Point Cloud Extraction
To extract a point cloud in PLY format from a trained scene, we provide the script [convertpth_to_ply.py](convertpth_to_ply.py), which can be used as follows:
   ```bash
   python convertpth_to_ply.py -output "./output/name_scene" -iter num_iter
   # For example, if the 'hotdog' scene was trained for 29999 iterations, you can use:
   # python convertpth_to_ply.py -output "./output/hotdog" -iter 29999
   ```

The generated PLY point cloud will be located in the folder `./output/scene/saved_pc/`.

# Visualization (Ubuntu only)
To visualize a trained scene, we provide the script [main_gui.py](main_gui.py), which opens a GUI to display the trained scene:

   ```bash
   # Two ways to use the GUI:
   
   # Using the folder of the trained scene and the desired iteration
   python main_gui.py -output "./output/name_scene" -iter num_iter

   # Using a PLY point cloud:
   python main_gui.py -ply_path "path_to_ply_file"
   ```

## Camera Controls

### First Person Camera
In *First Person* mode, you can use the keyboard keys to move the camera in different directions.

- **Direction Keys**:
  - `Z`: Move forward
  - `Q`: Move backward
  - `S`: Move left
  - `D`: Move right
  - `A`: Move down
  - `E`: Move up  
  

- **View Control with Right Click**:
  - **Right Click + Move Mouse Up**: Look up
  - **Right Click + Move Mouse Down**: Look down
  - **Right Click + Move Mouse Left**: Look left
  - **Right Click + Move Mouse Right**: Look right

> **Note**: Ensure that the *First Person* camera mode is active for these controls to work.

### Trackball Camera
In *Trackball* mode, the camera can be controlled with the mouse to freely view around an object.

- **Left Click**: Rotate the camera around the object. Hold down the left mouse button and move the mouse to rotate around the object.
- **Right Click**: Pan. Hold down the right mouse button and move the mouse to shift the view laterally or vertically.
- **Mouse Wheel**: Zoom in and out. Scroll the wheel to adjust the camera's distance from the object.

> **Note**: Ensure that the *Trackball* camera mode is active for these controls to work.

# Camera Path Rendering

To render a camera path from a trained point cloud, use the script as follows:
```bash
python render_camera_path.py -output "./output" -camera_path_filename "camera_path.json" -name_video "my_video"
```
This script loads a pre-trained model, renders images along a specified camera path, and saves them in `output/camera_path/images/`. A video is then generated from the images and saved in `output/camera_path/video/`.

The `camera_path.json` file, which defines the camera path, can be generated using [NeRFStudio](https://nerf.studio/) by training a similar scene and then exporting a `camera_path.json` file through NeRFStudio's graphical user interface. 
To maintain consistency with our method, you should use the `ns-train` command with the following options: 

```bash
--assume_colmap_world_coordinate_convention=False \
--orientation_method=none \
--center_method=none \
--auto-scale-poses=False \
```

# Processing Your Own Scenes with COLMAP

To use your own scenes, ensure your dataset is structured correctly for the COLMAP loaders. The directory must include an `images` folder containing your image files and a `sparse` folder with subdirectories containing `cameras.bin`, `images.bin`, and `points3D.bin` files obtained using COLMAP reconstruction. Note that the camera models used for COLMAP reconstruction must be either `SIMPLE_PINHOLE` or `PINHOLE`. 

The dataset structure must be as follows:

```
<location>
|---images
|   |---<image 0>
|   |---<image 1>
|   |---...
|---sparse
    |---0
        |---cameras.bin
        |---images.bin
        |---points3D.bin
```

# Acknowledgements

We thank the authors of [Python-Optix](https://github.com/mortacious/python-optix), upon which our project is based, as well as the authors of [NeRF](https://github.com/bmild/nerf) and [Mip-NeRF 360](https://github.com/google-research/multinerf) for providing their datasets. Finally, we would like to acknowledge the authors of [3D Gaussian Splatting](https://github.com/graphdeco-inria/gaussian-splatting), as our project's dataloader is inspired by the one used in 3DGS; and [Mip-Splatting](https://github.com/autonomousvision/mip-splatting) for the calculation of the minimum sizes of the Gaussians as a function of the cameras.



# Citation
If you find our code or paper useful, please cite
```bibtex
@misc{blanc2025raygaussxacceleratinggaussianbasedray,
      title={RayGaussX: Accelerating Gaussian-Based Ray Marching for Real-Time and High-Quality Novel View Synthesis}, 
      author={Hugo Blanc and Jean-Emmanuel Deschaud and Alexis Paljic},
      year={2025},
      eprint={2509.07782},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2509.07782}, 
}
```
and
```bibtex
@INPROCEEDINGS{blanc2025raygauss,
  author={Blanc, Hugo and Deschaud, Jean-Emmanuel and Paljic, Alexis},
  booktitle={2025 IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)}, 
  title={RayGauss: Volumetric Gaussian-Based Ray Casting for Photorealistic Novel View Synthesis}, 
  year={2025},
  volume={},
  number={},
  pages={1808-1817},
  keywords={Training;Hands;Casting;Computer vision;Rendering (computer graphics);Neural radiance field;Inference algorithms;Slabs;Kernel;Videos;volume ray casting;differentiable rendering;radiance fields;novel view synthesis},
  doi={10.1109/WACV61041.2025.00183}
}
```
