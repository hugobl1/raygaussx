#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
from typing import NamedTuple
from pathlib import Path
import json
from decimal import Decimal, ROUND_HALF_UP
import logging
logger = logging.getLogger(__name__)
from PIL import Image

import numpy as np
from plyfile import PlyData, PlyElement
from skimage.restoration import denoise_tv_chambolle
from scene.colmap_loader import read_extrinsics_text, read_intrinsics_text, qvec2rotmat, \
    read_extrinsics_binary, read_intrinsics_binary, read_points3D_binary, read_points3D_text
from utils.graphics_utils import getWorld2View2, focal2fov, fov2focal
from utils.sh_utils import SH2RGB
from utils.graphics_utils import BasicPointCloud



class CameraInfo(NamedTuple):
    uid: int
    R: np.array
    T: np.array
    FoVy: np.array
    FoVx: np.array
    image: np.array
    image_path: str
    image_name: str
    width: int
    height: int
    Px: float = 0.5
    Py: float = 0.5
    z_near: float = 0.1
    z_far: float = 100.0

class SceneInfo(NamedTuple):
    """
    Holds scene-related data required for training and evaluating a Novel View Synthesis model.

    Attributes:
        train_cameras (list): Cameras used for training.
        test_cameras (list): Cameras used for evaluation.
        nerf_normalization (dict): Scene normalization parameters.
    """
    train_cameras: list
    test_cameras: list
    nerf_normalization: dict

def getNerfppNorm(cam_info):
    def get_center_and_diag(cam_centers):
        cam_centers = np.hstack(cam_centers)
        avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
        center = avg_cam_center
        dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
        diagonal = np.max(dist) # Half the diagonal of the bounding box
        return center.flatten(), diagonal

    cam_centers = []

    for cam in cam_info:
        W2C = getWorld2View2(cam.R, cam.T)
        C2W = np.linalg.inv(W2C)
        cam_centers.append(C2W[:3, 3:4])

    center, diagonal = get_center_and_diag(cam_centers)
    radius = diagonal * 1.1

    translate = -center

    return {"translate": translate, "radius": radius}

def readColmapCameras(cam_extrinsics, cam_intrinsics, images_folder,zmin_zmax=None,resize_im=False,scale=1):
    cam_infos = []
    for idx, key in enumerate(cam_extrinsics):
        extr = cam_extrinsics[key]
        intr = cam_intrinsics[extr.camera_id]
        height = intr.height
        width = intr.width

        uid = intr.id
        R = np.transpose(qvec2rotmat(extr.qvec))
        T = np.array(extr.tvec)

        if intr.model=="SIMPLE_PINHOLE":
            focal_length_x = intr.params[0]
            FoVy = focal2fov(focal_length_x, height)
            FoVx = focal2fov(focal_length_x, width)
            ppx= intr.params[1]/float(width)
            ppy= intr.params[2]/float(height)
        elif intr.model=="PINHOLE":
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            FoVy = focal2fov(focal_length_y, height)
            FoVx = focal2fov(focal_length_x, width)
            ppx= intr.params[2]/float(width)
            ppy= intr.params[3]/float(height)
        else:
            assert False, "Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!"
        image_path = os.path.join(images_folder, os.path.basename(extr.name))
        image_name = os.path.basename(image_path).split(".")[0]
        image = Image.open(image_path)
        if resize_im:
            orig_w, orig_h = image.size
            resolution = round(orig_w/(scale)), round(orig_h/(scale))
            image=image.resize(resolution)


        if zmin_zmax is not None:
            z_near, z_far = zmin_zmax[idx]
            cam_info = CameraInfo(uid=uid, R=R, T=T, FoVy=FoVy, FoVx=FoVx, image=image,
                                    image_path=image_path, image_name=image_name, width=width, height=height,
                                    Px=ppx,Py=ppy,z_near=z_near, z_far=z_far)
        else:
            cam_info = CameraInfo(uid=uid, R=R, T=T, FoVy=FoVy, FoVx=FoVx, image=image,
                              image_path=image_path, image_name=image_name, width=width, height=height,Px=ppx,Py=ppy)
        cam_infos.append(cam_info)
    return cam_infos

def fetchPly(path):
    plydata = PlyData.read(path)
    vertices = plydata['vertex']
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0
    #print("colors",colors)
    # print("colors",positions)
    #If normals are not present, just return zeros
    try:
        normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T
    except:
        normals = np.zeros_like(positions)
        # normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T
    
    return BasicPointCloud(points=positions, colors=colors, normals=normals)

def storePly(path, xyz, rgb):
    # Define the dtype for the structured array
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    
    normals = np.zeros_like(xyz)

    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(path)

def readColmapSceneInfo(path, images, eval, config,llffhold=8):
    logger.info("Reading COLMAP scene info from %s", path)
    try:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.bin")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.bin")
        cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
    except:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.txt")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.txt")
        cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)

    reading_dir = "images" if images == None else images
    #Open poses_bounds.npy if it exists
    if os.path.exists(os.path.join(path, "poses_bounds.npy")):
        poses_bounds = np.load(os.path.join(path, "poses_bounds.npy"))
        zmin_zmax = poses_bounds[:, 15:17]            
        cam_infos_unsorted = readColmapCameras(cam_extrinsics=cam_extrinsics, cam_intrinsics=cam_intrinsics, images_folder=os.path.join(path, reading_dir), zmin_zmax=zmin_zmax)
    else:
        cam_infos_unsorted = readColmapCameras(cam_extrinsics=cam_extrinsics, cam_intrinsics=cam_intrinsics, images_folder=os.path.join(path, reading_dir))
    
    # cam_infos_unsorted = readColmapCameras(cam_extrinsics=cam_extrinsics, cam_intrinsics=cam_intrinsics, images_folder=os.path.join(path, reading_dir))
    cam_infos = sorted(cam_infos_unsorted.copy(), key = lambda x : x.image_name)

    if eval:
        train_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold != 0]
        test_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold == 0]
    else:
        train_cam_infos = cam_infos
        test_cam_infos = []
    nerf_normalization = getNerfppNorm(train_cam_infos)

    #########################################Mip NeRF Multi-scale like dataset############################################
    own_resize_im=True #If true, the image will be resized there, if false, we use the image folder with the resized images, to match mip-splatting way of training
    
    #Merge config.scene.train_resolution_scales and config.scene.test_resolution_scales
    scales_list=config.scene.train_resolution_scales+config.scene.test_resolution_scales
    #Extract the unique values 
    scales_list = list(set(scales_list))
    if len(scales_list) > 0:
        train_cam_infos = {}
        test_cam_infos = {}
        for scale in scales_list:
            #Check if images folder exists for this resolution
            if own_resize_im:
                images_folder=os.path.join(path, reading_dir)
            else:
                images_folder = os.path.join(path, reading_dir + f"_{int(scale)}") if scale != 1.0 else os.path.join(path, reading_dir)
            if os.path.exists(images_folder):
                scaled_intrinsics=cam_intrinsics.copy()
                #Scale the intrinsics
                for key,_ in  scaled_intrinsics.items():
                    #Round to nearest integer
                    nearest_width =int( Decimal(scaled_intrinsics[key].width/scale).to_integral_value(rounding=ROUND_HALF_UP))
                    ratio_x = nearest_width/scaled_intrinsics[key].width
                    scaled_intrinsics[key]=scaled_intrinsics[key]._replace(width=nearest_width)
                    nearest_height =int( Decimal(scaled_intrinsics[key].height/scale).to_integral_value(rounding=ROUND_HALF_UP))
                    ratio_y = nearest_height/scaled_intrinsics[key].height
                    scaled_intrinsics[key]=scaled_intrinsics[key]._replace(height=nearest_height)
                    if scaled_intrinsics[key].model=="SIMPLE_PINHOLE":
                        scaled_intrinsics[key]=scaled_intrinsics[key]._replace(model="PINHOLE")
                        f=scaled_intrinsics[key].params[0]
                        cx,cy=scaled_intrinsics[key].params[1],scaled_intrinsics[key].params[2]
                        fx,fy=f*ratio_x,f*ratio_y
                        cx,cy=cx*ratio_x,cy*ratio_y
                    elif scaled_intrinsics[key].model=="PINHOLE":
                        fx,fy,cx,cy=scaled_intrinsics[key].params[0:4]
                        fx,fy,cx,cy=fx*ratio_x,fy*ratio_y,cx*ratio_x,cy*ratio_y
                    scaled_params=np.array([fx,fy,cx,cy])
                    scaled_intrinsics[key]=scaled_intrinsics[key]._replace(params=scaled_params)
                #Load it
                #Open poses_bounds.npy if it exists
                if os.path.exists(os.path.join(path, "poses_bounds.npy")):
                    poses_bounds = np.load(os.path.join(path, "poses_bounds.npy"))
                    zmin_zmax = poses_bounds[:, 15:17]
                    cam_infos_unsorted = readColmapCameras(cam_extrinsics=cam_extrinsics, cam_intrinsics=scaled_intrinsics, images_folder=images_folder, zmin_zmax=zmin_zmax,resize_im=own_resize_im,scale=scale)
                else:
                    cam_infos_unsorted = readColmapCameras(cam_extrinsics=cam_extrinsics, cam_intrinsics=scaled_intrinsics, images_folder=images_folder,resize_im=own_resize_im,scale=scale)
                
                # cam_infos_unsorted = readColmapCameras(cam_extrinsics=cam_extrinsics, cam_intrinsics=cam_intrinsics, images_folder=os.path.join(path, reading_dir))
                cam_infos = sorted(cam_infos_unsorted.copy(), key = lambda x : x.image_name)           
                        
                if eval:
                    train_cam_infos_scale = [c for idx, c in enumerate(cam_infos) if idx % llffhold != 0]
                    test_cam_infos_scale = [c for idx, c in enumerate(cam_infos) if idx % llffhold == 0]
                else:
                    train_cam_infos_scale = cam_infos
                    test_cam_infos_scale = []
                train_cam_infos[scale] = train_cam_infos_scale
                test_cam_infos[scale] = test_cam_infos_scale
            else:
                print(f"Images folder for scale {scale} not found, skipping...")
    ########################################################################################################################

    if config.pointcloud.init_method=="ply":
        ply_path = config.pointcloud.ply.ply_path
        # print("ply path", ply_path)
        # ply_path = os.path.join(path, "sparse/0/",config.pointcloud.ply.ply_name)
        # print("ply path", ply_path)
        # ply_path = os.path.join(path, "sparse/0/point_cloud.ply")
        bin_path = os.path.join(path, "sparse/0/points3D.bin")
        txt_path = os.path.join(path, "sparse/0/points3D.txt")
        if not os.path.exists(ply_path):
            print("Converting point3d.bin to .ply, will happen only the first time you open the scene.")
            try:
                xyz, rgb, _ = read_points3D_binary(bin_path)
            except:
                xyz, rgb, _ = read_points3D_text(txt_path)
            storePly(ply_path, xyz, rgb)
    # try:
    #     pcd = fetchPly(ply_path)
    # except:
    #     pcd = None

    scene_info = SceneInfo(train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           )
    return scene_info

def readCamerasFromTransforms(path, transformsfile, white_background, extension=".png", denoising_image = False):
    cam_infos = []

    with open(os.path.join(path, transformsfile)) as json_file:
        contents = json.load(json_file)
        fovx = contents["camera_angle_x"]

        frames = contents["frames"]
        for idx, frame in enumerate(frames):
            cam_name = os.path.join(path, frame["file_path"] + extension)

            # NeRF 'transform_matrix' is a camera-to-world transform
            c2w = np.array(frame["transform_matrix"])
            # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
            c2w[:3, 1:3] *= -1
            # get the world-to-camera transform and set R, T
            w2c = np.linalg.inv(c2w)
            R = np.transpose(w2c[:3,:3])  # R is stored transposed due to 'glm' in CUDA code
            T = w2c[:3, 3]

            image_path = os.path.join(path, cam_name)

            image_name = Path(cam_name).stem
            image = Image.open(image_path)

            im_data = np.array(image.convert("RGBA"))

            bg = np.array([1,1,1]) if white_background else np.array([0, 0, 0])

            norm_data = im_data / 255.0
            arr = norm_data[:,:,:3] * norm_data[:, :, 3:4] + bg * (1 - norm_data[:, :, 3:4])

            if (denoising_image):
                arr = denoise_tv_chambolle(arr, weight=0.02, eps=0.0002, max_num_iter=200, channel_axis=-1)
                denoised_image = Image.fromarray(np.array(arr*255.0, dtype=np.byte), "RGB")
                denoised_cam_name = os.path.join(path, frame["file_path"] +'_denoised' + extension)
                denoised_image_path = os.path.join(path, denoised_cam_name)
                denoised_image.save(denoised_image_path)

            image = Image.fromarray(np.array(arr*255.0, dtype=np.byte), "RGB")
            

            FoVy = focal2fov(fov2focal(fovx, image.size[0]), image.size[1])
            FoVy = FoVy
            FoVx = fovx

            cam_infos.append(CameraInfo(uid=idx, R=R, T=T, FoVy=FoVy, FoVx=FoVx, image=image,
                            image_path=image_path, image_name=image_name, width=image.size[0], height=image.size[1]))
            
    return cam_infos

def readNerfSyntheticInfo(path, white_background, eval, config, extension=".png"):
    logger.info("Reading NeRF Synthetic scene info from %s", path)
    try:
        logger.info("Reading Training Transforms")
        train_cam_infos = readCamerasFromTransforms(path, "transforms_train.json", white_background, extension)
        logger.info("Reading Test Transforms")
        test_cam_infos = readCamerasFromTransforms(path, "transforms_test.json", white_background, extension)
        
        if not eval:
            train_cam_infos.extend(test_cam_infos)
            test_cam_infos = []
    except:
        logger.info("Reading All Transforms")
        cam_infos = readCamerasFromTransforms(path, "transforms.json", white_background, extension)

        if eval:
            train_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % 3 != 0]
            test_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % 3 == 0]

    nerf_normalization = getNerfppNorm(train_cam_infos)

    if config.pointcloud.init_method=="ply":
        # ply_path=os.path.join(path,config.pointcloud.ply.name_ply)
        ply_path = config.pointcloud.ply.ply_path
        if not os.path.exists(ply_path): #If the ply file does not exist, we generate a random point cloud
            # Since this data set has no colmap data, we start with random points
            num_pts = config.pointcloud.ply.n_rnd_pts
            print(f"Generating random point cloud ({num_pts})...")
            
            # We create random points inside the bounds of the synthetic Blender scenes

            xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3
            # xyz = np.random.random((num_pts, 3)) * 25.0 - 1.3
            shs = np.random.random((num_pts, 3)) / 255.0
            # pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)))

            storePly(ply_path, xyz, SH2RGB(shs) * 255) 
            
    scene_info = SceneInfo(train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           )
    return scene_info


def readMultiScale(path, white_background,split, only_highres=False):
    #Read Multiscale but only works for the NeRF multiscale dataset
    cam_infos = []
    
    #print("read split:", split)
    with open(os.path.join(path, 'metadata.json'), 'r') as fp:
        meta = json.load(fp)[split]
        
    meta = {k: np.array(meta[k]) for k in meta}
    
    #if only_highres is True, we only load the high resolution images else we load all images and store it by resolution in a dictionary
    if not only_highres:
        cam_infos = {}
        cam_infos[1.0]=[]
        cam_infos[2.0]=[]
        cam_infos[4.0]=[]
        cam_infos[8.0]=[]
    # should now have ['pix2cam', 'cam2world', 'width', 'height'] in self.meta
    for idx, relative_path in enumerate(meta['file_path']):
        if only_highres and not relative_path.endswith("d0.png"):
            continue
        image_path = os.path.join(path, relative_path)
        image_name = Path(image_path).stem
        
        # NeRF 'transform_matrix' is a camera-to-world transform
        c2w = meta["cam2world"][idx]
        # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
        c2w[:3, 1:3] *= -1

        # get the world-to-camera transform and set R, T
        w2c = np.linalg.inv(c2w)
        R = np.transpose(w2c[:3,:3])  # R is stored transposed due to 'glm' in CUDA code
        T = w2c[:3, 3]

        image = Image.open(image_path)

        im_data = np.array(image.convert("RGBA"))

        bg = np.array([1,1,1]) if white_background else np.array([0, 0, 0])

        norm_data = im_data / 255.0
        arr = norm_data[:,:,:3] * norm_data[:, :, 3:4] + bg * (1 - norm_data[:, :, 3:4])
        image = Image.fromarray(np.array(arr*255.0, dtype=np.byte), "RGB")

        fovx = focal2fov(meta["focal"][idx], image.size[0])
        FoVy = focal2fov(meta["focal"][idx], image.size[1])
        FoVy = FoVy 
        FoVx = fovx

        if only_highres:
            cam_infos.append(CameraInfo(uid=idx, R=R, T=T, FoVy=FoVy, FoVx=FoVx, image=image,
                        image_path=image_path, image_name=image_name, width=image.size[0], height=image.size[1]))
        else:
            width = meta["width"][idx] #Not very clean but ok for now
            scale=800.0/width
            cam_infos[scale].append(CameraInfo(uid=idx, R=R, T=T, FoVy=FoVy, FoVx=FoVx, image=image,
                        image_path=image_path, image_name=image_name, width=image.size[0], height=image.size[1]))
    return cam_infos


def readMultiScaleNerfSyntheticInfo(path,name_ply, white_background, eval,load_allres=False):
    logger.info("Reading Multi-scale NeRF Synthetic scene info from %s", path)
    train_cam_infos = readMultiScale(path, white_background, "train", only_highres=(not load_allres))
    #By default, we load all resolutions for testing
    test_cam_infos = readMultiScale(path, white_background, "test", only_highres=False)
    if not eval:
        print("adding test cameras to training")
        train_cam_infos.extend(test_cam_infos)
        test_cam_infos = []

    #If train_cam_infos is a dictionary, then it is a native multi-scale dataset
    if isinstance(train_cam_infos, dict):
        nerf_normalization = getNerfppNorm(train_cam_infos[1.0])
    else:
        nerf_normalization = getNerfppNorm(train_cam_infos)

    # ply_path = os.path.join(path, "fused.ply")
    ply_path = os.path.join(path, name_ply)
    print("ply path", ply_path)
    if not os.path.exists(ply_path):
        # Since this data set has no colmap data, we start with random points
        num_pts = 100_000
        print(f"Generating random point cloud ({num_pts})...")
        
        # We create random points inside the bounds of the synthetic Blender scenes
        xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3
        shs = np.random.random((num_pts, 3)) / 255.0
        pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)))

        storePly(ply_path, xyz, SH2RGB(shs) * 255)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info

def readCamerasFromNSVFPoses(path, idx, white_background, extension=".png"):
    cam_infos = []
    all_poses = sorted(os.listdir(os.path.join(path, "pose")))
    all_rgbs = sorted(os.listdir(os.path.join(path, "rgb")))

    with open(os.path.join(path, "intrinsics.txt")) as f:
        focal = float(f.readline().split()[0])
    for i in idx:
        cam_name = os.path.join(path, "pose", all_poses[i])
        c2w = np.loadtxt(cam_name)
        w2c = np.linalg.inv(c2w)

        R = np.transpose(w2c[:3, :3])
        T = w2c[:3, 3]

        image_path = os.path.join(path, "rgb", all_rgbs[i])
        image_name = Path(cam_name).stem
        image = Image.open(image_path)

        im_data = np.array(image.convert("RGBA"))

        bg = np.array([1, 1, 1]) if white_background else np.array([0, 0, 0])

        norm_data = im_data / 255.0
        arr = norm_data[:, :, :3] * norm_data[:, :, 3:4] + bg * (1 - norm_data[:, :, 3:4])
        image = Image.fromarray(np.array(arr * 255.0, dtype=np.byte), "RGB")

        # given focal in pixel unit
        FoVy = focal2fov(focal, image.size[1])
        FoVx = focal2fov(focal, image.size[0])

        cam_infos.append(CameraInfo(uid=idx, R=R, T=T, FoVy=FoVy, FoVx=FoVx, image=image,
                            image_path=image_path, image_name=image_name, width=image.size[0], height=image.size[1]))


    return cam_infos


def readNSVFSyntheticInfo(path, white_background, eval, config, extension=".png"):
    logger.info("Reading NSVF Synthetic scene info from %s", path)
    all_rgbs = sorted(os.listdir(os.path.join(path, "rgb")))

    train_idx = [idx for idx, file_name in enumerate(all_rgbs) if file_name.startswith("0_")]
    test_idx = [idx for idx, file_name in enumerate(all_rgbs) if file_name.startswith("2_")]

    #print("Reading Training Transforms")
    train_cam_infos = readCamerasFromNSVFPoses(path, train_idx, white_background, extension)
    #print("Reading Test Transforms")
    test_cam_infos = readCamerasFromNSVFPoses(path, test_idx, white_background, extension)

    if not eval:
        train_cam_infos.extend(test_cam_infos)
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    if config.pointcloud.init_method=="ply":
        # ply_path=os.path.join(path,config.pointcloud.ply.name_ply)
        ply_path = config.pointcloud.ply.ply_path
        if not os.path.exists(ply_path): #If the ply file does not exist, we generate a random point cloud
            # Since this data set has no colmap data, we start with random points
            num_pts = config.pointcloud.ply.n_rnd_pts
            print(f"Generating random point cloud ({num_pts})...")
            
            # We create random points inside the bounds of the synthetic Blender scenes
            xyz = np.random.random((num_pts, 3)) * 2.0 - 1.0
            shs = np.random.random((num_pts, 3)) / 255.0
            # pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)))

            storePly(ply_path, xyz, SH2RGB(shs) * 255) 
            
    scene_info = SceneInfo(train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           )
    return scene_info


sceneLoadTypeCallbacks = {
    "Colmap": readColmapSceneInfo,
    "Blender" : readNerfSyntheticInfo,
    "Multi-scale": readMultiScaleNerfSyntheticInfo,
    "NSVF": readNSVFSyntheticInfo
}