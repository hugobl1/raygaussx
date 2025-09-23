from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional
import logging

import numpy as np
import torch
from omegaconf import OmegaConf

from classes import point_cloud, scene


def _parse_available_iterations(models_dir: Path, logger: Optional[logging.Logger] = None) -> List[int]:
    """
    Parse checkpoint iteration numbers from filenames in `models_dir`.
    Expects files containing 'chkpnt' and ending with '.pth', e.g. '...chkpnt500.pt'.
    Returns a sorted, de-duplicated list of integers.
    """
    iters: List[int] = []
    if not models_dir.exists() or not models_dir.is_dir():
        if logger:
            logger.warning("Models directory does not exist or is not a directory: %s", models_dir)
        return iters

    for p in models_dir.iterdir():
        if not (p.is_file() and "chkpnt" in p.name and p.suffix == ".pth"):
            continue
        stem = p.stem  # e.g. "...chkpnt500"
        try:
            it_val = int(stem.split("chkpnt", 1)[1])
            iters.append(it_val)
        except Exception:
            if logger:
                logger.debug("Skipping non-matching checkpoint name: %s", p.name)

    iters = sorted(set(iters))
    return iters


def _to_uint8_images(cam_infos: list) -> List[np.ndarray]:
    """
    Convert camera original images (assumed torch tensors CxHxW in [0,1])
    to numpy uint8 arrays (H, W, C), vertically flipped to match your GUI convention.
    """
    imgs: List[np.ndarray] = []
    for ci in cam_infos:
        # (C,H,W) -> (H,W,C), to CPU, to numpy, flip vertically, scale to 0..255, cast
        arr = (ci.original_image.permute(1, 2, 0).cpu().numpy()[::-1, :, :] * 255.0).astype(np.uint8)
        imgs.append(arr)
    return imgs


def load_from_output(
    output_dir: Path,
    *,
    device: torch.device | None = None,
    preload_images: bool = True,
    logger: Optional[logging.Logger] = None,
) -> Tuple[point_cloud.PointCloud, Dict[str, Any]]:
    """
    Load a trained scene from an output folder produced by training.

    Returns:
        pointcloud: the PointCloud instance (model is NOT restored here)
        scene_data: dict with keys:
            - config: OmegaConf config
            - train_cam_infos, test_cam_infos: lists of camera infos
            - train_images, test_images: lists of np.uint8 images (if preload_images=True)
            - dt_step: float from config.scene.dt
            - dynamic_sampling: bool from config.scene.dynamic_sampling
            - available_iterations: sorted list of checkpoint iteration ints
            - path_available_iterations: Path to the 'models' directory
    """
    log = logger or logging.getLogger(__name__)
    output_dir = output_dir.resolve()

    # --- config
    cfg_path = output_dir / "config" / "config.yml"
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config file not found: {cfg_path}")
    try:
        cfg = OmegaConf.load(str(cfg_path))
    except Exception as e:
        raise RuntimeError(f"Failed to load config: {cfg_path}") from e

    # --- checkpoints list
    models_dir = output_dir / "models"
    available_iterations = _parse_available_iterations(models_dir, log)
    if not available_iterations:
        log.warning("No checkpoints found in %s", models_dir)
    print("available_iterations", available_iterations)
    # --- point cloud + scene
    dev = device or torch.device("cuda")
    pc = point_cloud.PointCloud(data_type="float32", device=dev)
    scn = scene.Scene(
        config=cfg,
        pointcloud=pc,
        train_resolution_scales=cfg.scene.train_resolution_scales,
        test_resolution_scales=cfg.scene.test_resolution_scales,
        init_pc=False,
    )

    train_cam_infos = scn.getTrainCameras()
    test_cam_infos = scn.getTestCameras()

    # --- images (optional preload to uint8 HxWxC)
    if preload_images:
        train_images = _to_uint8_images(train_cam_infos)
        test_images = _to_uint8_images(test_cam_infos)
    else:
        train_images = []
        test_images = []

    scene_data: Dict[str, Any] = {
        "config": cfg,
        "train_cam_infos": train_cam_infos,
        "test_cam_infos": test_cam_infos,
        "train_images": train_images,
        "test_images": test_images,
        "dt_step": cfg.scene.dt,
        "dynamic_sampling": cfg.scene.dynamic_sampling,
        "available_iterations": available_iterations,
        "path_available_iterations": str(models_dir),
    }
    return pc, scene_data


def load_from_ply(
    ply_path: Path,
    *,
    device: torch.device | None = None,
) -> Tuple[point_cloud.PointCloud, Dict[str, Any]]:
    """
    Load a point cloud directly from a rg_ply file.

    Returns:
        pointcloud: the PointCloud instance (data loaded from ply)
        scene_data: dict with keys:
            - dt_step: default timestep (0.005)
            - dynamic_sampling: True
    """
    if not ply_path.exists() or not ply_path.is_file():
        raise FileNotFoundError(f"PLY file not found: {ply_path}")

    dev = device or torch.device("cuda")
    pc = point_cloud.PointCloud(data_type="float32", device=dev)
    # Some APIs expect str rather than Path
    pc.load_from_rg_ply(str(ply_path.resolve()))

    scene_data: Dict[str, Any] = {
        "dt_step": 0.005,
        "dynamic_sampling": True,
    }
    return pc, scene_data
