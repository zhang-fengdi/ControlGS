#
# This file is part of ControlGS.
# 
# ControlGS is a derivative work based on 3D Gaussian Splatting (3DGS), originally developed by 
# the GRAPHDECO research group at Inria.
#
# Copyright (C) 2025, Fengdi Zhang
# Portions copyright (C) 2023, Inria - GRAPHDECO research group
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact: zhangfd22@mails.tsinghua.edu.cn
# Original authors contact: george.drettakis@inria.fr
#

import os

import torch
import torchvision.utils

from scene.cameras import Camera
import numpy as np
from utils.general_utils import PILtoTorch
from utils.graphics_utils import fov2focal
import cv2

WARNED = False

def loadCam(args, id, cam_info, resolution_scale, is_nerf_synthetic, is_test_dataset):
    if cam_info.depth_path != "":
        try:
            if is_nerf_synthetic:
                invdepthmap = cv2.imread(cam_info.depth_path, -1).astype(np.float32) / 512
            else:
                invdepthmap = cv2.imread(cam_info.depth_path, -1).astype(np.float32) / float(2**16)

        except FileNotFoundError:
            print(f"Error: The depth file at path '{cam_info.depth_path}' was not found.")
            raise
        except IOError:
            print(f"Error: Unable to open the image file '{cam_info.depth_path}'. It may be corrupted or an unsupported format.")
            raise
        except Exception as e:
            print(f"An unexpected error occurred when trying to read depth at {cam_info.depth_path}: {e}")
            raise
    else:
        invdepthmap = None
        
    orig_w, orig_h = cam_info.image.size
    if args.resolution in [1, 2, 4, 8]:
        resolution = round(orig_w/(resolution_scale * args.resolution)), round(orig_h/(resolution_scale * args.resolution))
    else:  # should be a type that converts to float
        if args.resolution == -1:
            if orig_w > 1600:
                global WARNED
                if not WARNED:
                    print("[ INFO ] Encountered quite large input images (>1.6K pixels width), rescaling to 1.6K.\n "
                        "If this is not desired, please explicitly specify '--resolution/-r' as 1")
                    WARNED = True
                global_down = orig_w / 1600
            else:
                global_down = 1
        else:
            global_down = orig_w / args.resolution

        scale = float(global_down) * float(resolution_scale)
        resolution = (int(orig_w / scale), int(orig_h / scale))

    image = PILtoTorch(cam_info.image, resolution)
    mask = None
    if cam_info.mask is not None:
        mask = PILtoTorch(cam_info.mask, resolution)
        if mask.shape[0] != 1:
            mask = mask[:1, ...]
        mask[mask > 0] = 1.

        if args.preview_masked_images:
            masked_preview = torch.clone(image)
            masked_preview[0, mask[0] == 0.] /= 4.
            masked_preview[0, mask[0] == 0.] += 0.75
            preview_save_path = os.path.join(args.model_path, "mask_preview", cam_info.image_name.replace("/", "_"))
            os.makedirs(os.path.dirname(preview_save_path), exist_ok=True)
            torchvision.utils.save_image(masked_preview, preview_save_path)

    return Camera(resolution, colmap_id=cam_info.uid, R=cam_info.R, T=cam_info.T, 
                  FoVx=cam_info.FovX, FoVy=cam_info.FovY, depth_params=cam_info.depth_params,
                  image=image, mask=mask, invdepthmap=invdepthmap,
                  image_name=cam_info.image_name, uid=id, data_device=args.data_device,
                  train_test_exp=args.train_test_exp, is_test_dataset=is_test_dataset, is_test_view=cam_info.is_test)

def cameraList_from_camInfos(cam_infos, resolution_scale, args, is_nerf_synthetic, is_test_dataset):
    camera_list = []

    for id, c in enumerate(cam_infos):
        camera_list.append(loadCam(args, id, c, resolution_scale, is_nerf_synthetic, is_test_dataset))

    return camera_list

def camera_to_JSON(id, camera : Camera):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = camera.R.transpose()
    Rt[:3, 3] = camera.T
    Rt[3, 3] = 1.0

    W2C = np.linalg.inv(Rt)
    pos = W2C[:3, 3]
    rot = W2C[:3, :3]
    serializable_array_2d = [x.tolist() for x in rot]
    camera_entry = {
        'id' : id,
        'img_name' : camera.image_name,
        'width' : camera.width,
        'height' : camera.height,
        'position': pos.tolist(),
        'rotation': serializable_array_2d,
        'fy' : fov2focal(camera.FovY, camera.height),
        'fx' : fov2focal(camera.FovX, camera.width)
    }
    return camera_entry