import argparse
import json
import os

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

from unik3d.models import UniK3D
from unik3d.utils.camera import (MEI, OPENCV, BatchCamera, Fisheye624, Pinhole,
                                 Spherical)
from unik3d.utils.visualization import colorize, save_file_ply


def save(rgb, outputs, name, base_path, save_map=False, save_pointcloud=False):

    os.makedirs(base_path, exist_ok=True)

    depth = outputs["depth"]
    rays = outputs["rays"]
    points = outputs["points"]

    depth = depth.cpu().numpy()
    rays = ((rays + 1) * 127.5).clip(0, 255)
    if save_map:
        Image.fromarray(colorize(depth.squeeze())).save(
            os.path.join(base_path, f"{name}_depth.png")
        )
        Image.fromarray(rays.squeeze().permute(1, 2, 0).byte().cpu().numpy()).save(
            os.path.join(base_path, f"{name}_rays.png")
        )

    if save_pointcloud:
        predictions_3d = points.permute(0, 2, 3, 1).reshape(-1, 3).cpu().numpy()
        rgb = rgb.permute(1, 2, 0).reshape(-1, 3).cpu().numpy()
        save_file_ply(predictions_3d, rgb, os.path.join(base_path, f"{name}.ply"))


def infer(model, args):
    rgb = np.array(Image.open(args.input))
    rgb_torch = torch.from_numpy(rgb).permute(2, 0, 1)

    camera = None
    camera_path = args.camera_path
    if camera_path is not None:
        with open(camera_path, "r") as f:
            camera_dict = json.load(f)

        params = torch.tensor(camera_dict["params"])
        name = camera_dict["name"]
        assert name in ["Fisheye624", "Spherical", "OPENCV", "Pinhole", "MEI"]
        camera = eval(name)(params=params)

    outputs = model.infer(rgb=rgb_torch, camera=camera, normalize=True, rays=None)
    name = args.input.split("/")[-1].split(".")[0]
    save(rgb_torch, outputs, name=name, base_path=args.output, save_map=args.save, save_pointcloud=args.save_ply)


if __name__ == "__main__":
    # Arguments
    parser = argparse.ArgumentParser(description='Training script', conflict_handler='resolve')
    parser.add_argument("--input", type=str, required=True, help="Path to input image.")
    parser.add_argument("--output", type=str, required=True, help="Path to output directory.")
    parser.add_argument("--config-file", type=str, required=True, default="./configs/eval/vitl.json", help="Path to config file. Please check ./configs/eval.")
    parser.add_argument("--camera-path", type=str, default=None, help="Path to camera parameters json file. See assets/demo for a few examples. The file needs a 'name' field with the camera model from unik3d/utils/camera.py and a 'params' field with the camera parameters as in the corresponding class docstring.")
    parser.add_argument("--save", action="store_true", help="Save outputs as (colorized) png.")
    parser.add_argument("--save-ply", action="store_true", help="Save pointcloud as ply.")
    parser.add_argument("--resolution-level", type=int, default=9, help="Resolution level in [0,10). Higher values means it will resize to larger resolution which increases details but decreases speed. Lower values lead to opposite.", choices=list(range(10)))
    parser.add_argument("--interpolation-mode", type=str, default="bilinear", help="Output interpolation.", choices=["nearest", "nearest-exact", "bilinear"])
    args = parser.parse_args()

    print("Torch version:", torch.__version__)
    version = args.config_file.split("/")[-1].split(".")[0]
    name = f"unik3d-{version}"
    model = UniK3D.from_pretrained(f"lpiccinelli/{name}")

    model.resolution_level = args.resolution_level
    model.interpolation_mode = args.interpolation_mode

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()

    infer(model, args)
