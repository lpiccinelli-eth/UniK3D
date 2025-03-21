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

SAVE = False
BASE_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "assets", "demo"
)


def infer(model, rgb_path, camera_path, rays=None):
    rgb = np.array(Image.open(rgb_path))
    rgb_torch = torch.from_numpy(rgb).permute(2, 0, 1)

    camera = None
    if camera_path is not None:
        with open(camera_path, "r") as f:
            camera_dict = json.load(f)

        params = torch.tensor(camera_dict["params"])
        name = camera_dict["name"]
        assert name in ["Fisheye624", "Spherical", "OPENCV", "Pinhole", "MEI"]
        camera = eval(name)(params=params)

    outputs = model.infer(rgb=rgb_torch, camera=camera, normalize=True, rays=rays)

    return rgb_torch, outputs


def infer_equirectangular(model, rgb_path):
    rgb = np.array(Image.open(rgb_path))
    rgb_torch = torch.from_numpy(rgb).permute(2, 0, 1)

    # assuming full equirectangular image horizontally
    H, W = rgb.shape[:2]
    hfov_half = np.pi
    vfov_half = np.pi * H / W
    assert vfov_half <= np.pi / 2

    params = [W, H, hfov_half, vfov_half]
    camera = Spherical(params=torch.tensor([1.0] * 4 + params))

    outputs = model.infer(rgb=rgb_torch, camera=camera, normalize=True)
    return rgb_torch, outputs


def save(rgb, outputs, name, base_path, save_pointcloud=False):
    depth = outputs["depth"]
    rays = outputs["rays"]
    points = outputs["points"]

    depth = depth.cpu().numpy()
    rays = ((rays + 1) * 127.5).clip(0, 255)

    Image.fromarray(colorize(depth.squeeze())).save(
        os.path.join(base_path, f"{name}_depth.png")
    )
    Image.fromarray(rgb.squeeze().permute(1, 2, 0).cpu().numpy()).save(
        os.path.join(base_path, f"{name}_rgb.png")
    )
    Image.fromarray(rays.squeeze().permute(1, 2, 0).byte().cpu().numpy()).save(
        os.path.join(base_path, f"{name}_rays.png")
    )

    if save_pointcloud:
        predictions_3d = points.permute(0, 2, 3, 1).reshape(-1, 3).cpu().numpy()
        rgb = rgb.permute(1, 2, 0).reshape(-1, 3).cpu().numpy()
        save_file_ply(predictions_3d, rgb, os.path.join(base_path, f"{name}.ply"))


def demo(model):
    # RGB + CAMERA
    rgb, outputs = infer(
        model,
        os.path.join(BASE_PATH, f"scannet.jpg"),
        os.path.join(BASE_PATH, "scannet.json"),
    )
    if SAVE:
        save(rgb, outputs, name="scannet", base_path=BASE_PATH)

    # get GT and pred
    pts_pred = outputs["points"].squeeze().cpu().permute(1, 2, 0).numpy()
    pts_gt = np.load("./assets/demo/scannet.npy").astype(float)
    mask = np.linalg.norm(pts_gt, axis=-1) > 0
    error = np.linalg.norm(pts_pred - pts_gt, axis=-1)
    error = np.mean(error[mask] ** 2) ** 0.5

    # Trade-off between speed and resolution
    model.resolution_level = 1
    rgb, outputs = infer(
        model,
        os.path.join(BASE_PATH, f"scannet.jpg"),
        os.path.join(BASE_PATH, "scannet.json"),
    )
    if SAVE:
        save(rgb, outputs, name="scannet_lowres", base_path=BASE_PATH)

    # RGB
    rgb, outputs = infer(model, os.path.join(BASE_PATH, f"poorthings.jpg"), None)
    if SAVE:
        save(rgb, outputs, name="poorthings", base_path=BASE_PATH)

    # RGB + CAMERA
    rgb, outputs = infer(
        model,
        os.path.join(BASE_PATH, f"dl3dv.png"),
        os.path.join(BASE_PATH, "dl3dv.json"),
    )
    if SAVE:
        save(rgb, outputs, name="dl3dv", base_path=BASE_PATH)

    # EQUIRECTANGULAR
    rgb, outputs = infer_equirectangular(
        model, os.path.join(BASE_PATH, f"equirectangular.jpg")
    )
    if SAVE:
        save(rgb, outputs, name="equirectangular", base_path=BASE_PATH)

    print("Output keys are", outputs.keys())

    if SAVE:
        print("Done! Results saved in", BASE_PATH)

    print(f"RMSE on 3D clouds for ScanNet++ sample: {100*error:.1f}cm")


if __name__ == "__main__":
    print("Torch version:", torch.__version__)
    type_ = "l"  # available types: s, b, l
    name = f"unik3d-vit{type_}"
    model = UniK3D.from_pretrained(f"lpiccinelli/{name}")

    # set resolution level in [0,10) and output interpolation
    model.resolution_level = 9
    model.interpolation_mode = "bilinear"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()

    demo(model)
