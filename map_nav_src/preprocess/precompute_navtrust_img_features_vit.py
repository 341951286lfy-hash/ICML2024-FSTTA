#!/usr/bin/env python3
import argparse
import json
import math
import os
import sys
from pathlib import Path

import h5py
import numpy as np
import torch
import timm
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

import MatterSim

# 正确加入仓库中的 map_nav_src 目录
THIS_DIR = Path(__file__).resolve().parent
MAP_NAV_SRC_DIR = THIS_DIR.parent
sys.path.insert(0, str(MAP_NAV_SRC_DIR))

from utils.navtrust_rgb import (  # noqa: E402
    NAVTRUST_RGB_CORRUPTIONS,
    apply_navtrust_rgb_corruption,
    make_deterministic_rng,
)


VIEWPOINT_SIZE = 36
WIDTH = 640
HEIGHT = 480
VFOV = 60


def parse_args():
    parser = argparse.ArgumentParser("Precompute NavTrust-corrupted ViT features for FSTTA")
    parser.add_argument("--model_name", type=str, default="vit_base_patch16_224")
    parser.add_argument(
        "--checkpoint_file",
        type=str,
        default=None,
        help="本地 ViT checkpoint；若为空且 --use_timm_pretrained=True，则使用 timm 的 ImageNet 预训练权重"
    )
    parser.add_argument("--use_timm_pretrained", action="store_true", default=False)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch_size", type=int, default=12)

    parser.add_argument("--connectivity_dir", type=str, required=True)
    parser.add_argument("--scan_dir", type=str, required=True)
    parser.add_argument("--output_file", type=str, required=True)
    parser.add_argument("--overwrite", action="store_true", default=False)

    parser.add_argument(
        "--corruption",
        type=str,
        default="motion_blur",
        choices=NAVTRUST_RGB_CORRUPTIONS
    )
    parser.add_argument("--severity", type=float, default=0.6)
    parser.add_argument("--base_seed", type=int, default=0)

    return parser.parse_args()


def load_viewpoint_ids(connectivity_dir: str):
    viewpoint_ids = []

    scans_file = os.path.join(connectivity_dir, "scans.txt")
    with open(scans_file, "r") as f:
        scans = [x.strip() for x in f.readlines() if x.strip()]

    for scan in scans:
        conn_file = os.path.join(connectivity_dir, f"{scan}_connectivity.json")
        with open(conn_file, "r") as f:
            data = json.load(f)

        for item in data:
            if item["included"]:
                viewpoint_ids.append((scan, item["image_id"]))

    return viewpoint_ids


def build_simulator(connectivity_dir: str, scan_dir: str):
    sim = MatterSim.Simulator()
    sim.setDatasetPath(scan_dir)
    sim.setNavGraphPath(connectivity_dir)
    sim.setRenderingEnabled(True)
    sim.setDiscretizedViewingAngles(True)
    sim.setCameraResolution(WIDTH, HEIGHT)
    sim.setCameraVFOV(math.radians(VFOV))
    sim.setBatchSize(1)
    sim.initialize()
    return sim


def render_36_views(sim, scan_id: str, viewpoint_id: str):
    """
    返回长度为 36 的 RGB 图像列表，每个元素 shape 为 [H, W, 3]。
    """
    images = []

    for ix in range(VIEWPOINT_SIZE):
        if ix == 0:
            sim.newEpisode([scan_id], [viewpoint_id], [0.0], [math.radians(-30)])
        elif ix % 12 == 0:
            sim.makeAction([0], [1.0], [1.0])
        else:
            sim.makeAction([0], [1.0], [0.0])

        state = sim.getState()[0]
        assert state.viewIndex == ix

        rgb = np.array(state.rgb, copy=True)
        if rgb.dtype != np.uint8:
            rgb = np.clip(rgb, 0, 255).astype(np.uint8)

        images.append(rgb)

    return images


def build_transform():
    return transforms.Compose([
        transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225)
        ),
    ])


def _strip_prefix_if_needed(state_dict):
    new_sd = {}
    for k, v in state_dict.items():
        nk = k
        if nk.startswith("module."):
            nk = nk[len("module."):]
        if nk.startswith("model."):
            nk = nk[len("model."):]
        new_sd[nk] = v
    return new_sd


def build_model(args):
    model = timm.create_model(
        args.model_name,
        pretrained=(args.use_timm_pretrained and args.checkpoint_file is None),
        num_classes=0,
    )

    if args.checkpoint_file is not None:
        ckpt = torch.load(args.checkpoint_file, map_location="cpu")
        if isinstance(ckpt, dict):
            if "state_dict" in ckpt:
                ckpt = ckpt["state_dict"]
            elif "model" in ckpt:
                ckpt = ckpt["model"]

        ckpt = _strip_prefix_if_needed(ckpt)
        missing, unexpected = model.load_state_dict(ckpt, strict=False)
        print(f"[ViT] loaded checkpoint: {args.checkpoint_file}")
        print(f"[ViT] missing keys: {len(missing)}, unexpected keys: {len(unexpected)}")

    model.eval()
    return model


@torch.no_grad()
def extract_features(model, transform, images_rgb, device: str, batch_size: int):
    tensors = []
    for img in images_rgb:
        pil = Image.fromarray(img)
        tensors.append(transform(pil))

    x = torch.stack(tensors, dim=0)

    feats = []
    for st in range(0, x.size(0), batch_size):
        ed = min(st + batch_size, x.size(0))
        xb = x[st:ed].to(device, non_blocking=True)
        fb = model(xb)

        # 兼容不同 timm 版本输出
        if isinstance(fb, (tuple, list)):
            fb = fb[0]

        if fb.ndim == 3:       # [B, N, C]
            fb = fb[:, 0]
        elif fb.ndim == 4:     # [B, C, H, W]
            fb = fb.mean(dim=(2, 3))

        feats.append(fb.detach().cpu())

    feats = torch.cat(feats, dim=0).numpy().astype(np.float32)
    return feats


def main():
    args = parse_args()

    if (not args.use_timm_pretrained) and (args.checkpoint_file is None):
        raise ValueError("必须提供 --checkpoint_file，或者显式加 --use_timm_pretrained")

    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)

    viewpoint_ids = load_viewpoint_ids(args.connectivity_dir)
    print(f"[INFO] loaded {len(viewpoint_ids)} included viewpoints")
    print(f"[INFO] corruption={args.corruption}, severity={args.severity}")

    sim = build_simulator(args.connectivity_dir, args.scan_dir)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = build_model(args).to(device)
    transform = build_transform()

    h5_mode = "w" if args.overwrite else "a"
    with h5py.File(args.output_file, h5_mode) as fout:
        for scan_id, viewpoint_id in tqdm(viewpoint_ids, desc="precompute"):
            key = f"{scan_id}_{viewpoint_id}"

            if (key in fout) and (not args.overwrite):
                continue

            images = render_36_views(sim, scan_id, viewpoint_id)

            corrupted_images = []
            for view_idx, img in enumerate(images):
                rng_key = f"{scan_id}|{viewpoint_id}|{view_idx}|{args.corruption}|{args.severity}"
                rng = make_deterministic_rng(rng_key, args.base_seed)

                cimg = apply_navtrust_rgb_corruption(
                    img_rgb=img,
                    corruption=args.corruption,
                    severity=args.severity,
                    rng=rng,
                )
                corrupted_images.append(cimg)

            feats = extract_features(
                model=model,
                transform=transform,
                images_rgb=corrupted_images,
                device=str(device),
                batch_size=args.batch_size,
            )

            if feats.shape[0] != 36:
                raise RuntimeError(f"{key}: expected 36 views, got {feats.shape}")
            if feats.ndim != 2:
                raise RuntimeError(f"{key}: expected [36, C], got {feats.shape}")

            if key in fout:
                del fout[key]
            fout.create_dataset(key, data=feats, compression="gzip")

    print(f"[DONE] saved to: {args.output_file}")


if __name__ == "__main__":
    main()
