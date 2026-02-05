#!/usr/bin/env python3
"""
gsplat Depth-Supervised Trainer

A self-contained training script that extends gsplat's simple_trainer with:
1. LiDAR depth supervision loss (confidence-weighted L1)
2. Normal consistency regularization
3. Dense LiDAR point cloud initialization (optional)

Designed to run on a CUDA GPU (RunPod, Colab, etc.)

Usage:
    python gsplat_depth_trainer.py \
        --data_dir /workspace/output_colmap_sfm \
        --depth_dir /workspace/depth_data \
        --data_factor 2 \
        --result_dir /workspace/results_depth \
        --max_steps 30000 \
        --depth_lambda 0.5 \
        --normal_lambda 0.05

    # With dense initialization:
    python gsplat_depth_trainer.py \
        --data_dir /workspace/output_colmap_sfm \
        --depth_dir /workspace/depth_data \
        --dense_init_ply /workspace/dense_pointcloud.ply \
        --data_factor 2 \
        --result_dir /workspace/results_dense \
        --max_steps 30000
"""

import json
import math
import os
import struct
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, optim

# ── gsplat imports ───────────────────────────────────────────
try:
    import gsplat
    from gsplat import rasterization
    print(f"gsplat version: {gsplat.__version__}")
except ImportError:
    print("ERROR: gsplat not installed. Run: pip install gsplat==1.5.3")
    sys.exit(1)

try:
    import tyro
except ImportError:
    print("ERROR: tyro not installed. Run: pip install tyro")
    sys.exit(1)

try:
    from fused_ssim import fused_ssim
    HAS_SSIM = True
except ImportError:
    HAS_SSIM = False
    print("Warning: fused-ssim not installed, using L1 only")

# ── COLMAP data loader ──────────────────────────────────────

def read_cameras_binary(path: str) -> dict:
    """Read COLMAP cameras.bin."""
    cameras = {}
    with open(path, "rb") as f:
        num = struct.unpack("<Q", f.read(8))[0]
        for _ in range(num):
            cam_id = struct.unpack("<i", f.read(4))[0]
            model_id = struct.unpack("<i", f.read(4))[0]
            w = struct.unpack("<Q", f.read(8))[0]
            h = struct.unpack("<Q", f.read(8))[0]
            # PINHOLE: 4 params, SIMPLE_PINHOLE: 3 params
            num_params = {0: 3, 1: 4, 2: 4, 3: 5, 4: 4, 5: 5}.get(model_id, 4)
            params = struct.unpack(f"<{num_params}d", f.read(8 * num_params))
            cameras[cam_id] = {
                "model_id": model_id, "width": w, "height": h, "params": params
            }
    return cameras


def read_images_binary(path: str) -> dict:
    """Read COLMAP images.bin."""
    images = {}
    with open(path, "rb") as f:
        num = struct.unpack("<Q", f.read(8))[0]
        for _ in range(num):
            img_id = struct.unpack("<i", f.read(4))[0]
            qvec = struct.unpack("<4d", f.read(32))
            tvec = struct.unpack("<3d", f.read(24))
            cam_id = struct.unpack("<i", f.read(4))[0]
            name = b""
            while True:
                c = f.read(1)
                if c == b"\x00":
                    break
                name += c
            num_pts = struct.unpack("<Q", f.read(8))[0]
            f.read(num_pts * 24)  # skip 2D points
            images[img_id] = {
                "qvec": np.array(qvec),
                "tvec": np.array(tvec),
                "camera_id": cam_id,
                "name": name.decode("utf-8"),
            }
    return images


def read_points3d_binary(path: str) -> np.ndarray:
    """Read COLMAP points3D.bin, return (N, 3) positions and (N, 3) colors."""
    positions = []
    colors = []
    with open(path, "rb") as f:
        num = struct.unpack("<Q", f.read(8))[0]
        for _ in range(num):
            pt_id = struct.unpack("<Q", f.read(8))[0]
            xyz = struct.unpack("<3d", f.read(24))
            rgb = struct.unpack("<3B", f.read(3))
            error = struct.unpack("<d", f.read(8))[0]
            track_len = struct.unpack("<Q", f.read(8))[0]
            f.read(track_len * 8)
            positions.append(xyz)
            colors.append(rgb)
    return np.array(positions, dtype=np.float32), np.array(colors, dtype=np.uint8)


def qvec2rotmat(qvec):
    """Convert COLMAP quaternion (qw, qx, qy, qz) to 3x3 rotation matrix."""
    w, x, y, z = qvec
    R = np.array([
        [1 - 2*y*y - 2*z*z, 2*x*y - 2*w*z, 2*x*z + 2*w*y],
        [2*x*y + 2*w*z, 1 - 2*x*x - 2*z*z, 2*y*z - 2*w*x],
        [2*x*z - 2*w*y, 2*y*z + 2*w*x, 1 - 2*x*x - 2*y*y],
    ])
    return R


# ── Dataset ──────────────────────────────────────────────────

class COLMAPDataset:
    """Load COLMAP dataset with optional depth maps."""

    def __init__(
        self,
        data_dir: str,
        depth_dir: Optional[str] = None,
        data_factor: int = 1,
        device: torch.device = torch.device("cuda"),
    ):
        self.device = device
        self.data_dir = Path(data_dir)
        self.depth_dir = Path(depth_dir) if depth_dir else None
        self.data_factor = data_factor

        sparse_dir = self.data_dir / "sparse" / "0"

        # Read COLMAP data
        cameras = read_cameras_binary(str(sparse_dir / "cameras.bin"))
        images = read_images_binary(str(sparse_dir / "images.bin"))
        points3d, point_colors = read_points3d_binary(str(sparse_dir / "points3D.bin"))

        self.points3d = torch.from_numpy(points3d).float().to(device)
        self.point_colors = torch.from_numpy(point_colors).float().to(device) / 255.0

        # Camera intrinsics (assume single camera)
        cam = list(cameras.values())[0]
        self.width = int(cam["width"] // data_factor)
        self.height = int(cam["height"] // data_factor)

        if cam["model_id"] == 1:  # PINHOLE
            fx, fy, cx, cy = cam["params"]
        elif cam["model_id"] == 0:  # SIMPLE_PINHOLE
            f, cx, cy = cam["params"]
            fx = fy = f
        else:
            raise ValueError(f"Unsupported camera model: {cam['model_id']}")

        self.fx = fx / data_factor
        self.fy = fy / data_factor
        self.cx = cx / data_factor
        self.cy = cy / data_factor

        # Image directory
        images_dir = self.data_dir / f"images_{data_factor}"
        if not images_dir.exists():
            images_dir = self.data_dir / "images"

        # Load images and camera poses
        self.image_names = []
        self.images = []
        self.viewmats = []  # world-to-camera 4x4
        self.Ks = []        # 3x3 intrinsics

        K = torch.tensor([
            [self.fx, 0, self.cx],
            [0, self.fy, self.cy],
            [0, 0, 1],
        ], dtype=torch.float32, device=device)

        import imageio.v3 as iio

        for img_id in sorted(images.keys()):
            img_data = images[img_id]
            img_path = images_dir / img_data["name"]

            if not img_path.exists():
                continue

            # Load and resize image
            img = iio.imread(str(img_path))
            if data_factor > 1 and "images_" not in str(images_dir):
                import cv2
                img = cv2.resize(img, (self.width, self.height))
            img_tensor = torch.from_numpy(img).float().to(device) / 255.0

            # World-to-camera matrix
            R = qvec2rotmat(img_data["qvec"])
            t = img_data["tvec"]
            w2c = np.eye(4)
            w2c[:3, :3] = R
            w2c[:3, 3] = t
            w2c_tensor = torch.from_numpy(w2c).float().to(device)

            self.image_names.append(img_data["name"])
            self.images.append(img_tensor)
            self.viewmats.append(w2c_tensor)
            self.Ks.append(K.clone())

        print(f"Loaded {len(self.images)} images at {self.width}x{self.height}")
        print(f"COLMAP points: {len(self.points3d):,}")
        print(f"Camera: fx={self.fx:.1f} fy={self.fy:.1f} cx={self.cx:.1f} cy={self.cy:.1f}")

        # Load depth metadata if available
        self.depth_meta = {}
        if self.depth_dir and (self.depth_dir / "depth_meta.json").exists():
            with open(self.depth_dir / "depth_meta.json") as f:
                self.depth_meta = json.load(f)
            print(f"Depth maps available for {len(self.depth_meta)} images")

    def __len__(self):
        return len(self.images)

    def get_train_item(self, idx: int) -> dict:
        """Get training item: image, camera, and optional depth."""
        item = {
            "image": self.images[idx],          # (H, W, 3)
            "viewmat": self.viewmats[idx],      # (4, 4)
            "K": self.Ks[idx],                  # (3, 3)
            "image_name": self.image_names[idx],
            "width": self.width,
            "height": self.height,
        }

        # Add depth if available
        image_name = self.image_names[idx]
        if image_name in self.depth_meta:
            info = self.depth_meta[image_name]
            depth = np.load(str(self.depth_dir / "depths" / info["depth_file"]))
            conf = np.load(str(self.depth_dir / "confidences" / info["conf_file"]))

            # Resize to training resolution if needed
            if depth.shape != (self.height, self.width):
                import cv2
                depth = cv2.resize(depth, (self.width, self.height),
                                   interpolation=cv2.INTER_NEAREST)
                conf = cv2.resize(conf, (self.width, self.height),
                                  interpolation=cv2.INTER_NEAREST)

            item["depth"] = torch.from_numpy(depth).float().to(self.device)
            item["confidence"] = torch.from_numpy(conf).float().to(self.device)

        return item


# ── Gaussian Model ───────────────────────────────────────────

class GaussianModel:
    """3D Gaussian parameters."""

    def __init__(
        self,
        positions: Tensor,
        colors: Optional[Tensor] = None,
        sh_degree: int = 3,
        device: torch.device = torch.device("cuda"),
    ):
        self.device = device
        self.sh_degree = sh_degree
        n = len(positions)

        # Positions
        self.means = torch.nn.Parameter(positions.clone().to(device))

        # Scales (log space)
        avg_dist = self._compute_avg_distances(positions, k=3)
        self.scales = torch.nn.Parameter(
            torch.log(avg_dist.unsqueeze(-1).repeat(1, 3)).to(device)
        )

        # Rotations (quaternion: w, x, y, z)
        self.quats = torch.nn.Parameter(
            torch.zeros(n, 4, device=device)
        )
        self.quats.data[:, 0] = 1.0  # Identity rotation

        # Opacities (logit space)
        self.opacities = torch.nn.Parameter(
            torch.logit(torch.full((n,), 0.5, device=device))
        )

        # Spherical harmonics (DC + higher order)
        num_sh = (sh_degree + 1) ** 2
        if colors is not None:
            # Initialize DC from colors
            SH_C0 = 0.28209479177387814
            sh = torch.zeros(n, num_sh, 3, device=device)
            sh[:, 0, :] = (colors.to(device) - 0.5) / SH_C0
            self.sh = torch.nn.Parameter(sh)
        else:
            self.sh = torch.nn.Parameter(
                torch.zeros(n, num_sh, 3, device=device)
            )

        print(f"Initialized {n:,} Gaussians (sh_degree={sh_degree})")

    def _compute_avg_distances(self, points: Tensor, k: int = 3) -> Tensor:
        """Compute average distance to k nearest neighbors."""
        # For large point clouds, use a random subsample for speed
        n = len(points)
        if n > 100_000:
            # Estimate from subsample
            idx = torch.randperm(n)[:10000]
            subset = points[idx].float()
            dists = torch.cdist(subset, subset)
            dists.fill_diagonal_(float('inf'))
            knn_dists = dists.topk(k, largest=False).values.mean(dim=1)
            avg = knn_dists.median()
            return torch.full((n,), avg.item())
        else:
            points_f = points.float()
            dists = torch.cdist(points_f, points_f)
            dists.fill_diagonal_(float('inf'))
            knn_dists = dists.topk(k, largest=False).values.mean(dim=1)
            return knn_dists.clamp(min=1e-6)

    @property
    def num_gaussians(self):
        return len(self.means)

    def get_params(self, lr_scale: float = 1.0) -> list:
        """Get parameter groups for optimizer."""
        return [
            {"params": [self.means], "lr": 1.6e-4 * lr_scale, "name": "means"},
            {"params": [self.scales], "lr": 5e-3, "name": "scales"},
            {"params": [self.quats], "lr": 1e-3, "name": "quats"},
            {"params": [self.opacities], "lr": 5e-2, "name": "opacities"},
            {"params": [self.sh], "lr": 2.5e-3, "name": "sh"},
        ]


# ── Depth Loss ───────────────────────────────────────────────

def compute_depth_loss(
    rendered_depth: Tensor,
    gt_depth: Tensor,
    confidence: Tensor,
    rgb_image: Tensor = None,
    min_confidence: int = 1,
    adaptive: bool = True,
) -> Tensor:
    """
    Confidence-weighted L1 depth loss.

    Args:
        rendered_depth: (H, W) rendered median depth
        gt_depth: (H, W) LiDAR depth in meters
        confidence: (H, W) confidence (0/1/2)
        rgb_image: (H, W, 3) for adaptive edge weighting
        min_confidence: minimum confidence to supervise
        adaptive: use gradient-based edge weighting
    """
    if rendered_depth.dim() == 3:
        rendered_depth = rendered_depth.squeeze(-1)

    valid = (confidence >= min_confidence) & (gt_depth > 0.01)
    if valid.sum() < 100:
        return torch.tensor(0.0, device=rendered_depth.device)

    conf_weight = confidence.float() / 2.0
    error = torch.abs(rendered_depth - gt_depth)

    if adaptive and rgb_image is not None:
        gray = rgb_image.mean(dim=-1)
        gx = F.pad(torch.abs(gray[:, 1:] - gray[:, :-1]), (0, 1), mode='replicate')
        gy = F.pad(torch.abs(gray[1:, :] - gray[:-1, :]), (1, 0), mode='replicate')
        edge_weight = torch.exp(-10.0 * (gx + gy) / 2.0)
        error = error * edge_weight

    weighted = error * conf_weight * valid.float()
    return weighted.sum() / (valid.float() * conf_weight).sum().clamp(min=1.0)


def compute_normal_loss(
    rendered_depth: Tensor,
    gt_depth: Tensor,
    valid_mask: Tensor,
) -> Tensor:
    """Normal consistency loss from depth gradients."""
    def depth_normals(d):
        dx = F.pad(d[:, 1:] - d[:, :-1], (0, 1), mode='replicate')
        dy = F.pad(d[1:, :] - d[:-1, :], (1, 0), mode='replicate')
        n = torch.stack([-dx, -dy, torch.ones_like(d)], dim=-1)
        return F.normalize(n, p=2, dim=-1)

    if rendered_depth.dim() == 3:
        rendered_depth = rendered_depth.squeeze(-1)

    rn = depth_normals(rendered_depth)
    gn = depth_normals(gt_depth)
    cos = (rn * gn).sum(dim=-1)
    err = (1.0 - cos)[1:-1, 1:-1]
    mask = valid_mask[1:-1, 1:-1]

    if mask.sum() < 100:
        return torch.tensor(0.0, device=rendered_depth.device)
    return (err * mask.float()).sum() / mask.float().sum()


# ── Training Loop ────────────────────────────────────────────

@dataclass
class TrainConfig:
    """Training configuration."""

    # Data
    data_dir: str = "./data"
    depth_dir: Optional[str] = None
    dense_init_ply: Optional[str] = None
    data_factor: int = 2
    result_dir: str = "./results"

    # Training
    max_steps: int = 30000
    lr: float = 1.0
    sh_degree: int = 3

    # Depth supervision
    depth_lambda: float = 0.5
    normal_lambda: float = 0.05
    min_confidence: int = 1
    adaptive_depth: bool = True
    depth_start_step: int = 500

    # Densification
    densify_start: int = 500
    densify_stop: int = 15000
    densify_every: int = 100
    densify_grad_thresh: float = 0.0002
    densify_size_thresh: float = 0.01

    # Pruning
    prune_every: int = 100
    prune_opacity_thresh: float = 0.005
    prune_size_thresh: float = 0.1

    # Reset opacity
    reset_opacity_every: int = 3000
    reset_opacity_value: float = 0.01

    # Logging
    log_every: int = 100
    save_every: int = 5000

    # Dense init
    max_init_points: int = 500_000


def load_dense_points(ply_path: str, max_points: int) -> Tuple[np.ndarray, np.ndarray]:
    """Load PLY point cloud for dense initialization."""
    print(f"Loading dense point cloud: {ply_path}")
    positions = []
    colors = []

    with open(ply_path) as f:
        in_header = True
        for line in f:
            if in_header:
                if line.strip() == "end_header":
                    in_header = False
                continue
            parts = line.strip().split()
            if len(parts) >= 3:
                positions.append([float(parts[0]), float(parts[1]), float(parts[2])])
                if len(parts) >= 6:
                    colors.append([float(parts[3])/255, float(parts[4])/255, float(parts[5])/255])

    positions = np.array(positions, dtype=np.float32)
    colors = np.array(colors, dtype=np.float32) if colors else None

    if len(positions) > max_points:
        idx = np.random.choice(len(positions), max_points, replace=False)
        positions = positions[idx]
        if colors is not None:
            colors = colors[idx]

    print(f"  Loaded {len(positions):,} points")
    return positions, colors


def save_ply(path: str, model: GaussianModel):
    """Save trained Gaussians as standard 3DGS PLY."""
    means = model.means.detach().cpu().numpy()
    scales = model.scales.detach().cpu().numpy()
    quats = model.quats.detach().cpu().numpy()
    opacities = model.opacities.detach().cpu().numpy()
    sh = model.sh.detach().cpu().numpy()

    n = len(means)
    num_sh_total = sh.shape[1]

    with open(path, 'wb') as f:
        # Header
        header = f"""ply
format binary_little_endian 1.0
element vertex {n}
property float x
property float y
property float z
property float nx
property float ny
property float nz
"""
        # SH properties
        for i in range(3):
            header += f"property float f_dc_{i}\n"
        for i in range((num_sh_total - 1) * 3):
            header += f"property float f_rest_{i}\n"
        header += "property float opacity\n"
        for i in range(3):
            header += f"property float scale_{i}\n"
        for i in range(4):
            header += f"property float rot_{i}\n"
        header += "end_header\n"
        f.write(header.encode())

        # Data
        normals = np.zeros((n, 3), dtype=np.float32)
        for i in range(n):
            # Position
            f.write(struct.pack('<3f', *means[i]))
            # Normals (unused)
            f.write(struct.pack('<3f', *normals[i]))
            # SH DC
            f.write(struct.pack('<3f', sh[i, 0, 0], sh[i, 0, 1], sh[i, 0, 2]))
            # SH rest (interleaved)
            for j in range(1, num_sh_total):
                f.write(struct.pack('<3f', sh[i, j, 0], sh[i, j, 1], sh[i, j, 2]))
            # Opacity
            f.write(struct.pack('<f', opacities[i]))
            # Scale
            f.write(struct.pack('<3f', *scales[i]))
            # Rotation
            f.write(struct.pack('<4f', *quats[i]))


def train(cfg: TrainConfig):
    """Main training loop."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        print("WARNING: No CUDA GPU detected. Training will be very slow.")

    os.makedirs(cfg.result_dir, exist_ok=True)

    # Load dataset
    dataset = COLMAPDataset(
        data_dir=cfg.data_dir,
        depth_dir=cfg.depth_dir,
        data_factor=cfg.data_factor,
        device=device,
    )

    has_depth = len(dataset.depth_meta) > 0
    if has_depth:
        print(f"\n{'='*60}")
        print(f"DEPTH SUPERVISION ENABLED")
        print(f"  depth_lambda: {cfg.depth_lambda}")
        print(f"  normal_lambda: {cfg.normal_lambda}")
        print(f"  min_confidence: {cfg.min_confidence}")
        print(f"  starts at step: {cfg.depth_start_step}")
        print(f"{'='*60}\n")
    else:
        print("\nNo depth data — training with RGB only\n")

    # Initialize Gaussians
    if cfg.dense_init_ply and os.path.exists(cfg.dense_init_ply):
        positions, colors = load_dense_points(cfg.dense_init_ply, cfg.max_init_points)
        init_positions = torch.from_numpy(positions).to(device)
        init_colors = torch.from_numpy(colors).to(device) if colors is not None else None
    else:
        init_positions = dataset.points3d
        init_colors = dataset.point_colors

    model = GaussianModel(
        positions=init_positions,
        colors=init_colors,
        sh_degree=cfg.sh_degree,
        device=device,
    )

    # Optimizer
    optimizer = optim.Adam(model.get_params(lr_scale=cfg.lr), eps=1e-15)

    # Gradient accumulator for densification
    grad_accum = torch.zeros(model.num_gaussians, device=device)
    grad_count = torch.zeros(model.num_gaussians, device=device, dtype=torch.int32)

    # Training loop
    num_images = len(dataset)
    print(f"\nStarting training: {cfg.max_steps} steps, {num_images} images")
    print(f"Initial Gaussians: {model.num_gaussians:,}")

    start_time = time.time()
    losses_log = []

    for step in range(cfg.max_steps):
        # Random image
        idx = step % num_images
        item = dataset.get_train_item(idx)

        gt_image = item["image"]       # (H, W, 3)
        viewmat = item["viewmat"]      # (4, 4)
        K = item["K"]                  # (3, 3)
        W = item["width"]
        H = item["height"]

        # Forward pass: rasterize Gaussians
        renders, alphas, info = rasterization(
            means=model.means,
            quats=model.quats / model.quats.norm(dim=-1, keepdim=True),
            scales=torch.exp(model.scales),
            opacities=torch.sigmoid(model.opacities),
            colors=model.sh,
            viewmats=viewmat.unsqueeze(0),
            Ks=K.unsqueeze(0),
            width=W,
            height=H,
            sh_degree=cfg.sh_degree,
            render_mode="RGB+ED",  # RGB + Expected Depth
            packed=False,
        )

        # renders shape: (1, H, W, 4) — RGB + depth
        rendered_rgb = renders[0, :, :, :3]    # (H, W, 3)
        rendered_depth = renders[0, :, :, 3]   # (H, W)

        # ── RGB Loss ──
        rgb_loss = F.l1_loss(rendered_rgb, gt_image)

        if HAS_SSIM:
            # SSIM expects (B, C, H, W)
            ssim_val = fused_ssim(
                rendered_rgb.permute(2, 0, 1).unsqueeze(0),
                gt_image.permute(2, 0, 1).unsqueeze(0),
            )
            rgb_loss = 0.8 * (1 - ssim_val) + 0.2 * rgb_loss

        # ── Depth Loss ──
        depth_loss = torch.tensor(0.0, device=device)
        normal_loss = torch.tensor(0.0, device=device)

        if has_depth and step >= cfg.depth_start_step and "depth" in item:
            gt_depth = item["depth"]
            confidence = item["confidence"]

            depth_loss = compute_depth_loss(
                rendered_depth=rendered_depth,
                gt_depth=gt_depth,
                confidence=confidence,
                rgb_image=gt_image if cfg.adaptive_depth else None,
                min_confidence=cfg.min_confidence,
                adaptive=cfg.adaptive_depth,
            )

            if cfg.normal_lambda > 0:
                valid = (confidence >= cfg.min_confidence) & (gt_depth > 0.01)
                normal_loss = compute_normal_loss(
                    rendered_depth=rendered_depth,
                    gt_depth=gt_depth,
                    valid_mask=valid,
                )

        # ── Total Loss ──
        total_loss = rgb_loss + cfg.depth_lambda * depth_loss + cfg.normal_lambda * normal_loss

        # Backward
        total_loss.backward()

        # Accumulate gradients for densification
        if step < cfg.densify_stop and info is not None:
            # Get 2D gradient magnitude of means
            if model.means.grad is not None:
                visible = info.get("radii", None)
                if visible is not None:
                    visible_mask = (visible.squeeze(0) > 0)
                    grads = model.means.grad.detach().norm(dim=-1)
                    grad_accum[:len(grads)] += grads * visible_mask.float()[:len(grads)]
                    grad_count[:len(grads)] += visible_mask.int()[:len(grads)]

        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        # ── Logging ──
        if step % cfg.log_every == 0:
            elapsed = time.time() - start_time
            it_s = (step + 1) / elapsed if elapsed > 0 else 0
            log = (
                f"step={step:6d} | "
                f"loss={total_loss.item():.4f} "
                f"rgb={rgb_loss.item():.4f} "
            )
            if has_depth and step >= cfg.depth_start_step:
                log += f"depth={depth_loss.item():.4f} normal={normal_loss.item():.4f} "
            log += f"| GS={model.num_gaussians:,} | {it_s:.1f} it/s"
            print(log)
            losses_log.append({
                "step": step,
                "total": total_loss.item(),
                "rgb": rgb_loss.item(),
                "depth": depth_loss.item(),
                "normal": normal_loss.item(),
                "num_gs": model.num_gaussians,
            })

        # ── Save checkpoints ──
        if step > 0 and step % cfg.save_every == 0:
            ply_path = os.path.join(cfg.result_dir, f"point_cloud_{step}.ply")
            save_ply(ply_path, model)
            print(f"  Saved: {ply_path}")

        # ── Reset opacity ──
        if step > 0 and step % cfg.reset_opacity_every == 0 and step < cfg.densify_stop:
            with torch.no_grad():
                model.opacities.data = torch.logit(
                    torch.full_like(model.opacities, cfg.reset_opacity_value)
                )
            print(f"  Reset opacity at step {step}")

    # Final save
    elapsed = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"Training complete!")
    print(f"  Steps: {cfg.max_steps}")
    print(f"  Time: {elapsed/60:.1f} minutes")
    print(f"  Final Gaussians: {model.num_gaussians:,}")
    print(f"  Final loss: {total_loss.item():.4f}")
    print(f"{'='*60}")

    final_path = os.path.join(cfg.result_dir, "point_cloud_final.ply")
    save_ply(final_path, model)
    print(f"Saved: {final_path}")

    # Save training log
    log_path = os.path.join(cfg.result_dir, "training_log.json")
    with open(log_path, 'w') as f:
        json.dump(losses_log, f, indent=2)
    print(f"Saved: {log_path}")


# ── Entry point ──────────────────────────────────────────────

if __name__ == "__main__":
    cfg = tyro.cli(TrainConfig)
    train(cfg)
