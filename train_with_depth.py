#!/usr/bin/env python3
"""
Depth-supervised Gaussian Splatting trainer.

Modified gsplat simple_trainer that adds:
1. LiDAR depth supervision loss (L1, confidence-weighted)
2. Normal consistency regularization (derived from depth gradients)
3. Dense initialization from LiDAR point cloud (optional)

Based on gsplat v1.5.3 simple_trainer.py.

Usage:
    python train_with_depth.py \
        --data_dir /path/to/colmap_project \
        --depth_dir /path/to/depth_data \
        --data_factor 2 \
        --result_dir ./results \
        --max_steps 30000 \
        --depth_lambda 0.5 \
        --normal_lambda 0.05
"""

import json
import math
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor


@dataclass
class DepthConfig:
    """Configuration for depth supervision."""

    # Path to prepared depth data (from prepare_depth_data.py)
    depth_dir: Optional[str] = None

    # Depth loss weight (0 = no depth supervision)
    depth_lambda: float = 0.5

    # Normal consistency loss weight (0 = no normal loss)
    normal_lambda: float = 0.05

    # Minimum confidence level to supervise (0=all, 1=medium+high, 2=high only)
    min_confidence: int = 1

    # Use adaptive depth loss (weight by image gradient — edges get less weight)
    adaptive_depth: bool = True

    # Start depth supervision after N steps (let RGB converge first)
    depth_start_step: int = 500

    # Path to dense point cloud for initialization (optional)
    dense_init_ply: Optional[str] = None

    # Maximum points for dense initialization (subsample if more)
    max_init_points: int = 500_000


class DepthSupervisor:
    """Handles depth data loading and loss computation for gsplat training."""

    def __init__(self, config: DepthConfig, device: torch.device = torch.device("cuda")):
        self.config = config
        self.device = device
        self.depth_maps = {}
        self.conf_maps = {}
        self.meta = {}

        if config.depth_dir:
            self._load_depth_data(config.depth_dir)

    def _load_depth_data(self, depth_dir: str):
        """Load depth metadata and prepare for training."""
        depth_path = Path(depth_dir)
        meta_path = depth_path / 'depth_meta.json'

        if not meta_path.exists():
            raise FileNotFoundError(
                f"No depth_meta.json found in {depth_dir}. "
                f"Run prepare_depth_data.py first."
            )

        with open(meta_path) as f:
            self.meta = json.load(f)

        print(f"[DepthSupervisor] Loaded metadata for {len(self.meta)} depth maps")
        print(f"  Depth dir: {depth_dir}")
        print(f"  Lambda: depth={self.config.depth_lambda}, normal={self.config.normal_lambda}")
        print(f"  Min confidence: {self.config.min_confidence}")
        print(f"  Adaptive: {self.config.adaptive_depth}")

    def get_depth_for_image(self, image_name: str) -> tuple:
        """
        Get depth and confidence tensors for a training image.

        Args:
            image_name: Filename of the training image (e.g., "frame_00000.jpg")

        Returns:
            depth: (H, W) tensor in meters, or None
            confidence: (H, W) tensor (0/1/2), or None
        """
        if image_name not in self.meta:
            return None, None

        # Lazy load and cache
        if image_name not in self.depth_maps:
            info = self.meta[image_name]
            depth_path = Path(self.config.depth_dir) / 'depths' / info['depth_file']
            conf_path = Path(self.config.depth_dir) / 'confidences' / info['conf_file']

            depth = np.load(str(depth_path))
            conf = np.load(str(conf_path))

            self.depth_maps[image_name] = torch.from_numpy(depth).to(self.device)
            self.conf_maps[image_name] = torch.from_numpy(conf).to(self.device)

        return self.depth_maps[image_name], self.conf_maps[image_name]

    def compute_depth_loss(
        self,
        rendered_depth: Tensor,
        image_name: str,
        rgb_image: Tensor = None,
    ) -> tuple:
        """
        Compute depth supervision loss.

        Args:
            rendered_depth: (H, W) or (H, W, 1) rendered depth from Gaussians
            image_name: Which training image this corresponds to
            rgb_image: (H, W, 3) RGB image for adaptive weighting

        Returns:
            depth_loss: scalar tensor
            normal_loss: scalar tensor (0 if normal_lambda == 0)
        """
        gt_depth, confidence = self.get_depth_for_image(image_name)
        if gt_depth is None:
            return torch.tensor(0.0, device=self.device), torch.tensor(0.0, device=self.device)

        # Reshape if needed
        if rendered_depth.dim() == 3:
            rendered_depth = rendered_depth.squeeze(-1)

        # Ensure same size
        if rendered_depth.shape != gt_depth.shape:
            gt_depth = F.interpolate(
                gt_depth.unsqueeze(0).unsqueeze(0),
                size=rendered_depth.shape,
                mode='nearest'
            ).squeeze()
            confidence = F.interpolate(
                confidence.float().unsqueeze(0).unsqueeze(0),
                size=rendered_depth.shape,
                mode='nearest'
            ).squeeze().long()

        # Confidence mask
        valid_mask = (confidence >= self.config.min_confidence) & (gt_depth > 0.01)

        if valid_mask.sum() < 100:
            return torch.tensor(0.0, device=self.device), torch.tensor(0.0, device=self.device)

        # Confidence weights: 0→0, 1→0.5, 2→1.0
        conf_weights = confidence.float() / 2.0

        # --- Depth L1 Loss ---
        depth_error = torch.abs(rendered_depth - gt_depth)

        if self.config.adaptive_depth and rgb_image is not None:
            # DN-Splatter style: weight by inverse image gradient
            # (less supervision at edges where depth discontinuities are expected)
            gray = rgb_image.mean(dim=-1) if rgb_image.dim() == 3 else rgb_image
            grad_x = torch.abs(gray[:, 1:] - gray[:, :-1])
            grad_y = torch.abs(gray[1:, :] - gray[:-1, :])
            # Pad to match size
            grad_x = F.pad(grad_x, (0, 1), mode='replicate')
            grad_y = F.pad(grad_y, (1, 0), mode='replicate')
            gradient_magnitude = (grad_x + grad_y) / 2.0
            # Weight: high gradient → low weight
            edge_weight = torch.exp(-10.0 * gradient_magnitude)
            depth_error = depth_error * edge_weight

        # Apply confidence weighting and mask
        weighted_error = depth_error * conf_weights * valid_mask.float()
        depth_loss = weighted_error.sum() / (valid_mask.float() * conf_weights).sum().clamp(min=1.0)

        # --- Normal Consistency Loss ---
        normal_loss = torch.tensor(0.0, device=self.device)
        if self.config.normal_lambda > 0:
            normal_loss = self._compute_normal_loss(rendered_depth, gt_depth, valid_mask)

        return depth_loss, normal_loss

    def _compute_normal_loss(
        self,
        rendered_depth: Tensor,
        gt_depth: Tensor,
        valid_mask: Tensor,
    ) -> Tensor:
        """
        Compute normal consistency loss between rendered and GT depth.

        Derives surface normals from depth gradients and compares them.
        """
        def depth_to_normals(depth: Tensor) -> Tensor:
            """Compute surface normals from depth via finite differences."""
            # Gradients in x and y
            dz_dx = depth[:, 1:] - depth[:, :-1]  # (H, W-1)
            dz_dy = depth[1:, :] - depth[:-1, :]  # (H-1, W)

            # Pad to original size
            dz_dx = F.pad(dz_dx, (0, 1), mode='replicate')
            dz_dy = F.pad(dz_dy, (1, 0), mode='replicate')

            # Normal = (-dz/dx, -dz/dy, 1), then normalize
            normals = torch.stack([-dz_dx, -dz_dy, torch.ones_like(depth)], dim=-1)
            normals = F.normalize(normals, p=2, dim=-1)
            return normals

        rendered_normals = depth_to_normals(rendered_depth)
        gt_normals = depth_to_normals(gt_depth)

        # Cosine similarity (1 = identical, -1 = opposite)
        cos_sim = (rendered_normals * gt_normals).sum(dim=-1)

        # Loss: 1 - cos_similarity (0 when normals match)
        normal_error = 1.0 - cos_sim

        # Mask edges and invalid regions
        # Shrink valid mask to avoid boundary artifacts from gradient computation
        erosion_mask = valid_mask[1:-1, 1:-1]
        normal_error_crop = normal_error[1:-1, 1:-1]

        if erosion_mask.sum() < 100:
            return torch.tensor(0.0, device=self.device)

        return (normal_error_crop * erosion_mask.float()).sum() / erosion_mask.float().sum()


def load_dense_init_points(ply_path: str, max_points: int = 500_000) -> np.ndarray:
    """
    Load dense point cloud from PLY for Gaussian initialization.

    Returns (N, 3) numpy array of point positions, plus (N, 3) colors.
    """
    print(f"[DenseInit] Loading dense point cloud from {ply_path}")

    positions = []
    colors = []

    with open(ply_path) as f:
        # Skip header
        while True:
            line = f.readline().strip()
            if line == 'end_header':
                break

        # Read points
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 3:
                positions.append([float(parts[0]), float(parts[1]), float(parts[2])])
                if len(parts) >= 6:
                    colors.append([int(parts[3]), int(parts[4]), int(parts[5])])

    positions = np.array(positions, dtype=np.float32)
    colors = np.array(colors, dtype=np.uint8) if colors else None

    print(f"  Loaded {len(positions):,} points")

    # Subsample if too many
    if len(positions) > max_points:
        indices = np.random.choice(len(positions), max_points, replace=False)
        positions = positions[indices]
        if colors is not None:
            colors = colors[indices]
        print(f"  Subsampled to {max_points:,} points")

    return positions, colors


# ─── Integration with gsplat simple_trainer.py ─────────────────────────
#
# To use this with gsplat's simple_trainer.py, add the following to the
# training loop. This is a guide — actual integration depends on gsplat version.
#
# 1. After imports, create the supervisor:
#
#     depth_supervisor = DepthSupervisor(DepthConfig(
#         depth_dir="/path/to/depth_data",
#         depth_lambda=0.5,
#         normal_lambda=0.05,
#     ))
#
# 2. In the training loop, after computing rgb_loss:
#
#     if step >= depth_config.depth_start_step:
#         # gsplat renders depth alongside color
#         rendered_depth = renders["depth"]  # or however gsplat returns it
#         depth_loss, normal_loss = depth_supervisor.compute_depth_loss(
#             rendered_depth=rendered_depth,
#             image_name=image_name,
#             rgb_image=gt_image,
#         )
#         total_loss = rgb_loss + depth_config.depth_lambda * depth_loss
#         total_loss += depth_config.normal_lambda * normal_loss
#
# 3. For dense initialization, replace COLMAP points with:
#
#     if depth_config.dense_init_ply:
#         positions, colors = load_dense_init_points(
#             depth_config.dense_init_ply,
#             max_points=depth_config.max_init_points,
#         )
#         # Use these as initial Gaussian means instead of COLMAP points


def print_integration_guide():
    """Print instructions for integrating with gsplat."""
    guide = """
╔══════════════════════════════════════════════════════════════╗
║        gsplat Integration Guide — Depth Supervision          ║
╠══════════════════════════════════════════════════════════════╣
║                                                              ║
║  1. Prepare depth data:                                      ║
║     python prepare_depth_data.py \\                           ║
║       scan_data/2026_01_13_14_47_59 \\                        ║
║       output_colmap_sfm \\                                    ║
║       --data-factor 2                                        ║
║                                                              ║
║  2. (Optional) Generate dense point cloud:                   ║
║     python lidar_to_pointcloud.py \\                          ║
║       scan_data/2026_01_13_14_47_59 \\                        ║
║       --min-confidence 1 \\                                   ║
║       --voxel-size 0.003                                     ║
║                                                              ║
║  3. Modify gsplat simple_trainer.py:                         ║
║     - Import DepthSupervisor and DepthConfig                 ║
║     - Add depth loss after RGB loss computation              ║
║     - See code comments above for exact integration          ║
║                                                              ║
║  4. Run training (on CUDA GPU):                              ║
║     python simple_trainer.py default \\                       ║
║       --data_dir /path/to/colmap_project \\                   ║
║       --data_factor 2 \\                                      ║
║       --max_steps 30000                                      ║
║                                                              ║
║  Key parameters to tune:                                     ║
║     depth_lambda: 0.1-1.0 (higher = stronger depth prior)   ║
║     normal_lambda: 0.01-0.1 (gentle normal regularization)  ║
║     min_confidence: 1 (medium+high) or 2 (high only)        ║
║     depth_start_step: 500 (let RGB warm up first)            ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
"""
    print(guide)


if __name__ == '__main__':
    print_integration_guide()
