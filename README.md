# LiDAR-Enhanced Gaussian Splatting

Improved 3D Gaussian Splatting using iPhone LiDAR depth and confidence data as supervision signals during training.

## What This Does

Standard Gaussian splatting only uses RGB images and sparse COLMAP points. This project adds **LiDAR depth supervision** to the training loop, using the depth maps and confidence data that the 3D Scanner App captures alongside each image.

### Improvements Over Standard Pipeline

1. **Depth supervision loss** — Penalizes Gaussians whose rendered depth deviates from LiDAR ground truth
2. **Dense point cloud initialization** — Initialize Gaussians from LiDAR depth back-projection instead of sparse COLMAP points (55k → 500k+ points)
3. **Confidence-weighted loss** — Weight depth loss by LiDAR confidence (0=low, 1=medium, 2=high)
4. **Normal regularization** — Derive surface normals from depth maps to constrain Gaussian orientation

## Data Format

From 3D Scanner App export:

| File | Resolution | Format | Description |
|------|-----------|--------|-------------|
| `frame_XXXXX.jpg` | 1920×1440 | JPEG | RGB image (45 keyframes) |
| `frame_XXXXX.json` | — | JSON | Camera pose + intrinsics |
| `depth_XXXXX.exr` | 192×256 | EXR float32 | LiDAR depth in meters |
| `conf_XXXXX.png` | 192×256 | PNG uint8 | Depth confidence (0/1/2) |

Depth/confidence are 5.625× smaller than images and exist for all 269 frames (not just the 45 with JPGs).

## Pipeline

```
1. prepare_colmap_with_depth.py  — COLMAP format + depth maps for training images
2. lidar_to_pointcloud.py        — Dense initialization from LiDAR back-projection
3. train_with_depth.py           — gsplat trainer with depth supervision (runs on CUDA GPU)
```

## References

- [DN-Splatter](https://arxiv.org/abs/2403.17822) — Depth and Normal Priors for Gaussian Splatting (WACV 2025)
- [SplaTAM](https://arxiv.org/abs/2312.02126) — Gaussian SLAM with RGB-D (CVPR 2024)
- [gsplat](https://github.com/nerfstudio-project/gsplat) — CUDA-accelerated Gaussian splatting library
- [2DGS](https://surfsplatting.github.io/) — 2D Gaussian Splatting for better geometry (SIGGRAPH 2024)
