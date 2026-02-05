#!/usr/bin/env python3
"""
Prepare depth and confidence maps for gsplat training.

Takes 3D Scanner App export data and creates a training-ready directory with:
- Depth maps resized to match training images (or downscaled versions)
- Confidence maps resized to match
- Metadata JSON mapping image filenames to their depth/confidence files

Output structure:
    output_dir/
        depths/          # Float32 .npy depth maps (matching training image resolution)
        confidences/     # Uint8 .npy confidence maps
        depth_meta.json  # Mapping of image filename → depth/conf filenames
"""

import argparse
import json
import os
from pathlib import Path

os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'
import cv2
import numpy as np
from tqdm import tqdm


def prepare_depth_maps(
    scan_dir: Path,
    colmap_dir: Path,
    output_dir: Path,
    data_factor: int = 1,
):
    """
    Prepare depth maps aligned to COLMAP training images.

    Args:
        scan_dir: 3D Scanner App export directory
        colmap_dir: COLMAP project directory (with images/)
        output_dir: Where to save prepared depth data
        data_factor: Downscale factor matching gsplat training (1=full, 2=half, etc.)
    """
    # Find training images
    images_dir = colmap_dir / 'images'
    if data_factor > 1:
        images_dir_scaled = colmap_dir / f'images_{data_factor}'
        if images_dir_scaled.exists():
            images_dir = images_dir_scaled

    image_files = sorted(images_dir.glob('frame_*.jpg'))
    if not image_files:
        raise FileNotFoundError(f"No images found in {images_dir}")

    # Get target resolution from first image
    sample_img = cv2.imread(str(image_files[0]))
    target_h, target_w = sample_img.shape[:2]
    print(f"Target resolution: {target_w}×{target_h} (data_factor={data_factor})")

    # Create output directories
    depths_dir = output_dir / 'depths'
    confs_dir = output_dir / 'confidences'
    depths_dir.mkdir(parents=True, exist_ok=True)
    confs_dir.mkdir(parents=True, exist_ok=True)

    meta = {}
    processed = 0

    for img_path in tqdm(image_files, desc="Preparing depth maps"):
        frame_name = img_path.stem  # e.g., "frame_00000"
        frame_num = int(frame_name.replace('frame_', ''))

        # Find matching depth and confidence
        depth_path = scan_dir / f"depth_{frame_num:05d}.exr"
        conf_path = scan_dir / f"conf_{frame_num:05d}.png"

        if not depth_path.exists():
            print(f"  Warning: no depth for {frame_name}")
            continue

        # Load depth (float32, meters)
        depth = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
        if len(depth.shape) == 3:
            depth = depth[:, :, 0]
        depth = depth.astype(np.float32)

        # Load confidence
        if conf_path.exists():
            conf = cv2.imread(str(conf_path), cv2.IMREAD_UNCHANGED)
            if len(conf.shape) == 3:
                conf = conf[:, :, 0]
            conf = conf.astype(np.uint8)
        else:
            # Default to full confidence
            conf = np.ones_like(depth, dtype=np.uint8) * 2

        # Resize depth to target resolution
        # Use INTER_NEAREST for depth to avoid interpolation artifacts at edges
        depth_resized = cv2.resize(depth, (target_w, target_h),
                                   interpolation=cv2.INTER_NEAREST)

        # For confidence, also use nearest neighbor
        conf_resized = cv2.resize(conf, (target_w, target_h),
                                  interpolation=cv2.INTER_NEAREST)

        # Save as numpy arrays (fast to load during training)
        depth_filename = f"{frame_name}_depth.npy"
        conf_filename = f"{frame_name}_conf.npy"

        np.save(str(depths_dir / depth_filename), depth_resized)
        np.save(str(confs_dir / conf_filename), conf_resized)

        meta[img_path.name] = {
            'depth_file': depth_filename,
            'conf_file': conf_filename,
            'original_depth_shape': list(depth.shape),
            'resized_shape': [target_h, target_w],
            'depth_min': float(depth[depth > 0].min()) if (depth > 0).any() else 0,
            'depth_max': float(depth.max()),
            'depth_mean': float(depth[depth > 0].mean()) if (depth > 0).any() else 0,
            'high_conf_pct': float((conf == 2).sum() / conf.size * 100),
        }
        processed += 1

    # Save metadata
    meta_path = output_dir / 'depth_meta.json'
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2)

    print(f"\n✅ Prepared {processed} depth maps")
    print(f"   Depths: {depths_dir}")
    print(f"   Confidences: {confs_dir}")
    print(f"   Metadata: {meta_path}")
    print(f"   Resolution: {target_w}×{target_h}")

    # Print depth statistics
    if meta:
        depths_min = min(v['depth_min'] for v in meta.values())
        depths_max = max(v['depth_max'] for v in meta.values())
        avg_conf = np.mean([v['high_conf_pct'] for v in meta.values()])
        print(f"   Depth range: {depths_min:.3f} - {depths_max:.3f} meters")
        print(f"   Avg high-confidence: {avg_conf:.1f}%")


def main():
    parser = argparse.ArgumentParser(
        description='Prepare LiDAR depth maps for gsplat depth-supervised training'
    )
    parser.add_argument(
        'scan_dir',
        type=Path,
        help='Path to 3D Scanner App export folder'
    )
    parser.add_argument(
        'colmap_dir',
        type=Path,
        help='Path to COLMAP project directory (with images/)'
    )
    parser.add_argument(
        '-o', '--output',
        type=Path,
        default=None,
        help='Output directory for prepared depth data. Default: <colmap_dir>/depth_data'
    )
    parser.add_argument(
        '--data-factor',
        type=int,
        default=2,
        help='Training image downscale factor (must match gsplat --data_factor). Default: 2'
    )

    args = parser.parse_args()

    if args.output is None:
        args.output = args.colmap_dir / 'depth_data'

    prepare_depth_maps(
        scan_dir=args.scan_dir,
        colmap_dir=args.colmap_dir,
        output_dir=args.output,
        data_factor=args.data_factor,
    )


if __name__ == '__main__':
    main()
