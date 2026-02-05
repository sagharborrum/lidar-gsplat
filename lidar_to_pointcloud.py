#!/usr/bin/env python3
"""
Back-project LiDAR depth maps into a dense 3D point cloud.

Uses camera intrinsics + poses from ARKit JSON metadata, plus LiDAR depth
and confidence maps, to create a dense colored point cloud suitable for
initializing Gaussian splatting.

Output: COLMAP-compatible points3D.bin or PLY point cloud.
"""

import argparse
import json
import os
from pathlib import Path

os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'
import cv2
import numpy as np
from scipy.spatial.transform import Rotation
from tqdm import tqdm


def load_depth(exr_path: str) -> np.ndarray:
    """Load depth map from EXR file. Returns float32 array in meters."""
    depth = cv2.imread(exr_path, cv2.IMREAD_UNCHANGED)
    if depth is None:
        raise FileNotFoundError(f"Cannot read depth: {exr_path}")
    # EXR might have multiple channels — take first if needed
    if len(depth.shape) == 3:
        depth = depth[:, :, 0]
    return depth.astype(np.float32)


def load_confidence(conf_path: str) -> np.ndarray:
    """Load confidence map. Returns uint8 array (0=low, 1=medium, 2=high)."""
    conf = cv2.imread(conf_path, cv2.IMREAD_UNCHANGED)
    if conf is None:
        raise FileNotFoundError(f"Cannot read confidence: {conf_path}")
    if len(conf.shape) == 3:
        conf = conf[:, :, 0]
    return conf.astype(np.uint8)


def parse_arkit_json(json_path: str) -> dict:
    """Parse ARKit camera metadata from 3D Scanner App JSON."""
    with open(json_path) as f:
        data = json.load(f)

    # Camera-to-world pose (row-major flat array)
    pose_flat = data['cameraPoseARFrame']
    c2w = np.array(pose_flat).reshape(4, 4, order='C')

    # 3x3 intrinsics (row-major flat array)
    K_flat = data['intrinsics']
    K = np.array(K_flat).reshape(3, 3)

    return {
        'c2w': c2w,
        'fx': K[0, 0],
        'fy': K[1, 1],
        'cx': K[0, 2],
        'cy': K[1, 2],
        'motion_quality': data.get('motionQuality', 0),
    }


def backproject_depth(
    depth: np.ndarray,
    confidence: np.ndarray,
    c2w: np.ndarray,
    fx: float, fy: float, cx: float, cy: float,
    image: np.ndarray = None,
    min_confidence: int = 1,
    depth_scale_x: float = 1.0,
    depth_scale_y: float = 1.0,
) -> tuple:
    """
    Back-project depth map into 3D world coordinates.

    Args:
        depth: (H, W) depth in meters
        confidence: (H, W) confidence 0/1/2
        c2w: 4x4 camera-to-world matrix (ARKit)
        fx, fy, cx, cy: camera intrinsics (at IMAGE resolution)
        image: (imgH, imgW, 3) RGB image for coloring points
        min_confidence: minimum confidence to include (0, 1, or 2)
        depth_scale_x: ratio of image width to depth width
        depth_scale_y: ratio of image height to depth height

    Returns:
        points: (N, 3) world coordinates
        colors: (N, 3) RGB colors (0-255) or None
    """
    h, w = depth.shape

    # Scale intrinsics to depth resolution
    fx_d = fx / depth_scale_x
    fy_d = fy / depth_scale_y
    cx_d = cx / depth_scale_x
    cy_d = cy / depth_scale_y

    # Create pixel grid
    u, v = np.meshgrid(np.arange(w), np.arange(h))

    # Filter by confidence and valid depth
    mask = (confidence >= min_confidence) & (depth > 0.01) & (depth < 100.0)
    u_valid = u[mask]
    v_valid = v[mask]
    d_valid = depth[mask]

    # Back-project to camera coordinates
    # ARKit camera: x-right, y-up, z-toward-viewer (negative z forward)
    x_cam = (u_valid - cx_d) * d_valid / fx_d
    y_cam = (v_valid - cy_d) * d_valid / fy_d
    z_cam = -d_valid  # ARKit: depth is along -Z

    # Stack into (N, 3) camera-space points
    points_cam = np.stack([x_cam, y_cam, z_cam], axis=-1)

    # Transform to world coordinates
    R = c2w[:3, :3]
    t = c2w[:3, 3]
    points_world = (R @ points_cam.T).T + t

    # Get colors from image (if provided)
    colors = None
    if image is not None:
        # Map depth pixel coordinates to image pixel coordinates
        u_img = (u_valid * depth_scale_x).astype(int)
        v_img = (v_valid * depth_scale_y).astype(int)
        u_img = np.clip(u_img, 0, image.shape[1] - 1)
        v_img = np.clip(v_img, 0, image.shape[0] - 1)
        colors = image[v_img, u_img, :]  # BGR from cv2

    return points_world, colors


def subsample_points(points: np.ndarray, colors: np.ndarray,
                     voxel_size: float = 0.002) -> tuple:
    """
    Voxel-based subsampling to reduce point cloud density.

    Args:
        points: (N, 3) point coordinates
        colors: (N, 3) point colors
        voxel_size: voxel edge length in meters (0.002 = 2mm)

    Returns:
        subsampled points and colors
    """
    # Quantize to voxel grid
    voxel_coords = np.floor(points / voxel_size).astype(np.int64)
    voxel_coords = np.ascontiguousarray(voxel_coords)

    # Hash-based unique detection
    # Shift to positive coordinates and create a unique key per voxel
    mins = voxel_coords.min(axis=0)
    shifted = voxel_coords - mins
    # Create unique integer key (works for reasonably sized scenes)
    max_vals = shifted.max(axis=0) + 1
    keys = shifted[:, 0] * max_vals[1] * max_vals[2] + shifted[:, 1] * max_vals[2] + shifted[:, 2]

    _, unique_idx = np.unique(keys, return_index=True)

    return points[unique_idx], colors[unique_idx] if colors is not None else None


def write_ply(path: str, points: np.ndarray, colors: np.ndarray = None):
    """Write point cloud to PLY format."""
    n = len(points)
    has_color = colors is not None

    with open(path, 'w') as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {n}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        if has_color:
            f.write("property uchar red\n")
            f.write("property uchar green\n")
            f.write("property uchar blue\n")
        f.write("end_header\n")

        for i in range(n):
            line = f"{points[i, 0]:.6f} {points[i, 1]:.6f} {points[i, 2]:.6f}"
            if has_color:
                r, g, b = int(colors[i, 2]), int(colors[i, 1]), int(colors[i, 0])  # BGR→RGB
                line += f" {r} {g} {b}"
            f.write(line + "\n")


def write_colmap_points3d_bin(path: str, points: np.ndarray, colors: np.ndarray):
    """Write points to COLMAP binary points3D.bin format."""
    import struct

    with open(path, 'wb') as f:
        # Number of points
        f.write(struct.pack('<Q', len(points)))

        for i in range(len(points)):
            point_id = i + 1
            x, y, z = points[i]
            r = int(colors[i, 2]) if colors is not None else 128  # BGR→RGB
            g = int(colors[i, 1]) if colors is not None else 128
            b = int(colors[i, 0]) if colors is not None else 128
            error = 1.0  # Placeholder reprojection error

            # POINT3D_ID, X, Y, Z, R, G, B, ERROR
            f.write(struct.pack('<Q', point_id))
            f.write(struct.pack('<ddd', x, y, z))
            f.write(struct.pack('<BBB', r, g, b))
            f.write(struct.pack('<d', error))

            # TRACK[] — number of track elements, then (IMAGE_ID, POINT2D_IDX) pairs
            # We write 0 track elements for dense points
            f.write(struct.pack('<Q', 0))


def main():
    parser = argparse.ArgumentParser(
        description='Back-project LiDAR depth into dense 3D point cloud'
    )
    parser.add_argument(
        'scan_dir',
        type=Path,
        help='Path to 3D Scanner App export folder'
    )
    parser.add_argument(
        '-o', '--output',
        type=Path,
        default=None,
        help='Output file path (.ply or directory for COLMAP format)'
    )
    parser.add_argument(
        '--min-confidence',
        type=int,
        default=1,
        choices=[0, 1, 2],
        help='Minimum confidence level (0=all, 1=medium+high, 2=high only). Default: 1'
    )
    parser.add_argument(
        '--voxel-size',
        type=float,
        default=0.002,
        help='Voxel size for subsampling in meters. Default: 0.002 (2mm)'
    )
    parser.add_argument(
        '--quality-threshold',
        type=float,
        default=0.5,
        help='Minimum ARKit motionQuality. Default: 0.5'
    )
    parser.add_argument(
        '--use-all-frames',
        action='store_true',
        help='Use all frames (not just those with JPGs). Colors will be interpolated.'
    )
    parser.add_argument(
        '--format',
        choices=['ply', 'colmap'],
        default='ply',
        help='Output format. Default: ply'
    )

    args = parser.parse_args()
    scan_dir = args.scan_dir

    if args.output is None:
        args.output = Path(f"dense_pointcloud.ply")

    # Find all frame JSON files
    json_files = sorted(scan_dir.glob('frame_*.json'))
    if not json_files:
        raise FileNotFoundError(f"No frame_*.json files in {scan_dir}")

    print(f"Found {len(json_files)} frame metadata files")

    # Determine which frames to use
    all_points = []
    all_colors = []

    for json_path in tqdm(json_files, desc="Processing frames"):
        frame_num = int(json_path.stem.replace('frame_', ''))

        # Check for depth map
        depth_path = scan_dir / f"depth_{frame_num:05d}.exr"
        conf_path = scan_dir / f"conf_{frame_num:05d}.png"

        if not depth_path.exists() or not conf_path.exists():
            continue

        # Check for image (optional — for coloring)
        jpg_path = scan_dir / f"frame_{frame_num:05d}.jpg"
        has_image = jpg_path.exists()

        if not args.use_all_frames and not has_image:
            continue

        # Parse metadata
        try:
            meta = parse_arkit_json(str(json_path))
        except (json.JSONDecodeError, KeyError) as e:
            print(f"  Warning: skipping {json_path}: {e}")
            continue

        if meta['motion_quality'] < args.quality_threshold:
            continue

        # Load depth and confidence
        depth = load_depth(str(depth_path))
        confidence = load_confidence(str(conf_path))

        # Load image if available
        image = cv2.imread(str(jpg_path)) if has_image else None

        # Compute scale factors (image resolution / depth resolution)
        if image is not None:
            depth_scale_x = image.shape[1] / depth.shape[1]
            depth_scale_y = image.shape[0] / depth.shape[0]
        else:
            # Use intrinsics to infer image size
            # Standard 3D Scanner App: 1920x1440 images, 192x256 depth
            depth_scale_x = 1440 / depth.shape[1]  # width
            depth_scale_y = 1920 / depth.shape[0]  # height

        # Back-project
        points, colors = backproject_depth(
            depth=depth,
            confidence=confidence,
            c2w=meta['c2w'],
            fx=meta['fx'], fy=meta['fy'],
            cx=meta['cx'], cy=meta['cy'],
            image=image,
            min_confidence=args.min_confidence,
            depth_scale_x=depth_scale_x,
            depth_scale_y=depth_scale_y,
        )

        if len(points) > 0:
            all_points.append(points)
            if colors is not None:
                all_colors.append(colors)

    if not all_points:
        raise RuntimeError("No points generated — check data paths and confidence threshold")

    # Concatenate all frames
    points = np.concatenate(all_points)
    colors = np.concatenate(all_colors) if all_colors else None

    print(f"\nRaw points: {len(points):,}")

    # Voxel subsampling
    if args.voxel_size > 0:
        points, colors = subsample_points(points, colors, args.voxel_size)
        print(f"After voxel subsampling ({args.voxel_size*1000:.0f}mm): {len(points):,}")

    # Write output
    if args.format == 'ply':
        output_path = args.output if str(args.output).endswith('.ply') else args.output / 'dense.ply'
        os.makedirs(output_path.parent, exist_ok=True)
        write_ply(str(output_path), points, colors)
        print(f"\n✅ Saved PLY: {output_path} ({os.path.getsize(output_path) / 1e6:.1f} MB)")
    elif args.format == 'colmap':
        output_dir = args.output
        os.makedirs(output_dir / 'sparse' / '0', exist_ok=True)
        colmap_path = output_dir / 'sparse' / '0' / 'points3D.bin'
        write_colmap_points3d_bin(str(colmap_path), points, colors)
        print(f"\n✅ Saved COLMAP points3D.bin: {colmap_path}")

    print(f"   Total points: {len(points):,}")


if __name__ == '__main__':
    main()
