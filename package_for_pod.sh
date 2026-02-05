#!/bin/bash
# Package everything needed for RunPod deployment.
#
# Creates two archives:
#   1. lidar-gsplat.tar.gz   — Code (small, ~50KB)
#   2. scan_data.tar.gz      — Raw scan data with depth/confidence (larger)
#
# The COLMAP data should already be on the pod from the previous training run.
# If not, also upload colmap_data.tar.gz and colmap_images.tar.gz.

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$HOME/projects"
SCAN_DIR="$PROJECT_DIR/arkit-to-colmap/scan_data/2026_01_13_14_47_59"

echo "=== Packaging code ==="
cd "$SCRIPT_DIR"
tar -czf /tmp/lidar-gsplat.tar.gz \
    --exclude='.venv' \
    --exclude='__pycache__' \
    --exclude='*.ply' \
    --exclude='*.npy' \
    --exclude='depth_data' \
    -C "$PROJECT_DIR" lidar-gsplat/

ls -lh /tmp/lidar-gsplat.tar.gz

echo ""
echo "=== Packaging scan data (depth + confidence + metadata) ==="
# Only include depth, confidence, and JSON files (not the full images/mesh)
tar -czf /tmp/scan_data.tar.gz \
    -C "$PROJECT_DIR/arkit-to-colmap" \
    --include='scan_data/2026_01_13_14_47_59/depth_*.exr' \
    --include='scan_data/2026_01_13_14_47_59/conf_*.png' \
    --include='scan_data/2026_01_13_14_47_59/frame_*.json' \
    --include='scan_data/2026_01_13_14_47_59/frame_*.jpg' \
    scan_data/2026_01_13_14_47_59/ 2>/dev/null || \
tar -czf /tmp/scan_data.tar.gz \
    -C "$PROJECT_DIR/arkit-to-colmap/scan_data" \
    2026_01_13_14_47_59/

ls -lh /tmp/scan_data.tar.gz

echo ""
echo "=== Done! Upload to RunPod: ==="
echo "  scp -P <port> /tmp/lidar-gsplat.tar.gz /tmp/scan_data.tar.gz root@<ip>:/workspace/"
echo ""
echo "Then on the pod:"
echo "  cd /workspace"
echo "  tar xzf lidar-gsplat.tar.gz"
echo "  mkdir -p scan_data && tar xzf scan_data.tar.gz -C scan_data/ || tar xzf scan_data.tar.gz"
echo "  bash lidar-gsplat/run_on_pod.sh"
