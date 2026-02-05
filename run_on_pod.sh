#!/bin/bash
# Deploy and run LiDAR-enhanced Gaussian splatting on RunPod
#
# Usage (from local machine):
#   # 1. Create pod, SSH in
#   # 2. Upload data:
#   scp -P <port> prepare_and_train.tar.gz root@<ip>:/workspace/
#   # 3. SSH in and run:
#   cd /workspace && tar xzf prepare_and_train.tar.gz && bash run_on_pod.sh
#
# Or run each step manually — see below.

set -e

WORK=/workspace
DATA_DIR=$WORK/output_colmap_sfm
SCAN_DIR=$WORK/scan_data
DEPTH_DIR=$WORK/depth_data
RESULTS_BASELINE=$WORK/results_baseline
RESULTS_DEPTH=$WORK/results_depth
RESULTS_DENSE=$WORK/results_dense

echo "╔══════════════════════════════════════════════════════════╗"
echo "║   LiDAR-Enhanced Gaussian Splatting — RunPod Training   ║"
echo "╚══════════════════════════════════════════════════════════╝"
echo ""

# ── Step 1: Install dependencies ─────────────────────────────
echo "=== Installing dependencies ==="
pip install gsplat==1.5.3 imageio tyro fused-ssim opencv-python scipy tqdm numpy 2>&1 | tail -3

# Enable OpenEXR
export OPENCV_IO_ENABLE_OPENEXR=1

# ── Step 2: Prepare depth data ───────────────────────────────
if [ ! -d "$DEPTH_DIR" ]; then
    echo ""
    echo "=== Preparing depth maps ==="
    python3 $WORK/lidar-gsplat/prepare_depth_data.py \
        $SCAN_DIR \
        $DATA_DIR \
        --data-factor 2 \
        -o $DEPTH_DIR
fi

# ── Step 3: Create downscaled images ─────────────────────────
if [ ! -d "$DATA_DIR/images_2" ]; then
    echo ""
    echo "=== Creating downscaled images (data_factor=2) ==="
    mkdir -p $DATA_DIR/images_2
    python3 -c "
import cv2, os, glob
src = '$DATA_DIR/images'
dst = '$DATA_DIR/images_2'
for f in sorted(glob.glob(f'{src}/*.jpg')):
    img = cv2.imread(f)
    h, w = img.shape[:2]
    small = cv2.resize(img, (w//2, h//2), interpolation=cv2.INTER_AREA)
    cv2.imwrite(f'{dst}/{os.path.basename(f)}', small)
    print(f'  {os.path.basename(f)}: {w}x{h} → {w//2}x{h//2}')
"
fi

# ── Step 4: Generate dense point cloud ───────────────────────
DENSE_PLY=$WORK/dense_pointcloud.ply
if [ ! -f "$DENSE_PLY" ]; then
    echo ""
    echo "=== Generating dense LiDAR point cloud ==="
    python3 $WORK/lidar-gsplat/lidar_to_pointcloud.py \
        $SCAN_DIR \
        --min-confidence 1 \
        --voxel-size 0.003 \
        -o $DENSE_PLY
fi

echo ""
echo "=== Data prepared. Starting training runs ==="
echo ""

# ── Step 5a: BASELINE — Standard training (no depth) ─────────
echo "╔══════════════════════════════════════════════════════════╗"
echo "║  Run 1/3: BASELINE (RGB only, COLMAP points init)      ║"
echo "╚══════════════════════════════════════════════════════════╝"
python3 $WORK/lidar-gsplat/gsplat_depth_trainer.py \
    --data_dir $DATA_DIR \
    --data_factor 2 \
    --result_dir $RESULTS_BASELINE \
    --max_steps 30000 \
    --depth_lambda 0.0 \
    --normal_lambda 0.0 \
    2>&1 | tee $RESULTS_BASELINE/train.log

# ── Step 5b: DEPTH — Depth-supervised training ────────────────
echo ""
echo "╔══════════════════════════════════════════════════════════╗"
echo "║  Run 2/3: DEPTH SUPERVISED (LiDAR depth + normals)     ║"
echo "╚══════════════════════════════════════════════════════════╝"
python3 $WORK/lidar-gsplat/gsplat_depth_trainer.py \
    --data_dir $DATA_DIR \
    --depth_dir $DEPTH_DIR \
    --data_factor 2 \
    --result_dir $RESULTS_DEPTH \
    --max_steps 30000 \
    --depth_lambda 0.5 \
    --normal_lambda 0.05 \
    2>&1 | tee $RESULTS_DEPTH/train.log

# ── Step 5c: DENSE — Dense init + depth supervised ───────────
echo ""
echo "╔══════════════════════════════════════════════════════════╗"
echo "║  Run 3/3: DENSE INIT + DEPTH (full LiDAR pipeline)     ║"
echo "╚══════════════════════════════════════════════════════════╝"
python3 $WORK/lidar-gsplat/gsplat_depth_trainer.py \
    --data_dir $DATA_DIR \
    --depth_dir $DEPTH_DIR \
    --dense_init_ply $DENSE_PLY \
    --data_factor 2 \
    --result_dir $RESULTS_DENSE \
    --max_steps 30000 \
    --depth_lambda 0.5 \
    --normal_lambda 0.05 \
    --max_init_points 500000 \
    2>&1 | tee $RESULTS_DENSE/train.log

echo ""
echo "╔══════════════════════════════════════════════════════════╗"
echo "║                  ALL TRAINING COMPLETE                   ║"
echo "╠══════════════════════════════════════════════════════════╣"
echo "║  Results:                                                ║"
echo "║    Baseline: $RESULTS_BASELINE/point_cloud_final.ply    ║"
echo "║    Depth:    $RESULTS_DEPTH/point_cloud_final.ply       ║"
echo "║    Dense:    $RESULTS_DENSE/point_cloud_final.ply       ║"
echo "║                                                          ║"
echo "║  Convert to .splat and compare in viewer!                ║"
echo "╚══════════════════════════════════════════════════════════╝"
