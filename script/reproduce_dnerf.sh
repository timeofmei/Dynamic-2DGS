#!/usr/bin/env bash

set -euo pipefail

usage() {
    echo "Usage: $0 SCENE" >&2
    echo "Scenes: lego bouncingballs jumpingjacks hook mutant standup trex hellwarrior" >&2
}

if [[ $# -ne 1 ]]; then
    usage
    exit 2
fi

SCENE="$1"
case "$SCENE" in
    lego|bouncingballs|jumpingjacks|hook|mutant|standup|trex|hellwarrior) ;;
    *)
        echo "Unknown D-NeRF scene: $SCENE" >&2
        usage
        exit 2
        ;;
esac

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

PYTHON="${PYTHON:-python}"
GPU_ID="${GPU_ID:-0}"
ITERATIONS="${ITERATIONS:-80000}"
DATA_ROOT="${DNERF_DATA_ROOT:-$HOME/dataset/data}"
OUTPUT_ROOT="${DNERF_OUTPUT_ROOT:-$REPO_ROOT/outputs/paper_dnerf}"

SOURCE_PATH="$DATA_ROOT/$SCENE"
MODEL_BASE="$OUTPUT_ROOT/$SCENE"
MODEL_PATH="${MODEL_BASE}_node"

test -f "$SOURCE_PATH/transforms_train.json"
test -f "$SOURCE_PATH/transforms_test.json"
mkdir -p "$OUTPUT_ROOT"

echo "[D-NeRF] Training scene: $SCENE"
CUDA_VISIBLE_DEVICES="$GPU_ID" "$PYTHON" train_gui.py \
    --source_path "$SOURCE_PATH" \
    --model_path "$MODEL_BASE" \
    --deform_type node --hyper_dim 8 --node_num 1024 \
    --is_blender --eval --gt_alpha_mask_as_scene_mask --local_frame \
    --resolution 1 --W 800 --H 800 \
    --test_iterations "$ITERATIONS" --save_iterations "$ITERATIONS" \
    --iterations "$ITERATIONS"

test -f "$MODEL_PATH/cfg_args"

echo "[D-NeRF] Rendering and extracting meshes: $SCENE"
CUDA_VISIBLE_DEVICES="$GPU_ID" "$PYTHON" render_mesh.py \
    --source_path "$SOURCE_PATH" \
    --model_path "$MODEL_PATH" \
    --deform_type node --hyper_dim 8 --node_num 1024 \
    --is_blender --eval --local_frame --resolution 1

GT_RENDER_PATH="$MODEL_PATH/test/ours_${ITERATIONS}/gt_w"
MESH_IMAGE_PATH="$MODEL_PATH/mesh_image"
test -d "$GT_RENDER_PATH"
test -d "$MESH_IMAGE_PATH"
test "$(find "$GT_RENDER_PATH" -maxdepth 1 -type f -name '*.png' | wc -l)" -eq 20
test "$(find "$MESH_IMAGE_PATH" -maxdepth 1 -type f -name '*.png' | wc -l)" -eq 20

echo "[D-NeRF] Evaluating Gaussian rendering: $SCENE"
CUDA_VISIBLE_DEVICES="$GPU_ID" "$PYTHON" metrics.py -m "$MODEL_PATH"

echo "[D-NeRF] Evaluating mesh rendering: $SCENE"
CUDA_VISIBLE_DEVICES="$GPU_ID" "$PYTHON" metrics_mesh.py --model_path "$MODEL_PATH"

echo "[D-NeRF] Complete: $SCENE"
echo "  Model: $MODEL_PATH"
echo "  Mesh metrics: $MODEL_PATH/mesh_render_results.json"
