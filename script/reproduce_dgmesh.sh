#!/usr/bin/env bash

set -euo pipefail

usage() {
    echo "Usage: $0 SCENE" >&2
    echo "Scenes: duck horse bird beagle torus2sphere girlwalk" >&2
    echo "Set FORCE=1 to rerun completed training/rendering stages." >&2
    echo "Activate the dgmesh-eval environment first, or set DGMESH_EVAL_PYTHON." >&2
}

if [[ $# -ne 1 ]]; then
    usage
    exit 2
fi

SCENE="$1"
case "$SCENE" in
    duck|horse|bird|beagle|torus2sphere|girlwalk) ;;
    *)
        echo "Unknown DG-Mesh scene: $SCENE" >&2
        usage
        exit 2
        ;;
esac

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

PYTHON="${PYTHON:-python}"
EVAL_PYTHON="${DGMESH_EVAL_PYTHON:-python}"
GPU_ID="${GPU_ID:-0}"
ITERATIONS="${ITERATIONS:-80000}"
FORCE="${FORCE:-0}"
DATA_ROOT="${DGMESH_DATA_ROOT:-$HOME/autodl-tmp/dataset/dg-mesh}"
OUTPUT_ROOT="${DGMESH_OUTPUT_ROOT:-$REPO_ROOT/outputs/paper_dgmesh}"
EVAL_ROOT="${DGMESH_EVAL_ROOT:-$HOME/dgmesh_eval}"
EVAL_REPO_ROOT="${DGMESH_REPO_ROOT:-$HOME/DG-Mesh}"

SOURCE_PATH="$DATA_ROOT/$SCENE"
MODEL_BASE="$OUTPUT_ROOT/$SCENE"
MODEL_PATH="${MODEL_BASE}_node"
EVAL_DIR="$EVAL_ROOT/$SCENE"
GT_SOURCE="$SOURCE_PATH/mesh_gt"
PRED_PATH="$MODEL_PATH/train/ours_${ITERATIONS}"
POINT_CLOUD_CHECKPOINT="$MODEL_PATH/point_cloud/iteration_${ITERATIONS}/point_cloud.ply"
DEFORM_CHECKPOINT="$MODEL_PATH/deform/iteration_${ITERATIONS}/deform.pth"
EXPECTED_FRAMES=200

test -f "$SOURCE_PATH/transforms_train.json"
test -f "$SOURCE_PATH/transforms_test.json"
test -d "$GT_SOURCE"
test -f "$EVAL_REPO_ROOT/dgmesh/mesh_evaluation.py"
mkdir -p "$OUTPUT_ROOT"

training_complete() {
    [[ "$FORCE" != "1" ]] &&
        test -f "$MODEL_PATH/cfg_args" &&
        test -f "$POINT_CLOUD_CHECKPOINT" &&
        test -f "$DEFORM_CHECKPOINT"
}

render_complete() {
    [[ "$FORCE" != "1" ]] &&
        test -d "$PRED_PATH" &&
        [[ "$(find "$PRED_PATH" -maxdepth 1 -type f -name 'frame_*.ply' | wc -l)" -eq "$EXPECTED_FRAMES" ]]
}

if training_complete; then
    echo "[DG-Mesh] Training already complete, skipping: $SCENE (iteration $ITERATIONS)"
else
    echo "[DG-Mesh] Training scene: $SCENE"
    CUDA_VISIBLE_DEVICES="$GPU_ID" "$PYTHON" train_gui.py \
        --source_path "$SOURCE_PATH" \
        --model_path "$MODEL_BASE" \
        --deform_type node --hyper_dim 8 --node_num 1024 \
        --is_blender --eval --gt_alpha_mask_as_scene_mask --local_frame \
        --resolution 1 --W 800 --H 800 \
        --test_iterations "$ITERATIONS" --save_iterations "$ITERATIONS" \
        --iterations "$ITERATIONS"
fi

test -f "$MODEL_PATH/cfg_args"
test -f "$POINT_CLOUD_CHECKPOINT"
test -f "$DEFORM_CHECKPOINT"

if render_complete; then
    echo "[DG-Mesh] Mesh sequence already complete, skipping: $PRED_PATH"
else
    echo "[DG-Mesh] Rendering and extracting ${EXPECTED_FRAMES}-frame mesh sequence: $SCENE"
    CUDA_VISIBLE_DEVICES="$GPU_ID" "$PYTHON" render_mesh.py \
        --source_path "$SOURCE_PATH" \
        --model_path "$MODEL_PATH" \
        --deform_type node --hyper_dim 8 --node_num 1024 \
        --is_blender --eval --local_frame --resolution 1
fi

test -d "$PRED_PATH"
test "$(find "$PRED_PATH" -maxdepth 1 -type f -name 'frame_*.ply' | wc -l)" -eq "$EXPECTED_FRAMES"
test "$(find "$GT_SOURCE" -maxdepth 1 -type f -name '*.obj' | wc -l)" -eq "$EXPECTED_FRAMES"

echo "[DG-Mesh] Arranging evaluator inputs: $SCENE"
mkdir -p "$EVAL_DIR/gt" "$EVAL_DIR/DGMesh/dynamic_mesh"
cp "$PRED_PATH"/frame_*.ply "$EVAL_DIR/DGMesh/dynamic_mesh/"
cp "$SOURCE_PATH/transforms_train.json" "$EVAL_DIR/transforms_train.json"

mapfile -t GT_MESHES < <(find "$GT_SOURCE" -maxdepth 1 -type f -name '*.obj' -print0 | sort -zV)
for i in "${!GT_MESHES[@]}"; do
    ln -sfn "${GT_MESHES[$i]}" "$EVAL_DIR/gt/frame_${i}.obj"
done

echo "[DG-Mesh] Running CD/EMD evaluation: $SCENE"
(
    cd "$EVAL_REPO_ROOT/dgmesh"
    "$EVAL_PYTHON" mesh_evaluation.py \
        --path "$EVAL_DIR" \
        --eval_type dgmesh
)

echo "[DG-Mesh] Complete: $SCENE"
echo "  Model: $MODEL_PATH"
echo "  Evaluation: $EVAL_DIR/DGMesh/results"
