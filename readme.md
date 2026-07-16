<div align="center">
<h1 align="center">
  Dynamic 2D Gaussians: Geometrically Accurate Radiance Fields for Dynamic Objects
</h1>
  
### ACMMM 2025
  
### [arXiv Paper](https://arxiv.org/abs/2409.14072)

[Shuai Zhang](https://github.com/Shuaizhang7) <sup>1\*</sup>, [Guanjun Wu](https://guanjunwu.github.io/) <sup>2\*</sup>,[Zhoufeng Xie]() <sup>1</sup>,[Xinggang Wang](https://xwcv.github.io/) <sup>1</sup>,[Bin Feng](https://scholar.google.com/citations?user=nRc8u6gAAAAJ&hl=en&oi=ao) <sup>1</sup>,
[Wenyu Liu](http://eic.hust.edu.cn/professor/liuwenyu) <sup>1,📧</sup>

<sup>1</sup> School of Electronic Information and Communications, Huazhong University of Science and Technology \
<sup>2</sup>  School of Computer Science & Technology, Huazhong University of Science and Technology 

(\* Equal contributions.📧 Corresponding author) 

</div>

---

## Abstract

Reconstructing objects and extracting high-quality surfaces play a vital role in the real world. Current 4D representations show the ability to render high-quality novel views for dynamic objects but cannot reconstruct high-quality meshes due to their implicit or geometrically inaccurate representations. In this paper, we propose a novel representation that can reconstruct accurate meshes from sparse image input, named Dynamic 2D Gaussians (D-2DGS). We adopt 2D Gaussians for basic geometry representation and use sparse-controlled points to capture 2D Gaussian's deformation. By extracting the object mask from the rendered high-quality image and masking the rendered depth map, a high-quality dynamic mesh sequence of the object can be extracted. Experiments demonstrate that our D-2DGS is outstanding in reconstructing high-quality meshes from sparse input.

<div align="center">
  <img src="./assets/teaser.png" width="100%" height="100%">
</div>

*Framework of our D-2DGS. Sparse points are bonded with canonical 2D Gaussians. Deformation networks are used to predict each sparse control point's control signals given any timestamp. The image and depth are rendered by deformed 2D Gaussians with alpha blending. To get high-quality meshes, depth images are filtered by rendered images with RGB mask, and then TSDF is applied on multiview depth images and RGB images.*

## Demo

<div align="center">
  <img src="./assets/bouncingballs.gif" width="24.5%">
  <img src="./assets/horse.gif" width="24.5%">
  <img src="./assets/jumpingjacks.gif" width="24.5%">
  <img src="./assets/standup.gif" width="24.5%">

</div>

## Updates

- 2025-07-05: Accepted by ACMMM 2025

- 2024-09-24: Release code

## Installation

```bash
git clone --recursive git@github.com:hustvl/Dynamic-2DGS.git
cd Dynamic-2DGS
conda create --name dynamic-2dgs python=3.8.0
conda activate dynamic-2dgs

pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu121

pip install ./submodules/diff-surfel-rasterization
pip install ./submodules/simple-knn
pip install git+https://github.com/NVlabs/nvdiffrast/
pip install "git+https://github.com/facebookresearch/pytorch3d.git"

pip install -r requirements.txt
```

## Data
We use the following dataset:
* [D-NeRF](https://www.albertpumarola.com/research/D-NeRF/index.html): dynamic scenes of synthetic objects ([download](https://www.dropbox.com/s/0bf6fl0ye2vz3vr/data.zip?e=1&dl=0))
* [DG-Mesh](https://github.com/Isabella98Liu/DG-Mesh?tab=readme-ov-file): dynamic scenes of synthetic objects ([download](https://drive.google.com/file/d/1yBga2DsIKG6zQK9V2WextewvhaV8Xho3/view))

## Run

Activate the environment first:

```bash
source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate dynamic-2dgs
```

### Training

Train the `jumpingjacks` scene for 80,000 iterations. The training command writes the model to `outputs/jumpingjacks_node` because `train_gui.py` appends the selected deformation type to the model path.

```bash
CUDA_VISIBLE_DEVICES=0 python train_gui.py \
  --source_path "$HOME/autodl-tmp/dataset/data/jumpingjacks" \
  --model_path outputs/jumpingjacks \
  --deform_type node \
  --hyper_dim 8 --node_num 1024 \
  --is_blender --eval --gt_alpha_mask_as_scene_mask --local_frame \
  --resolution 1 --W 800 --H 800 \
  --load2gpu_on_the_fly --iterations 80000
```

`--load2gpu_on_the_fly` keeps camera images on the CPU and loads them when needed; it is included for the 8GB GPU setup and does not change the model architecture.

### Rendering

Run the complete rendering and mesh extraction pipeline for the trained model. This single command exports training/test renders, RGB/depth/normal visualizations, dynamic meshes, and mesh-rendered images. No `--skip_*` option is needed.

```bash
CUDA_VISIBLE_DEVICES=0 python render_mesh.py \
  --source_path "$HOME/autodl-tmp/dataset/data/jumpingjacks" \
  --model_path outputs/jumpingjacks_node \
  --deform_type node --hyper_dim 8 --node_num 1024 \
  --is_blender --eval --local_frame --resolution 1 \
  --load2gpu_on_the_fly
```

The main outputs are:

```text
outputs/jumpingjacks_node/test/ours_80000/renders/
outputs/jumpingjacks_node/test/ours_80000/gt/
outputs/jumpingjacks_node/test/ours_80000/gt_w/
outputs/jumpingjacks_node/train/ours_80000/frame_*.ply
outputs/jumpingjacks_node/mesh_image/
outputs/jumpingjacks_node/mesh_shape/
```

Optionally render a camera trajectory video:

```bash
CUDA_VISIBLE_DEVICES=0 python render_mesh_trajectory.py \
  --source_path "$HOME/autodl-tmp/dataset/data/jumpingjacks" \
  --model_path outputs/jumpingjacks_node \
  --deform_type node --hyper_dim 8 \
  --is_blender --eval --local_frame --resolution 1 \
  --load2gpu_on_the_fly
```

### Evaluation

Evaluate the Gaussian-rendered test images. Higher PSNR/SSIM and lower LPIPS are better. This evaluates the radiance-field rendering, not the extracted mesh geometry.

```bash
CUDA_VISIBLE_DEVICES=0 python metrics.py -m outputs/jumpingjacks_node
```

Evaluate the locally generated mesh-rendered images with the repository helper. `render_mesh.py` creates `gt_w/` by compositing the RGBA test images onto a white background, matching the white background used by the mesh renderer:

```bash
CUDA_VISIBLE_DEVICES=0 python metrics_mesh.py --model_path outputs/jumpingjacks_node
```

This compares `mesh_image/` with `test/ours_*/gt_w`. If an existing output was rendered before `gt_w` support was added, run `render_mesh.py` once with `--skip_train --skip_mesh` to regenerate the test images and GT without retraining or re-extracting the mesh. For the paper's D-NeRF Table 1 protocol, use white-background GT images; the paper evaluates rendered images from extracted meshes because D-NeRF has no ground-truth mesh. For DG-Mesh CD/EMD evaluation, use the official [DG-Mesh mesh evaluation](https://github.com/Isabella98Liu/DG-Mesh#mesh-evaluation) after arranging the generated `frame_*.ply` files in its required directory structure.

### Batch reproduction

The following blocks must be executed in order in the same terminal after activating `dynamic-2dgs`. They reproduce the main quantitative results of the paper: D-NeRF mesh-rendering results (Table 1), DG-Mesh mesh metrics (Table 3), and their averages (Table 2). The commands run one scene at a time on GPU 0 of the single RTX 4090D (24 GB VRAM).

The existing single-scene `jumpingjacks` output is not used here. Batch reproduction always trains every scene to 80,000 iterations and writes fresh results under `outputs/paper_dnerf/` and `outputs/paper_dgmesh/`.

#### 1. Define the scene lists

```bash
set -euo pipefail

# Server configuration: one RTX 4090D.
export CUDA_VISIBLE_DEVICES=0

DNERF_SCENES=(lego bouncingballs jumpingjacks hook mutant standup trex hellwarrior)
DGMESH_SCENES=(duck horse bird beagle torus2sphere girlwalk)
```

Because this server has 24 GB of VRAM, the batch commands below keep camera images on the GPU and omit `--load2gpu_on_the_fly`, which was intended for the original 8 GB GPU setup. Add that flag back to the training and rendering commands if VRAM usage becomes a problem.

#### 2. Train all D-NeRF scenes

```bash
for scene in "${DNERF_SCENES[@]}"; do
  test -f "$HOME/autodl-tmp/dataset/data/$scene/transforms_train.json"
  CUDA_VISIBLE_DEVICES=0 python train_gui.py \
    --source_path "$HOME/autodl-tmp/dataset/data/$scene" \
    --model_path "outputs/paper_dnerf/$scene" \
    --deform_type node --hyper_dim 8 --node_num 1024 \
    --is_blender --eval --gt_alpha_mask_as_scene_mask --local_frame \
    --resolution 1 --W 800 --H 800 \
    --test_iterations 80000 --save_iterations 80000 \
    --iterations 80000
done
```

#### 3. Render all D-NeRF scenes and extract meshes

```bash
for scene in "${DNERF_SCENES[@]}"; do
  test -f "outputs/paper_dnerf/${scene}_node/cfg_args"
  CUDA_VISIBLE_DEVICES=0 python render_mesh.py \
    --source_path "$HOME/autodl-tmp/dataset/data/$scene" \
    --model_path "outputs/paper_dnerf/${scene}_node" \
    --deform_type node --hyper_dim 8 --node_num 1024 \
    --is_blender --eval --local_frame --resolution 1
done
```

After this step, every scene must contain `train/ours_80000/frame_*.ply`, `mesh_image/`, and `test/ours_80000/gt_w/`. The `gt_w/` images are the white-background GT required by the paper's D-NeRF mesh-rendering protocol.

#### 4. Evaluate direct Gaussian rendering (repository diagnostic)

```bash
DNERF_MODELS=()
for scene in "${DNERF_SCENES[@]}"; do
  DNERF_MODELS+=("outputs/paper_dnerf/${scene}_node")
done

CUDA_VISIBLE_DEVICES=0 python metrics.py -m "${DNERF_MODELS[@]}"
```

This writes the standard Gaussian-rendering metrics to each model directory. It is an additional diagnostic and is not the mesh-rendering metric reported in D-NeRF Table 1.

#### 5. Evaluate D-NeRF mesh rendering (Table 1)

```bash
for scene in "${DNERF_SCENES[@]}"; do
  test "$(find "outputs/paper_dnerf/${scene}_node/mesh_image" -maxdepth 1 -name '*.png' | wc -l)" -eq 20
  test "$(find "outputs/paper_dnerf/${scene}_node/test/ours_80000/gt_w" -maxdepth 1 -name '*.png' | wc -l)" -eq 20
  CUDA_VISIBLE_DEVICES=0 python metrics_mesh.py \
    --model_path "outputs/paper_dnerf/${scene}_node"
done
```

Each scene's result is saved as `outputs/paper_dnerf/<scene>_node/mesh_render_results.json`. Aggregate the eight D-NeRF scenes for the D-NeRF part of Table 2:

```bash
python - <<'PY'
import json
from pathlib import Path

scenes = ("lego", "bouncingballs", "jumpingjacks", "hook", "mutant", "standup", "trex", "hellwarrior")
root = Path("outputs/paper_dnerf")
rows = []
for scene in scenes:
    path = root / f"{scene}_node" / "mesh_render_results.json"
    if not path.is_file():
        raise FileNotFoundError(path)
    rows.append(json.loads(path.read_text()))

average = {
    key: sum(row[key] for row in rows) / len(rows)
    for key in ("PSNR", "SSIM", "LPIPS")
}
result = {"per_scene": dict(zip(scenes, rows)), "average": average}
(root / "mesh_render_average.json").write_text(json.dumps(result, indent=2))
print(json.dumps(average, indent=2))
PY
```

#### 6. Train all DG-Mesh scenes

```bash
DGMESH_SCENES=(duck horse bird beagle torus2sphere girlwalk)

for scene in "${DGMESH_SCENES[@]}"; do
  test -f "$HOME/autodl-tmp/dataset/dg-mesh/$scene/transforms_train.json"
  CUDA_VISIBLE_DEVICES=0 python train_gui.py \
    --source_path "$HOME/autodl-tmp/dataset/dg-mesh/$scene" \
    --model_path "outputs/paper_dgmesh/$scene" \
    --deform_type node --hyper_dim 8 --node_num 1024 \
    --is_blender --eval --gt_alpha_mask_as_scene_mask --local_frame \
    --resolution 1 --W 800 --H 800 \
    --test_iterations 80000 --save_iterations 80000 \
    --iterations 80000
done
```

#### 7. Render all DG-Mesh scenes and extract the 200-frame mesh sequences

```bash
for scene in "${DGMESH_SCENES[@]}"; do
  test -f "outputs/paper_dgmesh/${scene}_node/cfg_args"
  CUDA_VISIBLE_DEVICES=0 python render_mesh.py \
    --source_path "$HOME/autodl-tmp/dataset/dg-mesh/$scene" \
    --model_path "outputs/paper_dgmesh/${scene}_node" \
    --deform_type node --hyper_dim 8 --node_num 1024 \
    --is_blender --eval --local_frame --resolution 1
  test "$(find "outputs/paper_dgmesh/${scene}_node/train/ours_80000" -maxdepth 1 -name 'frame_*.ply' | wc -l)" -eq 200
done
```

#### 8. Install the official DG-Mesh evaluator

The paper reports CD and EMD for the DG-Mesh dataset. The official evaluator is maintained in the [DG-Mesh repository](https://github.com/Isabella98Liu/DG-Mesh#mesh-evaluation), so install it in a separate Python 3.9 environment. The Python 3.9 environment is required by the evaluator's CUDA extension build.

```bash
DG_MESH_EVAL_ROOT="$HOME/DG-Mesh"
if [ ! -d "$DG_MESH_EVAL_ROOT" ]; then
  git clone --depth 1 https://github.com/Isabella98Liu/DG-Mesh.git "$DG_MESH_EVAL_ROOT"
fi

if ! conda env list | awk 'NR > 2 {print $1}' | grep -qx dgmesh-eval; then
  conda create -n dgmesh-eval python=3.9 -y
fi

conda activate dgmesh-eval
python -m pip install torch==2.1.0 torchvision==0.16.0 \
  --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r "$DG_MESH_EVAL_ROOT/requirements.txt"

cd "$DG_MESH_EVAL_ROOT/dgmesh/metrics/pytorch_structural_losses"
make clean
make CUDA_ARCH="-gencode arch=compute_89,code=sm_89"
cd "$OLDPWD"
```

The `compute_89` setting matches the RTX 4090D's CUDA compute capability (sm_89).

#### 9. Arrange the DG-Mesh GT and predictions

The official evaluator pairs sorted `gt/*.obj` and `dynamic_mesh/frame_*.ply` files. The following command creates the required directory structure and renames GT meshes by numeric frame order, preventing lexicographic filename mismatches:

```bash
DG_EVAL_ROOT="$HOME/dgmesh_eval"

for scene in "${DGMESH_SCENES[@]}"; do
  eval_dir="$DG_EVAL_ROOT/$scene"
  pred_dir="outputs/paper_dgmesh/${scene}_node/train/ours_80000"
  gt_source="$HOME/autodl-tmp/dataset/dg-mesh/$scene/mesh_gt"

  test "$(find "$pred_dir" -maxdepth 1 -name 'frame_*.ply' | wc -l)" -eq 200
  test "$(find "$gt_source" -maxdepth 1 -name '*.obj' | wc -l)" -eq 200

  mkdir -p "$eval_dir/gt" "$eval_dir/DG-Mesh/dynamic_mesh"
  cp "$pred_dir"/frame_*.ply "$eval_dir/DG-Mesh/dynamic_mesh/"
  cp "$HOME/autodl-tmp/dataset/dg-mesh/$scene/transforms_train.json" "$eval_dir/transforms_train.json"

  mapfile -t gt_meshes < <(find "$gt_source" -maxdepth 1 -name '*.obj' -print0 | sort -zV)
  for i in "${!gt_meshes[@]}"; do
    ln -sfn "${gt_meshes[$i]}" "$eval_dir/gt/frame_${i}.obj"
  done
done
```

#### 10. Run CD/EMD evaluation (Table 3)

```bash
cd "$DG_MESH_EVAL_ROOT/dgmesh"
for scene in "${DGMESH_SCENES[@]}"; do
  python mesh_evaluation.py \
    --path "$DG_EVAL_ROOT/$scene" \
    --eval_type dgmesh
done
cd "$OLDPWD"
```

Each scene's official result is written below `~/dgmesh_eval/<scene>/DG-Mesh/results/`.

#### 11. Aggregate DG-Mesh results for Table 2

```bash
python - <<'PY'
import json
import re
from pathlib import Path

scenes = ("duck", "horse", "bird", "beagle", "torus2sphere", "girlwalk")
root = Path.home() / "dgmesh_eval"
per_scene = {}
for scene in scenes:
    logs = sorted((root / scene / "DG-Mesh" / "results").glob("*/eval_results.txt"))
    if not logs:
        raise FileNotFoundError(f"No evaluator result for {scene}")
    text = logs[-1].read_text()
    cd = float(re.search(r"Average Chamfer distance: ([0-9.]+)", text).group(1))
    emd = float(re.search(r"Average EMD: ([0-9.]+)", text).group(1))
    per_scene[scene] = {"CD": cd, "EMD": emd}

average = {
    key: sum(row[key] for row in per_scene.values()) / len(per_scene)
    for key in ("CD", "EMD")
}
result = {"per_scene": per_scene, "average": average}
out = Path("outputs/paper_dgmesh/dgmesh_mesh_average.json")
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(result, indent=2))
print(json.dumps(average, indent=2))
PY
```

The resulting per-scene JSON files correspond to Table 1 and Table 3; the two average JSON files provide the D-NeRF and DG-Mesh components of Table 2. Table 4 is a hardware benchmark and cannot match the paper's RTX A5000 timing/memory numbers on a different GPU. Table 5 is a separate ablation study and is not part of this full-scene batch.

### 3D Printing
Reconstruct the mesh through our model and 3D print it:
<div align="center">
<img src="./assets/print.jpg" width="45%">
<img src="./assets/trex.gif" width="35%">
</div>

## Citation
If you find our work useful, please consider citing:
```BibTeX
@misc{zhang2024dynamic2dgaussiansgeometrically,
      title={Dynamic 2D Gaussians: Geometrically accurate radiance fields for dynamic objects}, 
      author={Shuai Zhang and Guanjun Wu and Xinggang Wang and Bin Feng and Wenyu Liu},
      year={2024},
      eprint={2409.14072},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2409.14072}, 
}
```

## Acknowledgement

* Our work is developed on the basis of [DG-Mesh](https://github.com/Isabella98Liu/DG-Mesh), [SCGS](https://github.com/yihua7/SC-GS) and [2DGS](https://github.com/hbb1/2d-gaussian-splatting/tree/main), thanks to these great works.

```
@misc{liu2024dynamic,
 title={Dynamic Gaussians Mesh: Consistent Mesh Reconstruction from Monocular Videos}, 
 author={Isabella Liu and Hao Su and Xiaolong Wang},
 year={2024},
 eprint={2404.12379},
 archivePrefix={arXiv},
 primaryClass={cs.CV}
}

@article{huang2023sc,
  title={SC-GS: Sparse-Controlled Gaussian Splatting for Editable Dynamic Scenes},
  author={Huang, Yi-Hua and Sun, Yang-Tian and Yang, Ziyi and Lyu, Xiaoyang and Cao, Yan-Pei and Qi, Xiaojuan},
  journal={arXiv preprint arXiv:2312.14937},
  year={2023}
}

@inproceedings{Huang2DGS2024,
    title={2D Gaussian Splatting for Geometrically Accurate Radiance Fields},
    author={Huang, Binbin and Yu, Zehao and Chen, Anpei and Geiger, Andreas and Gao, Shenghua},
    publisher = {Association for Computing Machinery},
    booktitle = {SIGGRAPH 2024 Conference Papers},
    year      = {2024},
    doi       = {10.1145/3641519.3657428}
}
```
