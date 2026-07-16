dataname='bouncingballs'
datetime='0920'
d_type=node

CUDA_VISIBLE_DEVICES=0 python train_gui.py --source_path "$HOME/dataset/data/$dataname" --model_path outputs/${dataname}_${datetime} --deform_type $d_type  --is_blender --eval --gt_alpha_mask_as_scene_mask --local_frame --resolution 1 --W 800 --H 800 --load2gpu_on_the_fly
CUDA_VISIBLE_DEVICES=0 python render_mesh.py --source_path "$HOME/dataset/data/$dataname" --model_path outputs/${dataname}_${datetime} --deform_type $d_type --hyper_dim 8 --is_blender --eval --local_frame --resolution 1 --load2gpu_on_the_fly
