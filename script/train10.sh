dataname='torus2sphere'
datetime='0801'
d_type=node

CUDA_VISIBLE_DEVICES=0 python render_mesh.py --source_path "$HOME/autodl-tmp/dataset/dg-mesh/$dataname" --model_path outputs/${dataname}_${datetime} --deform_type $d_type --hyper_dim 8 --is_blender --eval --local_frame --resolution 1 --load2gpu_on_the_fly
