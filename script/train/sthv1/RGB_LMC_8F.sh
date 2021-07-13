CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node 8 ./main_reg.py \
--fp 32 --lr 0.01 -b 8 -j 8 \
--root_path ./ --dataset somethingv1 \
--type EAN \
--agg_type TSN \
--input_size 224 --arch resnet50 --num_segments 8 \
--checkpoint_dir ./checkpoints/RGB_LMC_8F_sthv1 \
--optim sgd --epochs 70 --lr_steps 41 51 61 --gd 20 -ef 1 --wd 5e-4 --dp 0.5 \
