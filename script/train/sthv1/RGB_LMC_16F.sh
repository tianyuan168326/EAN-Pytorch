CUDA_VISIBLE_DEVICES=4,5,6,7 python -m torch.distributed.launch --nproc_per_node 4 ./main_reg.py \
--fp 32 --lr 0.01 -b 4 -j 8 \
--root_path ./ --dataset somethingv1 \
--type EAN_16f \
--agg_type TSN \
--input_size 224 --arch resnet50 --num_segments 16 \
--checkpoint_dir ./checkpoints/RGB_LMC_16F_sthv1 --evaluate \
--optim sgd --epochs 70 --lr_steps 41 51 61 --gd 20 -ef 1 --wd 5e-4 --dp 0.5 \
--resume /data/tianyuan/EAN-Pytorch/pretrained/Somethingv1.RGB_LMC_16f.pth.tar
