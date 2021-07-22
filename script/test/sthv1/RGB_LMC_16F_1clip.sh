CUDA_VISIBLE_DEVICES=8 python ./test_models.py \
--fp 32 --lr 0.01 -b 1 -j 4 \
--root_path ./ --dataset somethingv1 \
--type EAN_16f \
--agg_type TSN \
--input_size 224 --arch resnet50 --num_segments 16 --num_clips 1 --beta 2 --alpha 4 \
--resume ./pretrained/Somethingv1.RGB_LMC_16f.pth.tar