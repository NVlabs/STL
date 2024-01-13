#!/user/bin/env bash
# Copyright (c) 2022-2023, NVIDIA Corporation & Affiliates. All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, visit
# https://github.com/NVlabs/STL/blob/master/LICENSE

cd ../../
python3 -m torch.distributed.launch --nproc_per_node=8 main.py  ../../raid/imagenet_raw/ --model fan_small_12_p4_hybrid_token -b 128 --sched cosine --epochs 350 --opt adamw -j 16 --warmup-lr 1e-6 --warmup-epochs 5  --model-ema-decay 0.99992 --aa rand-m9-mstd0.5-inc1 --remode pixel --reprob 0.3 --lr 20e-4 --min-lr 1e-6 --weight-decay .05 --drop 0.0 --drop-path .25 --img-size 224 --mixup 0.8 --cutmix 1.0 --smoothing 0.1 --output ../trained_models/fan_small_12_p4_hybrid_stl/ --amp --model-ema --token-label --cls-weight 1.0 --dense-weight 1.0 --offline-model ../trained_models/fan_small_12_p4_hybrid_token_labeler/train/20220817-162417-fan_small_12_p4_hybrid_token-224/model_best.pth.tar 
