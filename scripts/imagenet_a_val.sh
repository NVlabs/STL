#!/bin/bash
# Copyright (c) 2022-2023, NVIDIA Corporation & Affiliates. All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, visit
# https://github.com/NVlabs/STL/blob/master/LICENSE

# Copyright (c) 2021-2022, NVIDIA Corporation & Affiliates. All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, visit
# https://github.com/NVlabs/FAN/blob/main/LICENSE

model_name=$1
ckpt=$2
CUDA_VISIBLE_DEVICES=0 python3 ./validate_ood.py \
        ../imagenet_a/ \
	--model $model_name \
        --img-size 224 \
        -b 128 \
        -j 32 \
	--no-test-pool \
        --imagenet_a \
	--results-file ../debug/ \
	--checkpoint $ckpt \
	--imagenet_a
