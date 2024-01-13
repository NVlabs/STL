# Copyright (c) 2022-2023, NVIDIA Corporation & Affiliates. All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, visit
# https://github.com/NVlabs/STL/blob/master/LICENSE

ngc batch run --instance dgx1v.32g.8.norm --result /results --name 'ml-model.fanV2_small_12_p4_hybrid_offline_self_training' --image "nvcr.io/nvidian/swaiinf/mypytorch:22.05-py3-fanv2" --org nvidian --team swaiinf --ace nv-us-west-2 --datasetid 103922:/dataset/imagenet_raw --datasetid 103923:/dataset/imagenet_raw_val --commandline "apt update; apt install tmux -y; cd ../FAN_V2/imagenet_tar/; bash -c \"bash preprocess_dataset_to_raid.sh \"; sleep 167h" --workspace ul1-WvoGTd2002uZIwJ9oQ:/FAN_V2 -p 8080 -p 8081 -p 8082 -p 8083 -p 8084 -p 8085
