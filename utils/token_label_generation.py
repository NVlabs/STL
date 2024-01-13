# Copyright (c) 2022-2023, NVIDIA Corporation & Affiliates. All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, visit
# https://github.com/NVlabs/STL/blob/master/LICENSE


import torch
# Token label generation
def token_label_generation(model, input, args, amp_autocast=None):
    model.eval()
    with torch.no_grad():
        if args.channels_last:
            input = input.contiguous(memory_format=torch.channels_last)
        
        with amp_autocast():
            output = model(input)
            aux_target = output[1]
          
            if args.softmax_token_label:
                aux_target_softmax = torch.nn.functional.softmax(aux_target, dim=-1)
                return aux_target_softmax
            elif args.single_token_label:
                aux_target = torch.mean(aux_target, dim=-2)
            
            if not args.no_gumbel:
                aux_target_one_hot = torch.nn.functional.gumbel_softmax(aux_target, tau=1, hard=True, dim=-1)
            else:
                
                aux_target_one_hot = torch.argmax(aux_target.softmax(-1),dim=-1)
                aux_target_one_hot = torch.nn.functional.one_hot(aux_target_one_hot, 1000)
            # add label smoothing
            aux_target_one_hot = token_label_smoothing(aux_target_one_hot, args.smoothing)
            

    return aux_target_one_hot


# Token label smoothing
def token_label_smoothing(y_hot, α=0.1, K=1000):
    # y_ls = (1 - α) * y_hot + α / K , K = # of class
    y_ls = (1-α)*y_hot + α/K
    
    return y_ls



