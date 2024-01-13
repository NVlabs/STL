# Copyright (c) 2022-2023, NVIDIA Corporation & Affiliates. All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, visit
# https://github.com/NVlabs/STL/blob/master/LICENSE

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class SoftTargetCrossEntropy(nn.Module):

    def __init__(self):
        super(SoftTargetCrossEntropy, self).__init__()

    def forward(self, x, target):
        N_rep = x.shape[0]
        N = target.shape[0]
        if not N==N_rep:
            target = target.repeat(N_rep//N,1)
        loss = torch.sum(-target * F.log_softmax(x, dim=-1), dim=-1)
        return loss.mean()
    
    

class TokenLabelCrossEntropy(nn.Module):
    """
    Token labeling loss.
    """
    def __init__(self, dense_weight=1.0, cls_weight = 1.0, mixup_active=True, classes = 1000, dynamic_dense_weight=False, epoch=310):
        """
        Constructor Token labeling loss.
        """
        super(TokenLabelCrossEntropy, self).__init__()


        self.CE = SoftTargetCrossEntropy()
        
        if dynamic_dense_weight and epoch <= 300:
            self.dense_weight = math.sqrt(epoch/300) * dense_weight 
        else:
            self.dense_weight = dense_weight
            
        self.mixup_active = mixup_active
        self.classes = classes
        self.cls_weight = cls_weight
        self.single_tl_training = False
        assert dense_weight+cls_weight>0
        
        
        
    def forward(self, x, target):
        
        if len(x) == 2 and len(x[1].shape) == 3:
            output, aux_output = x
            B,N,C = aux_output.shape
            cls_target, aux_target = target

        elif len(x) == 2 and len(x[1].shape) == 2:
            output, aux_output = x
            B,C = aux_output.shape
            cls_target, aux_target = target
            
        else:
            aux_output = x
            B,N,C = aux_output.shape
            aux_target = target
            self.single_tl_training = True
            
#         # process target_aux and aux_output for single-token-label
        if self.single_tl_training:
            aux_output = torch.mean(aux_output, dim=1)
            loss_aux = self.CE(aux_output, aux_target)

        elif len(aux_output.shape) == 2:
            loss_aux = self.CE(aux_output, aux_target)
            loss_cls = self.CE(output, cls_target)
            
        else:
            aux_target = aux_target.reshape(B*N,C)
            aux_output = aux_output.reshape(-1,C)
            loss_cls = self.CE(output, cls_target)
            loss_aux = self.CE(aux_output, aux_target)
        
        if self.single_tl_training:
            return self.dense_weight* loss_aux
        else:
            return self.cls_weight*loss_cls+self.dense_weight* loss_aux
    
    
    

        
