# Copyright (c) 2022-2023, NVIDIA Corporation & Affiliates. All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, visit
# https://github.com/NVlabs/STL/blob/master/LICENSE

""" Transforms Factory
Adapted for token labeling
"""
import math

import torch
from torchvision import transforms

from .random_augment_clean_teacher import rand_augment_transform
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD, DEFAULT_CROP_PCT
from .transforms import str_to_interp_mode, str_to_pil_interp, RandomResizedCropAndInterpolation, ToNumpy
# from timm.data.transforms import str_to_interp_mode, str_to_pil_interp, RandomResizedCropAndInterpolation, ToNumpy
from .random_erasing_clean_teacher import RandomErasing
import random


import torchvision
from torchvision.transforms import functional as torchvision_F
from PIL import Image


class ComposeWithCleanTeacher(torchvision.transforms.Compose):
    def __init__(self, **kwargs):
        super(ComposeWithCleanTeacher, self).__init__(**kwargs)

    def __call__(self, img_teacher, img):
        for t in self.transforms:
            if type(t).__name__ == 'CustomRandomHorizontalFlip':
                img_teacher, img = t(img_teacher, img)
            elif type(t).__name__ == 'CustomRandomVerticalFlip':
                img_teacher, img = t(img_teacher, img)
            elif type(t).__name__ == 'RandAugment':
                img_teacher, img = t(img_teacher, img)
            else:
                img_teacher, img = t(img_teacher, img)
        return img_teacher, img
    

class CustomRandomHorizontalFlip(torchvision.transforms.RandomHorizontalFlip):
    def __init__(self, **kwargs):
        super(CustomRandomHorizontalFlip, self).__init__(**kwargs)

    def __call__(self, img_teacher, img):
        if torch.rand(1) < self.p:
            return torchvision_F.hflip(img_teacher), torchvision_F.hflip(img) 
        return img_teacher, img

class CustomRandomVerticalFlip(torchvision.transforms.RandomVerticalFlip):
    def __init__(self, **kwargs):
        super(CustomRandomVerticalFlip, self).__init__(**kwargs)

    def __call__(self, img_teacher, img):
        if torch.rand(1) < self.p:
            return torchvision_F.vflip(img_teacher), torchvision_F.vflip(img)
        return img_teacher, img

    
class CustomToTensor(torchvision.transforms.ToTensor):
    def __init__(self, **kwargs):
        super(CustomToTensor, self).__init__(**kwargs)
        
    def __call__(self, pic_teacher, pic):
        """
        Args:
            pic (PIL Image or numpy.ndarray): Image to be converted to tensor.

        Returns:
            Tensor: Converted image.
        """
        return F.to_tensor(pic_teacher), F.to_tensor(pic)
    

class CustomNormalize(torchvision.transforms.Normalize):
    def __init__(self, **kwargs):
        super(CustomNormalize, self).__init__(**kwargs)
        
    def __call__(self, tensor_teacher, tensor):
        """
        Args:
            pic (PIL Image or numpy.ndarray): Image to be converted to tensor.

        Returns:
            Tensor: Converted image.
        """
        return F.normalize(tensor_teacher, self.mean, self.std, self.inplace), F.normalize(tensor, self.mean, self.std, self.inplace)
        
    


def transforms_imagenet_train(
        img_size=224,
        scale=None,
        ratio=None,
        hflip=0.5,
        vflip=0.,
        color_jitter=0.4,
        auto_augment=None,
        interpolation='random',
        use_prefetcher=False,
        mean=IMAGENET_DEFAULT_MEAN,
        std=IMAGENET_DEFAULT_STD,
        re_prob=0.,
        re_mode='const',
        re_count=1,
        re_num_splits=0,
        separate=False,
):
    """
    If separate==True, the transforms are returned as a tuple of 3 separate transforms
    for use in a mixing dataset that passes
     * all data through the first (primary) transform, called the 'teacher' data
     * a portion of the data through the secondary transform
     * normalizes and converts the branches above with the third, final transform
    """
    scale = tuple(scale or (0.08, 1.0))  # default imagenet scale range
    ratio = tuple(ratio or (3./4., 4./3.))  # default imagenet ratio range
    primary_tfl = [
        RandomResizedCropAndInterpolation(img_size, scale=scale, ratio=ratio, interpolation=interpolation)]
    if hflip > 0.:
        primary_tfl += [CustomRandomHorizontalFlip(p=hflip)]
    if vflip > 0.:
        primary_tfl += [CustomRandomVerticalFlip(p=vflip)]

    secondary_tfl = []
    if auto_augment:
        assert isinstance(auto_augment, str)
        if isinstance(img_size, (tuple, list)):
            img_size_min = min(img_size)
        else:
            img_size_min = img_size
        aa_params = dict(
            translate_const=int(img_size_min * 0.45),
            img_mean=tuple([min(255, round(255 * x)) for x in mean]),
        )
        if interpolation and interpolation != 'random':
            aa_params['interpolation'] = str_to_pil_interp(interpolation)
        if auto_augment.startswith('rand'):
            secondary_tfl += [rand_augment_transform(auto_augment, aa_params)]
        elif auto_augment.startswith('augmix'):
            aa_params['translate_pct'] = 0.3
            secondary_tfl += [augment_and_mix_transform(auto_augment, aa_params)]
        else:
            secondary_tfl += [auto_augment_transform(auto_augment, aa_params)]
    elif color_jitter is not None:
        # color jitter is enabled when not using AA
        if isinstance(color_jitter, (list, tuple)):
            # color jitter should be a 3-tuple/list if spec brightness/contrast/saturation
            # or 4 if also augmenting hue
            assert len(color_jitter) in (3, 4)
        else:
            # if it's a scalar, duplicate for brightness, contrast, and saturation, no hue
            color_jitter = (float(color_jitter),) * 3
        secondary_tfl += [transforms.ColorJitter(*color_jitter)]

    final_tfl = []
    if use_prefetcher:
        # prefetcher and collate will handle tensor conversion and norm
        final_tfl += [ToNumpy()]
    else:
        final_tfl += [
            CustomToTensor(),
            CustomNormalize(
                mean=torch.tensor(mean),
                std=torch.tensor(std))
        ]
        if re_prob > 0.:
            final_tfl.append(
                RandomErasing(re_prob, mode=re_mode, max_count=re_count, num_splits=re_num_splits, device='cpu'))


    return ComposeWithCleanTeacher(transforms=primary_tfl + secondary_tfl + final_tfl)


def create_transform_clean_teacher(
        input_size,
        is_training=False,
        use_prefetcher=False,
        no_aug=False,
        scale=None,
        ratio=None,
        hflip=0.5,
        vflip=0.,
        color_jitter=0.4,
        auto_augment=None,
        interpolation='bilinear',
        mean=IMAGENET_DEFAULT_MEAN,
        std=IMAGENET_DEFAULT_STD,
        re_prob=0.,
        re_mode='const',
        re_count=1,
        re_num_splits=0,
        crop_pct=None,
        tf_preprocessing=False,
        separate=False,):

    if isinstance(input_size, tuple):
        img_size = input_size[-2:]
    else:
        img_size = input_size

    transform = transforms_imagenet_train(
        img_size,
        scale=scale,
        ratio=ratio,
        hflip=hflip,
        vflip=vflip,
        color_jitter=color_jitter,
        auto_augment=auto_augment,
        interpolation=interpolation,
        use_prefetcher=use_prefetcher,
        mean=mean,
        std=std,
        re_prob=re_prob,
        re_mode=re_mode,
        re_count=re_count,
        re_num_splits=re_num_splits,
        separate=separate)

    return transform
