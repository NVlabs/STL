# Copyright (c) 2022-2023, NVIDIA Corporation & Affiliates. All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, visit
# https://github.com/NVlabs/STL/blob/master/LICENSE

""" AutoAugment, RandAugment, and AugMix for PyTorch
This code implements the searched ImageNet policies with various tweaks and improvements and
does not include any of the search code.
AA and RA Implementation adapted from:
    https://github.com/tensorflow/tpu/blob/master/models/official/efficientnet/autoaugment.py
AugMix adapted from:
    https://github.com/google-research/augmix
Papers:
    AutoAugment: Learning Augmentation Policies from Data - https://arxiv.org/abs/1805.09501
    Learning Data Augmentation Strategies for Object Detection - https://arxiv.org/abs/1906.11172
    RandAugment: Practical automated data augmentation... - https://arxiv.org/abs/1909.13719
    AugMix: A Simple Data Processing Method to Improve Robustness and Uncertainty - https://arxiv.org/abs/1912.02781
Hacked together by / Copyright 2019, Ross Wightman
"""
import random
import math
import re
from PIL import Image, ImageOps, ImageEnhance, ImageChops
import PIL
import numpy as np


_PIL_VER = tuple([int(x) for x in PIL.__version__.split('.')[:2]])

_FILL = (128, 128, 128)

_LEVEL_DENOM = 10.  # denominator for conversion from 'Mx' magnitude scale to fractional aug level for op arguments

_HPARAMS_DEFAULT = dict(
    translate_const=250,
    img_mean=_FILL,
)

if hasattr(Image, "Resampling"):
    _RANDOM_INTERPOLATION = (Image.Resampling.BILINEAR, Image.Resampling.BICUBIC)
    _DEFAULT_INTERPOLATION = Image.Resampling.BICUBIC
else:
    _RANDOM_INTERPOLATION = (Image.BILINEAR, Image.BICUBIC)
    _DEFAULT_INTERPOLATION = Image.BICUBIC


def _interpolation(kwargs):
    interpolation = kwargs.pop('resample', _DEFAULT_INTERPOLATION)
    if isinstance(interpolation, (list, tuple)):
        return random.choice(interpolation)
    else:
        return interpolation


def _check_args_tf(kwargs):
    if 'fillcolor' in kwargs and _PIL_VER < (5, 0):
        kwargs.pop('fillcolor')
    kwargs['resample'] = _interpolation(kwargs)


def shear_x(img, factor, **kwargs):
    _check_args_tf(kwargs)
    return img.transform(img.size, Image.AFFINE, (1, factor, 0, 0, 1, 0), **kwargs)


def shear_y(img, factor, **kwargs):
    _check_args_tf(kwargs)
    return img.transform(img.size, Image.AFFINE, (1, 0, 0, factor, 1, 0), **kwargs)


def translate_x_rel(img, pct, **kwargs):
    pixels = pct * img.size[0]
    _check_args_tf(kwargs)
    return img.transform(img.size, Image.AFFINE, (1, 0, pixels, 0, 1, 0), **kwargs)


def translate_y_rel(img, pct, **kwargs):
    pixels = pct * img.size[1]
    _check_args_tf(kwargs)
    return img.transform(img.size, Image.AFFINE, (1, 0, 0, 0, 1, pixels), **kwargs)


def translate_x_abs(img, pixels, **kwargs):
    _check_args_tf(kwargs)
    return img.transform(img.size, Image.AFFINE, (1, 0, pixels, 0, 1, 0), **kwargs)


def translate_y_abs(img, pixels, **kwargs):
    _check_args_tf(kwargs)
    return img.transform(img.size, Image.AFFINE, (1, 0, 0, 0, 1, pixels), **kwargs)


def rotate(img, degrees, **kwargs):
    _check_args_tf(kwargs)
    if _PIL_VER >= (5, 2):
        return img.rotate(degrees, **kwargs)
    elif _PIL_VER >= (5, 0):
        w, h = img.size
        post_trans = (0, 0)
        rotn_center = (w / 2.0, h / 2.0)
        angle = -math.radians(degrees)
        matrix = [
            round(math.cos(angle), 15),
            round(math.sin(angle), 15),
            0.0,
            round(-math.sin(angle), 15),
            round(math.cos(angle), 15),
            0.0,
        ]

        def transform(x, y, matrix):
            (a, b, c, d, e, f) = matrix
            return a * x + b * y + c, d * x + e * y + f

        matrix[2], matrix[5] = transform(
            -rotn_center[0] - post_trans[0], -rotn_center[1] - post_trans[1], matrix
        )
        matrix[2] += rotn_center[0]
        matrix[5] += rotn_center[1]
        return img.transform(img.size, Image.AFFINE, matrix, **kwargs)
    else:
        return img.rotate(degrees, resample=kwargs['resample'])


def auto_contrast(img, **__):
    return ImageOps.autocontrast(img)


def invert(img, **__):
    return ImageOps.invert(img)


def equalize(img, **__):
    return ImageOps.equalize(img)


def solarize(img, thresh, **__):
    return ImageOps.solarize(img, thresh)


def solarize_add(img, add, thresh=128, **__):
    lut = []
    for i in range(256):
        if i < thresh:
            lut.append(min(255, i + add))
        else:
            lut.append(i)
    if img.mode in ("L", "RGB"):
        if img.mode == "RGB" and len(lut) == 256:
            lut = lut + lut + lut
        return img.point(lut)
    else:
        return img


def posterize(img, bits_to_keep, **__):
    if bits_to_keep >= 8:
        return img
    return ImageOps.posterize(img, bits_to_keep)


def contrast(img, factor, **__):
    return ImageEnhance.Contrast(img).enhance(factor)


def color(img, factor, **__):
    return ImageEnhance.Color(img).enhance(factor)


def brightness(img, factor, **__):
    return ImageEnhance.Brightness(img).enhance(factor)


def sharpness(img, factor, **__):
    return ImageEnhance.Sharpness(img).enhance(factor)    

# def shear_x(img_teacher, img, factor, **kwargs):
#     _check_args_tf(kwargs)
#     return img_teacher.transform(img_teacher.size, Image.AFFINE, (1, factor, 0, 0, 1, 0), **kwargs), img.transform(img.size, Image.AFFINE, (1, factor, 0, 0, 1, 0), **kwargs)


# def shear_y(img_teacher, img, factor, **kwargs):
#     _check_args_tf(kwargs)
#     return img_teacher.transform(img_teacher.size, Image.AFFINE, (1, 0, 0, factor, 1, 0), **kwargs), img.transform(img.size, Image.AFFINE, (1, 0, 0, factor, 1, 0), **kwargs)


# def translate_x_rel(img_teacher, img, pct, **kwargs):
#     pixels = pct * img.size[0]
#     _check_args_tf(kwargs)
#     return img_teacher.transform(img_teacher.size, Image.AFFINE, (1, 0, pixels, 0, 1, 0), **kwargs), img.transform(img.size, Image.AFFINE, (1, 0, pixels, 0, 1, 0), **kwargs)


# def translate_y_rel(img_teacher, img, pct, **kwargs):
#     pixels = pct * img.size[1]
#     _check_args_tf(kwargs)
#     return img_teacher.transform(img_teacher.size, Image.AFFINE, (1, 0, 0, 0, 1, pixels), **kwargs), img.transform(img.size, Image.AFFINE, (1, 0, 0, 0, 1, pixels), **kwargs)


# def translate_x_abs(img_teacher, img, pixels, **kwargs):
#     _check_args_tf(kwargs)
#     return img_teacher.transform(img_teacher.size, Image.AFFINE, (1, 0, pixels, 0, 1, 0), **kwargs), img.transform(img.size, Image.AFFINE, (1, 0, pixels, 0, 1, 0), **kwargs)


# def translate_y_abs(img_teacher, img, pixels, **kwargs):
#     _check_args_tf(kwargs)
#     return img_teacher.transform(img_teacher.size, Image.AFFINE, (1, 0, 0, 0, 1, pixels), **kwargs), img.transform(img.size, Image.AFFINE, (1, 0, 0, 0, 1, pixels), **kwargs)


# def rotate(img_teacher, img, degrees, **kwargs):
#     _check_args_tf(kwargs)
#     if _PIL_VER >= (5, 2):
#         return img.rotate(degrees, **kwargs)
#     elif _PIL_VER >= (5, 0):
#         w, h = img.size
#         post_trans = (0, 0)
#         rotn_center = (w / 2.0, h / 2.0)
#         angle = -math.radians(degrees)
#         matrix = [
#             round(math.cos(angle), 15),
#             round(math.sin(angle), 15),
#             0.0,
#             round(-math.sin(angle), 15),
#             round(math.cos(angle), 15),
#             0.0,
#         ]

#         def transform(x, y, matrix):
#             (a, b, c, d, e, f) = matrix
#             return a * x + b * y + c, d * x + e * y + f

#         matrix[2], matrix[5] = transform(
#             -rotn_center[0] - post_trans[0], -rotn_center[1] - post_trans[1], matrix
#         )
#         matrix[2] += rotn_center[0]
#         matrix[5] += rotn_center[1]
#         return img_teacher.transform(img_teacher.size, Image.AFFINE, matrix, **kwargs), img.transform(img.size, Image.AFFINE, matrix, **kwargs)
#     else:
#         return img_teacher.rotate(degrees, resample=kwargs['resample']), img.rotate(degrees, resample=kwargs['resample'])


# def auto_contrast(img_teacher, img, **__):
#     return img_teacher, ImageOps.autocontrast(img)


# def invert(img_teacher, img, **__):
#     return img_teacher, ImageOps.invert(img)


# def equalize(img_teacher, img, **__):
#     return img_teacher, ImageOps.equalize(img)


# def solarize(img_teacher, img, thresh, **__):
#     return img_teacher, ImageOps.solarize(img, thresh)


# def solarize_add(img_teacher, img, add, thresh=128, **__):
#     lut = []
#     for i in range(256):
#         if i < thresh:
#             lut.append(min(255, i + add))
#         else:
#             lut.append(i)
#     if img.mode in ("L", "RGB"):
#         if img.mode == "RGB" and len(lut) == 256:
#             lut = lut + lut + lut
#         return img_teacher, img.point(lut)
#     else:
#         return img_teacher, img


# def posterize(img_teacher, img, bits_to_keep, **__):
#     if bits_to_keep >= 8:
#         return img_teacher, img
#     return img_teacher, ImageOps.posterize(img, bits_to_keep)


# def contrast(img_teacher, img, factor, **__):
#     return img_teacher, ImageEnhance.Contrast(img).enhance(factor)


# def color(img_teacher, img, factor, **__):
#     return img_teacher, ImageEnhance.Color(img).enhance(factor)


# def brightness(img_teacher, img, factor, **__):
#     return img_teacher, ImageEnhance.Brightness(img).enhance(factor)


# def sharpness(img_teacher, img, factor, **__):
#     return img_teacher, ImageEnhance.Sharpness(img).enhance(factor)


def _randomly_negate(v):
    """With 50% prob, negate the value"""
    return -v if random.random() > 0.5 else v


def _rotate_level_to_arg(level, _hparams):
    # range [-30, 30]
    level = (level / _LEVEL_DENOM) * 30.
    level = _randomly_negate(level)
    return level,


def _enhance_level_to_arg(level, _hparams):
    # range [0.1, 1.9]
    return (level / _LEVEL_DENOM) * 1.8 + 0.1,


def _enhance_increasing_level_to_arg(level, _hparams):
    # the 'no change' level is 1.0, moving away from that towards 0. or 2.0 increases the enhancement blend
    # range [0.1, 1.9] if level <= _LEVEL_DENOM
    level = (level / _LEVEL_DENOM) * .9
    level = max(0.1, 1.0 + _randomly_negate(level))  # keep it >= 0.1
    return level,


def _shear_level_to_arg(level, _hparams):
    # range [-0.3, 0.3]
    level = (level / _LEVEL_DENOM) * 0.3
    level = _randomly_negate(level)
    return level,


def _translate_abs_level_to_arg(level, hparams):
    translate_const = hparams['translate_const']
    level = (level / _LEVEL_DENOM) * float(translate_const)
    level = _randomly_negate(level)
    return level,


def _translate_rel_level_to_arg(level, hparams):
    # default range [-0.45, 0.45]
    translate_pct = hparams.get('translate_pct', 0.45)
    level = (level / _LEVEL_DENOM) * translate_pct
    level = _randomly_negate(level)
    return level,


def _posterize_level_to_arg(level, _hparams):
    # As per Tensorflow TPU EfficientNet impl
    # range [0, 4], 'keep 0 up to 4 MSB of original image'
    # intensity/severity of augmentation decreases with level
    return int((level / _LEVEL_DENOM) * 4),


def _posterize_increasing_level_to_arg(level, hparams):
    # As per Tensorflow models research and UDA impl
    # range [4, 0], 'keep 4 down to 0 MSB of original image',
    # intensity/severity of augmentation increases with level
    return 4 - _posterize_level_to_arg(level, hparams)[0],


def _posterize_original_level_to_arg(level, _hparams):
    # As per original AutoAugment paper description
    # range [4, 8], 'keep 4 up to 8 MSB of image'
    # intensity/severity of augmentation decreases with level
    return int((level / _LEVEL_DENOM) * 4) + 4,


def _solarize_level_to_arg(level, _hparams):
    # range [0, 256]
    # intensity/severity of augmentation decreases with level
    return int((level / _LEVEL_DENOM) * 256),


def _solarize_increasing_level_to_arg(level, _hparams):
    # range [0, 256]
    # intensity/severity of augmentation increases with level
    return 256 - _solarize_level_to_arg(level, _hparams)[0],


def _solarize_add_level_to_arg(level, _hparams):
    # range [0, 110]
    return int((level / _LEVEL_DENOM) * 110),


LEVEL_TO_ARG = {
    'AutoContrast': None,
    'Equalize': None,
    'Invert': None,
    'Rotate': _rotate_level_to_arg,
    # There are several variations of the posterize level scaling in various Tensorflow/Google repositories/papers
    'Posterize': _posterize_level_to_arg,
    'PosterizeIncreasing': _posterize_increasing_level_to_arg,
    'PosterizeOriginal': _posterize_original_level_to_arg,
    'Solarize': _solarize_level_to_arg,
    'SolarizeIncreasing': _solarize_increasing_level_to_arg,
    'SolarizeAdd': _solarize_add_level_to_arg,
    'Color': _enhance_level_to_arg,
    'ColorIncreasing': _enhance_increasing_level_to_arg,
    'Contrast': _enhance_level_to_arg,
    'ContrastIncreasing': _enhance_increasing_level_to_arg,
    'Brightness': _enhance_level_to_arg,
    'BrightnessIncreasing': _enhance_increasing_level_to_arg,
    'Sharpness': _enhance_level_to_arg,
    'SharpnessIncreasing': _enhance_increasing_level_to_arg,
    'ShearX': _shear_level_to_arg,
    'ShearY': _shear_level_to_arg,
    'TranslateX': _translate_abs_level_to_arg,
    'TranslateY': _translate_abs_level_to_arg,
    'TranslateXRel': _translate_rel_level_to_arg,
    'TranslateYRel': _translate_rel_level_to_arg,
}


NAME_TO_OP = {
    'AutoContrast': auto_contrast,
    'Equalize': equalize,
    'Invert': invert,
    'Rotate': rotate,
    'Posterize': posterize,
    'PosterizeIncreasing': posterize,
    'PosterizeOriginal': posterize,
    'Solarize': solarize,
    'SolarizeIncreasing': solarize,
    'SolarizeAdd': solarize_add,
    'Color': color,
    'ColorIncreasing': color,
    'Contrast': contrast,
    'ContrastIncreasing': contrast,
    'Brightness': brightness,
    'BrightnessIncreasing': brightness,
    'Sharpness': sharpness,
    'SharpnessIncreasing': sharpness,
    'ShearX': shear_x,
    'ShearY': shear_y,
    'TranslateX': translate_x_abs,
    'TranslateY': translate_y_abs,
    'TranslateXRel': translate_x_rel,
    'TranslateYRel': translate_y_rel,
}

NAME_TO_OP_TEACHER = {
    'AutoContrast': None,
    'Equalize': None,
    'Invert': None,
    'Rotate': rotate,
    'Posterize': None,
    'PosterizeIncreasing': None,
    'PosterizeOriginal': None,
    'Solarize': None,
    'SolarizeIncreasing': None,
    'SolarizeAdd': None,
    'Color': None,
    'ColorIncreasing': None,
    'Contrast': None,
    'ContrastIncreasing': None,
    'Brightness': None,
    'BrightnessIncreasing': None,
    'Sharpness': None,
    'SharpnessIncreasing': None,
    'ShearX': shear_x,
    'ShearY': shear_y,
    'TranslateX': translate_x_abs,
    'TranslateY': translate_y_abs,
    'TranslateXRel': translate_x_rel,
    'TranslateYRel': translate_y_rel,
}


class AugmentOp:

    def __init__(self, name, prob=0.5, magnitude=10, hparams=None):
        hparams = hparams or _HPARAMS_DEFAULT
        self.name = name
        self.aug_fn = NAME_TO_OP[name]
        self.aug_fn_teacher = NAME_TO_OP_TEACHER[name]
        self.level_fn = LEVEL_TO_ARG[name]
        self.prob = prob
        self.magnitude = magnitude
        self.hparams = hparams.copy()
        self.kwargs = dict(
            fillcolor=hparams['img_mean'] if 'img_mean' in hparams else _FILL,
            resample=hparams['interpolation'] if 'interpolation' in hparams else _RANDOM_INTERPOLATION,
        )

        # If magnitude_std is > 0, we introduce some randomness
        # in the usually fixed policy and sample magnitude from a normal distribution
        # with mean `magnitude` and std-dev of `magnitude_std`.
        # NOTE This is my own hack, being tested, not in papers or reference impls.
        # If magnitude_std is inf, we sample magnitude from a uniform distribution
        self.magnitude_std = self.hparams.get('magnitude_std', 0)
        self.magnitude_max = self.hparams.get('magnitude_max', None)

    def __call__(self, img_teacher, img):
        if self.prob < 1.0 and random.random() > self.prob:
            return img_teacher, img
        magnitude = self.magnitude
        if self.magnitude_std > 0:
            # magnitude randomization enabled
            if self.magnitude_std == float('inf'):
                magnitude = random.uniform(0, magnitude)
            elif self.magnitude_std > 0:
                magnitude = random.gauss(magnitude, self.magnitude_std)
        # default upper_bound for the timm RA impl is _LEVEL_DENOM (10)
        # setting magnitude_max overrides this to allow M > 10 (behaviour closer to Google TF RA impl)
        upper_bound = self.magnitude_max or _LEVEL_DENOM
        magnitude = max(0., min(magnitude, upper_bound))
        level_args = self.level_fn(magnitude, self.hparams) if self.level_fn is not None else tuple()
        if self.aug_fn_teacher is not None:
            aug_img_teacher = self.aug_fn_teacher(img_teacher, *level_args, **self.kwargs)
        else:
            aug_img_teacher = img_teacher
        return aug_img_teacher, self.aug_fn(img, *level_args, **self.kwargs)

    def __repr__(self):
        fs = self.__class__.__name__ + f'(name={self.name}, p={self.prob}'
        fs += f', m={self.magnitude}, mstd={self.magnitude_std}'
        if self.magnitude_max is not None:
            fs += f', mmax={self.magnitude_max}'
        fs += ')'
        return fs





_RAND_TRANSFORMS = [
    'AutoContrast',
    'Equalize',
    'Invert',
    'Rotate',
    'Posterize',
    'Solarize',
    'SolarizeAdd',
    'Color',
    'Contrast',
    'Brightness',
    'Sharpness',
    'ShearX',
    'ShearY',
    'TranslateXRel',
    'TranslateYRel',
    #'Cutout'  # NOTE I've implement this as random erasing separately
]


_RAND_INCREASING_TRANSFORMS = [
    'AutoContrast',
    'Equalize',
    'Invert',
    'Rotate',
    'PosterizeIncreasing',
    'SolarizeIncreasing',
    'SolarizeAdd',
    'ColorIncreasing',
    'ContrastIncreasing',
    'BrightnessIncreasing',
    'SharpnessIncreasing',
    'ShearX',
    'ShearY',
    'TranslateXRel',
    'TranslateYRel',
    #'Cutout'  # NOTE I've implement this as random erasing separately
]



# These experimental weights are based loosely on the relative improvements mentioned in paper.
# They may not result in increased performance, but could likely be tuned to so.
_RAND_CHOICE_WEIGHTS_0 = {
    'Rotate': 0.3,
    'ShearX': 0.2,
    'ShearY': 0.2,
    'TranslateXRel': 0.1,
    'TranslateYRel': 0.1,
    'Color': .025,
    'Sharpness': 0.025,
    'AutoContrast': 0.025,
    'Solarize': .005,
    'SolarizeAdd': .005,
    'Contrast': .005,
    'Brightness': .005,
    'Equalize': .005,
    'Posterize': 0,
    'Invert': 0,
}


def _select_rand_weights(weight_idx=0, transforms=None):
    transforms = transforms or _RAND_TRANSFORMS
    assert weight_idx == 0  # only one set of weights currently
    rand_weights = _RAND_CHOICE_WEIGHTS_0
    probs = [rand_weights[k] for k in transforms]
    probs /= np.sum(probs)
    return probs


def rand_augment_ops(magnitude=10, hparams=None, transforms=None):
    hparams = hparams or _HPARAMS_DEFAULT
    transforms = transforms or _RAND_TRANSFORMS
    return [AugmentOp(
        name, prob=0.5, magnitude=magnitude, hparams=hparams) for name in transforms]


class RandAugment:
    def __init__(self, ops, num_layers=2, choice_weights=None):
        self.ops = ops
        self.num_layers = num_layers
        self.choice_weights = choice_weights

    def __call__(self, img_teacher, img):
        # no replacement when using weighted choice
        ops = np.random.choice(
            self.ops, self.num_layers, replace=self.choice_weights is None, p=self.choice_weights)
        for op in ops:
            img_teacher, img = op(img_teacher, img)
        return img_teacher, img

    def __repr__(self):
        fs = self.__class__.__name__ + f'(n={self.num_layers}, ops='
        for op in self.ops:
            fs += f'\n\t{op}'
        fs += ')'
        return fs


def rand_augment_transform(config_str, hparams):
    """
    Create a RandAugment transform
    :param config_str: String defining configuration of random augmentation. Consists of multiple sections separated by
    dashes ('-'). The first section defines the specific variant of rand augment (currently only 'rand'). The remaining
    sections, not order sepecific determine
        'm' - integer magnitude of rand augment
        'n' - integer num layers (number of transform ops selected per image)
        'w' - integer probabiliy weight index (index of a set of weights to influence choice of op)
        'mstd' -  float std deviation of magnitude noise applied, or uniform sampling if infinity (or > 100)
        'mmax' - set upper bound for magnitude to something other than default of  _LEVEL_DENOM (10)
        'inc' - integer (bool), use augmentations that increase in severity with magnitude (default: 0)
    Ex 'rand-m9-n3-mstd0.5' results in RandAugment with magnitude 9, num_layers 3, magnitude_std 0.5
    'rand-mstd1-w0' results in magnitude_std 1.0, weights 0, default magnitude of 10 and num_layers 2
    :param hparams: Other hparams (kwargs) for the RandAugmentation scheme
    :return: A PyTorch compatible Transform
    """
    magnitude = _LEVEL_DENOM  # default to _LEVEL_DENOM for magnitude (currently 10)
    num_layers = 2  # default to 2 ops per image
    weight_idx = None  # default to no probability weights for op choice
    transforms = _RAND_TRANSFORMS
    config = config_str.split('-')
    assert config[0] == 'rand'
    config = config[1:]
    for c in config:
        cs = re.split(r'(\d.*)', c)
        if len(cs) < 2:
            continue
        key, val = cs[:2]
        if key == 'mstd':
            # noise param / randomization of magnitude values
            mstd = float(val)
            if mstd > 100:
                # use uniform sampling in 0 to magnitude if mstd is > 100
                mstd = float('inf')
            hparams.setdefault('magnitude_std', mstd)
        elif key == 'mmax':
            # clip magnitude between [0, mmax] instead of default [0, _LEVEL_DENOM]
            hparams.setdefault('magnitude_max', int(val))
        elif key == 'inc':
            if bool(val):
                transforms = _RAND_INCREASING_TRANSFORMS
        elif key == 'm':
            magnitude = int(val)
        elif key == 'n':
            num_layers = int(val)
        elif key == 'w':
            weight_idx = int(val)
        else:
            assert False, 'Unknown RandAugment config section'
    ra_ops = rand_augment_ops(magnitude=magnitude, hparams=hparams, transforms=transforms)
    choice_weights = None if weight_idx is None else _select_rand_weights(weight_idx)
    return RandAugment(ra_ops, num_layers, choice_weights=choice_weights)


