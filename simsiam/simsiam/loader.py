# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from PIL import ImageFilter, Image
import random
import torchvision.transforms.functional as F
import numpy as np


class TwoCropsTransform:
    """Take two random crops of one image as the query and key."""

    def __init__(self, base_transform):
        self.base_transform = base_transform

    def __call__(self, x):
        q = self.base_transform(x)
        k = self.base_transform(x)
        return [q, k]


class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x
    

class SquarePad:
    def __call__(self, image):
        w, h = image.size
        max_wh = np.max([w, h])
        hp = int((max_wh - w) / 2)
        vp = int((max_wh - h) / 2)
        padding = (hp, vp, hp, vp)
        return F.pad(image, padding, 0, 'constant')

class CutOut:
    """Random image cutout based on https://github.com/hysts/pytorch_cutout"""

    def __init__(self, mask_size_range=(0.1, 0.3), cutout_inside=True, mask_color=0):
        self.mask_size_range =  mask_size_range
        self.cutout_inside = cutout_inside
        self.mask_color = mask_color
    
    def __call__(self, image):

        h, w = image.size
        # make sure that the image can be cut when range = 1
        min_wh = np.min([w, h])

        mask_size = int(random.uniform(self.mask_size_range[0], self.mask_size_range[1]) * min_wh)
         
        mask_size_half = mask_size // 2
        offset = 1 if mask_size % 2 == 0 else 0
        
        
        if self.cutout_inside:
            cxmin, cxmax = mask_size_half, w + offset - mask_size_half
            cymin, cymax = mask_size_half, h + offset - mask_size_half
        else:
            cxmin, cxmax = 0, w + offset
            cymin, cymax = 0, h + offset

        cx = np.random.randint(cxmin, cxmax)
        cy = np.random.randint(cymin, cymax)
        xmin = cx - mask_size_half
        ymin = cy - mask_size_half
        xmax = xmin + mask_size
        ymax = ymin + mask_size
        xmin = max(0, xmin)
        ymin = max(0, ymin)
        xmax = min(w, xmax)
        ymax = min(h, ymax)
        np_image = np.asarray(image).copy()
        np_image[ymin:ymax, xmin:xmax] = self.mask_color
        image = Image.fromarray(np.uint8(np_image))
        return image
