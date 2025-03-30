import numpy as np
import torch.nn as nn
import torch
from torch.nn import Conv2d, BatchNorm2d, ReLU

# Replacement for mmcv.cnn.ConvModule
class ConvModule(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 bias=True,
                 norm_cfg=None,
                 act_cfg=None):
        super().__init__()
        
        # Create convolution layer
        self.conv = Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias)
        
        # Add normalization if specified
        self.norm = None
        if norm_cfg is not None:
            self.norm = BatchNorm2d(out_channels)
        
        # Add activation if specified
        self.activate = None
        if act_cfg is not None:
            self.activate = ReLU(inplace=True)
    
    def forward(self, x):
        x = self.conv(x)
        if self.norm is not None:
            x = self.norm(x)
        if self.activate is not None:
            x = self.activate(x)
        return x

# Replacement for mmseg.ops.resize
def resize(input,
           size=None,
           scale_factor=None,
           mode='bilinear',
           align_corners=None):
    return torch.nn.functional.interpolate(
        input,
        size=size,
        scale_factor=scale_factor,
        mode=mode,
        align_corners=align_corners)

# Simplified registry system to replace mmseg.models.builder.HEADS
class Registry:
    _registry = {}
    
    @classmethod
    def register_module(cls, name=None):
        def decorator(module_class):
            module_name = name if name is not None else module_class.__name__
            cls._registry[module_name] = module_class
            return module_class
        return decorator
    
    @classmethod
    def get(cls, name):
        return cls._registry.get(name)

HEADS = Registry()

# Base class replacement
class BaseDecodeHead(nn.Module):
    def __init__(self, input_transform=None, **kwargs):
        super().__init__()
        # Add any common initialization here
        self.input_transform = input_transform

    def forward(self, inputs):
        raise NotImplementedError

@HEADS.register_module()
class IdentityHead(BaseDecodeHead):
    """Panoptic Feature Pyramid Networks.
    This head is the implementation of `Semantic FPN
    <https://arxiv.org/abs/1901.02446>`_.
    Args:
        feature_strides (tuple[int]): The strides for input feature maps.
            stack_lateral. All strides suppose to be power of 2. The first
            one is of largest resolution.
    """

    def __init__(self, **kwargs):
        super().__init__(input_transform=None, **kwargs)
        self.conv_seg = None

    def forward(self, inputs):
        return inputs