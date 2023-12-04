#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
https://github.com/caiyunapp/leibniz/blob/master/leibniz/nn/activation.py
Created on Tue Mar 19 10:30:02 2019
@author: Sebastien M. Popoff
Based on https://openreview.net/forum?id=H1T2hmZAb
"""

import torch
from torch.nn import *
from complexFunctions import *
import torch.nn.functional as F
import torch.nn as nn

def apply_complex(fr, fi, input, dtype = torch.complex64):
    return (fr(input.real)-fi(input.imag)).type(dtype) \
            + 1j*(fr(input.imag)+fi(input.real)).type(dtype)

class ComplexDropout1d(nn.Module):
    def __init__(self, p=0.5, inplace=True):
        super(ComplexDropout1d, self).__init__()
        self.p = p
        self.inplace = inplace

    def forward(self, input):
        return F.dropout(input.real, p=self.p, training=self.training, inplace=self.inplace) +\
               1j * F.dropout(input.imag, p=self.p, training=self.training, inplace=self.inplace)

class ComplexDropout2d(nn.Module):
    def __init__(self, p=0.5, inplace=True):
        super(ComplexDropout2d, self).__init__()
        self.p = p
        self.inplace = inplace

    def forward(self, input):
        return F.dropout2d(input.real, p=self.p, training=self.training, inplace=self.inplace) +\
               1j * F.dropout2d(input.imag, p=self.p, training=self.training, inplace=self.inplace)

class ComplexMaxPool2d(Module):
    def __init__(self,kernel_size,stride= None,padding = 0,dilation = 1,return_indices = False,ceil_mode = False):
        super(ComplexMaxPool2d,self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.ceil_mode = ceil_mode
        self.return_indices = return_indices

    def forward(self,input):
        return complex_max_pool2d(input,kernel_size = self.kernel_size,
                                stride = self.stride, padding = self.padding,
                                dilation = self.dilation, ceil_mode = self.ceil_mode,
                                return_indices = self.return_indices)

class ComplexAvgPool1d(nn.Module):
    def __init__(self):
        super(ComplexAvgPool1d, self).__init__()
        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, input):
        return self.pool(input.real) + 1j * self.pool(input.imag)

class ComplexAvgPool2d(nn.Module):
    def __init__(self):
        super(ComplexAvgPool2d, self).__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, input):
        return self.pool(input.real) + 1j * self.pool(input.imag)

class ComplexMatmul(Module):
    def forward(self, A, B):
        return complex_matmul(A, B)

class ComplexReLU(Module):
    def forward(self, input):
        return complex_relu(input)
class ComplexGeLU(Module):
    def forward(self, input):
        return complex_gelu(input)
class ComplexTanh(Module):
    def forward(self, input):
        return complex_tanh(input)
class ComplexSigmoid(Module):
    def forward(self, input):
        return complex_sigmoid(input)

class ComplexConvTranspose2d(Module):
    def __init__(self,in_channels, out_channels, kernel_size, stride=1, padding=0,
                 output_padding=0, groups=1, bias=True, dilation=1, padding_mode='zeros'):
        super(ComplexConvTranspose2d, self).__init__()

        self.conv_tran_r = ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding,
                                       output_padding, groups, bias, dilation, padding_mode)
        self.conv_tran_i = ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding,
                                       output_padding, groups, bias, dilation, padding_mode)

    def forward(self,input):
        return apply_complex(self.conv_tran_r, self.conv_tran_i, input)

class ComplexConv1d(nn.Module):
    def __init__(self,in_channels, out_channels, kernel_size=3, stride=1, padding=0, dilation=1, groups=1, bias=True):
        super(ComplexConv1d, self).__init__()
        self.rconv = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.iconv = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)

    def forward(self, input):
        return self.rconv(input.real) - self.iconv(input.imag) + 1j * (self.rconv(input.imag) + self.iconv(input.real))

class ComplexConv2d(nn.Module):
    def __init__(self,in_channels, out_channels, kernel_size=3, stride=1, padding=0, dilation=1, groups=1, bias=True):
        super(ComplexConv2d, self).__init__()
        self.rconv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.iconv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)

    def forward(self, input):
        return self.rconv(input.real) - self.iconv(input.imag) + 1j * (self.rconv(input.imag) + self.iconv(input.real))

class ComplexLinear(nn.Module):
    def __init__(self, in_channels, out_channels, bias=True):
        super(ComplexLinear, self).__init__()
        self.rfc = nn.Linear(in_channels, out_channels, bias)
        self.ifc = nn.Linear(in_channels, out_channels, bias)

    def forward(self, input):
        return self.rfc(input.real) - self.ifc(input.imag) + 1j * (self.rfc(input.imag) + self.ifc(input.real))

class ComplexBatchNorm1d(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True):
        super(ComplexBatchNorm1d, self).__init__()
        self.rbn = nn.BatchNorm1d(num_features, eps, momentum, affine, track_running_stats)
        self.ibn = nn.BatchNorm1d(num_features, eps, momentum, affine, track_running_stats)

    def forward(self, input):
        return self.rbn(input.real) + 1j * self.ibn(input.imag)

class ComplexBatchNorm2d(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True):
        super(ComplexBatchNorm2d, self).__init__()
        self.rbn = nn.BatchNorm2d(num_features, eps, momentum, affine, track_running_stats)
        self.ibn = nn.BatchNorm2d(num_features, eps, momentum, affine, track_running_stats)

    def forward(self, input):
        return self.rbn(input.real) + 1j * self.ibn(input.imag)