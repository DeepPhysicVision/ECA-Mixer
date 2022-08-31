import torch
from complexLayers import *
import numpy as np
from torch import nn
from einops.layers.torch import Rearrange

def propagator(Nx,Ny,z,wavelength,deltaX,deltaY):
    k = 1/wavelength
    x = np.expand_dims(np.arange(np.ceil(-Nx/2),np.ceil(Nx/2),1)*(1/(Nx*deltaX)),axis=0)
    y = np.expand_dims(np.arange(np.ceil(-Ny/2),np.ceil(Ny/2),1)*(1/(Ny*deltaY)),axis=1)
    y_new = np.repeat(y,Nx,axis=1)
    x_new = np.repeat(x,Ny,axis=0)
    kp = np.sqrt(y_new**2+x_new**2)
    term=k**2-kp**2
    term=np.maximum(term,0)
    phase = np.exp(1j*2*np.pi*z*np.sqrt(term))
    phase = phase[np.newaxis,np.newaxis,:,:]
    return phase

def dwt_init(x):
    x01 = x[:, :, 0::2, :] / 2
    x02 = x[:, :, 1::2, :] / 2
    x1 = x01[:, :, :, 0::2]
    x2 = x02[:, :, :, 0::2]
    x3 = x01[:, :, :, 1::2]
    x4 = x02[:, :, :, 1::2]
    x_LL = x1 + x2 + x3 + x4
    x_HL = -x1 - x2 + x3 + x4
    x_LH = -x1 + x2 - x3 + x4
    x_HH = x1 - x2 - x3 + x4
    return torch.cat((x_LL, x_HL, x_LH, x_HH), 1)

def iwt_init(x):
    r = 2
    in_batch, in_channel, in_height, in_width = x.size()
    out_batch, out_channel, out_height, out_width = in_batch, int(in_channel / (r ** 2)), r * in_height, r * in_width
    x1 = x[:, 0:out_channel, :, :] / 2
    x2 = x[:, out_channel:out_channel * 2, :, :] / 2
    x3 = x[:, out_channel * 2:out_channel * 3, :, :] / 2
    x4 = x[:, out_channel * 3:out_channel * 4, :, :] / 2

    h = torch.zeros([out_batch, out_channel, out_height, out_width])  #.float().cuda()
    h = torch.complex(h,h).cuda()

    h[:, :, 0::2, 0::2] = x1 - x2 - x3 + x4
    h[:, :, 1::2, 0::2] = x1 - x2 + x3 - x4
    h[:, :, 0::2, 1::2] = x1 + x2 - x3 - x4
    h[:, :, 1::2, 1::2] = x1 + x2 + x3 + x4
    return h

class AttentionNet(nn.Module):
    def __init__(self, k=3, p=1, ch=18):
        super(AttentionNet, self).__init__()
        self.act = ComplexReLU()

        self.init = nn.Sequential(
            ComplexConv2d(3, ch, 1, stride=1, padding=0),
            self.act,
            ComplexBatchNorm2d(ch),
            ComplexConv2d(ch, ch, k, stride=1, padding=p),
            self.act,
            ComplexBatchNorm2d(ch))

        self.nonlinear = nn.Sequential(
            ComplexConv2d(ch*64, ch, k, stride=1, padding=p),
            ComplexTanh(),
            ComplexConv2d(ch, ch*64, k, stride=1, padding=p),
            self.act)

        self.out = nn.Sequential(
            ComplexConv2d(ch, ch, k, stride=1, padding=p),
            self.act,
            ComplexConv2d(ch, ch, 1, stride=1, padding=0),
            self.act,
            ComplexConv2d(ch, 3,  1, stride=1, padding=0))

        self.attention1 = nn.Sequential(
            ComplexAvgPool2d(),
            Rearrange('b c d n -> b (n d) c'),
            ComplexConv1d(1, 1, kernel_size=7, padding=3),
            ComplexSigmoid(),
            Rearrange('b (n d) c -> b c d n', n=1))

        self.attention2 = nn.Sequential(
            Rearrange('b c d n -> b (n d) c'),
            ComplexAvgPool1d(),
            Rearrange('b nd c -> b c nd'),
            ComplexConv1d(1, 1, kernel_size=7, padding=3),
            ComplexSigmoid())

    def forward(self, x):
        d0  = self.init(x)

        d1 = dwt_init(d0)
        x1 = self.attention1(d1)
        d1 = d1 * x1.expand_as(d1)
        x2 = self.attention2(d1)
        x2 = Rearrange('b c (n d)  -> b c n d', d=d1.shape[-1])(x2)
        d1 = d1 * x2.expand_as(d1)

        d2 = dwt_init(d1)
        x1 = self.attention1(d2)
        d2 = d2 * x1.expand_as(d2)
        x2 = self.attention2(d2)
        x2 = Rearrange('b c (n d)  -> b c n d', d=d2.shape[-1])(x2)
        d2 = d2 * x2.expand_as(d2)

        d3 = dwt_init(d2)
        x1 = self.attention1(d3)
        d3 = d3 * x1.expand_as(d3)
        x2 = self.attention2(d3)
        x2 = Rearrange('b c (n d)  -> b c n d', d=d3.shape[-1])(x2)
        d3 = d3 * x2.expand_as(d3)

        d3 = self.nonlinear(d3)

        u3 = iwt_init(d3)
        x1 = self.attention1(u3)
        u3 = u3 * x1.expand_as(u3)
        x2 = self.attention2(u3)
        x2 = Rearrange('b c (n d)  -> b c n d', d=u3.shape[-1])(x2)
        u3 = u3 * x2.expand_as(u3)

        u2 = iwt_init(u3)
        x1 = self.attention1(u2)
        u2 = u2 * x1.expand_as(u2)
        x2 = self.attention2(u2)
        x2 = Rearrange('b c (n d)  -> b c n d', d=u2.shape[-1])(x2)
        u2 = u2 * x2.expand_as(u2)

        u1 = iwt_init(u2)
        out = self.out(u1)
        return out

def iwt_initR(x):
    r = 2
    in_batch, in_channel, in_height, in_width = x.size()
    out_batch, out_channel, out_height, out_width = in_batch, int(in_channel / (r ** 2)), r * in_height, r * in_width
    x1 = x[:, 0:out_channel, :, :] / 2
    x2 = x[:, out_channel:out_channel * 2, :, :] / 2
    x3 = x[:, out_channel * 2:out_channel * 3, :, :] / 2
    x4 = x[:, out_channel * 3:out_channel * 4, :, :] / 2

    h = torch.zeros([out_batch, out_channel, out_height, out_width]).float().cuda()

    h[:, :, 0::2, 0::2] = x1 - x2 - x3 + x4
    h[:, :, 1::2, 0::2] = x1 - x2 + x3 - x4
    h[:, :, 0::2, 1::2] = x1 + x2 - x3 - x4
    h[:, :, 1::2, 1::2] = x1 + x2 + x3 + x4
    return h