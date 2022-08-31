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

################################################### Net1 Attention
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

        #### change the kernel and pad
        self.attention1 = nn.Sequential(
            ComplexAvgPool2d(), #1 c 1 1
            Rearrange('b c d n -> b (n d) c'),  #1 1 c
            ComplexConv1d(1, 1, kernel_size=7, padding=3),   #1 1 c
            ComplexSigmoid(),
            Rearrange('b (n d) c -> b c d n', n=1))  #1 c 1 1

        self.attention2 = nn.Sequential(
            # the 2 rearrange don't change the sequence of channel and token, just because the Pool function can't direct calculate the c channel to 1 
            Rearrange('b c d n -> b (n d) c'),
            ComplexAvgPool1d(), #1 dn 1
            Rearrange('b nd c -> b c nd'),  #1 1 dn
            ComplexConv1d(1, 1, kernel_size=7, padding=3),
            ComplexSigmoid())

    def forward(self, x):
        d0  = self.init(x)  # init feature:16 c*h*w

        # down sample
        d1 = dwt_init(d0)   # 4c * h/2 * w/2
        x1 = self.attention1(d1)
        d1 = d1 * x1.expand_as(d1)
        x2 = self.attention2(d1)
        x2 = Rearrange('b c (n d)  -> b c n d', d=d1.shape[-1])(x2)
        d1 = d1 * x2.expand_as(d1)

        d2 = dwt_init(d1)   # 16c * h/4 * w/4
        x1 = self.attention1(d2)
        d2 = d2 * x1.expand_as(d2)
        x2 = self.attention2(d2)
        x2 = Rearrange('b c (n d)  -> b c n d', d=d2.shape[-1])(x2)
        d2 = d2 * x2.expand_as(d2)

        # middle transform
        d3 = dwt_init(d2)   # 64c * h/8 * w/8
        x1 = self.attention1(d3)
        d3 = d3 * x1.expand_as(d3)
        x2 = self.attention2(d3)
        x2 = Rearrange('b c (n d)  -> b c n d', d=d3.shape[-1])(x2)
        d3 = d3 * x2.expand_as(d3)

        d3 = self.nonlinear(d3) # nolinear layer

        # up sample
        u3 = iwt_init(d3)   # 16c * h/4 * w/4
        x1 = self.attention1(u3)
        u3 = u3 * x1.expand_as(u3)
        x2 = self.attention2(u3)
        x2 = Rearrange('b c (n d)  -> b c n d', d=u3.shape[-1])(x2)
        u3 = u3 * x2.expand_as(u3)

        u2 = iwt_init(u3)   # 4c * h/2 * w/2
        x1 = self.attention1(u2)
        u2 = u2 * x1.expand_as(u2)
        x2 = self.attention2(u2)
        x2 = Rearrange('b c (n d)  -> b c n d', d=u2.shape[-1])(x2)
        u2 = u2 * x2.expand_as(u2)

        u1 = iwt_init(u2)   # c*h*w

        # super resolution
        #x1 = self.attention1(u1)
        #u1 = u1 * x1.expand_as(u1)
        #x2 = self.attention2(u1)
        #x2 = Rearrange('b c (n d)  -> b c n d', d=u1.shape[-1])(x2)
        #u1 = u1 * x2.expand_as(u1)

        #u0 = iwt_init(u1)

        # output
        out = self.out(u1)  # 3*h*w
        return out

############################################ Net2 Conv
class ConvNet(nn.Module):
    def __init__(self, k=3, p=1):
        super(ConvNet, self).__init__()
        self.act = ComplexReLU()
        self.conv_init = nn.Sequential( 
            ComplexConv2d(3, 16,  1, stride=1, padding=0),
            self.act,
            ComplexBatchNorm2d(16),
            ComplexConv2d(16, 16, k, stride=1, padding=p),
            self.act,
            ComplexBatchNorm2d(16)
        )
        
        self.conv_1 = nn.Sequential(   
            ComplexConv2d(64, 64, k, stride=1, padding=p),
            self.act,
            ComplexBatchNorm2d(64),
            ComplexConv2d(64, 64, k, stride=1, padding=p),
            self.act,
            ComplexBatchNorm2d(64)
        )
        self.conv_2 = nn.Sequential(   
            ComplexConv2d(256, 256, k, stride=1, padding=p),
            self.act,
            ComplexBatchNorm2d(256),
            ComplexConv2d(256, 256, k, stride=1, padding=p),
            self.act,
            ComplexBatchNorm2d(256)
        )
        self.conv_3 = nn.Sequential(   
            ComplexConv2d(1024, 1024, k, stride=1, padding=p),
            self.act,
            ComplexBatchNorm2d(1024),
            ComplexConv2d(1024, 1024, k, stride=1, padding=p),
            self.act,
            ComplexBatchNorm2d(1024)
        )        
        
        self.conv_nonlinear = nn.Sequential(   
            ComplexConv2d(1024, 1024, k, stride=1, padding=p),
            self.act,
            ComplexBatchNorm2d(1024),
            ComplexConv2d(1024, 16, k, stride=1, padding=p),
            ComplexTanh()
        )
        self.deconv_1 = nn.Sequential(
            ComplexConv2d(16, 1024, k, stride=1, padding=p),
            self.act,
            ComplexBatchNorm2d(1024),
            ComplexConv2d(1024, 1024, k, stride=1, padding=p),
            self.act,
            ComplexBatchNorm2d(1024)
        )
        
        self.deconv_2 = nn.Sequential(
            ComplexConv2d(256, 256, k, stride=1, padding=p),
            self.act,
            ComplexBatchNorm2d(256),
            ComplexConv2d(256, 256, k, stride=1, padding=p),
            self.act,
            ComplexBatchNorm2d(256)
        )
        self.deconv_3 = nn.Sequential(
            ComplexConv2d(64, 64, k, stride=1, padding=p),
            self.act,
            ComplexBatchNorm2d(64),
            ComplexConv2d(64, 64, k, stride=1, padding=p),
            self.act,
            ComplexBatchNorm2d(64)
        )
        self.deconv_4 = nn.Sequential(
            ComplexConv2d(16, 16, k, stride=1, padding=p),
            self.act,
            ComplexConv2d(16, 16, 1, stride=1, padding=0),
            self.act,
            ComplexConv2d(16, 3, 1, stride=1, padding=0)
        )
        
    def forward(self, x):
        d1 = self.conv_init(x)
        
        d2 = dwt_init(d1)
        d2 = self.conv_1(d2)
        d3 = dwt_init(d2)
        d3 = self.conv_2(d3)
        d4 = dwt_init(d3)
        
        d4 = self.conv_3(d4)
        d5 = self.conv_nonlinear(d4)
        u4 = self.deconv_1(d5)
        
        u3 = iwt_init(u4)
        u3 = self.deconv_2(u3)
        u2 = iwt_init(u3)
        u2 = self.deconv_3(u2)
        u1 = iwt_init(u2)
        u1 = self.deconv_4(u1)
        
        #print(d1.shape, d2.shape, d3.shape, d4.shape, d5.shape)
        #print(u4.shape, u3.shape, u2.shape, u1.shape)
        return u1

################################################### Net3 EcaNet
class EcaNet(nn.Module):
    def __init__(self, k=3, p=1, ch=16):
        super(EcaNet, self).__init__()
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

        #### change the kernel and pad
        self.attention1 = nn.Sequential(
            ComplexAvgPool2d(), #1 c 1 1
            Rearrange('b c d n -> b (n d) c'),  #1 1 c
            ComplexConv1d(1, 1, kernel_size=7, padding=3),   #1 1 c
            ComplexSigmoid(),
            Rearrange('b (n d) c -> b c d n', n=1))  #1 c 1 1

        self.attention2 = nn.Sequential(
            # the 2 rearrange don't change the sequence of channel and token, just because the Pool function can't direct calculate the c channel to 1
            Rearrange('b c d n -> b (n d) c'),
            ComplexAvgPool1d(), #1 dn 1
            Rearrange('b nd c -> b c nd'),  #1 1 dn
            ComplexConv1d(1, 1, kernel_size=7, padding=3),
            ComplexSigmoid())

    def forward(self, x):
        d0  = self.init(x)  # init feature:16 c*h*w

        # down sample
        d1 = dwt_init(d0)   # 4c * h/2 * w/2
        x1 = self.attention1(d1)
        d1 = d1 * x1.expand_as(d1)
        #x2 = self.attention2(d1)
        #x2 = Rearrange('b c (n d)  -> b c n d', d=d1.shape[-1])(x2)
        #d1 = d1 * x2.expand_as(d1)

        d2 = dwt_init(d1)   # 16c * h/4 * w/4
        x1 = self.attention1(d2)
        d2 = d2 * x1.expand_as(d2)
        #x2 = self.attention2(d2)
        #x2 = Rearrange('b c (n d)  -> b c n d', d=d2.shape[-1])(x2)
        #d2 = d2 * x2.expand_as(d2)

        # middle transform
        d3 = dwt_init(d2)   # 64c * h/8 * w/8
        x1 = self.attention1(d3)
        d3 = d3 * x1.expand_as(d3)
        #x2 = self.attention2(d3)
        #x2 = Rearrange('b c (n d)  -> b c n d', d=d3.shape[-1])(x2)
        #d3 = d3 * x2.expand_as(d3)

        d3 = self.nonlinear(d3) # nolinear layer

        # up sample
        u3 = iwt_init(d3)   # 16c * h/4 * w/4
        x1 = self.attention1(u3)
        u3 = u3 * x1.expand_as(u3)
        #x2 = self.attention2(u3)
        #x2 = Rearrange('b c (n d)  -> b c n d', d=u3.shape[-1])(x2)
        #u3 = u3 * x2.expand_as(u3)

        u2 = iwt_init(u3)   # 4c * h/2 * w/2
        x1 = self.attention1(u2)
        u2 = u2 * x1.expand_as(u2)
        #x2 = self.attention2(u2)
        #x2 = Rearrange('b c (n d)  -> b c n d', d=u2.shape[-1])(x2)
        #u2 = u2 * x2.expand_as(u2)

        u1 = iwt_init(u2)   # c*h*w
        #x1 = self.attention1(u1)
        #u1 = u1 * x1.expand_as(u1)
        #x2 = self.attention2(u1)
        #x2 = Rearrange('b c (n d)  -> b c n d', n=u1.shape[-1])(x2)
        #u1 = u1 * x2.expand_as(u1)

        # super resolution
        #u0 = iwt_init(u1)

        # output
        out = self.out(u1)  # 3*h*w
        return out

################################################### Net1 Attention for drawing model
class AttentionNetDraw(nn.Module):
    def __init__(self, k=3, p=1, ch=16):
        super(AttentionNetDraw, self).__init__()
        self.act = ComplexReLU()

        self.init = nn.Sequential(
            ComplexConv2d(3, ch, 1, stride=1, padding=0),
            self.act,
            ComplexBatchNorm2d(ch),
            ComplexConv2d(ch, ch, k, stride=1, padding=p),
            self.act,
            ComplexBatchNorm2d(ch))

        self.nonlinear = nn.Sequential(
            ComplexConv2d(ch*16, ch, k, stride=1, padding=p),
            ComplexTanh(),
            ComplexConv2d(ch, ch*16, k, stride=1, padding=p),
            self.act)

        self.out = nn.Sequential(
            ComplexConv2d(ch, ch, k, stride=1, padding=p),
            self.act,
            ComplexConv2d(ch, ch, 1, stride=1, padding=0),
            self.act,
            ComplexConv2d(ch, 3,  1, stride=1, padding=0))

        #### change the kernel and pad
        self.attention1 = nn.Sequential(
            ComplexAvgPool2d(), #1 c 1 1
            Rearrange('b c d n -> b (n d) c'),  #1 1 c
            ComplexConv1d(1, 1, kernel_size=7, padding=3),   #1 1 c
            ComplexSigmoid(),
            Rearrange('b (n d) c -> b c d n', n=1))  #1 c 1 1

        self.attention2 = nn.Sequential(
            Rearrange('b c d n -> b (n d) c'),
            ComplexAvgPool1d(), #1 dn 1
            Rearrange('b nd c -> b c nd'),  #1 1 dn
            ComplexConv1d(1, 1, kernel_size=7, padding=3),
            ComplexSigmoid())

    def forward(self, x):
        d0  = self.init(x)  # init feature:16 c*h*w

        # down sample
        d1 = dwt_init(d0)   # 4c * h/2 * w/2
        x1 = self.attention1(d1)
        d1 = d1 * x1.expand_as(d1)
        x2 = self.attention2(d1)
        x2 = Rearrange('b c (n d)  -> b c n d', d=d1.shape[-1])(x2)
        d1 = d1 * x2.expand_as(d1)

        d2 = dwt_init(d1)   # 16c * h/4 * w/4
        x1 = self.attention1(d2)
        d2 = d2 * x1.expand_as(d2)
        x2 = self.attention2(d2)
        x2 = Rearrange('b c (n d)  -> b c n d', d=d2.shape[-1])(x2)
        d2 = d2 * x2.expand_as(d2)

        # middle transform
        d3 = self.nonlinear(d2)

        # up sample
        u3 = iwt_init(d3)   # 4c * h/2 * w/2
        x1 = self.attention1(u3)
        u3 = u3 * x1.expand_as(u3)
        x2 = self.attention2(u3)
        x2 = Rearrange('b c (n d)  -> b c n d', d=u3.shape[-1])(x2)
        u3 = u3 * x2.expand_as(u3)

        u2 = iwt_init(u3)   # c * h * w
        x1 = self.attention1(u2)
        u2 = u2 * x1.expand_as(u2)
        x2 = self.attention2(u2)
        x2 = Rearrange('b c (n d)  -> b c n d', d=u2.shape[-1])(x2)
        u2 = u2 * x2.expand_as(u2)

        # output
        out = self.out(u2)  # 3*h*w
        return out

## Autofocusing
class ZNet(nn.Module):
    def __init__(self):
        super(ZNet, self).__init__()
        self.linear1 = nn.Linear(40, 20, bias=True)
        self.acn = nn.ReLU()
        self.linear2 = nn.Linear(20, 1,  bias=True)

    def forward(self, x):
        y = self.linear1(x)
        y = self.acn(y)
        y = self.linear2(y)
        return y

## phase unwrap
def unwrap(x):
    y = x % (2 * np.pi)
    return torch.where(y > np.pi, 2 * np.pi - y, y)
def fft2dc(x):
    return np.fft.fftshift(np.fft.fft2(x))
def ifft2dc(x):
    return np.fft.ifft2(np.fft.fftshift(x))
def Phase_unwrapping(in_,Ny,Nx):
    f = np.zeros((Nx, Ny))
    for ii in range(Nx):
        for jj in range(Ny):
            x = ii - Nx / 2
            y = jj - Ny / 2
            f[ii, jj] = x ** 2 + y ** 2
    a = ifft2dc(fft2dc(np.cos(in_) * ifft2dc(fft2dc(np.sin(in_)) * f)) / (f + 0.000001))
    b = ifft2dc(fft2dc(np.sin(in_) * ifft2dc(fft2dc(np.cos(in_)) * f)) / (f + 0.000001))
    out = np.real(a - b)
    return out

## Phase generate
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
class PhaseNet(nn.Module):
    def __init__(self, k=3, p=1):
        super(PhaseNet, self).__init__()
        self.act = nn.GELU()
        self.conv_init = nn.Sequential(
            nn.Conv2d(1, 16, 5, stride=1, padding=2),
            self.act,
            nn.BatchNorm2d(16),
            nn.Conv2d(16, 16, k, stride=1, padding=p),
            self.act,
            nn.BatchNorm2d(16)
        )
        self.conv_1 = nn.Sequential(
            nn.Conv2d(64, 64, k, stride=1, padding=p),
            self.act,
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, k, stride=1, padding=p),
            self.act,
            nn.BatchNorm2d(64),
        )
        self.conv_2 = nn.Sequential(
            nn.Conv2d(256, 256, k, stride=1, padding=p),
            self.act,
            nn.BatchNorm2d(256),
            nn.Conv2d(256, 256, k, stride=1, padding=p),
            self.act,
            nn.BatchNorm2d(256),
        )
        self.conv_nonlinear = nn.Sequential(
            nn.Conv2d(1024, 1024, k, stride=1, padding=p),
            self.act,
            nn.BatchNorm2d(1024),
            nn.Conv2d(1024, 16, k, stride=1, padding=p),
            nn.Tanh(),  # tanh
        )
        self.deconv_1 = nn.Sequential(
            nn.Conv2d(16, 1024, k, stride=1, padding=p),
            self.act,
            nn.BatchNorm2d(1024),
            nn.Conv2d(1024, 1024, k, stride=1, padding=p),
            self.act,
            nn.BatchNorm2d(1024),
        )
        self.deconv_2 = nn.Sequential(
            nn.Conv2d(256, 256, k, stride=1, padding=p),
            self.act,
            nn.BatchNorm2d(256),
            nn.Conv2d(256, 256, k, stride=1, padding=p),
            self.act,
            nn.BatchNorm2d(256),
        )
        self.deconv_3 = nn.Sequential(
            nn.Conv2d(64, 64, k, stride=1, padding=p),
            self.act,
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, k, stride=1, padding=p),
            self.act,
            nn.BatchNorm2d(64),
        )
        self.deconv_4 = nn.Sequential(
            nn.Conv2d(16, 16, k, stride=1, padding=p),
            self.act,
            # nn.BatchNorm2d(16),
            nn.Conv2d(16, 16, k, stride=1, padding=p),
            self.act,
            # nn.BatchNorm2d(16),
            nn.Conv2d(16, 1, k, stride=1, padding=p),
        )

    def forward(self, x):
        x = self.conv_init(x)
        x = dwt_init(x)
        x = self.conv_1(x)
        x = dwt_init(x)
        x = self.conv_2(x)
        x = dwt_init(x)
        x = self.conv_nonlinear(x)

        x = self.deconv_1(x)
        x = iwt_initR(x)
        x = self.deconv_2(x)
        x = iwt_initR(x)
        x = self.deconv_3(x)
        x = iwt_initR(x)
        x = self.deconv_4(x)
        return x
        
## SRCNN 
class HNet(nn.Module):
    def __init__(self, inp_dim=1, mod_dim1=64, mod_dim2=32):
        super(HNet, self).__init__()

        self.seq = nn.Sequential(
            nn.Conv2d(inp_dim, mod_dim1, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(mod_dim1),
            nn.ReLU(inplace=True),

            nn.Conv2d(mod_dim1, mod_dim2, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(mod_dim2),
            nn.ReLU(inplace=True),

            nn.Conv2d(mod_dim2, mod_dim1, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(mod_dim1),
            nn.ReLU(inplace=True),

            nn.Conv2d(mod_dim1, inp_dim, kernel_size=1, stride=1, padding=0)
        )

    def forward(self, x):
        return self.seq(x)
