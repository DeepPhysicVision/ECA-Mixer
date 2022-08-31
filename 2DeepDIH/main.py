from common import *
from networks import *
import torch  
import torch.nn as nn
import numpy as np
import PIL.Image as Image
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from torch import optim
from torch.autograd import Variable
import torch.nn.functional as F
from torchsummary import summary
import cv2
import time
from thop import profile

# change: wavelength=636,530,470
def main(Nx = 680, Ny = 680, z = 850, wavelength = 630e-3, deltaX = 1.85, deltaY = 1.85):
    dir = '1951'
    ch = 0
    best = 0.0039
    loss_txt = f'./{dir}/Loss_{ch}.txt'
    img = Image.open(f'./{dir}/{ch}.bmp')
    pil2tensor = transforms.ToTensor()
    
    tensor_img = pil2tensor(img)
    g = tensor_img.numpy()
    g = np.sqrt(g)
    g = (g-np.min(g))/(np.max(g)-np.min(g))

    phase = propagator(Nx,Ny,z,wavelength,deltaX,deltaY)
    eta = np.fft.ifft2(np.fft.fft2(g)*np.fft.fftshift(np.conj(phase)))

    criterion_1 = RECLoss()
    model = Net().cuda()
    optimer_1 = optim.Adam(model.parameters(), lr=5e-3)

    device = torch.device("cuda")
    epoch_1 = 10000
    period = 20000
    eta = torch.from_numpy(np.concatenate([np.real(eta)[np.newaxis,:,:], np.imag(eta)[np.newaxis,:,:]], axis = 1))
    holo = torch.from_numpy(np.concatenate([np.real(g)[np.newaxis,:,:], np.imag(g)[np.newaxis,:,:]], axis = 1))

    TrainTime = time.time()
    for i in range(epoch_1):
        in_img = eta.to(device)
        target = holo.to(device)
        
        out = model(in_img) 
        l1_loss = criterion_1(out,target, Nx,Ny,z,wavelength,deltaX,deltaY)
        loss = l1_loss

        optimer_1.zero_grad()
        loss.backward()
        optimer_1.step()

        if loss < best:
            break
        '''
            best = loss
            outtemp = out.cpu().data.squeeze(0).squeeze(1)
            plotout = torch.sqrt(outtemp[0, :, :] ** 2 + outtemp[1, :, :] ** 2)
            plotout = (plotout - torch.min(plotout)) / (torch.max(plotout) - torch.min(plotout))
            amplitude = np.array(plotout)
            amplitude = amplitude.astype('float32') * 255
            cv2.imwrite(f'./{dir}/Amplitude/Best_{ch}.bmp', amplitude)

            plotout_p = (torch.atan(outtemp[1, :, :] / outtemp[0, :, :])).numpy()
            plotout_p = Phase_unwrapping(in_=plotout_p, size=Nx)
            plotout_p = (plotout_p - np.min(plotout_p)) / (np.max(plotout_p) - np.min(plotout_p))
            phase = np.array(plotout_p)
            phase = phase.astype('float32') * 255
            cv2.imwrite(f'./{dir}/Phase/Best_{ch}.bmp', phase)

        print('epoch [{}/{}]     Loss: {}'.format(i + 1, epoch_1, l1_loss.cpu().data.numpy()), best)
        output = "Epoch: %d , Loss: %f" % (i, loss)
        with open(loss_txt, "a+") as f:
            f.write(output + '\n')
            f.close()
        '''
        '''
        if ((i) % period) == 0:
            outtemp = out.cpu().data.squeeze(0).squeeze(1)
            outtemp = outtemp
            plotout = torch.sqrt(outtemp[0,:,:]**2 + outtemp[1,:,:]**2)
            plotout = (plotout - torch.min(plotout))/(torch.max(plotout)-torch.min(plotout))
            amplitude = np.array(plotout)
            amplitude = amplitude.astype('float32')*255
            cv2.imwrite("./results/Amplitude/iter%d.bmp"%(i), amplitude)
            
            plotout_p = (torch.atan(outtemp[1,:,:]/outtemp[0,:,:])).numpy()
            plotout_p = Phase_unwrapping(plotout_p)
            plotout_p = (plotout_p - np.min(plotout_p))/(np.max(plotout_p)-np.min(plotout_p))
            phase = np.array(plotout_p)
            phase = phase.astype('float32')*255
            cv2.imwrite("./results/Phase/iter%d.bmp"%(i), phase)        
        '''

    TrainTime = time.time() - TrainTime
    inp = in_img.view(1,1,in_img.shape[-3],in_img.shape[-2],in_img.shape[-1])
    flops, params = profile(model, inputs=inp)
    temp = np.zeros((3,3))
    np.savetxt(f'./{dir}/{ch}_Time_{TrainTime}+Flops_{flops/1e9}G+Params_{params/1e6}M',temp)

if __name__ == '__main__':
    main()