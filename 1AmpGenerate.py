from networks import *
import torch
import numpy as np
import PIL.Image as Image
import torchvision.transforms as transforms
import cv2
pil2tensor = transforms.ToTensor()

def main(z=4, deltaX=1.85e-3):
    dir = 'Ocean'
    amp = Image.open(f'./Color/{dir}/A.bmp')
    tensor_amp = pil2tensor(amp)
    tensor_amp = (tensor_amp - torch.min(tensor_amp)) / (torch.max(tensor_amp) - torch.min(tensor_amp))
    print(tensor_amp.shape)

    phase = Image.open(f'./Color/{dir}/A.bmp')
    tensor_phase = pil2tensor(phase)
    tensor_phase = (tensor_phase - torch.min(tensor_phase)) / (torch.max(tensor_phase) - torch.min(tensor_phase))

    for i in range(3):
        #input = tensor_amp[i,:,:] * torch.exp(1j * np.pi * tensor_phase[i,:,:])
        #input = torch.exp(1j * np.pi * tensor_phase[i,:,:])
        input = tensor_amp[i,:,:]
        input = input.view(1,1,input.shape[-2],input.shape[-1])

        wave = [636e-6, 530e-6, 470e-6]
        wavelength = wave[i]
        transfer = propagator(input.shape[-1],input.shape[-2],z,wavelength,deltaX,deltaX)
        eta      = np.fft.ifft2(np.fft.fft2(input.numpy()) * np.fft.fftshift(transfer))

        ampl = np.squeeze(np.abs(eta) * np.abs(eta))
        ampl = (ampl - np.min(ampl)) / (np.max(ampl) - np.min(ampl))
        ampl = ampl.astype('float32') * 255.0
        cv2.imwrite(f'./Color/{dir}/{i}.bmp', ampl)

if __name__ == '__main__':
    main()