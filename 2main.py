# pip install thop
from thop import profile
import os
import PIL.Image as Image
import torchvision.transforms as transforms
from einops.layers.torch import Rearrange
import torch.nn.functional as F
import cv2
import time
from torch import optim
from networks import *
import torch
import scipy.io as scio

#### change 1
def main(z=1, deltaX=1.85e-3):
    Model    = AttentionNet().cuda()  # change 2 AttentionNet  EcaNet  ConvNet
    Optimer  = optim.Adam(Model.parameters(), lr=1e-2)  #5e-3  2k:1e-2

    dir = 'QiuYin'    # change 3 dir
    device = torch.device("cuda")
    epoch  = 2000
    period = 300
    es     = 0
    best = 0.011  #### change 4
    wave = [636e-6, 530e-6, 470e-6]

    src_save = f'Color/{dir}/AttentionNet'     #### change 5
    folder = os.path.exists(src_save)
    if not folder:
        os.makedirs(f'./{src_save}/Amp')
        os.makedirs(f'./{src_save}/Phase')
    loss_txt = f'./{src_save}/Loss.txt'

    Diffraction = []
    Back = []
    for j in range(3):
        src_data    = f'{dir}/{j}'
        diff        = Image.open(f'./Color/{src_data}.bmp')
        pil2tensor  = transforms.ToTensor()
        tensor_diff = pil2tensor(diff)
        tensor_diff = torch.sqrt(tensor_diff)
        tensor_diff = (tensor_diff - torch.min(tensor_diff)) / (torch.max(tensor_diff) - torch.min(tensor_diff))
        tensor_diff = tensor_diff.view(1,1,tensor_diff.shape[-2],tensor_diff.shape[-1])

        ########## back propagate to image plane
        Nx = tensor_diff.shape[-1]
        Ny = tensor_diff.shape[-2]

        wavelength = wave[j]
        transfer = propagator(Nx, Ny, -z, wavelength, deltaX, deltaX)
        eta = np.fft.ifft2(np.fft.fft2(tensor_diff.numpy()) * np.fft.fftshift(transfer))

        ampl = np.abs(eta)
        ampl = np.squeeze(ampl)
        ampl = (ampl - np.min(ampl)) / (np.max(ampl) - np.min(ampl))
        ampl = ampl.astype('float32') * 255.0
        cv2.imwrite(f'./{src_save}/Amp/Back_{j}.bmp', ampl)

        p = np.angle(eta)
        p = np.squeeze(p)
        scio.savemat(f'./{src_save}/Phase/Back_{j}.mat', {'data': p})
        p = (p - np.min(p)) / (np.max(p) - np.min(p))
        p = p.astype('float32') * 255.0
        cv2.imwrite(f'./{src_save}/Phase/Back_{j}.bmp', p)

        Diffraction.append(tensor_diff)
        Back.append(torch.from_numpy(eta))
    Diffraction = torch.cat(Diffraction, 1)
    Back = torch.cat(Back,1)

    prop = []
    for g in range(3):
        wavelength = wave[g]
        pp = propagator(Nx, Ny, z, wavelength, deltaX, deltaX)
        pp = torch.from_numpy(pp)
        prop.append(pp)
    prop = torch.cat(prop, 1)
    prop = prop.to(device)

    ########################################### main training
    BackTarget   = Back.type(torch.complex64).to(device)
    #BackTarget = Rearrange('b c (h p1) (w p2) -> (b h w) c p1 p2', p1=200, p2=1240)(BackTarget)

    target = Diffraction.to(device)
    complex_target = target.type(torch.complex64)

    TrainTime = time.time()
    for i in range(epoch):
        out  = Model(BackTarget)
        #out = Rearrange('(b h w) c p1 p2 -> b c (h p1) (w p2)', b=1, h=8)(out)

        intenity = []
        for g in range(3):
            OutField = torch.fft.ifft2(torch.fft.fft2(out[:,g,:,:]) * torch.fft.fftshift(prop[:,g,:,:]))
            OutAmp   = torch.abs(OutField)
            intenity.append(OutAmp)
        intenity = torch.stack(intenity, 1)

        Loss = torch.mean(torch.abs(intenity*intenity - target*target)) / 2
        Optimer.zero_grad()
        Loss.backward()
        Optimer.step()

        if Loss < best:
            #break
            #best = Loss
            Super = out.cpu().detach().numpy()
            for n in range(3):
                SuperA = np.squeeze(np.abs(Super)[:,n,:,:])
                SuperA = (SuperA - np.min(SuperA)) / (np.max(SuperA) - np.min(SuperA))
                SuperA = SuperA.astype('float32') * 255.0
                cv2.imwrite(f'./{src_save}/Amp/Best_{n}.bmp', SuperA)

                SuperP = np.squeeze(np.angle(Super)[:,n,:,:])
                scio.savemat(f'./{src_save}/Phase/Best_{n}.mat', {'data':SuperP})
                SuperP = Phase_unwrapping(SuperP, Ny=SuperP.shape[-1], Nx=SuperP.shape[-2])
                SuperP = (SuperP - np.min(SuperP)) / (np.max(SuperP) - np.min(SuperP))
                SuperP = SuperP.astype('float32') * 255.0
                cv2.imwrite(f'./{src_save}/Phase/Best_{n}.bmp', SuperP)
            break

        output = "E: %d  ,L: %f" % (i, Loss)
        with open(loss_txt, "a+") as f:
            f.write(output + '\n')
            f.close

        #endtime = time.time() - TrainTime
        #print('E:', i, 'L:', Loss.item(), 'T:', endtime)
        print('Epoch:', i, 'Loss:', Loss.item(), 'Best:', best)

        '''
        if ((i) % period) == 0:
            Super  = out.cpu().detach().numpy()
            for m in range(3):
                SuperA = np.squeeze(np.abs(Super)[:,m,:,:])
                SuperA = (SuperA - np.min(SuperA)) / (np.max(SuperA) - np.min(SuperA))
                SuperA = SuperA.astype('float32') * 255.0
                cv2.imwrite(f'./{src_save}/Amp/{i}_{m}.bmp', SuperA)

                SuperP = np.squeeze(np.angle(Super)[:,m,:,:])
                scio.savemat(f'./{src_save}/Phase/{i}_{m}.mat', {'data':SuperP})
                SuperP = Phase_unwrapping(SuperP, Ny=SuperP.shape[-1], Nx=SuperP.shape[-2])
                SuperP = (SuperP - np.min(SuperP)) / (np.max(SuperP) - np.min(SuperP))
                SuperP = SuperP.astype('float32') * 255.0
                cv2.imwrite(f'./{src_save}/Phase/{i}_{m}.bmp', SuperP)
                #cv2.imwrite(f'./{src_save}/Phase/{i}.bmp',  SuperP.transpose(1,2,0))
        '''

    TrainTime = time.time() - TrainTime
    flops, params = profile(Model, inputs=(BackTarget,))
    print('Training time', TrainTime)
    temp = np.zeros((3,3))
    np.savetxt(f'./{src_save}/Time_{TrainTime} + BestLoss_{best}.txt + LastLoss_{Loss}.txt + '
               f'Flops_{flops/1e9}G + Params_{params/1e6}M', temp)

if __name__ == '__main__':
    main()
