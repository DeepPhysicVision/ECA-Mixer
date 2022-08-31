import torch
import numpy as np
import PIL.Image as Image
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import cv2
import torch.nn.functional as F
import time
import torch.nn as nn
import utils
from einops.layers.torch import Rearrange
from torch import optim
from thop import profile    #pip install thop
from networks import *

#def main(Nx=512, z=1100e-3, wavelength=600e-6, deltaX=2.2e-3):  #512_USAF f1
#def main(Nx=512, z=620e-3,  wavelength=635e-6, deltaX=2.2e-3):  #512_Cell f2
#def main(Nx=1000, Ny=1000, z=1300e-3, wavelength=635e-6, deltaX=2e-3):  #DIH
#def main(Nx=256, z=1, wavelength=635e-6, deltaX=2.2e-3):   #AmpOnly PhaseOnly AmpPhase
def main(z=900e-3, deltaX=1.85e-3):   # exp color  usaf850
#def main(z=5, wavelength=635e-6, deltaX=2.2e-3):   # simulate color

    for j in range(3):
        wave = [606e-6, 512e-6, 452e-6]
        wavelength = wave[j]

        print('######### Start reconstruct siat',j)
        ############################# load diffraction
        src_data   = f'Siat{j}' ### change
        src_save   = f'Siat{j}/AttentionNet'
        loss_txt   = f'./Results/{src_save}/loss.txt'

        diff       = Image.open(f'./Data/{src_data}.bmp')
        pil2tensor = transforms.ToTensor()
        tensor_diff = pil2tensor(diff)

        # for 3channel image
        print(tensor_diff.shape)
        if tensor_diff.shape[-3] == 3:
            tensor_diff = tensor_diff[j,:,:]

        #tensor_diff = (tensor_diff - torch.min(tensor_diff)) / (torch.max(tensor_diff) - torch.min(tensor_diff))
        tensor_diff = torch.sqrt(tensor_diff)
        tensor_diff = tensor_diff.view(1,1,tensor_diff.shape[-2],tensor_diff.shape[-1])
        #tensor_diff = Rearrange('(b c) h w -> b c h w', c=1)(tensor_diff)

        Nx = tensor_diff.shape[-1]
        Ny = tensor_diff.shape[-2]

        ############################################ back propagate to image plane
        transfer = propagator(Nx,Ny,-z,wavelength,deltaX,deltaX)
        eta      = np.fft.ifft2(np.fft.fft2(tensor_diff.numpy()) * np.fft.fftshift(transfer))

        ampl = np.abs(eta)
        ampl = np.squeeze(ampl)
        ampl = (ampl - np.min(ampl)) / (np.max(ampl) - np.min(ampl))
        ampl = ampl.astype('float32') * 255.0
        cv2.imwrite(f'./Results/{src_save}/BackAmp.png', ampl)

        p = np.angle(eta)
        #p = Phase_unwrapping(p,Ny,Nx)
        p = np.squeeze(p)
        p = (p - np.min(p)) / (np.max(p) - np.min(p))
        p = p.astype('float32') * 255.0
        cv2.imwrite(f'./Results/{src_save}/BackPhase.png', p)

        ################################################ load model
        Model    = AttentionNet().cuda()
        Optimer  = optim.Adam(Model.parameters(), lr=5e-3)
        ModelH   = HNet().cuda()
        OptimerH = optim.Adam(ModelH.parameters(), lr=5e-3)
        ModelZ   = ZNet().cuda()
        OptimerZ = optim.Adam(ModelZ.parameters(), lr=5e-3)

        device = torch.device("cuda")
        epoch  = 3001
        period = 100
        es     = 0
        standard = 0.003
        best = standard
        TrainTime = time.time()

        ########################################### main training
        for i in range(epoch):
            target         = tensor_diff.to(device)
            complex_target = target.type(torch.complex64)
            BackTarget     = torch.from_numpy(eta).type(torch.complex64).to(device)
            #in_img = Rearrange('b c (h p1) (w p2) -> (b h w) c p1 p2', p1=256, p2=256)(in_img)

            '''
            ## model z
            #slm_phase = (-0.5 + 1.0 * torch.rand(target.shape)).to(device)
            #slm_phase = slm_phase.requires_grad_(True)
            #optvars = [{'params': slm_phase}]
            #optimizerP = optim.Adam(optvars, lr=5e-3)
            len = 40
            pp = torch.zeros(1, len).to(device)
            for j in range(len):
                pp[:, j] = j/1000 + z - len/2000
            zz = ModelZ(pp)
            LossZ = torch.mean(torch.abs(zz - z)) / 2
            zz = zz.cpu().detach().numpy()
            '''

            #ip  = ModelH(target)
            out = Model(complex_target)
            #out = Rearrange('(b h w) c p1 p2 -> b c (h p1) (w p2)', b=1, h=5)(out)

            trans = propagator(Nx, Ny, z, wavelength, deltaX, deltaX)
            trans = torch.from_numpy(trans).to(device)
            OutField = torch.fft.ifft2(torch.fft.fft2(out) * torch.fft.fftshift(trans))
            OutAmp   = torch.abs(OutField)
            OutPhase = torch.angle(OutField)

            #target = F.interpolate(target, size=[Nx*4, Nx*4], mode='bilinear')
            #plt.imshow(np.squeeze(target.cpu().detach().numpy()), cmap='gray')
            #plt.show()

            Loss = torch.mean(torch.abs(OutAmp*OutAmp - target*target)) / 2
            #LossH = torch.mean(torch.abs(ip*ip - OutPhase*OutPhase)) / 2
            #w  = torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=torch.float).cuda()
            #aa = torch.angle(out)
            #aa = torch.var(F.conv2d(aa, w.view(1, 1, 3, 3)))
            #LossP = torch.mean(torch.abs(aa)) / 2

            Optimer.zero_grad()
            #OptimerH.zero_grad()
            #LossP.backward(retain_graph=True)
            Loss.backward()
            Optimer.step()
            #OptimerH.step()

            if ((i) % period) == 0:
                color = 'gray'
                Super  = out.cpu().detach().numpy()
                SuperA = np.squeeze(np.abs(Super))
                SuperA = (SuperA - np.min(SuperA)) / (np.max(SuperA) - np.min(SuperA))
                SuperA = SuperA.astype('float32') * 255.0
                #SuperP = Phase_unwrapping(np.angle(Super), Shape=Nx)
                #plt.imsave(f'./Results/{src_save}/Amp/Iter%d.bmp' % (i), SuperA, cmap=color)
                cv2.imwrite(f'./Results/{src_save}/Amp/Iter%d.bmp' % (i), SuperA)
                SuperP = np.angle(Super)
                SuperP = np.squeeze(SuperP)
                SuperP = (SuperP - np.min(SuperP)) / (np.max(SuperP) - np.min(SuperP))
                SuperP = SuperP.astype('float32') * 255.0
                #plt.imsave(f'./Results/{src_save}/Phase/Iter%d.bmp'%(i), SuperP, cmap=color)
                cv2.imwrite(f'./Results/{src_save}/Phase/Iter%d.bmp'%(i), SuperP)

            if Loss < best:
                best = Loss
                es = 0
                Super = out.cpu().detach().numpy()
                SuperA = np.squeeze(np.abs(Super))
                SuperA = (SuperA - np.min(SuperA)) / (np.max(SuperA) - np.min(SuperA))
                SuperA = SuperA.astype('float32') * 255.0
                #plt.imsave(f'./Results/{src_save}/Amp/Best.bmp', SuperA, cmap=color)
                cv2.imwrite(f'./Results/{src_save}/Amp/Best.bmp', SuperA)
                SuperP = np.angle(Super)
                SuperP = np.squeeze(SuperP)
                SuperP = (SuperP - np.min(SuperP)) / (np.max(SuperP) - np.min(SuperP))
                SuperP = SuperP.astype('float32') * 255.0
                #plt.imsave(f'./Results/{src_save}/Phase/Best.bmp', SuperP, cmap=color)
                cv2.imwrite(f'./Results/{src_save}/Phase/Best.bmp', SuperP)

            else:
                if best < standard:
                    es += 1
                    print("Counter {} of 15".format(es))
                if es > 30:
                    print("Early stopping with best: ", best, "and loss for this epoch: ", Loss)
                    break

            print('Epoch:', i, 'Loss:', Loss.item(), 'Best:', best)

            output = "Epoch: %d  ,Loss: %f" % (i, Loss)
            with open(loss_txt, "a+") as f:
                f.write(output + '\n')
                f.close

        #parameters = filter(lambda p: p.requires_grad, Model.parameters())
        #parameters = sum([np.prod(p.size()) for p in parameters]) / 1_000_000
        #print('Trainable Parameters: %.3fM' % parameters)

        flops,params = profile(Model, inputs=(complex_target,))
        #print("FLOPs=",  str(flops/1e9) +'{}'.format("G"))
        #print("params=", str(params/1e6)+'{}'.format("M"))
        #params = params / 1e6
        #flops  = params / 1e9

        TrainTime = time.time() - TrainTime
        print('Training time', TrainTime)
        temp = np.zeros((3,3))
        np.savetxt(f'./Results/{src_save}/Time_{TrainTime} + BestLoss_{best}.txt + Flops_{flops/1e9}G + params_{params/1e6}M',temp)

if __name__ == '__main__':
    main()