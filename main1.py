import os
import PIL.Image as Image
import torchvision.transforms as transforms
import cv2
import time
from torch import optim
# pip install thop
from thop import profile
from networks import *

#### change 0
def main(z=1, deltaX=1.85e-3):
    src_save = 'Results/LotusZ1/AttentionNet'     #### change 1
    folder = os.path.exists(src_save)
    if not folder:
        os.makedirs(f'./{src_save}/Amp')
        os.makedirs(f'./{src_save}/Phase')
    loss_txt = f'./{src_save}/Loss.txt'

    wave = [606e-6, 512e-6, 452e-6]    #### [624e-6, 530e-6, 470e-6] for bone

    for j in range(3):
        src_data = f'LotusZ1/{j}'  #### change 2
        diff = Image.open(f'./Color/{src_data}.bmp')
        pil2tensor = transforms.ToTensor()
        tensor_diff = pil2tensor(diff)
        tensor_diff = (tensor_diff - torch.min(tensor_diff)) / (torch.max(tensor_diff) - torch.min(tensor_diff))
        tensor_diff = torch.sqrt(tensor_diff)
        tensor_diff = tensor_diff.view(1, 1, tensor_diff.shape[-2], tensor_diff.shape[-1])

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
        cv2.imwrite(f'./{src_save}/BackAmp{j}.bmp', ampl)

        p = np.angle(eta)
        p = np.squeeze(p)
        p = (p - np.min(p)) / (np.max(p) - np.min(p))
        p = p.astype('float32') * 255.0
        cv2.imwrite(f'./{src_save}/BackPhase{j}.bmp', p)

        ############################################ load model
        Model = AttentionNet().cuda()
        Optimer = optim.Adam(Model.parameters(), lr=5e-3)
        # ModelZ   = ZNet().cuda()
        # OptimerZ = optim.Adam(ModelZ.parameters(), lr=5e-3)

        device = torch.device("cuda")
        epoch = 1001
        period = 100
        es = 0
        standard = 0.005  ####change 3
        best = standard

        wavelength = wave[j]
        pp = propagator(Nx, Ny, z, wavelength, deltaX, deltaX)
        prop = torch.from_numpy(pp)
        prop = prop.to(device)

        ########################################### main training
        BackTarget = torch.from_numpy(eta).type(torch.complex64).to(device)
        target = tensor_diff.to(device)
        complex_target = target.type(torch.complex64)

        TrainTime = time.time()
        for i in range(epoch):
            out = Model(BackTarget)
            OutField = torch.fft.ifft2(torch.fft.fft2(out) * torch.fft.fftshift(prop))
            OutAmp = torch.abs(OutField)

            # super resolution
            # tran = F.interpolate(tran, size=[Nx, Ny], mode='bilinear')
            # lapulas
            # w  = torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=torch.float).cuda()
            # aa = torch.angle(out)
            # aa = torch.var(F.conv2d(aa, w.view(1, 1, 3, 3)))
            # LossP = torch.mean(torch.abs(aa)) / 2

            Loss = torch.mean(torch.abs(OutAmp * OutAmp - target * target)) / 2
            # Lossr = torch.mean(torch.abs(tran[:,0,:,:]*tran[:,0,:,:] - target[:,0,:,:]*target[:,0,:,:])) / 2
            # Lossg = torch.mean(torch.abs(tran[:,1,:,:]*tran[:,1,:,:] - target[:,1,:,:]*target[:,1,:,:])) / 2
            # Lossb = torch.mean(torch.abs(tran[:,2,:,:]*tran[:,2,:,:] - target[:,2,:,:]*target[:,2,:,:])) / 2
            Optimer.zero_grad()
            # Lossr.backward(retain_graph=True)
            # Lossg.backward(retain_graph=True)
            Loss.backward()
            Optimer.step()

            '''
            if ((i) % period) == 0:
                Super  = out.cpu().detach().numpy()
                SuperA = np.squeeze(np.abs(Super))
                SuperA = (SuperA - np.min(SuperA)) / (np.max(SuperA) - np.min(SuperA))
                SuperA = SuperA.astype('float32') * 255.0
                cv2.imwrite(f'./{src_save}/Amp/{i}_0.bmp', SuperA[0,:,:])
                cv2.imwrite(f'./{src_save}/Amp/{i}_1.bmp', SuperA[1,:,:])
                cv2.imwrite(f'./{src_save}/Amp/{i}_2.bmp', SuperA[2,:,:])
                SuperP = np.angle(Super)
                SuperP = np.squeeze(SuperP)
                SuperP = (SuperP - np.min(SuperP)) / (np.max(SuperP) - np.min(SuperP))
                SuperP = SuperP.astype('float32') * 255.0
                cv2.imwrite(f'./{src_save}/Phase/{i}_0.bmp', SuperP[0,:,:])
                cv2.imwrite(f'./{src_save}/Phase/{i}_1.bmp', SuperP[1,:,:])
                cv2.imwrite(f'./{src_save}/Phase/{i}_2.bmp', SuperP[2,:,:])
            '''

            if Loss < best:
                Super = out.cpu().detach().numpy()
                SuperA = np.squeeze(np.abs(Super))
                SuperA = (SuperA - np.min(SuperA)) / (np.max(SuperA) - np.min(SuperA))
                SuperA = SuperA.astype('float32') * 255.0
                cv2.imwrite(f'./{src_save}/Amp/Best{j}.bmp', SuperA)
                SuperP = np.angle(Super)
                SuperP = np.squeeze(SuperP)
                SuperP = (SuperP - np.min(SuperP)) / (np.max(SuperP) - np.min(SuperP))
                SuperP = SuperP.astype('float32') * 255.0
                cv2.imwrite(f'./{src_save}/Phase/Best{j}.bmp', SuperP)
                # cv2.imwrite(f'./{src_save}/Phase/Best.bmp', SuperP.transpose(1,2,0))
                break

            '''
            else:
                if best < standard:
                    es += 1
                    #print("Counter {} of 30".format(es))
                if es > 10:
                    print("Early stopping with best: ", best, "and loss for this epoch: ", Loss)
                    break
            '''
            print('Epoch:', i, 'Loss:', Loss.item(), 'Best:', best)
            output = "Epoch: %d  ,Loss: %f" % (i, Loss)
            with open(loss_txt, "a+") as f:
                f.write(output + '\n')
                f.close

        TrainTime = time.time() - TrainTime
        flops, params = profile(Model, inputs=(complex_target,))
        print('Training time', TrainTime)
        temp = np.zeros((3, 3))
        np.savetxt(
            f'./{src_save}/Time_{TrainTime} + BestLoss_{best}.txt + Flops_{flops / 1e9}G + Params_{params / 1e6}M',
            temp)


if __name__ == '__main__':
    main()