import matplotlib.pyplot as plt
import os
import PIL.Image as Image
import torchvision.transforms as transforms
from networks import *
import cv2
import torch
import torch.nn as nn
from torch.autograd import Variable
 
class TVLoss(nn.Module):
    def __init__(self,TVLoss_weight=1):
        super(TVLoss,self).__init__()
        self.TVLoss_weight = TVLoss_weight
 
    def forward(self,x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self._tensor_size(x[:,:,1:,:])  #算出总共求了多少次差
        count_w = self._tensor_size(x[:,:,:,1:])
        h_tv = torch.pow((x[:,:,1:,:]-x[:,:,:h_x-1,:]),2).sum()  
        # x[:,:,1:,:]-x[:,:,:h_x-1,:]就是对原图进行错位，分成两张像素位置差1的图片，第一张图片
        # 从像素点1开始（原图从0开始），到最后一个像素点，第二张图片从像素点0开始，到倒数第二个            
        # 像素点，这样就实现了对原图进行错位，分成两张图的操作，做差之后就是原图中每个像素点与相
        # 邻的下一个像素点的差。
        w_tv = torch.pow((x[:,:,:,1:]-x[:,:,:,:w_x-1]),2).sum()
        return self.TVLoss_weight*2*(h_tv/count_h+w_tv/count_w)/batch_size
 
    def _tensor_size(self,t):
        return t.size()[1]*t.size()[2]*t.size()[3]
 
def main():
    # x = Variable(torch.FloatTensor([[[1,2],[2,3]],[[1,2],[2,3]]]).view(1,2,2,2), requires_grad=True)
    # x = Variable(torch.FloatTensor([[[3,1],[4,3]],[[3,1],[4,3]]]).view(1,2,2,2), requires_grad=True)
    # x = Variable(torch.FloatTensor([[[1,1,1], [2,2,2],[3,3,3]],[[1,1,1], [2,2,2],[3,3,3]]]).view(1, 2, 3, 3), requires_grad=True)
    x = Variable(torch.FloatTensor([[[1, 2, 3], [2, 3, 4], [3, 4, 5]], [[1, 2, 3], [2, 3, 4], [3, 4, 5]]]).view(1, 2, 3, 3),requires_grad=True)
    addition = TVLoss()
    z = addition(x)
    print(x)
    print(z.data)
    z.backward()
    print(x.grad)
    
if __name__ == '__main__':
    main()







a = plt.imread('/opt/data/private/server/0130/Color/Phase/AttentionNet/Phase/Best_2.bmp')
plt.imshow(a, cmap='Blues')
plt.show()

def main():
    src = 'Results/0FinalResult/BoneFov/AttentionNet/Phase'
    diff        = Image.open(f'{src}/Best0.bmp')
    pil2tensor  = transforms.ToTensor()
    tensor_diff = pil2tensor(diff)
    pp = tensor_diff.numpy()
    pp = Phase_unwrapping(pp, Ny=pp.shape[-1], Nx=pp.shape[-2])

    pp = np.squeeze(pp)
    pp = (pp - np.min(pp)) / (np.max(pp) - np.min(pp))
    pp = pp.astype('float32') * 255.0
    cv2.imwrite(f'{src}/Best0_U.bmp', pp)

if __name__ == '__main__':
    main()

'''
# matlab
% Display the reconstructed phase
phase_target = wrapToPi(data);
figure; imagesc(data);colormap jet;axis off image;colorbar;
'''
