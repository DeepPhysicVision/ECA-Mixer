import imageio
from utils.utils import *
import PIL.Image as Image
import torchvision.transforms as transforms
from networks import *
pil2tensor = transforms.ToTensor()

result_dir = 'Color/QiuYin'
recon_amp = []

for i in range(3):
    img = Image.open(f'./{result_dir}/{i}.bmp')
    img = pil2tensor(img)
    img = img.view(img.shape[-2], img.shape[-1])
    img = (img - torch.min(img)) / (torch.max(img) - torch.min(img))
    recon_amp.append(img)
recon_amp = torch.stack(recon_amp, 0)

recon_amp = recon_amp.squeeze().cpu().detach().numpy()
recon_amp = recon_amp.transpose(1, 2, 0)

recon_srgb = srgb_lin2gamma(np.clip(recon_amp ** 2, 0.0, 1.0))
imageio.imwrite(f'./{result_dir}/Holo.bmp', (recon_srgb * np.iinfo(np.uint8).max).round().astype(np.uint8))