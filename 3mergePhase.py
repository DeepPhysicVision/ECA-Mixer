import imageio
from utils.utils import *
import PIL.Image as Image
import torchvision.transforms as transforms
from networks import *
pil2tensor = transforms.ToTensor()

# dl
target_dir = 'QiuYin'
target_amp = Image.open(f'./Color/{target_dir}/A.bmp')
target_amp = pil2tensor(target_amp)
target_amp = target_amp.view(target_amp.shape[-3],target_amp.shape[-2],target_amp.shape[-1])

result_dir = f'Color/{target_dir}/AttentionNet'
recon_amp = []

for i in range(3):
    img = Image.open(f'./{result_dir}/Phase/Best_{i}.bmp')
    img = pil2tensor(img)
    img = img.view(img.shape[-2], img.shape[-1])
    img = (img - torch.min(img)) / (torch.max(img) - torch.min(img))
    recon_amp.append(img)
recon_amp = torch.stack(recon_amp, 0)

recon_amp = recon_amp.squeeze().cpu().detach().numpy()
target_amp = target_amp.squeeze().cpu().detach().numpy()
recon_amp = recon_amp.transpose(1, 2, 0)
target_amp = target_amp.transpose(1, 2, 0)

recon_srgb = srgb_lin2gamma(np.clip(recon_amp ** 2, 0.0, 1.0))
imageio.imwrite(f'./{result_dir}/Dl_Phase.bmp', (recon_srgb * np.iinfo(np.uint8).max).round().astype(np.uint8))

psnrs = {'amp': [], 'lin': [], 'srgb': []}
ssims = {'amp': [], 'lin': [], 'srgb': []}
idxs = []

'''
# calculate metrics
psnr_val, ssim_val = get_psnr_ssim(recon_amp, target_amp, multichannel=3)
for domain in ['amp', 'lin', 'srgb']:
    psnrs[domain].append(psnr_val[domain])
    ssims[domain].append(ssim_val[domain])
    print(f'PSNR({domain}): {psnr_val[domain]},  SSIM({domain}): {ssim_val[domain]:.4f}')
'''

# back
recon_amp = []
for i in range(3):
    img = Image.open(f'./{result_dir}/Phase/Back_{i}.bmp')    ### asa
    img = pil2tensor(img)
    img = img.view(img.shape[-2], img.shape[-1])

    recon_amp.append(img)
recon_amp = torch.stack(recon_amp, 0)

recon_amp = recon_amp.squeeze().cpu().detach().numpy()
recon_amp = recon_amp.transpose(1, 2, 0)

recon_srgb = srgb_lin2gamma(np.clip(recon_amp ** 2, 0.0, 1.0))
imageio.imwrite(f'./{result_dir}/Back_Phase.bmp', (recon_srgb * np.iinfo(np.uint8).max).round().astype(np.uint8))  #asa

psnrs = {'amp': [], 'lin': [], 'srgb': []}
ssims = {'amp': [], 'lin': [], 'srgb': []}
idxs = []

psnr_val, ssim_val = get_psnr_ssim(recon_amp, target_amp, multichannel=3)
for domain in ['amp', 'lin', 'srgb']:
    psnrs[domain].append(psnr_val[domain])
    ssims[domain].append(ssim_val[domain])
    print(f'PSNR({domain}): {psnr_val[domain]},  SSIM({domain}): {ssim_val[domain]:.4f}')
