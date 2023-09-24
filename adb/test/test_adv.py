import numpy as np 
import torch
import os 
import torchvision.transforms as transform
from skimage import metrics
from argparse import ArgumentParser 
import kornia as K 
from PIL import Image
from tqdm import tqdm
from piq import ssim

def get_args():
    args_parse = ArgumentParser()
    args_parse.add_argument('--sd-ckpt', help = 'Path to the stable-diffusion checkpoint folder', required = True) 
    args_parse.add_argument('--config', help = "Path to configuration for fine-tuning", default = "")
    args_parse.add_argument('--seed', help = "Path to configuration for fine-tuning", default = 6969)

    args = args_parse.parse_args()
    return args

def eval_ssim(image1: torch.Tensor, image2: torch.Tensor):
    """
    args:
        image1, image2: torch.Tensor of shape [N, C, H ,W]
        pixel range: [0.0, 1.0]
    return:
        ssim_score
    """
    scores = []
    image_1 = image1.detach().permute(0,2,3,1).cpu().numpy() * 255
    image_2 = image2.detach().permute(0,2,3,1).cpu().numpy() * 255

    image_list1 = [Image.fromarray(img.astype(np.uint8)) for img in image_1]
    image_list2 = [Image.fromarray(img.astype(np.uint8)) for img in image_2]
    for idx, item in enumerate(image_list1):
        ref_image = np.array(item.convert('L'))
        wm_image = np.array(image_list2[idx].convert('L'))
        ssim_score = metrics.structural_similarity(ref_image, wm_image, gaussian_weights=True, sigma=1.5, use_sample_covariance = False, data_range=1.0)
        scores.append(ssim_score)
    
    return scores

def eval_psnr(image1: torch.Tensor, image2: torch.Tensor):

    image_1 = image1.detach().cpu().numpy()
    image_2 = image2.detach().cpu().numpy()
    return metrics.peak_signal_noise_ratio(image_1, image_2)

def test_perceptual_adv(adv_folder, clean_folder):

    adv_imgs = []
    clean_imgs = []

    adv_files = sorted(os.listdir(adv_folder))
    clean_files = sorted(os.listdir(clean_folder))
    clean_files = [filename for filename in clean_files if not os.path.isdir(filename)] 

    for idx, filename in enumerate(adv_files):
        adv_img = K.io.load_image(f"{adv_folder}/{filename}", K.io.ImageLoadType.RGB8, device="cuda").unsqueeze(0)
        clean_img = K.io.load_image(f"{clean_folder}/{clean_files[idx]}", K.io.ImageLoadType.RGB8, device="cuda").unsqueeze(0)
        clean_imgs.append(clean_img )  
        adv_imgs.append(adv_img)

    adv_imgs = torch.cat(adv_imgs) / 255.0
    clean_imgs = torch.cat(clean_imgs) / 255.0
    
    ssim_score = eval_ssim(adv_imgs, clean_imgs)
    # ssim_score = ssim(adv_imgs, clean_imgs, data_range = 1.0)

    psnr_score = eval_psnr(adv_imgs, clean_imgs)
    breakpoint()

    return ssim_score, psnr_score


def main():
    root = '/home/ubuntu/adbp/adb/outputs'
    aspl_folder = f"{root}/ASPL"
    res = []
    for folder in tqdm(sorted(os.listdir(aspl_folder))):
        adv_folder = f"{aspl_folder}/{folder}/adversarial/noise-ckpt/50"
        clean_folder = f"{aspl_folder}/{folder}/adversarial/image_before_addding_noise"
        
        ssim_score, psnr_score = test_perceptual_adv(adv_folder, clean_folder)
        res.append({
            "folder": folder,
            "ssim": ssim_score,
            "psnr": psnr_score
        }) 
    
if __name__ == '__main__':
    main()