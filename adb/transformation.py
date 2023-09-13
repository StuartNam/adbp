import torch.nn as nn
import kornia
import torch
import random
from collections import OrderedDict 
from PIL import Image
import os

def  attack_gaussian_blur():
    """
        Performing the gaussian blur

        kernel_size: int: [3..11]
        std: float: [1..2]
    """

    kernel = random.randint(3, 11)
    sigma = random.uniform(1.0, 3.0)
    print(f"Random: size: {kernel} -- std: {sigma}")

    return kornia.filters.GaussianBlur2d(kernel_size = (kernel, kernel), sigma = (sigma, sigma))

def attack_gaussian_noise():
    mean = 0
    std = random.uniform(0.,1.)
    p = 1.
    return kornia.augmentation.RandomGaussianNoise(mean=mean, std=std, p=p, same_on_batch = True)

def attack_color():

    prob = random.uniform(0.0, 1.0)
    if prob > 0.5:
        return kornia.color.RgbToBgr() 
    return kornia.color.BgrToRgb()
        
def attack_rotation(bs, angle = None):
    """Rotate clock-wise
    args:
        bs: batch_size
        angle: if none -> random angle for define the transform funtion
    """
    if angle is None:
        angle = random.randint(0,360)
        print(f"Random angle = {angle}")    
    if not isinstance(angle, torch.Tensor):
        angle = torch.ones(batch, dtype=torch.float32) * angle

    return kornia.geometry.transform.affwarp.Rotate(angle)

def attack_hflip():
    # prob = random.uniform(0.0, 1.0)
    # if prob > 0.5:
    return kornia.geometry.transform.Hflip()
    # return lambda x: x 
    
def attack_vflip():
    # prob = random.uniform(0.0, 1.0)
    # if prob > 0.5:
    return kornia.geometry.transform.Vflip()
    # return lambda x: x 
    
def attack_warping(params):
    pass

def get_corruption_from_config(params):
    """
    Get attack method ffrom the configuration. There are two types of transformation: intensity and geometry
    args: params: OrderedDict
    return:
        Dict: key: transformation type
              value: list[name of attack method]    
    example:
        {
            "intensity": [gaussian_blur, color, gaussian_noise],
            "geometry": [hflip, vflip]
        }
    """
    transform_dict = OrderedDict()
    for key, value in params.items():
        if isinstance(value, str):
            value = value.split(" ") 
            transform_dict[key] = value
        else:
            transform_dict[key] = value
    return transform_dict

def create_transform_dict(cfg):
    """
    Create a dictionary of transformations.
    With the intensity transformation: strengthness (hyper-params will be random)
    """
    transform_dict_info = get_corruption_from_config(cfg.attackers)
    transform_dict = dict()
    for typ, method_list in transform_dict_info.items():
        if typ != "sequential":
            for method in method_list:
                if method != "rotation":
                    transform_dict[method] = eval(f"attack_{method}()")
                else:
                    transform_dict[method] = eval(f"attack_method(bs = cfg.dataloader.batch_size)") 
    return transform_dict 

def main():
    image_path = './db_dataset/5/set_A/1033.jpg'
    image = Image.open(image_path)
    