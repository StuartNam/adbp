import kornia
import torch
import random
from PIL import Image
import torchvision.transforms as transforms
from omegaconf import OmegaConf

class Transformation():
    @classmethod
    def prepare(cls, image, dtype = torch.float32, requires_grad = False):
        """
            Prepare image before transformation

            Args:
            - image: PIL.Image - shape: (C, H, W) - dtype: uint8
            - dtype

            Return:
            - tensor: torch.Tensor - shape: (B, C, H, W) - dtype: dtype
        """
        return transforms.PILToTensor()(image).unsqueeze(0).to(dtype).requires_grad_(requires_grad)
    
    @classmethod
    def to_image(cls, image):
        """
            Transform tensor into PIL.Image to save or display
        """
        return transforms.ToPILImage()(image.squeeze(0).to(torch.uint8))
    
    @classmethod
    def get_pipe(cls, config):
        def chain(methods):
            from functools import reduce
            return lambda x: reduce(lambda v, f: f(v), methods, x)

        methods = []

        for method in config.transformation.methods:
            if method.name == 'gaussian_blur':
                methods.append(Transformation.gaussian_blur_2d(method.config))
            
            if method.name == 'gaussian_noisify':
                methods.append(Transformation.gaussian_noisify(method.config))

            if method.name == 'color_convert':
                methods.append(Transformation.color_convert(method.config))
            
            if method.name == 'rotate':
                methods.append(Transformation.rotate(method.config))
            
            if method.name == 'hflip':
                methods.append(Transformation.hflip(method.config))
            
            if method.name == 'vflip':
                methods.append(Transformation.vflip(method.config))

        if config.transformation.stack:
            return chain(methods)  
        
        return methods
    
    @classmethod
    def gaussian_blur_2d(cls, config = None):
        """
        Performing the gaussian blur with kernel_size and sigma
        Args:
        - config: {'kernel_size': ..., 'sigma': ...}
        """

        if config['kernel_size'] == 'random':
            kernel_size = random.choice([3, 5, 7, 9, 11])
        else:
            try:
                kernel_size = int(config['kernel_size'])
            except:
                print("gaussian_blur_2d(): Kernel size must be able to convert to int")

        if config['sigma'] == 'random':
            sigma = random.uniform(1.0, 2.0)
        else:
            try:
                sigma = float(config['sigma'])
            except:
                print("gaussian_blur_2d(): Sigma must be able to convert to float")

        return kornia.filters.GaussianBlur2d(
            kernel_size = (kernel_size, kernel_size),
            sigma = (sigma, sigma)
        )

    @classmethod
    def gaussian_noisify(cls, config = None):
        """
            Add Gaussian noise to the image
            Args:
                - config: None
        """
        mean = 0.0
        std = random.uniform(0.0, 1.0)
        p = 1.0

        return kornia.augmentation.RandomGaussianNoise(mean = mean, std = std, p = p, same_on_batch = True)

    @classmethod
    def color_convert(cls, config = None):
        prob = random.uniform(0.0, 1.0)
        if prob > 0.5:
            return kornia.color.RgbToBgr() 
        
        return kornia.color.BgrToRgb()
    
    @classmethod
    def rotate(cls, config = None):
        """
        Rotate clock-wise
        Args:
            - bs: batch_size
            - angle: if none -> random angle for define the transform funtion
        """

        if config.angle == 'random':
            angle = random.randint(0, 360)
            print(f"Random angle = {angle}")

        if not isinstance(config.angle, torch.Tensor):
            try:
                angle = torch.ones(config.batch_size, dtype = torch.float32) * int(config.angle)
            except:
                raise RuntimeError("Transformation.rotate(): Batch size is required")

        return kornia.geometry.transform.affwarp.Rotate(angle)

    @classmethod
    def hflip(cls, config = None):
        prob = random.uniform(0.0, 1.0)
        if prob > 0.5:
            return kornia.geometry.transform.Hflip()
        
        return lambda x: x 
    
    @classmethod
    def vflip(cls, config = None):
        prob = random.uniform(0.0, 1.0)
        if prob > 0.5:
            return kornia.geometry.transform.Vflip()
        
        return lambda x: x

def main():
    config = OmegaConf.load('config.yaml')

    pipe = Transformation.get_pipe(config)
    image = Image.open('./data/1033.jpg')

    image = Transformation.prepare(image, requires_grad = True)
    loss = pipe(image)
    loss = loss.mean()
    loss.backward()

    print(image.grad)
    image = Transformation.to_image(image)
    image.show()

if __name__ == '__main__':
    main()