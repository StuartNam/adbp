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

        for method in config.methods:
            if method.name == 'gaussian_blur':
                methods.append(Transformation.gaussian_blur_2d(method.params)) 
            elif method.name == 'gaussian_noisify':
                methods.append(Transformation.gaussian_noisify(method.params))
            elif method.name == 'color_convert':
                methods.append(Transformation.color_convert(method.params))
            elif method.name == 'rotate':
                methods.append(Transformation.rotate(method.params)) 
            elif method.name == 'hflip':
                methods.append(Transformation.hflip(method.params))     
            elif method.name == 'vflip':
                methods.append(Transformation.vflip(method.params))
            elif method.name == 'motion_blur':
                methods.append(Transformation.motion_blur(method.params))

        if config.chained:
            return chain(methods)  
        
        return methods
    
    @classmethod
    def get_pipe_name(cls, config):
        name = ''
        for method in config.methods:
            if method.name == 'gaussian_blur':
                name += 'gb-'
            elif method.name == 'gaussian_noisify':
                name += 'gn-'
            elif method.name == 'color_convert':
                name += 'cc-'
            elif method.name == 'rotate':
                name += 'rt-'
            elif method.name == 'hflip':
                name += 'hf-'
            elif method.name == 'vflip':
                name += 'vf-'
            elif method.name == 'motion_blur':
                name += 'mb-'

        return name

    @classmethod
    def gaussian_blur_2d(cls, params = None):
        """
        Performing the gaussian blur with kernel_size and sigma
        Args:
        - params: {'kernel_size': ..., 'sigma': ...}
        """

        if params.kernel_size == 'random':
            kernel_size = random.choice([3, 5, 7, 9, 11])
        else:
            try:
                kernel_size = int(params.kernel_size)
            except:
                raise RuntimeError("gaussian_blur_2d(): Kernel size must be able to convert to int")

        if params.sigma == 'random':
            sigma = random.uniform(1.0, 2.0)
        else:
            try:
                sigma = float(params.sigma)
            except:
                raise RuntimeError("gaussian_blur_2d(): Sigma must be able to convert to float")

        return kornia.filters.GaussianBlur2d(
            kernel_size = (kernel_size, kernel_size),
            sigma = (sigma, sigma)
        )

    @classmethod
    def gaussian_noisify(cls, params = None):
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
    def color_convert(cls, params = None):
        prob = random.uniform(0.0, 1.0)
        if prob > 0.5:
            return kornia.color.RgbToBgr() 
        
        return kornia.color.BgrToRgb()
    
    @classmethod
    def rotate(cls, params = None):
        """
        Rotate clock-wise
        Args:
            - bs: batch_size
            - angle: if none -> random angle for define the transform funtion
        """

        if params.angle == 'random':
            angle = random.randint(0, 360)
            print(f"Random angle = {angle}")

        if not isinstance(params.angle, torch.Tensor):
            try:
                angle = torch.ones(params.batch_size, dtype = torch.float32) * int(params.angle)
            except:
                raise RuntimeError("Transformation.rotate(): Batch size is required")

        return kornia.geometry.transform.affwarp.Rotate(angle)

    @classmethod
    def hflip(cls, params = None):
        prob = random.uniform(0.0, 1.0)
        if prob > 0.5:
            return kornia.geometry.transform.Hflip()
        
        return lambda x: x 
    
    @classmethod
    def vflip(cls, params = None):
        prob = random.uniform(0.0, 1.0)
        if prob > 0.5:
            return kornia.geometry.transform.Vflip()
        
        return lambda x: x

    @classmethod
    def motion_blur(cls, params = None):
        return kornia.filters.MotionBlur(
            kernel_size = params.kernel_size,
            angle = params.angle,
            direction = params.direction
        )

    @classmethod
    def enhance_brightness(cls, params = None):
        pass
    
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
