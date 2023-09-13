import os
import argparse
from tqdm import tqdm
from omegaconf import OmegaConf
from transformation import Transformation
from PIL import Image

def parse_args(input_args = None):
    parser = argparse.ArgumentParser()

    parser.add_argument( # --target_dir )
        '--target_dir',
        type = str,
        default = None,
        required = True,
        help = "Target folder where the transformation is applied"
    )

    parser.add_argument( # --output_dir )
        '--output_dir',
        type = str,
        default = None,
        required = True,
        help = "Output folder where the transformed images are saved"
    )
    
    parser.add_argument( # --config='config/transform.yaml' )
        '--config',
        type = str,
        default = 'config/transform.yaml',
        required = False,
        help = ".yaml file containing transformation definition"
    )

    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    return args

def main(args):
    target_dir = args.target_dir
    output_dir = args.output_dir
    config = args.config

    config = OmegaConf.load(config)

    if not os.path.exists(target_dir): # target_dir must exist
        raise RuntimeError("transform.py - main(): Target_dir doesnot exist. It must exist.")
    
    target_filenames = os.listdir(target_dir) # Get all filenames inside target_dir

    pipename = Transformation.get_pipe_name(config)
    output_dir = os.path.join(output_dir, pipename) # Make the output_dir if not exist yet
    os.makedirs(output_dir, exist_ok = True)

    pipe = Transformation.get_pipe(config)
    
    progress_bar = tqdm(target_filenames, desc = 'Transform', unit = 'image')
    for target_filename in progress_bar:
        target_file = os.path.join(target_dir, target_filename)
        image = Image.open(target_file)
        image = Transformation.prepare(image)
        transformed_image = pipe(image)
        transformed_image = Transformation.to_image(transformed_image)
        
        output_file = os.path.join(output_dir, target_filename)
        transformed_image.save(output_file)
        
if __name__ == '__main__':
    args = parse_args()
    main(args)
