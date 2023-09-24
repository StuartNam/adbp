import os
import shutil

target_dir = 'outputs/ASPL/'

id_dirs = os.listdir(target_dir)

for id_dir in id_dirs:
    dir_path = f"{target_dir}/{id_dir}"
    dirs = os.listdir(dir_path)
    for dir in dirs:
        if dir == 'dreambooth' or dir == 'DREAMBOOTH':
            shutil.rmtree(f"{dir_path}/{dir}")