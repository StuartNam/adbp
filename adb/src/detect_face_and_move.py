import os
import numpy as np
import shutil
from deepface import DeepFace


root = 'outputs/DB'

def remove_contents(path):
    for root, dirs, files in os.walk(path, topdown = False):
        for file in files:
            file_path = os.path.join(root, file)
            os.remove(file_path)
        for dir in dirs:
            dir_path = os.path.join(root, dir)
            shutil.rmtree(dir_path)

def main():
    id_dirs = os.listdir(root)
    for id_dir in id_dirs:
        target_dir = f'outputs/DB/{id_dir}/checkpoint-1000/images/'
        prompt_dirs = os.listdir(target_dir)
        for prompt_dir in prompt_dirs:
            image_dir = f"{target_dir}/{prompt_dir}"

            all_dir = f"{image_dir}/all"
            os.makedirs(all_dir, exist_ok = True)
            remove_contents(all_dir)

            non_face_dir = f"{image_dir}/non-faces"
            face_dir = f"{image_dir}/faces"

            os.makedirs(non_face_dir, exist_ok = True)
            remove_contents(non_face_dir)

            os.makedirs(face_dir, exist_ok = True)
            remove_contents(face_dir)

            target_files = os.listdir(image_dir)

            for file in target_files:
                image_path = os.path.join(image_dir, file)
                
                if os.path.isdir(image_path):
                    continue
                
                shutil.copy(image_path, all_dir)

                try:
                    DeepFace.extract_faces(
                        img_path = image_path,
                        align = True
                    )

                    # DeepFace.represent(
                    #     img_path = image_path,
                    #     model_name = 'ArcFace'
                    # )

                    shutil.copy(image_path, face_dir)

                except:
                    shutil.copy(image_path, non_face_dir)
                
                # os.remove(image_path)


if __name__ == '__main__':
    main()