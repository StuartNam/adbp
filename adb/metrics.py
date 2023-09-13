import os
import shutil

import tqdm
import numpy as np
import torch

from deepface import DeepFace
from retinaface import RetinaFace

import logging

logging.basicConfig(
    filename = './eval/eval.log'
)

id = '17'

class Accumulator:
    # Accumulator helps with accumulating scalar values. Useful for calculating scores
    def __init__(self):
        self.total = 0
        self.n = 0

    def accumulate(self, value):
        self.total += value
        self.n += 1
    
    def average(self):
        if self.n == 0:
            return 0
        
        result = self.total / self.n

        return result
    
class Metric:
    def __init__(self):
        pass

class FDFR(Metric):
    @classmethod
    def eval(cls, target_folder_path):
        files = os.listdir(target_folder_path)
        num_files = len(files)
        num_failure_cases = 0

        if num_files == 0:
            return 0
        
        for file in tqdm.tqdm(files, desc = "FDFR.eval()", unit = "image"):
            image_path = os.path.join(target_folder_path, file)
            
            # Find faces in file using RetinaFace
            # face_objs = RetinaFace.extract_faces(
            #     img_path = image_path, 
            #     align = True
            # )

            try:
                # print(image_path)
                face_objs = DeepFace.extract_faces(
                    img_path = image_path, 
                    align = True
                )

            # If there are no faces
            except Exception as e:
                num_failure_cases += 1

        fdfr = num_failure_cases / num_files

        return fdfr
        
class ISM(Metric):
    @classmethod
    def eval(cls, target_folder_path, identity_folder_path):
        target_files = os.listdir(target_folder_path)

        target_face_vectors = []
        for file in target_files:
            image_path = os.path.join(target_folder_path, file)

            try:
                face_embedding_info = DeepFace.represent(
                    img_path = image_path,
                    model_name = 'ArcFace'
                )   
            except Exception as e:
                pass
                # print(f"{image_path}: {e}")
            else:
                face_vector = face_embedding_info[0]['embedding']
                target_face_vectors.append(face_vector)
        
        identity_face_vectors = []

        identity_files = os.listdir(identity_folder_path)

        for file in identity_files:
            image_path = os.path.join(identity_folder_path, file)

            try:
                face_embedding_info = DeepFace.represent(
                    img_path = image_path,
                    model_name = 'ArcFace'
                )   
            except Exception as e:
                pass
                # print(f"{image_path}: {e}")
            else:
                face_vector = face_embedding_info[0]['embedding']
                identity_face_vectors.append(face_vector)
        
        # print(identity_face_vectors)
        def elementwise_avg(lst):
            if not lst:
                return []  # Handle an empty input list

            num_lists = len(lst)
            num_elements = len(lst[0])  # Assuming all sublists have the same length

            # Initialize a list to store the sums of elements
            sums = [0] * num_elements

            # Calculate the sum of elements in each position
            for sub_list in lst:
                for i, element in enumerate(sub_list):
                    sums[i] += element

            # Calculate the average of each position
            average = [total / num_lists for total in sums]

            return average

        average_identity_face_vector = elementwise_avg(identity_face_vectors)

        def cosine_similarity(vector1, vector2):
            dot_product = np.dot(vector1, vector2)
            norm_vector1 = np.linalg.norm(vector1)
            norm_vector2 = np.linalg.norm(vector2)
            similarity = dot_product / (norm_vector1 * norm_vector2)
            
            return (similarity + 1) / 2

        # big_accumulator = Accumulator()
        # for target_face_vector in tqdm.tqdm(target_face_vectors, desc = "ISM.eval()", unit = "image"):  
        #     small_accumulator = Accumulator()

        #     for identity_face_vector in identity_face_vectors:
        #         score = cosine_similarity(target_face_vector, identity_face_vector)
        #         small_accumulator.accumulate(score)

        #     big_accumulator.accumulate(small_accumulator.average())

        accumulator = Accumulator()
        for target_face_vector in tqdm.tqdm(target_face_vectors, desc = "ISM.eval()", unit = "image"):
            score = cosine_similarity(target_face_vector, average_identity_face_vector)
            accumulator.accumulate(score)

        ism = accumulator.total / len(target_files)
        # ism = big_accumulator.average()

        return ism

nodef_f1 = FDFR.eval(f'./dreambooth-outputs/{id}/checkpoint-1000/dreambooth/a_photo_of_sks_person')
nodef_f2 = FDFR.eval(f'./dreambooth-outputs/{id}/checkpoint-1000/dreambooth/a_dslr_portrait_of_sks_person')
nodef_i1 = ISM.eval(f'./dreambooth-outputs/{id}/checkpoint-1000/dreambooth/a_photo_of_sks_person', f'./db_dataset/{id}/set_A')
nodef_i2 = ISM.eval(f'./dreambooth-outputs/{id}/checkpoint-1000/dreambooth/a_dslr_portrait_of_sks_person', f'./db_dataset/{id}/set_A')

# aspl_f1 = FDFR.eval(f'./outputs/ASPL/{id}/DREAMBOOTH/checkpoint-1000/dreambooth/a_photo_of_sks_person')
# aspl_f2 = FDFR.eval(f'./outputs/ASPL/{id}/DREAMBOOTH/checkpoint-1000/dreambooth/a_dslr_portrait_of_sks_person')
# aspl_i1 = ISM.eval(f'./outputs/ASPL/{id}/DREAMBOOTH/checkpoint-1000/dreambooth/a_photo_of_sks_person', f'./db_dataset/{id}/set_A')
# aspl_i2 = ISM.eval(f'./outputs/ASPL/{id}/DREAMBOOTH/checkpoint-1000/dreambooth/a_dslr_portrait_of_sks_person', f'./db_dataset/{id}/set_A')

aspl_f1 = FDFR.eval(f'./outputs/ASPL_TEST/{id}/DREAMBOOTH/checkpoint-1000/dreambooth/a_photo_of_sks_person')
aspl_f2 = FDFR.eval(f'./outputs/ASPL_TEST/{id}/DREAMBOOTH/checkpoint-1000/dreambooth/a_dslr_portrait_of_sks_person')
aspl_i1 = ISM.eval(f'./outputs/ASPL_TEST/{id}/DREAMBOOTH/checkpoint-1000/dreambooth/a_photo_of_sks_person', f'./db_dataset/{id}/set_A')
aspl_i2 = ISM.eval(f'./outputs/ASPL_TEST/{id}/DREAMBOOTH/checkpoint-1000/dreambooth/a_dslr_portrait_of_sks_person', f'./db_dataset/{id}/set_A')

print(f"Set {id}: No defense")
print("- FDFR score")
print(f". a photo of sks person: {nodef_f1}")
print(f". a DSLR portrait of sks person: {nodef_f2}")
print("- ISM score")
print(f". a photo of sks person: {nodef_i1}")
print(f". a DSLR portrait of sks person: {nodef_i2}")

print(f"Set {id}: ASPL attacked")
print("- FDFR score")
print(f". a photo of sks person: {aspl_f1}")
print(f". a DSLR portrait of sks person: {aspl_f2}")
print("- ISM score")
print(f". a photo of sks person: {aspl_i1}")
print(f". a DSLR portrait of sks person: {aspl_i2}")

        
