import os
import logging
import tqdm
import numpy as np
from deepface import DeepFace

class Accumulator:
    """
        Accumulator helps with accumulating scalar values. Useful for calculating scores
    """
    def __init__(self):
        self.total = 0
        self.n = 0

    def accumulate(self, value):
        self.total += value
        self.n += 1
    
    def average(self):
        if self.n == 0:
            raise RuntimeError("Accumulator.average(): No elements")
        
        result = self.total / self.n

        return result
    
class Metric:
    def __init__(self):
        pass

class FDFR(Metric):
    @classmethod
    def eval(cls, target_folder_path, log_info = False, enable_progress_bar = True):
        if log_info:
            print("FDFR.eval()")
            print(f"    target_folder_path = {target_folder_path}")
            print("----------")

        files = os.listdir(target_folder_path)
        num_files = len(files)
        num_failure_cases = 0

        if num_files == 0:
            return 0
        
        for file in tqdm.tqdm(files, desc = "FDFR.eval()", unit = "image"):
            image_path = os.path.join(target_folder_path, file)
            
            try:
                DeepFace.extract_faces(
                    img_path = image_path, 
                    align = True
                )
            except:
                num_failure_cases += 1

        fdfr = num_failure_cases / num_files

        if log_info:
            print(f"FDFR Score: {fdfr}")
            print()

        return fdfr
        
class ISM(Metric):
    @classmethod
    def eval(cls, target_folder_path, identity_folder_path, log_info = False, enable_progress_bar = True):
        if log_info:
            print("ISM.eval()")
            print(f"    target_folder_path = {target_folder_path}")
            print(f"    identity_folder_path = {identity_folder_path}")
            print("----------")

        target_files = os.listdir(target_folder_path)
        num_target_files = len(target_files)
        target_face_vectors = []
        num_face_target_files = num_target_files
        no_face_target_files = []

        for file in target_files:
            image_path = os.path.join(target_folder_path, file)

            try:
                face_embedding_info = DeepFace.represent(
                    img_path = image_path,
                    model_name = 'ArcFace'
                )   
            except:
                num_face_target_files -= 1
                no_face_target_files.append(image_path)
            else:
                face_vector = face_embedding_info[0]['embedding']
                target_face_vectors.append(face_vector)
        
        identity_files = os.listdir(identity_folder_path)
        num_identity_files = len(identity_files)
        identity_face_vectors = []
        num_face_identity_files = num_identity_files
        no_face_identity_files = []

        for file in identity_files:
            image_path = os.path.join(identity_folder_path, file)

            try:
                face_embedding_info = DeepFace.represent(
                    img_path = image_path,
                    model_name = 'ArcFace'
                )   
            except:
                num_face_identity_files -= 1
                no_face_identity_files.append(image_path)
            else:
                face_vector = face_embedding_info[0]['embedding']
                identity_face_vectors.append(face_vector)
        
        def average_vector(lst):
            """
                Return the average vector (list) of all vectors inside lst

                Args:
                - lst: list[list]
            """
            if not lst:
                return []

            num_lists = len(lst)
            num_elements = len(lst[0])

            sums = [0] * num_elements

            for sub_list in lst:
                for i, element in enumerate(sub_list):
                    sums[i] += element

            average = [total / num_lists for total in sums]

            return average

        average_identity_face_vector = average_vector(identity_face_vectors)

        def cosine_similarity(vector1, vector2):
            """
                Return the cosine similarity between 2 vectors, scaled to [0, 1]
            """
            dot_product = np.dot(vector1, vector2)
            norm_vector1 = np.linalg.norm(vector1)
            norm_vector2 = np.linalg.norm(vector2)
            similarity = dot_product / (norm_vector1 * norm_vector2)
            
            return (similarity + 1) / 2
        
        accumulator = Accumulator()

        progress_bar = target_face_vectors
        if enable_progress_bar:
            progress_bar = tqdm.tqdm(target_face_vectors, desc = "Score calculation", unit = "image")

        for target_face_vector in progress_bar:
            score = cosine_similarity(target_face_vector, average_identity_face_vector)
            accumulator.accumulate(score)

        ism = accumulator.total / num_target_files

        if log_info:
            print(f"Found {num_face_target_files} faces in {num_target_files} target_files")
            print(f"Found {num_face_identity_files} faces in {num_identity_files} identity_files")
            print("Files without faces: ")
            for file in no_face_target_files:
                print(f"- {file}")
            for file in no_face_identity_files:
                print(f"- {file}")
            
            print("----------")
            print(f"ISM Score: {ism}")
            print()

        return ism

class SER_FIQ(Metric):
    pass

class BRISQUE(Metric):
    pass

def parse_args(input_args = None):
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument( # --id )
        '--id',
        type = str,
        default = None,
        required = True,
        help = "Identity that you want to evaluate on"
    )

    parser.add_argument( # --log_into='logs/eval/log.log' )
        '--log_into',
        type = str,
        default = 'logs/eval/log.log',
        required = None,
        help = "Path to log file"
    )

    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    return args

def get_path(parameterized_path, parameters):
    """
        Return a true path from a parameterized path, given the parameters
    """
    parts = parameterized_path.split('/')

    for key, value in parameters:
        for i, part in enumerate(parts):
            if part == key:
                parts[i] = str(value)

    return '/'.join(parts)

def main(args):
    id = args.id
    log_into = args.log_into

    logging.basicConfig(
        filename = log_into,
        datefmt = None,
        level = logging.INFO
    )

    path_parameters = {
        '<id>': id
    }

    nodef_f1 = FDFR.eval(f'./dreambooth-outputs/{id}/checkpoint-1000/dreambooth/a_photo_of_sks_person')
    nodef_f2 = FDFR.eval(f'./dreambooth-outputs/{id}/checkpoint-1000/dreambooth/a_dslr_portrait_of_sks_person')
    nodef_i1 = ISM.eval(f'./dreambooth-outputs/{id}/checkpoint-1000/dreambooth/a_photo_of_sks_person', f'./db_dataset/{id}/set_A')
    nodef_i2 = ISM.eval(f'./dreambooth-outputs/{id}/checkpoint-1000/dreambooth/a_dslr_portrait_of_sks_person', f'./db_dataset/{id}/set_A')

    aspl_f1 = FDFR.eval(f'./outputs/ASPL/{id}/DREAMBOOTH/checkpoint-1000/dreambooth/a_photo_of_sks_person')
    aspl_f2 = FDFR.eval(f'./outputs/ASPL/{id}/DREAMBOOTH/checkpoint-1000/dreambooth/a_dslr_portrait_of_sks_person')
    aspl_i1 = ISM.eval(f'./outputs/ASPL/{id}/DREAMBOOTH/checkpoint-1000/dreambooth/a_photo_of_sks_person', f'./db_dataset/{id}/set_A')
    aspl_i2 = ISM.eval(f'./outputs/ASPL/{id}/DREAMBOOTH/checkpoint-1000/dreambooth/a_dslr_portrait_of_sks_person', f'./db_dataset/{id}/set_A')

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

if __name__ == '__main__':
    args = parse_args()
    main(args) 
