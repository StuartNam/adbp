import os
import numpy as np
from deepface import DeepFace
from tqdm import tqdm

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
    def eval(cls, target_dir, log_info = False, enable_progress_bar = True):
        if log_info:
            print("FDFR.eval()")
            print(f"    target_dir = {target_dir}")
            print("----------")

        files = os.listdir(target_dir)
        num_files = len(files)
        num_failure_cases = 0

        if num_files == 0:
            return 0
        
        for file in tqdm(files, desc = "FDFR.eval()", unit = "image"):
            image_path = os.path.join(target_dir, file)
            
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
    def eval(cls, target_dir, identity_dir, log_info = False, enable_progress_bar = True):
        if log_info:
            print("ISM.eval()")
            print(f"    target_dir = {target_dir}")
            print(f"    identity_dir = {identity_dir}")
            print("----------")

        target_files = os.listdir(target_dir)
        num_target_files = len(target_files)
        target_face_vectors = []
        num_face_target_files = num_target_files
        no_face_target_files = []

        for file in target_files:
            image_path = os.path.join(target_dir, file)

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
        
        identity_files = os.listdir(identity_dir)
        num_identity_files = len(identity_files)
        identity_face_vectors = []
        num_face_identity_files = num_identity_files
        no_face_identity_files = []

        for file in identity_files:
            image_path = os.path.join(identity_dir, file)

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
            progress_bar = tqdm(target_face_vectors, desc = "Score calculation", unit = "image")

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
    @classmethod
    def eval(self, target_dir):
        return "Not implemented"

class BRISQUE(Metric):
    @classmethod
    def eval(self, target_dir):
        return "Not implemented"
