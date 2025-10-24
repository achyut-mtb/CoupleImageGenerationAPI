import numpy as np
import pandas as pd
import cv2

from itertools import permutations

# InsightFace
import insightface
from insightface.utils import face_align

from config import Config

class FaceProcessor:

    def __init__(self, 
                 providers=("CPUExecutionProvider",), 
                 use_cpu=True):
        
        ### setup insightface & arcface 
        ctx_id = -1 if use_cpu else 0
        self.app = insightface.app.FaceAnalysis(name="buffalo_l", providers=list(providers))
        self.app.prepare(ctx_id=ctx_id, det_size=(640, 640))
        self.arcface = insightface.model_zoo.get_model(Config.ARCFACE_MODEL_PATH, providers=list(providers))
        self.arcface.prepare(ctx_id=ctx_id)
        self.swapper = insightface.model_zoo.get_model(Config.INSWAPPER_MODEL_PATH, providers=list(providers))


    # ---------------- Utility Functions ---------------- #
    def get_embedding(self, face, img):
        face_img = face_align.norm_crop(img, landmark=face.kps).astype(np.uint8)
        emb = self.arcface.get_feat([face_img])[0]
        return emb / np.linalg.norm(emb)
    

    def match_faces(self, target_img, faces_target, ref1_img, ref2_img, ref1_face, ref2_face, min_sim=0.25):

        if len(faces_target) < 2:
            print("⚠️ Less than 2 faces detected in target. Cannot guarantee 1-to-1 match.")
            return {}
        
        ref1_emb = self.get_embedding(ref1_face, ref1_img)
        ref2_emb = self.get_embedding(ref2_face, ref2_img)
        target_embs = [self.get_embedding(tf, target_img) for tf in faces_target]

        sims = np.array([
            [np.dot(ref1_emb, t_emb) for t_emb in target_embs],
            [np.dot(ref2_emb, t_emb) for t_emb in target_embs],
        ])

        best_assignment, best_score = None, -1e9
        for perm in permutations(range(len(faces_target)), 2):
            score = sims[0, perm[0]] + sims[1, perm[1]]
            if score > best_score:
                best_score = score
                best_assignment = {"person1": perm[0], "person2": perm[1]}

        return best_assignment or {}


    def get_detected_faces(self, image):
        try:
            return self.app.get(image)
        
        except:
            return None
        

    def apply_face_swap(self, couple_image, couple_faces, ref1_img, ref2_img, ref1_face, ref2_face):

        try:
            match = self.match_faces(couple_image, couple_faces, ref1_img, ref2_img, ref1_face, ref2_face)
            print("Matches:", match)

            if "person1" in match:
                try:
                    couple_image = self.swapper.get(couple_image, couple_faces[match["person1"]], ref1_face, paste_back=True)

                except Exception as e:
                    print(f"⚠️ Man swap failed: {e}")

            if "person2" in match:
                try:
                    couple_image = self.swapper.get(couple_image, couple_faces[match["person2"]], ref2_face, paste_back=True)

                except Exception as e:
                    print(f"⚠️ Woman swap failed: {e}")

            print("Face swap successful")

            return couple_image
        
        except:
            return None
        
    def get_left_right_information(self, couple_image, ref1_face, ref2_face):
        return {}

        
    
        


