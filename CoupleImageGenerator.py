import os
import cv2
import json
import copy
import math  # === NEW ===
import random
import shutil
import argparse 
import numpy as np
import pandas as pd
from io import BytesIO
from PIL import Image

from rembg import remove

# Google Gemini SDK (acts as Nano-Banana/i2i host in this script)
import google.genai as genai
from google.genai import types

# InsightFace
import insightface
from insightface.utils import face_align

from typing import Optional, Tuple, Dict
from itertools import permutations

from PoseExtractor import PoseExtractor

from utils import cm_to_feet_inches, _pil_to_part, enforce_aspect

class CoupleImageGenerator:

    def __init__(
        self,
        api_key,
        pose_dir="pose_images",
        interests_csv="resources/Bgs - Interests - Image Descriptions.csv",
        providers=("CPUExecutionProvider",),
        use_cpu=True,
        male_csv="resources/Hinge_Profile_Data - Combined - Male.csv",
        female_csv="resources/Hinge_Profile_Data - Combined - Female.csv",
        prompts_dir="prompts"
    ):
        self.client = genai.Client(api_key=api_key)

        self.prompts_dir = prompts_dir

        ctx_id = -1 if use_cpu else 0
        self.app = insightface.app.FaceAnalysis(name="buffalo_l", providers=list(providers))
        self.app.prepare(ctx_id=ctx_id, det_size=(640, 640))
        self.arcface = insightface.model_zoo.get_model("models/w600k_r50.onnx", providers=list(providers))
        self.arcface.prepare(ctx_id=ctx_id)
        self.swapper = insightface.model_zoo.get_model("models/inswapper_128.onnx", providers=list(providers))

        self.pose_extractor = PoseExtractor()
        self.pose_dir = pose_dir
        self.last_valid_pose = None
        self.last_pose_path = None

        # Interests
        self.interests = None
        if os.path.exists(interests_csv):
            try:
                self.interests = pd.read_csv(interests_csv)
            except Exception as e:
                print(f"⚠️ Failed to load interests CSV: {e}")
        if self.interests is None or "Image Description" not in (self.interests.columns if self.interests is not None else []):
            self.interests = pd.DataFrame([{
                "Image Description": "simple indoor studio background, soft gradient wall, neutral tones",
                "Light": "soft, diffused, even lighting"
            }])

        # CSVs (for height/ethnicity)
        self.male_csv = pd.read_csv(male_csv) if os.path.exists(male_csv) else pd.DataFrame()
        self.female_csv = pd.read_csv(female_csv) if os.path.exists(female_csv) else pd.DataFrame()

        os.makedirs("resources/canvas", exist_ok=True)


    # ---------------- Utility Functions ---------------- #
    def get_embedding(self, face, img):
        face_img = face_align.norm_crop(img, landmark=face.kps).astype(np.uint8)
        emb = self.arcface.get_feat([face_img])[0]
        return emb / np.linalg.norm(emb)

    def match_faces(self, faces_target, ref1_img, ref2_img, ref1_face, ref2_face, target_img, min_sim=0.25):
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

        # if best_assignment is not None:
        #     if sims[0, best_assignment["person1"]] < min_sim or sims[1, best_assignment["person2"]] < min_sim:
        #         print("⚠️ Similarity too low for reliable assignment. Skipping enforced mapping.")
        #         return {}
        return best_assignment or {}


    # Simplified: pull by "Sl. no." only  === NEW ===
    def get_height_ethnicity_by_id(self, gender: str, slno) -> tuple:
        try:
            slno = int(slno)
        except:
            return None, "Not Known"
        df = self.male_csv if gender.lower().startswith("m") else self.female_csv
        if df.empty or "Sl. no." not in df.columns:
            return None, "Not Known"
        row = df[df["Sl. no."] == slno]
        if row.empty:
            return None, "Not Known"
        r = row.iloc[0]
        height_cm = None
        if "Height_CM" in r and pd.notna(r["Height_CM"]):
            try:
                height_cm = float(r["Height_CM"])
            except:
                height_cm = None
        ethnicity = r["Ethnicity"] if "Ethnicity" in r and pd.notna(r["Ethnicity"]) else "Not Known"
        return height_cm, ethnicity

    def get_prompts(self, type_couple, man_height_cm, woman_height_cm, background_desc, background_light):

        try:
            ## load system prompt 
            system_prompt_filename = os.path.join(self.prompts_dir, type_couple + "_system_prompt.txt")
            system_prompt = ""
            with open(system_prompt_filename, "r") as f:
                system_prompt = f.read()
            

            ## load prompt
            prompt_filename = os.path.join(self.prompts_dir, type_couple + "_prompt.txt")
            prompt = ""
            with open(prompt_filename, "r") as f:
                prompt = f.read()

            prompt = prompt.replace("{{first_person_height_cm}}", str(man_height_cm))
            prompt = prompt.replace("{{second_person_height_cm}}", str(woman_height_cm))
            prompt = prompt.replace("{{background_desc}}", str(background_desc))
            prompt = prompt.replace("{{background_light}}", str(background_light))

            return prompt, system_prompt
        
        except:
            print("Prompts can not be loaded ..")
            return "", ""

    def generate_couple_photo(
        self,
        temp_dir="temp",
        pair_label="pair",
        pose_image_path = "pose_images/couple_image.jpg",
        canvas_width=1024,
        canvas_height=2200,
        fov_v_deg=40.0,
        depth_m=3.0,
        overlap_ratio=0.06,
        man_single_img_path=None,
        woman_single_img_path=None,
        background_desc="A simple plain background",
        background_light="Daylight, light source present in front of persons",
        man_height_cm="Not given, estimate from photo",
        woman_height_cm="Not given, estimate from photo",
        type_couple="straight" ## -- VALUES: ["straight", "lesbian", "gay"]
    ):
        
        """
            Arguments: 
                1. temp_dir: A root directory in which new directory will be created to store temporary files 
                2. pair_label: A unique identifier for pair 
                3. man_single_img_path: Path of first person's image 
                4. woman_single_img_path: Path of second person's image 
                5. man_height_cm: Height of first person
                6. woman_height_cm: Height of second person
                7. background_desc: Type of background needs to be in couple photo
                8. background_light: In what lightning condition, persons should be present
                9. type_couple: Is couple: straight, gay, lesbian or trans ?? 
        """

        try:
            pair_dir = os.path.join(temp_dir, pair_label)
            os.makedirs(pair_dir, exist_ok=True)

            print("Loading images .. ")
            print(man_single_img_path, woman_single_img_path, end='\n')

            # === Identity reference (full.jpg) ===
            # man_img   = cv2.imread(os.path.join(men_dir, "full.jpg"))
            # woman_img = cv2.imread(os.path.join(women_dir, "full.jpg"))
            man_img = cv2.imread(man_single_img_path)
            woman_img = cv2.imread(woman_single_img_path)

            if man_img is None or woman_img is None:
                raise RuntimeError("Missing full.jpg in men/women dirs")


            print("Getting face ..")

            man_faces = self.app.get(man_img)
            woman_faces = self.app.get(woman_img)
            if not man_faces or not woman_faces:
                raise RuntimeError("No face detected in reference images")
            man_face, woman_face = man_faces[0], woman_faces[0]

            ### Pose map creation 
            pose_path = os.path.join(pair_dir, "pose_skeleton.png")
            # pose_img = self.pose_extractor.extract_skeleton_pose(pose_image_path, pose_path)

            if type(man_height_cm) == str:
                man_height_cm = 170

            if type(woman_height_cm) == str: 
                woman_height_cm = 170

            print("Extracting pose .. ")

            pose_img = self.pose_extractor.extract_and_scale_pose(pose_image_path, 
                                                                  pose_path, 
                                                                  man_height_cm=man_height_cm,
                                                                  woman_height_cm=woman_height_cm)

            # pose_img = self.pose_extractor.get_openpose_skeleton(pose_image_path, pose_path)

            if pose_img is None: 
                raise RuntimeError("Pose extraction failed")
            
            print("Loading prompt .. ")

            prompt, system_prompt = self.get_prompts(type_couple,
                                                     man_height_cm,
                                                     woman_height_cm,
                                                     background_desc,
                                                     background_light)
            
            if prompt == "" or system_prompt == "":
                raise RuntimeError("Prompts can not be loaded .. ")

            generate_content_config = types.GenerateContentConfig(
                system_instruction=[
                        types.Part.from_text(text=system_prompt),
                    ]
            )

            parts = [
                types.Part(text=prompt),
                _pil_to_part(Image.open(man_single_img_path)),
                _pil_to_part(Image.open(woman_single_img_path)),
                _pil_to_part(Image.open(pose_path))
            ]

            print("Calling model .. ")

            contents = [types.Content(role="user", parts=parts)]
            response = self.client.models.generate_content(
                model="gemini-2.5-flash-image",
                contents=contents,
                config=generate_content_config
            )

            pre_swapping = None
            for cand in getattr(response, "candidates", []):
                for prt in cand.content.parts:
                    if getattr(prt, "inline_data", None):
                        buf = np.frombuffer(prt.inline_data.data, np.uint8)
                        im = cv2.imdecode(buf, cv2.IMREAD_COLOR)
                        if im is not None:
                            pre_swapping = im
                            break
                if pre_swapping is not None:
                    break
            if pre_swapping is None:
                raise RuntimeError("❌ No valid couple image generated")
            

            print("Face swap .. ")

            # === Step 6: Face swap ===
            target = pre_swapping.copy()
            faces_target = self.app.get(target)
            print("Number of faces: ", len(faces_target))

            if len(faces_target) >= 2:
                match = self.match_faces(faces_target, man_img, woman_img, man_face, woman_face, target)
                print("Matches:", match)

                if "person1" in match:
                    try:
                        target = self.swapper.get(target, faces_target[match["person1"]], man_face, paste_back=True)
                    except Exception as e:
                        print(f"⚠️ Man swap failed: {e}")

                if "person2" in match:
                    try:
                        target = self.swapper.get(target, faces_target[match["person2"]], woman_face, paste_back=True)
                    except Exception as e:
                        print(f"⚠️ Woman swap failed: {e}")

            # Save
            raw_path = os.path.join(pair_dir, "raw_output.png")
            final_path = os.path.join(pair_dir, "couple_final.png")
            cv2.imwrite(raw_path, pre_swapping)
            cv2.imwrite(final_path, target)

            return pre_swapping, target, man_img, woman_img, pose_path

        except Exception as e:
            print(f"Error in generate_couple_photo: {e}")
            return None, None, None, None, None

if __name__ == "__main__":


    person_dir = "Set_2_body_face/Men/18-22/1/"
    role_label = "man"
    person_id = 1
    man_height_cm = 170
    ethnicity = "American"
    use_pose = False

    api_key = "AIzaSyBlNNUl7RYCx1jsWL8WLOQa-7CAVP2_4X4"