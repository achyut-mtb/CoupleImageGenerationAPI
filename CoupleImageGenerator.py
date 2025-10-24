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
from CoupleImagePostprocessor import CoupleImagePostprocessor
from FaceProcessor import FaceProcessor

from utils import cm_to_feet_inches, _pil_to_part, enforce_aspect

from config import Config

import time

class CoupleImageGenerator:

    def __init__(
        self,
        api_key,
        providers=("CPUExecutionProvider",),
        use_cpu=True,
        male_csv="resources/Hinge_Profile_Data - Combined - Male.csv",
        female_csv="resources/Hinge_Profile_Data - Combined - Female.csv",
        prompts_dir=Config.PROMPTS_DIR
    ):
        self.client = genai.Client(api_key=api_key)

        self.prompts_dir = prompts_dir

        ## pose extractor 
        self.pose_extractor = PoseExtractor()

        ## postprocessor
        self.postprocessor = CoupleImagePostprocessor(api_key=api_key)

        ## face processor
        self.face_processor = FaceProcessor(providers=providers, use_cpu=use_cpu)

        # CSVs (for height/ethnicity)
        self.male_csv = pd.read_csv(male_csv) if os.path.exists(male_csv) else pd.DataFrame()
        self.female_csv = pd.read_csv(female_csv) if os.path.exists(female_csv) else pd.DataFrame()
                

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

    def get_prompts(self, type_couple, p1_height_cm, p2_height_cm, background_desc, background_light):

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

            prompt = prompt.replace("{{first_person_height_cm}}", str(p1_height_cm))
            prompt = prompt.replace("{{second_person_height_cm}}", str(p2_height_cm))
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
        pose_image_path = Config.POSE_IMAGE_PATH,
        canvas_width=1024,
        canvas_height=2200,
        fov_v_deg=40.0,
        depth_m=3.0,
        overlap_ratio=0.06,
        p1_img_path=None,
        p2_img_path=None,
        background_desc="A simple plain background",
        background_light="Daylight, light source present in front of persons",
        p1_height_cm="Not given, estimate from photo",
        p2_height_cm="Not given, estimate from photo",
        type_couple="straight" ## -- VALUES: ["straight", "lesbian", "gay"]
    ):
        
        """
            Arguments: 
                1. temp_dir: A root directory in which new directory will be created to store temporary files 
                2. pair_label: A unique identifier for pair 
                3. p1_img_path: Path of first person's image 
                4. p2_img_path: Path of second person's image 
                5. p1_height_cm: Height of first person
                6. p2_height_cm: Height of second person
                7. background_desc: Type of background needs to be in couple photo
                8. background_light: In what lightning condition, persons should be present
                9. type_couple: Is couple: straight, gay, lesbian or trans ?? 
        """

        try:
            pair_dir = os.path.join(temp_dir, pair_label)
            os.makedirs(pair_dir, exist_ok=True)

            print("Loading images .. ")
            print(p1_img_path, p2_img_path, end='\n')

            start = time.time()
            p1_img = cv2.imread(p1_img_path)
            p2_img = cv2.imread(p2_img_path)

            print("Time taken in image loading:", (time.time()-start))

            if p1_img is None or p2_img is None:
                raise RuntimeError("Missing full.jpg in men/women dirs")


            print("Getting face ..")

            p1_faces = self.face_processor.get_detected_faces(p1_img)
            p2_faces = self.face_processor.get_detected_faces(p2_img)

            if not p1_faces or not p2_faces:
                raise RuntimeError("No face detected in reference images")
            p1_face, p2_face = p1_faces[0], p2_faces[0]

            # print("Time taken till face detection:", (time.time()-start))

            ### Pose map creation 
            pose_path = os.path.join(pair_dir, "pose_skeleton.png")
            # pose_img = self.pose_extractor.extract_skeleton_pose(pose_image_path, pose_path)

            if type(p1_height_cm) == str:
                p1_height_cm = 170

            if type(p2_height_cm) == str: 
                p2_height_cm = 170

            print("Extracting pose .. ")

            pose_img = self.pose_extractor.extract_and_scale_pose(pose_image_path, 
                                                                  pose_path, 
                                                                  man_height_cm=p1_height_cm,
                                                                  woman_height_cm=p2_height_cm)

            # pose_img = self.pose_extractor.get_openpose_skeleton(pose_image_path, pose_path)

            if pose_img is None: 
                raise RuntimeError("Pose extraction failed")
            
            print("Loading prompt .. ")

            prompt, system_prompt = self.get_prompts(type_couple,
                                                     p1_height_cm,
                                                     p2_height_cm,
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
                _pil_to_part(Image.open(p1_img_path)),
                _pil_to_part(Image.open(p2_img_path)),
                _pil_to_part(Image.open(pose_path))
            ]

            print("Calling model .. ")

            contents = [types.Content(role="user", parts=parts)]
            response = self.client.models.generate_content(
                model=Config.IMAGE_GEN_MODEL_NAME,
                contents=contents,
                config=generate_content_config
            )

            # print("Time taken till first response generation:", (time.time()-start))

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
            faces_target = self.face_processor.get_detected_faces(target)
            print("Number of faces: ", len(faces_target))

            if len(faces_target) >= 2:
                target = self.face_processor.apply_face_swap(target,
                                                             faces_target,
                                                             p1_img,
                                                             p2_img, 
                                                             p1_face,
                                                             p2_face)
                
                if target is None: ## if face swap is unsuccessful
                    return False

            else: ## if two persons are not in the output image, return false for retry 
                return False

            # print("Time taken till face swap:", (time.time()-start))

            # Save
            preswapped_path = os.path.join(pair_dir, "raw_output.png")
            postswapped_path = os.path.join(pair_dir, "postswapped_output.png")
            cv2.imwrite(preswapped_path, pre_swapping)
            cv2.imwrite(postswapped_path, target)
            
            ### get left right information 
            
            faces_target = self.face_processor.get_detected_faces(target)
            left_right_info = self.face_processor.get_left_right_information(target, 
                                                                             faces_target, 
                                                                             p1_img, 
                                                                             p2_img,
                                                                             p1_face, 
                                                                             p2_face)
            
            print("Left right information:", left_right_info)

            # print("Time taken till image saving:", (time.time()-start))


            ### Postprocessing
            
            ## Step 1 -- Centerally align people  --- ### 
            postprocessed_path = os.path.join(pair_dir, "postprocessed_output.png")
            status = self.postprocessor.align_person_center_with_headspace(postswapped_path, postprocessed_path)

            # print("Time taken till step 1 postprocessing:", (time.time()-start))

            if not status:
                raise RuntimeError("Some error in postprocessing .. ")
            
            ## Step 2 -- Generate background -- ### 
            final_path = os.path.join(pair_dir, "final_output.png")
            status = self.postprocessor.generate_background(postprocessed_path, final_path, background_desc, background_light)

            if not status:
                raise RuntimeError("Some error in generating background .. ")
            
            # print("Time taken till step 2 postprocessing:", (time.time()-start))
        
            return True

        except Exception as e:
            print(f"Error in generate_couple_photo: {e}")
            return False