import cv2
import numpy as np
from rembg import remove
import os
from PIL import Image
from google import genai
import mediapipe as mp
from google.genai import types

from utils import _pil_to_part

from config import CONFIG

class CoupleImagePostprocessor:

    def __init__(self, api_key):

        self.client = genai.Client(api_key=api_key)

        ## minimum area to detect humans 
        self.min_component_area = 2000

    def load_prompt(self, background_desc, background_light):

        ### load prompt
        prompt = ""
        with open(os.path.join(CONFIG["PROMPTS_DIR"], "couple_image_postprocessing_prompt.txt"), 'r') as f:
            prompt = f.read()

        prompt = prompt.replace("[[background_desc]]", background_desc)
        prompt = prompt.replace("[[background_light]]", background_light)

        return prompt
        

    def get_ankle_crop_y(self, rgba_image):

        """
        Detect ankle points using MediaPipe Pose and return the Y coordinate
        (in pixels) to crop the image till that point.
        """
        mp_pose = mp.solutions.pose
        pose = mp_pose.Pose(static_image_mode=True, model_complexity=2)
        image_rgb = cv2.cvtColor(rgba_image[:, :, :3], cv2.COLOR_BGR2RGB)
        results = pose.process(image_rgb)

        pose.close()

        if not results.pose_landmarks:
            print("⚠️ No pose detected; skipping ankle cropping.")
            return rgba_image.shape[0]  # full height fallback

        h, w = rgba_image.shape[:2]
        landmarks = results.pose_landmarks.landmark

        # Get left & right ankle normalized coordinates (0–1)
        left_ankle_y = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE].y * h
        right_ankle_y = landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE].y * h

        # Take the max (lower) ankle point
        ankle_y = int(max(left_ankle_y, right_ankle_y))

        # # Remove small margin (5% of height) if needed
        ankle_y = min(ankle_y - int(CONFIG["ANKLE_MARGIN"] * h), h)

        return ankle_y
    

    def align_person_center_with_headspace(self, 
                                           input_path, 
                                           output_path,
                                           canvas_size=(1800, 2400),
                                           headspace_ratio=CONFIG["HEADSPACE_VALUE"],
                                           bottom_ratio=CONFIG["BOTTOMSPACE_VALUE"]):
        
        """
        Uses rembg + PIL to remove background, then creates a new
        1800x2400 white canvas with 16% headspace & cropped till ankles.
        """

        try:
            # --- Step 1: Load and remove background with rembg ---
            input_image = Image.open(input_path).convert("RGBA")
            output_image = remove(input_image)
            rgba = np.array(output_image)

            if rgba.shape[2] != 4:
                raise RuntimeError("Expected RGBA after background removal")

            # --- Step 2: Extract bounding box of foreground ---

            alpha = rgba[:, :, 3]
            binary_mask = (alpha > 10).astype(np.uint8) * 255

            # Find connected components
            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)
            clean_mask = np.zeros_like(binary_mask)

            for i in range(1, num_labels):  # skip background
                area = stats[i, cv2.CC_STAT_AREA]
                if area >= self.min_component_area:
                    clean_mask[labels == i] = 255

            coords = cv2.findNonZero(clean_mask)
            if coords is None:
                raise RuntimeError("No visible person found after cleaning mask")

            x, y, w, h_box = cv2.boundingRect(coords)
            y2 = y + h_box
            alpha = rgba[:, :, 3]
            coords = cv2.findNonZero(alpha)
            if coords is None:
                raise RuntimeError("No visible foreground found in image")

            x, y, w, h = cv2.boundingRect(coords)
            cropped = rgba[y:y+h, x:x+w]

            # --- Step 3: Detect ankle position & crop till ankles ---
            ankle_y = self.get_ankle_crop_y(cropped)
            cropped = cropped[:ankle_y, :, :]  # crop till ankles

            # --- Step 4: Compute scaling to fit height (headspace & bottom) ---
            canvas_w, canvas_h = canvas_size
            usable_height = int(canvas_h * (1 - headspace_ratio - bottom_ratio))
            scale = usable_height / cropped.shape[0]

            new_w = int(cropped.shape[1] * scale)
            new_h = int(cropped.shape[0] * scale)
            resized = cv2.resize(cropped, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)

            # --- Step 5: Create white canvas and position subject ---
            canvas = np.ones((canvas_h, canvas_w, 3), dtype=np.uint8) * 255
            x_offset = (canvas_w - new_w) // 2
            y_offset = int(canvas_h * headspace_ratio)

            # --- Step 6: Alpha blend ---
            alpha_resized = resized[:, :, 3] / 255.0
            for c in range(3):
                canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w, c] = (
                    alpha_resized * resized[:, :, c] +
                    (1 - alpha_resized) * canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w, c]
                )

            # --- Step 7: Convert BGR → RGB before saving ---
            canvas_rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)

            # --- Step 8: Save output ---
            cv2.imwrite(output_path, canvas_rgb)
            print(f"✅ Saved composed couple image to {output_path}")

            return True

        except: 
            return False
        
    def generate_background(self,
                            inp_path, 
                            out_path,
                            background_desc,
                            background_light,
                            target_size=(1800, 2400)):

        try:
            image = Image.open(inp_path)

            prompt = self.load_prompt(background_desc=background_desc, 
                                      background_light=background_light)
            
            print(f"Generating background for {inp_path}...")

            response = self.client.models.generate_content(
                model=CONFIG["IMAGE_GEN_MODEL_NAME"],
                contents=[prompt, image],
            )

            gen_img_bgr = None

            for candidate in response.candidates:
                for part in candidate.content.parts:
                    if getattr(part, "inline_data", None):
                        gen_img_bgr = cv2.imdecode(np.frombuffer(part.inline_data.data, np.uint8), cv2.IMREAD_COLOR)
                        if gen_img_bgr is not None:
                            break
                if gen_img_bgr is not None:
                    break

            if gen_img_bgr is None:
                raise RuntimeError(f"No image returned for single generation")

            # Resize to target size
            resized = cv2.resize(gen_img_bgr, target_size, interpolation=cv2.INTER_CUBIC)

            # img = Image.fromarray(cv2.cvtColor(resized, cv2.COLOR_BGR2RGB))
            # img.save(out_path.replace(".png", ".webp"), format="WEBP", quality=85, method=6)

            cv2.imwrite(out_path, resized)

            return True

        except:
            return False