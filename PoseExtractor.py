import os
import cv2
from PIL import Image
from ultralytics import YOLO
import numpy as np
import pandas as pd
from io import BytesIO
import mediapipe as mp
# from OpenPose.infer import pose_estimation

# ---------------- Pose Extractor ---------------- #
class PoseExtractor:
    def __init__(self, yolo_model="models/yolov8n.pt"):
        # Mediapipe Pose
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(static_image_mode=True, model_complexity=2)
        self.mp_drawing = mp.solutions.drawing_utils

        # YOLOv8 person detector
        self.detector = YOLO(yolo_model)


    def safe_load_pose_image(self, path, resize_dim=(1200, 1600)):
        """
        Load any image safely for pose extraction.
        Ensures 3-channel BGR output with 3:4 aspect ratio (center-cropped).
        """
        img = cv2.imread(path)
        if img is None:
            raise RuntimeError(f"Could not read pose image {path}")

        # If grayscale or single channel
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        # If RGBA
        elif img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        # If single-channel with shape (H, W, 1)
        elif img.shape[2] == 1:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        # If already RGB
        elif img.shape[2] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        # Ensure dtype is uint8
        if img.dtype != np.uint8:
            img = (img * 255).astype(np.uint8)  # for float images

        # Final sanity check
        if img.ndim != 3 or img.shape[2] != 3:
            raise RuntimeError(f"Pose image {path} is not 3-channel BGR after conversion!")

        # --- Center crop to 3:4 aspect ratio ---
        h, w = img.shape[:2]
        target_ratio = 3 / 4
        current_ratio = w / h

        if current_ratio > target_ratio:
            # Too wide → crop width
            new_w = int(h * target_ratio)
            new_h = h
        else:
            # Too tall → crop height
            new_w = w
            new_h = int(w / target_ratio)

        x1 = (w - new_w) // 2
        y1 = (h - new_h) // 2
        img = img[y1:y1 + new_h, x1:x1 + new_w]

        # --- Resize after crop ---
        if resize_dim is not None:
            img = cv2.resize(img, resize_dim, interpolation=cv2.INTER_AREA)

        return img

    def extract_and_scale_pose(self, img_path, out_path,
                           man_height_cm=170, woman_height_cm=160,
                           conf_threshold=0.5):
        """
        Extract skeletons for two persons from a pose image,
        scale them according to target heights,
        and draw them side-by-side on a canvas of same size as input image.
        """
        img_bgr = self.safe_load_pose_image(img_path)
        h, w = img_bgr.shape[:2]

        # canvas = same size as input image
        canvas = np.ones((h, w, 3), dtype=np.uint8) * 255  

        # detect persons with YOLO
        results = self.detector(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))[0]
        if results.boxes.shape[0] < 2:
            print("⚠️ Less than 2 persons detected.")
            return None

        print(f"Height: {man_height_cm}, {woman_height_cm}")

        persons = []
        for det in results.boxes:
            x1, y1, x2, y2 = det.xyxy[0].cpu().numpy().astype(int)
            conf = float(det.conf[0])
            class_id = int(det.cls[0])
            if class_id != 0 or conf < conf_threshold:
                continue

            crop = img_bgr[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            results_pose = self.pose.process(crop_rgb)
            if not results_pose.pose_landmarks:
                continue

            landmarks = []
            for lm in results_pose.pose_landmarks.landmark:
                px = int(lm.x * (x2 - x1) + x1)
                py = int(lm.y * (y2 - y1) + y1)
                landmarks.append((px, py))

            persons.append(landmarks)

        if len(persons) != 2:
            print("⚠️ Did not get exactly 2 skeletons.")
            return None

        man_pts, woman_pts = persons  # assign order (left/right can be forced later)

        # === scaling logic ===
        tallest = max(man_height_cm, woman_height_cm)
        base_h = int(h * 0.9)  # use 90% of input height as reference

        def scale_skeleton(points, target_height_cm):
            min_y, max_y = min(p[1] for p in points), max(p[1] for p in points)
            orig_h = max_y - min_y
            if orig_h == 0:
                return points
            new_h = int(base_h * (target_height_cm / tallest))
            scale = new_h / orig_h
            return [(int(x*scale), int(y*scale)) for (x,y) in points]

        man_scaled = scale_skeleton(man_pts, man_height_cm)
        woman_scaled = scale_skeleton(woman_pts, woman_height_cm)

        # align both skeletons’ feet to bottom of canvas
        baseline = h - 20
        def align_baseline(points):
            max_y = max(p[1] for p in points)
            dy = baseline - max_y
            return [(x, y+dy) for (x,y) in points]

        man_scaled = align_baseline(man_scaled)
        woman_scaled = align_baseline(woman_scaled)

        # optional: ensure man is left, woman is right
        man_center_x = np.mean([p[0] for p in man_scaled])
        woman_center_x = np.mean([p[0] for p in woman_scaled])
        if man_center_x > woman_center_x:
            # swap if YOLO gave them in reverse order
            man_scaled, woman_scaled = woman_scaled, man_scaled

        # === draw skeletons ===
        for (x,y) in man_scaled:
            if 0 <= x < w and 0 <= y < h:
                cv2.circle(canvas, (x,y), 8, (255,0,0), -1)  # blue = man
        for (x,y) in woman_scaled:
            if 0 <= x < w and 0 <= y < h:
                cv2.circle(canvas, (x,y), 8, (0,0,255), -1)  # red = woman

        cv2.imwrite(out_path, canvas)
        return Image.fromarray(canvas)

    # def get_openpose_skeleton(self, img_path, out_img_path):

    #     with open(img_path, 'rb') as f:
    #         img_bytes = f.read()

    #     pose_estimation(img_bytes, output_image_path=out_img_path)

    #     return Image.open(out_img_path)


if __name__ == "__main__":

    inp_image_path = "pose_images/couple_image.jpg"
    out_image_path = "pose_image.png"

    man_height_cm = 170
    woman_height_cm = 160 

    pose_extractor = PoseExtractor()

    _ = pose_extractor.extract_and_scale_pose(
        inp_image_path,
        out_image_path,
        man_height_cm=man_height_cm,
        woman_height_cm=woman_height_cm
    )