from io import BytesIO
from PIL import Image
from google.genai import types
import numpy as np


def cm_to_feet_inches(cm: int):
    total_inches = round(cm / 2.54)
    feet = total_inches // 12
    inches = total_inches % 12
    return f"{feet}′{inches}″ ({cm} cm)"


def _pil_to_part(pil_img: Image.Image):
    buf = BytesIO()
    pil_img.save(buf, format="PNG")
    return types.Part(inline_data=types.Blob(mime_type="image/png", data=buf.getvalue()))


def enforce_aspect(image, target_aspect=3/4):

    h, w = image.shape[:2]
    current_aspect = w / h

    if abs(current_aspect - target_aspect) < 1e-3:
        return image

    if current_aspect < target_aspect:
        new_w = int(h * target_aspect)
        result = np.full((h, new_w, 3), 255, dtype=np.uint8)
        x_offset = (new_w - w) // 2
        result[:, x_offset:x_offset + w] = image
        return result

    else:
        new_h = int(w / target_aspect)
        result = np.full((new_h, w, 3), 255, dtype=np.uint8)
        y_offset = (new_h - h) // 2
        result[y_offset:y_offset + h, :] = image
        return result

