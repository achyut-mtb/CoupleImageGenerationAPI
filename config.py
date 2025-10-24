import os

CONFIG = {
    "IMAGE_GEN_MODEL_NAME": "gemini-2.5-flash-image",
    "ARCFACE_MODEL_PATH": "models/w600k_r50.onnx",
    "INSWAPPER_MODEL_PATH": "models/inswapper_128.onnx",
    "PROMPTS_DIR": "prompts",
    "POSE_IMAGE_PATH": "pose_images/couple_image.jpg",
    "HEADSPACE_VALUE": 0.16,
    "BOTTOMSPACE_VALUE": 0,
    "ANKLE_MARGIN": 0.02,
    "API_KEY": os.getenv("GEMINI_API_KEY")    
}