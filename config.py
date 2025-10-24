import os

class Config:
    """
    Configuration settings for the image processing pipeline.
    All variables are defined as static class attributes.
    """
    
    # Model and File Paths
    IMAGE_GEN_MODEL_NAME: str = "gemini-2.5-flash-image"
    ARCFACE_MODEL_PATH: str = "models/w600k_r50.onnx"
    INSWAPPER_MODEL_PATH: str = "models/inswapper_128.onnx"
    PROMPTS_DIR: str = "prompts"
    POSE_IMAGE_PATH: str = "pose_images/couple_image.jpg"

    # Image Spacing and Margin Values
    HEADSPACE_VALUE: float = 0.16
    BOTTOMSPACE_VALUE: float = 0.0
    ANKLE_MARGIN: float = 0.02

    # API Key (Retrieved from Environment)
    API_KEY: str = os.getenv("GEMINI_API_KEY")