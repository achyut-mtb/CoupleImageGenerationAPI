import os
import uvicorn
from fastapi import FastAPI, BackgroundTasks, HTTPException
from pydantic import BaseModel
from typing import Optional

from CoupleImageGenerator import CoupleImageGenerator  # <-- your class file
from config import CONFIG


# =============================
# Initialize FastAPI
# =============================
app = FastAPI(title="Couple Image Generator API", version="1.0")

# Initialize generator (load models once!)
generator = CoupleImageGenerator(api_key=CONFIG["API_KEY"])

# =============================
# Request Model
# =============================
class CoupleRequest(BaseModel):
    pair_label: str
    man_image_path: str
    woman_image_path: str
    type_couple: str = "straight"  # "straight" | "gay" | "lesbian" | "trans"
    man_height_cm: float
    woman_height_cm: float
    background_desc: Optional[str] = "A simple plain background"
    background_light: Optional[str] = "Daylight, light source present in front of persons"
    pose_image_path: Optional[str] = "pose_images/couple_pose.jpg"


# =============================
# Response Model
# =============================
class CoupleResponse(BaseModel):
    status: str
    message: str
    final_output_path: Optional[str] = None


# =============================
# Background Task
# =============================
def process_generation(req: CoupleRequest):
    try:
        status = generator.generate_couple_photo(
            pair_label=req.pair_label,
            type_couple=req.type_couple,
            man_single_img_path=req.man_image_path,
            woman_single_img_path=req.woman_image_path,
            man_height_cm=req.man_height_cm,
            woman_height_cm=req.woman_height_cm,
            pose_image_path=req.pose_image_path
        )

        if not status:
            print("⚠️ Generation failed.")
        else:
            print(f"✅ Couple image generated for {req.pair_label}")

    except Exception as e:
        print(f"❌ Background task failed: {e}")


# =============================
# Main Endpoint
# =============================
@app.post("/generate-couple", response_model=CoupleResponse)
async def generate_couple(req: CoupleRequest, background_tasks: BackgroundTasks):

    pair_dir = os.path.join("resources/canvas", req.pair_label)
    os.makedirs(pair_dir, exist_ok=True)

    try:
        status = generator.generate_couple_photo(
            pair_label=req.pair_label,
            type_couple=req.type_couple,
            man_single_img_path=req.man_image_path,
            woman_single_img_path=req.woman_image_path,
            man_height_cm=req.man_height_cm,
            woman_height_cm=req.woman_height_cm,
            pose_image_path=req.pose_image_path
        )

        if not status:
            raise HTTPException(status_code=500, detail="Image generation failed")

        final_path = os.path.join(pair_dir, "final_output.png")

        return CoupleResponse(
            status="success",
            message="Couple image generated successfully",
            final_output_path=final_path
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# =============================
# Run locally
# =============================
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)
