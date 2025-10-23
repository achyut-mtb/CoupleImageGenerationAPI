import requests

# --- API endpoint ---
url = "http://localhost:8080/generate-couple"

# --- Request body ---
payload = {
    "pair_label": "pair_001",
    "man_image_path": "../Pair Image Generation/Chirag_Pairs/new_outputs/11_43/Person1/single_face_swapped.png",
    "woman_image_path": "../Pair Image Generation/Chirag_Pairs/new_outputs/11_43/Person2/single_face_swapped.png",
    "type_couple": "straight",
    "man_height_cm": 178.0,
    "woman_height_cm": 165.0,
    "background_desc": "A cozy indoor background with warm lighting",
    "background_light": "Soft evening light from the front",
    "pose_image_path": "pose_images/couple_image.jpg"
}

# --- Send POST request ---
response = requests.post(url, json=payload)

# --- Print response ---
print("Status Code:", response.status_code)
print("Response JSON:", response.json())
