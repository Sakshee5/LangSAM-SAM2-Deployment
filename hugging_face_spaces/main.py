import os
import logging
import json
import base64
from typing import Dict, Any

# Configure logging 
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Set Hugging Face cache directory to /tmp
os.environ["HF_HOME"] = "/tmp/huggingface"
os.environ["TRANSFORMERS_CACHE"] = "/tmp/huggingface"
os.environ["TORCH_HOME"] = "/tmp/torch"

from fastapi import FastAPI, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from PIL import Image
import io
import numpy as np
from lang_sam import LangSAM
import supervision as sv
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
import torch
import cv2
from dotenv import load_dotenv
import openai
import requests
from io import BytesIO

load_dotenv()
client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

app = FastAPI()

# Enable CORS for all origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create cache directories in /tmp
os.makedirs("/tmp/huggingface", exist_ok=True)
os.makedirs("/tmp/torch", exist_ok=True)

# Load the langSAM model
logger.info("Loading LangSAM model...")
langsam_model = LangSAM()
logger.info("LangSAM model loaded successfully")

# Load SAM2 Model
logger.info("Loading SAM2 model...")
sam2_checkpoint = "sam2.1_hiera_small.pt"
model_cfg = "configs/sam2.1/sam2.1_hiera_s.yaml"
device = torch.device("cpu")

sam2_model = build_sam2(model_cfg, sam2_checkpoint, device=device)
predictor = SAM2ImagePredictor(sam2_model)
logger.info("SAM2 model loaded successfully")

@app.get("/")
async def root():
    return {"message": "LangSAM API is running!"}

def create_mask_overlay(image: np.ndarray, mask: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """Create a mask overlay on the original image."""
    # Create a colored mask (blue color)
    colored_mask = np.zeros_like(image)
    colored_mask[mask > 0] = [30, 144, 255]  # Blue color
    
    # Add contour
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(colored_mask, contours, -1, (255, 255, 255), thickness=2)
    
    # Blend with original image
    overlay = cv2.addWeighted(image, 1 - alpha, colored_mask, alpha, 0)
    return overlay

def create_mask_only(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Create an image showing only the masked region."""
    # Create a black background
    result = np.zeros_like(image)
    # Copy only the masked region
    result[mask > 0] = image[mask > 0]
    return result

def image_to_base64(image: np.ndarray) -> str:
    """Convert numpy array image to base64 string."""
    _, buffer = cv2.imencode('.png', cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    return base64.b64encode(buffer).decode('utf-8')

def draw_image(image_rgb, masks, xyxy, probs, labels):
    mask_annotator = sv.MaskAnnotator()
    # Create class_id for each unique label
    unique_labels = list(set(labels))
    class_id_map = {label: idx for idx, label in enumerate(unique_labels)}
    class_id = [class_id_map[label] for label in labels]

    # Add class_id to the Detections object
    detections = sv.Detections(
        xyxy=xyxy,
        mask=masks.astype(bool),
        confidence=probs,
        class_id=np.array(class_id),
    )
    annotated_image = mask_annotator.annotate(scene=image_rgb.copy(), detections=detections)
    return annotated_image

def load_image_from_url(url):
    """Fetch image from URL and load it into memory."""
    try:
        logger.info(f"Fetching image from URL: {url}")
        response = requests.get(url)
        response.raise_for_status()
        return Image.open(BytesIO(response.content))
    except Exception as e:
        logger.error(f"Error loading image from URL: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Error loading image from URL: {str(e)}")

prompt = """You will be provided with a complete product name, which may contain brand names, extra details, and categories. Your task is to extract only the core product name (apparel or accessory) while removing brand names, categories, and unnecessary words and convert it's meaning to a basic clothing or accessory category.

Examples:
Beachwood Luxe Paneled Unitard — Girlfriend Collective → Dress
100 cotton strappy top · Black, White, Red, Peach · T-shirts And Polo Shirts | Massimo Dutti → Shirt
Wide-leg co-ord trousers with pleats · Green · Dressy | Massimo Dutti → Pants
BLANKNYC Wide Leg Jean in Radio Star | REVOLVE → Jeans

Basically, you need to convert the product name to a basic clothing or accessory category like Shirt, Pants, Dress, Jeans, etc.
Now, extract the core product name from the following:

{product_name}"""

def chat(product_name: str = Form(...)):
    try:
        logger.info(f"Processing product name: {product_name}")
        completion = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt.format(product_name=product_name)}],
        )
        result = completion.choices[0].message.content
        logger.info(f"OpenAI response: {result}")
        return result
    except Exception as e:
        logger.error(f"Error in OpenAI chat: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing product name: {str(e)}")

@app.post("/segment/sam2")
async def segment_image(
    image_url: str = Form(...), 
    x: int = Form(...), 
    y: int = Form(...)
):
    """Segment image using SAM2 with a single input point."""
    try:
        logger.info(f"Starting SAM2 segmentation for image URL: {image_url}")
        image_pil = load_image_from_url(image_url)
        image_array = np.array(image_pil)
        
        logger.info("Setting image in SAM2 predictor")
        predictor.set_image(image_array)
        
        input_point = np.array([[x, y]])
        input_label = np.array([1])  # Foreground point
        
        logger.info("Running SAM2 prediction")
        masks, scores, logits = predictor.predict(
            point_coords=input_point,
            point_labels=input_label,
            multimask_output=True,
        )

        # Get top mask
        top_mask = masks[np.argmax(scores)]

        # Create different versions of the result
        overlay_image = create_mask_overlay(image_array, top_mask)
        mask_only_image = create_mask_only(image_array, top_mask)
        
        # Convert images to base64
        original_b64 = image_to_base64(image_array)
        overlay_b64 = image_to_base64(overlay_image)
        mask_only_b64 = image_to_base64(mask_only_image)
        
        # Create response
        response = {
            "original": original_b64,
            "overlay": overlay_b64,
            "mask_only": mask_only_b64,
            "score": float(scores[np.argmax(scores)])
        }
        
        logger.info("SAM2 segmentation completed successfully")
        return response
    except Exception as e:
        logger.error(f"Error in SAM2 segmentation: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error in SAM2 segmentation: {str(e)}")

@app.post("/segment/langsam")
async def segment_image(image_url: str = Form(...), text_prompt: str = Form(...)):
    # process text prompt using openai chat
    text_prompt = chat(text_prompt)
    try:
        logger.info(f"Starting LangSAM segmentation for image URL: {image_url} with prompt: {text_prompt}")
        image_pil = load_image_from_url(image_url)
        image_array = np.array(image_pil)
        
        # Run segmentation
        logger.info("Running LangSAM prediction")
        results = langsam_model.predict([image_pil], [text_prompt])
        
        # Get the first (best) mask
        mask = results[0]["masks"][0]
        
        # Create different versions of the result
        overlay_image = create_mask_overlay(image_array, mask)
        mask_only_image = create_mask_only(image_array, mask)
        
        # Convert images to base64
        original_b64 = image_to_base64(image_array)
        overlay_b64 = image_to_base64(overlay_image)
        mask_only_b64 = image_to_base64(mask_only_image)
        
        # Create response
        response = {
            "original": original_b64,
            "overlay": overlay_b64,
            "mask_only": mask_only_b64,
            "boxes": results[0]["boxes"].tolist(),
            "scores": results[0]["scores"].tolist(),
            "labels": results[0]["labels"],
            "product_tag": text_prompt
        }
        
        logger.info("LangSAM segmentation completed successfully")
        return response
    except Exception as e:
        logger.error(f"Error in LangSAM segmentation: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error in LangSAM segmentation: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7860)
