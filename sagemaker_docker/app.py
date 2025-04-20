import os
import json
import base64
from io import BytesIO
import requests

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware

from PIL import Image
import numpy as np
import cv2
import torch

import openai
import boto3

from lang_sam import LangSAM
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

app = FastAPI()

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def get_secret(secret_name_or_arn):
    region_name = "us-east-2"
    session = boto3.session.Session()
    client = session.client(service_name='secretsmanager', region_name=region_name)
    
    secret_id = secret_name_or_arn
    
        
    response = client.get_secret_value(SecretId=secret_id)
    secret = response['SecretString']
    return json.loads(secret)


def init_openai_client():
    # Try environment variable first
    api_key = os.environ.get('OPENAI_API_KEY')
    
    if not api_key:
        print("Failed to get OpenAI API key")
        return None
        
    return openai.OpenAI(api_key=api_key)

def chat(product_name: str):
    openai_prompt =  """You will be provided with a complete product name, which may contain brand names, extra details, and categories. Your task is to extract only the core product name (apparel or accessory) while removing brand names, categories, and unnecessary words and convert it's meaning to a basic clothing or accessory category.

Examples:
Beachwood Luxe Paneled Unitard — Girlfriend Collective → Dress
100 cotton strappy top · Black, White, Red, Peach · T-shirts And Polo Shirts | Massimo Dutti → Shirt
Wide-leg co-ord trousers with pleats · Green · Dressy | Massimo Dutti → Pants
BLANKNYC Wide Leg Jean in Radio Star | REVOLVE → Jeans

Basically, you need to convert the product name to a basic clothing or accessory category like Shirt, Pants, Dress, Jeans, etc.
Now, extract the core product name from the following:

{product_name}"""

    client = init_openai_client()

    messages = [{"role": "user", "content": openai_prompt.format(product_name=product_name)}]
          
    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
    )
    result = completion.choices[0].message.content
    
    return result


def load_sam_model():
    import os
    
    # Model is now directly in /app
    model_path = os.path.join("/app", "sam2.1_hiera_small.pt")
    
    # Config is in the expected SAM2 location
    config_path = "configs/sam2.1/sam2.1_hiera_s.yaml"
    
    # Print the current directory and confirm file existence
    print(f"Current directory: {os.getcwd()}")
    print(f"Model path: {model_path}, exists: {os.path.exists(model_path)}")
    print(f"Config path: {config_path}, exists: {os.path.exists(config_path)}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Now use the standard path that should be in Hydra's search paths
    sam2_model = build_sam2(config_path, model_path, device=device)
    sam2_predictor = SAM2ImagePredictor(sam2_model)
    
    return sam2_predictor

# Helper functions
def create_mask_overlay(image: np.ndarray, mask: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    colored_mask = np.zeros_like(image)
    # Fix potential boolean ambiguity by ensuring mask is properly processed
    mask_bool = mask > 0
    colored_mask[mask_bool] = [30, 144, 255]
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(colored_mask, contours, -1, (255, 255, 255), thickness=2)
    return cv2.addWeighted(image, 1 - alpha, colored_mask, alpha, 0)

def create_mask_only(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    result = np.zeros_like(image)
    # Fix potential boolean ambiguity
    mask_bool = mask > 0
    result[mask_bool] = image[mask_bool]
    return result

def image_to_base64(image: np.ndarray) -> str:
    _, buffer = cv2.imencode('.png', cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    return base64.b64encode(buffer).decode('utf-8')

def load_image_from_url(url):
    response = requests.get(url)
    response.raise_for_status()
    return Image.open(BytesIO(response.content))


async def segment_langsam(image_url: str, text_prompt: str):
    print(f"Processing langsam request with image_url={image_url}, text_prompt={text_prompt}")

    langsam_model = LangSAM()
        
    processed_prompt = chat(text_prompt)
    print(f"Processed prompt: {processed_prompt}")
    
    image_pil = load_image_from_url(image_url)
    image_np = np.array(image_pil)
    print("Image loaded successfully")
    
    results = langsam_model.predict([image_pil], [processed_prompt])
    print(f"Model prediction results: {results}")
    
    if not results:
        return {
            "original": image_to_base64(image_np),
            "error": "No results returned from model"
        }
    
    if 'masks' not in results[0] or results[0]['masks'] is None:
        return {
            "original": image_to_base64(image_np),
            "error": "No masks found in results"
        }
    
    masks = results[0]['masks']
    if isinstance(masks, np.ndarray) and masks.size == 0:
        return {
            "original": image_to_base64(image_np),
            "error": "Empty masks array returned"
        }
    
    if isinstance(masks, list) and len(masks) == 0:
        return {
            "original": image_to_base64(image_np),
            "error": "Empty masks list returned"
        }
    
    # Get the first mask
    if isinstance(masks, list):
        mask = masks[0]
    else:
        mask = masks[0] if masks.shape[0] > 0 else None
    
    if mask is None:
        return {
            "original": image_to_base64(image_np),
            "error": "No valid mask found"
        }
    
    overlay = create_mask_overlay(image_np, mask)
    mask_only = create_mask_only(image_np, mask)
    
    # Handle scores and boxes more safely
    boxes = []
    if "boxes" in results[0] and results[0]["boxes"] is not None:
        # Convert to list only if it's a numpy array, otherwise keep as is
        if isinstance(results[0]["boxes"], np.ndarray):
            boxes = results[0]["boxes"].tolist()
        else:
            boxes = results[0]["boxes"]
        
    scores = []
    if "scores" in results[0] and results[0]["scores"] is not None:
        # Convert to list only if it's a numpy array, otherwise keep as is
        if isinstance(results[0]["scores"], np.ndarray):
            scores = results[0]["scores"].tolist()
        else:
            scores = results[0]["scores"]
    
    return {
        "original": image_to_base64(image_np),
        "overlay": image_to_base64(overlay),
        "mask_only": image_to_base64(mask_only),
        "boxes": boxes,
        "scores": scores,
        "tag": processed_prompt,
    }

async def segment_sam2(image_url: str, x: int, y: int):
    print(f"Processing sam2 request with image_url={image_url}, x={x}, y={y}")
    image_pil = load_image_from_url(image_url)
    image_np = np.array(image_pil)
    print("Image loaded successfully")

    sam2_predictor = load_sam_model()
    print("Model loaded successfully")
    
    sam2_predictor.set_image(image_np)
    print("Image set successfully")
    
    input_point = np.array([[x, y]])
    input_label = np.array([1])
    masks, scores, _ = sam2_predictor.predict(
        point_coords=input_point,
        point_labels=input_label,
        multimask_output=True,
    )
    print(f"Model prediction completed: masks={masks.shape if masks is not None else None}, scores={scores}")
    
    # More explicit check for mask presence
    if not isinstance(masks, np.ndarray) or masks.size == 0 or masks.shape[0] == 0:
        return {
            "original": image_to_base64(image_np),
            "error": "No masks generated"
        }
    
    # Get the top mask safely
    best_score_idx = np.argmax(scores) if scores.size > 0 else 0
    if best_score_idx < masks.shape[0]:
        top_mask = masks[best_score_idx]
        overlay = create_mask_overlay(image_np, top_mask)
        mask_only = create_mask_only(image_np, top_mask)
        
        return {
            "original": image_to_base64(image_np),
            "overlay": image_to_base64(overlay),
            "mask_only": image_to_base64(mask_only),
            "score": float(scores[best_score_idx]) if scores.size > best_score_idx else 0.0
        }
    else:
        return {
            "original": image_to_base64(image_np),
            "error": "Invalid mask index"
        }

@app.get("/ping")
def ping():
    return {"status": "healthy"}

@app.post("/invocations")
async def sagemaker_invoke(request: Request):
    """SageMaker invocation endpoint that routes to appropriate function"""
    body = await request.json()
    action = body.get("action", "")
    print(f"Received action: {action}")
    
    if action == "langsam":
        return await segment_langsam(
            image_url=body.get("image_url", ""), 
            text_prompt=body.get("text_prompt", "")
        )
    
    elif action == "sam2":
        return await segment_sam2(
            image_url=body.get("image_url", ""),
            x=body.get("x", 0),
            y=body.get("y", 0)
        )
    
    else:
        raise HTTPException(status_code=400, detail="Invalid action specified")
    