import os
import json
import base64
import numpy as np
import torch
import cv2
from PIL import Image
from io import BytesIO
import requests
import boto3
import threading
import time
from openai import OpenAI
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('/opt/ml/code/debug.log')
    ]
)
logger = logging.getLogger(__name__)

# Global variables to track initialization
initialization_status = {"status": "not_started", "message": "Initialization has not begun"}
models = None

def get_secret(secret_name):
    logger.info(f"Attempting to get secret: {secret_name}")
    region_name = os.getenv('AWS_DEFAULT_REGION')
    session = boto3.session.Session()
    client = session.client(service_name='secretsmanager', region_name=region_name)
    try:
        response = client.get_secret_value(SecretId=secret_name)
        logger.info("Successfully retrieved secret")
    except Exception as e:
        logger.error(f"Error retrieving secret: {e}") 
        return None
    
    secret = response['SecretString']
    return json.loads(secret)

def model_fn(model_dir):
    """Load the models when the container starts up."""
    logger.info("Starting model initialization...")
    global initialization_status
    
    # Update status to initializing
    initialization_status = {"status": "initializing", "message": "Models are loading..."}
    
    # Return immediately with a dummy model for health checks
    dummy_model = {"dummy": True}
    
    # Start initialization in a separate thread
    def initialize_models():
        global initialization_status, models
        try:
            logger.info("Starting model initialization thread...")
            # Import these inside the thread to avoid blocking startup
            logger.info("Importing LangSAM...")
            from lang_sam import LangSAM
            logger.info("Importing SAM2...")
            from sam2.build_sam import build_sam2
            from sam2.sam2_image_predictor import SAM2ImagePredictor

            # Fetch the OpenAI API key from Secrets Manager
            logger.info("Fetching OpenAI API key...")
            secret_name = "OPENAI_API_KEY"
            secret = get_secret(secret_name)
            openai_api_key = secret.get('OPENAI_API_KEY')
            if not openai_api_key:
                logger.error("OPENAI_API_KEY not found in secrets")
                initialization_status = {"status": "error", "message": "OPENAI_API_KEY not found in secrets"}
                return
            
            # Initialize OpenAI client
            logger.info("Initializing OpenAI client...")
            client = OpenAI(api_key=openai_api_key)
            
            # Initialize S3 client
            logger.info("Initializing S3 client...")
            s3_client = boto3.client('s3')
            
            # Check device availability
            logger.info("Checking device availability...")
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            logger.info(f"Using device: {device}")
            
            # Load SAM2 Model
            logger.info("Loading SAM2 model...")
            sam2_model_path = os.path.join(model_dir, "sam2.1_hiera_small.pt")
            
            # If not in model_dir, check if we need to download from S3
            if not os.path.exists(sam2_model_path):
                logger.info(f"Model not found in {sam2_model_path}, trying S3...")
                model_bucket = os.environ.get('MODEL_BUCKET')
                if model_bucket:
                    try:
                        logger.info(f"Downloading from S3 bucket {model_bucket}...")
                        s3_client.download_file(
                            model_bucket,
                            'sam2-model/sam2.1_hiera_small.pt',
                            sam2_model_path
                        )
                        logger.info("Download complete.")
                    except Exception as e:
                        logger.error(f"Error downloading model from S3: {e}")
                        initialization_status = {"status": "error", "message": f"Could not download model from S3: {str(e)}"}
                        return
            
            logger.info("Building SAM2 model...")
            config_path = os.path.join(model_dir, "configs/sam2.1/sam2.1_hiera_s.yaml")
            if not os.path.exists(config_path):
                logger.error(f"Missing config file: {config_path}")
                initialization_status = {"status": "error", "message": f"Missing config file: {config_path}"}
                return
            
            sam2_model = build_sam2(config_path, sam2_model_path, device=device)
            predictor = SAM2ImagePredictor(sam2_model)
            logger.info("SAM2 model loaded successfully")
            
            # Load LangSAM model
            logger.info("Loading LangSAM model...")
            try:
                langsam_model = LangSAM()
                logger.info("LangSAM model loaded successfully")
            except Exception as e:
                logger.error(f"Error loading LangSAM model: {e}")
                initialization_status = {"status": "error", "message": f"Error loading LangSAM model: {str(e)}"}
                return
            
            # Store models in global variable
            models = {
                "langsam": langsam_model,
                "sam2": predictor,
                "openai_client": client
            }
            
            initialization_status = {"status": "ready", "message": "Models loaded successfully"}
            logger.info("All models initialized successfully")
            
        except Exception as e:
            logger.error(f"Error during model initialization: {e}", exc_info=True)
            initialization_status = {"status": "error", "message": f"Error during model initialization: {str(e)}"}
    
    # Start the initialization in a separate thread
    init_thread = threading.Thread(target=initialize_models)
    init_thread.daemon = True  # Allow the thread to exit when the main program exits
    init_thread.start()
    
    return dummy_model

def input_fn(request_body, request_content_type):
    """Parse the input request body."""
    if request_content_type == "application/json":
        return json.loads(request_body)
    else:
        raise ValueError(f"Unsupported content type: {request_content_type}")

def predict_fn(input_data, model):
    """Make predictions using the loaded models."""
    global initialization_status, models
    
    endpoint_type = input_data.get("endpoint_type")
    
    # Handle health check request - always works regardless of model loading status
    if endpoint_type == "health":
        startup_time = None
        if initialization_status["status"] == "initializing":
            startup_time = "in progress"
        
        return {
            "status": initialization_status["status"],
            "message": initialization_status["message"],
            "models_loaded": {
                "initialization_complete": initialization_status["status"] == "ready",
                "startup_time": startup_time
            }
        }
    
    # For other endpoints, check if models are loaded
    if initialization_status["status"] != "ready":
        waiting_time = 0
        max_wait = 10  # Maximum wait time in seconds
        
        # Check if this is just at the beginning of initialization and wait briefly
        while initialization_status["status"] == "initializing" and waiting_time < max_wait:
            time.sleep(1)
            waiting_time += 1
        
        # If still not ready, return status
        if initialization_status["status"] != "ready":
            return {
                "error": "Models are still initializing",
                "status": initialization_status["status"],
                "message": initialization_status["message"],
                "retry_after_seconds": 30  # Suggest a retry time
            }
    
    # Now process the actual request types
    if endpoint_type == "langsam":
        return predict_langsam(input_data, models)
    elif endpoint_type == "sam2":
        return predict_sam2(input_data, models)
    else:
        raise ValueError(f"Unsupported endpoint type: {endpoint_type}")

def ping():
    """Health check endpoint for SageMaker"""
    return {"status": "healthy"}

prompt = """You will be provided with a complete product name, which may contain brand names, extra details, and categories. Your task is to extract only the core product name (apparel or accessory) while removing brand names, categories, and unnecessary words and convert it's meaning to a basic clothing or accessory category.

Examples:
Beachwood Luxe Paneled Unitard — Girlfriend Collective → Dress
100 cotton strappy top · Black, White, Red, Peach · T-shirts And Polo Shirts | Massimo Dutti → Shirt
Wide-leg co-ord trousers with pleats · Green · Dressy | Massimo Dutti → Pants
BLANKNYC Wide Leg Jean in Radio Star | REVOLVE → Jeans

Basically, you need to convert the product name to a basic clothing or accessory category like Shirt, Pants, Dress, Jeans, etc.
Now, extract the core product name from the following:

{product_name}"""

def chat(product_name, client):
    try:
        completion = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt.format(product_name=product_name)}],
        )
        result = completion.choices[0].message.content
        return result
    except Exception as e:
        print(f"Error in OpenAI chat: {str(e)}")
        raise
    
def predict_langsam(input_data, models):
    """Make predictions using LangSAM model."""
    image_url = input_data["image_url"]
    text_prompt = input_data["text_prompt"]

    # process text prompt using openai chat
    text_prompt = chat(text_prompt, models["openai_client"])
    
    # Load image
    response = requests.get(image_url)
    image_pil = Image.open(BytesIO(response.content))
    image_array = np.array(image_pil)
    
    # Run segmentation
    results = models["langsam"].predict([image_pil], [text_prompt])
    
    # Check if results are valid
    if not results or not results[0].get("masks") or len(results[0]["masks"]) == 0:
        return {
            "error": "No masks found for the given prompt",
            "overlay": None,
            "original": None,
            "mask_only": None,
            "product_tag": text_prompt
        }
    
    # Get the first (best) mask
    mask = results[0]["masks"][0]
    
    # Create mask overlay
    overlay = create_mask_overlay(image_array, mask)
    
    # Create mask-only image
    mask_only = np.zeros_like(image_array)
    mask_only[mask > 0] = [255, 255, 255]  # White color for mask
    
    # Convert to base64
    overlay_b64 = image_to_base64(overlay)
    original_b64 = image_to_base64(image_array)
    mask_only_b64 = image_to_base64(mask_only)
    
    return {
        "overlay": overlay_b64,
        "original": original_b64,
        "mask_only": mask_only_b64,
        "product_tag": text_prompt
    }

def predict_sam2(input_data, models):
    """Make predictions using SAM2 model."""
    image_url = input_data["image_url"]
    x = input_data["x"]
    y = input_data["y"]
    
    # Load image
    response = requests.get(image_url)
    image_pil = Image.open(BytesIO(response.content))
    image_array = np.array(image_pil)
    
    # Set image in predictor
    models["sam2"].set_image(image_array)
    
    # Run prediction
    input_point = np.array([[x, y]])
    input_label = np.array([1])  # Foreground point
    
    masks, scores, _ = models["sam2"].predict(
        point_coords=input_point,
        point_labels=input_label,
        multimask_output=True,
    )
    
    # Check if prediction succeeded
    if masks is None or len(masks) == 0:
        return {
            "error": "No masks found for the given point",
            "overlay": None,
            "original": None,
            "mask_only": None,
            "score": 0.0
        }
    
    # Get top mask
    top_mask = masks[np.argmax(scores)]
    
    # Create mask overlay
    overlay = create_mask_overlay(image_array, top_mask)
    
    # Create mask-only image
    mask_only = np.zeros_like(image_array)
    mask_only[top_mask > 0] = [255, 255, 255]  # White color for mask
    
    # Convert to base64
    overlay_b64 = image_to_base64(overlay)
    original_b64 = image_to_base64(image_array)
    mask_only_b64 = image_to_base64(mask_only)
    
    return {
        "overlay": overlay_b64,
        "original": original_b64,
        "mask_only": mask_only_b64,
        "score": float(scores[np.argmax(scores)])
    }

def create_mask_overlay(image: np.ndarray, mask: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """Create a mask overlay on the original image."""
    colored_mask = np.zeros_like(image)
    colored_mask[mask > 0] = [30, 144, 255]  # Blue color
    
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(colored_mask, contours, -1, (255, 255, 255), thickness=2)
    
    overlay = cv2.addWeighted(image, 1 - alpha, colored_mask, alpha, 0)
    return overlay

def image_to_base64(image: np.ndarray) -> str:
    """Convert numpy array image to base64 string."""
    _, buffer = cv2.imencode('.png', cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    return base64.b64encode(buffer).decode('utf-8')

def output_fn(prediction, content_type):
    """Format the prediction output."""
    if content_type == "application/json":
        return json.dumps(prediction)
    else:
        raise ValueError(f"Unsupported content type: {content_type}")
