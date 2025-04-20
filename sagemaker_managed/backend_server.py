from flask import Flask, request, jsonify
from flask_cors import CORS
import boto3
import json
import logging
from botocore.config import Config
import os
# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Initialize SageMaker runtime client with longer timeout
sagemaker_runtime = boto3.client(
    'sagemaker-runtime', 
    region_name=os.getenv('AWS_DEFAULT_REGION'),
    config=Config(
        connect_timeout=300,  # 5 minutes
        read_timeout=300,     # 5 minutes
    )
)

@app.route('/health', methods=['GET'])
def health_check():
    try:
        response = sagemaker_runtime.invoke_endpoint(
            EndpointName='sam-segmentation',
            ContentType='application/json',
            Body=json.dumps({"endpoint_type": "health"})
        )
        return jsonify(json.loads(response['Body'].read().decode()))
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/segment/langsam', methods=['POST'])
def langsam_proxy():
    try:
        data = request.json
        payload = {
            "endpoint_type": "langsam",
            "image_url": data.get("image_url"),
            "text_prompt": data.get("text_prompt")
        }
        
        response = sagemaker_runtime.invoke_endpoint(
            EndpointName='sam-segmentation',
            ContentType='application/json',
            Body=json.dumps(payload)
        )
        return jsonify(json.loads(response['Body'].read().decode()))
    except Exception as e:
        logger.error(f"LangSAM proxy failed: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/segment/sam2', methods=['POST'])
def sam2_proxy():
    try:
        data = request.json
        payload = {
            "endpoint_type": "sam2",
            "image_url": data.get("image_url"),
            "x": data.get("x"),
            "y": data.get("y")
        }
        
        response = sagemaker_runtime.invoke_endpoint(
            EndpointName='sam-segmentation',
            ContentType='application/json',
            Body=json.dumps(payload)
        )
        return jsonify(json.loads(response['Body'].read().decode()))
    except Exception as e:
        logger.error(f"SAM2 proxy failed: {str(e)}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=7860) 