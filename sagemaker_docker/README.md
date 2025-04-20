# Lang-SAM and SAM2 Deployment on AWS SageMaker using Dockerfile

A unified API endpoint that handles both text-based segmentation (Lang-SAM) and point-based segmentation (SAM2) using a containerized deployment on AWS SageMaker.

## Table of Contents
- [Overview](#overview)
- [Prerequisites](#prerequisites)
- [Quick Start](#quick-start)
- [Architecture](#architecture)
- [Models](#models)
- [Deployment](#deployment)
- [Usage](#usage)
- [Web UI Integration](#web-ui-integration)
- [Troubleshooting](#troubleshooting)
- [Acknowledgments](#acknowledgments)

## Overview

This project combines two segmentation models:
1. **Lang-SAM**: A language-guided segmentation model that segments objects based on text descriptions
2. **SAM2**: A point-based segmentation model that segments objects based on user-provided points

The repository root directory includes a web UI (`index.html`) that provides a user-friendly interface for interacting with the LangSAM and SAM2 model. However, to use this UI with the AWS SageMaker deployment, we need to set up a backend API that handles the authentication and communication with SageMaker. Here's how to set it up:

## Prerequisites

1. **AWS Account** with:
   - SageMaker access
   - ECR repository
   - S3 bucket
   - Secrets Manager access
   - IAM role with appropriate permissions

2. **Local Development Environment**:
   - Docker
   - AWS CLI configured
   - Python 3.11+
   - Required Python packages (see `requirements.txt`)

3. **API Keys**:
   - OpenAI API key stored in AWS Secrets Manager

## Quick Start

1. Clone the repository:
```bash
git clone https://github.com/Sakshee5/LangSAM-SAM2-Deployment.git
cd LangSAM-SAM2-Deployment/sagemaker_docker
```

2. Download required model files:
   - Download `sam2.1_hiera_small.pt` from official SAM2 repo
   - Download `configs/sam2.1/sam2.1_hiera_s.yaml` from official SAM2 repo
   - Place both files in a `/tmp` directory

3. Configure AWS credentials:
```bash
aws configure
```

4. Build and push Docker image:
```bash
./build_and_push.sh
```

Note: For Docker versions >0.26 and Docker Desktop 4.40.0, set:
```bash
DOCKER_BUILDKIT=0
```
to ensure image creation in a NON-OCI format (required for AWS SageMaker).

5. Deploy to SageMaker:
```bash
python deployment.py
```

6. Test the endpoints:
```bash
python run.py
```

## Architecture

The system follows a microservices architecture with these key components:

1. **FastAPI Application** (`app.py`):
   - Handles CORS configuration
   - Provides API endpoints for both Lang-SAM and SAM2
   - Manages image processing and segmentation
   - Integrates with OpenAI for text processing

2. **Docker Container**:
   - Base image: Python 3.11-slim
   - Includes all necessary system dependencies
   - Configured with PyTorch and CUDA support
   - Contains model files (baked into image)

3. **AWS Infrastructure**:
   - SageMaker endpoint for model serving
   - ECR repository for Docker image storage
   - S3 bucket for model file storage
   - Secrets Manager for API key management

### Technical Decisions

1. **IAM Role Permissions**:
   - Model files are baked into Docker image (no S3 access needed)
   - OpenAI API key passed as environment variable
   - Secrets Manager access handled in `deployment.py`

2. **Model Management**:
   - Models included in Docker image during build
   - Ensures container has all necessary files
   - Eliminates runtime S3 dependencies

## Models

The system uses two state-of-the-art segmentation models:

1. **Lang-SAM**:
   - Language-guided segmentation
   - Processes text descriptions
   - Returns segmented objects based on text input

2. **SAM2**:
   - Point-based segmentation
   - Processes user-provided coordinates
   - Returns segmented objects based on point input

## Deployment

The deployment process involves:

1. Building the Docker image
2. Pushing to ECR
3. Creating SageMaker endpoint
4. Configuring environment variables
5. Testing the deployment

Detailed deployment steps are in the [Quick Start](#quick-start) section.

## Usage

### API Endpoints

The system provides a single endpoint `/invocations` with these actions:

1. **Lang-SAM Segmentation**:
```json
{
    "action": "langsam",
    "image_url": "https://example.com/image.jpg",
    "text_prompt": "black jeans"
}
```

2. **SAM2 Segmentation**:
```json
{
    "action": "sam2",
    "image_url": "https://example.com/image.jpg",
    "x": 100,
    "y": 100
}
```

3. **Health Check**:
```json
{
    "action": "ping"
}
```

### Testing

1. **Local Testing**:
```bash
python test_local.py
```

2. **SageMaker Testing**:
```bash
python run.py
```

## Web UI Integration

The repository includes a web UI (`index.html`) for interacting with the models. To use it with AWS SageMaker:

1. **Backend Setup**:
   - Create Lambda function for SageMaker endpoint communication
   - Set up API Gateway to expose the Lambda function

2. **Frontend Configuration**:
   - Update endpoint URL in `index.html` to point to API Gateway
   - Replace Hugging Face endpoints with API Gateway endpoints

Example Lambda function:
```python
import boto3
import json

def lambda_handler(event, context):
    runtime = boto3.client('runtime.sagemaker')
    body = json.loads(event['body'])
    
    response = runtime.invoke_endpoint(
        EndpointName='sam-segmentation',
        ContentType='application/json',
        Body=json.dumps(body).encode('utf-8')
    )
    
    return {
        'statusCode': 200,
        'body': response['Body'].read().decode()
    }
```

## Troubleshooting

1. **Docker Build Issues**:
   - Verify model files are in correct location
   - Check Docker daemon status
   - Confirm AWS credentials

2. **SageMaker Deployment Issues**:
   - Check IAM role permissions
   - Verify ECR repository
   - Ensure model files are properly uploaded

3. **API Issues**:
   - Verify OpenAI API key
   - Check image URL accessibility
   - Review CORS configuration

## Acknowledgments

- [Lang-SAM](https://github.com/luca-medeiros/lang-segment-anything)
- [SAM2](https://github.com/facebookresearch/segment-anything)
- [FastAPI](https://fastapi.tiangolo.com/)
- [AWS SageMaker](https://aws.amazon.com/sagemaker/)
