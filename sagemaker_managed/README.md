# Lang-SAM / SAM2 GPU Deployment | Using Amazon SageMaker PyTorch Container

The deployment is an attempt to containerize using Amazon SageMaker's PyTorch container.

## Technical Challenges

The deployment faced several technical challenges:

1. **Version Compatibility Issues**:
   - LangSAM requires PyTorch 2.4.1, which is not supported by the SageMaker container
   - Python 3.11+ requirement for LangSAM conflicts with SageMaker's Python version support

2. **Deployment Limitations**:
   - Endpoint deployment completes successfully but testing consistently results in timeouts
   - Limited access to CloudWatch logs makes debugging challenging
   - The root cause is likely version incompatibility between dependencies

## Project Structure

### Core Components

1. **`deploy_to_sagemaker.py`**
   - Handles the SageMaker model deployment process
   - Manages model artifact packaging and S3 upload
   - Configures the SageMaker endpoint with GPU support
   - Implements progress tracking for large file uploads

2. **`inference.py`**
   - Core ation
   - Handles image processing and segmentation
   - Manages API endinference logic for both Lang-SAM and SAM2 models
   - Implements model loading and initializpoints for different segmentation types
   - Includes OpenAI integration for text prompt processing

3. **`backend_server.py`**
   - Flask-based proxy server for SageMaker endpoint
   - Implements CORS support for web interface
   - Provides health check endpoints
   - Manages request routing to appropriate segmentation models

4. **`requirements.txt`**
   - Specifies all project dependencies with version constraints
   - Includes packages:
     - FastAPI and Uvicorn for web serving
     - OpenAI for text processing
     - Boto3 and SageMaker SDK for AWS integration
     - OpenCV and Pillow for image processing
     - Custom Lang-SAM package from GitHub

5. **`.env`** : environment variables needed
- OPENAI_API_KEY
- MODEL_BUCKET
- AWS_ACCESS_KEY_ID
- AWS_SECRET_ACCESS_KEY
- AWS_DEFAULT_REGION
- AWS_ARN_ROLE