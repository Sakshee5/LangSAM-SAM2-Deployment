import os
import sagemaker
from sagemaker.pytorch import PyTorchModel
import shutil
import boto3
import logging
# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    # Use the role ARN directly
    role = os.getenv('AWS_ARN_ROLE')
    model_bucket = os.getenv('MODEL_BUCKET')
    
    # Set the region explicitly
    region = os.getenv('AWS_DEFAULT_REGION')
    boto3.setup_default_session(region_name=region)

    # Initialize SageMaker session with explicit region
    sagemaker_session = sagemaker.Session(
        boto_session=boto3.Session(region_name=region),
        default_bucket=model_bucket
    )

    
    # Create a temporary directory for model artifacts
    model_dir = "model_artifacts"
    os.makedirs(model_dir, exist_ok=True)
    
    local_model_path = "sam2.1_hiera_small.pt"
    
    logger.info(f"Using local model file at {local_model_path}")
    shutil.copy(local_model_path, os.path.join(model_dir, 'sam2.1_hiera_small.pt'))

    # Copy configs directory
    if os.path.exists('configs'):
        shutil.copytree('configs', os.path.join(model_dir, 'configs'), dirs_exist_ok=True)
    else:
        logger.warning("configs directory not found locally")
    
    # Copy requirements.txt
    shutil.copy('requirements.txt', os.path.join(model_dir, 'requirements.txt'))
    
    # Copy inference.py
    shutil.copy('inference.py', os.path.join(model_dir, 'inference.py'))
    

    # Create a model.tar.gz file
    logger.info("Creating model.tar.gz")
    shutil.make_archive("model", "gztar", model_dir)
    
    # Use boto3 directly for upload with progress tracking
    logger.info(f"Uploading model.tar.gz to S3 bucket {model_bucket}")
    
    class ProgressPercentage:
        def __init__(self, filename):
            self._filename = filename
            self._size = os.path.getsize(filename)
            self._seen_so_far = 0
            self._lock = threading.Lock()
            
        def __call__(self, bytes_amount):
            with self._lock:
                self._seen_so_far += bytes_amount
                percentage = (self._seen_so_far / self._size) * 100
                logger.info(f"Upload progress: {percentage:.2f}% ({self._seen_so_far}/{self._size} bytes)")
    
    import threading
    s3_client = boto3.client('s3')
    object_name = "sam2-model/model.tar.gz"
    
    try:
        logger.info(f"Starting upload of file size: {os.path.getsize('model.tar.gz')} bytes")
        s3_client.upload_file(
            'model.tar.gz', 
            model_bucket, 
            object_name,
            Callback=ProgressPercentage('model.tar.gz')
        )
        logger.info("Upload completed successfully")
    except Exception as e:
        logger.error(f"Upload failed: {str(e)}")
        raise
    
    s3_model_path = f"s3://{model_bucket}/{object_name}"
    logger.info(f"Model uploaded to: {s3_model_path}")
    
    # Create the model
    logger.info("Creating SageMaker model")
    model = PyTorchModel(
        model_data=s3_model_path,
        role=role,
        framework_version="2.4",
        py_version="py310",
        entry_point="inference.py",
        env={"MODEL_BUCKET": model_bucket},
        sagemaker_session=sagemaker_session
    )
    
    # Deploy the model with serverless configuration
    logger.info("Deploying endpoint")
    predictor = model.deploy(
        instance_type="ml.g4dn.xlarge",  # Use GPU instance
        initial_instance_count=1,
        endpoint_name="sam-segmentation",
        wait=True,
        container_startup_health_check_timeout=300,  # 5 minutes timeout
    )
    
    logger.info(f"Endpoint deployed successfully: {predictor.endpoint_name}")
    
    # Clean up
    logger.info("Cleaning up temporary files")
    shutil.rmtree(model_dir)
    os.remove("model.tar.gz")

if __name__ == "__main__":
    main()