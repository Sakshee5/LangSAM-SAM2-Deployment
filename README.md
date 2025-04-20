# LangSAM + SAM2 Deployment

The repository contains code for LangSAM and SAM2 deployment using three approaches
### 1. Hugging Face Spaces (Uses CPU) - `\hugging_face_spaces`
- Deployment: [Link](https://huggingface.co/spaces/sakshee05/langSAM)
- Run `index.html` from the root directory. Uses endpoints deployed on Hugging Face Spaces.

### 2. Sagemaker Managed (Uses GPU) - `\sagemaker_managed`
- Using Amazon SageMaker PyTorch Container
- Did not work. Check `sagemaker_managed/README.md` for more details

### 3. Sagemaker Custom (Uses GPU) - `\sagemaker_docker`
- This is `Bring your own container (BYOC)` approach where deployment is containerized using Docker and deployed on AWS SageMaker.
- Works and endpoint has been deployed under name `sam-segmentation`. Check `sagemaker_docker/README.md` for more details on replicating / using the endpoint.

## Use-Case

Designed for Fashivly. Provides image segmentation capabilities for fashion products. The system enables:

- Automatic segmentation of fashion items from product images using natural language descriptions
- Precise extraction of specific fashion elements (e.g., "red dress", "leather handbag", "sneakers") from complex product names scraped from ecommerce websites which act as LangSAM input.
- Point-based segmentation using SAM2 as a fallback.


## Features

- Language-guided image segmentation using Lang-SAM (API endpoint deployed to HF)
- Point input guided Image Segmentation with SAM2 (API endpoint deployed to HF)
- FastAPI backend with CORS support
- Web interface for easy interaction (For Hugging Face, Sagemaker endpoint will have to be exposed through API Gateway, check sagemaker-docker `README.md` for more details)
- OpenAI integration for product name processing
- Option to download original images and masks

## References

- Lang-SAM: [GitHub Repository](https://github.com/luca-medeiros/lang-segment-anything)
- SAM2: [Original Implementation](https://github.com/facebookresearch/segment-anything)
