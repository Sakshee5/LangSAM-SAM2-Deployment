# Hugging Face Lang-SAM + SAM2 Deployment (CPU)

- Hugging Face Deployment: [Link](https://huggingface.co/spaces/sakshee05/langSAM)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/Sakshee5/LangSAM-SAM2-Deployment.git
cd LangSAM-SAM2-Deployment/hugging_face_spaces
```

2. Create and activate a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Model Setup

### Lang-SAM Model
The Lang-SAM model will be automatically downloaded when you first run the application.

### SAM2 Model
Since the SAM2 model weights are too large for GitHub, need to be downloaded separately:

1. Download the SAM2 model weights:
   - Model: `sam2.1_hiera_small.pt`
   - Place it in the root directory of the project

2. Download the model configuration:
   - Config file: `configs/sam2.1/sam2.1_hiera_s.yaml`
   - Place it in the `configs/sam2.1/` directory

## Environment Setup

Create a `.env` file in the root directory with your OpenAI API key:
```
OPENAI_API_KEY=your_api_key_here
```

## Running the Application

1. Start the FastAPI server:
```bash
uvicorn main:app --reload
```

2. Run `index.html` to test. Either use the deployed Hugging Face endpoints already set up or replace with `localhost` to test locally

## API Endpoints

- `GET /`: Health check endpoint
- `POST /segment`: Image segmentation endpoint
- `POST /openai/chat`: Product name processing endpoint
