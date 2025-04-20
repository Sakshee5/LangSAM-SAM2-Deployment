import boto3
import json
import base64
import time
from PIL import Image
import io
import os
import argparse
from datetime import datetime

def save_base64_image(base64_string, output_path):
    """Save a base64 encoded image to a file"""
    try:
        image_data = base64.b64decode(base64_string)
        image = Image.open(io.BytesIO(image_data))
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        image.save(output_path)
        print(f"Saved image to {output_path}")
    except Exception as e:
        print(f"Error saving image: {str(e)}")

def invoke_segmentation(endpoint_name, payload, run_name="test"):
    """Invoke a SageMaker endpoint and measure time"""
    runtime = boto3.client('runtime.sagemaker')
    
    start_time = time.time()
    try:
        response = runtime.invoke_endpoint(
            EndpointName=endpoint_name,
            ContentType='application/json',
            Body=json.dumps(payload).encode('utf-8')
        )
        end_time = time.time()
        elapsed_time = end_time - start_time
        
        # Parse the response
        result = json.loads(response['Body'].read().decode())
        
        # Create timestamp for unique folder names
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f"results/{run_name}_{timestamp}"
        
        # Save returned images
        if 'original' in result:
            save_base64_image(result['original'], f"{output_dir}/original_image.png")
        
        if 'overlay' in result:
            save_base64_image(result['overlay'], f"{output_dir}/mask_overlay.png")
        
        if 'mask_only' in result:
            save_base64_image(result['mask_only'], f"{output_dir}/mask_only.png")
        
        # Print additional information
        print(f"\nResults for {run_name}:")
        print(f"Time taken: {elapsed_time:.4f} seconds")
        
        if 'boxes' in result:
            print("Bounding boxes:", result['boxes'])
        
        if 'scores' in result:
            print("Confidence scores:", result['scores'])
        
        if 'labels' in result:
            print("Labels:", result['labels'])
        
        if 'tag' in result:
            print("Tag:", result['tag'])
        
        # Check for errors
        if 'error' in result:
            print("Error:", result['error'])
        
        if "message" in result:
            print("Message:", result['message'])
        
        if "status" in result:
            print("Status:", result['status'])
            
        return elapsed_time, result
        
    except Exception as e:
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Error invoking endpoint: {str(e)}")
        print(f"Time taken: {elapsed_time:.4f} seconds")
        return elapsed_time, {"error": str(e)}

def test_segmentation_methods(endpoint_name, image_url, text_prompt=None, point_x=10, point_y=10, runs=2):
    """Test both segmentation methods with multiple runs"""
    
    # Create payloads
    payload_langsam = {
        "action": "langsam",
        "image_url": image_url,
        "text_prompt": text_prompt or "the main object"
    }
    
    payload_sam2 = {
        "action": "sam2",
        "image_url": image_url,
        "x": point_x,
        "y": point_y
    }
    
    results = {
        "langsam": [],
        "sam2": []
    }
    
    # Test LangSAM
    print("\n" + "="*50)
    print(f"TESTING LANGSAM WITH PROMPT: '{text_prompt}'")
    print("="*50)
    
    for i in range(runs):
        print(f"\nLangSAM Run #{i+1} ({'cold start' if i==0 else 'warm run'}):")
        elapsed, result = invoke_segmentation(
            endpoint_name, 
            payload_langsam, 
            f"langsam_run{i+1}"
        )
        results["langsam"].append({"time": elapsed, "result": result})
    
    # Test SAM2
    print("\n" + "="*50)
    print(f"TESTING SAM2 WITH POINT: ({point_x}, {point_y})")
    print("="*50)
    
    for i in range(runs):
        print(f"\nSAM2 Run #{i+1} ({'cold start' if i==0 else 'warm run'}):")
        elapsed, result = invoke_segmentation(
            endpoint_name, 
            payload_sam2, 
            f"sam2_run{i+1}"
        )
        results["sam2"].append({"time": elapsed, "result": result})
    
    # Print summary
    print("\n" + "="*50)
    print("PERFORMANCE SUMMARY")
    print("="*50)
    
    print("\nLangSAM:")
    for i, run in enumerate(results["langsam"]):
        print(f"  Run #{i+1} ({'cold start' if i==0 else 'warm run'}): {run['time']:.4f} seconds")
    
    print("\nSAM2:")
    for i, run in enumerate(results["sam2"]):
        print(f"  Run #{i+1} ({'cold start' if i==0 else 'warm run'}): {run['time']:.4f} seconds")
    
    # Calculate improvements
    if runs > 1:
        langsam_improvement = (results["langsam"][0]["time"] - results["langsam"][1]["time"]) / results["langsam"][0]["time"] * 100
        sam2_improvement = (results["sam2"][0]["time"] - results["sam2"][1]["time"]) / results["sam2"][0]["time"] * 100
        
        print("\nPerformance Improvement (Cold Start vs. Warm Run):")
        print(f"  LangSAM: {langsam_improvement:.2f}% faster")
        print(f"  SAM2: {sam2_improvement:.2f}% faster")
    
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test AWS SageMaker segmentation endpoints")
    parser.add_argument("--endpoint", default="sam-segmentation", help="SageMaker endpoint name")
    parser.add_argument("--image-url", default="https://d1lt4rjd5n2mo1.cloudfront.net/product-images/e809edc4-1226-43b6-8a3b-1fe8943d52fa.jpeg", 
                        help="URL of the image to segment")
    parser.add_argument("--text-prompt", default="Black Authentic Barrel Leg Jean", 
                        help="Text prompt for LangSAM")
    parser.add_argument("--point-x", type=int, default=10, help="X coordinate for SAM2 point prompting")
    parser.add_argument("--point-y", type=int, default=10, help="Y coordinate for SAM2 point prompting")
    parser.add_argument("--runs", type=int, default=2, help="Number of runs for each method")
    
    args = parser.parse_args()
    
    test_segmentation_methods(
        args.endpoint,
        args.image_url,
        args.text_prompt,
        args.point_x,
        args.point_y,
        args.runs
    )