import requests

# Local FastAPI server URL
BASE_URL = "http://localhost:8080"

def test_ping():
    print("Sending ping request to local server...")
    # Make request to local server
    response = requests.get(f"{BASE_URL}/ping")
    print("Response status code:", response.status_code)
    print("Response:", response.json())


def test_langsam():
    print("Sending langsam request to local server...")
    payload = {
        "action": "langsam",
        "image_url": "https://d1lt4rjd5n2mo1.cloudfront.net/product-images/e809edc4-1226-43b6-8a3b-1fe8943d52fa.jpeg", 
        "text_prompt": "Black Authentic Barrel Leg Jean",
    }
    response = requests.post(f"{BASE_URL}/invocations", json=payload)
    print("Response status code:", response.status_code)
    
    result = response.json()
    print("Full response keys:", result.keys())
    
    if "error" in result:
        print("Error:", result["error"])
        return
        
    print("Bounding boxes:", result.get('boxes', []))
    print("Confidence scores:", result.get('scores', []))
    print("Tag:", result.get('tag', 'Not found'))


def test_sam2():
    print("Sending sam2 request to local server...")
    payload = {
        "action": "sam2",
        "image_url": "https://d1lt4rjd5n2mo1.cloudfront.net/product-images/e809edc4-1226-43b6-8a3b-1fe8943d52fa.jpeg", 
        "x": 10,
        "y": 10
    }
    response = requests.post(f"{BASE_URL}/invocations", json=payload)       
    print("Response status code:", response.status_code)
    result = response.json()

    if "error" in result:
        print("Error:", result["error"])
        return

    print("Response Keys:", result.keys())

if __name__ == "__main__":
    test_ping() 
    # test_langsam()   # does not work because of chat() requiring openai api key which is passed as a secret during AWS deployment and it is not passed during local testing
    test_sam2()