import os
import torch
import numpy as np
from PIL import Image
import time
import base64
import io
import cv2
import asyncio  # Added import
from typing import Dict, Optional
import mss  # For screen capture
from fastapi import FastAPI, HTTPException, BackgroundTasks, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
from torchvision import transforms  # Added import

# Import the model architecture
from src.model import ComponentClassifier

# FastAPI app
app = FastAPI(
    title="Car Component State Detection API",
    description="API for detecting the state of car components in real-time",
    version="1.0.0"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create static directory and placeholder image if they don't exist
os.makedirs("static", exist_ok=True)
placeholder_path = "static/placeholder.png"
if not os.path.exists(placeholder_path):
    # Create a simple placeholder image
    placeholder = np.ones((300, 300, 3), dtype=np.uint8) * 240  # Light gray
    cv2.putText(
        placeholder,
        "No image captured",
        (50, 150),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (100, 100, 100),
        2
    )
    cv2.imwrite(placeholder_path, placeholder)

# Setup templates directory
os.makedirs("templates", exist_ok=True)
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

# Configuration
CONFIG = {
    "model_path": "./trained_models/car_component_classifier_model_resnet50_max.pt",
    "image_size": 224,
    "embedding_dim": 256,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "capture_interval": 0.5,  # Seconds between captures
    "monitor": 1,  # Monitor to capture (0 is primary)
    "capture_area": {
        "top": 0,      # Initial values - will be dynamically set
        "left": 0,     # Initial values - will be dynamically set
        "width": 800,  # Adjust based on your 3D view size
        "height": 600  # Adjust based on your 3D view size
    }
}

# Try to load capture config if it exists
try:
    if os.path.exists("capture_config.json"):
        with open("capture_config.json", "r") as f:
            import json
            config_data = json.load(f)
            CONFIG["monitor"] = config_data.get("monitor", CONFIG["monitor"])
            CONFIG["capture_area"] = config_data.get("capture_area", CONFIG["capture_area"])
            print(f"Loaded capture configuration: {CONFIG['capture_area']}")
except Exception as e:
    print(f"Error loading capture config: {e}")

# Component names
COMPONENT_NAMES = ['Front Left Door', 'Front Right Door', 'Rear Left Door', 'Rear Right Door', 'Hood']

# Global variables
model = None
latest_predictions = {component: "Closed" for component in COMPONENT_NAMES}
latest_probabilities = {component: 0.0 for component in COMPONENT_NAMES}
latest_inference_time = 0.0
latest_capture_image = None
is_capturing = False

# Pydantic models
class CaptureArea(BaseModel):
    top: int
    left: int
    width: int
    height: int

class ComponentState(BaseModel):
    predictions: Dict[str, str]
    probabilities: Dict[str, float]
    inference_time: float


            
# Load the model
def get_model():
    global model
    if model is None:
        try:
            model = ComponentClassifier(embedding_dim=CONFIG["embedding_dim"])
            checkpoint = torch.load(CONFIG["model_path"], map_location=CONFIG["device"])
            model.load_state_dict(checkpoint['model_state_dict'])
            model.to(CONFIG["device"])
            model.eval()
            print(f"Model loaded from {CONFIG['model_path']} using {CONFIG['device']}")
        except Exception as e:
            print(f"Error loading model: {e}")
            # Create a dummy model for testing if real model fails
            model = "dummy"
    return model

# Transform for inference
inference_transform = transforms.Compose([
    transforms.Resize((CONFIG["image_size"], CONFIG["image_size"])),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Add this new preprocessing function
def process_car_image(img_array, target_size=(224, 224), margin_percent=15):
    """
    Apply the same preprocessing technique used in training:
    1. Detect car by separating background
    2. Crop to car boundaries with margin
    3. Center car in new canvas
    4. Resize while preserving aspect ratio
    """
    # Convert to grayscale for processing if it's not already
    if len(img_array.shape) == 3:
        gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
    else:
        gray = img_array.copy()
    
    # Get image dimensions
    h, w = gray.shape[:2]
    
    # Find the background color (most common color in corners)
    corners = [
        gray[0:10, 0:10],      # top-left
        gray[0:10, w-10:w],     # top-right
        gray[h-10:h, 0:10],     # bottom-left
        gray[h-10:h, w-10:w]    # bottom-right
    ]
    
    # Flatten corners and find most common value
    corner_pixels = np.concatenate([c.flatten() for c in corners])
    bg_color = int(np.median(corner_pixels))
    
    # Create mask: non-background pixels
    threshold = 20  # Adjust based on background uniformity
    lower_bound = max(0, bg_color - threshold)
    upper_bound = min(255, bg_color + threshold)
    
    # Create the mask
    mask = cv2.inRange(gray, lower_bound, upper_bound)
    mask = cv2.bitwise_not(mask)  # Invert so car is white (255)
    
    # Clean up the mask with morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    
    # Find contours in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Default to original image if processing fails
    processed_img = img_array.copy()
    
    if contours:
        # Find the largest contour (should be the car)
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Get bounding box
        x, y, w_car, h_car = cv2.boundingRect(largest_contour)
        
        # Add margin
        margin_x = int(w_car * margin_percent / 100)
        margin_y = int(h_car * margin_percent / 100)
        
        # Calculate new coordinates with margin, ensuring they stay within image bounds
        x1 = max(0, x - margin_x)
        y1 = max(0, y - margin_y)
        x2 = min(w, x + w_car + margin_x)
        y2 = min(h, y + h_car + margin_y)
        
        # Crop image to car with margin
        cropped = img_array[y1:y2, x1:x2]
        
        # Get cropped dimensions
        h_crop, w_crop = cropped.shape[:2]
        
        # Determine scaling factor to fit within target size while preserving aspect ratio
        scale = min(target_size[0] / w_crop, target_size[1] / h_crop)
        
        # Calculate new dimensions
        new_w = int(w_crop * scale)
        new_h = int(h_crop * scale)
        
        # Resize the cropped image
        resized = cv2.resize(cropped, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
        # Create canvas of target size with background color
        canvas = np.ones((target_size[1], target_size[0], 3), dtype=np.uint8) * bg_color
        
        # Calculate position to center the car
        pos_x = (target_size[0] - new_w) // 2
        pos_y = (target_size[1] - new_h) // 2
        
        # Place the resized car on the canvas
        if len(img_array.shape) == 3:  # Color image
            canvas[pos_y:pos_y+new_h, pos_x:pos_x+new_w] = resized
        else:  # Grayscale image
            for c in range(3):
                canvas[pos_y:pos_y+new_h, pos_x:pos_x+new_w, c] = resized
        
        processed_img = canvas
    
    return processed_img

# Modified preprocess_image function to incorporate our preprocessing
def preprocess_image(image):
    if isinstance(image, str):  # If image path
        img_array = cv2.imread(image)
    elif isinstance(image, np.ndarray):  # If numpy array (from screen capture)
        img_array = image
    else:
        # If it's a PIL Image
        img_array = np.array(image)
        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    
    # Apply car detection and preprocessing
    processed_img = process_car_image(
        img_array, 
        target_size=(CONFIG["image_size"], CONFIG["image_size"]),
        margin_percent=15
    )
    
    # Convert to PIL Image
    pil_image = Image.fromarray(cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB))
    
    # Apply normalization and convert to tensor
    return inference_transform(pil_image).unsqueeze(0)

# Predict component states
async def predict_component_states(image_tensor):
    global latest_predictions, latest_probabilities, latest_inference_time
    
    model_instance = get_model()
    if model_instance == "dummy":
        # Return dummy predictions for testing
        return (
            {name: "Closed" for name in COMPONENT_NAMES},
            {name: 0.1 for name in COMPONENT_NAMES},
            0.01
        )
    
    with torch.no_grad():
        start_time = time.time()
        image_tensor = image_tensor.to(CONFIG["device"])
        
        # Forward pass through component classifier
        outputs = model_instance(image_tensor)
        
        # Apply sigmoid to get probabilities
        probabilities = torch.sigmoid(outputs)
        
        # Binary classification based on threshold
        predictions = (probabilities >= 0.8).cpu().numpy()[0]
        
        inference_time = time.time() - start_time
    
    result = {}
    probs = {}
    for i, name in enumerate(COMPONENT_NAMES):
        result[name] = "Open" if predictions[i] else "Closed"
        probs[name] = float(probabilities.cpu().numpy()[0][i])
    
    # Update global state
    latest_predictions = result
    latest_probabilities = probs
    latest_inference_time = inference_time
    
    return result, probs, inference_time

# Screen capture function
def capture_screen():
    with mss.mss() as sct:
        # Get information about the monitor
        monitor = sct.monitors[CONFIG["monitor"]]
        
        # Use configured capture area or default to full monitor
        capture_area = {
            "top": CONFIG["capture_area"]["top"],
            "left": CONFIG["capture_area"]["left"],
            "width": CONFIG["capture_area"]["width"],
            "height": CONFIG["capture_area"]["height"],
            "mon": CONFIG["monitor"],
        }
        
        # Capture the screen
        sct_img = sct.grab(capture_area)
        
        # Convert to numpy array
        img = np.array(sct_img)
        
        # Convert BGRA to BGR by slicing off the alpha channel
        img = img[:, :, :3]
        
        return img

# Continuous capture and prediction function
async def continuous_capture(background_tasks: BackgroundTasks):
    global is_capturing, latest_capture_image
    
    if is_capturing:
        return
    
    is_capturing = True
    
    try:
        while is_capturing:
            # Capture screen
            img = capture_screen()
            latest_capture_image = img.copy()
            
            # Save the raw captured image
            cv2.imwrite("static/latest_raw.jpg", img)
            
            # Apply our preprocessing before the model's transform
            processed_img = process_car_image(
                img, 
                target_size=(CONFIG["image_size"], CONFIG["image_size"]),
                margin_percent=15
            )
            
            # Save the preprocessed image for debugging
            cv2.imwrite("static/latest_processed.jpg", processed_img)
            
            # Preprocess and predict
            img_tensor = preprocess_image(img)
            await predict_component_states(img_tensor)
            
            # Sleep before next capture
            await asyncio.sleep(CONFIG["capture_interval"])
    except Exception as e:
        print(f"Error in continuous capture: {e}")
    finally:
        is_capturing = False

# Routes
@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    """Serve the main HTML page"""
    # Ensure the templates directory exists
    os.makedirs("templates", exist_ok=True)
       
    return templates.TemplateResponse(
        "index.html", 
        {"request": request}
    )

@app.get("/status")
async def get_status():
    """Get the latest component states"""
    return ComponentState(
        predictions=latest_predictions,
        probabilities=latest_probabilities,
        inference_time=latest_inference_time
    )

@app.get("/latest-image")
async def get_latest_image():
    """Get the latest captured image as base64"""
    global latest_capture_image
    
    if latest_capture_image is None:
        return {"status": "error", "message": "No image captured yet"}
    
    # Encode the image to base64
    _, buffer = cv2.imencode(".jpg", latest_capture_image)
    img_str = base64.b64encode(buffer).decode("utf-8")
    
    return {"status": "success", "image": f"data:image/jpeg;base64,{img_str}"}

@app.post("/set-capture-area")
async def set_capture_area(area: CaptureArea):
    """Set the screen capture area"""
    CONFIG["capture_area"]["top"] = area.top
    CONFIG["capture_area"]["left"] = area.left
    CONFIG["capture_area"]["width"] = area.width
    CONFIG["capture_area"]["height"] = area.height
    
    return {"status": "success", "message": "Capture area updated"}

@app.get("/start-capture")
async def start_capture(background_tasks: BackgroundTasks):
    """Start continuous screen capture and prediction"""
    background_tasks.add_task(continuous_capture, background_tasks)
    return {"status": "started", "message": "Screen capture started"}

@app.get("/stop-capture")
async def stop_capture():
    """Stop continuous screen capture"""
    global is_capturing
    is_capturing = False
    return {"status": "stopped", "message": "Screen capture stopped"}


# Run the FastAPI app if executing directly
if __name__ == "__main__":
    # Make sure get_model is called to load model
    get_model()
    
    # Start the FastAPI server with uvicorn
    print("Starting FastAPI server...")
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False)