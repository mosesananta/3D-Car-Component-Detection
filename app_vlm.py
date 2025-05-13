import os
import torch
import numpy as np
from PIL import Image
import time
import base64
import cv2
import json
import mss
from typing import Dict
from fastapi import FastAPI, HTTPException, File, UploadFile, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
from transformers import Idefics3ForConditionalGeneration, AutoProcessor
from peft import PeftModel, PeftConfig

# FastAPI app
app = FastAPI(
    title="Car Component VLM Detection API",
    description="API for detecting car component states using VLM (Vision Language Model)",
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

# Create static and templates directories
os.makedirs("static", exist_ok=True)
os.makedirs("templates", exist_ok=True)
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

# Configuration for VLM model
VLM_CONFIG = {
    "base_model_id": "HuggingFaceTB/SmolVLM-256M-Instruct",
    "adapter_path": "./trained_models/smolvlm-instruct-car-component-detection",
    "processor_id": "HuggingFaceTB/SmolVLM-256M-Instruct",  # Use the same processor as training
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "capture_area": {
        "top": 0,
        "left": 0,
        "width": 800,
        "height": 600
    },
    "monitor": 1  # Monitor to capture (0 is primary)
}

# Try to load capture config if it exists
try:
    if os.path.exists("capture_config.json"):
        with open("capture_config.json", "r") as f:
            config_data = json.load(f)
            VLM_CONFIG["monitor"] = config_data.get("monitor", VLM_CONFIG["monitor"])
            VLM_CONFIG["capture_area"] = config_data.get("capture_area", VLM_CONFIG["capture_area"])
            print(f"Loaded capture configuration: {VLM_CONFIG['capture_area']}")
except Exception as e:
    print(f"Error loading capture config: {e}")

# Global variables
vlm_model = None
vlm_processor = None
latest_description = ""
latest_inference_time = 0.0

# Pydantic models
class CaptureArea(BaseModel):
    top: int
    left: int
    width: int
    height: int

class VLMResponse(BaseModel):
    description: str
    inference_time: float
    image: str

# Load the VLM model
def get_vlm_model():
    global vlm_model, vlm_processor
    if vlm_model is None:
        try:
            print(f"Loading VLM base model from {VLM_CONFIG['base_model_id']}...")
            
            # Load the base model first
            base_model = Idefics3ForConditionalGeneration.from_pretrained(
                VLM_CONFIG['base_model_id'],
                device_map=VLM_CONFIG['device'],
                torch_dtype=torch.float16
            )
            
            # Then load the adapter/LoRA weights
            print(f"Loading LoRA adapter from {VLM_CONFIG['adapter_path']}...")
            vlm_model = PeftModel.from_pretrained(
                base_model,
                VLM_CONFIG['adapter_path'],
                is_trainable=False
            )
            
            # Load the processor
            vlm_processor = AutoProcessor.from_pretrained(VLM_CONFIG['processor_id'])
            
            print(f"VLM model with adapter loaded successfully using {VLM_CONFIG['device']}")
        except Exception as e:
            print(f"Error loading VLM model: {e}")
            # Create a dummy model for testing if real model fails
            vlm_model = "dummy"
    return vlm_model, vlm_processor

# Process car image function from original script
def process_car_image(img_array, target_size=(224, 224), margin_percent=15):
    """
    Apply preprocessing technique used in training:
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

# Screen capture function
def capture_screen():
    with mss.mss() as sct:
        # Get information about the monitor
        monitor = sct.monitors[VLM_CONFIG["monitor"]]
        
        # Use configured capture area
        capture_area = {
            "top": VLM_CONFIG["capture_area"]["top"],
            "left": VLM_CONFIG["capture_area"]["left"],
            "width": VLM_CONFIG["capture_area"]["width"],
            "height": VLM_CONFIG["capture_area"]["height"],
            "mon": VLM_CONFIG["monitor"],
        }
        
        # Capture the screen
        sct_img = sct.grab(capture_area)
        
        # Convert to numpy array
        img = np.array(sct_img)
        
        # Convert BGRA to BGR by slicing off the alpha channel
        img = img[:, :, :3]
        
        return img


# VLM inference function
async def generate_vlm_description(image):
    global latest_description, latest_inference_time
    
    model, processor = get_vlm_model()
    
    if model == "dummy":
        # Return dummy response for testing
        return "This is a dummy VLM description of the car state. The car's front left door and hood are open, while the other doors are closed.", 0.01
    
    # Process the image for VLM
    if isinstance(image, np.ndarray):
        # Convert from OpenCV format to PIL
        image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    elif isinstance(image, str):
        # If it's a filepath
        image = Image.open(image).convert('RGB')
    
    # Create a properly formatted sample that matches your training format
    query = "Describe the current state of the car doors and hood."
    
    sample = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": image,
                },
                {
                    "type": "text",
                    "text": query
                }
            ],
        },
        {
            "role": "assistant",
            "content": [
                {
                    "type": "text",
                    "text": ""  # Empty for generation
                }
            ],
        },
    ]
    
    # Prepare the text input by applying the chat template
    text_input = processor.apply_chat_template(
        sample,  # Use the formatted sample
        add_generation_prompt=True
    )
    
    image_inputs = []
    image_inputs.append([image])
    
    # Generate response
    start_time = time.time()
    
    try:
        # Prepare the inputs for the model
        model_inputs = processor(
            text=text_input,
            images=image_inputs,
            return_tensors="pt",
        ).to(VLM_CONFIG["device"])
        
        with torch.no_grad():
            # Generate text with the model
            generated_ids = model.generate(
                **model_inputs, 
                max_new_tokens=100,
                do_sample=False
            )
            
            # Trim the generated ids to remove the input ids
            trimmed_generated_ids = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(model_inputs.input_ids, generated_ids)
            ]
            
            # Decode the output text
            answer = processor.batch_decode(
                trimmed_generated_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False
            )[0]
        
        inference_time = time.time() - start_time
        
        # Update global state
        latest_description = answer
        latest_inference_time = inference_time
        
        return answer, inference_time
    
    except Exception as e:
        print(f"VLM inference error: {e}")
        return f"Error during VLM inference: {str(e)}", 0.0

# Convert OpenCV image to base64
def cv2_to_base64(cv2_image):
    _, buffer = cv2.imencode(".jpg", cv2_image)
    img_str = base64.b64encode(buffer).decode("utf-8")
    return f"data:image/jpeg;base64,{img_str}"


# Routes
@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    """Serve the main HTML page"""
    
    return templates.TemplateResponse(
        "vlm_inference.html", 
        {"request": request}
    )

@app.get("/get-capture-settings")
async def get_capture_settings():
    """Get the current capture settings"""
    return {
        "top": VLM_CONFIG["capture_area"]["top"],
        "left": VLM_CONFIG["capture_area"]["left"],
        "width": VLM_CONFIG["capture_area"]["width"],
        "height": VLM_CONFIG["capture_area"]["height"],
        "monitor": VLM_CONFIG["monitor"]
    }

@app.post("/set-capture-settings")
async def set_capture_settings(settings: Dict):
    """Set the screen capture settings"""
    try:
        VLM_CONFIG["capture_area"]["top"] = settings["top"]
        VLM_CONFIG["capture_area"]["left"] = settings["left"]
        VLM_CONFIG["capture_area"]["width"] = settings["width"]
        VLM_CONFIG["capture_area"]["height"] = settings["height"]
        VLM_CONFIG["monitor"] = settings["monitor"]
        
        # Save to capture_config.json for persistence
        with open("capture_config.json", "w") as f:
            json.dump({
                "monitor": VLM_CONFIG["monitor"],
                "capture_area": VLM_CONFIG["capture_area"]
            }, f, indent=4)
        
        return {"status": "success", "message": "Capture settings updated"}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.get("/capture-and-analyze")
async def capture_and_analyze():
    """Capture a single image and analyze it with VLM"""
    try:
        # Capture image
        img = capture_screen()
        
        # Save the raw captured image
        raw_path = os.path.join("static", "latest_raw.jpg")
        cv2.imwrite(raw_path, img)
        
        # Apply preprocessing
        processed_img = process_car_image(
            img, 
            target_size=(224, 224),
            margin_percent=15
        )
        
        # Save the preprocessed image
        processed_path = os.path.join("static", "latest_processed.jpg")
        cv2.imwrite(processed_path, processed_img)
        
        # Process with VLM
        description, inference_time = await generate_vlm_description(processed_img)
        
        # Create base64 image data
        img_base64 = cv2_to_base64(processed_img)
        
        return {
            "image": img_base64,
            "description": description,
            "inference_time": inference_time
        }
    except Exception as e:
        print(f"Error in capture and analyze: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/analyze-image")
async def analyze_image(image: UploadFile = File(...)):
    """Analyze an uploaded image using VLM"""
    try:
        # Read the image
        contents = await image.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # Save the uploaded image
        upload_path = os.path.join("static", "uploaded.jpg")
        cv2.imwrite(upload_path, img)
        
        # Apply preprocessing
        processed_img = process_car_image(
            img, 
            target_size=(224, 224),
            margin_percent=15
        )
        
        # Save the preprocessed image
        processed_path = os.path.join("static", "uploaded_processed.jpg")
        cv2.imwrite(processed_path, processed_img)
        
        # Process with VLM
        description, inference_time = await generate_vlm_description(processed_img)
        
        # Create base64 image data
        img_base64 = cv2_to_base64(processed_img)
        
        return {
            "image": img_base64,
            "description": description,
            "inference_time": inference_time
        }
    except Exception as e:
        print(f"Error analyzing image: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/latest-description")
async def get_latest_description():
    """Get the latest VLM description"""
    global latest_description, latest_inference_time
    
    return {
        "description": latest_description,
        "inference_time": latest_inference_time
    }

# Run the FastAPI app if executing directly
if __name__ == "__main__":
    # Load the VLM model
    get_vlm_model()
    
    # Start the FastAPI server with uvicorn
    print("Starting VLM inference server...")
    uvicorn.run("app_vlm:app", host="0.0.0.0", port=8000, reload=False)