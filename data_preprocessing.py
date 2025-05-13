import os
import cv2
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

def preprocess_car_images(dataset_path, output_path, target_size=(224, 224), margin_percent=10):
    """
    Preprocess car images to:
    1. Detect car by separating background
    2. Crop to car boundaries with margin
    3. Center car in new canvas
    4. Resize while preserving aspect ratio
    """
    # Create output directory
    os.makedirs(output_path, exist_ok=True)
    
    # Get list of all images
    image_files = [f for f in os.listdir(dataset_path) if f.endswith(('.png', '.jpg', '.jpeg'))]
    
    print(f"Processing {len(image_files)} images...")
    
    for img_file in tqdm(image_files):
        # Read image
        img_path = os.path.join(dataset_path, img_file)
        img = cv2.imread(img_path)
        
        if img is None:
            print(f"Warning: Could not read {img_file}")
            continue
        
        # Get image dimensions
        h, w = img.shape[:2]
        
        # Convert to grayscale for processing
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Find the background color (most common color in corners)
        corners = [
            gray[0:10, 0:10],      # top-left
            gray[0:10, w-10:w],     # top-right
            gray[h-10:h, 0:10],     # bottom-left
            gray[h-10:h, w-10:w]    # bottom-right
        ]
        
        # Flatten corners and find most common value
        corner_pixels = np.concatenate([c.flatten() for c in corners])
        bg_color = int(np.median(corner_pixels))  # Use int instead of np.uint8
        
        # Create mask: non-background pixels
        # Fix: Use scalar values for lower and upper bounds
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
            cropped = img[y1:y2, x1:x2]
            
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
            canvas[pos_y:pos_y+new_h, pos_x:pos_x+new_w] = resized
            
            # Save the processed image
            output_file = os.path.join(output_path, img_file)
            cv2.imwrite(output_file, canvas)
        else:
            print(f"Warning: No car detected in {img_file}")
            # Copy original image if no car detected (shouldn't happen in this dataset)
            output_file = os.path.join(output_path, img_file)
            cv2.imwrite(output_file, img)
    
    print("Processing complete!")

# Example usage:
preprocess_car_images(
    dataset_path="car_state_dataset_multilabel_large/images",
    output_path="car_state_dataset_preprocessed/images",
    target_size=(224, 224),
    margin_percent=15
)