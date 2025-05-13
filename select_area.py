import cv2
import numpy as np
import mss
import mss.tools
import time
import json
import os

def select_capture_area():
    """
    Interactive tool to select the capture area of the screen.
    Returns the selected area as a dictionary.
    """
    print("Screen Capture Area Selection Tool")
    print("=================================")
    print("1. First, we'll show you screenshots of available monitors.")
    print("2. Select a monitor number and then use mouse to select the area to capture.")
    print("3. The selected area will be saved for the app to use.")
    print()
    
    # Setup mss for screen capture
    with mss.mss() as sct:
        # Get information about available monitors
        for i, monitor in enumerate(sct.monitors):
            print(f"Monitor {i}: {monitor}")
            
            # Capture screenshot of this monitor
            screenshot = np.array(sct.grab(monitor))
            
            # Resize if too large for display
            height, width = screenshot.shape[:2]
            if width > 1200 or height > 800:
                scale_factor = min(1200/width, 800/height)
                new_width = int(width * scale_factor)
                new_height = int(height * scale_factor)
                screenshot = cv2.resize(screenshot, (new_width, new_height))
            
            # Display this monitor
            window_name = f"Monitor {i}"
            cv2.imshow(window_name, screenshot)
            cv2.waitKey(1000)  # Show for 1 second
            cv2.destroyWindow(window_name)
        
        # Ask user to select a monitor
        selected_monitor = int(input("Enter the monitor number to capture from: "))
        if selected_monitor < 0 or selected_monitor >= len(sct.monitors):
            print("Invalid monitor number. Using monitor 0.")
            selected_monitor = 0
        
        # Capture the selected monitor
        monitor = sct.monitors[selected_monitor]
        screenshot = np.array(sct.grab(monitor))
        
        # Resize if too large for display
        height, width = screenshot.shape[:2]
        display_scale = 1.0
        if width > 1200 or height > 800:
            display_scale = min(1200/width, 800/height)
            new_width = int(width * display_scale)
            new_height = int(height * display_scale)
            display_img = cv2.resize(screenshot, (new_width, new_height))
        else:
            display_img = screenshot.copy()
        
        # Variables for selection
        selection = {"top": 0, "left": 0, "width": 0, "height": 0}
        roi_selected = False
        start_point = (0, 0)
        end_point = (0, 0)
        
        # Mouse callback function
        def select_roi(event, x, y, flags, param):
            nonlocal start_point, end_point, roi_selected
            
            if event == cv2.EVENT_LBUTTONDOWN:
                # Reset selection
                roi_selected = False
                start_point = (x, y)
                end_point = (x, y)
                
            elif event == cv2.EVENT_MOUSEMOVE and flags == cv2.EVENT_FLAG_LBUTTON:
                end_point = (x, y)
                
            elif event == cv2.EVENT_LBUTTONUP:
                end_point = (x, y)
                roi_selected = True
                
                # Calculate real coordinates (accounting for scaling)
                left = min(start_point[0], end_point[0]) / display_scale
                top = min(start_point[1], end_point[1]) / display_scale
                right = max(start_point[0], end_point[0]) / display_scale
                bottom = max(start_point[1], end_point[1]) / display_scale
                
                # Update selection variables
                selection["top"] = int(top) + monitor["top"]
                selection["left"] = int(left) + monitor["left"]
                selection["width"] = int(right - left)
                selection["height"] = int(bottom - top)
                
                print(f"Selected area: {selection}")
        
        # Create window for selection
        window_name = "Select Capture Area (drag to select, press Enter when done)"
        cv2.namedWindow(window_name)
        cv2.setMouseCallback(window_name, select_roi)
        
        while True:
            # Create a copy of the display image to draw selection rectangle
            img_copy = display_img.copy()
            
            # Draw the current selection rectangle
            if start_point != end_point:
                cv2.rectangle(
                    img_copy,
                    start_point,
                    end_point,
                    (0, 255, 0),
                    2
                )
            
            # Display information on the image
            info_text = "Drag to select area, press Enter when done"
            cv2.putText(
                img_copy,
                info_text,
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2
            )
            
            if roi_selected:
                # Show selection details
                selection_text = f"Selected: {selection['width']}x{selection['height']}"
                cv2.putText(
                    img_copy,
                    selection_text,
                    (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2
                )
            
            # Show the image with selection rectangle
            cv2.imshow(window_name, img_copy)
            
            # Wait for key press
            key = cv2.waitKey(1) & 0xFF
            if key == 13:  # Enter key
                break
        
        # Close all windows
        cv2.destroyAllWindows()
        
        # Save the selection
        with open("capture_config.json", "w") as f:
            json.dump({
                "monitor": selected_monitor,
                "capture_area": selection
            }, f, indent=4)
        
        print(f"Capture configuration saved to capture_config.json")
        return selection

if __name__ == "__main__":
    select_capture_area()