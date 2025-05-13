import os
import time
import numpy as np
import itertools
import random
import datetime
from tqdm.auto import tqdm
import wandb
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager

class CarModelMultiLabelDataCollector:
    def __init__(self, url, dataset_path="car_state_dataset_multilabel_small", use_wandb=True):
        """Initialize the data collector with target URL and dataset storage path."""
        self.url = url
        self.dataset_path = dataset_path
        self.driver = None
        self.canvas = None
        self.buttons = {}
        self.component_states = {}
        self.use_wandb = use_wandb
        
        # Create dataset directory if it doesn't exist
        os.makedirs(self.dataset_path, exist_ok=True)
        os.makedirs(os.path.join(self.dataset_path, "images"), exist_ok=True)
        
        # Create labels file
        self.labels_file = os.path.join(self.dataset_path, "labels.csv")
        with open(self.labels_file, 'w') as f:
            f.write("filename,front_left_door,front_right_door,rear_left_door,rear_right_door,hood\n")
        
        # Timing configuration parameters
        self.page_load_time = 8.0
        self.view_stabilization_time = 0.5
        self.animation_time = 0.2
        self.post_action_delay = 0.2
        self.pre_screenshot_delay = 1.0
        
        # Component indices for labels
        self.component_indices = {
            "Front Left Door": 0,
            "Front Right Door": 1,
            "Rear Left Door": 2,
            "Rear Right Door": 3,
            "Hood": 4
        }
        
        # Global variables for canvas positioning
        self.canvas_center_x = 0
        self.canvas_center_y = 0
        self.canvas_width = 0
        self.canvas_height = 0
        
        # Variables for tracking statistics
        self.collection_start_time = None
        self.collection_stats = {
            "total_images": 0,
            "views_completed": 0,
            "component_states": {comp: {"open": 0, "closed": 0} for comp in self.component_indices.keys()},
            "elevation_distribution": {},
            "azimuth_distribution": {},
            "zoom_distribution": {},
            "pan_distribution": {},
            "errors": []
        }
        
        # Initialize wandb
        if self.use_wandb:
            self.init_wandb()
    
    def init_wandb(self):
        """Initialize Weights & Biases for experiment tracking."""
        try:
            run_name = f"car_data_collection_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
            wandb.init(
                project="car-component-detection",
                name=run_name,
                config={
                    "dataset_path": self.dataset_path,
                    "url": self.url,
                    "timing_params": {
                        "page_load_time": self.page_load_time,
                        "view_stabilization_time": self.view_stabilization_time,
                        "animation_time": self.animation_time,
                        "post_action_delay": self.post_action_delay,
                        "pre_screenshot_delay": self.pre_screenshot_delay
                    }
                }
            )
            print("Weights & Biases initialized successfully")
        except Exception as e:
            print(f"Error initializing Weights & Biases: {e}")
            self.use_wandb = False
    
    def setup_browser(self):
        """Initialize and set up the browser session."""
        chrome_options = Options()
        chrome_options.add_argument("--start-maximized")
        chrome_options.add_argument("--disable-infobars")
        chrome_options.add_argument("--disable-extensions")
        
        service = Service(ChromeDriverManager().install())
        self.driver = webdriver.Chrome(service=service, options=chrome_options)
        
        self.driver.get(self.url)
        print(f"Navigating to {self.url}")
        
        # Wait for the page to load completely
        time.sleep(self.page_load_time)
        print("Page load wait complete")
        
    def locate_elements(self):
        """Locate the canvas and control buttons on the page based on the HTML structure."""
        try:
            # Find the 3D canvas element (Three.js canvas)
            self.canvas = WebDriverWait(self.driver, 10).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, "canvas[data-engine='three.js r175']"))
            )
            print("Canvas element found")
            
            # Store canvas dimensions
            canvas_size = self.canvas.size
            self.canvas_width = canvas_size['width']
            self.canvas_height = canvas_size['height']
            self.canvas_center_x = self.canvas_width / 2
            self.canvas_center_y = self.canvas_height / 2
            
            print(f"Canvas dimensions: {self.canvas_width}x{self.canvas_height}")
            
            # Find all buttons in the control panel
            button_elements = self.driver.find_elements(By.CSS_SELECTOR, 
                "div[style*='position: absolute'] > button[style*='margin: 10px']")
            
            if len(button_elements) != 5:
                print(f"Warning: Expected 5 buttons, found {len(button_elements)}")
            
            # Map buttons to their components based on text content
            component_names = list(self.component_indices.keys())
            
            for button in button_elements:
                button_text = button.text.strip()
                if button_text in component_names:
                    self.buttons[button_text] = button
                    self.component_states[button_text] = False  # False = Closed, True = Open
                    print(f"Found control button for {button_text}")
            
            # Verify we found all buttons
            if len(self.buttons) != 5:
                print(f"Warning: Only mapped {len(self.buttons)} of 5 expected buttons")
                print(f"Found buttons for: {', '.join(self.buttons.keys())}")
                
        except Exception as e:
            print(f"Error locating elements: {e}")
            self.log_error("locate_elements", str(e))
            self.cleanup()
    
    def log_error(self, function_name, error_message):
        """Log an error to the collection stats and wandb."""
        error_info = {
            "timestamp": datetime.datetime.now().isoformat(),
            "function": function_name,
            "message": error_message
        }
        self.collection_stats["errors"].append(error_info)
        
        if self.use_wandb:
            wandb.log({"error": error_info})
    
    def get_safe_click_position(self):
        """Get a safe position in the bottom right corner of the canvas to avoid clicking car parts."""
        # Calculate a position in the bottom right corner (80% across, 80% down)
        safe_x = int(self.canvas_width * 0.8)
        safe_y = int(self.canvas_height * 0.8)
        
        return safe_x, safe_y
    
    def set_camera_view(self, angle_x, angle_y, zoom_level, pan_x=0, pan_y=0):
        """Set the camera to a specific view based on angle, zoom and pan.
        Uses safe positions to avoid accidental interaction with car components."""
        try:
            # Get safe position for mouse operations
            safe_x, safe_y = self.get_safe_click_position()
            
            # Calculate safe position offset from center
            safe_offset_x = safe_x - self.canvas_center_x
            safe_offset_y = safe_y - self.canvas_center_y
            
            # Add initial delay for stability
            time.sleep(self.post_action_delay)
            
            # Store initial component states to verify no accidental clicks
            initial_states = self.component_states.copy()
            
            # ROTATION: Use bottom right corner for safe dragging
            actions = ActionChains(self.driver)
            
            # Move to safe position first (bottom right corner)
            actions.move_to_element(self.canvas)
            actions.move_by_offset(safe_offset_x, safe_offset_y)
            actions.perform()
            time.sleep(0.2)
            
            # Perform the drag operation from safe position
            actions = ActionChains(self.driver)
            actions.move_to_element(self.canvas)
            actions.move_by_offset(safe_offset_x, safe_offset_y)
            actions.click_and_hold()
            time.sleep(0.2)
            actions.move_by_offset(angle_x, angle_y)
            time.sleep(0.2)
            actions.release()
            actions.perform()
            
            # Wait for camera movement to stabilize
            time.sleep(self.view_stabilization_time)
            
            # PAN: Apply panning if requested
            if pan_x != 0 or pan_y != 0:
                # Move back to safe position
                actions = ActionChains(self.driver)
                actions.move_to_element(self.canvas)
                actions.move_by_offset(safe_offset_x, safe_offset_y)
                actions.perform()
                time.sleep(0.2)
                
                # Right-click and drag for panning
                actions = ActionChains(self.driver)
                actions.move_to_element(self.canvas)
                actions.move_by_offset(safe_offset_x, safe_offset_y)
                actions.context_click()  # Right click
                time.sleep(0.2)
                actions.click_and_hold()
                time.sleep(0.2)
                actions.move_by_offset(pan_x, pan_y)
                time.sleep(0.2)
                actions.release()
                actions.perform()
                
                # Wait for camera pan to stabilize
                time.sleep(self.view_stabilization_time)
            
            # ZOOM: Apply zoom from safe position
            if zoom_level != 0:
                # Move back to safe position
                actions = ActionChains(self.driver)
                actions.move_to_element(self.canvas)
                actions.move_by_offset(safe_offset_x, safe_offset_y)
                actions.perform()
                time.sleep(0.2)
                
                # Apply zoom in smaller increments with delays
                increments = abs(zoom_level)
                direction = 120 if zoom_level > 0 else -120
                
                for _ in range(increments):
                    actions = ActionChains(self.driver)
                    actions.move_to_element(self.canvas)
                    actions.move_by_offset(safe_offset_x, safe_offset_y)
                    actions.scroll_by_amount(0, direction)
                    actions.perform()
                    time.sleep(0.3)  # Delay between scroll events
                
                # Wait for zoom to stabilize
                time.sleep(self.view_stabilization_time)
            
            # Check that we didn't accidentally change any component states
            for component, state in self.component_states.items():
                if state != initial_states[component]:
                    print(f"WARNING: {component} state accidentally changed during camera movement!")
                    # Reset the state
                    self.set_component_state(component, initial_states[component])
                
            print(f"Set camera view: angle({angle_x}, {angle_y}), zoom({zoom_level}), pan({pan_x}, {pan_y})")
            
        except Exception as e:
            print(f"Error setting camera view: {e}")
            self.log_error("set_camera_view", str(e))
    
    def set_component_state(self, component, desired_state):
        """Set a component to the desired state (True=Open, False=Closed)."""
        current_state = self.component_states[component]
        if current_state != desired_state:
            # Wait before clicking
            time.sleep(self.post_action_delay)
            
            # Click the component button
            # print(f"Clicking {component} button to change state")
            self.buttons[component].click()
            
            # Wait for animation to complete
            time.sleep(self.animation_time)
            
            # Update component state
            self.component_states[component] = desired_state
            # print(f"Set {component} to {'Open' if desired_state else 'Closed'}")
            
            # Additional stabilization time
            time.sleep(self.post_action_delay)
    
    def apply_component_states(self, states_config):
        """Apply a specific configuration of component states.
        states_config: Dict mapping component name to desired state (True=Open, False=Closed)
        """
        # First close everything for consistency
        for component in self.buttons.keys():
            if self.component_states[component] == True:  # If open
                self.set_component_state(component, False)  # Close it
        
        # Then set the desired states
        for component, desired_state in states_config.items():
            if desired_state == True:  # Only need to open doors that should be open
                self.set_component_state(component, desired_state)
        
        # Final stabilization delay
        time.sleep(self.post_action_delay)
    
    def capture_multi_label_data(self, filename, states, view_params=None):
        """Capture screenshot and record multi-label information using DOM manipulation
        to hide UI elements before taking the screenshot."""
        # Wait before screenshot to ensure UI is stable
        time.sleep(self.pre_screenshot_delay)
        
        try:
            # IMPORTANT: Hide all UI elements before taking the screenshot
            self.driver.execute_script("""
                // Store the original display states to restore later
                window.originalDisplayStates = {};
                
                // Hide all UI control elements (buttons, panels, etc.)
                var uiElements = document.querySelectorAll('button, div[style*="position: absolute"]');
                
                for (var i = 0; i < uiElements.length; i++) {
                    var element = uiElements[i];
                    
                    // Skip the canvas element itself
                    if (element.tagName.toLowerCase() === 'canvas') continue;
                    
                    // Store original display state
                    window.originalDisplayStates[i] = element.style.display;
                    
                    // Hide the element
                    element.style.display = 'none';
                }
                
                // Additional backup selector for any UI controls we might have missed
                var additionalControls = document.querySelectorAll('.ui-control, .button, .control-panel');
                for (var j = 0; j < additionalControls.length; j++) {
                    var control = additionalControls[j];
                    window.originalDisplayStates['additional_' + j] = control.style.display;
                    control.style.display = 'none';
                }
                
                // Remove any overlay texts or notifications
                var overlays = document.querySelectorAll('.overlay, .notification, .message');
                for (var k = 0; k < overlays.length; k++) {
                    var overlay = overlays[k];
                    window.originalDisplayStates['overlay_' + k] = overlay.style.display;
                    overlay.style.display = 'none';
                }
                
                return "UI elements hidden successfully";
            """)
            
            # print("UI elements temporarily hidden for clean screenshot")
            
            # Small delay to ensure UI hiding has been applied
            time.sleep(0.2)
            
            # Save the screenshot with only the 3D model visible
            image_path = os.path.join(self.dataset_path, "images", filename)
            self.driver.save_screenshot(image_path)
            
        except Exception as e:
            print(f"Error during UI hiding or screenshot: {e}")
            self.log_error("capture_multi_label_data", str(e))
            
        finally:
            # IMPORTANT: Always restore UI elements regardless of errors
            try:
                # Restore all UI elements
                self.driver.execute_script("""
                    // Restore all elements to their original display state
                    var uiElements = document.querySelectorAll('button, div[style*="position: absolute"]');
                    
                    for (var i = 0; i < uiElements.length; i++) {
                        var element = uiElements[i];
                        
                        // Skip the canvas element itself
                        if (element.tagName.toLowerCase() === 'canvas') continue;
                        
                        // Restore original display state if it was stored
                        if (window.originalDisplayStates && window.originalDisplayStates[i] !== undefined) {
                            element.style.display = window.originalDisplayStates[i];
                        } else {
                            // Default to block if original state wasn't stored
                            element.style.display = '';
                        }
                    }
                    
                    // Restore additional controls
                    var additionalControls = document.querySelectorAll('.ui-control, .button, .control-panel');
                    for (var j = 0; j < additionalControls.length; j++) {
                        var control = additionalControls[j];
                        if (window.originalDisplayStates && window.originalDisplayStates['additional_' + j] !== undefined) {
                            control.style.display = window.originalDisplayStates['additional_' + j];
                        } else {
                            control.style.display = '';
                        }
                    }
                    
                    // Restore overlays
                    var overlays = document.querySelectorAll('.overlay, .notification, .message');
                    for (var k = 0; k < overlays.length; k++) {
                        var overlay = overlays[k];
                        if (window.originalDisplayStates && window.originalDisplayStates['overlay_' + k] !== undefined) {
                            overlay.style.display = window.originalDisplayStates['overlay_' + k];
                        } else {
                            overlay.style.display = '';
                        }
                    }
                    
                    // Clean up
                    delete window.originalDisplayStates;
                    
                    return "UI elements restored successfully";
                """)
                # print("UI elements restored after screenshot")
            except Exception as e:
                print(f"Error restoring UI elements: {e}")
                self.log_error("restore_UI_elements", str(e))
        
        # Create label vector (1=Open, 0=Closed)
        label_vector = [0, 0, 0, 0, 0]  # Default all closed
        for component, is_open in states.items():
            if is_open:
                index = self.component_indices[component]
                label_vector[index] = 1
                self.collection_stats["component_states"][component]["open"] += 1
            else:
                self.collection_stats["component_states"][component]["closed"] += 1
        
        # Write to labels file
        with open(self.labels_file, 'a') as f:
            f.write(f"{filename},{','.join(map(str, label_vector))}\n")
        
        # Create states string
        states_display = []
        for component, is_open in states.items():
            state_text = "Open" if is_open else "Closed"
            states_display.append(f"{component}:{state_text}")
        
        # Update collection stats
        self.collection_stats["total_images"] += 1
        
        # Update distribution stats if view parameters are provided
        if view_params:
            elevation = view_params.get("elevation")
            azimuth = view_params.get("azimuth")
            zoom = view_params.get("zoom")
            pan = view_params.get("pan")
            
            # Update distributions
            if elevation is not None:
                self.collection_stats["elevation_distribution"][elevation] = \
                    self.collection_stats["elevation_distribution"].get(elevation, 0) + 1
            
            if azimuth is not None:
                self.collection_stats["azimuth_distribution"][azimuth] = \
                    self.collection_stats["azimuth_distribution"].get(azimuth, 0) + 1
            
            if zoom is not None:
                self.collection_stats["zoom_distribution"][zoom] = \
                    self.collection_stats["zoom_distribution"].get(zoom, 0) + 1
            
            if pan is not None:
                pan_key = f"{pan[0]},{pan[1]}"
                self.collection_stats["pan_distribution"][pan_key] = \
                    self.collection_stats["pan_distribution"].get(pan_key, 0) + 1
        
        # Log to wandb if enabled
        if self.use_wandb and self.collection_stats["total_images"] % 25 == 0:  # Log every 25 images
            self.log_to_wandb(image_path, states, view_params)
        
        # print(f"Captured {filename} with states: {' '.join(states_display)}")
        return True
    
    def log_to_wandb(self, image_path, states, view_params=None):
        """Log metrics and images to Weights & Biases."""
        if not self.use_wandb:
            return
        
        try:
            # Calculate images per minute
            elapsed_time = (time.time() - self.collection_start_time) / 60  # minutes
            images_per_minute = self.collection_stats["total_images"] / max(0.1, elapsed_time)
            
            # Create log dictionary
            log_dict = {
                "total_images": self.collection_stats["total_images"],
                "views_completed": self.collection_stats["views_completed"],
                "images_per_minute": images_per_minute,
                "elapsed_minutes": elapsed_time
            }
            
            # Add component state distributions
            for component, states_count in self.collection_stats["component_states"].items():
                for state, count in states_count.items():
                    log_dict[f"states/{component}_{state}"] = count
            
            # Log sample image occasionally
            if self.collection_stats["total_images"] % 100 == 0:  # Every 100 images
                # Format state information for image caption
                states_text = ", ".join([f"{c}:{'Open' if s else 'Closed'}" for c, s in states.items()])
                
                # Add view parameters to caption if available
                view_text = ""
                if view_params:
                    view_text = f"Elevation: {view_params.get('elevation')}, " + \
                               f"Azimuth: {view_params.get('azimuth')}, " + \
                               f"Zoom: {view_params.get('zoom')}, " + \
                               f"Pan: {view_params.get('pan')}"
                
                # Create image caption
                caption = f"Image {self.collection_stats['total_images']} | {states_text} | {view_text}"
                
                # Log the image
                log_dict["sample_image"] = wandb.Image(image_path, caption=caption)
            
            # Log to wandb
            wandb.log(log_dict)
            
        except Exception as e:
            print(f"Error logging to wandb: {e}")
            self.log_error("log_to_wandb", str(e))
    
    def generate_state_combinations(self, count=15):
        """Generate a diverse set of component state combinations."""
        components = list(self.buttons.keys())
        state_combinations = []
        
        # 1. Add baseline (all closed)
        all_closed = {comp: False for comp in components}
        state_combinations.append(all_closed)
        
        # 2. Add all individual component open states
        for component in components:
            one_open = {comp: (comp == component) for comp in components}
            state_combinations.append(one_open)
        
        # 3. Add various combinations of two components open
        two_component_combinations = list(itertools.combinations(components, 2))
        for combo in two_component_combinations:
            two_open = {comp: (comp in combo) for comp in components}
            state_combinations.append(two_open)
        
        # 4. Add various combinations of three components open
        three_component_combinations = list(itertools.combinations(components, 3))
        selected_threes = random.sample(three_component_combinations, 
                                     min(5, len(three_component_combinations)))
        for combo in selected_threes:
            three_open = {comp: (comp in combo) for comp in components}
            state_combinations.append(three_open)
        
        # 5. Add all open
        all_open = {comp: True for comp in components}
        state_combinations.append(all_open)
        
        # 6. Add random combinations to reach desired count
        while len(state_combinations) < count:
            random_state = {comp: random.choice([True, False]) for comp in components}
            # Avoid duplicates
            if random_state not in state_combinations:
                state_combinations.append(random_state)
        
        # Randomize order
        random.shuffle(state_combinations)
        
        # Return up to the requested count
        return state_combinations[:count]
    
    def collect_multi_label_data(self):
        """Collect comprehensive dataset with various combinations of component states."""
        try:
            # Start timing the collection process
            self.collection_start_time = time.time()
            
            elevation_angles = np.linspace(0, 9, 10)  # 10 different height angles
            azimuth_angles = np.linspace(0, 150, 15)      # 15 different rotations around car
            zoom_levels = [0]                     
            pan_variations = [(0, 0)]
            
            # Number of state combinations to capture per camera view
            combinations_per_view = 32  
            
            # Calculate total views for progress reporting
            total_views = len(elevation_angles) * len(azimuth_angles) * len(zoom_levels) * len(pan_variations)
            total_images = total_views * combinations_per_view
            
            print("Elevation angles: ")
            print(elevation_angles)
            
            print("Azimuth angles: ")
            print(azimuth_angles)
            
            print(f"Starting ENHANCED data collection with {total_views} camera views!")
            print(f"Each view will have {combinations_per_view} state combinations")
            print(f"Expected dataset size: {total_images} images")
            
            # Initialize tqdm for overall progress
            with tqdm(total=total_images, desc="Total Progress", unit="img") as pbar_total:
                # For each camera angle, zoom level, and pan variation
                for elevation_idx, elevation in enumerate(elevation_angles):
                    for azimuth_idx, azimuth in enumerate(azimuth_angles):
                        for zoom_idx, zoom in enumerate(zoom_levels):
                            # Initialize tqdm for pan variations
                            pan_desc = f"Elev:{elevation:.1f}° Azim:{azimuth:.1f}° Zoom:{zoom}"
                            with tqdm(pan_variations, desc=pan_desc, leave=False) as pbar_pan:
                                for pan_x, pan_y in pbar_pan:
                                    # Calculate rotation values
                                    angle_x = np.cos(np.radians(azimuth)) * 10
                                    angle_y = np.sin(np.radians(elevation)) * 10
                                    
                                    # # Add some random jitter to angles for more natural variations
                                    # jitter_x = random.uniform(-10, 10)
                                    # jitter_y = random.uniform(-10, 10)
                                    # angle_x += jitter_x
                                    # angle_y += jitter_y
                                    
                                    # Set the camera view with pan
                                    self.set_camera_view(azimuth, elevation, zoom, pan_x, pan_y)
                                    
                                    # Generate state combinations for this view
                                    state_combinations = self.generate_state_combinations(combinations_per_view)
                                    
                                    # Capture each combination for this camera view
                                    state_desc = f"Pan:({pan_x},{pan_y})"
                                    with tqdm(state_combinations, desc=state_desc, leave=False) as pbar_states:
                                        for combo_idx, states in enumerate(pbar_states):
                                            # Generate unique filename with view parameters encoded
                                            view_id = (elevation_idx * 1000000 + 
                                                      azimuth_idx * 10000 + 
                                                      zoom_idx * 100 +
                                                      pan_variations.index((pan_x, pan_y)))
                                            filename = f"view{view_id:08d}_combo{combo_idx:02d}.png"
                                            
                                            # Apply the component states
                                            self.apply_component_states(states)
                                            
                                            # Create view params dict for logging
                                            view_params = {
                                                "elevation": float(elevation),
                                                "azimuth": float(azimuth),
                                                "zoom": int(zoom),
                                                "pan": (pan_x, pan_y)
                                            }
                                            
                                            # Capture the image and record labels
                                            self.capture_multi_label_data(filename, states, view_params)
                                            
                                            # Update state progress
                                            pbar_states.set_postfix({"img": self.collection_stats["total_images"]})
                                            
                                            # Update total progress
                                            pbar_total.update(1)
                                            
                                            # Update wandb metrics regularly
                                            if self.use_wandb and self.collection_stats["total_images"] % 10 == 0:
                                                elapsed_time = (time.time() - self.collection_start_time) / 60
                                                images_per_minute = self.collection_stats["total_images"] / max(0.1, elapsed_time)
                                                pbar_total.set_postfix({
                                                    "img/min": f"{images_per_minute:.1f}",
                                                    "elapsed": f"{elapsed_time:.1f}min"
                                                })
                                    
                                    # Increment view counter
                                    self.collection_stats["views_completed"] += 1
                                    
                                    # Update pan progress
                                    pbar_pan.set_postfix({"views": self.collection_stats["views_completed"]})
                                    
                                    # Log to wandb after each view is complete
                                    if self.use_wandb:
                                        wandb.log({
                                            "views_completed": self.collection_stats["views_completed"],
                                            "progress_percentage": (self.collection_stats["views_completed"] / total_views) * 100
                                        })
            
            # Collection complete
            elapsed_time = (time.time() - self.collection_start_time) / 60
            print(f"\nData collection complete!")
            print(f"Total images captured: {self.collection_stats['total_images']}")
            print(f"Total time: {elapsed_time:.2f} minutes")
            print(f"Average speed: {self.collection_stats['total_images'] / max(0.1, elapsed_time):.2f} images/minute")
            
            # Final wandb log
            if self.use_wandb:
                wandb.log({
                    "collection_complete": True,
                    "final_image_count": self.collection_stats["total_images"],
                    "total_collection_time_minutes": elapsed_time
                })
            
        except Exception as e:
            print(f"Error during data collection: {e}")
            self.log_error("collect_multi_label_data", str(e))
            raise e
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Close the browser and clean up resources."""
        if self.driver:
            self.driver.quit()
            print("Browser session closed")
        
        # Close wandb run
        if self.use_wandb:
            wandb.finish()
    
    def run(self):
        """Execute the full data collection process."""
        try:
            self.setup_browser()
            self.locate_elements()
            self.collect_multi_label_data()
        except Exception as e:
            print(f"Error in data collection process: {e}")
            self.log_error("run", str(e))
            self.cleanup()
            raise e


if __name__ == "__main__":
    # URL of the 3D car model interface
    TARGET_URL = "https://euphonious-concha-ab5c5d.netlify.app/"
    
    # Create and run the collector
    collector = CarModelMultiLabelDataCollector(url=TARGET_URL, use_wandb=False)
    collector.run()