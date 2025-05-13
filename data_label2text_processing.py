import pandas as pd
import os

def create_descriptive_labels(input_csv_path, output_csv_path):
    """
    Convert binary labels to descriptive text for VLM/MLLM training
    
    Args:
        input_csv_path: Path to the original label.csv with binary values
        output_csv_path: Path to save the new CSV with text descriptions
    """
    # Read the original labels file
    df = pd.read_csv(input_csv_path)
    
    # Component names in a more natural order for text generation
    components = [
        "front left door",
        "front right door", 
        "rear left door", 
        "rear right door", 
        "hood"
    ]
    
    # Column names in the original CSV
    column_names = [
        "front_left_door",
        "front_right_door",
        "rear_left_door",
        "rear_right_door",
        "hood"
    ]
    
    # Create a new dataframe for the text labels
    text_df = pd.DataFrame()
    text_df['filename'] = df['filename']
    text_df['text_description'] = ""
    
    # Process each row
    for idx, row in df.iterrows():
        # Get the status of each component (0=Closed, 1=Open)
        statuses = [int(row[col]) for col in column_names]
        
        # Identify which components are open and which are closed
        open_components = [components[i] for i in range(len(components)) if statuses[i] == 1]
        closed_components = [components[i] for i in range(len(components)) if statuses[i] == 0]
        
        # Generate the descriptive text
        if len(open_components) == 0:
            # All components are closed
            description = "All doors and the hood of the car are closed."
        elif len(closed_components) == 0:
            # All components are open
            description = "All doors and the hood of the car are open."
        else:
            # Some components are open, some are closed
            if len(open_components) == 1:
                open_text = f"The car's {open_components[0]} is open."
            elif len(open_components) == 2:
                open_text = f"The car's {open_components[0]} and {open_components[1]} are open."
            else:
                # For 3 or more components, use comma formatting with "and" before the last item
                open_list = ", ".join(open_components[:-1]) + f", and {open_components[-1]}"
                open_text = f"The car's {open_list} are open."
            
            if len(closed_components) == 1:
                closed_text = f"The {closed_components[0]} remains closed."
            elif len(closed_components) == 2:
                closed_text = f"The {closed_components[0]} and {closed_components[1]} remain closed."
            else:
                # For 3 or more components, use comma formatting with "and" before the last item
                closed_list = ", ".join(closed_components[:-1]) + f", and {closed_components[-1]}"
                closed_text = f"The {closed_list} remain closed."
            
            description = f"{open_text} {closed_text}"
        
        # Store the description
        text_df.at[idx, 'text_description'] = description
    
    # Save the new CSV file
    text_df.to_csv(output_csv_path, index=False)
    print(f"Created text description labels at: {output_csv_path}")
    
    # Print a few examples
    print("\nExample Descriptions:")
    for i in range(min(5, len(text_df))):
        print(f"Image: {text_df.iloc[i]['filename']}")
        print(f"Description: {text_df.iloc[i]['text_description']}")
        print()

if __name__ == "__main__":
    # Paths
    dataset_path = "car_state_dataset_multilabel_large"  # Update this to your dataset path
    input_csv = os.path.join(dataset_path, "labels.csv")
    output_csv = os.path.join(dataset_path, "text_labels.csv")
    
    # Create the text labels
    create_descriptive_labels(input_csv, output_csv)