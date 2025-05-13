import os
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
class CarComponentDataset(Dataset):
    """Dataset for car component state detection."""
    
    def __init__(self, data_path, labels_df, transform=None):
        """
        Args:
            data_path (str): Path to the dataset directory
            labels_df (DataFrame): DataFrame with image filenames and labels
            transform (callable, optional): Optional transform to be applied to images
        """
        self.data_path = data_path
        self.labels_df = labels_df.reset_index(drop=True)  # Reset index to avoid issues
        self.transform = transform
        
        # Create a mapping of state combination to sample indices
        self.state_to_indices = {}
        for i, row in self.labels_df.iterrows():
            state = ''.join(map(str, row.iloc[1:].astype(int).tolist()))
            if state not in self.state_to_indices:
                self.state_to_indices[state] = []
            self.state_to_indices[state].append(i)
        
    def __len__(self):
        return len(self.labels_df)
    
    def __getitem__(self, idx):
        # Get image filename and full path
        img_name = self.labels_df.iloc[idx, 0]
        img_path = os.path.join(self.data_path, "images", img_name)
        
        # Load image
        image = Image.open(img_path).convert('RGB')
        
        # Apply transforms if any
        if self.transform:
            image = self.transform(image)
        
        # Get labels (all columns except the first one which is the filename)
        labels = self.labels_df.iloc[idx, 1:].values.astype(np.float32)
        
        return image, torch.tensor(labels)