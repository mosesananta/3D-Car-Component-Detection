import os
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import pandas as pd

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
    
class CarLLaVADataset(Dataset):
    """Dataset for training LLaVA-style car component model"""
    
    def __init__(self, 
                 image_dir, 
                 text_labels_path, 
                 tokenizer, 
                 transform=None,
                 max_length=256):
        self.image_dir = image_dir
        self.text_df = pd.read_csv(text_labels_path)
        self.tokenizer = tokenizer
        self.transform = transform
        self.max_length = max_length
        
    def __len__(self):
        return len(self.text_df)
    
    def __getitem__(self, idx):
        # Get image and text description
        img_name = self.text_df.iloc[idx]['filename']
        description = self.text_df.iloc[idx]['text_description']
        
        # Load and transform image
        img_path = os.path.join(self.image_dir, "images", img_name)
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        
        # Create chat messages for SmolLM2
        messages = [
            {"role": "user", "content": "Examine this car image and describe which doors and hood are open or closed.\n<image></image>"},
            {"role": "assistant", "content": description}
        ]
        
        # Use the model's chat template to format the messages
        input_text = self.tokenizer.apply_chat_template(messages, tokenize=False)
        
        # Tokenize
        encodings = self.tokenizer(
            input_text,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.max_length
        )
        
        input_ids = encodings.input_ids.squeeze(0)
        attention_mask = encodings.attention_mask.squeeze(0)
        
        # Create labels for autoregressive training
        labels = input_ids.clone()
        
        # Set -100 (ignore index) for all tokens from the user message
        # Find the first assistant message token index
        assistant_idx = input_text.find("assistant")
        if assistant_idx != -1:
            # Find the token position approximately
            assistant_enc = self.tokenizer.encode(input_text[:assistant_idx], add_special_tokens=False)
            # Set all tokens before assistant response to -100
            labels[:len(assistant_enc)] = -100
            
        # Also set -100 for padding tokens
        labels[labels == self.tokenizer.pad_token_id] = -100
        
        return {
            "image": image,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }