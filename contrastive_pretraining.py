import os
import random
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import wandb
from sklearn.model_selection import train_test_split

from src.dataset import CarComponentDataset
from src.model import ViewInvariantModel
from src.loss import info_nce_loss

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# Configuration
CONFIG = {
    "dataset_path": "/kaggle/input/3dcardata/car_state_dataset_preprocessed",
    "batch_size": 128,
    "num_epochs": 500,
    "learning_rate": 0.001,
    "weight_decay": 1e-5,
    "image_size": 224,
    "embedding_dim": 256,
    "temperature": 0.07,
    "use_wandb": False,
    "model_save_path": "car_component_view_invariant_model.pt"
}


def evaluate_view_invariance(model, val_loader, device):
    """
    Evaluate how well the model has learned view-invariant features.
    
    Args:
        model: The trained model
        val_loader: Validation data loader
        device: Computation device (CPU/GPU)
        
    Returns:
        Separation score between intra-state and inter-state feature similarities
    """
    model.eval()
    
    # Store features by component state
    state_to_features = {}
    
    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc="Evaluating view invariance"):
            images = images.to(device)
            features, _ = model(images)
            
            # Convert labels to string representation
            for i, label in enumerate(labels):
                state_str = ''.join(map(str, label.int().tolist()))
                if state_str not in state_to_features:
                    state_to_features[state_str] = []
                state_to_features[state_str].append(features[i].cpu())
    
    # Measure intra-state feature consistency vs. inter-state separation
    intra_similarities = []
    inter_similarities = []
    
    # Process each state combination
    for state1, feat_list1 in state_to_features.items():
        # Skip if too few samples
        if len(feat_list1) < 2:
            continue
            
        # Calculate intra-state similarity (same component states, different views)
        for i in range(len(feat_list1)):
            for j in range(i+1, len(feat_list1)):
                f1 = F.normalize(feat_list1[i], dim=0)
                f2 = F.normalize(feat_list1[j], dim=0)
                sim = torch.dot(f1, f2).item()
                intra_similarities.append(sim)
        
        # Calculate inter-state similarity (different component states)
        for state2, feat_list2 in state_to_features.items():
            if state1 == state2:
                continue
                
            for f1 in feat_list1[:min(len(feat_list1), 5)]:  # Limit comparisons for efficiency
                for f2 in feat_list2[:min(len(feat_list2), 5)]:
                    f1 = F.normalize(f1, dim=0)
                    f2 = F.normalize(f2, dim=0)
                    sim = torch.dot(f1, f2).item()
                    inter_similarities.append(sim)
    
    # Average similarities
    avg_intra_sim = sum(intra_similarities) / len(intra_similarities) if intra_similarities else 0
    avg_inter_sim = sum(inter_similarities) / len(inter_similarities) if inter_similarities else 0
    
    # The larger the difference, the better the view invariance
    separation = avg_intra_sim - avg_inter_sim
    
    print(f"View Invariance Evaluation:")
    print(f"  Avg. similarity between same states: {avg_intra_sim:.4f}")
    print(f"  Avg. similarity between different states: {avg_inter_sim:.4f}")
    print(f"  Separation (higher is better): {separation:.4f}")
    
    return separation


def create_similar_state_batch_sampler(dataset, batch_size):
    """
    Creates a generator function that yields batches with similar states
    grouped together for better contrastive learning.
    
    Args:
        dataset: The CarComponentDataset with state_to_indices mapping
        batch_size: Desired batch size
    
    Returns:
        Generator function that yields batches of indices
    """
    
    def batch_generator():
        # Get all unique states
        all_states = list(dataset.state_to_indices.keys())
        all_indices = list(range(len(dataset)))
        
        # Create random batches that try to include multiple samples of the same states
        random.shuffle(all_indices)
        
        while all_indices:
            # Start a new batch
            batch = []
            
            # Randomly select a few states to focus on in this batch
            selected_states = random.sample(all_states, min(4, len(all_states)))
            
            # Try to add samples from these states
            for state in selected_states:
                # Get indices for this state that are still available
                available_indices = [idx for idx in dataset.state_to_indices[state] if idx in all_indices]
                
                # Sample up to batch_size // 4 indices from this state
                if available_indices:
                    samples_to_take = min(len(available_indices), batch_size // 4)
                    state_samples = random.sample(available_indices, samples_to_take)
                    
                    # Add to batch and remove from available indices
                    batch.extend(state_samples)
                    for idx in state_samples:
                        all_indices.remove(idx)
            
            # Fill remaining batch slots with random indices
            remaining = batch_size - len(batch)
            if remaining > 0 and all_indices:
                random_samples = all_indices[:remaining]
                batch.extend(random_samples)
                all_indices = all_indices[remaining:]
            
            # Yield the batch if it's not empty
            if batch:
                yield batch
            
            # If we don't have enough indices left for a full batch, just use what's left
            if len(all_indices) < batch_size // 2:
                if all_indices:
                    yield all_indices
                break
    
    return batch_generator


def train_view_invariant_model():
    """Train a view-invariant feature extractor using contrastive learning."""
    print("Starting view-invariant feature learning...")
    
    # Initialize wandb if enabled
    if CONFIG["use_wandb"]:
        wandb.init(project="car-component-detection", name="view-invariant-features", config=CONFIG)
    
    # 1. Prepare data
    # ---------------
    # Load labels file
    labels_path = os.path.join(CONFIG["dataset_path"], "labels.csv")
    all_labels = pd.read_csv(labels_path)
    
    # Split into train and validation sets
    train_df, val_df = train_test_split(all_labels, test_size=0.2, random_state=42)
    
    print(f"Training set: {len(train_df)} images")
    print(f"Validation set: {len(val_df)} images")
    
    # Define data transforms (minimal since we're using expanded dataset)
    transform = transforms.Compose([
        transforms.Resize((CONFIG["image_size"], CONFIG["image_size"])),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Create datasets
    train_dataset = CarComponentDataset(CONFIG["dataset_path"], train_df, transform=transform)
    val_dataset = CarComponentDataset(CONFIG["dataset_path"], val_df, transform=transform)
    
    # Create data loaders
    # For training, use the batch generator that groups similar states
    train_loader = DataLoader(
        train_dataset,
        batch_size=CONFIG["batch_size"],
        shuffle=True,  # We'll use standard shuffling for simplicity
        num_workers=4,
        persistent_workers=True,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=CONFIG["batch_size"],
        shuffle=False,
        num_workers=4,
        persistent_workers=True,
        pin_memory=True
    )
    
    # 2. Create model and optimizer
    # -----------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model = ViewInvariantModel(embedding_dim=CONFIG["embedding_dim"])
    model = model.to(device)
    
    # Define optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=CONFIG["learning_rate"],
        weight_decay=CONFIG["weight_decay"]
    )
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=5,
        verbose=True
    )
    
    # 3. Training loop
    # ---------------
    best_separation = -float('inf')
    best_loss = float('inf')
    patience_counter = 0
    early_stopping_patience = 15
    
    # Lists to store metrics for plotting
    train_losses = []
    separations = []
    
    # For each epoch
    for epoch in range(CONFIG["num_epochs"]):
        print(f"\nEpoch {epoch+1}/{CONFIG['num_epochs']}")
        
        # Training phase
        model.train()
        train_loss = 0.0
        batch_count = 0
        
        # Track progress with tqdm
        train_pbar = tqdm(train_loader, desc=f"Training", leave=True)
        
        for images, labels in train_pbar:
            # Move data to device
            images = images.to(device)
            labels = labels.to(device)
            
            # Skip batches with only one sample per state (can't do contrastive learning)
            label_strs = [''.join(map(str, l.int().tolist())) for l in labels]
            state_counts = {}
            for s in label_strs:
                state_counts[s] = state_counts.get(s, 0) + 1
                
            has_contrastive_pairs = any(count > 1 for count in state_counts.values())
            
            if not has_contrastive_pairs:
                continue
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward pass
            _, projections = model(images)
            
            # Compute contrastive loss
            loss = info_nce_loss(projections, labels, CONFIG["temperature"])
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            # Update statistics
            train_loss += loss.item()
            batch_count += 1
            
            # Update progress bar
            train_pbar.set_postfix({"loss": loss.item()})
        
        # Calculate average training loss
        avg_train_loss = train_loss / batch_count if batch_count > 0 else 0
        train_losses.append(avg_train_loss)
        
        print(f"Train Loss: {avg_train_loss:.4f}")
        
        # Evaluate view invariance
        if epoch % 5 == 0 or epoch == CONFIG["num_epochs"] - 1:
            separation = evaluate_view_invariance(model, val_loader, device)
            separations.append(separation)
            
            # Update scheduler based on separation metric
            scheduler.step(-separation)  # Negative because higher separation is better
            
            # Log to wandb if enabled
            if CONFIG["use_wandb"]:
                wandb.log({
                    "epoch": epoch + 1,
                    "train_loss": avg_train_loss,
                    "view_invariance_separation": separation,
                    "learning_rate": optimizer.param_groups[0]["lr"]
                })
            
            # Save best model based on separation metric
            if separation > best_separation:
                best_separation = separation
                patience_counter = 0
                
                # Save the best model
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'separation': separation,
                    'loss': avg_train_loss
                }, CONFIG["model_save_path"])
                
                print(f"Model saved to {CONFIG['model_save_path']} (separation: {separation:.4f})")
            else:
                patience_counter += 1
                print(f"Early stopping patience: {patience_counter}/{early_stopping_patience}")
                
                if patience_counter >= early_stopping_patience:
                    print("Early stopping triggered! No improvement in view invariance.")
                    break
        else:
            # Save based on loss when not evaluating separation
            if avg_train_loss < best_loss:
                best_loss = avg_train_loss
                
                # Save intermediate model
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': avg_train_loss
                }, "intermediate_" + CONFIG["model_save_path"])
    
    # 4. Plot training curves
    # ----------------------
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses)
    plt.xlabel('Epoch')
    plt.ylabel('Contrastive Loss')
    plt.title('Training Loss')
    
    if separations:
        plt.subplot(1, 2, 2)
        plt.plot(range(0, len(separations) * 5, 5), separations)
        plt.xlabel('Epoch')
        plt.ylabel('Feature Separation')
        plt.title('View Invariance Quality')
    
    plt.tight_layout()
    plt.savefig('view_invariant_training.png')
    
    print("\nView-invariant feature learning complete!")
    print(f"Best separation score: {best_separation:.4f}")
    
    # Close wandb if enabled
    if CONFIG["use_wandb"]:
        wandb.finish()
    
    return model

if __name__ == "__main__":
    train_view_invariant_model()