import os
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
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split

# Import view-invariant model definition
from src.dataset import CarComponentDataset
from src.model import ViewInvariantModel, ComponentClassifier
from src.loss import info_nce_loss, FocalLoss


# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Configuration
CONFIG = {
    "dataset_path": "/kaggle/input/3dcardata/car_state_dataset_preprocessed",
    "batch_size": 64,
    "num_epochs": 100,
    "feature_extractor_lr": 0.00005,  # 10x lower for pre-trained features
    "classifier_heads_lr": 0.0005,     # Higher for new classifier heads
    "weight_decay": 1e-5,
    "image_size": 224,
    "use_wandb": False,
    "early_stopping_patience": 10,
    "pretrained_model_path": "car_component_view_invariant_model.pt",
    "final_model_save_path": "car_component_classifier_model.pt",
    "freeze_feature_extractor": False,  # Whether to freeze the pre-trained feature extractor
    "use_contrastive_loss": True,  # Whether to include contrastive loss during fine-tuning
    "contrastive_loss_weight": 0.2,  # Weight for contrastive loss component
    "embedding_dim": 256,  # Must match pre-trained model
    "use_focal_loss": True,
    "focal_loss_alpha": 0.25,  # Weight for the positive class in focal loss
    "focal_loss_gamma": 2.0   # Focusing parameter for focal loss
}



def evaluate_components(model, data_loader, device, threshold=0.8):
    """
    Evaluate model performance on each component.
    
    Args:
        model: The trained model
        data_loader: Data loader for evaluation
        device: Computation device (CPU/GPU)
        threshold: Threshold for binary classification
    """
    model.eval()
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in tqdm(data_loader, desc="Evaluating components"):
            images = images.to(device)
            outputs = model(images)
            
            # Apply sigmoid and threshold
            preds = (torch.sigmoid(outputs) >= threshold).float()
            
            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
    
    # Convert to numpy arrays
    all_preds = np.vstack(all_preds)
    all_labels = np.vstack(all_labels)
    
    # Component names for reporting
    component_names = ['Front Left Door', 'Front Right Door', 'Rear Left Door', 'Rear Right Door', 'Hood']
    
    # Calculate metrics for each component
    metrics = {}
    for i, name in enumerate(component_names):
        # Extract predictions and labels for this component
        comp_preds = all_preds[:, i]
        comp_labels = all_labels[:, i]
        
        # Calculate metrics
        accuracy = accuracy_score(comp_labels, comp_preds)
        precision = precision_score(comp_labels, comp_preds, zero_division=0)
        recall = recall_score(comp_labels, comp_preds, zero_division=0)
        f1 = f1_score(comp_labels, comp_preds, zero_division=0)
        conf_matrix = confusion_matrix(comp_labels, comp_preds)
        
        # Store metrics
        metrics[name] = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'confusion_matrix': conf_matrix
        }
    
    # Overall metrics
    metrics['overall'] = {
        'accuracy': accuracy_score(all_labels.flatten(), all_preds.flatten()),
        'precision': precision_score(all_labels.flatten(), all_preds.flatten(), zero_division=0),
        'recall': recall_score(all_labels.flatten(), all_preds.flatten(), zero_division=0),
        'f1': f1_score(all_labels.flatten(), all_preds.flatten(), zero_division=0)
    }
    
    return metrics

def create_optimizer(model, freeze_feature_extractor=False):
    """Create optimizer with different learning rates for different parts of the model."""
    
    if freeze_feature_extractor:
        # If feature extractor is frozen, only optimize classifier parameters
        return optim.AdamW(
            model.component_heads.parameters(),
            lr=CONFIG["classifier_heads_lr"],
            weight_decay=CONFIG["weight_decay"]
        )
    else:
        # Use different learning rates for feature extractor and classifier heads
        feature_params = model.base_model.parameters()
        classifier_params = model.component_heads.parameters()
        
        param_groups = [
            {'params': feature_params, 'lr': CONFIG["feature_extractor_lr"]},
            {'params': classifier_params, 'lr': CONFIG["classifier_heads_lr"]}
        ]
        
        return optim.AdamW(
            param_groups,
            weight_decay=CONFIG["weight_decay"]
        )

def train_component_classifier():
    """Fine-tune the view-invariant model for component classification."""
    print("Starting component classifier fine-tuning with Focal Loss...")
    
    # Initialize wandb if enabled
    if CONFIG["use_wandb"]:
        wandb.init(project="car-component-detection", name="component-classifier-focal", config=CONFIG)
    
    # 1. Prepare data
    # ---------------
    # Load labels file
    labels_path = os.path.join(CONFIG["dataset_path"], "labels.csv")
    all_labels = pd.read_csv(labels_path)
    
    # Split into train and validation sets
    train_df, val_df = train_test_split(all_labels, test_size=0.2, random_state=42)
    
    print(f"Training set: {len(train_df)} images")
    print(f"Validation set: {len(val_df)} images")
    
    # Define data transforms
    transform = transforms.Compose([
        transforms.Resize((CONFIG["image_size"], CONFIG["image_size"])),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Create datasets
    train_dataset = CarComponentDataset(CONFIG["dataset_path"], train_df, transform=transform)
    val_dataset = CarComponentDataset(CONFIG["dataset_path"], val_df, transform=transform)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=CONFIG["batch_size"],
        shuffle=True,
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
    
    # 2. Create model, loss function, and optimizer
    # --------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create classifier model with pre-trained feature extractor
    model = ComponentClassifier(embedding_dim=CONFIG["embedding_dim"])
    model.load_unfreeze_feature_extractor(
        pretrained_extractor_path=CONFIG["pretrained_model_path"],
        freeze_feature_extractor=CONFIG["freeze_feature_extractor"]
    )
    model = model.to(device)
    
    # Define Focal Loss for classification instead of BCE  
    if CONFIG["use_focal_loss"]:
        classification_criterion = FocalLoss(
            alpha=CONFIG["focal_loss_alpha"],
            gamma=CONFIG["focal_loss_gamma"]
        )
    else: 
        classification_criterion = nn.BCEWithLogitsLoss()
    
    # Define optimizer
    optimizer = create_optimizer(model, CONFIG["freeze_feature_extractor"])
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='max',
        factor=0.5,
        patience=5,
        verbose=True
    )
    
    # 3. Training loop
    # ---------------
    best_val_f1 = 0.0
    patience_counter = 0
    
    # Lists to store metrics for plotting
    train_losses = []
    val_f1_scores = []
    
    # For each epoch
    for epoch in range(CONFIG["num_epochs"]):
        print(f"\nEpoch {epoch+1}/{CONFIG['num_epochs']}")
        
        # Training phase
        model.train()
        train_loss = 0.0
        
        # Track progress with tqdm
        train_pbar = tqdm(train_loader, desc=f"Training", leave=True)
        
        for images, labels in train_pbar:
            # Move data to device
            images = images.to(device)
            labels = labels.to(device)
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward pass
            if CONFIG["use_contrastive_loss"]:
                # Get classification outputs and projections for contrastive loss
                outputs, _, projections = model(images, return_features=True)
                
                # Focal loss for classification
                cls_loss = classification_criterion(outputs, labels)
                
                # Contrastive loss (if a batch has positive pairs)
                cont_loss = info_nce_loss(projections, labels)
                
                # Combined loss
                loss = cls_loss + CONFIG["contrastive_loss_weight"] * cont_loss
            else:
                # Simple classification only with Focal Loss
                outputs = model(images)
                loss = classification_criterion(outputs, labels)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            # Update statistics
            train_loss += loss.item()
            
            # Update progress bar
            train_pbar.set_postfix({"loss": loss.item()})
        
        # Calculate average training loss
        avg_train_loss = train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        print(f"Train Loss: {avg_train_loss:.4f}")
        
        # Validation phase
        model.eval()
        
        # Evaluate on validation set
        val_metrics = evaluate_components(model, val_loader, device)
        
        # Print validation metrics
        print("\nValidation Metrics:")
        print(f"Overall - Accuracy: {val_metrics['overall']['accuracy']:.4f}, F1: {val_metrics['overall']['f1']:.4f}")
        
        for name in ['Front Left Door', 'Front Right Door', 'Rear Left Door', 'Rear Right Door', 'Hood']:
            print(f"{name} - F1: {val_metrics[name]['f1']:.4f}")
        
        # Track validation F1 score
        val_f1 = val_metrics['overall']['f1']
        val_f1_scores.append(val_f1)
        
        # Update learning rate scheduler
        scheduler.step(val_f1)
        
        # Log to wandb if enabled
        if CONFIG["use_wandb"]:
            log_dict = {
                "epoch": epoch + 1,
                "train_loss": avg_train_loss,
                "val_f1": val_f1,
                "val_accuracy": val_metrics['overall']['accuracy'],
                "learning_rate": optimizer.param_groups[0]["lr"]
            }
            # Add component-specific metrics
            for name in ['Front Left Door', 'Front Right Door', 'Rear Left Door', 'Rear Right Door', 'Hood']:
                for metric in ['accuracy', 'precision', 'recall', 'f1']:
                    log_dict[f"{name}_{metric}"] = val_metrics[name][metric]
            
            wandb.log(log_dict)
        
        # Early stopping check
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            patience_counter = 0
            
            # Save the best model
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'f1': val_f1,
                'loss': avg_train_loss
            }, CONFIG["final_model_save_path"])
            
            print(f"Model saved to {CONFIG['final_model_save_path']} (F1: {val_f1:.4f})")
        else:
            patience_counter += 1
            print(f"Early stopping patience: {patience_counter}/{CONFIG['early_stopping_patience']}")
            
            if patience_counter >= CONFIG['early_stopping_patience']:
                print("Early stopping triggered! No improvement in validation F1.")
                break
    
    # 4. Plot training curves
    # ----------------------
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    
    plt.subplot(1, 2, 2)
    plt.plot(val_f1_scores)
    plt.xlabel('Epoch')
    plt.ylabel('F1 Score')
    plt.title('Validation F1 Score')
    
    plt.tight_layout()
    plt.savefig('component_classifier_focal_training.png')
    
    # 5. Final evaluation
    # ------------------
    # Load the best model
    checkpoint = torch.load(CONFIG["final_model_save_path"])
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Evaluate on validation set
    final_metrics = evaluate_components(model, val_loader, device)
    
    # Print final results
    print("\nFinal Evaluation Results:")
    print(f"Overall Accuracy: {final_metrics['overall']['accuracy']:.4f}")
    print(f"Overall F1 Score: {final_metrics['overall']['f1']:.4f}")
    
    # Component-specific results with confusion matrices
    component_names = ['Front Left Door', 'Front Right Door', 'Rear Left Door', 'Rear Right Door', 'Hood']
    for name in component_names:
        print(f"\n{name}:")
        print(f"  Accuracy: {final_metrics[name]['accuracy']:.4f}")
        print(f"  Precision: {final_metrics[name]['precision']:.4f}")
        print(f"  Recall: {final_metrics[name]['recall']:.4f}")
        print(f"  F1 Score: {final_metrics[name]['f1']:.4f}")
        print(f"  Confusion Matrix:")
        print(final_metrics[name]['confusion_matrix'])
    
    print("\nComponent classifier fine-tuning with Focal Loss complete!")
    
    # Close wandb if enabled
    if CONFIG["use_wandb"]:
        wandb.finish()
    
    return model

if __name__ == "__main__":
    train_component_classifier()