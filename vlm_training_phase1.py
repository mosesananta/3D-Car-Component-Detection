import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import get_linear_schedule_with_warmup
from PIL import Image
import pandas as pd
from tqdm.auto import tqdm
import matplotlib.pyplot as plt

from src.model import ComponentClassifier, VisionProjectionLayer
from src.dataset import CarLLaVADataset
# Configuration
CONFIG = {
    "dataset_path": "/kaggle/input/3dcardata/car_state_dataset_multilabel",
    "text_labels_path": "/kaggle/working/text_labels.csv",
    "cnn_model_path": "/kaggle/input/car_component_classifier/pytorch/default/1/car_component_classifier_model_resnet50_max.pt",
    "llm_model_path": "HuggingFaceTB/SmolLM2-135M-Instruct",
    "output_dir": "car_llava_phase1",
    "batch_size": 16,  # Can be larger with the smaller model
    "gradient_accumulation_steps": 4,  # Effective batch size = 64
    "epochs": 10,  # Can train longer with the smaller model
    "learning_rate": 1e-4,  # Higher learning rate for projection layer only
    "warmup_ratio": 0.05,
    "max_length": 256,  # Can be smaller for this task
    "image_size": 224,
    "feature_dim": 2048,     # CNN feature dimension
    "classifier_dim": 5,    # Number of car components
    "hidden_dim": 768,      # Projection hidden dimension
    "llm_embedding_dim": 576,  # SmolLM2 embedding dimension
    "save_every_steps": 200
}

# Ensure output directory exists
os.makedirs(CONFIG["output_dir"], exist_ok=True)





class CarLLaVA_Phase1(nn.Module):
    """
    Phase 1 of Car-LLaVA model: CNN and LLM are frozen, only train projection layer
    """
    
    def __init__(self, 
                 cnn_model_path,              # Path to fine-tuned car component model
                 llm_model_path,              # Path to LLM
                 projection_hidden_dim=512,   # Hidden dim for projection layer
                 vision_feature_dim=512,      # CNN feature dimension
                 vision_classifier_dim=5,     # Number of components
                 llm_embedding_dim=576):      # SmolLM2 embedding dimension
        super().__init__()
        
        # Load CNN classifier model (includes both feature extractor and classifier)
        self.vision_model = ComponentClassifier(embedding_dim=256)
        checkpoint = torch.load(cnn_model_path)
        self.vision_model.load_state_dict(checkpoint['model_state_dict'])
        
        # Freeze CNN
        for param in self.vision_model.parameters():
            param.requires_grad = False
        print("CNN model loaded and frozen")
        
        # Load tokenizer and add special tokens
        self.tokenizer = AutoTokenizer.from_pretrained(llm_model_path)
        special_tokens = {"additional_special_tokens": ["<image>", "</image>"]}
        num_added = self.tokenizer.add_special_tokens(special_tokens)
        print(f"Added {num_added} special tokens to tokenizer")
        
        # Load LLM 
        self.llm = AutoModelForCausalLM.from_pretrained(llm_model_path)
        
        # Resize token embeddings to account for new special tokens
        self.llm.resize_token_embeddings(len(self.tokenizer))
        
        # Freeze LLM
        for param in self.llm.parameters():
            param.requires_grad = False
        print("LLM loaded and frozen")
        
        # Create projection layer (the only trainable part in phase 1)
        self.projection_layer = VisionProjectionLayer(
            vision_feature_dim=vision_feature_dim,
            vision_classifier_dim=vision_classifier_dim,
            llm_embedding_dim=llm_embedding_dim,
            hidden_dim=projection_hidden_dim
        )
        
        # Log trainable parameters
        trainable_params = sum(p.numel() for p in self.projection_layer.parameters() if p.requires_grad)
        print(f"Trainable parameters in projection layer: {trainable_params:,}")
    
    def forward(self, images, input_ids, attention_mask=None):
        # Process images through CNN (with gradients disabled)
        with torch.no_grad():
            self.vision_model.eval()
            logits, features, _ = self.vision_model(images, return_features=True)
            sigmoid_outputs = torch.sigmoid(logits)
        
        # Project visual features to LLM embedding space (this has gradients)
        visual_embedding = self.projection_layer(features, sigmoid_outputs)
        
        # Get LLM input embeddings (detached - we don't want to train the embedding table)
        embedding_layer = self.llm.get_input_embeddings()
        llm_inputs = embedding_layer(input_ids).detach()
        
        # Find positions of image tokens
        batch_size = input_ids.shape[0]
        image_token_id = self.tokenizer.convert_tokens_to_ids("<image>")
        
        # Replace <image> token embeddings with visual embeddings
        for b in range(batch_size):
            image_positions = (input_ids[b] == image_token_id).nonzero(as_tuple=True)[0]
            if len(image_positions) > 0:
                image_pos = image_positions[0]
                # Important: This connects our projection layer to the computation graph
                llm_inputs[b, image_pos] = visual_embedding[b]
        
        # Forward pass through LLM - WITHOUT using torch.no_grad()
        # Parameters are still frozen (requires_grad=False) but gradients can flow through
        outputs = self.llm(
            inputs_embeds=llm_inputs,
            attention_mask=attention_mask,
            return_dict=True
        )
        
        return outputs






def train_phase1():
    """Train Phase 1 of Car-LLaVA: projection layer only"""
    
    print("Starting Phase 1 training: projection layer integration")
    
    # Set up data transforms
    transform = transforms.Compose([
        transforms.Resize((CONFIG["image_size"], CONFIG["image_size"])),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Initialize model
    print("Initializing Car-LLaVA Phase 1 model")
    model = CarLLaVA_Phase1(
        cnn_model_path=CONFIG["cnn_model_path"],
        llm_model_path=CONFIG["llm_model_path"],
        projection_hidden_dim=CONFIG["hidden_dim"],
        vision_feature_dim=CONFIG["feature_dim"],
        vision_classifier_dim=CONFIG["classifier_dim"],
        llm_embedding_dim=CONFIG["llm_embedding_dim"]
    )
    
    # Create dataset
    print(f"Loading dataset from {CONFIG['text_labels_path']}")
    dataset = CarLLaVADataset(
        image_dir=CONFIG["dataset_path"],
        text_labels_path=CONFIG["text_labels_path"],
        tokenizer=model.tokenizer,
        transform=transform,
        max_length=CONFIG["max_length"]
    )
    
    # Split dataset
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    print(f"Dataset split: {train_size} training, {val_size} validation samples")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=CONFIG["batch_size"],
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=CONFIG["batch_size"],
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
    )
    
    # Set up optimizer - only optimize projection layer parameters
    optimizer = optim.AdamW(
        model.projection_layer.parameters(),
        lr=CONFIG["learning_rate"],
        weight_decay=0.01,
        betas=(0.9, 0.999)
    )
    
    # Calculate total training steps
    total_steps = len(train_loader) * CONFIG["epochs"] // CONFIG["gradient_accumulation_steps"]
    warmup_steps = int(total_steps * CONFIG["warmup_ratio"])
    
    # Set up learning rate scheduler
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )
    
    print(f"Training for {CONFIG['epochs']} epochs, {total_steps} total steps, {warmup_steps} warmup steps")
    
    # Move model to device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Handle multi-GPU setup if available
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs for training")
        model = nn.DataParallel(model)

    model = model.to(device)
    
    # Set up loss function
    loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
    
    # Training loop
    train_losses = []
    val_losses = []
    global_step = 0
    best_val_loss = float('inf')
    
    print("Starting training loop")
    for epoch in range(CONFIG["epochs"]):
        # Training
        model.train()
        epoch_loss = 0
        
        # Use tqdm for progress bar
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{CONFIG['epochs']}")
        for step, batch in enumerate(pbar):
   
            # Move batch to device
            batch = {k: v.to(device) for k, v in batch.items()}
            
            # Forward pass
            outputs = model(
                images=batch["image"],
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"]
            )
            
            # Calculate loss
            logits = outputs.logits
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = batch["labels"][..., 1:].contiguous()
            
            # Compute loss
            loss = loss_fn(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1)
            )
            
            # Scale loss for gradient accumulation
            loss = loss / CONFIG["gradient_accumulation_steps"]
                      
            # Backward pass
            loss.backward()
            
            # Update progress bar
            pbar.set_postfix({"loss": loss.item() * CONFIG["gradient_accumulation_steps"]})
            
            # Update weights and reset gradients
            if (step + 1) % CONFIG["gradient_accumulation_steps"] == 0 or step == len(train_loader) - 1:
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                
                global_step += 1
                
                # Save model checkpoint periodically
                if global_step % CONFIG["save_every_steps"] == 0:
                    checkpoint_path = os.path.join(CONFIG["output_dir"], f"checkpoint-step-{global_step}.pt")
                    
                    # Save only projection layer for efficiency, accounting for DataParallel
                    if isinstance(model, nn.DataParallel):
                        torch.save(model.module.projection_layer.state_dict(), checkpoint_path)
                    else:
                        torch.save(model.projection_layer.state_dict(), checkpoint_path)
                    
                    print(f"Saved checkpoint at step {global_step} to {checkpoint_path}")
            
            # Accumulate loss
            epoch_loss += loss.item() * CONFIG["gradient_accumulation_steps"]
        
        # Calculate average training loss
        avg_train_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # Validation
        model.eval()
        val_loss = 0
        
        print("Running validation...")
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                # Move batch to device
                batch = {k: v.to(device) for k, v in batch.items()}
                
                # Forward pass
                outputs = model(
                    images=batch["image"],
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"]
                )
                
                # Calculate loss
                logits = outputs.logits
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = batch["labels"][..., 1:].contiguous()
                
                # Compute loss
                loss = loss_fn(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1)
                )
                
                val_loss += loss.item()
        
        # Calculate average validation loss
        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        
        # Log epoch results
        print(f"Epoch {epoch+1}: Train Loss = {avg_train_loss:.4f}, Val Loss = {avg_val_loss:.4f}")
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_path = os.path.join(CONFIG["output_dir"], "best_projection_layer.pt")
            
            # Save only projection layer for efficiency, accounting for DataParallel
            if isinstance(model, nn.DataParallel):
                torch.save(model.module.projection_layer.state_dict(), best_model_path)
            else:
                torch.save(model.projection_layer.state_dict(), best_model_path)
            
            print(f"New best model saved to {best_model_path} with validation loss {best_val_loss:.4f}")
    
    # Save final model
    final_model_path = os.path.join(CONFIG["output_dir"], "final_projection_layer.pt")

    # Save only projection layer for efficiency, accounting for DataParallel
    if isinstance(model, nn.DataParallel):
        torch.save(model.module.projection_layer.state_dict(), final_model_path)
    else:
        torch.save(model.projection_layer.state_dict(), final_model_path)

    print(f"Final model saved to {final_model_path}")
    
    # Plot training and validation losses
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.savefig(os.path.join(CONFIG["output_dir"], 'loss_curve.png'))
    
    print("Phase 1 training complete!")
    
    return model

if __name__ == "__main__":
    train_phase1()