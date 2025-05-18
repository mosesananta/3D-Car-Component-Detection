import os
import numpy as np
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
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training


from src.model import ComponentClassifier, VisionProjectionLayer
from src.dataset import CarLLaVADataset

# Configuration
CONFIG = {
    "dataset_path": "/kaggle/input/3dcardata/car_state_dataset_multilabel",
    "text_labels_path": "/kaggle/working/text_labels.csv",
    "cnn_model_path": "/kaggle/input/car_component_classifier/pytorch/default/1/car_component_classifier_model_resnet50_max.pt",
    "llm_model_path": "HuggingFaceTB/SmolLM2-135M-Instruct",
    "phase1_model_path": "car_llava_phase1/best_projection_layer.pt",
    "output_dir": "car_llava_phase2",
    "batch_size": 16,
    "gradient_accumulation_steps": 4,
    "epochs": 15,
    "projection_lr": 5e-5,      # Lower learning rate for pre-trained projection layer
    "lora_lr": 2e-5,            # Learning rate for LoRA parameters
    "warmup_ratio": 0.05,
    "max_length": 256,
    "image_size": 224,
    "feature_dim": 2048,        # CNN feature dimension for ResNet50
    "classifier_dim": 5,        # Number of car components
    "hidden_dim": 768,          # Projection hidden dimension
    "llm_embedding_dim": 576,   # SmolLM2 embedding dimension
    "save_every_steps": 200,
    "weight_decay": 0.03,       # Increased weight decay
    # LoRA configuration
    "lora_r": 16,               # LoRA rank
    "lora_alpha": 32,           # LoRA alpha scaling
    "lora_dropout": 0.05        # LoRA dropout
}

# Ensure output directory exists
os.makedirs(CONFIG["output_dir"], exist_ok=True)


class CarLLaVA_Phase2(nn.Module):
    """
    Phase 2 of Car-LLaVA model: 
    - Projection layer initialized from Phase 1
    - LLM partially unfrozen and fine-tuned with LoRA
    """
    
    def __init__(self, 
                 cnn_model_path,              # Path to fine-tuned car component model
                 llm_model_path,              # Path to LLM
                 phase1_model_path,           # Path to trained projection layer from Phase 1
                 projection_hidden_dim=768,   # Hidden dim for projection layer
                 vision_feature_dim=2048,     # CNN feature dimension
                 vision_classifier_dim=5,     # Number of components
                 llm_embedding_dim=576,       # SmolLM2 embedding dimension
                 lora_r=16,                   # LoRA rank
                 lora_alpha=32,               # LoRA alpha
                 lora_dropout=0.05):          # LoRA dropout
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
        
        # Create projection layer
        self.projection_layer = VisionProjectionLayer(
            vision_feature_dim=vision_feature_dim,
            vision_classifier_dim=vision_classifier_dim,
            llm_embedding_dim=llm_embedding_dim,
            hidden_dim=projection_hidden_dim
        )
        
        # Load pre-trained projection layer weights from Phase 1
        try:
            projection_state_dict = torch.load(phase1_model_path)
            # Check if the keys match the current model
            matched_keys = 0
            for key in projection_state_dict.keys():
                if key in self.projection_layer.state_dict():
                    matched_keys += 1
            
            print(f"Found {matched_keys} matching keys in projection layer checkpoint")
            
            # Handle potential size mismatch if the model architecture changed
            if matched_keys / len(projection_state_dict.keys()) >= 0.5:  # At least 50% keys match
                # Create a new state dict with matching keys only
                cleaned_state_dict = {k: v for k, v in projection_state_dict.items() 
                                    if k in self.projection_layer.state_dict() and 
                                    v.shape == self.projection_layer.state_dict()[k].shape}
                
                # Load the matched weights
                self.projection_layer.load_state_dict(cleaned_state_dict, strict=False)
                print(f"Loaded {len(cleaned_state_dict)}/{len(self.projection_layer.state_dict())} layers from Phase 1 checkpoint")
            else:
                print("Warning: Phase 1 checkpoint structure doesn't match. Using random initialization.")
        except Exception as e:
            print(f"Error loading Phase 1 projection layer: {e}")
            print("Using random initialization for projection layer")
        
        # Load LLM 
        self.llm = AutoModelForCausalLM.from_pretrained(llm_model_path)
        
        # Resize token embeddings to account for new special tokens
        self.llm.resize_token_embeddings(len(self.tokenizer))
        
        # Apply LoRA to specific attention layers
        lora_target_modules = [
            "q_proj", 
            "k_proj",
            "v_proj", 
            "o_proj"
        ]
        
        # Configure LoRA
        self.lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=lora_target_modules,
            lora_dropout=lora_dropout,
            bias="none",
            task_type="CAUSAL_LM"
        )
        
        # Prepare model for LoRA fine-tuning
        self.llm = get_peft_model(self.llm, self.lora_config)
        self.llm.print_trainable_parameters()  # Print percentage of trainable parameters
    
    def forward(self, images, input_ids, attention_mask=None):
        # Process images through CNN (with gradients disabled)
        with torch.no_grad():
            self.vision_model.eval()
            logits, features, _ = self.vision_model(images, return_features=True)
            sigmoid_outputs = torch.sigmoid(logits)  # Get probabilities for component states
        
        # Project visual features to LLM embedding space
        visual_embedding = self.projection_layer(features, sigmoid_outputs)
        
        # Find positions of image tokens in the input
        batch_size = input_ids.shape[0]
        image_token_id = self.tokenizer.convert_tokens_to_ids("<image>")
        
        # Get LLM input embeddings
        llm_inputs = self.llm.get_input_embeddings()(input_ids)
        
        # Replace <image> token embeddings with visual embeddings
        for b in range(batch_size):
            # Find the position of <image> token
            image_positions = (input_ids[b] == image_token_id).nonzero(as_tuple=True)[0]
            if len(image_positions) > 0:
                # Replace the embedding at the <image> position
                image_pos = image_positions[0]
                llm_inputs[b, image_pos] = visual_embedding[b]
        
        # Forward pass through LLM
        outputs = self.llm(
            inputs_embeds=llm_inputs,
            attention_mask=attention_mask,
            return_dict=True
        )
        
        return outputs



def train_phase2():
    """Train Phase 2 of Car-LLaVA: LLM adaptation via LoRA"""
    
    print("Starting Phase 2 training: LLM adaptation")
    
    # Set up data transforms
    transform = transforms.Compose([
        transforms.Resize((CONFIG["image_size"], CONFIG["image_size"])),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Initialize model
    print("Initializing Car-LLaVA Phase 2 model")
    model = CarLLaVA_Phase2(
        cnn_model_path=CONFIG["cnn_model_path"],
        llm_model_path=CONFIG["llm_model_path"],
        phase1_model_path=CONFIG["phase1_model_path"],
        projection_hidden_dim=CONFIG["hidden_dim"],
        vision_feature_dim=CONFIG["feature_dim"],
        vision_classifier_dim=CONFIG["classifier_dim"],
        llm_embedding_dim=CONFIG["llm_embedding_dim"],
        lora_r=CONFIG["lora_r"],
        lora_alpha=CONFIG["lora_alpha"],
        lora_dropout=CONFIG["lora_dropout"]
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
    
    # Set up optimizer with different learning rates for different components
    optimizer_grouped_parameters = [
        {
            "params": model.projection_layer.parameters(),
            "lr": CONFIG["projection_lr"],
            "weight_decay": CONFIG["weight_decay"]
        },
        {
            "params": model.llm.parameters(),
            "lr": CONFIG["lora_lr"],
            "weight_decay": CONFIG["weight_decay"] * 0.5  # Lower weight decay for LoRA params
        }
    ]
    
    optimizer = optim.AdamW(
        optimizer_grouped_parameters,
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
                    checkpoint_path = os.path.join(CONFIG["output_dir"], f"checkpoint-step-{global_step}")
                    os.makedirs(checkpoint_path, exist_ok=True)
                    
                    # Save projection layer
                    if isinstance(model, nn.DataParallel):
                        torch.save(model.module.projection_layer.state_dict(), 
                                os.path.join(checkpoint_path, "projection_layer.pt"))
                        model.module.llm.save_pretrained(checkpoint_path)
                        model.module.tokenizer.save_pretrained(checkpoint_path)
                    else:
                        torch.save(model.projection_layer.state_dict(), 
                                os.path.join(checkpoint_path, "projection_layer.pt"))
                        model.llm.save_pretrained(checkpoint_path)
                        model.tokenizer.save_pretrained(checkpoint_path)
                    
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
        
        # Sample generation to check progress
        if device == torch.device("cuda"):
            torch.cuda.empty_cache()  # Free up memory
        
        # Generate a sample to show progress
        print("\nGenerating sample response...")
        try:
            model.eval()
            
            # Get a random sample from validation set
            random_idx = torch.randint(0, len(val_dataset), (1,)).item()
            val_sample = val_dataset[random_idx]
            sample_image = val_sample["image"].unsqueeze(0).to(device)
            
            tokenizer = model.module.tokenizer if isinstance(model, nn.DataParallel) else model.tokenizer
            
            # Get the ground truth for reference
            # Extract the actual ground truth label from the sample
            ground_truth = ""
            for i in range(len(val_sample["labels"])):
                if val_sample["labels"][i] != -100:
                    ground_truth += tokenizer.decode([val_sample["input_ids"][i]])
            
            print(f"Ground truth: {ground_truth}")
            
            # Create a new input for generation
            messages = [
                {"role": "user", "content": "Examine this car image and describe which doors and hood are open or closed.\n<image></image>"}
            ]
            
            # Access tokenizer through module if using DataParallel

            
            input_text = tokenizer.apply_chat_template(messages, tokenize=False)
            input_ids = tokenizer.encode(input_text, return_tensors="pt").to(device)
            
            # Generate response
            with torch.no_grad():
                # Get visual embedding - handle DataParallel wrapping
                with torch.no_grad():
                    if isinstance(model, nn.DataParallel):
                        model.module.vision_model.eval()
                        logits, features, _ = model.module.vision_model(sample_image, return_features=True)
                    else:
                        model.vision_model.eval()
                        logits, features, _ = model.vision_model(sample_image, return_features=True)
                        
                    sigmoid_outputs = torch.sigmoid(logits)
                    
                    # Print predicted component states for reference
                    predicted_states = (sigmoid_outputs > 0.8).cpu().numpy()[0]
                    component_names = ["Front Left Door", "Front Right Door", "Rear Left Door", "Rear Right Door", "Hood"]
                    print("CNN Predicted States:")
                    for i, name in enumerate(component_names):
                        state = "Open" if predicted_states[i] else "Closed"
                        print(f"  {name}: {state} ({sigmoid_outputs[0][i].item():.2f})")
                
                # Project visual features - handle DataParallel wrapping
                if isinstance(model, nn.DataParallel):
                    visual_embedding = model.module.projection_layer(features, sigmoid_outputs)
                    image_token_id = model.module.tokenizer.convert_tokens_to_ids("<image>")
                else:
                    visual_embedding = model.projection_layer(features, sigmoid_outputs)
                    image_token_id = model.tokenizer.convert_tokens_to_ids("<image>")
                
                # Get LLM embedding handlers based on whether using DataParallel
                if isinstance(model, nn.DataParallel):
                    llm = model.module.llm
                else:
                    llm = model.llm
                    
                # Replace <image> token embedding
                image_pos = (input_ids[0] == image_token_id).nonzero(as_tuple=True)[0]
                
                if len(image_pos) > 0:
                    image_pos = image_pos[0]
                    llm_inputs = llm.get_input_embeddings()(input_ids)
                    llm_inputs[0, image_pos] = visual_embedding[0]
                    
                    # Generate
                    outputs = llm.generate(
                        inputs_embeds=llm_inputs,
                        max_new_tokens=50,
                        temperature=0,
                        do_sample=False
                    )
                    
                    # Decode and print
                    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
                    print(f"Sample generation:\n{generated_text}")
                    
                    # Plot and save the image with generated text
                    plt.figure(figsize=(10, 10))
                    
                    # Extract image tensor and convert to numpy for plotting
                    img_tensor = val_sample["image"].cpu()
                    # Denormalize the image
                    inv_normalize = transforms.Normalize(
                        mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
                        std=[1/0.229, 1/0.224, 1/0.225]
                    )
                    img_tensor = inv_normalize(img_tensor)
                    # Convert to numpy and transpose dimensions for matplotlib
                    img_np = img_tensor.permute(1, 2, 0).numpy()
                    # Clip values to be within [0,1] range
                    img_np = np.clip(img_np, 0, 1)
                    
                    # Plot the image
                    plt.subplot(1, 1, 1)
                    plt.imshow(img_np)
                    plt.axis('off')
                    
                    # Create a nicely formatted caption
                    caption = f"Epoch {epoch+1} - Random Sample\n\n"
                    caption += "Generated:\n" + generated_text + "\n\n"
                    caption += "CNN Predicted States:\n"
                    for i, name in enumerate(component_names):
                        state = "Open" if predicted_states[i] else "Closed"
                        conf = sigmoid_outputs[0][i].item()
                        caption += f"{name}: {state} ({conf:.2f})\n"
                    
                    plt.title(caption, fontsize=10, loc='left')
                    
                    plt.show()
                    
                    # Save the figure
                    sample_img_path = os.path.join(CONFIG["output_dir"], f"sample_epoch_{epoch+1}.png")
                    plt.savefig(sample_img_path, bbox_inches='tight')
                    plt.close()
                    print(f"Sample image and text saved to {sample_img_path}")

        except Exception as e:
            print(f"Error during sample generation: {e}")
            import traceback
            traceback.print_exc()  # Print full traceback for debugging
                
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_path = os.path.join(CONFIG["output_dir"], "best_model")
            os.makedirs(best_model_path, exist_ok=True)
            
            # Save projection layer and LLM - handle DataParallel wrapping
            if isinstance(model, nn.DataParallel):
                torch.save(model.module.projection_layer.state_dict(), 
                        os.path.join(best_model_path, "projection_layer.pt"))
                model.module.llm.save_pretrained(best_model_path)
                model.module.tokenizer.save_pretrained(best_model_path)
            else:
                torch.save(model.projection_layer.state_dict(), 
                        os.path.join(best_model_path, "projection_layer.pt"))
                model.llm.save_pretrained(best_model_path)
                model.tokenizer.save_pretrained(best_model_path)
            
            print(f"New best model saved to {best_model_path} with validation loss {best_val_loss:.4f}")
    
    # Save final model
    final_model_path = os.path.join(CONFIG["output_dir"], "final_model")
    os.makedirs(final_model_path, exist_ok=True)

    # Save projection layer and LLM - handle DataParallel wrapping
    if isinstance(model, nn.DataParallel):
        torch.save(model.module.projection_layer.state_dict(), 
                os.path.join(final_model_path, "projection_layer.pt"))
        model.module.llm.save_pretrained(final_model_path)
        model.module.tokenizer.save_pretrained(final_model_path)
    else:
        torch.save(model.projection_layer.state_dict(), 
                os.path.join(final_model_path, "projection_layer.pt"))
        model.llm.save_pretrained(final_model_path)
        model.tokenizer.save_pretrained(final_model_path)

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
    
    print("Phase 2 training complete!")
    
    return model

if __name__ == "__main__":
    train_phase2()