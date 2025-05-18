import os
import numpy as np

import cv2
from PIL import Image

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM
import torchvision.transforms as transforms
from peft import PeftModel, PeftConfig

class ResidualBlock(nn.Module):
    """Basic residual block for ResNet architecture."""
    
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        
        # First convolution layer
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        # Second convolution layer
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # Skip connection (shortcut)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        
        # Activation function
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        residual = x
        
        # First conv block
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        # Second conv block
        out = self.conv2(out)
        out = self.bn2(out)
        
        # Add skip connection
        out += self.shortcut(residual)
        out = self.relu(out)
        
        return out
    
class BottleneckBlock(nn.Module):
    """Bottleneck block used in deeper ResNet architectures (50, 101, 152)."""
    
    expansion = 4  # Output channels expand by 4x
    
    def __init__(self, in_channels, out_channels, stride=1):
        super(BottleneckBlock, self).__init__()
        
        # Bottleneck design: reduce dimensions, process, then expand
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)
        
        self.relu = nn.ReLU(inplace=True)
        
        # Skip connection (shortcut)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels * self.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * self.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * self.expansion)
            )
    
    def forward(self, x):
        residual = x
        
        # First 1x1 conv
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        # 3x3 conv
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        
        # Second 1x1 conv for expansion
        out = self.conv3(out)
        out = self.bn3(out)
        
        # Add skip connection
        out += self.shortcut(residual)
        out = self.relu(out)
        
        return out


class ResNet(nn.Module):
    """Improved ResNet architecture with support for bottleneck blocks."""
    
    def __init__(self, block, num_blocks):
        super(ResNet, self).__init__()
        
        self.in_channels = 64
        
        # Initial convolution layer
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # Residual layers
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        
        # Global average pooling
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Initialize weights
        self._initialize_weights()
    
    def _make_layer(self, block, out_channels, num_blocks, stride):
        """Create a layer with multiple residual blocks."""
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            if hasattr(block, 'expansion'):
                self.in_channels = out_channels * block.expansion
            else:
                self.in_channels = out_channels
            
        return nn.Sequential(*layers)
    
    def _initialize_weights(self):
        """Initialize model weights for better convergence."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # Initial convolution
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        # Residual blocks
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        # Global average pooling
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        
        return x


    
class ViewInvariantModel(nn.Module):
    """Model with feature extractor and projection head for contrastive learning."""
    
    def __init__(self, embedding_dim=256, resnet_version=50):
        super(ViewInvariantModel, self).__init__()
        
        # Feature extractor (ResNet-50)
        if resnet_version == 18:
            self.feature_extractor = ResNet(ResidualBlock, [2, 2, 2, 2])  # ResNet-18
        elif resnet_version == 34:
            self.feature_extractor = ResNet(ResidualBlock, [3, 4, 6, 3])  # ResNet-34
        elif resnet_version == 26:
            self.feature_extractor = ResNet(BottleneckBlock, [2, 2, 2, 2])  # ResNet-26
        else:
            self.feature_extractor = ResNet(BottleneckBlock, [3, 4, 6, 3])  # ResNet-50
        
        # Projection head for contrastive learning
        if resnet_version == 18 or resnet_version == 34:
            # Smaller Projection      
            self.projection_head = nn.Sequential(
                nn.Linear(512, 512),
                nn.ReLU(inplace=True),
                nn.Linear(512, embedding_dim)
            )
        else:
            self.projection_head = nn.Sequential(
                nn.Linear(2048, 1024),  # 2048 = 512 * 4 (expansion factor)
                nn.ReLU(inplace=True),
                nn.Linear(1024, embedding_dim)
            )
        

    
    def forward(self, x):
        # Extract features
        features = self.feature_extractor(x)
        
        # Project features for contrastive learning
        projections = self.projection_head(features)
        
        return features, projections
    
class ComponentClassifier(nn.Module):
    """Multi-head classifier built on top of view-invariant feature extractor."""
    
    def __init__(self, embedding_dim, num_components=5):
        super(ComponentClassifier, self).__init__()
        
        # Load pre-trained view-invariant model
        self.base_model = ViewInvariantModel(embedding_dim=embedding_dim)
        
        # Get feature dimension from base model
        # self.feature_dim = 512  # This should match the feature extractor output
        self.feature_dim = 2048  
        
        # Component classification heads (one for each component)
        self.component_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.feature_dim, 128),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(128, 1)
            ) for _ in range(num_components)
        ])
        
    def load_unfreeze_feature_extractor(self, pretrained_extractor_path, freeze_feature_extractor=False):
        # Load pre-trained weights
        checkpoint = torch.load(pretrained_extractor_path)
        self.base_model.load_state_dict(checkpoint['model_state_dict'])
        
        # Freeze feature extractor if specified
        if freeze_feature_extractor:
            for param in self.base_model.feature_extractor.parameters():
                param.requires_grad = False
        
    def forward(self, x, return_features=False):
        # Get features and projections from base model
        features, projections = self.base_model(x)
        
        # Apply component-specific classifiers
        outputs = []
        for head in self.component_heads:
            outputs.append(head(features))
        
        # Stack outputs along dimension 1
        logits = torch.cat(outputs, dim=1)
        
        if return_features:
            return logits, features, projections
        return logits
    
class VisionProjectionLayer(nn.Module):
    """Projects visual features from CNN to the LLM embedding space"""
    
    def __init__(self, 
                 vision_feature_dim=2048,     # CNN feature dimension
                 vision_classifier_dim=5,     # Number of components 
                 llm_embedding_dim=576,       # SmolLM2 embedding dimension
                 hidden_dim=768):             # Projection hidden dimension
        super().__init__()
        
        # Feature projection path - added an extra layer
        self.feature_projection = nn.Sequential(
            nn.Linear(vision_feature_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),  # Added layer
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, llm_embedding_dim)
        )
        
        # Classification result projection path
        self.classifier_projection = nn.Sequential(
            nn.Linear(vision_classifier_dim, hidden_dim // 4),
            nn.LayerNorm(hidden_dim // 4),
            nn.GELU(),
            nn.Linear(hidden_dim // 4, llm_embedding_dim)
        )
        
        # Combined attention pooling
        self.attention_pool = nn.Sequential(
            nn.Linear(llm_embedding_dim * 2, llm_embedding_dim),
            nn.Tanh()
        )
        
    def forward(self, features, classifier_outputs):
        # Project features and classifier outputs
        feature_embedding = self.feature_projection(features)
        classifier_embedding = self.classifier_projection(classifier_outputs)
        
        # Combine projections
        combined = torch.cat([feature_embedding, classifier_embedding], dim=1)
        visual_embedding = self.attention_pool(combined)
        
        return visual_embedding

# CNN Component Detector + SmolLM2 Implementation
class CarComponentVLM(nn.Module):
    """
    Custom VLM that combines:
    1. CNN car component detector
    2. Projection layer
    3. SmolLM2 LLM with LoRA
    """
    
    def __init__(self, cnn_path, projection_path, llm_path, adapter_path):
        super().__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Load the CNN model
        self.vision_model = ComponentClassifier(embedding_dim=256)
        checkpoint = torch.load(cnn_path, map_location=self.device)
        self.vision_model.load_state_dict(checkpoint['model_state_dict'])
        self.vision_model.eval()
        
        # Load the projection layer
        self.projection_layer = VisionProjectionLayer(
            vision_feature_dim=2048,  # ResNet50 feature dim
            vision_classifier_dim=5,  # 5 car components
            llm_embedding_dim=576,    # SmolLM2 embedding dim
            hidden_dim=768
        )
        projection_state_dict = torch.load(projection_path, map_location=self.device)
        self.projection_layer.load_state_dict(projection_state_dict)
        self.projection_layer.eval()
        
        # Load the tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(llm_path)
        special_tokens = {"additional_special_tokens": ["<image>", "</image>"]}
        self.tokenizer.add_special_tokens(special_tokens)
        
        # Load the LoRA-tuned LLM
        
        self.llm = AutoModelForCausalLM.from_pretrained(llm_path)
        self.llm.resize_token_embeddings(len(self.tokenizer))
        
        # Load LoRA adapter
        lora_path = adapter_path
        self.llm = PeftModel.from_pretrained(self.llm, lora_path)
        self.llm.eval()
        
        # Move everything to device
        self.vision_model = self.vision_model.to(self.device)
        self.projection_layer = self.projection_layer.to(self.device)
        self.llm = self.llm.to(self.device)
        
        # Image preprocessing
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def generate(self, image, prompt=None, **kwargs):
        """
        Generate description for an image
        """
        if isinstance(image, np.ndarray):
            # Convert from OpenCV format to PIL
            image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        
        # Apply transform
        img_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        # Default prompt if none provided
        if prompt is None:
            prompt = "Examine this car image and describe which doors and hood are open or closed."
        
        # Create messages
        messages = [
            {"role": "user", "content": f"{prompt}\n<image></image>"}
        ]
        
        # Format with chat template
        input_text = self.tokenizer.apply_chat_template(messages, tokenize=False)
        input_ids = self.tokenizer.encode(input_text, return_tensors="pt").to(self.device)
        
        # Process image through CNN and get features
        with torch.no_grad():
            logits, features, _ = self.vision_model(img_tensor, return_features=True)
            sigmoid_outputs = torch.sigmoid(logits)
        
        # Project visual features to LLM embedding space
        visual_embedding = self.projection_layer(features, sigmoid_outputs)
        
        # Replace <image> token with visual embedding
        image_token_id = self.tokenizer.convert_tokens_to_ids("<image>")
        image_pos = (input_ids[0] == image_token_id).nonzero(as_tuple=True)[0]
        
        if len(image_pos) > 0:
            image_pos = image_pos[0]
            llm_inputs = self.llm.get_input_embeddings()(input_ids)
            llm_inputs[0, image_pos] = visual_embedding[0]
            
            # Generate text
            outputs = self.llm.generate(
                inputs_embeds=llm_inputs,
                max_new_tokens=kwargs.get("max_new_tokens", 100),
                do_sample=kwargs.get("do_sample", True),
                temperature=kwargs.get("temperature", 0.2),
              )
            
            # Decode output
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract only the generated part after the prompt
            prompt_end = generated_text.find("assistant")
            if prompt_end != -1:
                assistant_text = generated_text[prompt_end+len("assistant"):]
                assistant_start = assistant_text.find(": ")
                if assistant_start != -1:
                    return assistant_text[assistant_start+2:]
            
            return generated_text