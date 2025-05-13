import torch
import torch.nn as nn


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