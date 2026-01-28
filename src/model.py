"""
Staging Regression Model for Nuclear Layer Images

This module implements a ResNet-style CNN architecture for regression tasks,
specifically designed to predict staging values (0-1) from nuclear layer images.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple



class ResidualBlock(nn.Module):
    """
    Residual block with two convolutional layers and a skip connection.
    
    Implements the basic building block commonly used in ResNet architectures,
    with batch normalization and ReLU activation.
    """
    
    def __init__(
        self, 
        in_channels: int, 
        out_channels: int, 
        stride: int = 1, 
        downsample: Optional[nn.Module] = None
    ):
        """
        Initialize the residual block.
        
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            stride: Stride for the first convolution (default: 1)
            downsample: Optional downsampling layer for the skip connection
        """
        super(ResidualBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, 
            kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, 
            kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.downsample = downsample
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the residual block.
        
        Args:
            x: Input tensor of shape (B, C, H, W)
            
        Returns:
            Output tensor of shape (B, out_channels, H', W')
        """
        identity = x
        
        # First conv block
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        # Second conv block
        out = self.conv2(out)
        out = self.bn2(out)
        
        # Apply downsampling to identity if needed
        if self.downsample is not None:
            identity = self.downsample(x)
        
        # Add skip connection
        out += identity
        out = self.relu(out)
        
        return out


class Model(nn.Module):
    """
    ResNet-style CNN for regression or classification on nuclear layer images.

    For regression: Predicts a continuous staging value between 0 and 1.
    For classification: Predicts class probabilities for discrete staging classes.
    Designed to handle variable-width inputs through adaptive pooling.
    """

    def __init__(
        self,
        in_channels: int = 1,
        base_channels: int = 64,
        num_blocks: Tuple[int, int, int, int] = (2, 2, 2, 2),
        dropout_rate: float = 0.5,
        task_type: str = 'regression',
        num_classes: int = 1
    ):
        """
        Initialize the model.

        Args:
            in_channels: Number of input channels (default: 1 for grayscale)
            base_channels: Number of filters in the first layer (default: 64).
                           Controls the width of the entire network.
            num_blocks: Number of residual blocks in each layer (default: (2,2,2,2))
            dropout_rate: Dropout probability for regularization (default: 0.5)
            task_type: Task type ('regression' or 'classification')
            num_classes: Number of output classes (only used for classification)
        """
        super(Model, self).__init__()

        self.task_type = task_type
        self.num_classes = num_classes
        
        self.in_channels = base_channels
        
        # Initial convolution layer
        self.conv1 = nn.Conv2d(
            in_channels, base_channels, 
            kernel_size=7, stride=2, padding=3, bias=False
        )
        self.bn1 = nn.BatchNorm2d(base_channels)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # ResNet layers with increasing channels: base -> base*2 -> base*4 -> base*8
        c1 = base_channels
        c2 = base_channels * 2
        c3 = base_channels * 4
        c4 = base_channels * 8
        
        self.layer1 = self._make_layer(out_channels=c1, num_blocks=num_blocks[0], stride=1)
        self.layer2 = self._make_layer(out_channels=c2, num_blocks=num_blocks[1], stride=2)
        self.layer3 = self._make_layer(out_channels=c3, num_blocks=num_blocks[2], stride=2)
        self.layer4 = self._make_layer(out_channels=c4, num_blocks=num_blocks[3], stride=2)
        
        # Adaptive pooling to handle variable-width inputs
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # Task-specific output head
        self.fc1 = nn.Linear(c4, 128)
        self.dropout = nn.Dropout(p=dropout_rate)

        if self.task_type == 'regression':
            # Regression: single output with sigmoid activation
            self.fc2 = nn.Linear(128, 1)
        else:
            # Classification: num_classes outputs (no activation, CrossEntropyLoss expects logits)
            self.fc2 = nn.Linear(128, self.num_classes)

        # Initialize weights
        self._initialize_weights()
        
    def _make_layer(
        self, 
        out_channels: int, 
        num_blocks: int, 
        stride: int = 1
    ) -> nn.Sequential:
        """
        Create a layer with multiple residual blocks.
        
        Args:
            out_channels: Number of output channels for this layer
            num_blocks: Number of residual blocks in this layer
            stride: Stride for the first block (default: 1)
            
        Returns:
            Sequential module containing the residual blocks
        """
        downsample = None
        
        # Create downsampling layer if dimensions change
        if stride != 1 or self.in_channels != out_channels:
            downsample_layers = []
            
            # ResNet-D: Use AvgPool for spatial downsampling if needed
            if stride != 1:
                downsample_layers.append(nn.AvgPool2d(kernel_size=2, stride=stride))
            
            # Always projection with 1x1 Conv (stride 1) + BN
            downsample_layers.extend([
                nn.Conv2d(
                    self.in_channels, out_channels, 
                    kernel_size=1, stride=1, bias=False
                ),
                nn.BatchNorm2d(out_channels)
            ])
            
            downsample = nn.Sequential(*downsample_layers)
        
        layers = []
        # First block with potential downsampling
        layers.append(
            ResidualBlock(self.in_channels, out_channels, stride, downsample)
        )
        
        # Update in_channels for subsequent blocks
        self.in_channels = out_channels
        
        # Add remaining blocks
        for _ in range(1, num_blocks):
            layers.append(ResidualBlock(out_channels, out_channels))
        
        return nn.Sequential(*layers)
    
    def _initialize_weights(self) -> None:
        """
        Initialize model weights using He initialization for conv layers
        and Xavier for linear layers.
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the model.

        Args:
            x: Input tensor of shape (B, C, H, W)

        Returns:
            For regression: Predicted staging values of shape (B, 1) in range [0, 1]
            For classification: Logits of shape (B, num_classes)
        """
        # Initial conv + pooling
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        # ResNet layers
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # Global pooling
        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        # Output head
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)

        # Task-specific activation
        if self.task_type == 'regression':
            # Sigmoid to constrain output to [0, 1]
            x = torch.sigmoid(x)
        # For classification, return logits (no activation)

        return x


def create_staging_model(
    model_type: str = 'small',
    in_channels: int = 1,
    dropout_rate: float = 0.5,
    task_type: str = 'regression',
    num_classes: int = 1,
    base_channels: Optional[int] = None,
    num_blocks: Optional[Tuple[int, int, int, int]] = None
) -> Model:
    """
    Factory function to create different model variants.

    Args:
        model_type: Model size ('nano', 'tiny', 'small', 'medium', 'large')
        in_channels: Number of input channels (default: 1)
        dropout_rate: Dropout probability (default: 0.5)
        task_type: Task type ('regression' or 'classification')
        num_classes: Number of output classes (only used for classification)
        base_channels: Optional override for base channel count/width.
                       If None, derived from model_type defaults.
        num_blocks: Optional override for residual blocks structure.

    Returns:
        Initialized Model

    Raises:
        ValueError: If model_type is not recognized
    """
    # Configs: (default_base_channels, default_num_blocks)
    model_configs = {
        'nano':   (8, (1, 1, 1, 1)),  # Extremely lightweight (<150k params target)
        'tiny':   (32, (1, 1, 1, 1)), # ~700k params
        'small':  (64, (2, 2, 2, 2)), # ~11M params (Standard ResNet18-ish)
        'medium': (64, (3, 4, 6, 3)), # ~25M params (ResNet34-ish)
        'large':  (64, (3, 4, 23, 3)) # ResNet101-ish depth
    }

    if model_type not in model_configs:
        raise ValueError(
            f"Unknown model_type '{model_type}'. "
            f"Choose from: {list(model_configs.keys())}"
        )

    default_channels, default_blocks = model_configs[model_type]

    # Use overrides if provided, otherwise defaults
    final_channels = base_channels if base_channels is not None else default_channels
    final_blocks = num_blocks if num_blocks is not None else default_blocks

    return Model(
        in_channels=in_channels,
        base_channels=final_channels,
        num_blocks=final_blocks,
        dropout_rate=dropout_rate,
        task_type=task_type,
        num_classes=num_classes
    )


def get_model_summary(model: nn.Module, input_size: Tuple[int, int, int, int]) -> None:
    """
    Print model summary including parameter count and layer structure.
    
    Args:
        model: PyTorch model
        input_size: Input tensor size (B, C, H, W)
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print("=" * 80)
    print(f"Model: {model.__class__.__name__}")
    print("=" * 80)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Non-trainable parameters: {total_params - trainable_params:,}")
    print("=" * 80)
    print(f"{'Layer (type)':<25} {'Output Shape':<25} {'Param #':<15}")
    print("=" * 80)

    # Hooks to capture shape
    layer_outputs = []
    hooks = []

    def hook_fn(module, input, output):
        class_name = str(module.__class__).split(".")[-1].split("'")[0]
        module_idx = len(layer_outputs)
        
        m_key = f"{class_name}-{module_idx+1}"
        layer_outputs.append({
            "name": m_key,
            "shape": tuple(output.shape),
            "params": sum(p.numel() for p in module.parameters())
        })

    # Register hooks on top-level children
    for name, module in model.named_children():
        hooks.append(module.register_forward_hook(hook_fn))
    
    # Test forward pass
    model.eval()
    with torch.no_grad():
        dummy_input = torch.randn(input_size)
        # Verify device
        device = next(model.parameters()).device
        dummy_input = dummy_input.to(device)
        
        print(f"Input Shape: {tuple(dummy_input.shape)}")
        try:
            output = model(dummy_input)
            
            # Print captured layers
            for layer in layer_outputs:
                print(f"{layer['name']:<25} {str(layer['shape']):<25} {layer['params']:<15,}")
                
            print("=" * 80)
            print(f"Final Output: {tuple(output.shape)}")
            print(f"Output range: [{output.min():.4f}, {output.max():.4f}]")
            
        except Exception as e:
            print(f"Error during forward pass: {e}")
        finally:
            # Cleanup hooks
            for h in hooks:
                h.remove()
    print("=" * 80)


if __name__ == "__main__":
    # Example usage and testing
    print("Testing Staging Regression Model\n")
    
    # Create different model variants
    for model_type in ['nano', 'tiny', 'small']:
        print(f"\n{'='*80}")
        print(f"Testing {model_type.upper()} model")
        print('='*80)
        
        model = create_staging_model(model_type=model_type)
        
        # Test with typical nuclear layer dimensions
        batch_size = 256
        channels = 1
        height = 128
        width = 256
        
        get_model_summary(model, (batch_size, channels, height, width))
        
        print()
