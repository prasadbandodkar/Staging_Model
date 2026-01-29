"""
Staging Regression Model for Nuclear Layer Images

This module implements a ResNet-style CNN architecture for regression tasks,
specifically designed to predict staging values (0-1) from nuclear layer images.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation block for channel attention.
    
    Adaptively recalibrates channel-wise feature responses by explicitly
    modeling interdependencies between channels.
    """
    
    def __init__(self, channels: int, reduction: int = 16):
        """
        Initialize the SE block.
        
        Args:
            channels: Number of input channels
            reduction: Reduction ratio for the bottleneck (default: 16)
        """
        super(SEBlock, self).__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the SE block.
        
        Args:
            x: Input tensor of shape (B, C, H, W)
            
        Returns:
            Channel-wise recalibrated tensor of shape (B, C, H, W)
        """
        b, c, _, _ = x.size()
        # Squeeze: global spatial information
        y = self.squeeze(x).view(b, c)
        # Excitation: channel-wise attention weights
        y = self.excitation(y).view(b, c, 1, 1)
        # Scale: apply attention weights
        return x * y.expand_as(x)



class ResidualBlock(nn.Module):
    """
    Pre-activation residual block with GroupNorm, GELU, and SE attention.
    
    Uses pre-activation design (BN→ReLU→Conv) for better gradient flow,
    GroupNorm for batch-size independence, GELU for smoother gradients,
    and Squeeze-Excitation for channel-wise attention.
    """
    
    def __init__(
        self, 
        in_channels: int, 
        out_channels: int, 
        stride: int = 1, 
        downsample: Optional[nn.Module] = None,
        num_groups: int = 32
    ):
        """
        Initialize the pre-activation residual block.
        
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            stride: Stride for the first convolution (default: 1)
            downsample: Optional downsampling layer for the skip connection
            num_groups: Number of groups for GroupNorm (default: 32)
        """
        super(ResidualBlock, self).__init__()
        
        # Adjust num_groups if channels are too small
        num_groups = min(num_groups, out_channels)
        
        # Pre-activation design: GN → GELU → Conv
        self.gn1 = nn.GroupNorm(num_groups=num_groups, num_channels=in_channels)
        self.gelu1 = nn.GELU()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, 
            kernel_size=3, stride=stride, padding=1, bias=False
        )
        
        self.gn2 = nn.GroupNorm(num_groups=num_groups, num_channels=out_channels)
        self.gelu2 = nn.GELU()
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, 
            kernel_size=3, stride=1, padding=1, bias=False
        )
        
        # Squeeze-and-Excitation block
        self.se = SEBlock(out_channels, reduction=16)
        
        self.downsample = downsample
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the pre-activation residual block.
        
        Args:
            x: Input tensor of shape (B, C, H, W)
            
        Returns:
            Output tensor of shape (B, out_channels, H', W')
        """
        identity = x
        
        # Pre-activation: GN → GELU → Conv
        out = self.gn1(x)
        out = self.gelu1(out)
        out = self.conv1(out)
        
        # Second pre-activation block
        out = self.gn2(out)
        out = self.gelu2(out)
        out = self.conv2(out)
        
        # Squeeze-and-Excitation
        out = self.se(out)
        
        # Apply downsampling to identity if needed
        if self.downsample is not None:
            identity = self.downsample(x)
        
        # Add skip connection (no activation after addition in pre-activation design)
        out += identity
        
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
        num_classes: int = 1,
        activation: str = 'gelu',
        normalization_type: str = 'group_norm',
        num_groups: int = 32,
        se_enabled: bool = True,
        se_reduction: int = 16,
        pre_activation: bool = True
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
        self.num_groups = min(32, base_channels)  # Adjust for small models
        
        # Initial convolution layer
        self.conv1 = nn.Conv2d(
            in_channels, base_channels, 
            kernel_size=7, stride=2, padding=3, bias=False
        )
        self.gn1 = nn.GroupNorm(num_groups=self.num_groups, num_channels=base_channels)
        self.gelu = nn.GELU()
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

        # Shared feature extractor (deeper than before)
        self.shared_fc = nn.Sequential(
            nn.Linear(c4, 256),
            nn.GELU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(256, 128),
            nn.GELU(),
        )
        
        # Task-specific output heads
        if self.task_type == 'regression':
            # Regression: deeper head for single output
            self.task_head = nn.Sequential(
                nn.Dropout(p=dropout_rate * 0.5),
                nn.Linear(128, 64),
                nn.GELU(),
                nn.Linear(64, 1)
            )
        else:
            # Classification: task-specific head for multi-class output
            self.task_head = nn.Sequential(
                nn.Dropout(p=dropout_rate * 0.5),
                nn.Linear(128, self.num_classes)
            )

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
            
            # Adjust num_groups for downsampling projection
            num_groups_proj = min(self.num_groups, out_channels)
            
            # Always projection with 1x1 Conv (stride 1) + GN
            downsample_layers.extend([
                nn.Conv2d(
                    self.in_channels, out_channels, 
                    kernel_size=1, stride=1, bias=False
                ),
                nn.GroupNorm(num_groups=num_groups_proj, num_channels=out_channels)
            ])
            
            downsample = nn.Sequential(*downsample_layers)
        
        layers = []
        # First block with potential downsampling
        layers.append(
            ResidualBlock(self.in_channels, out_channels, stride, downsample, self.num_groups)
        )
        
        # Update in_channels for subsequent blocks
        self.in_channels = out_channels
        
        # Add remaining blocks
        for _ in range(1, num_blocks):
            layers.append(ResidualBlock(out_channels, out_channels, num_groups=self.num_groups))
        
        return nn.Sequential(*layers)
    
    def _initialize_weights(self) -> None:
        """
        Initialize model weights using He initialization for conv layers
        and Xavier for linear layers. Updated for GroupNorm.
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
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
        x = self.gn1(x)
        x = self.gelu(x)
        x = self.maxpool(x)

        # ResNet layers
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # Global pooling
        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        # Shared feature extractor
        x = self.shared_fc(x)
        
        # Task-specific head
        x = self.task_head(x)

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
    num_blocks: Optional[Tuple[int, int, int, int]] = None,
    activation: str = 'gelu',
    normalization_type: str = 'group_norm',
    num_groups: int = 32,
    se_enabled: bool = True,
    se_reduction: int = 16,
    pre_activation: bool = True
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
        activation: Activation function ('gelu', 'relu', 'mish', 'leaky_relu')
        normalization_type: Type of normalization ('group_norm' or 'batch_norm')
        num_groups: Number of groups for GroupNorm (only used if normalization_type='group_norm')
        se_enabled: Whether to use SE blocks for channel attention
        se_reduction: Reduction ratio for SE bottleneck
        pre_activation: Use pre-activation design (True) or post-activation (False)

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
        num_classes=num_classes,
        activation=activation,
        normalization_type=normalization_type,
        num_groups=num_groups,
        se_enabled=se_enabled,
        se_reduction=se_reduction,
        pre_activation=pre_activation
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
