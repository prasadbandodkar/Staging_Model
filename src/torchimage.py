import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Union, Optional

class TorchImage:
    """
    PyTorch-compatible image container with GPU-accelerated preprocessing and augmentation.
    
    Handles conversion between numpy arrays and PyTorch tensors, normalization,
    and data augmentation for training neural networks. Supports GPU operations
    for maximum performance.
    """
    
    def __init__(
        self, 
        image: Union[np.ndarray, torch.Tensor], 
        id: float,
        device: Optional[Union[str, torch.device]] = None,
        skip_normalization: bool = False
    ):
        """Initialize TorchImage with image data and ID.
        
        Args:
            image: Input image as numpy array or torch tensor
            id: Image identifier (regression value or class label)
            device: Device to place tensor on ('cpu', 'cuda', 'mps', etc.). 
                   If None, keeps tensor on its current device.
            skip_normalization: If True, assumes image is already in [0,1] range
        """
        self.id = id
        self.device = device
        self.I = self._prepare_tensor(image, skip_normalization)
        
    def _prepare_tensor(
        self, 
        image: Union[np.ndarray, torch.Tensor],
        skip_normalization: bool = False
    ) -> torch.Tensor:
        """Convert input to normalized torch tensor.
        
        Args:
            image: Input image
            skip_normalization: If True, skip normalization step
        Returns:
            Normalized torch tensor in [0, 1] range with shape [C, H, W]
        """
        if isinstance(image, np.ndarray):
            tensor = torch.tensor(image, dtype=torch.float32)
        elif isinstance(image, torch.Tensor):
            tensor = image.float()
        else:
            raise TypeError(f"Unsupported image type: {type(image)}")
            
        # Ensure image is in correct shape [C, H, W]
        if len(tensor.shape) == 2:
            tensor = tensor.unsqueeze(0)  # Add channel dimension
        elif len(tensor.shape) == 3 and tensor.shape[0] not in [1, 3]:
            tensor = tensor.permute(2, 0, 1)  # Convert HWC to CHW
        
        # Normalize to [0, 1] if needed
        # if not skip_normalization:
        #     if tensor.max() > 1.0:
        #         tensor = tensor / 255.0
        tensor = tensor / 255.0
        
        # Move to device if specified
        if self.device is not None:
            tensor = tensor.to(self.device)
            
        return tensor

    def augment_gpu(self) -> torch.Tensor:
        """Apply GPU-accelerated data augmentation using pure PyTorch operations.
        
        Uses native PyTorch operations that run entirely on GPU without requiring
        external dependencies like Kornia. Works with any number of channels.
        
        Returns:
            Augmented image tensor on the same device
        """
        img = self.I
        
        # Random horizontal flip (50% probability)
        if torch.rand(1, device=img.device).item() > 0.5:
            img = torch.flip(img, dims=[2])  # Flip along width dimension
        
        # Random vertical flip (50% probability)
        if torch.rand(1, device=img.device).item() > 0.5:
            img = torch.flip(img, dims=[1])  # Flip along height dimension
        
        # Random brightness adjustment (80% probability)
        if torch.rand(1, device=img.device).item() < 0.8:
            # Brightness factor: 1.0 ± 0.4 (range: [0.6, 1.4])
            brightness_factor = 1.0 + (torch.rand(1, device=img.device).item() - 0.5) * 0.8
            img = torch.clamp(img * brightness_factor, 0, 1)
        
        # Random contrast adjustment (80% probability)
        if torch.rand(1, device=img.device).item() < 0.8:
            mean = img.mean()
            # Contrast factor: 1.0 ± 0.3 (range: [0.7, 1.3])
            contrast_factor = 1.0 + (torch.rand(1, device=img.device).item() - 0.5) * 0.6
            img = torch.clamp((img - mean) * contrast_factor + mean, 0, 1)
        
        return img
    
    def augment(self, seed: Optional[int] = None) -> torch.Tensor:
        """Apply data augmentation (chooses GPU or CPU based on tensor device).
        
        Automatically uses GPU-accelerated augmentation if tensor is on GPU,
        otherwise falls back to CPU augmentation using torchvision.
        
        Args:
            seed: Optional random seed for reproducibility (testing/debugging only)
                 Note: Seeding affects CPU augmentation only
        Returns:
            Augmented image tensor
        """
        # If tensor is on GPU, use pure PyTorch GPU augmentation
        if self.I.is_cuda or self.I.device.type == 'mps':
            if seed is not None:
                # Set seed for GPU operations
                torch.manual_seed(seed)
            return self.augment_gpu()
        
        # CPU fallback using torchvision
        import torchvision.transforms as transforms
        
        # Create augmentation pipeline
        basic_transforms = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.ColorJitter(
                brightness=0.25,
                contrast=0.25,
                saturation=0.0,  # Disabled for grayscale compatibility
                hue=0.0          # Disabled for grayscale compatibility
            )
        ])
        
        # Apply with optional seed
        if seed is not None:
            state = torch.get_rng_state()
            torch.manual_seed(seed)
            result = basic_transforms(self.I)
            torch.set_rng_state(state)
        else:
            result = basic_transforms(self.I)
        
        return result

    def normalize(self, method: str = 'minmax') -> torch.Tensor:
        """Normalize the image using specified method.
        
        Args:
            method: Normalization method ('minmax' or 'standardize')
        Returns:
            Normalized image tensor
        """
        if method == 'minmax':
            imin, imax = self.I.min(), self.I.max()
            if imin == imax:
                return self.I
            return (self.I - imin) / (imax - imin)
        elif method == 'standardize':
            mean = self.I.mean()
            std = self.I.std()
            if std == 0:
                return self.I - mean
            return (self.I - mean) / std
        else:
            raise ValueError(f"Unsupported normalization method: {method}")
            
    @property
    def shape(self) -> Tuple[int, ...]:
        """Get image shape."""
        return tuple(self.I.shape)
    
    def to_numpy(self) -> np.ndarray:
        """Convert to numpy array."""
        return self.I.cpu().numpy()
    
    def to_device(self, device: Union[str, torch.device]) -> 'TorchImage':
        """Move image to specified device."""
        self.I = self.I.to(device)
        self.device = device
        return self