import torch
import torch.nn as nn
import torchvision.transforms as transforms
import numpy as np
from typing import Tuple, Union, Optional

class TorchImage:
    """
    PyTorch-compatible image container with preprocessing and augmentation.
    
    Handles conversion between numpy arrays and PyTorch tensors, normalization,
    and data augmentation for training neural networks.
    """
    
    def __init__(self, image: Union[np.ndarray, torch.Tensor], id: int):
        """Initialize TorchImage with image data and ID.
        
        Args:
            image: Input image as numpy array or torch tensor
            id: Image identifier
        """
        self.id = id
        self.I = self._prepare_tensor(image)
        
    def _prepare_tensor(self, image: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        """Convert input to normalized torch tensor.
        
        Args:
            image: Input image
        Returns:
            Normalized torch tensor
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
            
        return tensor / 255.0 if tensor.max() > 1.0 else tensor

    def augment(self, seed: Optional[int] = None) -> torch.Tensor:
        """Apply data augmentation to the image.
        
        Args:
            seed: Optional random seed for reproducibility (testing/debugging only)
        Returns:
            Augmented image tensor
        """
        # Create a generator if seed is provided (for testing)
        # This avoids modifying the global random state
        generator = None
        if seed is not None:
            generator = torch.Generator()
            generator.manual_seed(seed)
        
        # Basic augmentations
        basic_transforms = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.ColorJitter(
                brightness=0.2,
                contrast=0.1,
                saturation=0.1,
                hue=0.05
            )
        ])
        
        # Note: torchvision transforms don't support generator parameter
        # So for reproducible testing, we still need to set global seed temporarily
        if seed is not None:
            # Save current state
            state = torch.get_rng_state()
            torch.manual_seed(seed)
            result = basic_transforms(self.I)
            # Restore state
            torch.set_rng_state(state)
            return result
        else:
            # Training mode - use current random state
            return basic_transforms(self.I)

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
        return self