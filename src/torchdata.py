#
# TorchData class - PyTorch-specific data loading functionality
# Extends the base Data class with GPU-accelerated tensor loading capabilities
#

# Main imports
from typing import Tuple, Optional

# Machine learning imports
import torch
import torchvision.io
import torchvision.transforms.functional

# Local imports
from .data import Data, ListType


class TorchData(Data):
    """
    PyTorch-enabled data loader that extends the base Data class.
    
    Provides GPU-accelerated image loading using torchvision, with direct
    tensor loading to GPU and optional resizing for preprocessed pipelines.
    """
    
    def get_raw_image_torch(
        self, 
        folder: str, 
        idx: int, 
        list_type: ListType, 
        device: str = 'cpu'
    ) -> Tuple[torch.Tensor, float]:
        """
        Load a raw image from disk as a PyTorch tensor, optionally directly to GPU.
        
        This method is optimized for GPU-accelerated preprocessing pipelines.
        Uses torchvision.io.read_image for fast tensor loading.
        
        Args:
            folder: Name of the folder containing the image
            idx: Index of the image within the folder
            list_type: Type of dataset ('train', 'test', or 'val')
            device: Device to load the tensor to ('cpu', 'cuda', 'mps', etc.)
            target_size: Optional (height, width) to resize image to. If None, no resizing.
            
        Returns:
            Tuple of (image tensor [C, H, W] in range [0, 1], image ID)
            
        Raises:
            ValueError: If list_type is invalid
            KeyError: If folder is not found
            IndexError: If index is out of bounds
            FileNotFoundError: If image file cannot be loaded
        """
        # Check if list_type is valid
        list_dict = {
            'train': self.train_data,
            'test': self.test_data,
            'val': self.val_data
        }
        if list_type not in list_dict:
            raise ValueError(f"Invalid list_type '{list_type}'. Expected one of: {list(list_dict.keys())}")

        data = list_dict[list_type]

        # Check if folder is valid
        if folder not in data:
            raise KeyError(f"Folder '{folder}' not found in {list_type} data.")

        # Check if idx is valid
        if not (0 <= idx < len(data[folder])):
            raise IndexError(f"Index {idx} is out of bounds for folder '{folder}' with size {len(data[folder])}.")

        filename: str = data[folder].iloc[idx, 0]
        id_value: float = data[folder].iloc[idx, 1]
        
        # Load image as tensor
        try:
            # read_image returns tensor in [0, 255] range with shape [C, H, W]
            # Use GRAY mode for grayscale images
            I_tensor = torchvision.io.read_image(filename, mode=torchvision.io.ImageReadMode.GRAY)
            
            # Convert to float32 and normalize to [0, 1]
            # I_tensor = I_tensor.float() / 255.0
            
            # Move to specified device
            I_tensor = I_tensor.to(device)
            
        except Exception as e:
            raise FileNotFoundError(f"Image '{filename}' could not be loaded as tensor: {e}")

        return I_tensor, id_value
