"""Test the PyTorch data pipeline."""

# Basic imports
import torch
import random

# Local imports
from src.data import Data
from src.cvimage import CVImage
from src.torchimage import TorchImage


def get_torch_device():
    """
    Detect and return the best available PyTorch device.
    
    Returns:
        torch.device: The device to use (cuda, mps, or cpu)
    """
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
        print("PyTorch IS using CUDA")
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
        print("PyTorch IS using MPS")
    else:
        device = torch.device('cpu')
        print("PyTorch IS NOT using GPU")
    
    return device


def test_torch_pipeline(
    data_folder: str = '/Volumes/X2/Projects/phd/staging/data/data',
    test_ids: list = None,
    val_ids: list = None,
    seed: int = 42
):
    """
    Test the PyTorch data pipeline with the Data and TorchImage classes.
    
    Args:
        data_folder: Path to the data directory
        test_ids: List of test folder IDs
        val_ids: List of validation folder IDs
        seed: Random seed for reproducibility
    
    Returns:
        tuple: (image, device) - TorchImage on device and the device object
    """
    # Set defaults
    if test_ids is None:
        test_ids = [6, 7]
    if val_ids is None:
        val_ids = [21, 34]
    
    print(f"PyTorch version: {torch.__version__}")
    device = get_torch_device()
    print()
    
    # Set random seed for reproducibility
    random.seed(seed)
    torch.manual_seed(seed)
    
    # Initialize data loader
    print(f"Loading data from: {data_folder}")
    print(f"Test IDs: {test_ids}, Val IDs: {val_ids}")
    d = Data(data_folder, test=test_ids, val=val_ids)
    
    # Get a random training image
    I, id, folder, idx = d.get_random_image('train')
    print(f"\nSelected folder: {folder}, index: {idx}")
    
    # Create TorchImage
    image = TorchImage(CVImage(I, id).image, id)
    
    # Print image information
    print(f"Image shape: {image.I.shape}")
    print(f"Image dtype: {image.I.dtype}")
    print(f"Image id: {image.id}")
    print(f"Value range: [{torch.min(image.I):.4f}, {torch.max(image.I):.4f}]")
    
    # Move to device if not CPU
    if device.type != 'cpu':
        print(f"\nMoving image to {device.type.upper()}...")
        image_gpu = image.to_device(str(device.type))
        print(f"Image device: {image_gpu.I.device}")
        print(f"Value range on {device.type.upper()}: [{torch.min(image_gpu.I):.4f}, {torch.max(image_gpu.I):.4f}]")
        return image_gpu, device
    
    return image, device


if __name__ == "__main__":
    # Run the test
    try:
        image, device = test_torch_pipeline()
        print("\n✓ PyTorch pipeline test completed successfully!")
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        raise
