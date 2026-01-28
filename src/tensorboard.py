"""
TensorBoard Logger

Comprehensive logging for training metrics, sample predictions, and data transformations.
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
from typing import Dict, Optional, List, Tuple
from dataclasses import asdict


class TensorBoardLogger:
    """
    Wrapper around TensorBoard SummaryWriter for model training.
    
    Logs:
    - Scalars: loss, MAE, MSE, R², learning rate
    - Images: predictions vs targets, data transformations
    - Distributions: model weights, gradients
    - Hyperparameters: config + final metrics
    """
    
    def __init__(self, log_dir: str, enabled: bool = True):
        """
        Initialize TensorBoard logger.
        
        Args:
            log_dir: Directory for TensorBoard event files
            enabled: Whether logging is enabled
        """
        self.enabled = enabled
        self.log_dir = Path(log_dir)
        
        if self.enabled:
            self.log_dir.mkdir(parents=True, exist_ok=True)
            self.writer = SummaryWriter(str(self.log_dir))
            print(f"✓ TensorBoard logging enabled: {self.log_dir}")
        else:
            self.writer = None
            print("TensorBoard logging disabled")
    
    def log_scalar(self, tag: str, value: float, step: int):
        """Log a scalar value."""
        if self.enabled:
            self.writer.add_scalar(tag, value, step)
    
    def log_scalars(self, main_tag: str, tag_scalar_dict: Dict[str, float], step: int):
        """Log multiple scalars under a main tag."""
        if self.enabled:
            self.writer.add_scalars(main_tag, tag_scalar_dict, step)
    
    def log_metrics(self, metrics: Dict[str, float], step: int, prefix: str = ""):
        """
        Log training/validation metrics.
        
        Args:
            metrics: Dictionary of metrics (loss, mae, mse, r2)
            step: Current epoch/step
            prefix: Prefix for tags (e.g., 'train/' or 'val/')
        """
        if not self.enabled:
            return
        
        for key, value in metrics.items():
            self.writer.add_scalar(f"{prefix}{key}", value, step)
    
    def log_learning_rate(self, lr: float, step: int):
        """Log learning rate."""
        if self.enabled:
            self.writer.add_scalar("learning_rate", lr, step)
    
    def log_images(self, tag: str, images: torch.Tensor, step: int, max_images: int = 8):
        """
        Log a batch of images.
        
        Args:
            tag: Tag for the images
            images: Tensor of shape (N, C, H, W) or (N, H, W)
            step: Current step
            max_images: Maximum number of images to log
        """
        if not self.enabled:
            return
        
        # Limit number of images
        if images.shape[0] > max_images:
            images = images[:max_images]
        
        # Ensure 4D tensor (N, C, H, W)
        if images.ndim == 3:
            images = images.unsqueeze(1)
        
        # Normalize to [0, 1] if needed
        if images.min() < 0 or images.max() > 1:
            images = (images - images.min()) / (images.max() - images.min() + 1e-8)
        
        self.writer.add_images(tag, images, step)
    
    def log_predictions(
        self, 
        images: torch.Tensor, 
        predictions: torch.Tensor, 
        targets: torch.Tensor,
        step: int,
        num_samples: int = 4
    ):
        """
        Log sample predictions vs targets.
        
        Creates a visualization showing:
        - Input image
        - Predicted value
        - Target value
        - Error
        
        Args:
            images: Input images (N, C, H, W)
            predictions: Model predictions (N, 1)
            targets: Ground truth (N, 1)
            step: Current step
            num_samples: Number of samples to visualize
        """
        if not self.enabled:
            return
        
        # Limit samples
        num_samples = min(num_samples, images.shape[0])
        images = images[:num_samples].cpu()
        predictions = predictions[:num_samples].cpu().flatten()
        targets = targets[:num_samples].cpu().flatten()
        
        # Create figure
        fig, axes = plt.subplots(2, num_samples, figsize=(num_samples * 3, 6))
        if num_samples == 1:
            axes = axes.reshape(2, 1)
        
        for i in range(num_samples):
            # Show image
            ax_img = axes[0, i]
            img = images[i].squeeze()
            ax_img.imshow(img, cmap='gray')
            ax_img.axis('off')
            ax_img.set_title(f"Sample {i+1}")
            
            # Show prediction vs target
            ax_pred = axes[1, i]
            pred_val = predictions[i].item()
            target_val = targets[i].item()
            error = abs(pred_val - target_val)
            
            ax_pred.barh([0, 1], [target_val, pred_val], color=['blue', 'red'], alpha=0.7)
            ax_pred.set_yticks([0, 1])
            ax_pred.set_yticklabels(['Target', 'Pred'])
            ax_pred.set_xlim(0, 1)
            ax_pred.set_xlabel('Staging Value')
            ax_pred.set_title(f"Pred: {pred_val:.3f}\nTarget: {target_val:.3f}\nError: {error:.3f}")
            ax_pred.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Convert to tensor and log
        fig.canvas.draw()
        # Use buffer_rgba() which works across all backends
        # Note: get_width_height() returns (width, height) but buffer is in (height, width) order
        width, height = fig.canvas.get_width_height()
        img_array = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        # Calculate expected size and reshape accordingly
        expected_size = width * height * 4
        if img_array.size != expected_size:
            # DPI scaling may cause size mismatch, calculate actual dimensions
            actual_height = img_array.size // (width * 4)
            img_array = img_array.reshape(actual_height, width, 4)
        else:
            img_array = img_array.reshape(height, width, 4)
        # Convert RGBA to RGB
        img_array = img_array[:, :, :3]
        img_tensor = torch.from_numpy(img_array).permute(2, 0, 1) / 255.0
        
        self.writer.add_image("predictions/samples", img_tensor, step)
        plt.close(fig)
    
    def log_data_transformation_pipeline(
        self,
        original: torch.Tensor,
        preprocessed: Optional[torch.Tensor] = None,
        segmented: Optional[torch.Tensor] = None,
        unrolled: Optional[torch.Tensor] = None,
        final: Optional[torch.Tensor] = None,
        step: int = 0,
        num_samples: int = 4
    ):
        """
        Log data transformation pipeline stages.
        
        Shows how images are transformed through the preprocessing pipeline.
        
        Args:
            original: Original images (N, ...)
            preprocessed: After resize/padding (N, ...)
            segmented: Segmentation masks (N, ...)
            unrolled: Unrolled layer (N, ...)
            final: Final processed images (N, ...)
            step: Current step (usually 0 for initial logging)
            num_samples: Number of samples to show
        """
        if not self.enabled:
            return
        
        # Collect available stages
        stages = []
        stage_names = []
        
        if original is not None:
            stages.append(original[:num_samples].cpu())
            stage_names.append("Original")
        if preprocessed is not None:
            stages.append(preprocessed[:num_samples].cpu())
            stage_names.append("Preprocessed")
        if segmented is not None:
            stages.append(segmented[:num_samples].cpu())
            stage_names.append("Segmented")
        if unrolled is not None:
            stages.append(unrolled[:num_samples].cpu())
            stage_names.append("Unrolled")
        if final is not None:
            stages.append(final[:num_samples].cpu())
            stage_names.append("Final")
        
        if not stages:
            return
        
        # Create visualization
        num_stages = len(stages)
        num_samples = min(num_samples, stages[0].shape[0])
        
        fig, axes = plt.subplots(num_samples, num_stages, figsize=(num_stages * 3, num_samples * 3))
        if num_samples == 1:
            axes = axes.reshape(1, -1)
        if num_stages == 1:
            axes = axes.reshape(-1, 1)
        
        for i in range(num_samples):
            for j, (stage, name) in enumerate(zip(stages, stage_names)):
                ax = axes[i, j]
                img = stage[i].squeeze()
                ax.imshow(img, cmap='gray')
                ax.axis('off')
                if i == 0:
                    ax.set_title(name, fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        
        # Convert to tensor and log
        fig.canvas.draw()
        # Use buffer_rgba() which works across all backends
        # Note: get_width_height() returns (width, height) but buffer is in (height, width) order
        width, height = fig.canvas.get_width_height()
        img_array = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        # Calculate expected size and reshape accordingly
        expected_size = width * height * 4
        if img_array.size != expected_size:
            # DPI scaling may cause size mismatch, calculate actual dimensions
            actual_height = img_array.size // (width * 4)
            img_array = img_array.reshape(actual_height, width, 4)
        else:
            img_array = img_array.reshape(height, width, 4)
        # Convert RGBA to RGB
        img_array = img_array[:, :, :3]
        img_tensor = torch.from_numpy(img_array).permute(2, 0, 1) / 255.0
        
        self.writer.add_image("data_pipeline/transformation_stages", img_tensor, step)
        plt.close(fig)
        
        print(f"✓ Logged data transformation pipeline ({num_samples} samples, {num_stages} stages)")
    
    def log_model_graph(self, model: nn.Module, input_shape: Tuple):
        """
        Log model computational graph.
        
        Args:
            model: PyTorch model
            input_shape: Shape of input tensor (e.g., (1, 1, 128, 128))
        """
        if not self.enabled:
            return
        
        try:
            dummy_input = torch.zeros(input_shape)
            self.writer.add_graph(model, dummy_input)
            print("✓ Logged model graph")
        except Exception as e:
            print(f"Warning: Could not log model graph: {e}")
    
    def log_hyperparameters(self, hparams: Dict, metrics: Dict):
        """
        Log hyperparameters and final metrics.
        
        Args:
            hparams: Dictionary of hyperparameters
            metrics: Dictionary of final metrics
        """
        if not self.enabled:
            return
        
        # Flatten nested dicts
        flat_hparams = {}
        for key, value in hparams.items():
            if isinstance(value, dict):
                for subkey, subvalue in value.items():
                    flat_hparams[f"{key}/{subkey}"] = subvalue
            else:
                flat_hparams[key] = value
        
        # Convert any non-scalar values to strings
        for key in flat_hparams:
            if isinstance(flat_hparams[key], (list, tuple)):
                flat_hparams[key] = str(flat_hparams[key])
        
        self.writer.add_hparams(flat_hparams, metrics)
        print("✓ Logged hyperparameters")
    
    def close(self):
        """Close the writer."""
        if self.enabled and self.writer is not None:
            self.writer.close()
            print("✓ Closed TensorBoard logger")
