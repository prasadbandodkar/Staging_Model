"""
Training script

Implements comprehensive training pipeline with validation, checkpointing,
learning rate scheduling, and metrics logging.
"""

import os
import sys
import json
import random
from pathlib import Path
from typing import Dict, Tuple, Optional, List, Union
from datetime import datetime
from dataclasses import asdict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

# Import local modules
from .model import Model, create_staging_model, get_model_summary
from .torchdataset import TorchDataset
from .config import AppConfig, TrainingConfig, DataConfig, ModelConfig
from .run_manager import RunManager



class MetricsTracker:
    """
    Track and compute training metrics for regression or classification tasks.

    For regression, computes:
    - MSE (Mean Squared Error)
    - MAE (Mean Absolute Error)
    - R² Score (Coefficient of Determination)
    - Loss

    For classification, computes:
    - Accuracy (percentage of correct predictions)
    - Loss
    """

    def __init__(self, task_type: str = 'regression'):
        """
        Initialize metrics tracker.

        Args:
            task_type: 'regression' or 'classification'
        """
        self.task_type = task_type
        self.reset()

    def reset(self):
        """Reset all metrics."""
        self.losses = []
        self.predictions = []
        self.targets = []

    def update(self, loss: float, predictions: torch.Tensor, targets: torch.Tensor):
        """Update metrics with new batch."""
        self.losses.append(loss)
        # Store on CPU to save GPU memory. Use torch for final computation.
        self.predictions.append(predictions.detach().cpu())
        self.targets.append(targets.detach().cpu())

    def compute(self) -> Dict[str, float]:
        """Compute aggregated metrics."""
        # Average loss
        avg_loss = np.mean(self.losses)

        if self.task_type == 'regression':
            # Concatenate all batches efficiently
            preds = torch.cat(self.predictions).flatten()
            targs = torch.cat(self.targets).flatten()

            # Mean Squared Error
            mse = torch.mean((preds - targs) ** 2).item()

            # Mean Absolute Error
            mae = torch.mean(torch.abs(preds - targs)).item()

            # R² Score
            ss_res = torch.sum((targs - preds) ** 2)
            ss_tot = torch.sum((targs - torch.mean(targs)) ** 2)
            r2 = (1 - (ss_res / (ss_tot + 1e-8))).item()

            return {
                'loss': avg_loss,
                'mse': mse,
                'mae': mae,
                'r2': r2
            }
        else:  # classification
            # Concatenate all batches
            preds = torch.cat(self.predictions)  # (N, num_classes) logits
            targs = torch.cat(self.targets).long()  # (N,) class indices

            # Get predicted classes
            pred_classes = torch.argmax(preds, dim=1)

            # Accuracy
            correct = (pred_classes == targs).sum().item()
            total = targs.size(0)
            accuracy = correct / total

            return {
                'loss': avg_loss,
                'accuracy': accuracy
            }



class Trainer:
    """
    Comprehensive trainer for model.
    
    Handles training loop, validation, checkpointing, and metrics tracking.
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: TrainingConfig,
        device: torch.device,
        task_type: str = 'regression',
        logger=None
    ):
        """
        Initialize trainer.

        Args:
            model: PyTorch model to train
            train_loader: DataLoader for training data
            val_loader: DataLoader for validation data
            config: Training configuration object
            device: Device to train on (cpu or cuda)
            task_type: Task type ('regression' or 'classification')
            logger: Optional TensorBoardLogger instance
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device
        self.task_type = task_type
        self.logger = logger

        # Loss function
        self.criterion = self._get_loss_function()

        # Optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config.optimizer.learning_rate,
            weight_decay=config.optimizer.weight_decay
        )

        # Learning rate scheduler
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=config.scheduler.factor,
            patience=config.scheduler.patience
        )

        # Training state
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.best_val_epoch = 0
        self.epochs_without_improvement = 0

        # Checkpointing
        self.checkpoint_dir = Path(config.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # History (task-specific)
        if self.task_type == 'regression':
            self.history = {
                'train_loss': [],
                'train_mae': [],
                'train_r2': [],
                'val_loss': [],
                'val_mae': [],
                'val_r2': [],
                'learning_rates': []
            }
        else:  # classification
            self.history = {
                'train_loss': [],
                'train_accuracy': [],
                'val_loss': [],
                'val_accuracy': [],
                'learning_rates': []
            }
    
    def _get_loss_function(self) -> nn.Module:
        """
        Get loss function based on config and task type.

        For Regression:
        1. **MSE (Mean Squared Error)**:
           - Standard default for regression
           - Heavily penalizes outliers (large errors squared)

        2. **Smooth L1 / Huber**:
           - Robust regression
           - Behave like MSE near zero but like L1 for large errors

        For Classification:
        - **CrossEntropyLoss**: Standard for multi-class classification
        """
        if self.task_type == 'classification':
            # For classification, always use CrossEntropyLoss
            return nn.CrossEntropyLoss()

        # For regression, use configured loss type
        loss_type = self.config.loss.type

        if loss_type == 'mse':
            return nn.MSELoss()
        elif loss_type == 'smooth_l1':
            return nn.SmoothL1Loss()
        elif loss_type == 'huber':
            return nn.HuberLoss(delta=self.config.loss.huber_delta)
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")
    
    def train_epoch(self) -> Dict[str, float]:
        """
        Train for one epoch.

        Returns:
            Dictionary of training metrics
        """
        self.model.train()
        metrics = MetricsTracker(task_type=self.task_type)

        iterator = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch + 1} [Train]")

        for batch_idx, (images, targets, folder_ids) in enumerate(iterator):
            # Move to device (convert to float32 first for MPS compatibility)
            images = images.float().to(self.device)

            # Prepare targets based on task type
            if self.task_type == 'regression':
                targets = targets.float().unsqueeze(1).to(self.device)
            else:  # classification
                targets = targets.long().to(self.device)  # CrossEntropyLoss expects Long

            # Forward pass
            self.optimizer.zero_grad()
            predictions = self.model(images)
            loss = self.criterion(predictions, targets)

            # Backward pass
            loss.backward()

            # Gradient clipping
            if self.config.regularization.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.regularization.grad_clip
                )

            self.optimizer.step()

            # Update metrics
            metrics.update(loss.item(), predictions, targets)

            # Update progress bar
            iterator.set_postfix({'loss': loss.item()})

        return metrics.compute()
    
    def validate(self) -> Dict[str, float]:
        """
        Validate the model.

        Returns:
            Dictionary of validation metrics
        """
        self.model.eval()
        metrics = MetricsTracker(task_type=self.task_type)

        with torch.no_grad():
            iterator = tqdm(self.val_loader, desc=f"Epoch {self.current_epoch + 1} [Val]")

            for images, targets, folder_ids in iterator:
                # Move to device (convert to float32 first for MPS compatibility)
                images = images.float().to(self.device)

                # Prepare targets based on task type
                if self.task_type == 'regression':
                    targets = targets.float().unsqueeze(1).to(self.device)
                else:  # classification
                    targets = targets.long().to(self.device)

                # Forward pass
                predictions = self.model(images)
                loss = self.criterion(predictions, targets)

                # Update metrics
                metrics.update(loss.item(), predictions, targets)

                # Update progress bar
                iterator.set_postfix({'loss': loss.item()})

        return metrics.compute()
    
    def save_checkpoint(self, is_best: bool = False):
        """
        Save model checkpoint.
        
        Only saves when a new best model is found, ensuring efficient storage
        and allowing retraining from the best checkpoint.
        
        Args:
            is_best: Whether this is the best model so far
        """
        if not is_best:
            return
        
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'best_val_epoch': self.best_val_epoch,
            'history': self.history,
            'config': asdict(self.config) if hasattr(self.config, '__dataclass_fields__') else self.config
        }
        
        # Save best checkpoint only
        best_path = self.checkpoint_dir / 'checkpoint_best.pt'
        torch.save(checkpoint, best_path)
        print(f"✓ Saved best model (val_loss: {self.best_val_loss:.6f})")
    
    def load_checkpoint(self, checkpoint_path: str):
        """
        Load model checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint file
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.history = checkpoint.get('history', self.history)
        
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        self.best_val_epoch = checkpoint.get('best_val_epoch', 0)
        
        # Recover best_val_epoch from history if it wasn't saved (older checkpoints)
        if self.best_val_epoch == 0 and self.best_val_loss < float('inf') and len(self.history['val_loss']) > 0:
            try:
                # Assuming val_loss list aligns with epochs 0, 1, 2...
                # We need to be careful if history grows during training and we look for min locally
                # But typically history stores all epochs.
                val_losses = self.history['val_loss']
                self.best_val_epoch = np.argmin(val_losses)
                # Verify consistency
                if abs(val_losses[self.best_val_epoch] - self.best_val_loss) > 1e-6:
                    print(f"Warning: Saved best_val_loss ({self.best_val_loss}) differs from min history loss ({val_losses[self.best_val_epoch]}). Using history min.")
                    self.best_val_loss = val_losses[self.best_val_epoch]
            except Exception as e:
                print(f"Warning: Could not recover best epoch from history: {e}")
        # Try to restore config as dataclass if possible, but keep dict if loading old checkpoints
        loaded_config = checkpoint.get('config', {})
        if isinstance(loaded_config, dict):
            # Attempt to upgrade to TrainingConfig if keys match
            try:
                from .config import OptimizerConfig, LossConfig, SchedulerConfig, RegularizationConfig
                
                # Reconstruct nested dataclasses
                config_dict = {}
                
                # Handle optimizer config
                if 'optimizer' in loaded_config:
                    if isinstance(loaded_config['optimizer'], dict):
                        config_dict['optimizer'] = OptimizerConfig(**loaded_config['optimizer'])
                    else:
                        config_dict['optimizer'] = loaded_config['optimizer']
                
                # Handle loss config
                if 'loss' in loaded_config:
                    if isinstance(loaded_config['loss'], dict):
                        config_dict['loss'] = LossConfig(**loaded_config['loss'])
                    else:
                        config_dict['loss'] = loaded_config['loss']
                
                # Handle scheduler config
                if 'scheduler' in loaded_config:
                    if isinstance(loaded_config['scheduler'], dict):
                        config_dict['scheduler'] = SchedulerConfig(**loaded_config['scheduler'])
                    else:
                        config_dict['scheduler'] = loaded_config['scheduler']
                
                # Handle regularization config
                if 'regularization' in loaded_config:
                    if isinstance(loaded_config['regularization'], dict):
                        config_dict['regularization'] = RegularizationConfig(**loaded_config['regularization'])
                    else:
                        config_dict['regularization'] = loaded_config['regularization']
                
                # Copy scalar fields
                for key in ['batch_size', 'epochs', 'early_stopping', 'checkpoint_dir']:
                    if key in loaded_config:
                        config_dict[key] = loaded_config[key]
                
                self.config = TrainingConfig(**config_dict)
            except Exception as e:
                print(f"Warning: Could not convert loaded config to TrainingConfig: {e}")
                print(f"Using config as-is. This may cause issues if accessing nested attributes.")
                self.config = loaded_config
        else:
            self.config = loaded_config
        
        print(f"✓ Loaded checkpoint from epoch {self.current_epoch}")
    
    def train(self, num_epochs: int):
        """
        Train the model for specified number of epochs.
        
        Args:
            num_epochs: Number of epochs to train
        """
        print(f"\nStarting training for {num_epochs} epochs")
        print("=" * 80)
        
        # Log sample predictions at start if logger available
        if self.logger and hasattr(self, 'log_sample_predictions'):
            try:
                self._log_initial_predictions()
            except Exception as e:
                print(f"Warning: Could not log initial predictions: {e}")
        
        for epoch in range(num_epochs):
            self.current_epoch = epoch
            
            # Train
            train_metrics = self.train_epoch()
            
            # Validate
            val_metrics = self.validate()
            
            # Update scheduler
            self.scheduler.step(val_metrics['loss'])
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Update history
            self.history['train_loss'].append(train_metrics['loss'])
            self.history['val_loss'].append(val_metrics['loss'])
            self.history['learning_rates'].append(current_lr)

            if self.task_type == 'regression':
                self.history['train_mae'].append(train_metrics['mae'])
                self.history['train_r2'].append(train_metrics['r2'])
                self.history['val_mae'].append(val_metrics['mae'])
                self.history['val_r2'].append(val_metrics['r2'])
            else:  # classification
                self.history['train_accuracy'].append(train_metrics['accuracy'])
                self.history['val_accuracy'].append(val_metrics['accuracy'])

            # Log to TensorBoard
            if self.logger:
                self.logger.log_metrics(train_metrics, epoch, prefix='train/')
                self.logger.log_metrics(val_metrics, epoch, prefix='val/')
                self.logger.log_learning_rate(current_lr, epoch)

                # Log sample predictions every 5 epochs
                if epoch % 5 == 0:
                    self._log_sample_predictions(epoch)

            # Check for improvement
            is_best = val_metrics['loss'] < self.best_val_loss
            if is_best:
                self.best_val_loss = val_metrics['loss']
                self.best_val_epoch = self.current_epoch
                self.epochs_without_improvement = 0
            else:
                self.epochs_without_improvement += 1

            # Print epoch summary (task-specific)
            print(f"\nEpoch {epoch + 1}/{num_epochs}")
            if self.task_type == 'regression':
                best_mae = "N/A"
                if self.history['val_mae'] and len(self.history['val_mae']) > self.best_val_epoch:
                    best_mae = f"{self.history['val_mae'][self.best_val_epoch]:.6f}"

                print(f"  Train - Loss: {train_metrics['loss']:.6f}, MAE: {train_metrics['mae']:.6f}, R²: {train_metrics['r2']:.4f}")
                print(f"  Val   - Loss: {val_metrics['loss']:.6f}, MAE: {val_metrics['mae']:.6f}, R²: {val_metrics['r2']:.4f}")
                print(f"  Best  - Loss: {self.best_val_loss:.6f}, MAE: {best_mae} (Epoch {self.best_val_epoch + 1})")
            else:  # classification
                best_acc = "N/A"
                if self.history['val_accuracy'] and len(self.history['val_accuracy']) > self.best_val_epoch:
                    best_acc = f"{self.history['val_accuracy'][self.best_val_epoch]:.4f}"

                print(f"  Train - Loss: {train_metrics['loss']:.6f}, Accuracy: {train_metrics['accuracy']:.4f}")
                print(f"  Val   - Loss: {val_metrics['loss']:.6f}, Accuracy: {val_metrics['accuracy']:.4f}")
                print(f"  Best  - Loss: {self.best_val_loss:.6f}, Accuracy: {best_acc} (Epoch {self.best_val_epoch + 1})")

            print(f"  LR: {current_lr:.2e}")
            
            # Save checkpoint
            self.save_checkpoint(is_best=is_best)
            
            # Early stopping
            if self.config.early_stopping > 0:
                if self.epochs_without_improvement >= self.config.early_stopping:
                    print(f"\nEarly stopping triggered after {epoch + 1} epochs")
                    break
        
        print("\n" + "=" * 80)
        print("Training complete!")
        print(f"Best validation loss: {self.best_val_loss:.6f} (Epoch {self.best_val_epoch + 1})")
        
        # Save training history
        history_path = self.checkpoint_dir / 'training_history.json'
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=2)
        
        # Log final hyperparameters and metrics
        if self.logger:
            if self.task_type == 'regression':
                final_metrics = {
                    'final/best_val_loss': self.best_val_loss,
                    'final/best_val_mae': min(self.history['val_mae']),
                    'final/best_val_r2': max(self.history['val_r2'])
                }
            else:  # classification
                final_metrics = {
                    'final/best_val_loss': self.best_val_loss,
                    'final/best_val_accuracy': max(self.history['val_accuracy'])
                }

            self.logger.log_hyperparameters(
                hparams={'config': asdict(self.config)},
                metrics=final_metrics
            )
            self.logger.close()
    
    def _log_sample_predictions(self, epoch: int):
        """Log sample predictions to TensorBoard."""
        if not self.logger:
            return
        
        self.model.eval()
        with torch.no_grad():
            # Get one batch from validation
            images, targets, folder_ids = next(iter(self.val_loader))
            images = images.float().to(self.device)
            targets = targets.float().unsqueeze(1).to(self.device)
            
            predictions = self.model(images)
            
            self.logger.log_predictions(images, predictions, targets, epoch)
        
        self.model.train()



def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    # Make PyTorch deterministic (slower but reproducible)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False



def get_device(device_config: str = 'auto') -> torch.device:
    """
    Get the device based on configuration.

    Args:
        device_config: Device configuration ('auto', 'cpu', 'cuda', or 'mps')

    Returns:
        torch.device object
    """
    if device_config == 'cpu':
        return torch.device('cpu')
    elif device_config == 'cuda':
        if not torch.cuda.is_available():
            print("Warning: CUDA requested but not available. Using CPU.")
            return torch.device('cpu')
        return torch.device('cuda')
    elif device_config == 'mps':
        if not torch.backends.mps.is_available():
            print("Warning: MPS requested but not available. Using CPU.")
            return torch.device('cpu')
        return torch.device('mps')
    elif device_config == 'auto':
        if torch.cuda.is_available():
            return torch.device('cuda')
        elif torch.backends.mps.is_available():
            return torch.device('mps')
        else:
            return torch.device('cpu')
    else:
        raise ValueError(f"Unknown device config: {device_config}. Choose from: auto, cpu, cuda, mps")



def create_dataloaders(cfg: AppConfig, device: torch.device):
    """
    Create training and validation dataloaders.

    Args:
        cfg: Application configuration
        device: Device to use for training

    Returns:
        Tuple of (train_loader, val_loader, train_dataset, val_dataset)
    """
    print("\nLoading datasets...")

    # Create datasets
    train_dataset = TorchDataset(
        path=cfg.data.paths.root,
        test=cfg.data.splits.test_ids,
        val=cfg.data.splits.val_ids,
        ignore=cfg.data.splits.ignore_ids,
        type='train',
        size=(cfg.data.unroll_on_fly.img_height, cfg.data.unroll_on_fly.img_width),
        padding=cfg.data.unroll_on_fly.padding,
        npoints=cfg.data.unroll_on_fly.npoints,
        boundary_extension=cfg.data.unroll_on_fly.boundary_extension,
        sagittal_folder_prefixes=cfg.data.unroll_on_fly.sagittal_folder_prefixes,
        trunc_width=cfg.data.loading.trunc_width,
        metadata_path=cfg.data.paths.metadata,
        target_ppm=cfg.data.unroll_on_fly.target_ppm,
        data_augment=cfg.data.unroll_on_fly.interpolation_enabled,
        use_unroll_on_fly=cfg.data.loading.use_unroll_on_fly,
        augment_distribution=cfg.data.unroll_on_fly.interpolation_distribution,
        augment_beta_alpha=cfg.data.unroll_on_fly.interpolation_beta_alpha,
        augment_beta_beta=cfg.data.unroll_on_fly.interpolation_beta_beta,
        task_type=cfg.task.type,
        num_classes=cfg.task.classification.num_classes
    )

    val_dataset = TorchDataset(
        path=cfg.data.paths.root,
        test=cfg.data.splits.test_ids,
        val=cfg.data.splits.val_ids,
        ignore=cfg.data.splits.ignore_ids,
        type='val',
        size=(cfg.data.unroll_on_fly.img_height, cfg.data.unroll_on_fly.img_width),
        padding=cfg.data.unroll_on_fly.padding,
        npoints=cfg.data.unroll_on_fly.npoints,
        boundary_extension=cfg.data.unroll_on_fly.boundary_extension,
        sagittal_folder_prefixes=cfg.data.unroll_on_fly.sagittal_folder_prefixes,
        trunc_width=cfg.data.loading.trunc_width,
        metadata_path=cfg.data.paths.metadata,
        target_ppm=cfg.data.unroll_on_fly.target_ppm,
        data_augment=cfg.data.unroll_on_fly.interpolation_enabled,
        use_unroll_on_fly=cfg.data.loading.use_unroll_on_fly,
        augment_distribution=cfg.data.unroll_on_fly.interpolation_distribution,
        augment_beta_alpha=cfg.data.unroll_on_fly.interpolation_beta_alpha,
        augment_beta_beta=cfg.data.unroll_on_fly.interpolation_beta_beta,
        task_type=cfg.task.type,
        num_classes=cfg.task.classification.num_classes
    )

    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(val_dataset)}")

    # Create data loaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=True,
        num_workers=cfg.data.loading.num_workers,
        pin_memory=True if device.type in ['cuda', 'mps'] else False
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=False,
        num_workers=cfg.data.loading.num_workers,
        pin_memory=True if device.type in ['cuda', 'mps'] else False
    )

    return train_loader, val_loader, train_dataset, val_dataset


 
def train_model(cfg: AppConfig, resume_from: str = None, run_name: str = None):
    """
    High-level training function called by main.py.
    
    Args:
        cfg: Application configuration
        resume_from: Optional path to checkpoint to resume training from
        run_name: Optional custom run name (auto-generated if None)
    """
    # Set up run directory
    run_manager = RunManager(
        base_dir=cfg.runs.base_dir,
        auto_name=cfg.runs.auto_name
    )
    
    if resume_from:
        # Resuming from checkpoint - extract run name from path
        checkpoint_path = Path(resume_from)
        if 'runs/' in str(checkpoint_path):
            parts = checkpoint_path.parts
            run_idx = parts.index('runs')
            run_name = parts[run_idx + 1]
            run_dir = run_manager.get_run_dir(run_name)
            run_manager.current_run_dir = run_dir
            run_manager.current_run_name = run_name
            print(f"Resuming run: {run_name}")
        else:
            run_dir = run_manager.create_run(run_name)
    else:
        run_dir = run_manager.create_run(run_name)
    
    # Save config snapshot
    if cfg.runs.save_config:
        run_manager.save_config(cfg)
    
    # Set random seed
    set_seed(cfg.system.seed)

    # Get device
    device = get_device(cfg.system.device)
    print(f"Using device: {device}")
    print(f"Task type: {cfg.task.type}")
    if cfg.task.type == 'classification':
        print(f"Number of classes: {cfg.task.classification.num_classes}")
    print("\nConfiguration loaded from config.yml")

    # Create dataloaders
    train_loader, val_loader, train_dataset, val_dataset = create_dataloaders(cfg, device)

    # Create model
    print("\nCreating model...")
    model = create_staging_model(
        model_type=cfg.model.architecture,
        in_channels=1,
        dropout_rate=cfg.model.dropout,
        task_type=cfg.task.type,
        num_classes=cfg.task.classification.num_classes
    )

    # Print model summary if requested
    if cfg.model.show_summary:
        # Load a sample image to determine actual input dimensions
        print("\nDetermining input dimensions from sample image...")
        
        # Get one sample image from the already-created train_dataset
        sample_image, _, _ = train_dataset[0]
        
        # Extract dimensions: (C, H, W)
        _, img_height, img_width = sample_image.shape
        
        print(f"  Actual input dimensions: {img_height}×{img_width}")
        
        get_model_summary(
            model,
            (cfg.training.batch_size, 1, img_height, img_width)
        )

    # Update training config to use run-specific checkpoint directory
    cfg.training.checkpoint_dir = str(run_manager.get_checkpoint_dir())

    # Initialize TensorBoard logger if enabled
    logger = None
    if cfg.runs.tensorboard_enabled:
        from .tensorboard import TensorBoardLogger
        logger = TensorBoardLogger(
            log_dir=str(run_manager.get_logs_dir()),
            enabled=True
        )

    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=cfg.training,
        device=device,
        task_type=cfg.task.type,
        logger=logger
    )
    
    # Resume from checkpoint if specified
    if resume_from:
        print(f"\nResuming training from checkpoint: {resume_from}")
        trainer.load_checkpoint(resume_from)
    
    # Train
    trainer.train(num_epochs=cfg.training.epochs)
