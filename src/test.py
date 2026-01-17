"""
Testing module for model evaluation and visualization.

Provides functionality to test trained models on various datasets
and generate comprehensive visualization plots.
"""

from pathlib import Path
from typing import List, Optional

import torch
from torch.utils.data import DataLoader, ConcatDataset
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from .config import AppConfig
from .model import create_staging_model
from .data import TorchDataset
from .train import set_seed, get_device


def test_model(cfg: AppConfig, checkpoint_path: str, test_on: str = 'both', folders: List[int] = None):
    """
    Test the model and create visualization plots.
    
    Args:
        cfg: Application configuration
        checkpoint_path: Path to checkpoint to test
        test_on: Which dataset to test on ('test', 'val', 'both', or 'custom')
        folders: List of folder IDs for custom testing (only used when test_on='custom')
    """
    # Set random seed for reproducibility
    set_seed(cfg.seed)
    
    # Get device
    device = get_device(cfg.cpu)
    print(f"Using device: {device}")
    
    print("\nConfiguration loaded from config.yml")
    print(f"Testing model from checkpoint: {checkpoint_path}")
    
    # Determine which datasets to load based on test_on parameter
    datasets_to_combine = []
    dataset_names = []
    
    if test_on == 'custom':
        # Use custom folder IDs
        print(f"\nLoading custom dataset with folder IDs: {folders}")
        custom_dataset = TorchDataset(
            path=cfg.data.path,
            test=folders,
            val=[],
            type='test',
            size=(cfg.data.img_height, cfg.data.img_width),
            padding=cfg.data.padding,
            npoints=cfg.data.npoints,
            boundary_extension=cfg.data.boundary_extension,
            sagittal_folder_prefixes=cfg.data.sagittal_folder_prefixes,
            trunc_width=cfg.data.trunc_width,
            image_type=cfg.data.image_type
        )
        datasets_to_combine.append(custom_dataset)
        dataset_names.append(f"Custom (folders: {folders})")
        
    else:
        # Load test and/or val datasets
        if test_on in ['test', 'both']:
            print("\nLoading test dataset...")
            test_dataset = TorchDataset(
                path=cfg.data.path,
                test=cfg.data.test_ids,
                val=cfg.data.val_ids,
                type='test',
                size=(cfg.data.img_height, cfg.data.img_width),
                padding=cfg.data.padding,
                npoints=cfg.data.npoints,
                boundary_extension=cfg.data.boundary_extension,
                sagittal_folder_prefixes=cfg.data.sagittal_folder_prefixes,
                trunc_width=cfg.data.trunc_width,
                image_type=cfg.data.image_type
            )
            datasets_to_combine.append(test_dataset)
            dataset_names.append(f"Test (folders: {cfg.data.test_ids})")
        
        if test_on in ['val', 'both']:
            print("Loading validation dataset...")
            val_dataset = TorchDataset(
                path=cfg.data.path,
                test=cfg.data.test_ids,
                val=cfg.data.val_ids,
                type='val',
                size=(cfg.data.img_height, cfg.data.img_width),
                padding=cfg.data.padding,
                npoints=cfg.data.npoints,
                boundary_extension=cfg.data.boundary_extension,
                sagittal_folder_prefixes=cfg.data.sagittal_folder_prefixes,
                trunc_width=cfg.data.trunc_width,
                image_type=cfg.data.image_type
            )
            datasets_to_combine.append(val_dataset)
            dataset_names.append(f"Val (folders: {cfg.data.val_ids})")
    
    # Combine datasets if necessary
    if len(datasets_to_combine) == 1:
        combined_dataset = datasets_to_combine[0]
        print(f"Dataset size: {len(combined_dataset)}")
    else:
        combined_dataset = ConcatDataset(datasets_to_combine)
        for ds, name in zip(datasets_to_combine, dataset_names):
            print(f"{name} size: {len(ds)}")
        print(f"Combined dataset size: {len(combined_dataset)}")
    
    # Create dataloader
    test_loader = DataLoader(
        combined_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=False,
        num_workers=cfg.data.num_workers,
        pin_memory=True if device.type == 'cuda' else False
    )
    
    # Create model
    print("\nCreating model...")
    model = create_staging_model(
        model_type=cfg.model.model_type,
        in_channels=1,
        dropout_rate=cfg.model.dropout
    )
    model.to(device)
    
    # Load checkpoint
    print(f"Loading checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"✓ Loaded checkpoint from epoch {checkpoint['epoch']}")
    
    # Run testing and collect predictions with folder IDs
    print("\nRunning testing...")
    model.eval()
    all_predictions = []
    all_targets = []
    all_folder_ids = []
    
    with torch.no_grad():
        for images, targets, folder_ids in tqdm(test_loader, desc="Testing"):
            images = images.float().to(device)
            targets = targets.float().unsqueeze(1).to(device)
            
            predictions = model(images)
            
            all_predictions.append(predictions.cpu().numpy())
            all_targets.append(targets.cpu().numpy())
            all_folder_ids.append(folder_ids.cpu().numpy())
    
    # Flatten all predictions, targets, and folder IDs
    predictions = np.concatenate(all_predictions).flatten()
    targets = np.concatenate(all_targets).flatten()
    folder_ids = np.concatenate(all_folder_ids).flatten()
    
    # Calculate metrics
    mse = np.mean((predictions - targets) ** 2)
    mae = np.mean(np.abs(predictions - targets))
    ss_res = np.sum((targets - predictions) ** 2)
    ss_tot = np.sum((targets - np.mean(targets)) ** 2)
    r2 = 1 - (ss_res / ss_tot)
    
    # Print results
    print("\n" + "=" * 80)
    print("Test Results:")
    print(f"  Loss: {mse:.6f}")
    print(f"  MAE:  {mae:.6f}")
    print(f"  MSE:  {mse:.6f}")
    print(f"  R²:   {r2:.4f}")
    print("=" * 80)
    
    # Create visualization with folder ID color coding
    print("\nCreating visualization...")
    
    # Get unique folder IDs and assign colors
    unique_folders = np.unique(folder_ids)
    num_folders = len(unique_folders)
    
    # Use a colormap with distinct colors
    import matplotlib.cm as cm
    colormap = cm.get_cmap('tab10' if num_folders <= 10 else 'tab20')
    folder_colors = {fid: colormap(i / max(num_folders - 1, 1)) for i, fid in enumerate(unique_folders)}
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Predictions vs Targets by image number (color-coded by folder)
    image_numbers = np.arange(len(predictions))
    ax1 = axes[0, 0]
    for fid in unique_folders:
        mask = folder_ids == fid
        ax1.scatter(image_numbers[mask], targets[mask], alpha=0.6, s=30, 
                   label=f'Folder {int(fid)} (GT)', color=folder_colors[fid], marker='o')
        ax1.scatter(image_numbers[mask], predictions[mask], alpha=0.6, s=30, 
                   label=f'Folder {int(fid)} (Pred)', color=folder_colors[fid], marker='x')
    ax1.set_xlabel('Image Number', fontsize=12)
    ax1.set_ylabel('Staging Value', fontsize=12)
    ax1.set_title('Predictions vs Ground Truth by Image\n(Color-coded by Folder ID)', 
                 fontsize=14, fontweight='bold')
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Scatter plot - Predictions vs Targets (color-coded by folder)
    ax2 = axes[0, 1]
    
    # Define distinct markers for each folder
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h']
    
    for i, fid in enumerate(unique_folders):
        mask = folder_ids == fid
        marker = markers[i % len(markers)]
        ax2.scatter(targets[mask], predictions[mask], alpha=0.7, s=50, 
                   label=f'Folder {int(fid)}', color=folder_colors[fid],
                   marker=marker, edgecolors='black', linewidth=0.5)
    
    min_val = min(targets.min(), predictions.min())
    max_val = max(targets.max(), predictions.max())
    ax2.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect Prediction', linewidth=2)
    ax2.set_xlabel('Ground Truth', fontsize=12)
    ax2.set_ylabel('Predictions', fontsize=12)
    ax2.set_title('Prediction vs Ground Truth Scatter\n(Color-coded by Folder ID)', 
                 fontsize=14, fontweight='bold')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.set_aspect('equal', adjustable='box')
    
    # Plot 3: Residuals by image number (color-coded by folder)
    ax3 = axes[1, 0]
    residuals = predictions - targets
    for fid in unique_folders:
        mask = folder_ids == fid
        ax3.scatter(image_numbers[mask], residuals[mask], alpha=0.6, s=30, 
                   label=f'Folder {int(fid)}', color=folder_colors[fid])
    ax3.axhline(y=0, color='r', linestyle='--', linewidth=2)
    ax3.set_xlabel('Image Number', fontsize=12)
    ax3.set_ylabel('Residual (Prediction - Target)', fontsize=12)
    ax3.set_title('Prediction Residuals by Image\n(Color-coded by Folder ID)', 
                 fontsize=14, fontweight='bold')
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Histogram of residuals (side-by-side bars by folder)
    ax4 = axes[1, 1]
    
    # Determine bin edges for consistency across all folders
    all_residuals_min = residuals.min()
    all_residuals_max = residuals.max()
    bins = np.linspace(all_residuals_min, all_residuals_max, 31)
    
    # Calculate bar width and positions
    bar_width = (bins[1] - bins[0]) / (num_folders + 0.5)
    
    for i, fid in enumerate(unique_folders):
        mask = folder_ids == fid
        folder_residuals = residuals[mask]
        
        # Calculate histogram
        counts, _ = np.histogram(folder_residuals, bins=bins)
        
        # Offset bars based on folder index
        bin_centers = (bins[:-1] + bins[1:]) / 2
        offset = (i - num_folders/2 + 0.5) * bar_width
        
        # Plot bars
        ax4.bar(bin_centers + offset, counts, width=bar_width, 
               label=f'Folder {int(fid)}', color=folder_colors[fid], 
               alpha=0.7, edgecolor='black', linewidth=0.5)
    
    ax4.axvline(x=0, color='r', linestyle='--', linewidth=2, label='Zero Error')
    ax4.set_xlabel('Residual (Prediction - Target)', fontsize=12)
    ax4.set_ylabel('Frequency', fontsize=12)
    ax4.set_title('Distribution of Residuals\n(Side-by-side by Folder ID)', 
                 fontsize=14, fontweight='bold')
    ax4.legend(fontsize=9)
    ax4.grid(True, alpha=0.3, axis='y')
    
    # Add overall metrics
    metrics_text = f'MAE: {mae:.4f} | MSE: {mse:.4f} | R²: {r2:.4f}'
    fig.suptitle(f'Model Test Results\n{metrics_text}', 
                 fontsize=16, fontweight='bold', y=0.995)
    
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    
    # Save the plot
    checkpoint_dir = Path(checkpoint_path).parent
    plot_path = checkpoint_dir / 'test_results.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Saved visualization to: {plot_path}")
    
    plt.close()
