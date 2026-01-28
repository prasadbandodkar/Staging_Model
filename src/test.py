"""
Testing module for model evaluation and visualization.

Provides functionality to test trained models on various datasets
and generate comprehensive visualization plots for both regression
and classification tasks.
"""

from pathlib import Path
from typing import List, Optional

import torch
from torch.utils.data import DataLoader, ConcatDataset
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, classification_report

from .config import AppConfig
from .model import create_staging_model
from .torchdataset import TorchDataset
from .train import set_seed, get_device


def test_model(cfg: AppConfig, checkpoint_path: str, test_on: str = 'both', folders: List[int] = None):
    """
    Test the model and create visualization plots.
    
    Supports both regression and classification tasks.
    
    Args:
        cfg: Application configuration
        checkpoint_path: Path to checkpoint to test
        test_on: Which dataset to test on ('test', 'val', 'both', or 'custom')
        folders: List of folder IDs for custom testing (only used when test_on='custom')
    """
    # Set random seed for reproducibility
    set_seed(cfg.system.seed)
    
    # Get device
    device = get_device(cfg.system.device)
    print(f"Using device: {device}")
    print(f"Task type: {cfg.task.type}")
    
    print("\nConfiguration loaded from config.yml")
    print(f"Testing model from checkpoint: {checkpoint_path}")
    
    # Determine which datasets to load based on test_on parameter
    datasets_to_combine = []
    dataset_names = []
    
    if test_on == 'custom':
        # Use custom folder IDs
        print(f"\nLoading custom dataset with folder IDs: {folders}")
        custom_dataset = TorchDataset(
            path=cfg.data.paths.root,
            test=folders,
            val=[],
            ignore=cfg.data.splits.ignore_ids,
            type='test',
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
        datasets_to_combine.append(custom_dataset)
        dataset_names.append(f"Custom (folders: {folders})")
        
    else:
        # Load test and/or val datasets
        if test_on in ['test', 'both']:
            print("\nLoading test dataset...")
            test_dataset = TorchDataset(
                path=cfg.data.paths.root,
                test=cfg.data.splits.test_ids,
                val=cfg.data.splits.val_ids,
                ignore=cfg.data.splits.ignore_ids,
                type='test',
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
            datasets_to_combine.append(test_dataset)
            dataset_names.append(f"Test (folders: {cfg.data.splits.test_ids})")
        
        if test_on in ['val', 'both']:
            print("Loading validation dataset...")
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
            datasets_to_combine.append(val_dataset)
            dataset_names.append(f"Val (folders: {cfg.data.splits.val_ids})")
    
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
        num_workers=cfg.data.loading.num_workers,
        pin_memory=True if device.type == 'cuda' else False
    )
    
    # Create model
    print("\nCreating model...")
    model = create_staging_model(
        model_type=cfg.model.architecture,
        in_channels=1,
        dropout_rate=cfg.model.dropout,
        task_type=cfg.task.type,
        num_classes=cfg.task.classification.num_classes
    )
    model.to(device)
    
    # Load checkpoint
    print(f"Loading checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"✓ Loaded checkpoint from epoch {checkpoint['epoch']}")
    
    # Run testing and collect predictions with folder IDs
    print(f"\nRunning testing (task: {cfg.task.type})...")
    model.eval()
    all_predictions = []
    all_targets = []
    all_folder_ids = []
    
    with torch.no_grad():
        for images, targets, folder_ids in tqdm(test_loader, desc="Testing"):
            images = images.float().to(device)
            
            # Prepare targets based on task type
            if cfg.task.type == 'regression':
                targets = targets.float().unsqueeze(1).to(device)
            else:  # classification
                targets = targets.long().to(device)
            
            predictions = model(images)
            
            all_predictions.append(predictions.cpu().numpy())
            all_targets.append(targets.cpu().numpy())
            all_folder_ids.append(folder_ids.cpu().numpy())
    
    # Concatenate all predictions, targets, and folder IDs
    predictions = np.concatenate(all_predictions)
    targets = np.concatenate(all_targets)
    folder_ids = np.concatenate(all_folder_ids).flatten()
    
    # Task-specific processing
    if cfg.task.type == 'regression':
        predictions = predictions.flatten()
        targets = targets.flatten()
        _test_regression(cfg, predictions, targets, folder_ids, checkpoint_path)
    else:  # classification
        # Convert logits to class predictions
        pred_classes = np.argmax(predictions, axis=1)
        _test_classification(cfg, predictions, pred_classes, targets, folder_ids, checkpoint_path)


def _test_regression(cfg: AppConfig, predictions: np.ndarray, targets: np.ndarray, 
                     folder_ids: np.ndarray, checkpoint_path: str):
    """Run regression-specific testing and visualization."""
    
    # Calculate metrics
    mse = np.mean((predictions - targets) ** 2)
    mae = np.mean(np.abs(predictions - targets))
    ss_res = np.sum((targets - predictions) ** 2)
    ss_tot = np.sum((targets - np.mean(targets)) ** 2)
    r2 = 1 - (ss_res / ss_tot)
    
    # Print results
    print("\n" + "=" * 80)
    print("Regression Test Results:")
    print(f"  Loss: {mse:.6f}")
    print(f"  MAE:  {mae:.6f}")
    print(f"  MSE:  {mse:.6f}")
    print(f"  R²:   {r2:.4f}")
    print("=" * 80)
    
    # Create visualization with folder ID color coding
    print("\nCreating regression visualization...")
    
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
    fig.suptitle(f'Regression Test Results\n{metrics_text}', 
                 fontsize=16, fontweight='bold', y=0.995)
    
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    
    # Save the plot
    checkpoint_dir = Path(checkpoint_path).parent
    plot_path = checkpoint_dir / 'test_results_regression.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Saved regression visualization to: {plot_path}")
    
    plt.close()


def _create_per_folder_confusion_matrices(targets: np.ndarray, pred_classes: np.ndarray,
                                         folder_ids: np.ndarray, unique_folders: np.ndarray,
                                         num_classes: int, class_names: list,
                                         checkpoint_dir: Path):
    """Create separate confusion matrices for each folder ID."""

    num_folders = len(unique_folders)

    # Calculate grid dimensions
    cols = min(3, num_folders)
    rows = (num_folders + cols - 1) // cols

    # Create figure with subplots
    fig, axes = plt.subplots(rows, cols, figsize=(6*cols, 5*rows))
    if rows == 1 and cols == 1:
        axes = np.array([[axes]])
    elif rows == 1 or cols == 1:
        axes = axes.reshape(rows, cols)

    for idx, fid in enumerate(unique_folders):
        row = idx // cols
        col = idx % cols
        ax = axes[row, col]

        # Get data for this folder
        mask = folder_ids == fid
        folder_targets = targets[mask]
        folder_preds = pred_classes[mask]

        # Calculate confusion matrix for this folder
        cm_folder = np.zeros((num_classes, num_classes), dtype=np.int64)
        for t, p in zip(folder_targets, folder_preds):
            if 0 <= t < num_classes and 0 <= p < num_classes:
                cm_folder[t, p] += 1

        # Calculate accuracy
        folder_acc = np.mean(folder_preds == folder_targets)

        # Plot confusion matrix
        im = ax.imshow(cm_folder, interpolation='nearest', cmap='Blues')
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        # Set ticks and labels
        ax.set(xticks=np.arange(num_classes),
               yticks=np.arange(num_classes),
               xticklabels=class_names,
               yticklabels=class_names,
               ylabel='True Label',
               xlabel='Predicted Label')
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor", fontsize=9)
        plt.setp(ax.get_yticklabels(), fontsize=9)

        # Annotate confusion matrix
        thresh = cm_folder.max() / 2. if cm_folder.max() > 0 else 0.5
        for i in range(cm_folder.shape[0]):
            for j in range(cm_folder.shape[1]):
                if cm_folder[i, j] > 0:  # Only annotate non-zero entries
                    ax.text(j, i, format(cm_folder[i, j], 'd'),
                           ha="center", va="center", fontsize=8,
                           color="white" if cm_folder[i, j] > thresh else "black")

        # Set title with folder ID and accuracy
        ax.set_title(f'Folder {int(fid)}\nAcc: {folder_acc:.2%} (N={mask.sum()})',
                    fontsize=12, fontweight='bold')

    # Hide unused subplots
    for idx in range(num_folders, rows * cols):
        row = idx // cols
        col = idx % cols
        axes[row, col].axis('off')

    fig.suptitle('Per-Folder Confusion Matrices', fontsize=16, fontweight='bold')
    plt.tight_layout()

    # Save the plot
    plot_path = checkpoint_dir / 'test_results_per_folder_confusion_matrices.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved per-folder confusion matrices to: {plot_path}")

    plt.close()


def _test_classification(cfg: AppConfig, predictions: np.ndarray, pred_classes: np.ndarray,
                        targets: np.ndarray, folder_ids: np.ndarray, checkpoint_path: str):
    """Run classification-specific testing and visualization."""
    
    # Calculate metrics
    num_classes = cfg.task.classification.num_classes
    class_names = cfg.task.classification.class_names or [f"Class {i}" for i in range(num_classes)]
    
    # Ensure targets and predictions are proper integer arrays
    targets = targets.flatten().astype(np.int64)
    pred_classes = pred_classes.flatten().astype(np.int64)
    
    accuracy = np.mean(pred_classes == targets)
    
    # Get confusion matrix (manual computation to ensure proper dtype)
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(targets, pred_classes):
        if 0 <= t < num_classes and 0 <= p < num_classes:
            cm[t, p] += 1
    
    # Classification report
    report = classification_report(targets, pred_classes, 
                                   target_names=class_names, 
                                   labels=range(num_classes),
                                   zero_division=0)
    
    # Per-class accuracy
    per_class_acc = cm.diagonal() / cm.sum(axis=1).clip(min=1)
    
    # Get unique folder IDs
    unique_folders = np.unique(folder_ids)

    # Print results
    print("\n" + "=" * 80)
    print("Classification Test Results:")
    print(f"  Overall Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")

    print("\nPer-Folder Accuracy:")
    for fid in unique_folders:
        mask = folder_ids == fid
        if mask.sum() > 0:
            folder_acc = np.mean(pred_classes[mask] == targets[mask])
            print(f"  Folder {int(fid)}: {folder_acc:.4f} ({folder_acc*100:.2f}%) [N={mask.sum()}]")

    print("\nPer-Class Accuracy:")
    for i, (name, acc) in enumerate(zip(class_names, per_class_acc)):
        print(f"  {name}: {acc:.4f} ({acc*100:.2f}%)")
    print("\nClassification Report:")
    print(report)
    print("=" * 80)
    
    # Create visualization
    print("\nCreating classification visualization...")

    # Assign colors to folders
    num_folders = len(unique_folders)

    import matplotlib.cm as mpl_cm
    colormap = mpl_cm.get_cmap('tab10' if num_folders <= 10 else 'tab20')
    folder_colors = {fid: colormap(i / max(num_folders - 1, 1)) for i, fid in enumerate(unique_folders)}
    
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
    
    # Plot 1: Confusion Matrix
    ax1 = fig.add_subplot(gs[0, 0])
    im = ax1.imshow(cm, interpolation='nearest', cmap='Blues')
    ax1.figure.colorbar(im, ax=ax1)
    ax1.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=class_names,
           yticklabels=class_names,
           ylabel='True Label',
           xlabel='Predicted Label')
    plt.setp(ax1.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    # Annotate confusion matrix
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax1.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    ax1.set_title('Confusion Matrix', fontsize=14, fontweight='bold')
    
    # Plot 2: Normalized Confusion Matrix
    ax2 = fig.add_subplot(gs[0, 1])
    cm_normalized = cm.astype('float') / cm.sum(axis=1, keepdims=True).clip(min=1)
    im2 = ax2.imshow(cm_normalized, interpolation='nearest', cmap='Blues', vmin=0, vmax=1)
    ax2.figure.colorbar(im2, ax=ax2)
    ax2.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=class_names,
           yticklabels=class_names,
           ylabel='True Label',
           xlabel='Predicted Label')
    plt.setp(ax2.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    # Annotate normalized confusion matrix
    for i in range(cm_normalized.shape[0]):
        for j in range(cm_normalized.shape[1]):
            ax2.text(j, i, f"{cm_normalized[i, j]:.2f}",
                    ha="center", va="center",
                    color="white" if cm_normalized[i, j] > 0.5 else "black")
    ax2.set_title('Normalized Confusion Matrix', fontsize=14, fontweight='bold')
    
    # Plot 3: Per-Class Accuracy
    ax3 = fig.add_subplot(gs[0, 2])
    bars = ax3.bar(range(num_classes), per_class_acc, color='steelblue', edgecolor='black', linewidth=1.2)
    ax3.set_xticks(range(num_classes))
    ax3.set_xticklabels(class_names, rotation=45, ha="right")
    ax3.set_ylabel('Accuracy', fontsize=12)
    ax3.set_xlabel('Class', fontsize=12)
    ax3.set_title('Per-Class Accuracy', fontsize=14, fontweight='bold')
    ax3.set_ylim([0, 1.0])
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for i, (bar, acc) in enumerate(zip(bars, per_class_acc)):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{acc:.2%}',
                ha='center', va='bottom', fontsize=10)
    
    # Plot 4: Class Distribution (Ground Truth vs Predictions)
    ax4 = fig.add_subplot(gs[1, 0])
    x = np.arange(num_classes)
    width = 0.35
    
    true_counts = np.bincount(targets, minlength=num_classes)
    pred_counts = np.bincount(pred_classes, minlength=num_classes)

    ax4.bar(x - width/2, true_counts, width, label='Ground Truth',
            color='steelblue', edgecolor='black', linewidth=1.2)
    ax4.bar(x + width/2, pred_counts, width, label='Predictions',
            color='coral', edgecolor='black', linewidth=1.2)
    
    ax4.set_xlabel('Class', fontsize=12)
    ax4.set_ylabel('Count', fontsize=12)
    ax4.set_title('Class Distribution', fontsize=14, fontweight='bold')
    ax4.set_xticks(x)
    ax4.set_xticklabels(class_names, rotation=45, ha="right")
    ax4.legend()
    ax4.grid(True, alpha=0.3, axis='y')
    
    # Plot 5: Prediction Confidence Distribution
    ax5 = fig.add_subplot(gs[1, 1])
    # Get max probability for each prediction (confidence)
    confidences = np.max(predictions, axis=1)
    correct_mask = pred_classes == targets
    
    ax5.hist([confidences[correct_mask], confidences[~correct_mask]], 
             bins=30, label=['Correct', 'Incorrect'],
             color=['green', 'red'], alpha=0.6, edgecolor='black')
    ax5.set_xlabel('Confidence (Max Probability)', fontsize=12)
    ax5.set_ylabel('Frequency', fontsize=12)
    ax5.set_title('Prediction Confidence Distribution', fontsize=14, fontweight='bold')
    ax5.legend()
    ax5.grid(True, alpha=0.3, axis='y')
    
    # Plot 6: Folder-wise Accuracy
    ax6 = fig.add_subplot(gs[1, 2])
    folder_accuracies = []
    for fid in unique_folders:
        mask = folder_ids == fid
        if mask.sum() > 0:
            folder_acc = np.mean(pred_classes[mask] == targets[mask])
            folder_accuracies.append(folder_acc)
        else:
            folder_accuracies.append(0)
    
    bars = ax6.bar(range(len(unique_folders)), folder_accuracies, 
                   color=[folder_colors[fid] for fid in unique_folders],
                   edgecolor='black', linewidth=1.2)
    ax6.set_xticks(range(len(unique_folders)))
    ax6.set_xticklabels([f"Folder {int(fid)}" for fid in unique_folders], 
                        rotation=45, ha="right")
    ax6.set_ylabel('Accuracy', fontsize=12)
    ax6.set_xlabel('Folder ID', fontsize=12)
    ax6.set_title('Folder-wise Accuracy', fontsize=14, fontweight='bold')
    ax6.set_ylim([0, 1.0])
    ax6.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, acc in zip(bars, folder_accuracies):
        height = bar.get_height()
        ax6.text(bar.get_x() + bar.get_width()/2., height,
                f'{acc:.2%}',
                ha='center', va='bottom', fontsize=9)
    
    # Add overall metrics
    metrics_text = f'Overall Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)'
    fig.suptitle(f'Classification Test Results\n{metrics_text}', 
                 fontsize=16, fontweight='bold', y=0.995)
    
    # Save the plot
    checkpoint_dir = Path(checkpoint_path).parent
    plot_path = checkpoint_dir / 'test_results_classification.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Saved classification visualization to: {plot_path}")

    plt.close()

    # Create per-folder confusion matrices
    print("\nCreating per-folder confusion matrices...")
    _create_per_folder_confusion_matrices(
        targets, pred_classes, folder_ids, unique_folders,
        num_classes, class_names, checkpoint_dir
    )
