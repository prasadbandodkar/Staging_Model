#!/usr/bin/env python3
"""
Main entry point for model training and testing.

This script delegates to src.train and src.test modules.
All configuration is loaded from config.yml.

Usage:
    # Training
    python main.py --mode train
    python main.py --mode train --run-name my_experiment
    python main.py --mode train --resume runs/my_experiment/checkpoints/checkpoint_best.pt
    
    # Testing
    python main.py --mode test --checkpoint runs/my_experiment/checkpoints/checkpoint_best.pt
    python main.py --mode test --checkpoint checkpoint.pt --test-on val
    python main.py --mode test --checkpoint checkpoint.pt --test-on custom --folders 6 7 21
    
    # TensorBoard
    python main.py --mode tensorboard --run-name my_experiment
"""

import argparse
import subprocess
import sys
from pathlib import Path

from src.config import AppConfig
from src.run_manager import RunManager
from src.train import train_model
from src.test import test_model



def launch_tensorboard(cfg: AppConfig, run_name: str = None):
    """Launch TensorBoard server for a run."""
    run_manager = RunManager(base_dir=cfg.runs.base_dir)
    
    if run_name is None:
        latest_run = run_manager.get_latest_run()
        if latest_run:
            run_name = latest_run
            print(f"Using latest run: {latest_run}")
        else:
            print("Error: No runs found. Please specify --run-name")
            sys.exit(1)
    
    try:
        log_dir = run_manager.get_logs_dir(run_name)
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)
    
    print(f"\nLaunching TensorBoard for run: {run_name}")
    print(f"Log directory: {log_dir}")
    print("\nTensorBoard will be available at: http://localhost:6006")
    print("Press Ctrl+C to stop\n")
    
    try:
        subprocess.run(['tensorboard', '--logdir', str(log_dir)], check=True)
    except KeyboardInterrupt:
        print("\nTensorBoard stopped")
    except FileNotFoundError:
        print("\nError: TensorBoard not found. Install with: pip install tensorboard")
        sys.exit(1)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Training and testing for the staging regression model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
            Examples:
            # Train from scratch
            python main.py --mode train
            
            # Train with custom run name
            python main.py --mode train --run-name my_experiment
            
            # Resume training
            python main.py --mode train --resume runs/my_experiment/checkpoints/checkpoint_best.pt
            
            # Test on both test and validation sets
            python main.py --mode test --checkpoint runs/my_experiment/checkpoints/checkpoint_best.pt
            
            # Test on specific dataset
            python main.py --mode test --checkpoint checkpoint.pt --test-on test
            
            # Test on custom folders
            python main.py --mode test --checkpoint checkpoint.pt --test-on custom --folders 6 21 34
            
            # Launch TensorBoard
            python main.py --mode tensorboard --run-name my_experiment
        """
    )
    
    parser.add_argument(
        '--mode',
        type=str,
        choices=['train', 'test', 'tensorboard'],
        default='train',
        help='Mode to run: train, test, or tensorboard (default: train)'
    )
    
    # Training arguments
    parser.add_argument('--resume', type=str, help='Path to checkpoint to resume training from')
    parser.add_argument('--run-name', type=str, help='Custom name for training run')
    
    # Testing arguments
    parser.add_argument('--checkpoint', type=str, help='Path to checkpoint to test')
    parser.add_argument(
        '--test-on',
        type=str,
        choices=['test', 'val', 'both', 'custom'],
        default='both',
        help='Dataset to test on (default: both)'
    )
    parser.add_argument(
        '--folders',
        type=int,
        nargs='+',
        help='Custom folder IDs to test on (use with --test-on=custom)'
    )
    
    # Configuration
    parser.add_argument('--config', type=str, default='config.yml', help='Path to config file')
    
    args = parser.parse_args()
    
    # Load configuration
    try:
        cfg = AppConfig.load(args.config)
    except Exception as e:
        print(f"Error loading configuration from '{args.config}': {e}")
        print("Please ensure the config file exists and is valid.")
        sys.exit(1)
    
    # Delegate to appropriate module
    if args.mode == 'train':
        train_model(cfg, resume_from=args.resume, run_name=args.run_name)
    
    elif args.mode == 'test':
        if args.checkpoint is None:
            print("Error: --checkpoint is required in test mode")
            sys.exit(1)
        
        if args.test_on == 'custom' and args.folders is None:
            print("Error: --folders is required when --test-on=custom")
            sys.exit(1)
        
        if not Path(args.checkpoint).exists():
            print(f"Error: Checkpoint not found: {args.checkpoint}")
            sys.exit(1)
        
        test_model(cfg, args.checkpoint, test_on=args.test_on, folders=args.folders)
    
    elif args.mode == 'tensorboard':
        launch_tensorboard(cfg, run_name=args.run_name)


if __name__ == "__main__":
    main()
