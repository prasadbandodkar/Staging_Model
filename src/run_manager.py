"""
Run Management System

Handles creation and management of isolated training run directories.
Each run gets its own folder with checkpoints, logs, and config snapshot.
"""

import os
import json
import shutil
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, List
from dataclasses import asdict


class RunManager:
    """
    Manages training run directories and metadata.
    
    Each run creates:
    - runs/<run_name>/checkpoints/  - Model checkpoints
    - runs/<run_name>/logs/         - TensorBoard event files
    - runs/<run_name>/config.yml    - Config snapshot
    - runs/<run_name>/metrics.json  - Final metrics
    """
    
    def __init__(self, base_dir: str = "./runs", auto_name: bool = True):
        """
        Initialize run manager.
        
        Args:
            base_dir: Base directory for all runs
            auto_name: Auto-generate timestamp-based names if True
        """
        self.base_dir = Path(base_dir)
        self.auto_name = auto_name
        self.current_run_dir: Optional[Path] = None
        self.current_run_name: Optional[str] = None
        
    def create_run(self, run_name: Optional[str] = None) -> Path:
        """
        Create a new run directory.
        
        Args:
            run_name: Custom run name, or None to auto-generate
            
        Returns:
            Path to the created run directory
        """
        # Generate run name
        if run_name is None and self.auto_name:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            run_name = f"run_{timestamp}"
        elif run_name is None:
            raise ValueError("run_name required when auto_name is False")
        
        # Create run directory
        run_dir = self.base_dir / run_name
        
        # Check if run already exists
        if run_dir.exists():
            raise ValueError(f"Run '{run_name}' already exists at {run_dir}")
        
        # Create directory structure
        run_dir.mkdir(parents=True, exist_ok=False)
        (run_dir / "checkpoints").mkdir(exist_ok=True)
        (run_dir / "logs").mkdir(exist_ok=True)
        
        self.current_run_dir = run_dir
        self.current_run_name = run_name
        
        print(f"✓ Created run directory: {run_dir}")
        return run_dir
    
    def get_run_dir(self, run_name: Optional[str] = None) -> Path:
        """
        Get the directory for a specific run.
        
        Args:
            run_name: Run name, or None to use current run
            
        Returns:
            Path to run directory
        """
        if run_name is None:
            if self.current_run_dir is None:
                raise ValueError("No active run. Call create_run() first.")
            return self.current_run_dir
        
        run_dir = self.base_dir / run_name
        if not run_dir.exists():
            raise ValueError(f"Run '{run_name}' not found at {run_dir}")
        
        return run_dir
    
    def get_checkpoint_dir(self, run_name: Optional[str] = None) -> Path:
        """Get the checkpoints directory for a run."""
        return self.get_run_dir(run_name) / "checkpoints"
    
    def get_logs_dir(self, run_name: Optional[str] = None) -> Path:
        """Get the TensorBoard logs directory for a run."""
        return self.get_run_dir(run_name) / "logs"
    
    def save_config(self, config, run_name: Optional[str] = None):
        """
        Save configuration snapshot to run directory.
        
        Args:
            config: AppConfig instance
            run_name: Run name, or None to use current run
        """
        run_dir = self.get_run_dir(run_name)
        config_path = run_dir / "config.yml"
        
        # Save config (copy the original file)
        import yaml
        
        # Convert config to dict
        if hasattr(config, '__dict__'):
            config_dict = {}
            for key, value in config.__dict__.items():
                if hasattr(value, '__dict__'):
                    # Convert nested dataclasses
                    config_dict[key] = asdict(value) if hasattr(value, '__dataclass_fields__') else value.__dict__
                else:
                    config_dict[key] = value
        else:
            config_dict = config
        
        with open(config_path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)
        
        print(f"✓ Saved config snapshot to: {config_path}")
    
    def save_metrics(self, metrics: Dict, run_name: Optional[str] = None):
        """
        Save final metrics to run directory.
        
        Args:
            metrics: Dictionary of metrics
            run_name: Run name, or None to use current run
        """
        run_dir = self.get_run_dir(run_name)
        metrics_path = run_dir / "metrics.json"
        
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        print(f"✓ Saved metrics to: {metrics_path}")
    
    def list_runs(self) -> List[str]:
        """
        List all available runs.
        
        Returns:
            List of run names sorted by creation time (newest first)
        """
        if not self.base_dir.exists():
            return []
        
        runs = []
        for item in self.base_dir.iterdir():
            if item.is_dir():
                runs.append((item.name, item.stat().st_ctime))
        
        # Sort by creation time, newest first
        runs.sort(key=lambda x: x[1], reverse=True)
        
        return [name for name, _ in runs]
    
    def get_latest_run(self) -> Optional[str]:
        """Get the name of the most recently created run."""
        runs = self.list_runs()
        return runs[0] if runs else None
    
    def delete_run(self, run_name: str, confirm: bool = False):
        """
        Delete a run directory.
        
        Args:
            run_name: Run name to delete
            confirm: Must be True to actually delete
        """
        if not confirm:
            raise ValueError("Must set confirm=True to delete a run")
        
        run_dir = self.get_run_dir(run_name)
        shutil.rmtree(run_dir)
        print(f"✓ Deleted run: {run_name}")
