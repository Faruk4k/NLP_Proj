"""
Utility functions for configuration management, logging, and common operations.
Research-quality utilities for the POLAR project.
"""

import os
import yaml
import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional
from datetime import datetime
import numpy as np
import pandas as pd


class ConfigManager:
    """Manages YAML configuration files with validation and defaults."""
    
    def __init__(self, config_path: str):
        """Load and parse YAML configuration."""
        self.config_path = Path(config_path)
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        with open(self.config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        if self.config is None:
            raise ValueError(f"Config file is empty: {config_path}")
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value using dot notation (e.g., 'svm.kernel')."""
        keys = key.split('.')
        value = self.config
        
        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            if default is not None:
                return default
            raise KeyError(f"Configuration key not found: {key}")
    
    def __getitem__(self, key: str):
        """Allow dictionary-style access."""
        return self.get(key)
    
    def to_dict(self) -> Dict:
        """Convert configuration to dictionary."""
        return self.config


def setup_logging(
    log_dir: str = "logs",
    log_file: str = "training.log",
    log_level: str = "INFO",
    timestamp: bool = True
) -> logging.Logger:
    """Setup logging configuration for the experiment."""
    
    #Create log directory
    os.makedirs(log_dir, exist_ok=True)
    
    #Add timestamp to log filename
    if timestamp:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = f"{log_file.rsplit('.', 1)[0]}_{ts}.log"
    
    log_path = os.path.join(log_dir, log_file)
    
    #Configure logging
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler()
        ]
    )
    
    logger = logging.getLogger(__name__)
    logger.info(f"Logging initialized. Log file: {log_path}")
    
    return logger


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    np.random.seed(seed)
    import random
    random.seed(seed)
    
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


def load_csv_data(
    filepath: str,
    text_column: str = "text",
    label_columns: Optional[list] = None,
    id_column: str = "id"
) -> pd.DataFrame:
    """Load CSV data with validation."""
    
    df = pd.read_csv(filepath)
    
    #Validate required columns
    if text_column not in df.columns:
        raise ValueError(f"Text column '{text_column}' not found in {filepath}")
    
    if label_columns:
        for col in label_columns:
            if col not in df.columns:
                raise ValueError(f"Label column '{col}' not found in {filepath}")
    
    #Basic cleaning
    df[text_column] = df[text_column].astype(str).str.strip()
    df = df[df[text_column].str.len() > 0]  #Remove empty texts
    
    return df


def save_results(
    results: Dict[str, Any],
    output_dir: str,
    filename: str = "results",
    formats: list = None
) -> None:
    """Save results to multiple formats (JSON, CSV)."""
    
    if formats is None:
        formats = ["json", "csv"]
    
    os.makedirs(output_dir, exist_ok=True)
    
    #Save JSON
    if "json" in formats:
        json_path = os.path.join(output_dir, f"{filename}.json")
        with open(json_path, 'w') as f:
            json.dump(results, f, indent=4)
    
    #Save CSV (if results is a dataframe)
    if "csv" in formats and isinstance(results, pd.DataFrame):
        csv_path = os.path.join(output_dir, f"{filename}.csv")
        results.to_csv(csv_path, index=False)


def create_experiment_dir(base_dir: str, experiment_name: str, timestamp: bool = True) -> str:
    """Create experiment directory with optional timestamp."""
    
    if timestamp:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        exp_dir = os.path.join(base_dir, f"{experiment_name}_{ts}")
    else:
        exp_dir = os.path.join(base_dir, experiment_name)
    
    os.makedirs(exp_dir, exist_ok=True)
    
    return exp_dir


def get_device_info() -> Dict[str, Any]:
    """Get device information (CPU/GPU availability)."""
    
    try:
        import torch
        info = {
            "cuda_available": torch.cuda.is_available(),
            "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
            "cuda_version": torch.version.cuda,
            "device": "cuda" if torch.cuda.is_available() else "cpu"
        }
        
        # Add extra diagnostics
        if torch.cuda.is_available():
            info["cuda_device_name"] = torch.cuda.get_device_name(0)
            info["cuda_capability"] = torch.cuda.get_device_capability(0)
        
        return info
    except ImportError:
        return {
            "cuda_available": False,
            "cuda_device_count": 0,
            "cuda_version": None,
            "device": "cpu"
        }


def print_device_diagnostics(logger: Optional[logging.Logger] = None) -> None:
    """Print detailed device diagnostics."""
    
    try:
        import torch
        
        msg = (
            f"\n=== DEVICE DIAGNOSTICS ===\n"
            f"PyTorch version: {torch.__version__}\n"
            f"CUDA available: {torch.cuda.is_available()}\n"
        )
        
        if torch.cuda.is_available():
            msg += (
                f"CUDA version: {torch.version.cuda}\n"
                f"Device count: {torch.cuda.device_count()}\n"
                f"Current device: {torch.cuda.current_device()}\n"
                f"Device name: {torch.cuda.get_device_name(0)}\n"
                f"Device capability: {torch.cuda.get_device_capability(0)}\n"
            )
        else:
            msg += "No CUDA-capable GPU detected. Training will use CPU (slower).\n"
        
        msg += "=========================="
        
        if logger:
            logger.info(msg)
        else:
            print(msg)
    except Exception as e:
        if logger:
            logger.warning(f"Error getting device diagnostics: {e}")
        else:
            print(f"Error getting device diagnostics: {e}")


def print_config(config: ConfigManager, logger: Optional[logging.Logger] = None) -> None:
    """Pretty-print configuration."""
    config_dict = config.to_dict()
    config_str = json.dumps(config_dict, indent=2)
    
    if logger:
        logger.info(f"Configuration:\n{config_str}")
    else:
        print(f"Configuration:\n{config_str}")
