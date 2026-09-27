# src/config.py
import os
import yaml
import numpy as np
import random
from pathlib import Path

def load_config(config_path="configs/config.yaml"):
    """Loads YAML configuration file."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config

def set_seed(seed):
    """Ensures deterministic behavior across libraries."""
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

def setup_environment(config):
    """Creates necessary directories and sets seeds."""
    set_seed(config['project']['seed'])
    
    # Create required directories
    dirs_to_create = [
        config['data']['output_dir'],
        config['data']['model_dir'],
        "reports"
    ]
    for d in dirs_to_create:
        Path(d).mkdir(parents=True, exist_ok=True)
        
    return config