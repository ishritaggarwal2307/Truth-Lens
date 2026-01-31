"""
Configuration Loader for Truth-Lens
====================================
Loads and validates YAML configuration files.
"""

import os
import yaml
from typing import Dict, Any
from pathlib import Path


class Config:
    """Configuration management class."""
    
    def __init__(self, config_path: str = "configs/config.yaml"):
        """
        Initialize configuration.
        
        Args:
            config_path: Path to YAML configuration file
        """
        self.config_path = config_path
        self.config = self._load_config()
        self._create_directories()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
        
        with open(self.config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        return config
    
    def _create_directories(self):
        """Create necessary directories from config."""
        dirs_to_create = [
            self.config['paths']['models_dir'],
            self.config['paths']['logs_dir'],
            self.config['paths']['cache_dir'],
            self.config['paths']['temp_dir'],
            self.config['dataset']['paths']['raw_data'],
            self.config['dataset']['paths']['processed_data'],
        ]
        
        for directory in dirs_to_create:
            Path(directory).mkdir(parents=True, exist_ok=True)
    
    def get(self, key_path: str, default: Any = None) -> Any:
        """
        Get configuration value using dot notation.
        
        Args:
            key_path: Dot-separated path (e.g., 'model.features.mfcc.n_mfcc')
            default: Default value if key not found
            
        Returns:
            Configuration value
            
        Example:
            >>> config.get('model.features.mfcc.n_mfcc')
            40
        """
        keys = key_path.split('.')
        value = self.config
        
        try:
            for key in keys:
                value = value[key]
            return value
        except (KeyError, TypeError):
            return default
    
    def update(self, key_path: str, value: Any):
        """
        Update configuration value.
        
        Args:
            key_path: Dot-separated path
            value: New value
        """
        keys = key_path.split('.')
        config = self.config
        
        for key in keys[:-1]:
            config = config.setdefault(key, {})
        
        config[keys[-1]] = value
    
    def save(self, output_path: str = None):
        """
        Save configuration to file.
        
        Args:
            output_path: Output path (default: original config_path)
        """
        output_path = output_path or self.config_path
        
        with open(output_path, 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False)
    
    def __repr__(self) -> str:
        return f"Config(config_path='{self.config_path}')"


# Global config instance
_config_instance = None


def get_config(config_path: str = "configs/config.yaml") -> Config:
    """
    Get global configuration instance (Singleton pattern).
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Config instance
    """
    global _config_instance
    
    if _config_instance is None:
        _config_instance = Config(config_path)
    
    return _config_instance


if __name__ == "__main__":
    # Test configuration loading
    config = get_config()
    print(f"Model name: {config.get('model.name')}")
    print(f"Sample rate: {config.get('audio.sample_rate')}")
    print(f"Batch size: {config.get('training.batch_size')}")
