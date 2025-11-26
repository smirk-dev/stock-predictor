"""
Configuration management for the stock prediction system.
"""

import os
import yaml
from pathlib import Path
from typing import Dict, Any


class Config:
    """Configuration manager for loading and accessing config parameters."""
    
    def __init__(self, config_path: str = None):
        """
        Initialize configuration.
        
        Args:
            config_path: Path to the config.yaml file. If None, uses default location.
        """
        if config_path is None:
            # Get the project root directory
            self.project_root = Path(__file__).parent.parent
            config_path = self.project_root / "config.yaml"
        else:
            config_path = Path(config_path)
            self.project_root = config_path.parent
        
        with open(config_path, 'r') as f:
            self._config = yaml.safe_load(f)
        
        # Create necessary directories
        self._create_directories()
    
    def _create_directories(self):
        """Create necessary directories if they don't exist."""
        dirs = [
            self.get('training.checkpoint_dir'),
            self.get('training.model_save_dir'),
            self.get('visualization.plots_dir'),
            'logs',
            'outputs',
        ]
        
        for dir_path in dirs:
            full_path = self.project_root / dir_path
            full_path.mkdir(parents=True, exist_ok=True)
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get a configuration value using dot notation.
        
        Args:
            key: Configuration key in dot notation (e.g., 'data.train_split')
            default: Default value if key not found
            
        Returns:
            Configuration value
        """
        keys = key.split('.')
        value = self._config
        
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
                if value is None:
                    return default
            else:
                return default
        
        return value
    
    def get_path(self, key: str) -> Path:
        """
        Get a path configuration value as a Path object.
        
        Args:
            key: Configuration key
            
        Returns:
            Path object
        """
        path_str = self.get(key)
        if path_str:
            return self.project_root / path_str
        return None
    
    @property
    def data_config(self) -> Dict:
        """Get data configuration."""
        return self._config.get('data', {})
    
    @property
    def feature_config(self) -> Dict:
        """Get feature engineering configuration."""
        return self._config.get('features', {})
    
    @property
    def model_config(self) -> Dict:
        """Get model configuration."""
        return self._config.get('models', {})
    
    @property
    def training_config(self) -> Dict:
        """Get training configuration."""
        return self._config.get('training', {})
    
    @property
    def evaluation_config(self) -> Dict:
        """Get evaluation configuration."""
        return self._config.get('evaluation', {})
    
    @property
    def visualization_config(self) -> Dict:
        """Get visualization configuration."""
        return self._config.get('visualization', {})


# Global config instance
_config = None


def get_config(config_path: str = None) -> Config:
    """
    Get the global configuration instance.
    
    Args:
        config_path: Optional path to config file
        
    Returns:
        Config instance
    """
    global _config
    if _config is None:
        _config = Config(config_path)
    return _config
