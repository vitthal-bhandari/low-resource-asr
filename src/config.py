"""
Configuration management for the low-resource-asr project.

Loads configuration from environment variables and .env file.

Usage:
    from src.config import config
    
    api_key = config.mdc_api_key
    data_dir = config.data_dir
"""

import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

from dotenv import load_dotenv

# Load .env file from project root
PROJECT_ROOT = Path(__file__).parent.parent
load_dotenv(PROJECT_ROOT / ".env")


@dataclass
class Config:
    """Project configuration loaded from environment variables."""
    
    # API Keys
    mdc_api_key: Optional[str] = field(default_factory=lambda: os.getenv("MDC_API_KEY"))
    hf_token: Optional[str] = field(default_factory=lambda: os.getenv("HF_TOKEN"))
    wandb_api_key: Optional[str] = field(default_factory=lambda: os.getenv("WANDB_API_KEY"))
    
    # Paths
    project_root: Path = field(default_factory=lambda: PROJECT_ROOT)
    data_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "data")
    mozilla_data_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "data" / "mozilla_speech_data")
    models_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "models")
    results_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "results")
    
    # Mozilla Data Collective API
    mdc_api_base_url: str = "https://mozilladatacollective.com/api"
    mdc_train_dev_dataset_id: str = "cmfzu8u8wa555eq8onrk334h4"
    mdc_test_dataset_id: str = "cminc35no007no707hql26lzk"
    
    def __post_init__(self):
        """Ensure directories exist."""
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.mozilla_data_dir.mkdir(parents=True, exist_ok=True)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)
    
    def validate(self) -> list[str]:
        """
        Validate configuration and return list of missing required values.
        
        Returns:
            List of missing configuration keys (empty if all required configs present)
        """
        missing = []
        if not self.mdc_api_key:
            missing.append("MDC_API_KEY")
        return missing
    
    def get_mdc_download_url(self, dataset_id: str) -> str:
        """Get the API URL for downloading a dataset."""
        return f"{self.mdc_api_base_url}/datasets/{dataset_id}/download"


# Global config instance
config = Config()
