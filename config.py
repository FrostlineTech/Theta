from pathlib import Path

# Base project directory
PROJECT_DIR = Path(__file__).parent

# Model paths
MODEL_DIR = PROJECT_DIR / "models"
MODEL_FILE = MODEL_DIR / "final" / "final_model"  # Main model location - matches train_all_datasets.py
CHECKPOINTS_DIR = MODEL_DIR / "checkpoints"  # For training checkpoints
BACKUP_DIR = MODEL_DIR / "backups"  # For model backups

# Ensure directories exist
MODEL_DIR.mkdir(parents=True, exist_ok=True)
MODEL_FILE.mkdir(parents=True, exist_ok=True)
CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
BACKUP_DIR.mkdir(parents=True, exist_ok=True)
