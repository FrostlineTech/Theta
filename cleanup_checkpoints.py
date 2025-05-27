#!/usr/bin/env python
"""
Theta AI - Checkpoint Cleanup Script

This script helps save disk space by removing older model checkpoints
while preserving the most recent ones and the final model.
"""

import os
import shutil
import argparse
import logging
import glob
import re
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Default paths
MODEL_DIR = "models"
FINAL_MODEL_PATH = os.path.join(MODEL_DIR, "final", "final_model")
CHECKPOINTS_DIR = "theta_all_datasets_model"
BACKUPS_DIR = os.path.join(MODEL_DIR, "backups")

def get_checkpoint_epochs(checkpoint_dir):
    """Get a list of all checkpoint epochs in the directory."""
    checkpoint_dirs = glob.glob(os.path.join(checkpoint_dir, "checkpoint-epoch-*"))
    epochs = []
    
    for dir_path in checkpoint_dirs:
        match = re.search(r'checkpoint-epoch-(\d+)$', dir_path)
        if match:
            epoch = int(match.group(1))
            epochs.append((epoch, dir_path))
    
    # Sort by epoch number (ascending)
    epochs.sort()
    return epochs

def get_backup_timestamps(backups_dir):
    """Get a list of backup directories with their timestamps."""
    backup_dirs = glob.glob(os.path.join(backups_dir, "theta_backup_*"))
    backups = []
    
    for dir_path in backup_dirs:
        match = re.search(r'theta_backup_(\d{8}_\d{6})$', dir_path)
        if match:
            timestamp_str = match.group(1)
            try:
                timestamp = datetime.strptime(timestamp_str, "%Y%m%d_%H%M%S")
                backups.append((timestamp, dir_path))
            except ValueError:
                logger.warning(f"Couldn't parse timestamp from {dir_path}")
    
    # Sort by timestamp (oldest first)
    backups.sort()
    return backups

def cleanup_checkpoints(args):
    """Clean up old checkpoint directories."""
    # Check if checkpoints directory exists
    if not os.path.exists(args.checkpoints_dir):
        logger.warning(f"Checkpoints directory '{args.checkpoints_dir}' does not exist. Nothing to clean up.")
        return
    
    # Get all checkpoint epochs
    epochs = get_checkpoint_epochs(args.checkpoints_dir)
    
    if not epochs:
        logger.info(f"No checkpoints found in '{args.checkpoints_dir}'.")
        return
    
    # Determine which checkpoints to keep
    total_checkpoints = len(epochs)
    keep_count = min(args.keep, total_checkpoints)
    
    if keep_count == total_checkpoints:
        logger.info(f"Only {total_checkpoints} checkpoints exist, which is <= {args.keep}. Nothing to delete.")
        return
    
    # Keep the most recent checkpoints
    to_keep = epochs[-keep_count:]
    to_delete = epochs[:-keep_count]
    
    # Delete older checkpoints
    bytes_saved = 0
    for epoch, dir_path in to_delete:
        if os.path.exists(dir_path):
            # Calculate size before deleting
            dir_size = sum(os.path.getsize(os.path.join(dirpath, filename)) 
                          for dirpath, _, filenames in os.walk(dir_path) 
                          for filename in filenames)
            bytes_saved += dir_size
            
            # Delete the directory
            if not args.dry_run:
                logger.info(f"Deleting checkpoint: {dir_path}")
                shutil.rmtree(dir_path)
            else:
                logger.info(f"[DRY RUN] Would delete checkpoint: {dir_path}")
    
    # Convert to MB for better readability
    mb_saved = bytes_saved / (1024 * 1024)
    
    if not args.dry_run:
        logger.info(f"Deleted {len(to_delete)} checkpoints, keeping the {keep_count} most recent ones.")
        logger.info(f"Space saved: {mb_saved:.2f} MB")
    else:
        logger.info(f"[DRY RUN] Would delete {len(to_delete)} checkpoints, keeping the {keep_count} most recent ones.")
        logger.info(f"[DRY RUN] Space that would be saved: {mb_saved:.2f} MB")
    
    # Log what's being kept
    kept_epochs = [epoch for epoch, _ in to_keep]
    logger.info(f"Kept checkpoints for epochs: {kept_epochs}")

def cleanup_backups(args):
    """Clean up old model backups."""
    # Check if backups directory exists
    if not os.path.exists(args.backups_dir):
        logger.warning(f"Backups directory '{args.backups_dir}' does not exist. Nothing to clean up.")
        return
    
    # Get all backup timestamps
    backups = get_backup_timestamps(args.backups_dir)
    
    if not backups:
        logger.info(f"No backups found in '{args.backups_dir}'.")
        return
    
    # Determine which backups to keep
    total_backups = len(backups)
    keep_count = min(args.keep_backups, total_backups)
    
    if keep_count == total_backups:
        logger.info(f"Only {total_backups} backups exist, which is <= {args.keep_backups}. Nothing to delete.")
        return
    
    # Keep the most recent backups
    to_keep = backups[-keep_count:]
    to_delete = backups[:-keep_count]
    
    # Delete older backups
    bytes_saved = 0
    for _, dir_path in to_delete:
        if os.path.exists(dir_path):
            # Calculate size before deleting
            dir_size = sum(os.path.getsize(os.path.join(dirpath, filename)) 
                          for dirpath, _, filenames in os.walk(dir_path) 
                          for filename in filenames)
            bytes_saved += dir_size
            
            # Delete the directory
            if not args.dry_run:
                logger.info(f"Deleting backup: {dir_path}")
                shutil.rmtree(dir_path)
            else:
                logger.info(f"[DRY RUN] Would delete backup: {dir_path}")
    
    # Convert to MB for better readability
    mb_saved = bytes_saved / (1024 * 1024)
    
    if not args.dry_run:
        logger.info(f"Deleted {len(to_delete)} backups, keeping the {keep_count} most recent ones.")
        logger.info(f"Space saved: {mb_saved:.2f} MB")
    else:
        logger.info(f"[DRY RUN] Would delete {len(to_delete)} backups, keeping the {keep_count} most recent ones.")
        logger.info(f"[DRY RUN] Space that would be saved: {mb_saved:.2f} MB")

def main():
    """Main function to parse arguments and start cleanup."""
    parser = argparse.ArgumentParser(description="Clean up old Theta AI model checkpoints and backups")
    
    parser.add_argument("--checkpoints_dir", type=str, default=CHECKPOINTS_DIR,
                      help=f"Directory containing checkpoint directories (default: {CHECKPOINTS_DIR})")
    parser.add_argument("--backups_dir", type=str, default=BACKUPS_DIR,
                      help=f"Directory containing backup directories (default: {BACKUPS_DIR})")
    parser.add_argument("--keep", type=int, default=3,
                      help="Number of most recent checkpoints to keep (default: 3)")
    parser.add_argument("--keep_backups", type=int, default=2,
                      help="Number of most recent backups to keep (default: 2)")
    parser.add_argument("--clean_checkpoints", action="store_true",
                      help="Clean up checkpoint directories")
    parser.add_argument("--clean_backups", action="store_true",
                      help="Clean up backup directories")
    parser.add_argument("--clean_all", action="store_true",
                      help="Clean up both checkpoint and backup directories")
    parser.add_argument("--dry_run", action="store_true",
                      help="Show what would be deleted without actually deleting")
    
    args = parser.parse_args()
    
    # If no specific clean option is selected, show help
    if not (args.clean_checkpoints or args.clean_backups or args.clean_all):
        parser.print_help()
        return
    
    # Clean checkpoints if requested
    if args.clean_checkpoints or args.clean_all:
        logger.info("Cleaning up checkpoint directories...")
        cleanup_checkpoints(args)
    
    # Clean backups if requested
    if args.clean_backups or args.clean_all:
        logger.info("Cleaning up backup directories...")
        cleanup_backups(args)
    
    logger.info("Cleanup complete!")

if __name__ == "__main__":
    main()


#to delete all checkpoints and backups
#python cleanup_checkpoints.py --clean_all

#to delete only checkpoints
#python cleanup_checkpoints.py --clean_checkpoints

#to delete only backups
#python cleanup_checkpoints.py --clean_backups

#to delete all but 3 checkpoints
#python cleanup_checkpoints.py --keep 3

#to delete all but 2 backups
#python cleanup_checkpoints.py --keep_backups 2

#to delete all but 2 checkpoints and 1 backup

