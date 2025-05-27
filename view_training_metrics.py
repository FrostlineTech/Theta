#!/usr/bin/env python
"""
Theta AI - Training Metrics Viewer

This script allows you to view training metrics stored in the database.
It can display training progress, compare loss values across epochs,
and help track model improvements over time.
"""
#to view training metrics run: 
#python view_training_metrics.py --show-gpu --plot

#this will show the training metrics in a table and plot the loss curves for the training and validation data 
#we can use this database information to create a graph of the training and validation loss over time to see how the model is doing over time
#we can also use this information to see if the model is overfitting or underfitting
#if a developer notices that the model is overfitting or underfitting they can adjust the training parameters to improve the model

import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt
from tabulate import tabulate
from database import ConversationDatabase

def format_metrics(metrics, show_gpu=False):
    """Format metrics for display."""
    # Convert to pandas DataFrame for easier manipulation
    df = pd.DataFrame(metrics)
    
    # Format timestamp
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['formatted_time'] = df['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
    
    # Determine columns to display
    display_cols = ['epoch_number', 'formatted_time', 'train_loss', 'validation_loss']
    
    if 'model_version' in df.columns:
        display_cols.append('model_version')
    
    if show_gpu and 'gpu_temperature' in df.columns:
        display_cols.extend(['gpu_temperature', 'gpu_memory_usage', 'gpu_utilization'])
    
    # Select relevant columns and return as list of dicts
    return df[display_cols].to_dict('records')

def plot_loss_curves(metrics, output_file=None):
    """Plot train and validation loss curves."""
    df = pd.DataFrame(metrics)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('epoch_number')
    
    plt.figure(figsize=(12, 6))
    plt.plot(df['epoch_number'], df['train_loss'], label='Training Loss', marker='o')
    plt.plot(df['epoch_number'], df['validation_loss'], label='Validation Loss', marker='x')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Add gap between train and validation loss as annotations
    for i, row in df.iterrows():
        gap = row['validation_loss'] - row['train_loss']
        plt.annotate(f"{gap:.2f}", 
                     xy=(row['epoch_number'], (row['train_loss'] + row['validation_loss'])/2),
                     xytext=(5, 0), 
                     textcoords='offset points',
                     fontsize=8,
                     alpha=0.7)
    
    if output_file:
        plt.savefig(output_file)
        print(f"Plot saved to {output_file}")
    else:
        plt.show()

def main():
    """Main function to parse arguments and display metrics."""
    parser = argparse.ArgumentParser(description="View Theta AI training metrics")
    
    parser.add_argument("--limit", type=int, default=10,
                      help="Number of most recent metrics to display")
    parser.add_argument("--show-gpu", action="store_true",
                      help="Show GPU-related metrics")
    parser.add_argument("--plot", action="store_true",
                      help="Plot training and validation loss curves")
    parser.add_argument("--output", type=str, default=None,
                      help="Save plot to the specified file (e.g., 'plot.png')")
    parser.add_argument("--model", type=str, default=None,
                      help="Filter metrics by model version")
    
    args = parser.parse_args()
    
    # Initialize database connection
    db = ConversationDatabase()
    
    try:
        # Get metrics from database
        metrics = db.get_training_metrics(limit=args.limit)
        
        # Filter by model version if specified
        if args.model and metrics:
            metrics = [m for m in metrics if m.get('model_version') == args.model]
        
        if not metrics:
            print("No training metrics found in the database.")
            return
        
        # Format metrics for display
        formatted_metrics = format_metrics(metrics, show_gpu=args.show_gpu)
        
        # Print metrics as table
        headers = formatted_metrics[0].keys()
        rows = [m.values() for m in formatted_metrics]
        print(tabulate(rows, headers=headers, tablefmt='grid'))
        
        # Plot metrics if requested
        if args.plot:
            plot_loss_curves(metrics, output_file=args.output)
        
    except Exception as e:
        print(f"Error retrieving metrics: {e}")
    
if __name__ == "__main__":
    main()
