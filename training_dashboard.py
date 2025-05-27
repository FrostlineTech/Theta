#!/usr/bin/env python
"""
Theta AI - Unified Training & GPU Health Dashboard

A comprehensive dashboard for visualizing and analyzing Theta model training metrics
and GPU health information. This unified application connects to your database and 
provides real-time insights into training progress and hardware performance,
helping identify both model training issues and hardware concerns."""

#this script creates a gui for the training metrics dashboard
#to run the dashboard run: 
#streamlit run training_dashboard.py
#this will help us evaluate the health of our hardware used to train and the health of the model training overtime
#we can use this information to make informed decisions about the hardware we use to train the model and the model training process
#also graphing is cool 


import os
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from database import ConversationDatabase

# Page configuration
st.set_page_config(
    page_title="Theta AI Unified Dashboard",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        color: #4a86e8;
        margin-bottom: 1rem;
    }
    .gpu-header {
        font-size: 2.5rem;
        color: #FF5733;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #333;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f9f9f9;
        padding: 1rem;
        border-radius: 0.5rem;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        text-align: center;
    }
    .temperature-warning {
        color: #FF5733;
        font-weight: bold;
    }
    .temperature-normal {
        color: #33A1FF;
        font-weight: bold;
    }
    .temperature-high {
        color: #FFC300;
        font-weight: bold;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: bold;
        color: #4a86e8;
    }
    .gpu-metric-value {
        font-size: 2rem;
        font-weight: bold;
        color: #FF5733;
    }
    .metric-label {
        font-size: 1rem;
        color: #666;
    }
    .tab-content {
        padding: 1rem 0;
    }
    </style>
""", unsafe_allow_html=True)

def load_metrics_data(limit=100, model_version=None, date_range=None):
    """Load training metrics from database with optional filtering."""
    try:
        db = ConversationDatabase()
        metrics = db.get_training_metrics(limit=limit)
        
        if not metrics:
            st.error("No training metrics found in the database.")
            return None
            
        # Convert to pandas DataFrame
        df = pd.DataFrame(metrics)
        
        # Convert timestamp to datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Filter by model version if specified
        if model_version and model_version != "All":
            df = df[df['model_version'] == model_version]
            
        # Filter by date range if specified
        if date_range:
            start_date, end_date = date_range
            df = df[(df['timestamp'].dt.date >= start_date) & 
                    (df['timestamp'].dt.date <= end_date)]
            
        # Sort by timestamp
        df = df.sort_values('timestamp')
        
        return df
    except Exception as e:
        st.error(f"Error loading metrics: {e}")
        return None

def plot_loss_curves(df):
    """Create interactive loss curve plot with Plotly."""
    fig = go.Figure()
    
    # Add traces for training and validation loss
    fig.add_trace(go.Scatter(
        x=df['epoch_number'], 
        y=df['train_loss'],
        mode='lines+markers',
        name='Training Loss',
        line=dict(color='blue', width=2),
        marker=dict(size=8)
    ))
    
    fig.add_trace(go.Scatter(
        x=df['epoch_number'], 
        y=df['validation_loss'],
        mode='lines+markers',
        name='Validation Loss',
        line=dict(color='red', width=2),
        marker=dict(size=8)
    ))
    
    # Calculate the gap between training and validation loss
    df['loss_gap'] = df['validation_loss'] - df['train_loss']
    
    # Add gap trace
    fig.add_trace(go.Scatter(
        x=df['epoch_number'], 
        y=df['loss_gap'],
        mode='lines+markers',
        name='Gap (Val - Train)',
        line=dict(color='green', width=2, dash='dot'),
        marker=dict(size=6),
        visible='legendonly'  # Hidden by default
    ))
    
    # Customize layout
    fig.update_layout(
        title='Training and Validation Loss Over Time',
        xaxis_title='Epoch',
        yaxis_title='Loss',
        hovermode='x unified',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        height=500
    )
    
    # Add annotations for potentially concerning patterns
    if len(df) > 1:
        # Check for increasing validation loss with decreasing training loss
        for i in range(1, len(df)):
            if (df['train_loss'].iloc[i] < df['train_loss'].iloc[i-1] and 
                df['validation_loss'].iloc[i] > df['validation_loss'].iloc[i-1] and
                df['loss_gap'].iloc[i] > 1.0):  # Gap threshold
                
                fig.add_annotation(
                    x=df['epoch_number'].iloc[i],
                    y=df['validation_loss'].iloc[i],
                    text="Potential Overfitting",
                    showarrow=True,
                    arrowhead=1,
                    arrowcolor="red",
                    arrowsize=1,
                    ax=0,
                    ay=-40
                )
    
    return fig

def plot_gpu_metrics(df):
    """Create GPU metrics visualization."""
    if 'gpu_temperature' not in df.columns or df['gpu_temperature'].isna().all():
        return None
        
    # Extract numeric values if needed
    if 'gpu_temp_value' not in df.columns:
        try:
            # Extract temperature values
            df['gpu_temp_value'] = df['gpu_temperature'].str.extract(r'(\d+)').astype(float)
        except Exception:
            # Fallback method if regex fails
            try:
                df['gpu_temp_value'] = df['gpu_temperature'].apply(
                    lambda x: float(''.join(c for c in str(x) if c.isdigit() or c == '.')) if pd.notna(x) else None
                )
            except Exception:
                return None
    
    # Extract memory and utilization if available
    if 'gpu_memory_usage' in df.columns and df['gpu_memory_usage'].notna().any():
        try:
            # Extract memory data (format: "4000 MB / 8000 MB")
            df['gpu_memory_used'] = df['gpu_memory_usage'].str.extract(r'(\d+)').astype(float)
            df['gpu_memory_total'] = df['gpu_memory_usage'].str.extract(r'\d+ MB / (\d+)').astype(float)
            df['gpu_memory_percent'] = (df['gpu_memory_used'] / df['gpu_memory_total'] * 100).round(1)
        except Exception:
            pass
    
    if 'gpu_utilization' in df.columns and df['gpu_utilization'].notna().any():
        try:
            df['gpu_util_value'] = df['gpu_utilization'].str.extract(r'(\d+)').astype(float)
        except Exception:
            pass
    
    # Create figure with secondary y-axis
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    # Add temperature trace
    fig.add_trace(
        go.Scatter(
            x=df['epoch_number'] if 'epoch_number' in df.columns else df['timestamp'], 
            y=df['gpu_temp_value'],
            name="GPU Temperature (°C)",
            mode='lines+markers',
            line=dict(color='red', width=2),
            marker=dict(size=6)
        ),
        secondary_y=False,
    )
    
    # Add horizontal threshold lines for temperature
    x_values = df['epoch_number'] if 'epoch_number' in df.columns else df['timestamp']
    fig.add_shape(
        type="line", 
        x0=x_values.min(), 
        x1=x_values.max(),
        y0=75, 
        y1=75,
        line=dict(color="orange", width=2, dash="dash")
    )
    
    fig.add_shape(
        type="line", 
        x0=x_values.min(), 
        x1=x_values.max(),
        y0=85, 
        y1=85,
        line=dict(color="red", width=2, dash="dash")
    )
    
    # Add memory usage trace if available
    if 'gpu_memory_percent' in df.columns and df['gpu_memory_percent'].notna().any():
        fig.add_trace(
            go.Scatter(
                x=df['epoch_number'] if 'epoch_number' in df.columns else df['timestamp'], 
                y=df['gpu_memory_percent'],
                name="Memory Usage (%)",
                mode='lines+markers',
                line=dict(color='blue', width=2),
                marker=dict(size=6)
            ),
            secondary_y=True,
        )
    elif 'gpu_memory_used' in df.columns and df['gpu_memory_used'].notna().any():
        fig.add_trace(
            go.Scatter(
                x=df['epoch_number'] if 'epoch_number' in df.columns else df['timestamp'], 
                y=df['gpu_memory_used'],
                name="Memory Used (MB)",
                mode='lines+markers',
                line=dict(color='blue', width=2),
                marker=dict(size=6)
            ),
            secondary_y=True,
        )
    
    # Add utilization trace if available
    if 'gpu_util_value' in df.columns and df['gpu_util_value'].notna().any():
        fig.add_trace(
            go.Scatter(
                x=df['epoch_number'] if 'epoch_number' in df.columns else df['timestamp'], 
                y=df['gpu_util_value'],
                name="GPU Utilization (%)",
                mode='lines+markers',
                line=dict(color='green', width=2),
                marker=dict(size=6)
            ),
            secondary_y=True,
        )
    
    # Set titles
    fig.update_layout(
        title='GPU Metrics During Training',
        xaxis_title='Epoch' if 'epoch_number' in df.columns else 'Time',
        hovermode="x unified",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        height=400
    )
    
    # Set y-axes titles
    fig.update_yaxes(title_text="Temperature (°C)", secondary_y=False)
    fig.update_yaxes(title_text="Usage (%)", secondary_y=True)
    
    return fig

def calculate_metrics_summary(df):
    """Calculate summary metrics for the dashboard."""
    summary = {}
    
    if df is None or df.empty:
        return summary
    
    # Latest epoch info
    latest_epoch = df['epoch_number'].max()
    latest_df = df[df['epoch_number'] == latest_epoch]
    
    summary['latest_epoch'] = int(latest_epoch)
    summary['latest_train_loss'] = latest_df['train_loss'].values[0]
    summary['latest_val_loss'] = latest_df['validation_loss'].values[0]
    summary['latest_gap'] = summary['latest_val_loss'] - summary['latest_train_loss']
    
    # Min loss values
    summary['min_train_loss'] = df['train_loss'].min()
    summary['min_train_epoch'] = df.loc[df['train_loss'].idxmin(), 'epoch_number']
    summary['min_val_loss'] = df['validation_loss'].min()
    summary['min_val_epoch'] = df.loc[df['validation_loss'].idxmin(), 'epoch_number']
    
    # Calculate average gap
    df['gap'] = df['validation_loss'] - df['train_loss']
    summary['avg_gap'] = df['gap'].mean()
    
    # Calculate average loss decrease per epoch
    if len(df) >= 2:
        df_sorted = df.sort_values('epoch_number')
        first_train = df_sorted['train_loss'].iloc[0]
        last_train = df_sorted['train_loss'].iloc[-1]
        first_val = df_sorted['validation_loss'].iloc[0]
        last_val = df_sorted['validation_loss'].iloc[-1]
        
        epochs_count = len(df_sorted) - 1
        summary['avg_train_decrease'] = (first_train - last_train) / epochs_count
        summary['avg_val_decrease'] = (first_val - last_val) / epochs_count
    else:
        summary['avg_train_decrease'] = 0
        summary['avg_val_decrease'] = 0
    
    # Determine overfitting risk - adjusted for language models like Theta
    # For LLMs, a larger gap between training and validation loss is normal
    # Especially for models with reduced complexity (768 hidden size, 12 layers)
    if summary['latest_gap'] > 2.5 or summary['avg_gap'] > 2.2:
        summary['overfitting_risk'] = "High"
    elif summary['latest_gap'] > 1.8 or summary['avg_gap'] > 1.5:
        summary['overfitting_risk'] = "Medium"
    else:
        summary['overfitting_risk'] = "Low"
    
    # GPU metrics if available
    if 'gpu_temp_value' in df.columns and df['gpu_temp_value'].notna().any():
        summary['latest_temp'] = df.loc[df['timestamp'].idxmax(), 'gpu_temp_value']
        summary['max_temp'] = df['gpu_temp_value'].max()
        summary['avg_temp'] = df['gpu_temp_value'].mean()
        summary['temp_status'] = get_temperature_status(summary['latest_temp'])
    
    if 'gpu_util_value' in df.columns and df['gpu_util_value'].notna().any():
        summary['latest_util'] = df.loc[df['timestamp'].idxmax(), 'gpu_util_value']
        summary['max_util'] = df['gpu_util_value'].max()
        summary['avg_util'] = df['gpu_util_value'].mean()
    
    if 'gpu_memory_percent' in df.columns and df['gpu_memory_percent'].notna().any():
        summary['latest_mem'] = df.loc[df['timestamp'].idxmax(), 'gpu_memory_percent']
        summary['max_mem'] = df['gpu_memory_percent'].max()
        summary['avg_mem'] = df['gpu_memory_percent'].mean()
    
    return summary


def calculate_temperature_metrics(df):
    """Calculate temperature-related metrics."""
    metrics = {}
    
    if 'gpu_temp_value' not in df.columns or df['gpu_temp_value'].isna().all():
        return metrics
    
    # Latest temperature
    metrics["latest_temp"] = df['gpu_temp_value'].iloc[-1]
    
    # Maximum temperature
    metrics["max_temp"] = df['gpu_temp_value'].max()
    max_temp_idx = df['gpu_temp_value'].idxmax()
    metrics["max_temp_time"] = df.loc[max_temp_idx, 'timestamp']
    
    # Average temperature
    metrics["avg_temp"] = df['gpu_temp_value'].mean()
    
    # Time spent above warning and critical thresholds
    metrics["time_above_warning"] = len(df[df['gpu_temp_value'] >= 75])
    metrics["time_above_critical"] = len(df[df['gpu_temp_value'] >= 85])
    
    # Temperature trend (last 10 readings or all if less)
    sample_size = min(10, len(df))
    if sample_size > 1:
        recent_temps = df['gpu_temp_value'].iloc[-sample_size:]
        first_temp = recent_temps.iloc[0]
        last_temp = recent_temps.iloc[-1]
        metrics["temp_trend"] = last_temp - first_temp
    
    return metrics


def get_temperature_status(temp):
    """Get the temperature status based on value."""
    if temp is None:
        return "unknown"
    elif temp >= 85:
        return "critical"
    elif temp >= 75:
        return "warning"
    else:
        return "normal"


def get_temperature_class(status):
    """Get the CSS class for temperature status."""
    if status == "critical":
        return "temperature-warning"
    elif status == "warning":
        return "temperature-high"
    else:
        return "temperature-normal"

def main():
    """Main dashboard application."""
    # Header
    st.markdown('<div class="main-header">Theta AI Unified Dashboard</div>', unsafe_allow_html=True)
    st.markdown('Monitor your model training progress, analyze GPU health metrics, and detect potential issues.')
    
    # Sidebar filters
    st.sidebar.header("Filters")
    
    # Date range selector
    st.sidebar.subheader("Date Range")
    today = datetime.now().date()
    thirty_days_ago = today - timedelta(days=30)
    
    start_date = st.sidebar.date_input("Start Date", thirty_days_ago)
    end_date = st.sidebar.date_input("End Date", today)
    
    if start_date > end_date:
        st.sidebar.error("End date must be after start date")
        return
    
    date_range = (start_date, end_date)
    
    # Model version selector
    st.sidebar.subheader("Model Version")
    
    # Load data with minimal filter to get all model versions
    all_df = load_metrics_data(limit=1000)
    if all_df is not None:
        model_versions = ["All"] + sorted(all_df['model_version'].unique().tolist())
        selected_model = st.sidebar.selectbox("Select Model Version", model_versions)
    else:
        selected_model = None
    
    # Number of records to show
    st.sidebar.subheader("Data Limit")
    record_limit = st.sidebar.slider("Max Records to Load", min_value=10, max_value=1000, value=200, step=10)
    
    # Load filtered data
    df = load_metrics_data(limit=record_limit, model_version=selected_model, date_range=date_range)
    
    if df is not None and not df.empty:
        # Calculate metrics summary
        summary = calculate_metrics_summary(df)
        
        # Create tabs for Training, GPU Health and CPU Health
        tab1, tab2, tab3 = st.tabs(["📊 Training Metrics", "🔥 GPU Health", "💻 CPU Health"])
        
        # Training tab
        with tab1:
            st.markdown("### Training Summary")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.markdown(
                    f"""
                    <div class="metric-card">
                        <div class="metric-value">{summary['latest_epoch']}</div>
                        <div class="metric-label">Latest Epoch</div>
                    </div>
                    """, 
                    unsafe_allow_html=True
                )
            
            with col2:
                st.markdown(
                    f"""
                    <div class="metric-card">
                        <div class="metric-value">{summary['latest_train_loss']:.4f}</div>
                        <div class="metric-label">Latest Training Loss</div>
                    </div>
                    """, 
                    unsafe_allow_html=True
                )
                
            with col3:
                st.markdown(
                    f"""
                    <div class="metric-card">
                        <div class="metric-value">{summary['latest_val_loss']:.4f}</div>
                        <div class="metric-label">Latest Validation Loss</div>
                    </div>
                    """, 
                    unsafe_allow_html=True
                )
                
            with col4:
                risk_class = "temperature-warning" if summary['overfitting_risk'] == "High" else \
                            "temperature-high" if summary['overfitting_risk'] == "Medium" else "temperature-normal"
                st.markdown(
                    f"""
                    <div class="metric-card">
                        <div class="metric-value {risk_class}">{summary['overfitting_risk']}</div>
                        <div class="metric-label">Overfitting Risk</div>
                    </div>
                    """, 
                    unsafe_allow_html=True
                )
            
            # Loss curves
            st.markdown('<div class="sub-header">Loss Curves</div>', unsafe_allow_html=True)
            loss_fig = plot_loss_curves(df)
            st.plotly_chart(loss_fig, use_container_width=True)
            
            # Training insights
            st.markdown('<div class="sub-header">Training Insights</div>', unsafe_allow_html=True)
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Best Epochs:**")
                st.markdown(f"• Lowest Training Loss: **{summary['min_train_loss']:.4f}** (Epoch {summary['min_train_epoch']})")
                st.markdown(f"• Lowest Validation Loss: **{summary['min_val_loss']:.4f}** (Epoch {summary['min_val_epoch']})")
                
                st.markdown("**Progress Rate:**")
                st.markdown(f"• Avg. Training Loss Decrease: **{summary['avg_train_decrease']:.4f}** per epoch")
                st.markdown(f"• Avg. Validation Loss Decrease: **{summary['avg_val_decrease']:.4f}** per epoch")
            
            with col2:
                st.markdown("**Overfitting Analysis:**")
                st.markdown(f"• Current Gap (Val - Train): **{summary['latest_gap']:.4f}**")
                st.markdown(f"• Average Gap: **{summary['avg_gap']:.4f}**")
                
                # Recommendations based on metrics
                st.markdown("**Recommendations:**")
                if summary['overfitting_risk'] == "High":
                    st.markdown("• ⚠️ **High risk of overfitting detected.** Consider increasing dropout, reducing model complexity, or applying stronger regularization.")
                elif summary['overfitting_risk'] == "Medium":
                    st.markdown("• ⚠️ **Moderate risk of overfitting.** Monitor the validation loss closely in the next few epochs.")
                else:
                    st.markdown("• ✅ **Model training appears healthy.** Continue with current hyperparameters.")
                    
                # Add note about Theta model architecture
                st.markdown("• 🔄 Your model was recently modified with smaller hidden size (768) and fewer layers (12) which should help with generalization.")
        
        # GPU Health tab
        with tab2:
            # GPU health metrics at a glance
            st.markdown("### GPU Health Summary")
            
            # Process GPU data if not already processed
            if 'gpu_temperature' in df.columns and df['gpu_temperature'].notna().any() and 'gpu_temp_value' not in df.columns:
                try:
                    # Extract temperature values
                    df['gpu_temp_value'] = df['gpu_temperature'].str.extract(r'(\d+)').astype(float)
                    
                    # Process memory and utilization data too
                    if 'gpu_memory_usage' in df.columns and df['gpu_memory_usage'].notna().any():
                        df['gpu_memory_used'] = df['gpu_memory_usage'].str.extract(r'(\d+)').astype(float)
                        df['gpu_memory_total'] = df['gpu_memory_usage'].str.extract(r'\d+ MB / (\d+)').astype(float)
                        df['gpu_memory_percent'] = (df['gpu_memory_used'] / df['gpu_memory_total'] * 100).round(1)
                    
                    if 'gpu_utilization' in df.columns and df['gpu_utilization'].notna().any():
                        df['gpu_util_value'] = df['gpu_utilization'].str.extract(r'(\d+)').astype(float)
                except Exception as e:
                    st.error(f"Error processing GPU data: {e}")
            
            # Calculate GPU metrics for summary
            gpu_summary = {}
            if 'gpu_temp_value' in df.columns and df['gpu_temp_value'].notna().any():
                gpu_summary['latest_temp'] = df['gpu_temp_value'].iloc[-1]
                gpu_summary['max_temp'] = df['gpu_temp_value'].max()
                gpu_summary['avg_temp'] = df['gpu_temp_value'].mean()
                gpu_summary['temp_status'] = get_temperature_status(gpu_summary['latest_temp'])
            
            if 'gpu_util_value' in df.columns and df['gpu_util_value'].notna().any():
                gpu_summary['latest_util'] = df['gpu_util_value'].iloc[-1]
                gpu_summary['max_util'] = df['gpu_util_value'].max()
                gpu_summary['avg_util'] = df['gpu_util_value'].mean()
            
            if 'gpu_memory_percent' in df.columns and df['gpu_memory_percent'].notna().any():
                gpu_summary['latest_mem'] = df['gpu_memory_percent'].iloc[-1]
                gpu_summary['max_mem'] = df['gpu_memory_percent'].max()
                gpu_summary['avg_mem'] = df['gpu_memory_percent'].mean()
                
            # Calculate CPU metrics for summary
            cpu_summary = {}
            if 'cpu_temperature' in df.columns and df['cpu_temperature'].notna().any():
                cpu_summary['latest_temp'] = df['cpu_temperature'].iloc[-1]
                cpu_summary['max_temp'] = df['cpu_highest_temp'].iloc[-1] if 'cpu_highest_temp' in df.columns else df['cpu_temperature'].max()
                cpu_summary['avg_temp'] = df['cpu_temperature'].mean()
                cpu_summary['temp_status'] = get_temperature_status(cpu_summary['latest_temp'])
            
            if 'cpu_utilization' in df.columns and df['cpu_utilization'].notna().any():
                cpu_summary['latest_util'] = df['cpu_utilization'].iloc[-1]
                cpu_summary['max_util'] = df['cpu_highest_utilization'].iloc[-1] if 'cpu_highest_utilization' in df.columns else df['cpu_utilization'].max()
                cpu_summary['avg_util'] = df['cpu_utilization'].mean()
            
            # Only show if GPU data is available
            if 'gpu_temp_value' in df.columns and len(gpu_summary) > 0:
                # Temperature Summary
                st.markdown("#### Temperature Summary")
                
                # Create separate temperature cards
                col1, col2, col3, col4 = st.columns(4)
                
                # Current temperature
                current_temp = gpu_summary['latest_temp']
                current_temp_class = get_temperature_class(get_temperature_status(current_temp))
                with col1:
                    st.markdown(
                        f"""
                        <div class="metric-card">
                            <div class="metric-label">Current Temperature</div>
                            <div class="metric-value {current_temp_class}">{current_temp:.1f}°C</div>
                        </div>
                        """, 
                        unsafe_allow_html=True
                    )
                
                # Highest temperature
                max_temp = gpu_summary['max_temp']
                max_temp_class = get_temperature_class(get_temperature_status(max_temp))
                with col2:
                    st.markdown(
                        f"""
                        <div class="metric-card">
                            <div class="metric-label">Highest Temperature</div>
                            <div class="metric-value {max_temp_class}">{max_temp:.1f}°C</div>
                        </div>
                        """, 
                        unsafe_allow_html=True
                    )
                
                # Average temperature
                avg_temp = gpu_summary['avg_temp']
                avg_temp_class = get_temperature_class(get_temperature_status(avg_temp))
                with col3:
                    st.markdown(
                        f"""
                        <div class="metric-card">
                            <div class="metric-label">Average Temperature</div>
                            <div class="metric-value {avg_temp_class}">{avg_temp:.1f}°C</div>
                        </div>
                        """, 
                        unsafe_allow_html=True
                    )
                
                # Warning level status
                status = "NORMAL" if current_temp < 75 else "WARNING" if current_temp < 85 else "CRITICAL"
                status_class = "temperature-normal" if status == "NORMAL" else "temperature-high" if status == "WARNING" else "temperature-warning"
                with col4:
                    st.markdown(
                        f"""
                        <div class="metric-card">
                            <div class="metric-label">Warning Level</div>
                            <div class="metric-value {status_class}">{status}</div>
                        </div>
                        """, 
                        unsafe_allow_html=True
                    )
                
                # Temperature warning level meter in its own box
                st.markdown("#### Temperature Warning Level")
                
                # Create a custom temperature warning level indicator with colored gradient and marker
                st.markdown(
                    f"""
                    <style>
                    .warning-meter-container {{
                        background-color: #f0f2f6;
                        padding: 15px;
                        border-radius: 5px;
                        margin-bottom: 20px;
                    }}
                    .warning-meter {{
                        position: relative;
                        height: 20px;
                        background: linear-gradient(to right, #33A1FF, #33A1FF 74%, #FFC300 74%, #FFC300 84%, #FF5733 84%, #FF5733 100%);
                        border-radius: 10px;
                        margin-top: 10px;
                        margin-bottom: 30px;
                    }}
                    .warning-marker {{
                        position: absolute;
                        top: -7px;
                        left: {min(max(current_temp, 0), 100)}%;
                        width: 3px;
                        height: 35px;
                        background-color: black;
                        transform: translateX(-50%);
                    }}
                    .warning-marker-label {{
                        position: absolute;
                        top: -30px;
                        left: 50%;
                        transform: translateX(-50%);
                        background-color: black;
                        color: white;
                        padding: 2px 6px;
                        border-radius: 3px;
                        font-size: 12px;
                        white-space: nowrap;
                    }}
                    .warning-labels {{
                        display: flex;
                        justify-content: space-between;
                        margin-top: 5px;
                    }}
                    </style>
                    
                    <div class="warning-meter-container">
                        <div class="warning-meter">
                            <div class="warning-marker">
                                <div class="warning-marker-label">{current_temp:.1f}°C</div>
                            </div>
                        </div>
                        <div class="warning-labels">
                            <div>Normal (0-74°C)</div>
                            <div>Warning (75-84°C)</div>
                            <div>Critical (85°C+)</div>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
                
                # Add a status box below the meter
                status = "NORMAL" if current_temp < 75 else "WARNING" if current_temp < 85 else "CRITICAL"
                status_color = "#33A1FF" if status == "NORMAL" else "#FFC300" if status == "WARNING" else "#FF5733"
                
                st.markdown(
                    f"""
                    <div style="text-align: center; margin-top: 10px; margin-bottom: 20px;">
                        <div style="display: inline-block; background-color: {status_color}; color: {'white' if status != 'NORMAL' else 'black'}; 
                              padding: 5px 15px; border-radius: 5px; font-weight: bold;">
                            Current Status: {status}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
                
                # Create GPU metrics per epoch visualization
                if 'epoch_number' in df.columns and 'gpu_temp_value' in df.columns:
                    epoch_fig = go.Figure()
                    
                    # Get temperature data by epoch
                    epoch_temp = df.groupby('epoch_number')['gpu_temp_value'].agg(['mean', 'max']).reset_index()
                    
                    # Add temperature trace
                    epoch_fig.add_trace(go.Scatter(
                        x=epoch_temp['epoch_number'], 
                        y=epoch_temp['mean'],
                        mode='lines+markers',
                        name='Avg Temp (°C)',
                        line=dict(color='orange', width=2),
                        marker=dict(size=8)
                    ))
                    
                    # Add warning and critical temperature lines
                    epoch_fig.add_shape(
                        type="line", 
                        x0=epoch_temp['epoch_number'].min(), 
                        x1=epoch_temp['epoch_number'].max(),
                        y0=75, 
                        y1=75,
                        line=dict(color="orange", width=2, dash="dash")
                    )
                    
                    epoch_fig.add_shape(
                        type="line", 
                        x0=epoch_temp['epoch_number'].min(), 
                        x1=epoch_temp['epoch_number'].max(),
                        y0=85, 
                        y1=85,
                        line=dict(color="red", width=2, dash="dash")
                    )
                    
                    # Set titles and layout
                    epoch_fig.update_layout(
                        title="GPU Temperature by Epoch",
                        xaxis_title="Epoch Number",
                        yaxis_title="Temperature (°C)",
                        hovermode="x unified",
                        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                    )
                    
                    st.plotly_chart(epoch_fig, use_container_width=True)
                
                # GPU utilization and memory plots
                st.markdown('<div class="sub-header">GPU Utilization & Memory</div>', unsafe_allow_html=True)
                
                # Create utilization and memory plot
                if (('gpu_util_value' in df.columns and df['gpu_util_value'].notna().any()) or 
                    ('gpu_memory_percent' in df.columns and df['gpu_memory_percent'].notna().any())):
                    
                    util_fig = go.Figure()
                    
                    # Add utilization trace if available
                    if 'gpu_util_value' in df.columns and df['gpu_util_value'].notna().any():
                        util_fig.add_trace(
                            go.Scatter(
                                x=df['timestamp'], 
                                y=df['gpu_util_value'],
                                name="GPU Utilization (%)",
                                line=dict(color='green', width=2)
                            )
                        )
                    
                    # Add memory usage trace if available
                    if 'gpu_memory_percent' in df.columns and df['gpu_memory_percent'].notna().any():
                        util_fig.add_trace(
                            go.Scatter(
                                x=df['timestamp'], 
                                y=df['gpu_memory_percent'],
                                name="Memory Usage (%)",
                                line=dict(color='blue', width=2)
                            )
                        )
                    
                    # Set titles and layout
                    util_fig.update_layout(
                        xaxis_title="Time",
                        yaxis_title="Percentage (%)",
                        hovermode="x unified",
                        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                    )
                    
                    st.plotly_chart(util_fig, use_container_width=True)
                
                # Create per-epoch temperature and utilization chart
                if 'epoch_number' in df.columns:
                    st.markdown('<div class="sub-header">GPU Metrics Per Epoch</div>', unsafe_allow_html=True)
                    
                    # Group data by epoch
                    epoch_data = {}
                    
                    if 'gpu_temp_value' in df.columns:
                        epoch_temp = df.groupby('epoch_number')['gpu_temp_value'].agg(['mean', 'max']).reset_index()
                        epoch_data['temperatures'] = epoch_temp
                    
                    if 'gpu_util_value' in df.columns:
                        epoch_util = df.groupby('epoch_number')['gpu_util_value'].agg(['mean', 'max']).reset_index()
                        epoch_data['utilization'] = epoch_util
                    
                    if 'gpu_memory_percent' in df.columns:
                        epoch_mem = df.groupby('epoch_number')['gpu_memory_percent'].agg(['mean', 'max']).reset_index()
                        epoch_data['memory'] = epoch_mem
                    
                    # Create the figure
                    if epoch_data:
                        fig = go.Figure()
                        
                        # Add temperature traces
                        if 'temperatures' in epoch_data:
                            fig.add_trace(go.Scatter(
                                x=epoch_data['temperatures']['epoch_number'], 
                                y=epoch_data['temperatures']['max'],
                                mode='lines+markers',
                                name='Max Temp (°C)',
                                line=dict(color='red', width=2),
                                marker=dict(size=8)
                            ))
                            
                            fig.add_trace(go.Scatter(
                                x=epoch_data['temperatures']['epoch_number'], 
                                y=epoch_data['temperatures']['mean'],
                                mode='lines+markers',
                                name='Avg Temp (°C)',
                                line=dict(color='orange', width=2),
                                marker=dict(size=8)
                            ))
                            
                            # Add warning and critical temperature lines
                            fig.add_shape(
                                type="line", 
                                x0=epoch_data['temperatures']['epoch_number'].min(), 
                                x1=epoch_data['temperatures']['epoch_number'].max(),
                                y0=75, 
                                y1=75,
                                line=dict(color="orange", width=2, dash="dash")
                            )
                            
                            fig.add_shape(
                                type="line", 
                                x0=epoch_data['temperatures']['epoch_number'].min(), 
                                x1=epoch_data['temperatures']['epoch_number'].max(),
                                y0=85, 
                                y1=85,
                                line=dict(color="red", width=2, dash="dash")
                            )
                        
                        # Add utilization and memory traces if available
                        if 'utilization' in epoch_data:
                            fig.add_trace(go.Scatter(
                                x=epoch_data['utilization']['epoch_number'], 
                                y=epoch_data['utilization']['max'],
                                mode='lines+markers',
                                name='Max Utilization (%)',
                                line=dict(color='green', width=2),
                                marker=dict(size=8),
                                yaxis='y2'
                            ))
                        
                        if 'memory' in epoch_data:
                            fig.add_trace(go.Scatter(
                                x=epoch_data['memory']['epoch_number'], 
                                y=epoch_data['memory']['max'],
                                mode='lines+markers',
                                name='Max Memory Usage (%)',
                                line=dict(color='blue', width=2),
                                marker=dict(size=8),
                                yaxis='y2'
                            ))
                        
                        # Update layout
                        fig.update_layout(
                            title='GPU Metrics by Epoch',
                            xaxis_title='Epoch',
                            yaxis_title='Temperature (°C)',
                            yaxis2=dict(
                                title='Utilization/Memory (%)',
                                overlaying='y',
                                side='right',
                                range=[0, 100]
                            ),
                            hovermode='x unified',
                            legend=dict(
                                orientation="h",
                                yanchor="bottom",
                                y=1.02,
                                xanchor="right",
                                x=1
                            )
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                
                # GPU health analysis
                st.markdown('<div class="sub-header">GPU Health Analysis</div>', unsafe_allow_html=True)
                
                # Show analysis based on the gpu_summary we already calculated
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**Temperature Analysis:**")
                    if 'max_temp' in gpu_summary:
                        st.markdown(f"• Maximum temperature: **{gpu_summary['max_temp']:.1f}°C**")
                        st.markdown(f"• Average temperature: **{gpu_summary['avg_temp']:.1f}°C**")
                        
                        # Calculate time above thresholds
                        time_above_warning = len(df[df['gpu_temp_value'] >= 75])
                        time_above_critical = len(df[df['gpu_temp_value'] >= 85])
                        
                        if time_above_warning > 0:
                            st.markdown(f"• Time spent above 75°C: **{time_above_warning}** data points")
                        if time_above_critical > 0:
                            st.markdown(f"• Time spent above 85°C: **{time_above_critical}** data points")
                        
                        # Temperature per epoch chart
                        st.markdown("**Temperature per Epoch:**")
                        if 'epoch_number' in df.columns:
                            epoch_temp_df = df.groupby('epoch_number')['gpu_temp_value'].agg(['max', 'mean']).reset_index()
                            epoch_temp_df.columns = ['Epoch', 'Max Temp', 'Avg Temp']
                            st.dataframe(epoch_temp_df, hide_index=True)
                
                with col2:
                    st.markdown("**GPU Usage Analysis:**")
                    
                    # Determine if utilization is appropriate for the model architecture
                    if 'latest_util' in gpu_summary:
                        # For ML training, 100% utilization is normal and desired
                        if gpu_summary['latest_util'] < 30:
                            st.markdown(f"• ℹ️ **GPU is underutilized at {gpu_summary['latest_util']:.1f}%**")
                        elif gpu_summary['latest_util'] > 90:
                            st.markdown(f"• ✅ **GPU is optimally utilized at {gpu_summary['latest_util']:.1f}%**")
                        else:
                            st.markdown(f"• ✅ **GPU utilization is good at {gpu_summary['latest_util']:.1f}%**")
                        
                    # Memory usage analysis
                    if 'latest_mem' in gpu_summary:
                        if gpu_summary['latest_mem'] > 98:
                            st.markdown(f"• Memory usage is very high at **{gpu_summary['latest_mem']:.1f}%** - this is normal for ML training but close to limit.")
                        elif gpu_summary['latest_mem'] > 90:
                            st.markdown(f"• ✅ Memory usage is high at **{gpu_summary['latest_mem']:.1f}%** - this is optimal for ML training efficiency.")
                        else:
                            st.markdown(f"• Memory usage is at **{gpu_summary['latest_mem']:.1f}%**, which is good for current training parameters.")
                    
                    # Add note about model architecture
                    st.markdown("**Recommendations:**")
                    st.markdown(f"• Your model with reduced complexity (768 hidden size, 12 layers) should maintain good efficiency with the current hardware.")
                    st.markdown(f"• 100% GPU utilization is normal and optimal for machine learning workloads.")
        
        # Data tables in both tabs
        st.markdown('<div class="sub-header">Raw Training Data</div>', unsafe_allow_html=True)
        
        # Select relevant columns for display
        display_cols = ['timestamp', 'epoch_number', 'model_version', 'train_loss', 'validation_loss']
        if 'gpu_temp_value' in df.columns:
            display_cols.append('gpu_temp_value')
        if 'gpu_util_value' in df.columns:
            display_cols.append('gpu_util_value')
        if 'gpu_memory_percent' in df.columns:
            display_cols.append('gpu_memory_percent')
        
        # Display the dataframe
        st.dataframe(df[display_cols].sort_values('timestamp', ascending=False))
        
    else:
        st.warning("No data available with the current filters. Please adjust your filters or ensure your database contains training metrics.")

if __name__ == "__main__":
    main()
