#!/usr/bin/env python
"""
Theta AI - GPU Health Monitoring Dashboard

A dedicated dashboard for monitoring GPU health metrics during Theta AI model training.
Tracks temperature, memory usage, utilization, and records max temperatures reached.
"""

import os
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime, timedelta
from database import ConversationDatabase

# Initialize database connection
db = ConversationDatabase()

# Page configuration
st.set_page_config(
    page_title="Theta AI GPU Health Monitor",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
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
    }
    .metric-label {
        font-size: 1rem;
        color: #666;
    }
    </style>
""", unsafe_allow_html=True)

def load_gpu_metrics(limit=500, date_range=None):
    """Load GPU metrics from database with optional filtering."""
    try:
        metrics = db.get_training_metrics(limit=limit)
        
        if not metrics:
            st.error("No training metrics found in the database.")
            return None
            
        # Convert to pandas DataFrame
        df = pd.DataFrame(metrics)
        
        # Convert timestamp to datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Filter by date range if specified
        if date_range:
            start_date, end_date = date_range
            df = df[(df['timestamp'].dt.date >= start_date) & 
                    (df['timestamp'].dt.date <= end_date)]
            
        # Sort by timestamp
        df = df.sort_values('timestamp')
        
        # Extract numeric values from GPU metrics
        if 'gpu_temperature' in df.columns and df['gpu_temperature'].notna().any():
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
                    # Last resort - use a default value
                    df['gpu_temp_value'] = 40.0  # Default value
        
        if 'gpu_memory_usage' in df.columns and df['gpu_memory_usage'].notna().any():
            # Extract used memory and total memory
            df['gpu_memory_used'] = df['gpu_memory_usage'].str.extract(r'(\d+)').astype(float)
            df['gpu_memory_total'] = df['gpu_memory_usage'].str.extract(r'\d+ MB \/ (\d+)').astype(float)
            df['gpu_memory_percent'] = (df['gpu_memory_used'] / df['gpu_memory_total'] * 100).round(1)
        
        if 'gpu_utilization' in df.columns and df['gpu_utilization'].notna().any():
            df['gpu_util_value'] = df['gpu_utilization'].str.extract(r'(\d+)').astype(float)
            
        # Update max GPU temperature
        if 'gpu_temp_value' in df.columns:
            df['max_gpu_temp'] = df['gpu_temp_value'].expanding().max()
            
            # Update the database with max temperatures (only if they've changed)
            for index, row in df.iterrows():
                if pd.notna(row['gpu_temp_value']):
                    epoch = int(row['epoch_number'])
                    model_version = row['model_version']
                    current_temp = row['gpu_temp_value']
                    max_temp = row['max_gpu_temp']
                    
                    # Only update if the current temp equals the max (which means it's a new max)
                    if current_temp == max_temp and pd.notna(current_temp):
                        try:
                            # SQL to update the max_gpu_temperature
                            sql = """
                            UPDATE training_metrics
                            SET max_gpu_temperature = %s
                            WHERE epoch_number = %s AND model_version = %s
                            """
                            
                            with db.get_connection() as conn:
                                with conn.cursor() as cur:
                                    cur.execute(sql, (f"{max_temp}°C", epoch, model_version))
                                    conn.commit()
                        except Exception as e:
                            st.error(f"Error updating max GPU temperature: {e}")
        
        return df
    
    except Exception as e:
        st.error(f"Error loading GPU metrics: {e}")
        return None

def create_temperature_plot(df):
    """Create GPU temperature plot with threshold markers."""
    if 'gpu_temp_value' not in df.columns:
        return None
    
    fig = go.Figure()
    
    # Add temperature trace
    fig.add_trace(go.Scatter(
        x=df['timestamp'],
        y=df['gpu_temp_value'],
        mode='lines+markers',
        name='GPU Temperature (°C)',
        line=dict(color='#FF5733', width=2),
        marker=dict(size=6)
    ))
    
    # Add max temperature trace
    if 'max_gpu_temp' in df.columns:
        fig.add_trace(go.Scatter(
            x=df['timestamp'],
            y=df['max_gpu_temp'],
            mode='lines',
            name='Max Temperature Reached (°C)',
            line=dict(color='#900C3F', width=2, dash='dash')
        ))
    
    # Add warning threshold line at 75°C
    fig.add_trace(go.Scatter(
        x=[df['timestamp'].min(), df['timestamp'].max()],
        y=[75, 75],
        mode='lines',
        name='Warning Threshold (75°C)',
        line=dict(color='orange', width=1.5, dash='dot')
    ))
    
    # Add critical threshold line at 85°C
    fig.add_trace(go.Scatter(
        x=[df['timestamp'].min(), df['timestamp'].max()],
        y=[85, 85],
        mode='lines',
        name='Critical Threshold (85°C)',
        line=dict(color='red', width=1.5, dash='dot')
    ))
    
    # Customize layout
    fig.update_layout(
        title='GPU Temperature Over Time',
        xaxis_title='Time',
        yaxis_title='Temperature (°C)',
        yaxis=dict(range=[0, max(100, df['gpu_temp_value'].max() * 1.1)]),
        hovermode='x unified',
        height=500
    )
    
    return fig

def create_utilization_plot(df):
    """Create GPU utilization and memory usage plot."""
    if 'gpu_memory_percent' not in df.columns and 'gpu_util_value' not in df.columns:
        return None
    
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    # Add memory usage trace if available
    if 'gpu_memory_percent' in df.columns:
        fig.add_trace(go.Scatter(
            x=df['timestamp'],
            y=df['gpu_memory_percent'],
            mode='lines',
            name='Memory Usage (%)',
            line=dict(color='#3498DB', width=2)
        ), secondary_y=False)
    
    # Add utilization trace if available
    if 'gpu_util_value' in df.columns:
        fig.add_trace(go.Scatter(
            x=df['timestamp'],
            y=df['gpu_util_value'],
            mode='lines',
            name='GPU Utilization (%)',
            line=dict(color='#2ECC71', width=2)
        ), secondary_y=True)
    
    # Customize layout
    fig.update_layout(
        title='GPU Utilization and Memory Usage',
        xaxis_title='Time',
        hovermode='x unified',
        height=400
    )
    
    # Set y-axis titles
    fig.update_yaxes(title_text="Memory Usage (%)", secondary_y=False)
    fig.update_yaxes(title_text="GPU Utilization (%)", secondary_y=True)
    
    return fig

def calculate_temperature_metrics(df):
    """Calculate temperature-related metrics."""
    if df is None or 'gpu_temp_value' not in df.columns:
        return {}
    
    # Get the latest temperature
    latest_temp = df['gpu_temp_value'].iloc[-1] if not df['gpu_temp_value'].empty else None
    
    # Get the maximum temperature
    max_temp = df['gpu_temp_value'].max() if not df['gpu_temp_value'].empty else None
    
    # Calculate average temperature
    avg_temp = df['gpu_temp_value'].mean() if not df['gpu_temp_value'].empty else None
    
    # Get the timestamp when the max temperature was reached
    max_temp_idx = df['gpu_temp_value'].idxmax() if not df['gpu_temp_value'].empty else None
    max_temp_time = df.loc[max_temp_idx, 'timestamp'] if max_temp_idx is not None else None
    
    # Calculate time spent above 75°C (warning threshold)
    time_above_warning = len(df[df['gpu_temp_value'] > 75]) if not df['gpu_temp_value'].empty else 0
    
    # Calculate time spent above 85°C (critical threshold)
    time_above_critical = len(df[df['gpu_temp_value'] > 85]) if not df['gpu_temp_value'].empty else 0
    
    # Temperature trend (last 10 datapoints if available)
    recent_count = min(10, len(df))
    if recent_count > 1 and not df['gpu_temp_value'].empty:
        recent_temps = df['gpu_temp_value'].iloc[-recent_count:]
        temp_trend = recent_temps.iloc[-1] - recent_temps.iloc[0]
    else:
        temp_trend = None
    
    return {
        "latest_temp": latest_temp,
        "max_temp": max_temp,
        "avg_temp": avg_temp,
        "max_temp_time": max_temp_time,
        "time_above_warning": time_above_warning,
        "time_above_critical": time_above_critical,
        "temp_trend": temp_trend
    }

def get_temperature_status(temp):
    """Get the temperature status based on value."""
    if temp is None:
        return "unknown"
    elif temp > 85:
        return "critical"
    elif temp > 75:
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
    st.markdown('<div class="main-header">Theta AI GPU Health Monitor</div>', unsafe_allow_html=True)
    st.markdown(
        "Monitor GPU health during Theta AI model training to prevent hardware issues and ensure optimal performance."
    )
    
    # Sidebar filters
    st.sidebar.title("Filters")
    
    # Get date range for metrics
    metrics = db.get_training_metrics(limit=1)
    if metrics:
        df_all = pd.DataFrame(metrics)
        df_all['timestamp'] = pd.to_datetime(df_all['timestamp'])
        min_date = df_all['timestamp'].min().date()
        max_date = datetime.now().date()
    else:
        min_date = datetime.now().date() - timedelta(days=30)
        max_date = datetime.now().date()
    
    # Calculate default start date without going before min_date
    default_start = max(min_date, max_date - timedelta(days=7))
    
    date_range = st.sidebar.date_input(
        "Date Range",
        value=(default_start, max_date),
        min_value=min_date,
        max_value=max_date
    )
    
    # Load data with filters
    df = load_gpu_metrics(
        limit=1000,
        date_range=date_range if len(date_range) == 2 else None
    )
    
    if df is not None and not df.empty:
        # Extract numeric values from GPU temperature if needed
        if 'gpu_temperature' in df.columns and 'gpu_temp_value' not in df.columns:
            try:
                # Extract temperature values
                df['gpu_temp_value'] = df['gpu_temperature'].str.extract(r'(\d+)').astype(float)
            except Exception:
                # Fallback method if regex fails
                df['gpu_temp_value'] = df['gpu_temperature'].apply(
                    lambda x: float(''.join(c for c in str(x) if c.isdigit() or c == '.')) if pd.notna(x) else None
                )
        
        # Continue only if we have temperature data
        if 'gpu_temp_value' in df.columns and df['gpu_temp_value'].notna().any():
            # Calculate temperature metrics
            temp_metrics = calculate_temperature_metrics(df)
            
            # Temperature status
            latest_temp_status = get_temperature_status(temp_metrics.get("latest_temp"))
            max_temp_status = get_temperature_status(temp_metrics.get("max_temp"))
            
            # Temperature cards
            st.markdown('<div class="sub-header">GPU Temperature Overview</div>', unsafe_allow_html=True)
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown(
                    f"""
                    <div class="metric-card">
                        <div class="metric-value {get_temperature_class(latest_temp_status)}">
                            {temp_metrics.get("latest_temp", "N/A"):.1f}°C
                        </div>
                        <div class="metric-label">Current Temperature</div>
                    </div>
                    """, 
                    unsafe_allow_html=True
                )
            
            with col2:
                st.markdown(
                    f"""
                    <div class="metric-card">
                        <div class="metric-value {get_temperature_class(max_temp_status)}">
                            {temp_metrics.get("max_temp", "N/A"):.1f}°C
                        </div>
                        <div class="metric-label">Maximum Temperature</div>
                    </div>
                    """, 
                    unsafe_allow_html=True
                )
                
            with col3:
                st.markdown(
                    f"""
                    <div class="metric-card">
                        <div class="metric-value">
                            {temp_metrics.get("avg_temp", "N/A"):.1f}°C
                        </div>
                        <div class="metric-label">Average Temperature</div>
                    </div>
                    """, 
                    unsafe_allow_html=True
                )
            
            # Temperature plot
            temp_fig = create_temperature_plot(df)
            if temp_fig:
                st.plotly_chart(temp_fig, use_container_width=True)
            
            # Temperature analysis
            st.markdown('<div class="sub-header">Temperature Analysis</div>', unsafe_allow_html=True)
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Temperature Metrics:**")
                if temp_metrics.get("max_temp_time"):
                    max_temp_time_str = temp_metrics["max_temp_time"].strftime("%Y-%m-%d %H:%M:%S")
                    st.markdown(f"• Maximum temperature reached at: **{max_temp_time_str}**")
                
                if temp_metrics.get("time_above_warning") is not None:
                    st.markdown(f"• Time spent above 75°C: **{temp_metrics['time_above_warning']}** data points")
                
                if temp_metrics.get("time_above_critical") is not None:
                    st.markdown(f"• Time spent above 85°C: **{temp_metrics['time_above_critical']}** data points")
            
            with col2:
                st.markdown("**Temperature Trend:**")
                if temp_metrics.get("temp_trend") is not None:
                    trend_direction = "increasing" if temp_metrics["temp_trend"] > 0 else "decreasing"
                    st.markdown(f"• Recent trend: **{trend_direction}** by {abs(temp_metrics['temp_trend']):.1f}°C")
                
                # Temperature recommendations
                st.markdown("**Recommendations:**")
                if latest_temp_status == "critical":
                    st.markdown("• ⚠️ **CRITICAL: GPU temperature is dangerously high!** Consider stopping training immediately and checking cooling.")
                elif latest_temp_status == "warning":
                    st.markdown("• ⚠️ **WARNING: GPU temperature is approaching unsafe levels.** Consider lowering batch size or checking cooling.")
                else:
                    st.markdown("• ✅ GPU temperature is within normal operating range.")
            
            # Utilization and memory plot
            util_fig = create_utilization_plot(df)
            if util_fig:
                st.markdown('<div class="sub-header">GPU Utilization & Memory</div>', unsafe_allow_html=True)
                st.plotly_chart(util_fig, use_container_width=True)
                
                # Calculate utilization and memory metrics
                if 'gpu_util_value' in df.columns and df['gpu_util_value'].notna().any():
                    avg_util = df['gpu_util_value'].mean()
                    max_util = df['gpu_util_value'].max()
                    current_util = df['gpu_util_value'].iloc[-1]
                    
                    # Determine if utilization is appropriate for the model architecture
                    # Theta model has been modified to be smaller (768 hidden size, 12 layers)
                    # For ML training, 100% utilization is normal and desired
                    if max_util < 30:
                        util_status = "low"
                        util_message = "GPU is underutilized. Consider increasing batch size to improve training efficiency."
                    elif max_util > 90:
                        util_status = "optimal"
                        util_message = "GPU is fully saturated - this is normal and optimal for ML training."
                    else:
                        util_status = "good"
                        util_message = "GPU utilization is good for the current model architecture."
                    
                    # Add utilization analysis section
                    st.markdown('<div class="sub-header">Utilization & Memory Analysis</div>', unsafe_allow_html=True)
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("**Utilization Metrics:**")
                        st.markdown(f"• Average utilization: **{avg_util:.1f}%**")
                        st.markdown(f"• Maximum utilization: **{max_util:.1f}%**")
                        st.markdown(f"• Current utilization: **{current_util:.1f}%**")
                        
                        # Memory metrics if available
                        if 'gpu_memory_percent' in df.columns and df['gpu_memory_percent'].notna().any():
                            avg_mem = df['gpu_memory_percent'].mean()
                            max_mem = df['gpu_memory_percent'].max()
                            current_mem = df['gpu_memory_percent'].iloc[-1]
                            st.markdown(f"• Memory usage: **{current_mem:.1f}%** (Max: {max_mem:.1f}%)")
                    
                    with col2:
                        st.markdown("**Utilization Analysis:**")
                        if util_status == "low":
                            st.markdown(f"• ℹ️ **GPU is underutilized at {current_util:.1f}%**")
                        elif util_status == "optimal":
                            st.markdown(f"• ✅ **GPU is optimally utilized at {current_util:.1f}%**")
                        else:
                            st.markdown(f"• ✅ **GPU utilization is good at {current_util:.1f}%**")
                        
                        # Recommendations based on current architecture
                        st.markdown("**Recommendations:**")
                        st.markdown(f"• {util_message}")
                        
                        # For the recently modified model architecture
                        st.markdown(f"• Your model with reduced complexity (768 hidden size, 12 layers) should maintain good efficiency with appropriate batch sizes.")
                        
                        # Memory recommendations if available
                        if 'gpu_memory_percent' in df.columns and current_mem > 98:
                            st.markdown(f"• Memory usage is very high at **{current_mem:.1f}%** - this is normal for ML training but close to limit.")
                        elif 'gpu_memory_percent' in df.columns and current_mem > 90:
                            st.markdown(f"• ✅ Memory usage is high at **{current_mem:.1f}%** - this is optimal for ML training efficiency.")
                        elif 'gpu_memory_percent' in df.columns:
                            st.markdown(f"• Memory usage is at **{current_mem:.1f}%**, which is good for current training parameters.")
        else:
            st.warning("No processed GPU temperature data available. The dashboard needs to extract numeric values from the 'gpu_temperature' column.")
        
        # Data table
        st.markdown('<div class="sub-header">Raw GPU Data</div>', unsafe_allow_html=True)
        
        # Select columns for display
        display_cols = ['timestamp', 'epoch_number', 'model_version']
        if 'gpu_temp_value' in df.columns:
            display_cols.append('gpu_temp_value')
        if 'max_gpu_temp' in df.columns:
            display_cols.append('max_gpu_temp')
        if 'gpu_memory_percent' in df.columns:
            display_cols.append('gpu_memory_percent')
        if 'gpu_util_value' in df.columns:
            display_cols.append('gpu_util_value')
        
        # Create a display DataFrame with renamed columns
        display_df = df[display_cols].copy()
        display_df.columns = [
            'Timestamp', 'Epoch', 'Model Version', 
            'Temperature (°C)', 'Max Temperature (°C)',
            'Memory Usage (%)', 'GPU Utilization (%)' 
        ][:len(display_cols)]
        
        st.dataframe(display_df.sort_values('Timestamp', ascending=False))
        
    else:
        st.warning("No GPU temperature data available. Please check that your training script is capturing GPU metrics.")

if __name__ == "__main__":
    main()
