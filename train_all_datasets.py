"""
Theta AI - Comprehensive Training Script

This script trains the Theta model on all available datasets and overrides the final model.
It combines multiple training data sources including:
- General knowledge questions
- Code completion samples
- State capitals
- Smart-ass conversational data
- Any other datasets in the data folder
"""

import os
import argparse
import torch
import logging
import glob
import json
import random
import smtplib
import subprocess
import time
from datetime import datetime
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.utils import formatdate
from dotenv import load_dotenv
from torch.optim import AdamW
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from torch.utils.data import DataLoader, Dataset, RandomSampler
from tqdm import tqdm
from string import Template

# Import local modules
from model import ThetaModel, ThetaConfig
from database import ConversationDatabase

# Initialize the database
db = ConversationDatabase()

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Default paths
MODEL_DIR = "models"
FINAL_MODEL_PATH = os.path.join(MODEL_DIR, "final", "final_model")

# Email notification settings
EMAIL_ENABLED = True
EMAIL_ADDRESS = os.getenv('EMAIL_ADDRESS')
EMAIL_APP_PASSWORD = os.getenv('EMAIL_APP_PASSWORD')
EMAIL_SMTP_SERVER = 'smtp.gmail.com'
EMAIL_SMTP_PORT = 587


def get_gpu_temperature():
    """Get the current GPU temperature using nvidia-smi."""
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=temperature.gpu', '--format=csv,noheader,nounits'],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True
        )
        temp = result.stdout.strip()
        return int(temp) if temp.isdigit() else 0  # Return numeric value for comparisons
    except (subprocess.SubprocessError, FileNotFoundError):
        logger.warning("Could not get GPU temperature. nvidia-smi may not be available.")
        return 0
        
def get_cpu_temperature():
    """Get the current CPU temperature."""
    try:
        if os.name == 'nt':  # Windows
            # Use wmic for Windows systems
            result = subprocess.run(
                ['wmic', 'temperature', 'get', 'currenttemperature'],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=True
            )
            # Parse the output - typically second line has the value
            lines = result.stdout.strip().split('\n')
            if len(lines) > 1:
                temp = lines[1].strip()
                return int(temp) if temp.isdigit() else 0
            return 0
        else:  # Linux and others
            # Try sensors command for Linux systems
            result = subprocess.run(
                ['sensors'],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=True
            )
            # Parse output - look for Core temp readings
            for line in result.stdout.split('\n'):
                if 'Core' in line and '°C' in line:
                    # Extract temperature value
                    temp_match = re.search(r'\+([\d\.]+)°C', line)
                    if temp_match:
                        return int(float(temp_match.group(1)))
            return 0
    except (subprocess.SubprocessError, FileNotFoundError):
        logger.warning("Could not get CPU temperature.")
        return 0
        
def get_cpu_utilization():
    """Get the current CPU utilization percentage."""
    try:
        if os.name == 'nt':  # Windows
            # Use wmic for Windows
            result = subprocess.run(
                ['wmic', 'cpu', 'get', 'loadpercentage'],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=True
            )
            # Parse the output - typically second line has the value
            lines = result.stdout.strip().split('\n')
            if len(lines) > 1:
                util = lines[1].strip()
                return float(util) if util.replace('.', '', 1).isdigit() else 0.0
            return 0.0
        else:  # Linux and others
            # Use top command in batch mode
            result = subprocess.run(
                ['top', '-bn1'],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=True
            )
            # Parse output - look for Cpu(s) line
            for line in result.stdout.split('\n'):
                if line.startswith('%Cpu(s):'):
                    # Extract CPU usage percentage
                    parts = line.split(',')
                    for part in parts:
                        if 'us' in part:  # user space usage
                            usage = float(part.replace('us', '').strip())
                            return usage
            return 0.0
    except (subprocess.SubprocessError, FileNotFoundError, ValueError):
        logger.warning("Could not get CPU utilization.")
        return 0.0


def create_html_email(subject, content_blocks, gpu_temp=None):
    """Create a stylish HTML email using the external template file."""
    # Add GPU temperature to the content blocks if provided
    if gpu_temp:
        content_blocks.append(("Current GPU Temperature", gpu_temp))
    
    # Load the HTML template
    template_path = os.path.join(os.path.dirname(__file__), 'email_templates', 'theta_email_template.html')
    with open(template_path, 'r') as template_file:
        template_content = template_file.read()
    
    # Generate the content blocks HTML
    content_blocks_html = ""
    for label, value in content_blocks:
        content_blocks_html += f"""
            <div class="data-row">
                <div class="data-label">{label}</div>
                <div class="data-value">{value}</div>
            </div>
        """
    
    # Get current timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Replace template placeholders
    html = template_content.replace('{{subject}}', subject)
    html = html.replace('{{content_blocks}}', content_blocks_html)
    html = html.replace('{{timestamp}}', timestamp)
    
    return html

def send_email_notification(subject, content_blocks):
    """Send an email notification with formatted content blocks using the Theta email template.
    
    Args:
        subject (str): Email subject
        content_blocks (list): List of tuples with (label, value) for each section of content
    
    Returns:
        bool: True if email was sent successfully, False otherwise
    """
    try:
        # Check if email configuration is available
        if not all([EMAIL_ADDRESS, EMAIL_APP_PASSWORD, EMAIL_SMTP_SERVER, EMAIL_SMTP_PORT]):
            logger.warning("Email configuration not complete. Skipping email notification.")
            return False
        
        # Create message container - the correct MIME type is multipart/alternative
        msg = MIMEMultipart('alternative')
        msg['From'] = EMAIL_ADDRESS
        msg['To'] = EMAIL_ADDRESS
        msg['Subject'] = subject
        msg['Date'] = formatdate(localtime=True)
        
        # Load the email template
        template_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                                    "email_templates", "theta_email_template.html")
        
        try:
            with open(template_path, 'r', encoding='utf-8') as f:
                template_content = f.read()
        except Exception as e:
            logger.error(f"Failed to load email template: {e}")
            # Fallback to basic template if the file can't be loaded
            template_content = """<!DOCTYPE html>
<html>
<head>
    <title>{{subject}}</title>
</head>
<body>
    <h1>{{subject}}</h1>
    <div>{{content_blocks}}</div>
    <div>Generated: {{timestamp}}</div>
</body>
</html>"""
            
        # Format the content blocks into HTML
        html_blocks = []
        plain_blocks = []
        
        for label, value in content_blocks:
            # Clean and format the value
            if isinstance(value, (int, float)):
                value = f"{value}"
            else:
                value = str(value) if value is not None else "N/A"
            
            # Create HTML version
            html_blocks.append(f"""<div class="data-row">
            <div class="data-label">{label}</div>
            <div class="data-value">{value}</div>
        </div>""")
            
            # Create plain text version
            plain_blocks.append(f"{label}: {value}")
        
        # Combine the blocks
        html_content_blocks = "\n".join(html_blocks)
        plain_content = "\n\n".join(plain_blocks)
        
        # Create the email with the template
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Replace template placeholders
        html_content = template_content
        html_content = html_content.replace("{{subject}}", subject)
        html_content = html_content.replace("{{content_blocks}}", html_content_blocks)
        html_content = html_content.replace("{{timestamp}}", timestamp)
        
        # The email client will try to render the last part first
        # So we attach the plain text first, HTML last
        plain_part = MIMEText(plain_content, 'plain')
        html_part = MIMEText(html_content, 'html')
        
        # Attach parts into message container
        msg.attach(plain_part)
        msg.attach(html_part)  # Attach HTML last so it's preferred

        server = smtplib.SMTP(EMAIL_SMTP_SERVER, EMAIL_SMTP_PORT)
        server.starttls()
        server.login(EMAIL_ADDRESS, EMAIL_APP_PASSWORD)
        server.send_message(msg)
        server.quit()

        logger.info(f"Email notification sent: {subject}")
        return True
    except Exception as e:
        logger.error(f"Failed to send email: {e}")
        return False

class ThetaDataset(Dataset):
    """Custom dataset for Theta training data."""
    
    def __init__(self, tokenizer, data_pairs, max_length=512):
        self.tokenizer = tokenizer
        self.data_pairs = data_pairs
        self.max_length = max_length
    
    def __len__(self):
        return len(self.data_pairs)
    
    def __getitem__(self, idx):
        question, answer = self.data_pairs[idx]
        
        # Format as conversation
        input_text = f"User: {question}\nTheta:"
        target_text = answer
        
        # Tokenize inputs and targets in a single call
        encodings = self.tokenizer(
            input_text, 
            text_target=target_text,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        input_ids = encodings["input_ids"].squeeze()
        attention_mask = encodings["attention_mask"].squeeze()
        labels = encodings["labels"].squeeze()
        
        # Replace padding token id with -100 for labels (to ignore in loss)
        labels[labels == self.tokenizer.pad_token_id] = -100
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }

def load_qa_data(file_path):
    """Load question-answer pairs from a text file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    qa_pairs = []
    sections = content.split('\n\n')
    
    for section in sections:
        if not section.strip():
            continue
        
        lines = section.split('\n')
        question = None
        answer = None
        
        for line in lines:
            if line.startswith('Q:'):
                question = line[2:].strip()
            elif line.startswith('A:'):
                answer = line[2:].strip()
        
        if question and answer:
            qa_pairs.append((question, answer))
    
    return qa_pairs

def load_all_datasets():
    """Load all datasets from the data directory in curriculum learning order."""
    logger.info("Loading datasets in curriculum learning order...")
    
    # Organize dataset files in curriculum order (from basic to complex)
    # This staged loading helps the model learn fundamentals first
    curriculum_stages = [
        # Stage 1: Foundation datasets (basic sentence structure and coherence)
        [
            os.path.join("data", "foundation_sentences.txt"),
            os.path.join("data", "state_capitals.txt"),
        ],
        # Stage 2: Question-answer and dialogue datasets
        [
            os.path.join("data", "question_answer_pairs.txt"),
            os.path.join("data", "dialogue_progressions.txt"),
        ],
        # Stage 3: Grounded and specific expertise datasets
        [
            os.path.join("data", "grounded_responses.txt"),
            os.path.join("data", "general_knowledge.txt"),
            os.path.join("data", "code_completions.txt")
        ],
        # Stage 4: Personality and style datasets
        [
            os.path.join("data", "smartass_style_examples.txt"),
            os.path.join("data", "smartass_conversations.txt")
        ]
    ]
    
    # Flatten the curriculum stages into a single list of dataset files
    dataset_files = [file for stage in curriculum_stages for file in stage]
    
    # Check for any additional datasets not explicitly listed
    for additional_file in glob.glob(os.path.join("data", "*.txt")):
        if additional_file not in dataset_files and "README" not in additional_file:
            dataset_files.append(additional_file)
    
    all_qa_pairs = []
    dataset_counts = {}
    
    # Load each dataset
    for file_path in dataset_files:
        if os.path.exists(file_path):
            try:
                qa_pairs = load_qa_data(file_path)
                all_qa_pairs.extend(qa_pairs)
                dataset_name = os.path.basename(file_path)
                dataset_counts[dataset_name] = len(qa_pairs)
                logger.info(f"Loaded {len(qa_pairs)} QA pairs from {file_path}")
            except Exception as e:
                logger.warning(f"Error loading dataset {file_path}: {e}")
        else:
            logger.warning(f"Dataset file not found: {file_path}")
    
    # Also load any conversation data from the database
    try:
        logger.info("Loading conversation history from database...")
        db_pairs = db.get_training_pairs(limit=500)  # Get up to 500 pairs
        if db_pairs:
            dataset_counts["database_conversations"] = len(db_pairs)
            logger.info(f"Loaded {len(db_pairs)} conversation pairs from database")
            all_qa_pairs.extend(db_pairs)
    except Exception as e:
        logger.warning(f"Could not load database conversations: {e}")
    
    # Log dataset composition
    total_pairs = len(all_qa_pairs)
    logger.info(f"\nDataset composition:")
    for dataset_name, count in dataset_counts.items():
        percentage = (count / total_pairs) * 100 if total_pairs > 0 else 0
        logger.info(f"  {dataset_name}: {count} pairs ({percentage:.1f}%)")
        
    # Shuffle the combined dataset
    random.shuffle(all_qa_pairs)
    
    logger.info(f"Total training examples: {len(all_qa_pairs)}")
    return all_qa_pairs

def send_gpu_temperature_update():
    """Send an email with the current GPU temperature."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    gpu_temp = get_gpu_temperature()
    memory_usage = "N/A"
    gpu_utilization = "N/A"
    
    # Try to get GPU memory usage and utilization
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=memory.used,memory.total,utilization.gpu', '--format=csv,noheader,nounits'],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True
        )
        info = result.stdout.strip().split(',')
        if len(info) >= 2:
            memory_usage = f"{info[0].strip()} MB / {info[1].strip()} MB"
        if len(info) >= 3:
            gpu_utilization = f"{info[2].strip()}%"
    except (subprocess.SubprocessError, FileNotFoundError):
        pass
    
    subject = f"Theta AI - GPU Status Update"
    content_blocks = [
        ("Timestamp", timestamp),
        ("GPU Temperature", gpu_temp),
        ("GPU Memory Usage", memory_usage),
        ("GPU Utilization", gpu_utilization)
    ]
    
    html_content = create_html_email(subject, content_blocks)
    send_email_update(subject, html_content, is_html_content=True)
    logger.info(f"Sent GPU temperature update: {gpu_temp}")


def train_model(args):
    """Train the model on all datasets."""
    # Load all datasets
    data_pairs = load_all_datasets()
    
    logger.info(f"Total training examples: {len(data_pairs)}")
    
    # Send email notification about training start
    start_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    gpu_info = "GPU: " + torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU training"
    gpu_temp = get_gpu_temperature()
    
    # Try to get GPU memory and utilization info
    memory_usage = "N/A"
    gpu_utilization = "N/A"
    if torch.cuda.is_available():
        try:
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=memory.used,memory.total,utilization.gpu', '--format=csv,noheader,nounits'],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=True
            )
            info = result.stdout.strip().split(',')
            if len(info) >= 2:
                memory_usage = f"{info[0].strip()} MB / {info[1].strip()} MB"
            if len(info) >= 3:
                gpu_utilization = f"{info[2].strip()}%"
        except (subprocess.SubprocessError, FileNotFoundError):
            pass
    
    # Calculate estimated training time based on dataset size and epochs
    # This is a rough estimate that will be refined after the first epoch
    steps_per_epoch = len(data_pairs) // args.batch_size
    estimated_seconds_per_step = 0.5  # Default estimate for mid-range GPU
    
    # Adjust estimate based on known GPU
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0).lower()
        if "3090" in gpu_name or "a100" in gpu_name:
            estimated_seconds_per_step = 0.3  # Faster GPU
        elif "3060" in gpu_name:
            estimated_seconds_per_step = 0.5  # Mid-range GPU
        elif "1060" in gpu_name or "1650" in gpu_name:
            estimated_seconds_per_step = 0.8  # Older GPU
    else:
        # CPU training is much slower
        estimated_seconds_per_step = 2.0
    
    estimated_seconds_per_epoch = steps_per_epoch * estimated_seconds_per_step
    total_estimated_seconds = estimated_seconds_per_epoch * args.num_epochs
    
    estimated_hours = int(total_estimated_seconds // 3600)
    estimated_minutes = int((total_estimated_seconds % 3600) // 60)
    
    estimated_time = f"{estimated_hours}h {estimated_minutes}m (rough estimate)"
    
    content_blocks = [
        ("Time", start_time),
        ("Total examples", str(len(data_pairs))),
        ("Training device", gpu_info),
        ("GPU Temperature", gpu_temp),
        ("GPU Memory", memory_usage),
        ("GPU Utilization", gpu_utilization),
        ("Epochs", str(args.num_epochs)),
        ("Batch size", str(args.batch_size)),
        ("Learning rate", str(args.learning_rate)),
        ("Estimated training time", estimated_time)
    ]
    
    html_content = create_html_email("Theta AI Training Started", content_blocks)
    send_email_update("Theta AI Training Started", html_content, is_html_content=True)
    
    # Set device - use GPU by default
    use_gpu = torch.cuda.is_available() and not args.use_cpu
    device = torch.device("cuda" if use_gpu else "cpu")
    logger.info(f"Using device: {device} {'(GPU)' if use_gpu else '(CPU)'}")
    
    # Warn if CPU is being used
    if not use_gpu:
        if args.use_cpu:
            logger.warning("Using CPU as requested by --use_cpu flag")
        elif not torch.cuda.is_available():
            logger.warning("No CUDA-capable GPU detected. Using CPU instead.")
        logger.warning("Training on CPU will be significantly slower!")
    
    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Set default model path to final model if not specified
    if not args.model_path:
        args.model_path = FINAL_MODEL_PATH
        logger.info(f"No model path specified, will attempt to use final model at {FINAL_MODEL_PATH}")
    
    # Load or create model
    if os.path.exists(args.model_path):
        try:
            logger.info(f"Loading model from {args.model_path}")
            config_path = os.path.join(args.model_path, "config.json")
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    config_dict = json.load(f)
                config = ThetaConfig(**config_dict)
            else:
                logger.info("No config.json found, checking for theta_config.json")
                config_path = os.path.join(args.model_path, "theta_config.json")
                if os.path.exists(config_path):
                    with open(config_path, 'r') as f:
                        config_dict = json.load(f)
                    config = ThetaConfig(**config_dict)
                else:
                    config = ThetaConfig()
            
            model = ThetaModel(config)
            model_weights = os.path.join(args.model_path, "pytorch_model.bin")
            if os.path.exists(model_weights):
                logger.info(f"Loading weights from {model_weights}")
                model.load_state_dict(torch.load(model_weights, map_location=device))
                logger.info("Successfully loaded existing model")
            else:
                logger.warning(f"No weights file found at {model_weights}")
                logger.info("Using model configuration but initializing new weights")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            logger.info("Initializing a new model instead")
            config = ThetaConfig()
            model = ThetaModel(config)
    else:
        logger.info("Model path does not exist, initializing a new model")
        config = ThetaConfig()
        model = ThetaModel(config)
    
    model.to(device)
    
    # Create train/validation split (90/10)
    random.shuffle(data_pairs)
    split_idx = int(len(data_pairs) * 0.9)
    train_pairs = data_pairs[:split_idx]
    val_pairs = data_pairs[split_idx:]
    
    logger.info(f"Training on {len(train_pairs)} examples, validating on {len(val_pairs)} examples")
    
    # Create train dataset and dataloader
    train_dataset = ThetaDataset(tokenizer, train_pairs, max_length=args.max_length)
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(
        train_dataset,
        sampler=train_sampler,
        batch_size=args.batch_size,
        num_workers=0
    )
    
    # Create validation dataset and dataloader
    val_dataset = ThetaDataset(tokenizer, val_pairs, max_length=args.max_length)
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        num_workers=0
    )
    
    # Set up optimizer with weight decay for regularization
    # Apply weight decay to all parameters except bias and layer norm
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {
            'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            'weight_decay': 0.01
        },
        {
            'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            'weight_decay': 0.0
        }
    ]
    
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate)
    
    # Learning rate scheduler with longer warmup
    total_steps = len(train_dataloader) * args.num_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * 0.15),  # 15% warmup
        num_training_steps=total_steps
    )
    
    # Training loop
    logger.info("Starting training...")
    torch.cuda.empty_cache()
    model.train()
    
    training_start_time = time.time()
    last_temp_update_time = training_start_time
    epoch_start_times = []
    for epoch in range(args.num_epochs):
        epoch_start_time = time.time()
        epoch_start_times.append(epoch_start_time)
        model.train()
        epoch_loss = 0
        
        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{args.num_epochs}")
        
        for batch in progress_bar:
            # Move batch to device
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                return_dict=True
            )
            
            loss = outputs["loss"]
            
            # Backward pass
            loss.backward()
            
            # Update weights
            optimizer.step()
            scheduler.step()
            
            # Update metrics
            epoch_loss += loss.item()
            progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})
            
            # Send GPU temperature update every 15 minutes
            current_time = time.time()
            if current_time - last_temp_update_time >= 900:  # 900 seconds = 15 minutes
                # Pass current epoch and model version to save to database
                model_version = os.path.basename(args.output_dir)
                send_gpu_temperature_update(current_epoch=epoch+1, model_version=model_version)
                last_temp_update_time = current_time
        
        # Evaluate on validation set
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for val_batch in val_dataloader:
                val_input_ids = val_batch["input_ids"].to(device)
                val_attention_mask = val_batch["attention_mask"].to(device)
                val_labels = val_batch["labels"].to(device)
                
                outputs = model(
                    input_ids=val_input_ids,
                    attention_mask=val_attention_mask,
                    labels=val_labels,
                    return_dict=True
                )
                
                # Access loss from the dictionary returned by the model
                val_loss += outputs["loss"].item()
        
        # Calculate epoch statistics
        epoch_loss = epoch_loss / len(train_dataloader)
        val_loss = val_loss / len(val_dataloader)
        epoch_duration = time.time() - epoch_start_time
        epoch_duration_mins = int(epoch_duration // 60)
        epoch_duration_secs = int(epoch_duration % 60)
        
        logger.info(f"Epoch {epoch+1}/{args.num_epochs} - Train Loss: {epoch_loss:.6f} - Val Loss: {val_loss:.6f} - Time: {epoch_duration_mins}m {epoch_duration_secs}s")
        
        # Get GPU information for metrics database
        gpu_temp = get_gpu_temperature()
        memory_usage = "N/A"
        gpu_utilization = "N/A"
        
        if torch.cuda.is_available():
            try:
                result = subprocess.run(
                    ['nvidia-smi', '--query-gpu=memory.used,memory.total,utilization.gpu', '--format=csv,noheader,nounits'],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    check=True
                )
                info = result.stdout.strip().split(',')
                if len(info) >= 2:
                    memory_usage = f"{info[0].strip()} MB / {info[1].strip()} MB"
                if len(info) >= 3:
                    gpu_utilization = f"{info[2].strip()}%"
            except (subprocess.SubprocessError, FileNotFoundError):
                pass
                
        # Save metrics to database
        try:
            model_version = os.path.basename(args.output_dir)
            db.save_training_metrics(
                epoch_number=epoch+1,
                train_loss=epoch_loss,
                validation_loss=val_loss,
                model_version=model_version,
                dataset_size=len(data_pairs),
                learning_rate=args.learning_rate,
                notes=f"Duration: {epoch_duration_mins}m {epoch_duration_secs}s",
                gpu_temperature=gpu_temp,
                gpu_memory_usage=memory_usage,
                gpu_utilization=gpu_utilization
            )
            logger.info(f"Saved epoch {epoch+1} metrics to database with GPU info")
        except Exception as e:
            logger.error(f"Failed to save training metrics to database: {e}")
        
        # Get GPU information for the epoch email
        gpu_temp = get_gpu_temperature()
        memory_usage = "N/A"
        gpu_utilization = "N/A"
        
        if torch.cuda.is_available():
            try:
                result = subprocess.run(
                    ['nvidia-smi', '--query-gpu=memory.used,memory.total,utilization.gpu', '--format=csv,noheader,nounits'],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    check=True
                )
                info = result.stdout.strip().split(',')
                if len(info) >= 2:
                    memory_usage = f"{info[0].strip()} MB / {info[1].strip()} MB"
                if len(info) >= 3:
                    gpu_utilization = f"{info[2].strip()}%"
            except (subprocess.SubprocessError, FileNotFoundError):
                pass
                
        # Create styled HTML email for epoch completion
        epoch_title = f"Theta AI Training - Epoch {epoch+1}/{args.num_epochs}"
        estimated_remaining_mins = epoch_duration_mins * (args.num_epochs - epoch - 1)
        estimated_remaining_hours = estimated_remaining_mins // 60
        estimated_remaining_mins = estimated_remaining_mins % 60
        
        content_blocks = [
            ("Epoch", f"{epoch+1}/{args.num_epochs}"),
            ("Training Loss", f"{epoch_loss:.6f}"),
            ("Validation Loss", f"{val_loss:.6f}"),
            ("Duration", f"{epoch_duration_mins}m {epoch_duration_secs}s"),
            ("Estimated Remaining", f"{estimated_remaining_hours}h {estimated_remaining_mins}m"),
            ("GPU Temperature", gpu_temp),
            ("GPU Memory Usage", memory_usage),
            ("GPU Utilization", gpu_utilization)
        ]
        
        html_content = create_html_email(epoch_title, content_blocks)
        send_email_update(epoch_title, html_content, is_html_content=True)
        
        # Save checkpoint after each epoch
        checkpoint_dir = os.path.join(args.output_dir, f"checkpoint-epoch-{epoch+1}")
        os.makedirs(checkpoint_dir, exist_ok=True)
        torch.save(model.state_dict(), os.path.join(checkpoint_dir, "pytorch_model.bin"))
        if hasattr(model, 'config'):
            with open(os.path.join(checkpoint_dir, "config.json"), 'w') as f:
                json.dump(model.config.to_dict(), f)
        tokenizer.save_pretrained(checkpoint_dir)
        logger.info(f"Saved checkpoint to {checkpoint_dir}")
    
    # Save final model
    logger.info("Training complete, saving final model...")
    save_final_model(model, tokenizer)
    
    # Send email notification about training completion
    end_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    training_duration = time.time() - epoch_start_times[0] if epoch_start_times else 0
    hours = int(training_duration // 3600)
    minutes = int((training_duration % 3600) // 60)
    
    # Get GPU information for the completion email
    gpu_temp = get_gpu_temperature()
    memory_usage = "N/A"
    gpu_utilization = "N/A"
    gpu_info = "N/A"
    
    if torch.cuda.is_available():
        gpu_info = torch.cuda.get_device_name(0)
        try:
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=memory.used,memory.total,utilization.gpu', '--format=csv,noheader,nounits'],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=True
            )
            info = result.stdout.strip().split(',')
            if len(info) >= 2:
                memory_usage = f"{info[0].strip()} MB / {info[1].strip()} MB"
            if len(info) >= 3:
                gpu_utilization = f"{info[2].strip()}%"
        except (subprocess.SubprocessError, FileNotFoundError):
            pass
    
    content_blocks = [
        ("End Time", end_time),
        ("Final Loss", f"{epoch_loss:.6f}"),
        ("Total Duration", f"{hours}h {minutes}m"),
        ("GPU Used", gpu_info),
        ("Final GPU Temperature", gpu_temp),
        ("Final GPU Memory Usage", memory_usage),
        ("Final GPU Utilization", gpu_utilization),
        ("Model saved to", FINAL_MODEL_PATH)
    ]
    
    html_content = create_html_email("Theta AI Training Completed", content_blocks)
    send_email_update("Theta AI Training Completed", html_content, is_html_content=True)
    
    return model, tokenizer

def save_final_model(model, tokenizer):
    """Save the final model, overriding the existing one if present."""
    # Create directory for final model
    final_dir = os.path.dirname(FINAL_MODEL_PATH)
    os.makedirs(final_dir, exist_ok=True)
    
    # Create backup of existing model if it exists
    if os.path.exists(FINAL_MODEL_PATH):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = os.path.join(MODEL_DIR, "backups", f"theta_backup_{timestamp}")
        os.makedirs(backup_path, exist_ok=True)
        
        # If there's an existing model file, back it up
        model_file = os.path.join(FINAL_MODEL_PATH, "pytorch_model.bin")
        if os.path.exists(model_file):
            logger.info(f"Backing up existing model to {backup_path}")
            torch.save(torch.load(model_file), os.path.join(backup_path, "pytorch_model.bin"))
        
        # Backup config if it exists
        config_file = os.path.join(FINAL_MODEL_PATH, "config.json")
        if os.path.exists(config_file):
            with open(config_file, 'r') as src, open(os.path.join(backup_path, "config.json"), 'w') as dst:
                dst.write(src.read())
    
    # Save new model
    os.makedirs(FINAL_MODEL_PATH, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(FINAL_MODEL_PATH, "pytorch_model.bin"))
    
    # Save config
    if hasattr(model, 'config'):
        with open(os.path.join(FINAL_MODEL_PATH, "config.json"), 'w') as f:
            json.dump(model.config.to_dict(), f)
    
    # Save tokenizer
    tokenizer.save_pretrained(FINAL_MODEL_PATH)
    
    logger.info(f"Final model saved to {FINAL_MODEL_PATH}")

def send_system_metrics_update(current_epoch=None, model_version=None):
    """Send an email with GPU and CPU metrics and save to database.
    This function is designed to be called both during training and periodically (every 15 min).
    It also sends warning emails if temperatures exceed specified thresholds.
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Define temperature thresholds for warnings
    GPU_WARNING_THRESHOLD = 75  # Celsius
    CPU_WARNING_THRESHOLD = 70  # Celsius
    
    # Static variables to track highest temperatures and utilization
    if not hasattr(send_system_metrics_update, "highest_gpu_temp"):
        send_system_metrics_update.highest_gpu_temp = 0
    if not hasattr(send_system_metrics_update, "highest_cpu_temp"):
        send_system_metrics_update.highest_cpu_temp = 0
    if not hasattr(send_system_metrics_update, "highest_gpu_util"):
        send_system_metrics_update.highest_gpu_util = 0.0
    if not hasattr(send_system_metrics_update, "highest_cpu_util"):
        send_system_metrics_update.highest_cpu_util = 0.0
    
    # Get GPU temperature (now returns numeric value)
    gpu_temp = get_gpu_temperature()
    gpu_temp_display = f"{gpu_temp}°C"
    
    # Get CPU temperature and utilization
    cpu_temp = get_cpu_temperature()
    cpu_util = get_cpu_utilization()
    cpu_temp_display = f"{cpu_temp}°C" if cpu_temp > 0 else "N/A"
    cpu_util_display = f"{cpu_util:.1f}%" if cpu_util > 0 else "N/A"
    
    # Update highest values
    if gpu_temp > send_system_metrics_update.highest_gpu_temp:
        send_system_metrics_update.highest_gpu_temp = gpu_temp
    if cpu_temp > send_system_metrics_update.highest_cpu_temp:
        send_system_metrics_update.highest_cpu_temp = cpu_temp
    
    period_note = None
    
    # Get GPU memory usage and utilization
    memory_usage = None
    gpu_utilization = None
    
    try:
        if NVIDIA_SMI_AVAILABLE:
            # Execute the nvidia-smi command to get memory usage and utilization
            cmd = "nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits"
            result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, shell=True)
            
            # Parse the output
            if result.returncode == 0:
                info = result.stdout.strip().split(',')
                if len(info) >= 2:
                    memory_used = float(info[0].strip())
                    memory_total = float(info[1].strip())
                    memory_usage = (memory_used / memory_total) * 100  # Convert to percentage
                    
                if len(info) >= 3:
                    gpu_utilization = float(info[2].strip())
                    
                    # Update highest GPU utilization
                    if gpu_utilization > send_system_metrics_update.highest_gpu_util:
                        send_system_metrics_update.highest_gpu_util = gpu_utilization
                        
                # Update highest CPU utilization
                if cpu_util > send_system_metrics_update.highest_cpu_util:
                    send_system_metrics_update.highest_cpu_util = cpu_util
            
            # Create note about update type
            if current_epoch is not None:
                period_note = f"During training (Epoch {current_epoch})"
            else:
                period_note = "Periodic update (15-minute interval)"
            
            # Prepare email content blocks
            content_blocks = [
                ("Timestamp", timestamp),
                ("Update Type", period_note),
                ("GPU Temperature", gpu_temp_display),
                ("Highest GPU Temperature", f"{send_system_metrics_update.highest_gpu_temp}°C"),
                ("CPU Temperature", cpu_temp_display),
                ("Highest CPU Temperature", f"{send_system_metrics_update.highest_cpu_temp}°C"),
            ]
            
            if memory_usage is not None:
                content_blocks.append(("GPU Memory Usage", f"{memory_usage:.2f}%"))
                
            if gpu_utilization is not None:
                content_blocks.append(("GPU Utilization", f"{gpu_utilization:.2f}%"))
                content_blocks.append(("Highest GPU Utilization", f"{send_system_metrics_update.highest_gpu_util:.2f}%"))
                
            if cpu_util > 0:
                content_blocks.append(("CPU Utilization", cpu_util_display))
                content_blocks.append(("Highest CPU Utilization", f"{send_system_metrics_update.highest_cpu_util:.2f}%"))
                
            if model_version:
                content_blocks.append(("Model Version", model_version))
            
            # Check if we need to send a warning email
            needs_warning = False
            warning_messages = []
            
            if gpu_temp >= GPU_WARNING_THRESHOLD:
                needs_warning = True
                warning_messages.append(f"⚠️ WARNING: GPU temperature ({gpu_temp}°C) has exceeded the threshold of {GPU_WARNING_THRESHOLD}°C")
            
            if cpu_temp >= CPU_WARNING_THRESHOLD:
                needs_warning = True
                warning_messages.append(f"⚠️ WARNING: CPU temperature ({cpu_temp}°C) has exceeded the threshold of {CPU_WARNING_THRESHOLD}°C")
            
            # Send email(s)
            if needs_warning:
                warning_subject = f"⚠️ THETA TEMPERATURE WARNING - {timestamp}"
                warning_content = "\n".join(warning_messages)
                content_blocks.append(("WARNING", warning_content))
                
                send_email_notification(warning_subject, content_blocks)
                logger.warning(f"Temperature warning email sent: {warning_subject}")
            else:
                # Always send system metrics update email every 15 minutes as requested
                update_subject = f"Theta System Metrics - {timestamp}"
                send_email_notification(update_subject, content_blocks)
                logger.info(f"System metrics update email sent")
                
    except Exception as e:
        logger.error(f"Error getting system metrics: {e}")
        # Still try to send what we have
        if current_epoch is not None:
            period_note = f"During training (Epoch {current_epoch}) - Error: {str(e)}"
        else:
            period_note = f"Periodic update (15-minute interval) - Error: {str(e)}"
    
    # Set epoch for DB storage (0 for periodic updates)
    db_epoch = current_epoch if current_epoch is not None else 0
    
    # Save to database - use dummy values for training metrics when it's just a periodic update
    try:
        # Save to database with the appropriate epoch number
        db.save_training_metrics(
            epoch_number=db_epoch,
            train_loss=0.0,  # Placeholder
            validation_loss=0.0,  # Placeholder
            model_version=model_version or "periodic_update",
            notes=period_note,
            gpu_temperature=gpu_temp,
            gpu_memory_usage=memory_usage,
            gpu_utilization=gpu_utilization,
            cpu_temperature=cpu_temp,
            cpu_utilization=cpu_util,
            cpu_highest_temp=send_system_metrics_update.highest_cpu_temp,
            cpu_highest_utilization=send_system_metrics_update.highest_cpu_util
        )
        logger.info(f"Saved GPU and CPU metrics to database for epoch {db_epoch}")
    except Exception as e:
        logger.error(f"Error saving hardware metrics to database: {e}")

def schedule_system_metrics_updates():
    """Schedule system metrics updates to run every 15 minutes.
    This function sets up a background scheduler that will call send_system_metrics_update
    periodically, even when not actively training.
    """
    scheduler = BackgroundScheduler()
    scheduler.add_job(
        send_system_metrics_update,
        'interval',
        minutes=15,
        id='system_metrics_job',
        replace_existing=True
    )
    scheduler.start()
    logger.info("Scheduled system metrics updates (GPU and CPU) to run every 15 minutes")
    
    return scheduler

def send_error_notification(error_message, traceback_info):
    """Send an email notification about a training error."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    gpu_temp = get_gpu_temperature()
    cpu_temp = get_cpu_temperature()
    cpu_util = get_cpu_utilization()
    
    # Initialize variables
    memory_usage = "N/A"
    gpu_utilization = "N/A"
    gpu_info = "N/A"
    
    # Get detailed system information
    if torch.cuda.is_available():
        gpu_info = torch.cuda.get_device_name(0)
        
        if NVIDIA_SMI_AVAILABLE:
            try:
                cmd = "nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits"
                result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, shell=True)
                
                if result.returncode == 0:
                    info = result.stdout.strip().split(',')
                    if len(info) >= 2:
                        memory_used = float(info[0].strip())
                        memory_total = float(info[1].strip())
                        memory_usage = f"{memory_used} MB / {memory_total} MB ({(memory_used / memory_total * 100):.1f}%)"
                        
                    if len(info) >= 3:
                        gpu_utilization = f"{float(info[2].strip()):.1f}%"
            except Exception as e:
                logger.error(f"Error getting detailed GPU info for error notification: {e}")
    
    # Create email content blocks
    content_blocks = [
        ("ERROR", error_message),
        ("Timestamp", timestamp),
        ("GPU Information", gpu_info),
        ("GPU Temperature", f"{gpu_temp}°C"),
        ("GPU Memory Usage", memory_usage),
        ("GPU Utilization", gpu_utilization),
        ("CPU Temperature", f"{cpu_temp}°C"),
        ("CPU Utilization", f"{cpu_util:.1f}%" if cpu_util > 0 else "N/A"),
    ]
    
    # Add traceback information
    if traceback_info:
        content_blocks.append(("Traceback", f"<pre>{traceback_info}</pre>"))
    
    # Create email subject
    subject = f"⚠️ Theta AI Training Error Alert - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    
    # Send the notification using our template-based email function
    send_email_notification(subject, content_blocks)
    logger.error(f"Training error notification sent: {error_message}")

def main():
    """Main function to parse arguments and start training."""
    parser = argparse.ArgumentParser(description="Train Theta AI on all datasets")
    
    # Model arguments
    parser.add_argument("--model_path", type=str, default=None,
                      help="Path to existing model to continue training from")
    parser.add_argument("--output_dir", type=str, default="theta_all_datasets_model",
                      help="Directory to save model checkpoints")
    
    # Training arguments
    parser.add_argument("--batch_size", type=int, default=8, #batch size 8 most stable for 12gb RTX 3060
                      help="Batch size for training")
    parser.add_argument("--num_epochs", type=int, default=15, #epochs 15 for final training 5 for growing on already trained datasets 
                      help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=5e-5, #increased from 2e-5 for better learning
                      help="Learning rate")
    parser.add_argument("--max_length", type=int, default=512, #max length 512 most stable for 12gb RTX 3060
                      help="Maximum sequence length")
    
    # Other arguments
    parser.add_argument("--use_cpu", action="store_true",
                      help="Use  for training instead of GPU")
    parser.add_argument("--seed", type=int, default=42,
                      help="Random seed for reproducibility")
    parser.add_argument("--monitor_system", action="store_true", default=True,
                      help="Enable 15-minute GPU and CPU metrics monitoring")
    
    args = parser.parse_args()
    
    # Set random seeds
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        
    # Start system monitoring thread (sends updates every 15 minutes)
    if args.monitor_system:
        logger.info("Starting system metrics monitoring (updates every 15 minutes)")
        metrics_scheduler = schedule_system_metrics_updates()
        # Also send an initial update immediately
        send_system_metrics_update()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Train the model with error handling
    try:
        model, tokenizer = train_model(args)
        logger.info("Training complete!")
    except Exception as e:
        # Get traceback information
        import traceback
        traceback_str = traceback.format_exc()
        
        # Log the error
        logger.error(f"Training failed: {str(e)}\n{traceback_str}")
        
        # Send error notification email
        send_error_notification(str(e), traceback_str)
        
        # Re-raise the exception
        raise

if __name__ == "__main__":
    main()
