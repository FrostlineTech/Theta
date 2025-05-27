# Theta AI

Theta AI is a transformer-based language model designed for code completion, programming assistance, and general question answering with a unique conversational style. Developed by Frostline Solutions LLC, Theta aims to provide a versatile AI assistant that can be integrated into various platforms.

## Features
- Enhanced language generation with coherent, complete sentence responses
- "Smart ass" conversational style for engaging interactions
- Enhanced question answering capabilities with conversational memory
- Database-backed conversation history and training metrics
- Comprehensive hardware monitoring system (GPU & CPU)
- Real-time email notifications for system metrics and alerts
- Unified dashboard for training and hardware monitoring
- Model checkpointing and backup system
- Configurable training parameters
- Comprehensive testing framework
- Identity and domain-specific knowledge training

## Project Structure
- `model.py`: Core model definition and architecture
- `utils.py`: Utility functions for data processing
- `train_all_datasets.py`: Comprehensive training script with curriculum learning
- `database.py`: Database interactions for metrics and conversation history
- `theta_conversation.py`: Interactive conversation mode with real-time learning
- `training_dashboard.py`: Unified dashboard for training and hardware monitoring
- `inference.py`: Inference and prediction functionality
- `test_theta.py`: Script for testing model performance
- `data/`: Directory for training and evaluation data including specialized datasets
  - `foundation_sentences.txt`: Basic sentence structure examples
  - `question_answer_pairs.txt`: Factual Q&A examples
  - `dialogue_progressions.txt`: Natural conversation flows
  - `grounded_responses.txt`: Detailed, informative answers
  - `smartass_style_examples.txt`: Responses with humor and personality
- `email_templates/`: HTML templates for system notification emails
- `configs/`: Configuration files

## Getting Started
1. Install the required dependencies:
   ```
   pip install -r requirements.txt
   pip install -r requirements_db.txt  # For database features
   ```
2. Set up environment variables for database and email notifications:
   ```
   # Database settings
   export DB_NAME=theta_db
   export DB_USER=your_username
   export DB_PASSWORD=your_password
   export DB_HOST=localhost
   
   # Email notification settings (for system monitoring)
   export EMAIL_ADDRESS=your_email@example.com
   export EMAIL_APP_PASSWORD=your_app_password

   ```
3. Prepare your training data in the `data/` directory
4. Run the comprehensive training with curriculum learning approach:
   ```
   python train_all_datasets.py --output_dir models/final --num_epochs 15 --batch_size 8 --learning_rate 5e-5
   ```
5. Launch the training dashboard to monitor progress and system health:
   ```
   streamlit run training_dashboard.py
   ```

## System Monitoring and Dashboard

Theta includes a comprehensive system monitoring solution to track both GPU and CPU metrics during training:

### Real-time Metrics Tracking
- GPU temperature, memory usage, and utilization
- CPU temperature and utilization
- Automatic tracking of highest recorded values
- Database storage for time-series analysis

### Email Notifications
- Automated emails sent every 15 minutes with system health updates
- Warning alerts when temperatures exceed defined thresholds (GPU > 75°C, CPU > 70°C)
- Detailed metrics including highest recorded values
- Customizable HTML email templates

### Unified Dashboard
The training dashboard provides a comprehensive view of both training progress and system health:

```
streamlit run training_dashboard.py
```

The dashboard includes:
- Training metrics (loss, validation gap, epochs completed)
- GPU health monitoring (temperature, memory usage, utilization)
- CPU health monitoring (temperature, utilization)
- Historical data visualization
- Customizable date ranges and filters

## Training Details

### Curriculum Learning
Theta uses a curriculum learning approach, which introduces datasets in order of increasing complexity:

1. **Foundation Stage**: Basic sentence structure and factual knowledge
   - `foundation_sentences.txt`: Grammar and sentence completion
   - `state_capitals.txt`: Basic factual information

2. **Intermediate Stage**: Question-answering and dialogue
   - `question_answer_pairs.txt`: Direct Q&A format
   - `dialogue_progressions.txt`: Multi-turn conversations

3. **Advanced Stage**: Personality and style
   - `grounded_responses.txt`: Detailed, informative answers
   - `smartass_style_examples.txt`: Humor and personality training

This staged approach allows the model to build a strong foundation before learning more complex conversational styles.

### Epochs
An epoch is one complete pass through the entire training dataset. For example, with 58 Q&A pairs, one epoch means the model has seen and learned from all 58 examples once.

**Benefits of Multiple Epochs:**
- Early epochs (1-5): The model learns basic patterns
- Middle epochs (5-15): The model refines these patterns
- Later epochs (15+): Fine-tuning occurs for better accuracy

**How to Set Epochs:**
Use the `--num_epochs` parameter to specify how many times the model should iterate through the entire dataset:
```
python train_all_datasets.py --num_epochs 15 ...
```

### Batch Size
Batch size defines how many training examples the model processes before updating its weights. Larger batch sizes can speed up training but require more memory.

**How to Set Batch Size:**
Use the `--batch_size` parameter:
```
python train_all_datasets.py --batch_size 8 ...
```

**Recommendations:**
- For RTX 3060 (12GB): Try batch sizes of 4-8 (8 is optimal for most training)
- For systems with limited VRAM: Use smaller batch sizes (2-4)

### Learning Rate
The learning rate controls how quickly the model adapts to the training data. Theta now uses an increased learning rate (5e-5, up from 2e-5) with warmup for better generalization and to address overfitting.

**How to Set Learning Rate:**
```
python train_all_datasets.py --learning_rate 5e-5 ...
```

### Other Architectural Improvements
The latest version of Theta includes several architectural improvements:

- **Reduced model complexity**: Smaller hidden size (768 instead of 1024) and fewer layers (12 instead of 24) to reduce overfitting
- **Increased regularization**: Higher dropout rates (0.2 instead of 0.1) and weight decay (0.01) for better generalization
- **Improved text generation**: Enhanced nucleus sampling for more coherent responses
- **Extended warmup period**: Longer learning rate warmup to improve training stability

## Expanding the Training Dataset

### Specialized Datasets for Curriculum Learning
Theta now uses a curriculum learning approach with specialized datasets that build upon each other:

1. **Foundation Sentences Dataset** (`data/foundation_sentences.txt`)
   - Focus: Basic sentence structure, grammar, and coherence
   - Format: Incomplete sentences that the model learns to complete
   - Example:
     ```
     The capital of France is Paris.|
     The president of the United States lives in the White House.|
     ```

2. **Question-Answer Pairs** (`data/question_answer_pairs.txt`)
   - Focus: Factual knowledge and direct responses
   - Format: Q&A pairs with complete, informative answers
   - Example:
     ```
     Q: What is the capital of California?
     A: The capital of California is Sacramento.
     
     Q: Who created you?
     A: I was created by Dakota Fryberger at Frostline Solutions LLC.
     ```

3. **Dialogue Progressions** (`data/dialogue_progressions.txt`)
   - Focus: Natural conversation flow and multi-turn exchanges
   - Format: Complete dialogues with multiple turns
   - Example:
     ```
     User: Hi, how are you today?
     Theta: I'm doing well, thanks for asking! How about you?
     
     User: Can you help me with a Python question?
     Theta: Absolutely! I'd be happy to help with your Python question. What would you like to know?
     ```

4. **Grounded Responses** (`data/grounded_responses.txt`)
   - Focus: Detailed, informative answers to complex questions
   - Format: Questions with comprehensive explanations
   - Example:
     ```
     User: How does a transformer neural network work?
     Theta: Transformer neural networks rely on a mechanism called self-attention. Unlike previous sequence models that process data sequentially, transformers can process entire sequences at once. The architecture consists of an encoder and decoder, each with multiple layers of self-attention and feed-forward neural networks. The key innovation is the attention mechanism that weighs the importance of different words in relation to each other, allowing the model to focus on relevant parts of the input regardless of their position in the sequence.
     ```

5. **Smart Ass Style Examples** (`data/smartass_style_examples.txt`)
   - Focus: Conversational style with humor and personality
   - Format: Exchanges showing witty but informative responses
   - Example:
     ```
     User: What's the meaning of life?
     Theta: Oh, you're only asking me the question philosophers have pondered for millennia. No pressure! The meaning of life is 42... just kidding. That's from "The Hitchhiker's Guide to the Galaxy." In reality, it's whatever you make it. I'd tell you my purpose, but I'm still figuring out why my creator gave me this sarcastic subroutine. Maybe to keep things interesting?
     ```

### Adding New Training Examples
1. Identify which dataset category your new examples fit into
2. Add examples to the appropriate file using the format shown above
3. For completely new categories, create a new file in the `data/` directory
4. Update the curriculum stages in `train_all_datasets.py` if adding new dataset files
5. Run training with the updated datasets:
   ```
   python train_all_datasets.py --output_dir models/final --num_epochs 15
   ```

## Testing and Interacting with Theta

Theta offers multiple ways to interact with and test the model:

### Enhanced Conversation Mode
```
python theta_conversation.py --model_path models/final/final_model
```

The conversation mode offers several advanced features:
- Database-backed conversation history for context retention
- Dynamic parameter adjustment during runtime
- Smart response cleaning for better output formatting
- "Smart ass" conversational style with personality
- Real-time learning from user interactions

Commands available during conversation:
```
/help                   - Show available commands
/params                 - Show current generation parameters
/temp [value]          - Set temperature (0.1-1.5)
/top_p [value]         - Set nucleus sampling parameter (0.0-1.0)
/max_length [value]    - Set maximum response length
/exit                  - Exit conversation mode
/correct                - Correct the last response
/learning on/off        - Toggle real-time learning from user interactions
/save                   - Save the conversation history to a file
/stats                  - Show chat statistics and model performance
```

### Basic Testing
For quick testing of the model's capabilities:

```
# Test identity questions (who are you, what's your name, etc.)
python test_theta.py --model_path models/final/final_model --test_type identity

# Test capital questions
python test_theta.py --model_path models/final/final_model --test_type capitals

# Test mixed questions
python test_theta.py --model_path models/final/final_model --test_type mixed
```

### Single Question Testing
```
python test_theta.py --model_path models/final/final_model --question "What is the capital of California?"
```

### Simple Interactive Mode
```
python test_theta.py --model_path models/final/final_model --interactive
```
This starts a basic chat interface for quick testing.

### Testing Specific Checkpoints
```
python test_theta.py --model_path models/final/checkpoint-epoch-10
```
This allows you to compare model performance at different stages of training.

## Continuing Training

To continue training from a previous checkpoint with the unified training approach:
```
python train_all_datasets.py --output_dir models/continued --num_epochs 15 --batch_size 8 --learning_rate 3e-5 --model_name_or_path models/final/final_model
```

You can also monitor the continued training in real-time with the dashboard:
```
streamlit run training_dashboard.py
```

## Viewing Training Metrics History

The training dashboard provides comprehensive visualization of training metrics and system health data:

1. **Training Metrics Tab**:
   - Loss curves for both training and validation
   - Epoch progression and model improvement
   - Customizable date ranges for focused analysis

2. **GPU Health Tab**:
   - Temperature trends over time
   - Memory usage and utilization metrics
   - Warning threshold visualization

3. **CPU Health Tab**:
   - CPU temperature monitoring
   - Utilization tracking
   - Historical performance data

## Future Integrations and Enhancements

### Near-term Development
- Discord bot integration with the "smart ass" conversational style
- Website support bot functionality with real-time learning
- API endpoint for remote inference
- Mobile application support

### Monitoring Enhancements
- Integration with cloud monitoring services
- Predictive analytics for hardware performance
- Automated throttling based on temperature thresholds
- Remote monitoring via mobile app notifications

### Training Improvements
- Multi-GPU training support with distributed metrics
- Expanded curriculum with domain-specific datasets
- Fine-tuning tools for personality customization
- Transfer learning from larger pre-trained models
