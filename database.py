"""
Database handling for Theta conversations and memory.
"""
import os
import json
import logging
from datetime import datetime
from dotenv import load_dotenv
import psycopg2
from psycopg2.extras import RealDictCursor
from psycopg2.pool import SimpleConnectionPool
import threading
import time

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ConversationDatabase:
    """PostgreSQL database handler for conversation memory."""
    
    def __init__(self):
        """Initialize database connection pool."""
        self.conn_params = {
            'dbname': os.getenv('DB_NAME', 'theta_db'),
            'user': os.getenv('DB_USER', 'postgres'),
            'password': os.getenv('DB_PASSWORD', 'theta'),
            'host': os.getenv('DB_HOST', 'localhost'),
            'port': os.getenv('DB_PORT', '5432')
        }
        
        # Convert port to integer and ensure all parameters are strings
        params = {
            'dbname': str(self.conn_params['dbname'] or ''),
            'user': str(self.conn_params['user'] or ''),
            'password': str(self.conn_params['password'] or ''),
            'host': str(self.conn_params['host'] or ''),
            'port': int(self.conn_params['port'] or 5432)  # Default PostgreSQL port
        }
        
        # Create a connection pool with 5 connections
        self.pool = SimpleConnectionPool(1, 5, **params)
        self.local = threading.local()
        self.ensure_tables()
        
    def get_connection(self):
        """Get a database connection from the pool."""
        if not hasattr(self.local, 'connection') or self.local.connection.closed:
            self.local.connection = self.pool.getconn()
        return self.local.connection
    
    def release_connection(self, conn):
        """Return a connection to the pool."""
        self.pool.putconn(conn)
    
    def __del__(self):
        """Cleanup connection pool on object destruction."""
        try:
            self.pool.closeall()
        except:
            pass
    def ensure_tables(self):
        """Create necessary tables if they don't exist."""
        create_tables_sql = """
        CREATE TABLE IF NOT EXISTS conversations (
            id SERIAL PRIMARY KEY,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            user_input TEXT NOT NULL,
            model_response TEXT NOT NULL,
            user_feedback TEXT,
            loss FLOAT
        );
        
        CREATE TABLE IF NOT EXISTS training_data (
            id SERIAL PRIMARY KEY,
            question TEXT NOT NULL,
            answer TEXT NOT NULL,
            source VARCHAR(255),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            model_performance FLOAT,
            tags TEXT[]
        );
        
        CREATE TABLE IF NOT EXISTS model_feedback (
            id SERIAL PRIMARY KEY,
            conversation_id INTEGER REFERENCES conversations(id),
            feedback_type VARCHAR(50),
            feedback_text TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            action_taken TEXT,
            improvement_score FLOAT
        );
        
        CREATE TABLE IF NOT EXISTS training_metrics (
            id SERIAL PRIMARY KEY,
            epoch_number INTEGER NOT NULL,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            epoch_completion_date DATE DEFAULT CURRENT_DATE,
            train_loss REAL NOT NULL,
            validation_loss REAL NOT NULL,
            model_version TEXT,
            dataset_size INTEGER,
            learning_rate REAL,
            notes TEXT,
            gpu_temperature TEXT,
            gpu_memory_usage TEXT,
            gpu_utilization TEXT,
            cpu_temperature REAL,
            cpu_utilization REAL,
            cpu_highest_temp REAL,
            cpu_highest_utilization REAL
        );
        
        CREATE INDEX IF NOT EXISTS idx_conversations_timestamp 
        ON conversations(timestamp);
        
        CREATE INDEX IF NOT EXISTS idx_training_data_source 
        ON training_data(source);
        
        CREATE INDEX IF NOT EXISTS idx_model_feedback_conversation 
        ON model_feedback(conversation_id);
        
        CREATE INDEX IF NOT EXISTS idx_training_metrics_timestamp 
        ON training_metrics(timestamp);
        
        CREATE INDEX IF NOT EXISTS idx_training_metrics_epoch_completion_date 
        ON training_metrics(epoch_completion_date);
        """
        
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(create_tables_sql)
                    conn.commit()
                    logger.info("Database tables ensured")
        except Exception as e:
            logger.error(f"Error ensuring database tables: {e}")
            raise
    def execute_with_retry(self, operation, *args, max_retries=3):
        """Execute a database operation with retries."""
        last_error = None
        conn = None
        for attempt in range(max_retries):
            try:
                conn = self.get_connection()
                with conn.cursor(cursor_factory=RealDictCursor) as cur:
                    result = operation(cur, *args)
                    conn.commit()
                    return result
            except (psycopg2.Error, psycopg2.OperationalError) as e:
                last_error = e
                logger.warning(f"Database operation failed (attempt {attempt + 1}/{max_retries}): {e}")
                if conn:
                    conn.rollback()
                    self.release_connection(conn)
                    self.local.connection = None
                if attempt == max_retries - 1:
                    raise last_error
                time.sleep(1 * (attempt + 1))  # Exponential backoff
    
    def add_exchange(self, user_input, model_response, user_feedback=None, loss=None):
        """Add a conversation exchange to the database with retry mechanism."""
        def _operation(cur, *args):
            sql = """
            INSERT INTO conversations 
                (user_input, model_response, user_feedback, loss, timestamp)
            VALUES 
                (%s, %s, %s, %s, CURRENT_TIMESTAMP)
            RETURNING id;
            """
            cur.execute(sql, (user_input, model_response, user_feedback, loss))
            exchange_id = cur.fetchone()['id']
            logger.info(f"Saved conversation exchange with ID: {exchange_id}")
            return exchange_id
        
        try:
            return self.execute_with_retry(_operation)
        except Exception as e:
            logger.error(f"Error adding exchange to database after retries: {e}")
            raise

    def get_recent_exchanges(self, limit=5):
        """Get recent conversation exchanges."""
        sql = """
        SELECT 
            id, 
            timestamp, 
            user_input, 
            model_response, 
            user_feedback, 
            loss
        FROM conversations
        ORDER BY timestamp DESC
        LIMIT %s;
        """
        
        try:
            with self.get_connection() as conn:
                with conn.cursor(cursor_factory=RealDictCursor) as cur:
                    cur.execute(sql, (limit,))
                    return cur.fetchall()
        except Exception as e:
            logger.error(f"Error fetching recent exchanges: {e}")
            raise

    def get_training_pairs(self, limit=100):
        """Get conversation pairs for training."""
        sql = """
        SELECT 
            user_input, 
            COALESCE(user_feedback, model_response) as target_response
        FROM conversations
        ORDER BY 
            CASE WHEN user_feedback IS NOT NULL THEN 0 ELSE 1 END,
            timestamp DESC
        LIMIT %s;
        """
        
        try:
            with self.get_connection() as conn:
                with conn.cursor(cursor_factory=RealDictCursor) as cur:
                    cur.execute(sql, (limit,))
                    rows = cur.fetchall()
                    return [(row['user_input'], row['target_response']) for row in rows]
        except Exception as e:
            logger.error(f"Error fetching training pairs: {e}")
            raise

    def update_feedback(self, conversation_id, feedback):
        """Update user feedback for a conversation."""
        sql = """
        UPDATE conversations 
        SET 
            user_feedback = %s,
            updated_at = CURRENT_TIMESTAMP
        WHERE id = %s
        RETURNING id;
        """
        
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(sql, (feedback, conversation_id))
                    updated_id = cur.fetchone()
                    conn.commit()
                    if updated_id:
                        logger.info(f"Updated feedback for conversation {conversation_id}")
                        return True
                    return False
        except Exception as e:
            logger.error(f"Error updating feedback: {e}")
            raise

    def migrate_from_jsonl(self, jsonl_file):
        """Migrate existing conversations from JSONL file to database."""
        try:
            with open(jsonl_file, 'r') as f:
                for line in f:
                    if line.strip():
                        exchange = json.loads(line)
                        self.add_exchange(
                            user_input=exchange['user_input'],
                            model_response=exchange['model_response'],
                            user_feedback=exchange.get('user_feedback'),
                            loss=None  # Historical data might not have loss values
                        )
            logger.info(f"Successfully migrated conversations from {jsonl_file}")
        except Exception as e:
            logger.error(f"Error migrating from JSONL: {e}")
            raise

    def get_stats(self):
        """Get conversation statistics."""
        sql = """
        SELECT 
            COUNT(*) as total_conversations,
            COUNT(CASE WHEN user_feedback IS NOT NULL THEN 1 END) as feedback_count,
            AVG(CASE WHEN loss IS NOT NULL THEN loss END) as avg_loss,
            MIN(timestamp) as first_conversation,
            MAX(timestamp) as last_conversation
        FROM conversations;
        """
        
        try:
            with self.get_connection() as conn:
                with conn.cursor(cursor_factory=RealDictCursor) as cur:
                    cur.execute(sql)
                    return cur.fetchone()
        except Exception as e:
            logger.error(f"Error fetching statistics: {e}")
            raise

    def add_training_data(self, question, answer, source=None, tags=None, model_performance=None):
        """Add new training data to the database."""
        def _operation(cur, *args):
            sql = """
            INSERT INTO training_data 
                (question, answer, source, tags, model_performance, created_at)
            VALUES 
                (%s, %s, %s, %s, %s, CURRENT_TIMESTAMP)
            RETURNING id;
            """
            cur.execute(sql, (question, answer, source, tags, model_performance))
            training_id = cur.fetchone()['id']
            logger.info(f"Saved training data with ID: {training_id}")
            return training_id
        
        try:
            return self.execute_with_retry(_operation)
        except Exception as e:
            logger.error(f"Error adding training data: {e}")
            raise

    def add_model_feedback(self, conversation_id, feedback_type, feedback_text, action_taken=None, improvement_score=None):
        """Add model feedback for a conversation."""
        def _operation(cur, *args):
            sql = """
            INSERT INTO model_feedback 
                (conversation_id, feedback_type, feedback_text, action_taken, improvement_score, created_at)
            VALUES 
                (%s, %s, %s, %s, %s, CURRENT_TIMESTAMP)
            RETURNING id;
            """
            cur.execute(sql, (conversation_id, feedback_type, feedback_text, action_taken, improvement_score))
            feedback_id = cur.fetchone()['id']
            logger.info(f"Saved model feedback with ID: {feedback_id}")
            return feedback_id
        
        try:
            return self.execute_with_retry(_operation)
        except Exception as e:
            logger.error(f"Error adding model feedback: {e}")
            raise

    def get_all_training_data(self, limit=1000):
        """Get all training data including conversations and dedicated training data."""
        sql = """
        (SELECT 
            user_input as question,
            COALESCE(user_feedback, model_response) as answer,
            'conversation' as source,
            timestamp as created_at,
            loss as model_performance,
            ARRAY['conversation']::text[] as tags
        FROM conversations
        WHERE user_feedback IS NOT NULL OR model_response IS NOT NULL)
        UNION ALL
        (SELECT 
            question,
            answer,
            source,
            created_at,
            model_performance,
            tags
        FROM training_data)
        ORDER BY created_at DESC
        LIMIT %s;
        """
        
        try:
            with self.get_connection() as conn:
                with conn.cursor(cursor_factory=RealDictCursor) as cur:
                    cur.execute(sql, (limit,))
                    return cur.fetchall()
        except Exception as e:
            logger.error(f"Error fetching all training data: {e}")
            raise

    def get_model_feedback_for_conversation(self, conversation_id):
        """Get all feedback for a specific conversation."""
        sql = """
        SELECT 
            id,
            feedback_type,
            feedback_text,
            action_taken,
            improvement_score,
            created_at
        FROM model_feedback
        WHERE conversation_id = %s
        ORDER BY created_at DESC;
        """
        
        try:
            with self.get_connection() as conn:
                with conn.cursor(cursor_factory=RealDictCursor) as cur:
                    cur.execute(sql, (conversation_id,))
                    return cur.fetchall()
        except Exception as e:
            logger.error(f"Error fetching model feedback: {e}")
            raise
            
    def save_training_metrics(self, epoch_number, train_loss, validation_loss, model_version=None, dataset_size=None, learning_rate=None, notes=None, gpu_temperature=None, gpu_memory_usage=None, gpu_utilization=None, cpu_temperature=None, cpu_utilization=None, cpu_highest_temp=None, cpu_highest_utilization=None):
        """Save training metrics for a specific epoch."""
        def _operation(cur, *args):
            sql = """
            INSERT INTO training_metrics 
                (epoch_number, train_loss, validation_loss, model_version, dataset_size, learning_rate, notes, timestamp, epoch_completion_date, gpu_temperature, gpu_memory_usage, gpu_utilization, cpu_temperature, cpu_utilization, cpu_highest_temp, cpu_highest_utilization)
            VALUES 
                (%s, %s, %s, %s, %s, %s, %s, CURRENT_TIMESTAMP, CURRENT_DATE, %s, %s, %s, %s, %s, %s, %s)
            RETURNING id;
            """
            cur.execute(sql, (epoch_number, train_loss, validation_loss, model_version, dataset_size, learning_rate, notes, gpu_temperature, gpu_memory_usage, gpu_utilization, cpu_temperature, cpu_utilization, cpu_highest_temp, cpu_highest_utilization))
            metrics_id = cur.fetchone()['id']
            logger.info(f"Saved training metrics with ID: {metrics_id}")
            return metrics_id
        
        try:
            return self.execute_with_retry(_operation)
        except Exception as e:
            logger.error(f"Error saving training metrics: {e}")
            raise
            
    def get_training_metrics(self, limit=20):
        """Get recent training metrics."""
        sql = """
        SELECT 
            id,
            epoch_number,
            timestamp,
            epoch_completion_date,
            train_loss,
            validation_loss,
            model_version,
            dataset_size,
            learning_rate,
            notes,
            gpu_temperature,
            gpu_memory_usage,
            gpu_utilization,
            cpu_temperature,
            cpu_utilization,
            cpu_highest_temp,
            cpu_highest_utilization
        FROM training_metrics
        ORDER BY timestamp DESC
        LIMIT %s;
        """
        
        try:
            with self.get_connection() as conn:
                with conn.cursor(cursor_factory=RealDictCursor) as cur:
                    cur.execute(sql, (limit,))
                    return cur.fetchall()
        except Exception as e:
            logger.error(f"Error fetching training metrics: {e}")
            raise
