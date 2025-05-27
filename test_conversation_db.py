"""Test script to verify database integration with theta_conversation.py"""
import os
from theta_conversation import ConversationMemory

def test_conversation_db():
    # Initialize conversation memory with database
    memory = ConversationMemory()
    
    # Test adding an exchange
    test_input = "What is your name?"
    test_response = "My name is Theta."
    memory.add_exchange(test_input, test_response)
    print("\nAdded test exchange to database...")
    
    # Get recent exchanges
    print("\nRecent exchanges from database:")
    print("-" * 50)
    try:
        pairs = memory.get_training_pairs()
        for input_text, response in pairs[:5]:  # Show first 5 pairs
            print(f"\nUser: {input_text}")
            print(f"Theta: {response}")
            print("-" * 30)
    except Exception as e:
        print(f"Error getting training pairs: {e}")
    
    # Get database stats
    print("\nDatabase Statistics:")
    print("-" * 50)
    try:
        stats = memory.db.get_stats()
        for key, value in stats.items():
            print(f"{key}: {value}")
    except Exception as e:
        print(f"Error getting stats: {e}")

if __name__ == "__main__":
    test_conversation_db()
