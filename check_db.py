"""Script to check database contents"""
from database import ConversationDatabase

def main():
    db = ConversationDatabase()
    
    # Get conversation stats
    stats = db.get_stats()
    print("\nDatabase Statistics:")
    print("-" * 20)
    for key, value in stats.items():
        print(f"{key}: {value}")
    
    # Get recent conversations
    print("\nRecent Conversations:")
    print("-" * 20)
    recent = db.get_recent_exchanges(limit=5)
    for exchange in recent:
        print(f"\nTimestamp: {exchange['timestamp']}")
        print(f"User: {exchange['user_input']}")
        print(f"Theta: {exchange['model_response']}")
        if exchange['user_feedback']:
            print(f"Feedback: {exchange['user_feedback']}")
        print("-" * 20)

if __name__ == "__main__":
    main()
