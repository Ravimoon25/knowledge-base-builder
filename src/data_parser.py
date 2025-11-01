
"""
Data Parser Module
Handles parsing of CSV, Excel, and Text files into structured conversation format
"""

import pandas as pd
from typing import List, Dict, Any
import io

def parse_csv_conversations(file) -> List[Dict[str, Any]]:
    """
    Parse CSV file into list of conversations
    
    Args:
        file: Uploaded file object
        
    Returns:
        List of conversations, each with metadata and messages
    """
    try:
        # Read CSV
        df = pd.read_csv(file)
        
        # Detect column names (case-insensitive)
        df.columns = df.columns.str.lower().str.strip()
        
        # Try to identify speaker and message columns
        speaker_col = None
        message_col = None
        
        # Common speaker column names
        speaker_options = ['speaker', 'role', 'participant', 'user', 'from']
        for col in df.columns:
            if any(opt in col for opt in speaker_options):
                speaker_col = col
                break
        
        # Common message column names
        message_options = ['message', 'text', 'content', 'utterance', 'dialogue']
        for col in df.columns:
            if any(opt in col for opt in message_options):
                message_col = col
                break
        
        # If not found, use first two columns as fallback
        if speaker_col is None or message_col is None:
            if len(df.columns) >= 2:
                speaker_col = df.columns[0]
                message_col = df.columns[1]
            else:
                raise ValueError("Could not identify speaker and message columns")
        
        # Check for conversation ID column
        conv_id_col = None
        conv_id_options = ['conversation_id', 'conv_id', 'session_id', 'call_id', 'id']
        for col in df.columns:
            if any(opt in col for opt in conv_id_options):
                conv_id_col = col
                break
        
        # Parse conversations
        conversations = []
        
        if conv_id_col:
            # Group by conversation ID
            for conv_id, group in df.groupby(conv_id_col):
                messages = []
                for _, row in group.iterrows():
                    messages.append({
                        'speaker': str(row[speaker_col]).strip(),
                        'text': str(row[message_col]).strip()
                    })
                
                conversations.append({
                    'id': f"conv_{conv_id}",
                    'messages': messages,
                    'num_messages': len(messages)
                })
        else:
            # Treat entire file as one conversation
            messages = []
            for _, row in df.iterrows():
                messages.append({
                    'speaker': str(row[speaker_col]).strip(),
                    'text': str(row[message_col]).strip()
                })
            
            conversations.append({
                'id': 'conv_001',
                'messages': messages,
                'num_messages': len(messages)
            })
        
        return conversations
    
    except Exception as e:
        raise Exception(f"Error parsing CSV: {str(e)}")


def parse_excel_conversations(file) -> List[Dict[str, Any]]:
    """
    Parse Excel file into list of conversations
    
    Args:
        file: Uploaded file object
        
    Returns:
        List of conversations
    """
    try:
        # Read Excel
        df = pd.read_excel(file)
        
        # Convert to CSV-like format and use CSV parser
        csv_buffer = io.StringIO()
        df.to_csv(csv_buffer, index=False)
        csv_buffer.seek(0)
        
        return parse_csv_conversations(csv_buffer)
    
    except Exception as e:
        raise Exception(f"Error parsing Excel: {str(e)}")


def parse_text_conversations(file) -> List[Dict[str, Any]]:
    """
    Parse text file into list of conversations
    Expects format like:
    Speaker1: message
    Speaker2: message
    
    Args:
        file: Uploaded file object
        
    Returns:
        List of conversations
    """
    try:
        # Read text file
        content = file.read().decode('utf-8')
        lines = content.split('\n')
        
        messages = []
        current_speaker = None
        current_text = []
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Check if line starts with speaker pattern (Speaker: or Speaker - )
            if ':' in line or '-' in line:
                # Save previous message if exists
                if current_speaker and current_text:
                    messages.append({
                        'speaker': current_speaker,
                        'text': ' '.join(current_text).strip()
                    })
                    current_text = []
                
                # Parse new speaker and message
                if ':' in line:
                    parts = line.split(':', 1)
                elif '-' in line:
                    parts = line.split('-', 1)
                
                if len(parts) == 2:
                    current_speaker = parts[0].strip()
                    current_text = [parts[1].strip()]
                else:
                    current_text.append(line)
            else:
                # Continuation of previous message
                current_text.append(line)
        
        # Add last message
        if current_speaker and current_text:
            messages.append({
                'speaker': current_speaker,
                'text': ' '.join(current_text).strip()
            })
        
        conversations = [{
            'id': 'conv_001',
            'messages': messages,
            'num_messages': len(messages)
        }]
        
        return conversations
    
    except Exception as e:
        raise Exception(f"Error parsing text file: {str(e)}")


def validate_conversations(conversations: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Validate parsed conversations and return statistics
    
    Args:
        conversations: List of parsed conversations
        
    Returns:
        Dictionary with validation results and statistics
    """
    if not conversations:
        return {
            'valid': False,
            'error': 'No conversations found',
            'stats': {}
        }
    
    total_messages = sum(conv['num_messages'] for conv in conversations)
    
    # Check for common issues
    issues = []
    
    # Check if conversations have messages
    empty_convs = [conv for conv in conversations if conv['num_messages'] == 0]
    if empty_convs:
        issues.append(f"{len(empty_convs)} empty conversations found")
    
    # Check for speaker diversity
    all_speakers = set()
    for conv in conversations:
        for msg in conv['messages']:
            all_speakers.add(msg['speaker'].lower())
    
    if len(all_speakers) < 2:
        issues.append("Only one speaker detected - conversations need at least 2 speakers")
    
    stats = {
        'total_conversations': len(conversations),
        'total_messages': total_messages,
        'avg_messages_per_conversation': total_messages / len(conversations) if conversations else 0,
        'unique_speakers': len(all_speakers),
        'speakers': list(all_speakers)
    }
    
    return {
        'valid': len(issues) == 0,
        'issues': issues,
        'stats': stats
    }


def format_conversation_for_display(conversation: Dict[str, Any]) -> str:
    """
    Format a conversation for display in the UI
    
    Args:
        conversation: Parsed conversation dictionary
        
    Returns:
        Formatted string representation
    """
    output = []
    output.append(f"📝 Conversation ID: {conversation['id']}")
    output.append(f"💬 Messages: {conversation['num_messages']}")
    output.append("-" * 50)
    
    for idx, msg in enumerate(conversation['messages'][:10], 1):  # Show first 10 messages
        speaker = msg['speaker']
        text = msg['text'][:100] + "..." if len(msg['text']) > 100 else msg['text']
        output.append(f"{idx}. {speaker}: {text}")
    
    if conversation['num_messages'] > 10:
        output.append(f"\n... and {conversation['num_messages'] - 10} more messages")
    
    return "\n".join(output)


def get_sample_conversation_csv() -> pd.DataFrame:
    """
    Generate a sample conversation CSV for demonstration
    
    Returns:
        Sample DataFrame
    """
    data = {
        'conversation_id': [1, 1, 1, 1, 2, 2, 2, 2],
        'speaker': ['Agent', 'Customer', 'Agent', 'Customer', 'Agent', 'Customer', 'Agent', 'Customer'],
        'message': [
            'Hello! How can I help you today?',
            'I need help with my refund policy',
            'I can help you with that. Our refund policy allows returns within 30 days of purchase.',
            'Thank you, that\'s helpful!',
            'Good morning! What brings you here today?',
            'I want to know about your shipping options',
            'We offer standard shipping (5-7 days) and express shipping (2-3 days).',
            'Great, I\'ll go with express shipping.'
        ]
    }
    
    return pd.DataFrame(data)
