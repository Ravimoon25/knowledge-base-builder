"""
Knowledge Extractor Module
Extracts QA pairs from conversations using Claude AI
"""

from typing import List, Dict, Any
from anthropic import Anthropic
import streamlit as st
from src.claude_client import call_claude_with_json, estimate_tokens, estimate_cost

def load_extraction_prompt() -> str:
    """
    Load the extraction prompt template
    
    Returns:
        Prompt template string
    """
    try:
        with open('prompts/extraction_prompt.txt', 'r') as f:
            return f.read()
    except FileNotFoundError:
        # Fallback prompt if file not found
        return """Extract question-answer pairs from this conversation.
Return a JSON array with objects containing 'question', 'answer', and 'justification' fields.

Conversation:
{conversation}"""


def format_conversation_for_extraction(conversation: Dict[str, Any]) -> str:
    """
    Format a conversation for the extraction prompt
    
    Args:
        conversation: Parsed conversation dictionary
        
    Returns:
        Formatted conversation string
    """
    lines = []
    for msg in conversation['messages']:
        speaker = msg['speaker']
        text = msg['text']
        lines.append(f"{speaker}: {text}")
    
    return "\n".join(lines)


def extract_qa_pairs(
    client: Anthropic,
    conversation: Dict[str, Any],
    model: str = "claude-sonnet-4-5-20250929"
) -> Dict[str, Any]:
    """
    Extract QA pairs from a single conversation
    
    Args:
        client: Anthropic client instance
        conversation: Parsed conversation dictionary
        model: Claude model to use
        
    Returns:
        Dictionary with extraction results
    """
    try:
        # Load prompt template
        prompt_template = load_extraction_prompt()
        
        # Format conversation
        conv_text = format_conversation_for_extraction(conversation)
        
        # Create full prompt
        full_prompt = prompt_template.replace("{conversation}", conv_text)
        
        # Estimate tokens and cost
        input_tokens = estimate_tokens(full_prompt)
        estimated_output_tokens = 1000  # Rough estimate
        estimated_cost = estimate_cost(input_tokens, estimated_output_tokens, model)
        
        # Call Claude
        qa_pairs = call_claude_with_json(
            client=client,
            prompt=full_prompt,
            model=model,
            max_tokens=4000,
            system_prompt="You are a helpful assistant that extracts knowledge from conversations. Always respond with valid JSON."
        )
        
        # Validate response
        if not isinstance(qa_pairs, list):
            raise ValueError("Expected a list of QA pairs")
        
        # Add metadata
        result = {
            'conversation_id': conversation['id'],
            'qa_pairs': qa_pairs,
            'num_qa_pairs': len(qa_pairs),
            'input_tokens': input_tokens,
            'estimated_cost': estimated_cost,
            'success': True,
            'error': None
        }
        
        return result
    
    except Exception as e:
        return {
            'conversation_id': conversation.get('id', 'unknown'),
            'qa_pairs': [],
            'num_qa_pairs': 0,
            'input_tokens': 0,
            'estimated_cost': 0,
            'success': False,
            'error': str(e)
        }


def extract_from_multiple_conversations(
    client: Anthropic,
    conversations: List[Dict[str, Any]],
    model: str = "claude-sonnet-4-5-20250929",
    progress_callback=None
) -> List[Dict[str, Any]]:
    """
    Extract QA pairs from multiple conversations
    
    Args:
        client: Anthropic client instance
        conversations: List of parsed conversations
        model: Claude model to use
        progress_callback: Optional callback for progress updates
        
    Returns:
        List of extraction results
    """
    results = []
    total = len(conversations)
    
    for idx, conversation in enumerate(conversations):
        # Update progress
        if progress_callback:
            progress_callback(idx, total, conversation['id'])
        
        # Extract QA pairs
        result = extract_qa_pairs(client, conversation, model)
        results.append(result)
    
    return results


def summarize_extraction_results(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Summarize extraction results across all conversations
    
    Args:
        results: List of extraction results
        
    Returns:
        Summary statistics
    """
    total_conversations = len(results)
    successful = sum(1 for r in results if r['success'])
    failed = total_conversations - successful
    
    total_qa_pairs = sum(r['num_qa_pairs'] for r in results)
    total_cost = sum(r['estimated_cost'] for r in results)
    
    # Get all QA pairs
    all_qa_pairs = []
    for r in results:
        for qa in r['qa_pairs']:
            qa['source_conversation'] = r['conversation_id']
            all_qa_pairs.append(qa)
    
    return {
        'total_conversations': total_conversations,
        'successful': successful,
        'failed': failed,
        'total_qa_pairs': total_qa_pairs,
        'avg_qa_per_conversation': total_qa_pairs / total_conversations if total_conversations > 0 else 0,
        'total_estimated_cost': total_cost,
        'all_qa_pairs': all_qa_pairs
    }
