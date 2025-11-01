"""
Batch Processor Module
Handles batch processing of multiple conversations
"""

from typing import List, Dict, Any, Callable, Optional
from anthropic import Anthropic
import time
from src.extractor import extract_qa_pairs

def process_conversations_batch(
    client: Anthropic,
    conversations: List[Dict[str, Any]],
    model: str = "claude-sonnet-4-5-20250929",
    progress_callback: Optional[Callable] = None,
    delay_between_calls: float = 0.5
) -> List[Dict[str, Any]]:
    """
    Process multiple conversations in batch
    
    Args:
        client: Anthropic client instance
        conversations: List of parsed conversations
        model: Claude model to use
        progress_callback: Optional callback for progress updates
        delay_between_calls: Delay between API calls (in seconds) to avoid rate limits
        
    Returns:
        List of extraction results
    """
    results = []
    total = len(conversations)
    
    for idx, conversation in enumerate(conversations):
        # Progress update
        if progress_callback:
            progress_callback(idx, total, conversation['id'])
        
        # Extract QA pairs
        try:
            result = extract_qa_pairs(client, conversation, model)
            results.append(result)
            
            # Small delay to avoid rate limits
            if idx < total - 1:  # Don't delay after last one
                time.sleep(delay_between_calls)
        
        except Exception as e:
            # Add error result
            results.append({
                'conversation_id': conversation.get('id', f'conv_{idx}'),
                'qa_pairs': [],
                'num_qa_pairs': 0,
                'input_tokens': 0,
                'estimated_cost': 0,
                'success': False,
                'error': str(e)
            })
    
    return results


def aggregate_results(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Aggregate results from batch processing
    
    Args:
        results: List of extraction results
        
    Returns:
        Aggregated statistics and all QA pairs
    """
    total_conversations = len(results)
    successful = sum(1 for r in results if r['success'])
    failed = total_conversations - successful
    
    total_qa_pairs = sum(r['num_qa_pairs'] for r in results)
    total_cost = sum(r['estimated_cost'] for r in results)
    total_tokens = sum(r['input_tokens'] for r in results)
    
    # Collect all QA pairs with metadata
    all_qa_pairs = []
    for result in results:
        if result['success']:
            for qa in result['qa_pairs']:
                qa_with_meta = qa.copy()
                qa_with_meta['source_conversation'] = result['conversation_id']
                all_qa_pairs.append(qa_with_meta)
    
    # Get failed conversations
    failed_conversations = [
        {'id': r['conversation_id'], 'error': r['error']}
        for r in results if not r['success']
    ]
    
    return {
        'total_conversations': total_conversations,
        'successful': successful,
        'failed': failed,
        'total_qa_pairs': total_qa_pairs,
        'avg_qa_per_conversation': total_qa_pairs / successful if successful > 0 else 0,
        'total_cost': total_cost,
        'total_tokens': total_tokens,
        'all_qa_pairs': all_qa_pairs,
        'failed_conversations': failed_conversations,
        'success_rate': (successful / total_conversations * 100) if total_conversations > 0 else 0
    }


def export_qa_pairs_to_dict(qa_pairs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Export QA pairs to a clean dictionary format
    
    Args:
        qa_pairs: List of QA pairs with metadata
        
    Returns:
        Cleaned list of QA pairs
    """
    exported = []
    for idx, qa in enumerate(qa_pairs, 1):
        exported.append({
            'id': idx,
            'question': qa.get('question', ''),
            'answer': qa.get('answer', ''),
            'source': qa.get('source_conversation', 'unknown'),
            'justification': qa.get('justification', '')
        })
    return exported
