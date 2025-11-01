"""
Embeddings Module
Handles generation of embeddings for questions using OpenAI
"""

import streamlit as st
from openai import OpenAI
from typing import List, Dict, Any
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def get_openai_client() -> OpenAI:
    """
    Initialize and return OpenAI client
    
    Returns:
        OpenAI client instance
    """
    try:
        api_key = st.secrets["OPENAI_API_KEY"]
        return OpenAI(api_key=api_key)
    except KeyError:
        st.error("❌ OpenAI API key not found in secrets!")
        st.info("Please add OPENAI_API_KEY to Streamlit secrets")
        st.stop()
    except Exception as e:
        st.error(f"Error initializing OpenAI client: {str(e)}")
        st.stop()


def generate_embeddings(
    texts: List[str],
    client: OpenAI,
    model: str = "text-embedding-3-small"
) -> np.ndarray:
    """
    Generate embeddings for a list of texts
    
    Args:
        texts: List of text strings to embed
        client: OpenAI client instance
        model: Embedding model to use
        
    Returns:
        NumPy array of embeddings (shape: [len(texts), embedding_dim])
    """
    try:
        # OpenAI API accepts up to 2048 texts at once
        # We'll process in batches to be safe
        batch_size = 100
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            
            response = client.embeddings.create(
                input=batch,
                model=model
            )
            
            batch_embeddings = [item.embedding for item in response.data]
            all_embeddings.extend(batch_embeddings)
        
        return np.array(all_embeddings)
    
    except Exception as e:
        raise Exception(f"Error generating embeddings: {str(e)}")


def generate_embeddings_for_qa_pairs(
    qa_pairs: List[Dict[str, Any]],
    client: OpenAI
) -> List[Dict[str, Any]]:
    """
    Generate embeddings for questions in QA pairs
    
    Args:
        qa_pairs: List of QA pair dictionaries
        client: OpenAI client instance
        
    Returns:
        QA pairs with embeddings added
    """
    try:
        # Extract questions
        questions = [qa.get('question', '') for qa in qa_pairs]
        
        # Generate embeddings
        embeddings = generate_embeddings(questions, client)
        
        # Add embeddings to QA pairs
        qa_pairs_with_embeddings = []
        for qa, embedding in zip(qa_pairs, embeddings):
            qa_with_emb = qa.copy()
            qa_with_emb['embedding'] = embedding
            qa_pairs_with_embeddings.append(qa_with_emb)
        
        return qa_pairs_with_embeddings
    
    except Exception as e:
        raise Exception(f"Error adding embeddings to QA pairs: {str(e)}")


def calculate_similarity_matrix(embeddings: np.ndarray) -> np.ndarray:
    """
    Calculate pairwise cosine similarity matrix
    
    Args:
        embeddings: NumPy array of embeddings
        
    Returns:
        Similarity matrix (shape: [n, n])
    """
    return cosine_similarity(embeddings)


def find_similar_qa_pairs(
    qa_pairs_with_embeddings: List[Dict[str, Any]],
    threshold: float = 0.85
) -> List[List[int]]:
    """
    Find groups of similar QA pairs based on embedding similarity
    
    Args:
        qa_pairs_with_embeddings: QA pairs with embeddings
        threshold: Similarity threshold (0-1)
        
    Returns:
        List of groups (each group is a list of indices)
    """
    # Extract embeddings
    embeddings = np.array([qa['embedding'] for qa in qa_pairs_with_embeddings])
    
    # Calculate similarity matrix
    sim_matrix = calculate_similarity_matrix(embeddings)
    
    # Find similar pairs
    similar_groups = []
    used_indices = set()
    
    for i in range(len(embeddings)):
        if i in used_indices:
            continue
        
        # Find all items similar to this one
        similar_indices = np.where(sim_matrix[i] >= threshold)[0].tolist()
        
        if len(similar_indices) > 1:
            similar_groups.append(similar_indices)
            used_indices.update(similar_indices)
    
    return similar_groups


def estimate_embedding_cost(num_texts: int, model: str = "text-embedding-3-small") -> float:
    """
    Estimate cost of embedding generation
    
    Args:
        num_texts: Number of texts to embed
        model: Embedding model
        
    Returns:
        Estimated cost in USD
    """
    # text-embedding-3-small: $0.02 per 1M tokens
    # Rough estimate: ~100 tokens per question
    estimated_tokens = num_texts * 100
    cost = (estimated_tokens / 1_000_000) * 0.02
    return cost


def test_embeddings_connection(client: OpenAI) -> bool:
    """
    Test if OpenAI embeddings API is working
    
    Args:
        client: OpenAI client instance
        
    Returns:
        True if connection successful
    """
    try:
        response = client.embeddings.create(
            input=["test"],
            model="text-embedding-3-small"
        )
        return len(response.data) > 0
    except Exception as e:
        st.error(f"Embeddings test failed: {str(e)}")
        return False
