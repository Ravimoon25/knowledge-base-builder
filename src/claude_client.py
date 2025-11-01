"""
Claude Client Module
Handles all interactions with Anthropic's Claude API
"""

import streamlit as st
from anthropic import Anthropic
from typing import Optional, Dict, Any
import json

def get_claude_client() -> Anthropic:
    """
    Initialize and return Claude client
    
    Returns:
        Anthropic client instance
    """
    try:
        api_key = st.secrets["ANTHROPIC_API_KEY"]
        return Anthropic(api_key=api_key)
    except KeyError:
        st.error("❌ Anthropic API key not found in secrets!")
        st.info("Please add ANTHROPIC_API_KEY to Streamlit secrets")
        st.stop()
    except Exception as e:
        st.error(f"Error initializing Claude client: {str(e)}")
        st.stop()


def test_claude_connection(client: Anthropic) -> bool:
    """
    Test if Claude API is working
    
    Args:
        client: Anthropic client instance
        
    Returns:
        True if connection successful, False otherwise
    """
    try:
        response = client.messages.create(
            model="claude-sonnet-4-5-20250929",
            max_tokens=100,
            messages=[
                {"role": "user", "content": "Say 'Connection successful!' if you can read this."}
            ]
        )
        return "successful" in response.content[0].text.lower()
    except Exception as e:
        st.error(f"Connection test failed: {str(e)}")
        return False


def call_claude(
    client: Anthropic,
    prompt: str,
    model: str = "claude-sonnet-4-5-20250929",
    max_tokens: int = 4000,
    temperature: float = 0.7,
    system_prompt: Optional[str] = None
) -> str:
    """
    Make a call to Claude API
    
    Args:
        client: Anthropic client instance
        prompt: User prompt/message
        model: Model to use
        max_tokens: Maximum tokens in response
        temperature: Randomness (0-1)
        system_prompt: Optional system prompt
        
    Returns:
        Claude's response text
    """
    try:
        messages = [{"role": "user", "content": prompt}]
        
        kwargs = {
            "model": model,
            "max_tokens": max_tokens,
            "messages": messages,
            "temperature": temperature
        }
        
        if system_prompt:
            kwargs["system"] = system_prompt
        
        response = client.messages.create(**kwargs)
        
        return response.content[0].text
    
    except Exception as e:
        raise Exception(f"Error calling Claude: {str(e)}")


def call_claude_with_json(
    client: Anthropic,
    prompt: str,
    model: str = "claude-sonnet-4-5-20250929",
    max_tokens: int = 4000,
    system_prompt: Optional[str] = None
) -> Dict[Any, Any]:
    """
    Make a call to Claude API and expect JSON response
    
    Args:
        client: Anthropic client instance
        prompt: User prompt/message
        model: Model to use
        max_tokens: Maximum tokens in response
        system_prompt: Optional system prompt
        
    Returns:
        Parsed JSON response
    """
    try:
        response_text = call_claude(
            client=client,
            prompt=prompt,
            model=model,
            max_tokens=max_tokens,
            temperature=0.3,  # Lower temperature for more consistent JSON
            system_prompt=system_prompt
        )
        
        # Try to extract JSON from response
        # Sometimes Claude wraps JSON in markdown code blocks
        if "```json" in response_text:
            # Extract JSON from code block
            json_start = response_text.find("```json") + 7
            json_end = response_text.find("```", json_start)
            json_str = response_text[json_start:json_end].strip()
        elif "```" in response_text:
            # Extract from generic code block
            json_start = response_text.find("```") + 3
            json_end = response_text.find("```", json_start)
            json_str = response_text[json_start:json_end].strip()
        else:
            json_str = response_text.strip()
        
        # Parse JSON
        return json.loads(json_str)
    
    except json.JSONDecodeError as e:
        raise Exception(f"Failed to parse JSON response: {str(e)}\nResponse: {response_text[:200]}")
    except Exception as e:
        raise Exception(f"Error calling Claude with JSON: {str(e)}")


def estimate_tokens(text: str) -> int:
    """
    Rough estimation of token count
    
    Args:
        text: Input text
        
    Returns:
        Estimated token count
    """
    # Rough estimate: ~4 characters per token
    return len(text) // 4


def estimate_cost(input_tokens: int, output_tokens: int, model: str = "claude-sonnet-4-5-20250929") -> float:
    """
    Estimate API call cost
    
    Args:
        input_tokens: Number of input tokens
        output_tokens: Number of output tokens
        model: Model being used
        
    Returns:
        Estimated cost in USD
    """
    # Claude Sonnet 4.5 pricing (as of 2025)
    # Input: $3 per million tokens
    # Output: $15 per million tokens
    
    if "sonnet" in model.lower():
        input_cost = (input_tokens / 1_000_000) * 3.0
        output_cost = (output_tokens / 1_000_000) * 15.0
    elif "opus" in model.lower():
        # Claude Opus pricing
        input_cost = (input_tokens / 1_000_000) * 15.0
        output_cost = (output_tokens / 1_000_000) * 75.0
    else:
        # Default to Sonnet pricing
        input_cost = (input_tokens / 1_000_000) * 3.0
        output_cost = (output_tokens / 1_000_000) * 15.0
    
    return input_cost + output_cost
