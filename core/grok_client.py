"""
NexInsight - LLM Client Compatibility Layer
Provides backwards-compatible GrokClient interface mapped to core.llm_client.LLMClient.
"""

from core.llm_client import LLMClient

# Alias GrokClient to LLMClient for seamless backward compatibility
GrokClient = LLMClient

__all__ = ["GrokClient", "LLMClient"]
