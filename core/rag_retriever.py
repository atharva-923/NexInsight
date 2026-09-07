"""
NexInsight - RAG Retriever Compatibility Layer
Provides backwards-compatible RAGRetriever interface mapped to core.rag_engine.RAGEngine.
"""

from core.rag_engine import RAGEngine

# Alias RAGRetriever to RAGEngine for seamless backward compatibility
class RAGRetriever(RAGEngine):
    @classmethod
    def build_grounded_context(cls, active_dataset, query, verified_result=None):
        return cls.retrieve_relevant_context(query, active_dataset, verified_result=verified_result)

__all__ = ["RAGRetriever", "RAGEngine"]
