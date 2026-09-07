"""
NexInsight - Groq AI Client
Handles communication with the Groq API with strict secret handling,
configurable model selection, timeout safeguards, and error isolation.
"""

import os
from typing import Dict, Any, List, Optional
import requests

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


class LLMClient:
    """Lightweight, resilient interface for the Groq API."""

    DEFAULT_MODEL = "openai/gpt-oss-120b"
    API_URL = "https://api.groq.com/openai/v1/chat/completions"

    @classmethod
    def get_api_key(cls, explicit_key: Optional[str] = None) -> Optional[str]:
        """
        Retrieves Groq API key with priority:
        1. Explicit parameter (if passed as non-None)
        2. os.environ['GROQ_API_KEY']
        3. os.environ['XAI_API_KEY'] (fallback compatibility)
        """
        if explicit_key is not None:
            return explicit_key.strip() if explicit_key.strip() else None
        env_key = os.environ.get("GROQ_API_KEY", "")
        if env_key and env_key.strip():
            return env_key.strip()
        fallback_key = os.environ.get("XAI_API_KEY", "")
        if fallback_key and fallback_key.strip():
            return fallback_key.strip()
        return None

    @classmethod
    def get_model(cls, explicit_model: Optional[str] = None) -> str:
        """Retrieves configured Groq model name."""
        if explicit_model and explicit_model.strip():
            return explicit_model.strip()
        env_model = os.environ.get("GROQ_MODEL", "")
        if env_model and env_model.strip():
            return env_model.strip()
        return cls.DEFAULT_MODEL

    @classmethod
    def is_configured(cls, explicit_key: Optional[str] = None) -> bool:
        """Returns True if a non-empty API key is present."""
        return bool(cls.get_api_key(explicit_key))

    @classmethod
    def test_connection(cls, explicit_key: Optional[str] = None, explicit_model: Optional[str] = None) -> Dict[str, Any]:
        """
        Sends a minimal verification ping to confirm API key validity.
        Never reveals the full key in outputs.
        """
        api_key = cls.get_api_key(explicit_key)
        if not api_key:
            return {"success": False, "error": "GROQ_API_KEY is not configured."}

        model = cls.get_model(explicit_model)
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": "Ping. Respond with OK."}],
            "max_tokens": 10,
            "temperature": 0.0
        }

        # Determine endpoint URL
        url = cls.API_URL
        if api_key.startswith("xai-"):
            url = "https://api.x.ai/v1/chat/completions"

        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=12)
            if resp.status_code == 200:
                data = resp.json()
                content = data["choices"][0]["message"]["content"].strip()
                return {"success": True, "message": f"Connected successfully to {model} via Groq.", "reply": content}
            elif resp.status_code == 401:
                return {"success": False, "error": "Invalid API Key (HTTP 401 Unauthorized)."}
            elif resp.status_code == 404:
                return {"success": False, "error": f"Configured model '{model}' was not found (HTTP 404)."}
            elif resp.status_code == 429:
                return {"success": False, "error": "Rate limit exceeded (HTTP 429)."}
            else:
                return {"success": False, "error": f"API returned HTTP status code {resp.status_code}."}
        except requests.exceptions.Timeout:
            return {"success": False, "error": "API connection timed out (12s threshold)."}
        except requests.exceptions.ConnectionError:
            return {"success": False, "error": "Network connection error reaching Groq API."}
        except Exception as e:
            return {"success": False, "error": f"API request error: {type(e).__name__}"}

    @classmethod
    def generate_chat_completion(
        cls,
        system_prompt: str,
        user_prompt: str,
        conversation_history: Optional[List[Dict[str, str]]] = None,
        explicit_key: Optional[str] = None,
        explicit_model: Optional[str] = None,
        temperature: float = 0.2,
        timeout: int = 20
    ) -> Dict[str, Any]:
        """
        Executes a grounded chat completion against the Groq API.
        Includes error handling, automatic timeout protection, and history formatting.
        """
        api_key = cls.get_api_key(explicit_key)
        if not api_key:
            return {
                "success": False,
                "error": "No Groq API Key configured. Please set GROQ_API_KEY environment variable.",
                "content": ""
            }

        model = cls.get_model(explicit_model)
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }

        # Build messages payload with short conversational context
        messages = [{"role": "system", "content": system_prompt}]

        if conversation_history:
            # Keep only the last 4 turns to conserve tokens and cost
            recent_turns = conversation_history[-4:]
            for turn in recent_turns:
                role = turn.get("role", "user")
                content = turn.get("content", "")
                if role in ["user", "assistant"] and content:
                    messages.append({"role": role, "content": content})

        messages.append({"role": "user", "content": user_prompt})

        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": 1024
        }

        url = cls.API_URL
        if api_key.startswith("xai-"):
            url = "https://api.x.ai/v1/chat/completions"

        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=timeout)
            if resp.status_code == 200:
                data = resp.json()
                content = data["choices"][0]["message"]["content"].strip()
                content = content.replace("\u202f", " ").replace("\u00a0", " ")
                tokens_used = data.get("usage", {}).get("total_tokens", 0)
                return {
                    "success": True,
                    "content": content,
                    "model": model,
                    "tokens_used": tokens_used
                }
            elif resp.status_code == 401:
                return {
                    "success": False,
                    "error": "Authentication failed: Invalid Groq API Key (HTTP 401).",
                    "content": ""
                }
            elif resp.status_code == 404:
                return {
                    "success": False,
                    "error": f"Model '{model}' is unavailable on Groq. Please select an available model (e.g. 'openai/gpt-oss-120b', 'openai/gpt-oss-20b', 'qwen/qwen3.8-27b') in Settings.",
                    "content": ""
                }
            elif resp.status_code == 429:
                return {
                    "success": False,
                    "error": "Groq rate limit exceeded. Please wait a moment before querying again.",
                    "content": ""
                }
            else:
                return {
                    "success": False,
                    "error": f"Groq API returned HTTP {resp.status_code}.",
                    "content": ""
                }
        except requests.exceptions.Timeout:
            return {
                "success": False,
                "error": f"Groq request timed out after {timeout} seconds.",
                "content": ""
            }
        except requests.exceptions.ConnectionError:
            return {
                "success": False,
                "error": "Failed to connect to Groq endpoint. Check internet connection.",
                "content": ""
            }
        except Exception as e:
            return {
                "success": False,
                "error": f"API error: {type(e).__name__}",
                "content": ""
            }


# Backwards compatibility alias
GrokClient = LLMClient
