"""
Base Agent Framework

Provides abstract base class for all LLM-powered vision agents.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
import json
import base64
from pathlib import Path
import os


class VisionAgent(ABC):
    """
    Abstract base class for all LLM agents in the multi-agent system.
    
    Each agent is a specialized "witness" that provides specific insights
    about PDF form field detection.
    """
    
    def __init__(
        self,
        provider: str = "openai",  # "openai" or "anthropic"  
        model: str = "gpt-4o-mini",
        api_key: Optional[str] = None,
        temperature: float = 0.1,
        max_tokens: int = 1000
    ):
        """
        Initialize agent.
        
        Args:
            provider: LLM provider ("openai" or "anthropic")
            model: Model to use
            api_key: API key (defaults to env var)
            temperature: Sampling temperature (0.0-1.0)
            max_tokens: Max response tokens
        """
        self.provider = provider
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        
        # Initialize client
        if provider == "openai":
            try:
                import openai
                self.client = openai.OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))
            except ImportError:
                raise ImportError("openai package required. Install with: pip install openai")
        elif provider == "anthropic":
            try:
                from anthropic import Anthropic
                self.client = Anthropic(api_key=api_key or os.getenv("ANTHROPIC_API_KEY"))
            except ImportError:
                raise ImportError("anthropic package required. Install with: pip install anthropic")
        else:
            raise ValueError(f"Unsupported provider: {provider}")
        
        # Cost tracking
        self.cost_tracker = {
            "calls": 0,
            "total_tokens": 0,
            "total_cost": 0.0,
            "call_history": []
        }
    
    @abstractmethod
    def get_system_prompt(self) -> str:
        """
        Return the system prompt for this agent.
        
        This defines the agent's role and expertise.
        """
        pass
    
    @abstractmethod
    def get_user_prompt(self, **kwargs) -> str:
        """
        Generate user prompt with inputs.
        
        Args:
            **kwargs: Agent-specific inputs
        
        Returns:
            Formatted user prompt
        """
        pass
    
    @abstractmethod
    def parse_response(self, response: str) -> Dict[str, Any]:
        """
        Parse and validate agent response.
        
        Args:
            response: Raw LLM response
        
        Returns:
            Validated response dict
        
        Raises:
            ValueError: If response is invalid
        """
        pass
    
    def call(
        self,
        image: Optional[bytes] = None,
        image_detail: str = "high",
        **kwargs
    ) -> Dict[str, Any]:
        """
        Call the agent with inputs.
        
        Args:
            image: Optional image bytes (PNG/JPEG)
            image_detail: Detail level for vision ("low" or "high")
            **kwargs: Inputs for user prompt
        
        Returns:
            Parsed response dict
        """
        system_prompt = self.get_system_prompt()
        user_prompt = self.get_user_prompt(**kwargs)
        
        # Call appropriate provider
        if self.provider == "openai":
            response_text, tokens = self._call_openai(
                system_prompt, 
                user_prompt, 
                image,
                image_detail
            )
        elif self.provider == "anthropic":
            response_text, tokens = self._call_anthropic(
                system_prompt,
                user_prompt,
                image
            )
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")
        
        # Track cost
        cost = self._estimate_cost(tokens)
        self.cost_tracker["calls"] += 1
        self.cost_tracker["total_tokens"] += tokens
        self.cost_tracker["total_cost"] += cost
        self.cost_tracker["call_history"].append({
            "tokens": tokens,
            "cost": cost,
            "had_image": image is not None
        })
        
        # Parse and validate
        try:
            parsed = self.parse_response(response_text)
            parsed["_metadata"] = {
                "agent": self.__class__.__name__,
                "model": self.model,
                "tokens": tokens,
                "cost": cost
            }
            return parsed
        except json.JSONDecodeError as e:
            raise ValueError(f"Agent returned invalid JSON: {e}\nResponse: {response_text}")
        except Exception as e:
            raise ValueError(f"Failed to parse agent response: {e}")
    
    def _call_openai(
        self,
        system: str,
        user: str,
        image: Optional[bytes],
        image_detail: str
    ) -> tuple[str, int]:
        """
        Call OpenAI API.
        
        Returns:
            (response_text, token_count)
        """
        messages = [
            {"role": "system", "content": system}
        ]
        
        if image:
            # Encode image to base64
            image_b64 = base64.b64encode(image).decode('utf-8')
            
            # Determine MIME type (assume PNG, could be enhanced)
            mime_type = "image/png"
            
            messages.append({
                "role": "user",
                "content": [
                    {"type": "text", "text": user},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{mime_type};base64,{image_b64}",
                            "detail": image_detail
                        }
                    }
                ]
            })
        else:
            messages.append({"role": "user", "content": user})
        
        # Call API
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            response_format={"type": "json_object"},  # Force JSON output
            temperature=self.temperature,
            max_tokens=self.max_tokens
        )
        
        content = response.choices[0].message.content
        tokens = response.usage.total_tokens
        
        return content, tokens
    
    def _call_anthropic(
        self,
        system: str,
        user: str,
        image: Optional[bytes]
    ) -> tuple[str, int]:
        """
        Call Anthropic API.
        
        Returns:
            (response_text, token_count)
        """
        # Anthropic has different API structure
        content_parts = [{"type": "text", "text": user}]
        
        if image:
            # Encode image
            image_b64 = base64.b64encode(image).decode('utf-8')
            content_parts.append({
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/png",
                    "data": image_b64
                }
            })
        
        response = self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            system=system,
            messages=[{
                "role": "user",
                "content": content_parts
            }]
        )
        
        content = response.content[0].text
        tokens = response.usage.input_tokens + response.usage.output_tokens
        
        return content, tokens
    
    def _estimate_cost(self, tokens: int) -> float:
        """
        Estimate API cost based on model and tokens.
        
        Args:
            tokens: Total tokens used
        
        Returns:
            Estimated cost in USD
        """
        # Pricing as of 2024 (approximate)
        pricing = {
            "gpt-4o-mini": 0.15 / 1_000_000,  # $0.15 per 1M tokens (avg input/output)
            "gpt-4o": 2.50 / 1_000_000,        # $2.50 per 1M tokens
            "gpt-4-turbo": 10.00 / 1_000_000,  # $10.00 per 1M tokens
            "claude-3-haiku": 0.25 / 1_000_000,
            "claude-3-sonnet": 3.00 / 1_000_000,
            "claude-3-opus": 15.00 / 1_000_000
        }
        
        rate = pricing.get(self.model, 1.00 / 1_000_000)  # Default $1 per 1M
        return tokens * rate
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get cost and usage statistics.
        
        Returns:
            Dict with calls, tokens, cost, and call history
        """
        return {
            "calls": self.cost_tracker["calls"],
            "total_tokens": self.cost_tracker["total_tokens"],
            "total_cost": round(self.cost_tracker["total_cost"], 4),
            "avg_cost_per_call": round(
                self.cost_tracker["total_cost"] / max(self.cost_tracker["calls"], 1),
                4
            ),
            "call_history": self.cost_tracker["call_history"]
        }
    
    def reset_stats(self):
        """Reset cost tracking statistics."""
        self.cost_tracker = {
            "calls": 0,
            "total_tokens": 0,
            "total_cost": 0.0,
            "call_history": []
        }


class TextOnlyAgent(VisionAgent):
    """
    Base class for agents that don't need vision.
    
    Useful for referee, learning agents that work with structured data.
    """
    
    def call(self, **kwargs) -> Dict[str, Any]:
        """
        Call agent without image.
        
        Args:
            **kwargs: Inputs for user prompt
        
        Returns:
            Parsed response dict
        """
        return super().call(image=None, **kwargs)

