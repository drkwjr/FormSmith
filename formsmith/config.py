"""
Centralized Configuration Management

Loads settings from .env and provides validation and budget tracking.
"""

import os
from pathlib import Path
from dotenv import load_dotenv
from typing import Optional

# Load .env file
load_dotenv()


class Config:
    """Centralized configuration management."""
    
    # API Keys
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    OPENAI_ORG_ID: str = os.getenv("OPENAI_ORG_ID", "")
    ANTHROPIC_API_KEY: str = os.getenv("ANTHROPIC_API_KEY", "")
    
    # Models
    OPENAI_MODEL: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    ANTHROPIC_MODEL: str = os.getenv("ANTHROPIC_MODEL", "claude-3-haiku-20240307")
    
    # Agent Settings
    USE_LLM_AGENTS: bool = os.getenv("USE_LLM_AGENTS", "true").lower() == "true"
    MAX_LLM_CALLS_PER_FORM: int = int(os.getenv("MAX_LLM_CALLS_PER_FORM", "50"))
    AGENT_CONFIDENCE_THRESHOLD: float = float(os.getenv("AGENT_CONFIDENCE_THRESHOLD", "0.6"))
    LEARNING_MODE: bool = os.getenv("LEARNING_MODE", "true").lower() == "true"
    
    # Cost Tracking
    MONTHLY_BUDGET_USD: float = float(os.getenv("MONTHLY_BUDGET_USD", "100.0"))
    ALERT_THRESHOLD_USD: float = float(os.getenv("ALERT_THRESHOLD_USD", "80.0"))
    
    # File Persistence
    OUTPUT_DIR: Path = Path(os.getenv("OUTPUT_DIR", "output"))
    BACKUP_DIR: Path = Path(os.getenv("BACKUP_DIR", "backups"))
    AUTO_BACKUP: bool = os.getenv("AUTO_BACKUP", "true").lower() == "true"
    
    @classmethod
    def validate(cls) -> list[str]:
        """Validate configuration and return list of errors."""
        errors = []
        
        if not cls.OPENAI_API_KEY:
            errors.append("OPENAI_API_KEY not set in .env")
        
        if cls.USE_LLM_AGENTS and not (cls.OPENAI_API_KEY or cls.ANTHROPIC_API_KEY):
            errors.append("USE_LLM_AGENTS=true but no API keys provided")
        
        # Create directories if they don't exist
        cls.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        if cls.AUTO_BACKUP:
            cls.BACKUP_DIR.mkdir(parents=True, exist_ok=True)
        
        return errors
    
    @classmethod
    def check_budget(cls, current_cost: float) -> tuple[bool, str]:
        """Check if within budget."""
        if current_cost >= cls.MONTHLY_BUDGET_USD:
            return False, f"Monthly budget exceeded: ${current_cost:.2f} / ${cls.MONTHLY_BUDGET_USD:.2f}"
        elif current_cost >= cls.ALERT_THRESHOLD_USD:
            return True, f"Warning: Approaching budget limit: ${current_cost:.2f} / ${cls.MONTHLY_BUDGET_USD:.2f}"
        return True, ""
    
    @classmethod
    def to_dict(cls) -> dict:
        """Export configuration as dict (for logging/debugging)."""
        return {
            "openai_model": cls.OPENAI_MODEL,
            "anthropic_model": cls.ANTHROPIC_MODEL,
            "use_llm_agents": cls.USE_LLM_AGENTS,
            "max_llm_calls_per_form": cls.MAX_LLM_CALLS_PER_FORM,
            "agent_confidence_threshold": cls.AGENT_CONFIDENCE_THRESHOLD,
            "learning_mode": cls.LEARNING_MODE,
            "monthly_budget_usd": cls.MONTHLY_BUDGET_USD,
            "alert_threshold_usd": cls.ALERT_THRESHOLD_USD,
            "output_dir": str(cls.OUTPUT_DIR),
            "backup_dir": str(cls.BACKUP_DIR),
            "auto_backup": cls.AUTO_BACKUP,
            "api_keys_configured": {
                "openai": bool(cls.OPENAI_API_KEY),
                "anthropic": bool(cls.ANTHROPIC_API_KEY)
            }
        }

