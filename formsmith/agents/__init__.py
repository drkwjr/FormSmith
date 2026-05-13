"""
Multi-Agent System for PDF Field Detection

Specialized LLM agents that provide "vision" and "understanding" to the field detection system.
"""

from .base_agent import VisionAgent, TextOnlyAgent
from .field_spotter import FieldSpotterAgent
from .layout_analyst import LayoutAnalystAgent
from .validator import ValidatorAgent
from .referee import RefereeAgent
from .position_advisor import PositionAdvisorAgent
from .learning_agent import LearningAgent

__all__ = [
    'VisionAgent',
    'TextOnlyAgent',
    'FieldSpotterAgent',
    'LayoutAnalystAgent',
    'ValidatorAgent',
    'RefereeAgent',
    'PositionAdvisorAgent',
    'LearningAgent'
]
