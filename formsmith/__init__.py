"""
FormSmith — PDF form analysis, field detection, and intelligent placement.

A pipeline for taking a blank PDF, understanding its structure, detecting where
form fields belong, placing them precisely, and emitting interview-engine-ready
artifacts (Docassemble-compatible YAML, or generic JSON).

Pipeline stages:
    analyze → detect → place → validate → emit

The vision-agent stage (formsmith.agents) is optional and requires a
vision-capable LLM API key (OpenAI, Anthropic Claude, Google Gemini).
"""

from .pattern_learner import PatternLearner
from .field_mapper import FieldMapper
from .learned_field_detector import LearnedFieldDetector
from .schemas import FieldDefinition, FormDefinition, DetectionResult
from .output_formatter import FieldCollectionExporter, export_all_formats
from .interview_yaml_generator import InterviewYAMLGenerator

__all__ = [
    "PatternLearner",
    "FieldMapper",
    "LearnedFieldDetector",
    "FieldDefinition",
    "FormDefinition",
    "DetectionResult",
    "FieldCollectionExporter",
    "export_all_formats",
    "InterviewYAMLGenerator",
]

__version__ = "0.1.0"
