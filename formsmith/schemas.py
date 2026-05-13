#!/usr/bin/env python3
"""
Field Definition Schemas

Single source of truth for field definitions.
Supports both PDF creation and interview YAML generation.
"""

from dataclasses import dataclass, field, asdict
from typing import List, Optional, Dict, Any
import json


@dataclass
class FieldDefinition:
    """
    Canonical field definition supporting PDF creation and interview-engine workflows (Docassemble-compatible).
    
    This is the single source of truth that bridges:
    - PDF field creation (PyMuPDF widgets)
    - interview generation (YAML)
    - Field validation and quality assurance
    """
    
    # === PDF-specific properties ===
    pdf_name: str                      # Internal PDF field name
    pdf_type: str                      # "text", "checkbox", "radio", "signature"
    bbox: List[float]                  # [x0, y0, x1, y1] in PDF coordinates
    page: int                          # 0-indexed page number
    
    # === Interview-engine properties (Docassemble-compatible) ===
    interview_variable: str                   # Variable name in interview (snake_case)
    interview_type: str                       # "text", "yesno", "date", "signature"
    interview_datatype: str                   # interview datatype for field
    interview_label: str                      # Human-readable label for questions
    interview_field_group: Optional[str] = None  # Logical grouping (petitioner_info, etc.)
    
    # === Detection metadata ===
    detection_method: str = "unknown"  # "learned_pattern", "opencv", "existing_field"
    confidence: float = 0.0            # 0.0 - 1.0 confidence score
    source_label: Optional[str] = None # Original label text from PDF
    
    # === Validation properties ===
    required: bool = True              # Is field required in interview?
    validation_rules: Dict[str, Any] = field(default_factory=dict)  # Validation constraints
    
    def to_pdf_widget_params(self) -> Dict[str, Any]:
        """
        Convert to PyMuPDF widget creation parameters.
        
        Returns:
            Dictionary suitable for fitz.Widget creation
        """
        widget_type_map = {
            "text": 7,       # fitz.PDF_WIDGET_TYPE_TEXT
            "checkbox": 2,   # fitz.PDF_WIDGET_TYPE_CHECKBOX
            "radio": 3,      # fitz.PDF_WIDGET_TYPE_RADIOBUTTON
            "signature": 8,  # fitz.PDF_WIDGET_TYPE_SIGNATURE
            "combobox": 4,   # fitz.PDF_WIDGET_TYPE_COMBOBOX
            "listbox": 5     # fitz.PDF_WIDGET_TYPE_LISTBOX
        }
        
        return {
            "field_name": self.pdf_name,
            "field_type": widget_type_map.get(self.pdf_type, 7),
            "rect": self.bbox,
            "page": self.page
        }
    
    def to_yaml_field(self) -> Dict[str, Any]:
        """
        Convert to interview YAML field definition.
        
        Returns:
            Dictionary for YAML question field
        """
        yaml_field = {
            "variable": self.interview_variable,
            "label": self.interview_label,
            "datatype": self.interview_datatype,
            "required": self.required
        }
        
        # Add validation if present
        if self.validation_rules:
            yaml_field.update(self.validation_rules)
        
        return yaml_field
    
    def to_attachment_field_mapping(self) -> Dict[str, str]:
        """
        Convert to interview attachment field mapping (Docassemble-style).
        
        Returns:
            Dictionary mapping PDF field name to interview variable
        """
        return {
            self.pdf_name: f"${{ {self.interview_variable} }}"
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'FieldDefinition':
        """Create from dictionary."""
        # Handle validation_rules default
        if 'validation_rules' not in data:
            data['validation_rules'] = {}
        return cls(**data)
    
    def __post_init__(self):
        """Validate field definition after initialization."""
        # Validate bbox
        if len(self.bbox) != 4:
            raise ValueError(f"bbox must have 4 values, got {len(self.bbox)}")
        
        x0, y0, x1, y1 = self.bbox
        if x1 <= x0:
            raise ValueError(f"Invalid bbox: x1 ({x1}) must be > x0 ({x0})")
        if y1 <= y0:
            raise ValueError(f"Invalid bbox: y1 ({y1}) must be > y0 ({y0})")
        
        # Validate confidence
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"confidence must be in [0, 1], got {self.confidence}")
        
        # Validate page
        if self.page < 0:
            raise ValueError(f"page must be >= 0, got {self.page}")


@dataclass
class FormDefinition:
    """
    Collection of fields representing a complete form.
    """
    
    pdf_name: str                      # Source PDF filename
    form_type: Optional[str] = None    # Form type identifier (ma_divorce_joint_petition)
    interview_module: Optional[str] = None  # Interview-engine module name (e.g., Docassemble package)
    fields: List[FieldDefinition] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def add_field(self, field_def: FieldDefinition):
        """Add a field to the form."""
        self.fields.append(field_def)
    
    def get_fields_by_page(self, page: int) -> List[FieldDefinition]:
        """Get all fields on a specific page."""
        return [f for f in self.fields if f.page == page]
    
    def get_fields_by_group(self, group: str) -> List[FieldDefinition]:
        """Get all fields in a logical group."""
        return [f for f in self.fields if f.interview_field_group == group]
    
    def get_field_groups(self) -> List[str]:
        """Get list of unique field groups."""
        groups = set()
        for field in self.fields:
            if field.interview_field_group:
                groups.add(field.interview_field_group)
        return sorted(list(groups))
    
    def get_low_confidence_fields(self, threshold: float = 0.75) -> List[FieldDefinition]:
        """Get fields with confidence below threshold."""
        return [f for f in self.fields if f.confidence < threshold]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "pdf_name": self.pdf_name,
            "form_type": self.form_type,
            "interview_module": self.interview_module,
            "total_fields": len(self.fields),
            "fields": [f.to_dict() for f in self.fields],
            "metadata": self.metadata
        }
    
    def to_json(self, indent: int = 2) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=indent)
    
    def save(self, output_path: str):
        """Save to JSON file."""
        with open(output_path, 'w') as f:
            f.write(self.to_json())
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'FormDefinition':
        """Create from dictionary."""
        fields = [FieldDefinition.from_dict(f) for f in data.get('fields', [])]
        return cls(
            pdf_name=data['pdf_name'],
            form_type=data.get('form_type'),
            interview_module=data.get('interview_module'),
            fields=fields,
            metadata=data.get('metadata', {})
        )
    
    @classmethod
    def load(cls, input_path: str) -> 'FormDefinition':
        """Load from JSON file."""
        with open(input_path, 'r') as f:
            data = json.load(f)
        return cls.from_dict(data)


@dataclass
class DetectionResult:
    """
    Result of field detection on a PDF.
    """
    
    pdf_path: str
    form_definition: FormDefinition
    detection_summary: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Calculate detection summary."""
        fields = self.form_definition.fields
        
        self.detection_summary = {
            "total_fields": len(fields),
            "fields_by_type": self._count_by_attribute(fields, 'pdf_type'),
            "fields_by_method": self._count_by_attribute(fields, 'detection_method'),
            "fields_by_group": self._count_by_attribute(fields, 'interview_field_group'),
            "avg_confidence": sum(f.confidence for f in fields) / len(fields) if fields else 0.0,
            "high_confidence_count": len([f for f in fields if f.confidence >= 0.85]),
            "medium_confidence_count": len([f for f in fields if 0.75 <= f.confidence < 0.85]),
            "low_confidence_count": len([f for f in fields if f.confidence < 0.75])
        }
    
    def _count_by_attribute(self, fields: List[FieldDefinition], attr: str) -> Dict[str, int]:
        """Count fields grouped by attribute."""
        counts = {}
        for field in fields:
            value = getattr(field, attr, None)
            if value:
                counts[value] = counts.get(value, 0) + 1
        return counts
    
    def print_summary(self):
        """Print detection summary."""
        print("\n" + "="*80)
        print("  DETECTION RESULT SUMMARY")
        print("="*80)
        print(f"\nPDF: {self.pdf_path}")
        print(f"Total Fields: {self.detection_summary['total_fields']}")
        print(f"Average Confidence: {self.detection_summary['avg_confidence']:.2f}")
        print(f"\nConfidence Distribution:")
        print(f"  • High (≥0.85): {self.detection_summary['high_confidence_count']}")
        print(f"  • Medium (0.75-0.85): {self.detection_summary['medium_confidence_count']}")
        print(f"  • Low (<0.75): {self.detection_summary['low_confidence_count']}")
        print(f"\nField Types:")
        for field_type, count in self.detection_summary['fields_by_type'].items():
            print(f"  • {field_type}: {count}")
        print(f"\nDetection Methods:")
        for method, count in self.detection_summary['fields_by_method'].items():
            print(f"  • {method}: {count}")
        if self.detection_summary['fields_by_group']:
            print(f"\nField Groups:")
            for group, count in self.detection_summary['fields_by_group'].items():
                print(f"  • {group}: {count}")

