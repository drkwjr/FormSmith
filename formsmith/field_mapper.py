#!/usr/bin/env python3
"""
Field Mapper

Single source of truth for mapping PDF fields to interview variables.
Handles type inference, naming conventions, and metadata enrichment.
"""

import re
from typing import Dict, Optional, Any, List, Tuple


class FieldMapper:
    """
    Maps PDF field properties to interview conventions.
    
    This is the single source of truth for:
    - PDF field type → interview datatype
    - Label text → interview variable name
    - Field metadata → Interview question properties
    """
    
    # PDF type to interview datatype mapping
    TYPE_MAPPING = {
        "text": "text",
        "checkbox": "yesno",
        "radio": "radio",
        "signature": "signature",
        "combobox": "dropdown",
        "listbox": "list"
    }
    
    # Keywords for semantic type inference
    TYPE_KEYWORDS = {
        "date": ["date", "dob", "birth", "filed", "signed", "effective"],
        "email": ["email", "e-mail"],
        "phone": ["phone", "telephone", "tel", "fax"],
        "ssn": ["ssn", "social security"],
        "zip": ["zip", "postal"],
        "currency": ["income", "amount", "fee", "cost", "price", "salary"],
        "number": ["number", "count", "quantity", "age"]
    }
    
    # Legal vocabulary for MA court forms
    LEGAL_VOCABULARY = {
        "petitioner", "defendant", "plaintiff", "respondent",
        "docket", "bbo", "case", "court", "judge", "attorney",
        "name", "address", "city", "state", "zip", "phone", "email",
        "date", "birth", "marriage", "divorce", "custody",
        "child", "children", "support", "alimony",
        "property", "asset", "debt", "income", "indigent"
    }
    
    def __init__(self, learned_patterns: Optional[Dict] = None):
        """
        Initialize mapper with optional learned patterns.
        
        Args:
            learned_patterns: Dictionary of learned naming/typing patterns
        """
        self.learned_patterns = learned_patterns or {}
        self.naming_patterns = learned_patterns.get('naming', {}) if learned_patterns else {}
        self.used_names = set()  # Track used names for disambiguation
        
    def map_field_type(self, pdf_field_type: str, label_text: Optional[str] = None) -> Tuple[str, str]:
        """
        Map PDF field type to interview type and datatype.
        
        Args:
            pdf_field_type: PDF field type (text, checkbox, etc.)
            label_text: Optional label text for semantic inference
            
        Returns:
            Tuple of (interview_type, interview_datatype)
        """
        # Base mapping
        interview_type = self.TYPE_MAPPING.get(pdf_field_type, "text")
        interview_datatype = interview_type
        
        # Semantic inference for text fields
        if pdf_field_type == "text" and label_text:
            label_lower = label_text.lower()
            
            # Check for semantic types
            for semantic_type, keywords in self.TYPE_KEYWORDS.items():
                if any(keyword in label_lower for keyword in keywords):
                    interview_datatype = semantic_type
                    break
        
        return interview_type, interview_datatype
    
    def generate_field_name(self, 
                          label_text: Optional[str],
                          field_type: str,
                          field_index: int = 0,
                          pdf_field_name: Optional[str] = None) -> str:
        """
        Generate interview-compatible variable name.
        
        Priority:
        1. Check learned naming patterns (from examples)
        2. Parse PDF field name if available and meaningful
        3. Generate from label text
        4. Fallback to generic name
        
        Args:
            label_text: Label text from PDF
            field_type: Field type (text, checkbox, etc.)
            field_index: Index for disambiguation
            pdf_field_name: Original PDF field name
            
        Returns:
            interview-compatible variable name (snake_case)
        """
        # 1. Check learned patterns first
        if label_text and self.naming_patterns:
            label_mappings = self.naming_patterns.get('label_mappings', {})
            if label_text in label_mappings:
                return self._disambiguate_name(label_mappings[label_text])
        
        # 2. Try to use PDF field name if meaningful
        if pdf_field_name and self._is_meaningful_name(pdf_field_name):
            name = self._normalize_name(pdf_field_name)
            if name:
                return self._disambiguate_name(name)
        
        # 3. Generate from label text
        if label_text:
            name = self._generate_name_from_label(label_text, field_type)
            if name:
                return self._disambiguate_name(name)
        
        # 4. Fallback to generic name
        generic_name = f"{field_type}_field_{field_index}"
        return self._disambiguate_name(generic_name)
    
    def _is_meaningful_name(self, name: str) -> bool:
        """
        Check if PDF field name is meaningful (not auto-generated).
        """
        # Reject obviously auto-generated names
        auto_patterns = [
            r'^field_?\d+$',
            r'^text_?\d+$',
            r'^check_?\d+$',
            r'^unnamed',
            r'^widget',
            r'^form_?field'
        ]
        
        name_lower = name.lower()
        for pattern in auto_patterns:
            if re.match(pattern, name_lower):
                return False
        
        return True
    
    def _generate_name_from_label(self, label_text: str, field_type: str) -> Optional[str]:
        """
        Generate field name from label text.
        
        Examples:
            "Petitioner's Name:" → "petitioner_name"
            "Date of Birth:" → "date_of_birth"
            "☐ Married" → "is_married" (checkbox)
        """
        # Remove common punctuation and prefixes
        text = label_text.strip()
        
        # Remove checkbox symbols
        text = re.sub(r'^[☐☑✓✗\[\]]+\s*', '', text)
        
        # Remove trailing punctuation
        text = re.sub(r'[:.?!]+$', '', text)
        
        # Remove possessive 's
        text = text.replace("'s", "")
        
        # Split into words
        words = re.findall(r'\b\w+\b', text)
        
        if not words:
            return None
        
        # Convert to snake_case
        name_parts = [word.lower() for word in words if len(word) > 1]
        
        if not name_parts:
            return None
        
        # Apply naming rules
        name = '_'.join(name_parts)
        
        # Add prefix for checkboxes
        if field_type == "checkbox" and not name.startswith('is_'):
            name = f"is_{name}"
        
        # Normalize with legal vocabulary
        name = self._apply_vocabulary_normalization(name)
        
        return name
    
    def _normalize_name(self, name: str) -> Optional[str]:
        """
        Normalize any name to interview conventions.
        """
        # Convert to lowercase
        name = name.lower()
        
        # Replace non-alphanumeric with underscore
        name = re.sub(r'[^a-z0-9_]', '_', name)
        
        # Remove multiple underscores
        name = re.sub(r'_+', '_', name)
        
        # Remove leading/trailing underscores
        name = name.strip('_')
        
        # Must start with letter
        if name and not name[0].isalpha():
            name = 'field_' + name
        
        return name if name else None
    
    def _apply_vocabulary_normalization(self, name: str) -> str:
        """
        Normalize using legal vocabulary (prefer standard terms).
        
        Example: "plaintiff" → "petitioner" for MA divorce forms
        """
        # Common synonyms in legal forms
        synonyms = {
            'plaintiff': 'petitioner',  # MA divorce uses petitioner
            'fname': 'first_name',
            'lname': 'last_name',
            'dob': 'date_of_birth',
            'addr': 'address',
            'tel': 'phone',
            'ph': 'phone',
        }
        
        # Replace parts
        for old, new in synonyms.items():
            name = name.replace(old, new)
        
        return name
    
    def _disambiguate_name(self, name: str) -> str:
        """
        Add suffix if name already used.
        """
        if name not in self.used_names:
            self.used_names.add(name)
            return name
        
        # Add numeric suffix
        counter = 2
        while f"{name}_{counter}" in self.used_names:
            counter += 1
        
        disambiguated = f"{name}_{counter}"
        self.used_names.add(disambiguated)
        return disambiguated
    
    def infer_field_properties(self, 
                              field_def: Dict[str, Any],
                              label_text: Optional[str] = None) -> Dict[str, Any]:
        """
        Add interview-engine-specific metadata to field definition.
        
        Args:
            field_def: Basic field definition (bbox, type, etc.)
            label_text: Optional label text
            
        Returns:
            Enhanced field definition with interview metadata
        """
        pdf_type = field_def.get('type', 'text')
        pdf_name = field_def.get('name', '')
        
        # Map types
        interview_type, interview_datatype = self.map_field_type(pdf_type, label_text)
        
        # Generate variable name
        interview_variable = self.generate_field_name(
            label_text=label_text,
            field_type=pdf_type,
            field_index=field_def.get('index', 0),
            pdf_field_name=pdf_name
        )
        
        # Generate human-readable label for interview
        interview_label = self._generate_question_label(label_text, interview_variable)
        
        # Infer field grouping
        interview_field_group = self._infer_field_group(interview_variable)
        
        # Add validation rules
        validation_rules = self._infer_validation_rules(interview_datatype, label_text)
        
        # Enhanced field definition
        enhanced = {
            **field_def,  # Keep original properties
            "interview_variable": interview_variable,
            "interview_type": interview_type,
            "interview_datatype": interview_datatype,
            "interview_label": interview_label,
            "interview_field_group": interview_field_group,
            "validation": validation_rules,
            "required": self._is_likely_required(label_text, interview_variable)
        }
        
        return enhanced
    
    def _generate_question_label(self, label_text: Optional[str], variable_name: str) -> str:
        """
        Generate human-readable label for interview question.
        """
        if label_text:
            # Clean up label text
            label = label_text.strip()
            label = re.sub(r'^[☐☑✓✗\[\]]+\s*', '', label)  # Remove checkbox symbols
            label = re.sub(r'[:.]+$', '', label)  # Remove trailing punctuation
            return label
        
        # Generate from variable name
        words = variable_name.replace('_', ' ').split()
        if words and words[0] == 'is':
            # Checkbox: "is_married" → "Married?"
            return ' '.join(words[1:]).capitalize() + '?'
        else:
            # Regular field: "petitioner_name" → "Petitioner name"
            return ' '.join(words).capitalize()
    
    def _infer_field_group(self, variable_name: str) -> Optional[str]:
        """
        Infer logical field grouping from variable name.
        
        Groups fields for better interview organization.
        """
        name_lower = variable_name.lower()
        
        if 'petitioner' in name_lower:
            return 'petitioner_info'
        elif 'defendant' in name_lower or 'respondent' in name_lower:
            return 'defendant_info'
        elif any(word in name_lower for word in ['docket', 'case', 'court']):
            return 'case_info'
        elif any(word in name_lower for word in ['child', 'children', 'custody']):
            return 'children_info'
        elif any(word in name_lower for word in ['property', 'asset', 'debt', 'income']):
            return 'financial_info'
        elif 'attorney' in name_lower or 'bbo' in name_lower:
            return 'attorney_info'
        
        return None
    
    def _infer_validation_rules(self, datatype: str, label_text: Optional[str]) -> Dict[str, Any]:
        """
        Infer validation rules based on field type.
        """
        rules = {}
        
        if datatype == "email":
            rules["pattern"] = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        elif datatype == "phone":
            rules["pattern"] = r'^\(?([0-9]{3})\)?[-. ]?([0-9]{3})[-. ]?([0-9]{4})$'
        elif datatype == "zip":
            rules["pattern"] = r'^\d{5}(-\d{4})?$'
        elif datatype == "ssn":
            rules["pattern"] = r'^\d{3}-?\d{2}-?\d{4}$'
        elif datatype == "text":
            # Default text field limits
            rules["max_length"] = 200
        elif datatype == "number":
            rules["min"] = 0
        elif datatype == "currency":
            rules["min"] = 0
            rules["currency_symbol"] = "$"
        
        return rules
    
    def _is_likely_required(self, label_text: Optional[str], variable_name: str) -> bool:
        """
        Infer if field is likely required.
        
        Most fields are required unless they're optional checkboxes or
        explicitly marked as optional.
        """
        if label_text:
            label_lower = label_text.lower()
            if 'optional' in label_lower or '(if applicable)' in label_lower:
                return False
        
        # Names, addresses, dates are typically required
        required_keywords = ['name', 'address', 'date', 'docket', 'case']
        if any(keyword in variable_name for keyword in required_keywords):
            return True
        
        # Default to required
        return True


# Convenience function for quick mapping
def map_pdf_field_to_interview(field_def: Dict[str, Any],
                                  label_text: Optional[str] = None,
                                  learned_patterns: Optional[Dict] = None) -> Dict[str, Any]:
    """
    Quick function to map a PDF field to interview format.
    
    Args:
        field_def: PDF field definition
        label_text: Optional label text
        learned_patterns: Optional learned patterns
        
    Returns:
        Enhanced field definition with interview metadata
    """
    mapper = FieldMapper(learned_patterns)
    return mapper.infer_field_properties(field_def, label_text)

