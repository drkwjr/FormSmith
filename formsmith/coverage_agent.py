#!/usr/bin/env python3
"""Agent A – coverage assurance for PDF fields"""

import json
import os
from dataclasses import dataclass
from typing import Dict, List

import fitz  # PyMuPDF
from PIL import Image
from openai import OpenAI
from dotenv import load_dotenv


load_dotenv()


@dataclass
class CoverageReport:
    detected_field_count: int
    target_region_count: int
    missing_targets: List[Dict]
    extra_fields: List[Dict]
    notes: str
    raw_response: Dict
    token_usage: Dict

    def to_dict(self) -> Dict:
        return {
            "detected_field_count": self.detected_field_count,
            "target_region_count": self.target_region_count,
            "missing_targets": self.missing_targets,
            "extra_fields": self.extra_fields,
            "notes": self.notes,
            "raw_response": self.raw_response,
            "token_usage": self.token_usage,
        }


def _load_json(path: str) -> Dict:
    with open(path, "r") as f:
        return json.load(f)


def _render_pdf_page(pdf_path: str, page_index: int = 0, dpi: int = 150) -> str:
    """Render PDF page to base64 PNG string for LLM vision"""
    doc = fitz.open(pdf_path)
    if page_index >= len(doc):
        raise ValueError(f"PDF only has {len(doc)} pages; page_index {page_index} invalid")

    page = doc[page_index]
    matrix = fitz.Matrix(dpi / 72, dpi / 72)
    pix = page.get_pixmap(matrix=matrix)
    image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    doc.close()

    import base64
    from io import BytesIO

    buf = BytesIO()
    image.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode()
    return f"data:image/png;base64,{b64}"


def run_coverage_agent(
    pdf_path: str,
    fields_json: str,
    targets_json: str,
    output_path: str,
    model: str = "gpt-5-mini",
    page_index: int = 0,
) -> CoverageReport:
    fields = _load_json(fields_json)
    targets = _load_json(targets_json)

    field_list = fields.get("fields", [])
    target_list = targets.get("regions", [])

    pdf_image = _render_pdf_page(pdf_path, page_index=page_index)

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    prompt = (
        "You are Agent A: an obsessive coverage inspector for court form fields.\n"
        "Goal: ensure every visual fillable area has a corresponding field entry, and no field is missing a visual target.\n\n"
        "Input includes: (1) page image, (2) JSON of detected fields with bounding boxes, (3) JSON of target regions derived from underlines/checkboxes.\n\n"
        "Rules:\n"
        "- Only trust what you can see on the form image.\n"
        "- Treat target regions as the authoritative list of fillable areas.\n"
        "- Compare each target with the detected fields.\n"
        "- If a target has no field covering it, mark it missing.\n"
        "- If a field lacks a matching target area, mark it extra.\n"
        "- Provide short notes for ambiguous or hard-to-see cases.\n\n"
        "Respond strictly in JSON with keys: missing_targets (array of target IDs), extra_fields (array of field names), notes (short string)."
    )

    messages = [
        {
            "role": "system",
            "content": "You are a meticulous QA inspector for legal form coverage.",
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "text", "text": f"Detected fields JSON:\n{json.dumps(field_list)}"},
                {"type": "text", "text": f"Target regions JSON:\n{json.dumps(target_list)}"},
                {"type": "image_url", "image_url": {"url": pdf_image, "detail": "high"}},
            ],
        },
    ]

    response = client.chat.completions.create(
        model=model,
        messages=messages,
        response_format={"type": "json_object"},
    )

    usage = response.usage
    content = json.loads(response.choices[0].message.content)

    missing_targets = content.get("missing_targets", [])
    extra_fields = content.get("extra_fields", [])
    notes = content.get("notes", "")

    report = CoverageReport(
        detected_field_count=len(field_list),
        target_region_count=len(target_list),
        missing_targets=missing_targets,
        extra_fields=extra_fields,
        notes=notes,
        raw_response=content,
        token_usage={
            "prompt_tokens": usage.prompt_tokens,
            "completion_tokens": usage.completion_tokens,
            "total_tokens": usage.total_tokens,
        },
    )

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(report.to_dict(), f, indent=2)

    return report


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Run Agent A coverage check")
    parser.add_argument("pdf", help="Path to PDF form")
    parser.add_argument("fields", help="Path to fields JSON export")
    parser.add_argument("targets", help="Path to target regions JSON")
    parser.add_argument("output", help="Where to write coverage report JSON")
    parser.add_argument("--page", type=int, default=0)
    parser.add_argument("--model", default="gpt-5-mini")

    args = parser.parse_args()

    run_coverage_agent(
        pdf_path=args.pdf,
        fields_json=args.fields,
        targets_json=args.targets,
        output_path=args.output,
        model=args.model,
        page_index=args.page,
    )


if __name__ == "__main__":
    main()

