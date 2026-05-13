#!/usr/bin/env python3
"""Agent B – precision refinement for PDF field placement"""

import base64
import json
import os
from dataclasses import dataclass
from io import BytesIO
from typing import Dict, List, Tuple

import fitz  # PyMuPDF
from PIL import Image, ImageDraw
from openai import OpenAI
from dotenv import load_dotenv


load_dotenv()


# Estimated pricing constants for gpt-5-mini (USD per token)
# Adjust these if OpenAI updates the published rates.
PROMPT_COST_PER_TOKEN = 0.000003
COMPLETION_COST_PER_TOKEN = 0.000006


@dataclass
class Refinement:
    field_name: str
    field_type: str
    original_bbox: Tuple[float, float, float, float]
    offsets: Tuple[float, float, float, float]
    confidence: float
    notes: str

    def apply(self) -> Tuple[float, float, float, float]:
        x0, y0, x1, y1 = self.original_bbox
        dx, dy, dw, dh = self.offsets
        return (x0 + dx, y0 + dy, x1 + dx + dw, y1 + dy + dh)

    def to_dict(self) -> Dict:
        return {
            "field_name": self.field_name,
            "field_type": self.field_type,
            "original_bbox": list(self.original_bbox),
            "offsets": list(self.offsets),
            "confidence": self.confidence,
            "notes": self.notes,
            "adjusted_bbox": list(self.apply()),
        }


def _load_json(path: str) -> Dict:
    with open(path, "r") as f:
        return json.load(f)


def _bbox_to_rect(bbox: List[float]) -> fitz.Rect:
    return fitz.Rect(bbox[0], bbox[1], bbox[2], bbox[3])


def _clip_margin(rect: fitz.Rect, margin: float, page_rect: fitz.Rect) -> fitz.Rect:
    expanded = fitz.Rect(
        rect.x0 - margin,
        rect.y0 - margin,
        rect.x1 + margin,
        rect.y1 + margin,
    )
    return expanded & page_rect


def _render_field_crop(
    doc: fitz.Document,
    page_index: int,
    bbox: List[float],
    margin: float = 12,
    dpi: int = 200,
) -> Tuple[str, int, int]:
    page = doc[page_index]
    page_rect = page.rect
    rect = _bbox_to_rect(bbox)
    clip = _clip_margin(rect, margin, page_rect)

    matrix = fitz.Matrix(dpi / 72, dpi / 72)
    pix = page.get_pixmap(matrix=matrix, clip=clip)
    image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

    # Draw the original bbox overlay (converted to clipped coordinates)
    draw = ImageDraw.Draw(image, "RGBA")

    scale_x = pix.width / clip.width
    scale_y = pix.height / clip.height

    box_left = (rect.x0 - clip.x0) * scale_x
    box_right = (rect.x1 - clip.x0) * scale_x
    box_bottom = (clip.y1 - rect.y0) * scale_y
    box_top = (clip.y1 - rect.y1) * scale_y

    draw.rectangle(
        [box_left, box_top, box_right, box_bottom],
        outline=(255, 0, 0, 255),
        width=3,
    )

    buffer = BytesIO()
    image.save(buffer, format="PNG")
    b64 = base64.b64encode(buffer.getvalue()).decode()

    return f"data:image/png;base64,{b64}", pix.width, pix.height


def _call_refinement_llm(
    client: OpenAI,
    model: str,
    field: Dict,
    crop_image: str,
    target_region: Dict = None,
    max_offset: float = 30.0,
) -> Dict:
    field_name = field.get("name", "unnamed")
    field_type = field.get("type", "unknown")
    bbox = field.get("bbox", [])

    target_text = (
        f"Target region metadata: {json.dumps(target_region)}\n"
        if target_region
        else ""
    )

    prompt = (
        "You are Agent B: a precision field placement expert for court forms.\n"
        "Task: inspect the cropped image (with red box showing current field) and suggest adjustments so the field perfectly covers the intended form area.\n\n"
        "Constraints:\n"
        "- Provide numeric offsets in PDF coordinate space (same orientation as bbox).\n"
        "- Offsets are limited to +/- {max_offset} pixels.\n"
        "- Return JSON with keys offsets {{\"dx\": ..., \"dy\": ..., \"dw\": ..., \"dh\": ...}}, confidence (0-1), and notes.\n"
        "- Offsets semantics: new_x0 = x0 + dx, new_y0 = y0 + dy, new_x1 = x1 + dx + dw, new_y1 = y1 + dy + dh.\n"
        "- If no change is needed, set offsets to 0.\n"
        "- If unsure, keep offsets 0 and explain in notes.\n"
    ).format(max_offset=max_offset)

    messages = [
        {
            "role": "system",
            "content": "You are a meticulous layout specialist ensuring fields align with their printed targets.",
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {
                    "type": "text",
                    "text": (
                        f"Field name: {field_name}\nField type: {field_type}\n"
                        f"Current bbox: {bbox}\n{target_text}"
                    ),
                },
                {"type": "image_url", "image_url": {"url": crop_image, "detail": "high"}},
            ],
        },
    ]

    response = client.chat.completions.create(
        model=model,
        messages=messages,
        response_format={"type": "json_object"},
    )

    return {
        "content": json.loads(response.choices[0].message.content),
        "usage": {
            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
            "total_tokens": response.usage.total_tokens,
        },
    }


def run_refinement(
    pdf_path: str,
    fields_json: str,
    targets_json: str,
    output_json: str,
    model: str = "gpt-5-mini",
    page_index: int = 0,
    max_offset: float = 30.0,
    baseline_json: str | None = None,
) -> Dict:
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    fields = _load_json(fields_json)
    targets = _load_json(targets_json)

    field_list = fields.get("refinements") or fields.get("fields", [])
    target_map = {region["id"]: region for region in targets.get("regions", [])}

    baseline_map = {}
    if baseline_json:
        baseline_data = _load_json(baseline_json)
        for entry in baseline_data.get("refinements", []):
            name = entry.get("field_name") or entry.get("name")
            if name:
                baseline_map[name] = entry.get("adjusted_bbox")

    doc = fitz.open(pdf_path)

    results: List[Refinement] = []
    token_summary = {"prompt": 0, "completion": 0, "total": 0}

    for field in field_list:
        field_name = field.get("field_name") or field.get("name") or "unnamed"
        if field.get("page", page_index) != page_index:
            continue

        bbox = field.get("adjusted_bbox") or field.get("bbox")
        if not bbox or len(bbox) != 4:
            bbox = baseline_map.get(field_name)
        if not bbox or len(bbox) != 4:
            continue

        bbox = list(bbox)
        field["bbox"] = bbox

        crop_image, _, _ = _render_field_crop(doc, page_index, bbox)

        target_id = field.get("target_id")
        response = _call_refinement_llm(
            client,
            model,
            field,
            crop_image,
            target_region=target_map.get(target_id),
            max_offset=max_offset,
        )

        content = response["content"]
        usage = response["usage"]

        offsets_dict = content.get("offsets", {}) or {}
        dx = offsets_dict.get("dx", 0)
        dy = offsets_dict.get("dy", 0)
        dw = offsets_dict.get("dw", 0)
        dh = offsets_dict.get("dh", 0)

        def clamp(value: float) -> float:
            return max(-max_offset, min(max_offset, float(value)))

        offsets = (clamp(dx), clamp(dy), clamp(dw), clamp(dh))

        refinement = Refinement(
            field_name=field_name,
            field_type=field.get("field_type") or field.get("type", "unknown"),
            original_bbox=tuple(bbox),
            offsets=offsets,
            confidence=float(content.get("confidence", 0.0)),
            notes=str(content.get("notes", "")),
        )
        results.append(refinement)

        token_summary["prompt"] += usage["prompt_tokens"]
        token_summary["completion"] += usage["completion_tokens"]
        token_summary["total"] += usage["total_tokens"]

    doc.close()

    cost_usd = (
        token_summary["prompt"] * PROMPT_COST_PER_TOKEN
        + token_summary["completion"] * COMPLETION_COST_PER_TOKEN
    )

    output = {
        "pdf": os.path.basename(pdf_path),
        "model": model,
        "max_offset": max_offset,
        "baseline": baseline_json,
        "refinements": [r.to_dict() for r in results],
        "token_usage": token_summary,
        "cost_usd": cost_usd,
    }

    os.makedirs(os.path.dirname(output_json), exist_ok=True)
    with open(output_json, "w") as f:
        json.dump(output, f, indent=2)

    return output


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Run Agent B refinement")
    parser.add_argument("pdf", help="Path to PDF form")
    parser.add_argument("fields", help="Path to fields JSON export")
    parser.add_argument("targets", help="Path to target regions JSON")
    parser.add_argument("output", help="Where to write refined fields JSON")
    parser.add_argument("--model", default="gpt-5-mini")
    parser.add_argument("--page", type=int, default=0)
    parser.add_argument("--max-offset", type=float, default=30.0)

    args = parser.parse_args()

    run_refinement(
        pdf_path=args.pdf,
        fields_json=args.fields,
        targets_json=args.targets,
        output_json=args.output,
        model=args.model,
        page_index=args.page,
        max_offset=args.max_offset,
    )


if __name__ == "__main__":
    main()

