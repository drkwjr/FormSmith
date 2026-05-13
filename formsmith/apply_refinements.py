#!/usr/bin/env python3
"""Apply refined field offsets to the original PDF using PyMuPDF widgets."""

import hashlib
import json
import logging
import os
from typing import Dict, Tuple

import fitz  # PyMuPDF


logger = logging.getLogger(__name__)


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


PDF_WIDGET_TYPE_MAP = {
    "text": fitz.PDF_WIDGET_TYPE_TEXT,
    "date": fitz.PDF_WIDGET_TYPE_TEXT,
    "email": fitz.PDF_WIDGET_TYPE_TEXT,
    "phone": fitz.PDF_WIDGET_TYPE_TEXT,
    "address": fitz.PDF_WIDGET_TYPE_TEXT,
    "signature": fitz.PDF_WIDGET_TYPE_SIGNATURE,
    "checkbox": fitz.PDF_WIDGET_TYPE_CHECKBOX,
}


DIRTY_WIDGET_FIELDS = (
    "field_name",
    "field_label",
    "rect",
    "text_color",
    "text_font",
    "text_size",
    "border_color",
    "border_style",
    "border_width",
    "fill_color",
    "choices",
    "button_caption",
)


def load_refinements(path: str) -> Dict:
    with open(path, "r") as f:
        return json.load(f)


def to_rect(bbox: Tuple[float, float, float, float]) -> fitz.Rect:
    return fitz.Rect(*bbox)


MIN_HEIGHT_MAP = {
    "text": 14.0,
    "date": 14.0,
    "email": 14.0,
    "phone": 14.0,
    "address": 14.0,
    "signature": 18.0,
    "checkbox": 12.0,
}

MAX_HEIGHT_MAP = {
    "checkbox": 18.0,
}


def _clamp_dimension(value: float, minimum: float, maximum: float | None = None) -> float:
    value = max(minimum, value)
    if maximum is not None:
        value = min(maximum, value)
    return value


def _enforce_geometry(entry: Dict) -> Tuple[Tuple[float, float, float, float], str, Dict]:
    bbox = entry.get("adjusted_bbox") or entry.get("original_bbox")
    if not bbox:
        raise ValueError("Missing bbox for field")

    x0, y0, x1, y1 = [float(v) for v in bbox]
    field_type = (entry.get("field_type") or entry.get("type") or "text").lower()
    # Ensure positive width/height without altering the user's placement.
    if x1 <= x0:
        x1 = x0 + 1.0
    if y1 <= y0:
        y1 = y0 + 1.0

    width = x1 - x0
    height = y1 - y0

    geometry_meta = {
        "enforced_height": height,
        "enforced_width": width,
    }

    rect = fitz.Rect(x0, y0, x1, y1)
    return rect, field_type, geometry_meta


def create_widget(page: fitz.Page, entry: Dict, page_height: float) -> fitz.Widget:
    rect, field_type, geometry_meta = _enforce_geometry(entry)

    field_name = entry.get("field_name") or entry.get("name") or "unnamed"
    widget_type = PDF_WIDGET_TYPE_MAP.get(field_type, fitz.PDF_WIDGET_TYPE_TEXT)

    widget = fitz.Widget()
    widget.field_name = field_name
    widget.field_label = entry.get("field_label") or field_name
    widget.field_type = widget_type
    widget.rect = rect

    # Make all fields borderless
    widget.border_width = 0
    widget.border_color = None
    widget.fill_color = None
    widget.border_style = "S"  # Solid border with width 0
    
    if widget_type == fitz.PDF_WIDGET_TYPE_TEXT:
        widget.text_color = (0, 0, 0)
        widget.text_font = "helv"
        widget.text_size = 11
    elif widget_type == fitz.PDF_WIDGET_TYPE_CHECKBOX:
        widget.button_caption = "✓"
    elif widget_type == fitz.PDF_WIDGET_TYPE_SIGNATURE:
        widget.text_color = (0, 0, 0)
        widget.text_font = "helv"
        widget.text_size = 11

    entry.setdefault("geometry_meta", geometry_meta)
    entry["final_bbox"] = [rect.x0, rect.y0, rect.x1, rect.y1]

    return widget


def _build_entry_lookup(refinements: list[Dict]) -> Dict[Tuple[str, int], Dict]:
    counters: Dict[str, int] = {}
    lookup: Dict[Tuple[str, int], Dict] = {}
    for entry in refinements:
        name = entry.get("field_name") or entry.get("name") or "unnamed"
        idx = counters.get(name, 0)
        counters[name] = idx + 1
        lookup[(name, idx)] = entry
    return lookup


def apply_refinements(
    template_pdf: str,
    refined_json: str,
    output_pdf: str,
    page_index: int = 0,
) -> Dict:
    logger.info(
        "apply_refinements.start",
        extra={
            "template_pdf": template_pdf,
            "template_sha": _sha256(template_pdf),
            "refined_json": refined_json,
            "output_pdf": output_pdf,
        },
    )

    data = load_refinements(refined_json)
    refinements = data.get("refinements", [])
    expected_pdf = data.get("pdf")

    if expected_pdf:
        assert (
            os.path.basename(expected_pdf) == os.path.basename(template_pdf)
        ), f"Template mismatch: json expects {expected_pdf}, got {template_pdf}"
 
    doc = fitz.open(template_pdf)
    assert (
        doc.page_count == data.get("page_count", doc.page_count)
    ), f"Page count mismatch: pdf={doc.page_count}, json={data.get('page_count')}"

    for page_number in range(doc.page_count):
        page = doc[page_number]
        logger.info(
            "apply_refinements.page_info",
            extra={
                "page": page_number,
                "mediabox": tuple(page.mediabox),
                "cropbox": tuple(page.cropbox),
                "rotation": page.rotation,
            },
        )
        assert (
            page.rotation in (0, 90, 180, 270)
        ), f"Unexpected rotation {page.rotation} on page {page_number}"

    if page_index >= len(doc):
        raise ValueError(
            f"PDF has {len(doc)} pages; page_index {page_index} is invalid"
        )

    page = doc[page_index]

    # Remove existing widgets
    widget = page.first_widget
    while widget:
        next_widget = widget.next
        page.delete_widget(widget)
        widget = next_widget

    sample = [
        {
            "field_name": entry.get("field_name"),
            "page": entry.get("page", 0),
            "bbox": entry.get("adjusted_bbox"),
        }
        for entry in refinements[:3]
    ]
    logger.info(
        "apply_refinements.refinements_loaded",
        extra={
            "count": len(refinements),
            "sample": sample,
        },
    )

    entry_lookup = _build_entry_lookup(refinements)
    written = []
    name_counters: Dict[str, int] = {}

    for entry in refinements:
        fname = entry.get("field_name") or entry.get("name") or "unnamed"
        idx = name_counters.get(fname, 0)
        name_counters[fname] = idx + 1

        source_entry = entry_lookup.get((fname, idx), entry)
        widget = create_widget(page, source_entry, page.rect.height)
        page.add_widget(widget)
        written.append((fname, idx))

    doc.save(output_pdf)
    doc.close()

    # Verify output widgets for debugging
    with fitz.open(output_pdf) as out_doc:
        page = out_doc[page_index]
        widgets = list(page.widgets())
        widget_sample = [
            {
                "field_name": widget.field_name,
                "rect": (
                    round(widget.rect.x0, 2),
                    round(widget.rect.y0, 2),
                    round(widget.rect.x1, 2),
                    round(widget.rect.y1, 2),
                ),
            }
            for widget in widgets[:3]
        ]
        logger.info(
            "apply_refinements.output_summary",
            extra={
                "page": page_index,
                "widget_count": len(widgets),
                "widget_sample": widget_sample,
            },
        )

    return {
        "output_pdf": output_pdf,
        "field_count": len(written),
        "fields_written": written,
    }


def main():
    import argparse

    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description="Apply refined field positions")
    parser.add_argument("template_pdf", help="Path to original (flat) PDF template")
    parser.add_argument("refined_json", help="Path to refined fields JSON")
    parser.add_argument("output_pdf", help="Where to save updated fillable PDF")
    parser.add_argument("--page", type=int, default=0, help="Page index to apply refinements")

    args = parser.parse_args()
    result = apply_refinements(
        args.template_pdf,
        args.refined_json,
        args.output_pdf,
        page_index=args.page,
    )
    print(f"Wrote {result['field_count']} fields to {result['output_pdf']}")


if __name__ == "__main__":
    main()
