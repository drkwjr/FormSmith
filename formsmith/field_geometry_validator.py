#!/usr/bin/env python3
"""Deterministic geometry extraction and validation for PDF form fields"""

import json
import math
import os
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple, Iterable

import fitz  # PyMuPDF
from PIL import Image, ImageDraw, ImageFont


@dataclass
class TargetRegion:
    """Represents an ideal geometric target for a form field"""

    id: str
    type: str
    bbox: Tuple[float, float, float, float]
    corners: Dict[str, Tuple[float, float]]
    detection_source: str
    metadata: Dict[str, float]

    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "type": self.type,
            "bbox": list(self.bbox),
            "corners": {k: list(v) for k, v in self.corners.items()},
            "detection_source": self.detection_source,
            "metadata": self.metadata,
        }


class FieldGeometryValidator:
    """Extracts deterministic target regions and validates field placement"""

    def __init__(self, pdf_path: str, page_index: int = 0):
        self.pdf_path = pdf_path
        self.page_index = page_index
        self.doc = fitz.open(pdf_path)

        if page_index >= len(self.doc):
            raise ValueError(
                f"PDF has {len(self.doc)} pages; page_index {page_index} is invalid"
            )

        self.page = self.doc[page_index]
        self.page_rect = self.page.rect

    def close(self):
        self.doc.close()

    def extract_target_regions(self) -> List[TargetRegion]:
        """Derive target regions from visual cues (lines, rectangles)"""

        drawings = self.page.get_drawings()
        regions: List[TargetRegion] = []

        underline_index = 0
        checkbox_index = 0
        signature_index = 0
        rectangle_index = 0

        def add_region(region: TargetRegion) -> None:
            rounded_bbox = tuple(round(coord, 1) for coord in region.bbox)
            existing = next(
                (
                    r
                    for r in regions
                    if tuple(round(c, 1) for c in r.bbox) == rounded_bbox
                    and r.type == region.type
                ),
                None,
            )
            if existing is None:
                regions.append(region)

        for drawing in drawings:
            dtype = drawing.get("type")
            items = drawing.get("items", [])

            for item in items:
                if not item:
                    continue

                operator = item[0]

                if operator == "re":
                    rect = item[1]
                    if not isinstance(rect, fitz.Rect):
                        continue

                    xmin, ymin, xmax, ymax = rect
                    width = xmax - xmin
                    height = ymax - ymin

                    if width <= 0 or height <= 0:
                        continue

                    aspect_ratio = width / height if height else 0

                    if height <= 4 and width >= 35:
                        line_type = "signature_line" if width >= 175 else "underline"
                        if line_type == "signature_line":
                            signature_index += 1
                            region_id = f"signature_{signature_index:03d}"
                        else:
                            underline_index += 1
                            region_id = f"underline_{underline_index:03d}"

                        region_height = max(4.0, height * 2)
                        ymin_adj = ymin - (region_height - height) / 2
                        ymax_adj = ymax + (region_height - height) / 2

                        corners = {
                            "top_left": (xmin, ymax_adj),
                            "top_right": (xmax, ymax_adj),
                            "bottom_left": (xmin, ymin_adj),
                            "bottom_right": (xmax, ymin_adj),
                        }

                        add_region(
                            TargetRegion(
                                id=region_id,
                                type=line_type,
                                bbox=(xmin, ymin_adj, xmax, ymax_adj),
                                corners=corners,
                                detection_source="rect_line",
                                metadata={
                                    "width": width,
                                    "height": height,
                                    "aspect_ratio": aspect_ratio,
                                },
                            )
                        )

                    elif 8 <= width <= 36 and 8 <= height <= 36 and 0.65 <= aspect_ratio <= 1.35:
                        checkbox_index += 1
                        region_id = f"checkbox_{checkbox_index:03d}"

                        corners = {
                            "top_left": (xmin, ymax),
                            "top_right": (xmax, ymax),
                            "bottom_left": (xmin, ymin),
                            "bottom_right": (xmax, ymin),
                        }

                        add_region(
                            TargetRegion(
                                id=region_id,
                                type="checkbox",
                                bbox=(xmin, ymin, xmax, ymax),
                                corners=corners,
                                detection_source="rect_checkbox",
                                metadata={
                                    "width": width,
                                    "height": height,
                                    "aspect_ratio": aspect_ratio,
                                },
                            )
                        )

                    else:
                        rectangle_index += 1
                        corners = {
                            "top_left": (xmin, ymax),
                            "top_right": (xmax, ymax),
                            "bottom_left": (xmin, ymin),
                            "bottom_right": (xmax, ymin),
                        }

                        add_region(
                            TargetRegion(
                                id=f"rectangle_{rectangle_index:03d}",
                                type="rectangle",
                                bbox=(xmin, ymin, xmax, ymax),
                                corners=corners,
                                detection_source="rect_generic",
                                metadata={
                                    "width": width,
                                    "height": height,
                                    "aspect_ratio": aspect_ratio,
                                },
                            )
                        )

                elif operator == "qu":
                    quad = item[1]
                    if not isinstance(quad, fitz.Quad):
                        continue

                    rect = quad.rect
                    xmin, ymin, xmax, ymax = rect
                    width = xmax - xmin
                    height = ymax - ymin

                    if width <= 0 or height <= 0:
                        continue

                    aspect_ratio = width / height if height else 0

                    if 8 <= width <= 36 and 8 <= height <= 36 and 0.6 <= aspect_ratio <= 1.4:
                        checkbox_index += 1
                        region_id = f"checkbox_{checkbox_index:03d}"

                        corners = {
                            "top_left": (xmin, ymax),
                            "top_right": (xmax, ymax),
                            "bottom_left": (xmin, ymin),
                            "bottom_right": (xmax, ymin),
                        }

                        add_region(
                            TargetRegion(
                                id=region_id,
                                type="checkbox",
                                bbox=(xmin, ymin, xmax, ymax),
                                corners=corners,
                                detection_source="quad_checkbox",
                                metadata={
                                    "width": width,
                                    "height": height,
                                    "aspect_ratio": aspect_ratio,
                                },
                            )
                        )
                    else:
                        rectangle_index += 1
                        corners = {
                            "top_left": (xmin, ymax),
                            "top_right": (xmax, ymax),
                            "bottom_left": (xmin, ymin),
                            "bottom_right": (xmax, ymin),
                        }

                        add_region(
                            TargetRegion(
                                id=f"rectangle_{rectangle_index:03d}",
                                type="rectangle",
                                bbox=(xmin, ymin, xmax, ymax),
                                corners=corners,
                                detection_source="quad_generic",
                                metadata={
                                    "width": width,
                                    "height": height,
                                    "aspect_ratio": aspect_ratio,
                                },
                            )
                        )

        return regions

    def save_target_regions(
        self,
        output_json: str,
        overlay_image: Optional[str] = None,
        dpi: int = 150,
    ) -> List[TargetRegion]:
        regions = self.extract_target_regions()

        os.makedirs(os.path.dirname(output_json), exist_ok=True)
        with open(output_json, "w") as f:
            json.dump(
                {
                    "pdf": os.path.basename(self.pdf_path),
                    "page_index": self.page_index,
                    "page_width": float(self.page_rect.width),
                    "page_height": float(self.page_rect.height),
                    "regions": [region.to_dict() for region in regions],
                },
                f,
                indent=2,
            )

        if overlay_image:
            self._render_overlay(regions, overlay_image, dpi)

        return regions

    def _render_overlay(
        self,
        regions: List[TargetRegion],
        overlay_image: str,
        dpi: int = 150,
    ) -> None:
        matrix = fitz.Matrix(dpi / 72, dpi / 72)
        pix = self.page.get_pixmap(matrix=matrix)
        image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        draw = ImageDraw.Draw(image, "RGBA")

        scale_x = pix.width / self.page_rect.width
        scale_y = pix.height / self.page_rect.height

        try:
            font = ImageFont.truetype("Helvetica", 14)
        except Exception:
            font = ImageFont.load_default()

        color_map = {
            "checkbox": (0, 200, 0, 150),
            "underline": (0, 128, 255, 150),
            "signature_line": (255, 128, 0, 150),
            "rectangle": (200, 0, 200, 150),
        }

        for region in regions:
            color = color_map.get(region.type, (255, 0, 0, 150))
            xmin, ymin, xmax, ymax = region.bbox

            left = xmin * scale_x
            right = xmax * scale_x
            top = pix.height - (ymax * scale_y)
            bottom = pix.height - (ymin * scale_y)

            draw.rectangle([left, top, right, bottom], outline=color, width=3)
            draw.rectangle([left, top, right, bottom], fill=(color[0], color[1], color[2], 40))

            label = f"{region.id}\n{region.type}"
            bbox = draw.textbbox((left, top), label, font=font)
            label_width = bbox[2] - bbox[0]
            label_height = bbox[3] - bbox[1]
            label_x = left
            label_y = max(0, top - label_height - 4)

            draw.rectangle(
                [label_x, label_y, label_x + label_width + 4, label_y + label_height + 4],
                fill=(0, 0, 0, 170),
            )
            draw.text((label_x + 2, label_y + 2), label, fill=(255, 255, 255), font=font)

        os.makedirs(os.path.dirname(overlay_image), exist_ok=True)
        image.save(overlay_image)

    @staticmethod
    def validate_field_alignment(
        fields: List[Dict],
        targets: List[TargetRegion],
        tolerance_checkbox: float = 2.0,
        tolerance_default: float = 3.0,
    ) -> Dict:
        """Compare field boxes to target regions and produce alignment report"""

        report = {
            "matches": [],
            "unmatched_fields": [],
            "unmatched_targets": [],
        }

        targets_remaining = targets.copy()

        for field in fields:
            field_bbox = field.get("bbox")
            if not field_bbox:
                continue

            field_center = (
                (field_bbox[0] + field_bbox[2]) / 2,
                (field_bbox[1] + field_bbox[3]) / 2,
            )

            best_target = None
            best_distance = float("inf")

            for target in targets_remaining:
                target_bbox = target.bbox
                target_center = (
                    (target_bbox[0] + target_bbox[2]) / 2,
                    (target_bbox[1] + target_bbox[3]) / 2,
                )

                distance = math.dist(field_center, target_center)
                if distance < best_distance:
                    best_distance = distance
                    best_target = target

            if best_target is None:
                report["unmatched_fields"].append(field)
                continue

            # Measure corner deltas
            deltas = {}
            field_corners = {
                "top_left": (field_bbox[0], field_bbox[3]),
                "top_right": (field_bbox[2], field_bbox[3]),
                "bottom_left": (field_bbox[0], field_bbox[1]),
                "bottom_right": (field_bbox[2], field_bbox[1]),
            }

            for corner_name, field_corner in field_corners.items():
                target_corner = best_target.corners[corner_name]
                deltas[corner_name] = (
                    field_corner[0] - target_corner[0],
                    field_corner[1] - target_corner[1],
                )

            # Determine pass/fail using tolerance by target type
            tolerance = (
                tolerance_checkbox
                if best_target.type == "checkbox"
                else tolerance_default
            )

            max_delta = max(
                max(abs(dx), abs(dy)) for dx, dy in deltas.values()
            )
            passed = max_delta <= tolerance

            report["matches"].append(
                {
                    "field": field,
                    "target": best_target.to_dict(),
                    "corner_deltas": deltas,
                    "center_distance": best_distance,
                    "tolerance": tolerance,
                    "passed": passed,
                }
            )

            targets_remaining.remove(best_target)

        # Remaining unmatched targets
        report["unmatched_targets"] = [target.to_dict() for target in targets_remaining]
        return report


def _load_fields_from_json(path: str) -> List[Dict]:
    with open(path, "r") as f:
        data = json.load(f)
    return data.get("fields", [])


def _load_refinements_from_json(path: str) -> List[Dict]:
    with open(path, "r") as f:
        data = json.load(f)
    return data.get("refinements", data.get("fields", []))


def _horizontal_overlap(field_bbox: Tuple[float, float, float, float], target_bbox: Tuple[float, float, float, float]) -> float:
    left = max(field_bbox[0], target_bbox[0])
    right = min(field_bbox[2], target_bbox[2])
    return max(0.0, right - left)


def _clamp(value: float, min_value: float, max_value: float) -> float:
    return max(min_value, min(value, max_value))


def _expand_bbox(
    bbox: Tuple[float, float, float, float],
    padding: float,
    page_width: float,
    page_height: float,
) -> Tuple[float, float, float, float]:
    xmin, ymin, xmax, ymax = bbox
    xmin = _clamp(xmin - padding, 0.0, page_width)
    xmax = _clamp(xmax + padding, 0.0, page_width)
    ymin = _clamp(ymin - padding, 0.0, page_height)
    ymax = _clamp(ymax + padding, 0.0, page_height)
    if xmax <= xmin:
        xmax = min(page_width, xmin + 1.0)
    if ymax <= ymin:
        ymax = min(page_height, ymin + 1.0)
    return xmin, ymin, xmax, ymax


def _round_bbox(bbox: Tuple[float, float, float, float], precision: int = 2) -> Tuple[float, float, float, float]:
    return tuple(round(coord, precision) for coord in bbox)


def _target_center(target: TargetRegion) -> Tuple[float, float]:
    bbox = target.bbox
    return (bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0


def _bbox_center(bbox: Tuple[float, float, float, float]) -> Tuple[float, float]:
    return (bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0


def _bbox_width(bbox: Tuple[float, float, float, float]) -> float:
    return bbox[2] - bbox[0]


def _bbox_height(bbox: Tuple[float, float, float, float]) -> float:
    return bbox[3] - bbox[1]


def _find_best_target(
    field_bbox: Tuple[float, float, float, float],
    candidates: Iterable[TargetRegion],
    max_vertical_diff: float,
    min_overlap_ratio: float,
) -> Optional[TargetRegion]:
    field_center_x, field_center_y = _bbox_center(field_bbox)
    field_width = _bbox_width(field_bbox)

    best_target: Optional[TargetRegion] = None
    best_score = float("inf")

    for target in candidates:
        target_bbox = target.bbox
        target_center_x, target_center_y = _target_center(target)
        vertical_diff = abs(field_center_y - target_center_y)
        if vertical_diff > max_vertical_diff:
            continue

        overlap = _horizontal_overlap(field_bbox, target_bbox)
        overlap_ratio = overlap / max(field_width, 1.0)
        if overlap_ratio < min_overlap_ratio:
            continue

        target_width = _bbox_width(target_bbox)
        width_diff = abs(target_width - field_width)
        center_dx = abs(field_center_x - target_center_x)

        score = vertical_diff * 2.0 + width_diff * 0.05 + center_dx * 0.1 - overlap_ratio * 5.0
        if score < best_score:
            best_score = score
            best_target = target

    return best_target


def _find_nearest_checkbox(
    field_bbox: Tuple[float, float, float, float],
    candidates: Iterable[TargetRegion],
) -> Optional[TargetRegion]:
    field_center = _bbox_center(field_bbox)
    best_target: Optional[TargetRegion] = None
    best_distance = float("inf")

    for target in candidates:
        target_center = _target_center(target)
        distance = math.dist(field_center, target_center)
        if distance < best_distance:
            best_distance = distance
            best_target = target

    return best_target


def snap_fields_to_targets(
    fields: List[Dict],
    targets: List[TargetRegion],
    page_width: float,
    page_height: float,
    text_height: float = 18.0,
    signature_height: float = 22.0,
    text_x_padding: float = 3.0,
    checkbox_padding: float = 0.5,
    min_width: float = 8.0,
) -> List[Dict]:
    """Snap baseline fields to detected target geometry."""

    lines = [t for t in targets if t.type in {"signature_line", "underline"}]
    checkboxes = [t for t in targets if t.type == "checkbox"]

    snapped: List[Dict] = []

    for field in fields:
        name = field.get("name") or field.get("field_name")
        field_type = (field.get("type") or "text").lower()
        original_bbox = tuple(field.get("bbox", ()))
        if len(original_bbox) != 4:
            continue

        entry: Dict[str, Optional[str]] = {
            "field_name": name,
            "field_type": field_type,
            "original_bbox": list(original_bbox),
        }

        snapped_bbox: Optional[Tuple[float, float, float, float]] = None
        method = "fallback_original"
        target_id: Optional[str] = None
        target_type: Optional[str] = None

        if field_type == "checkbox":
            target = _find_nearest_checkbox(original_bbox, checkboxes)
            if target is not None:
                snapped_bbox = _expand_bbox(target.bbox, checkbox_padding, page_width, page_height)
                method = "checkbox_snap"
                target_id = target.id
                target_type = target.type
        else:
            target = _find_best_target(
                original_bbox,
                lines,
                max_vertical_diff=28.0,
                min_overlap_ratio=0.15,
            )

            if target is not None:
                target_id = target.id
                target_type = target.type
                target_bbox = target.bbox

                line_left, line_bottom, line_right, line_top = target_bbox

                desired_height = signature_height if name and "signature" in name.lower() else text_height
                half_height = desired_height / 2.0
                line_center_y = (line_bottom + line_top) / 2.0

                new_bottom = _clamp(line_center_y - half_height, 0.0, page_height)
                new_top = _clamp(line_center_y + half_height, 0.0, page_height)

                orig_left, orig_bottom, orig_right, orig_top = original_bbox

                new_left = max(line_left, orig_left - text_x_padding)
                new_right = min(line_right, orig_right + text_x_padding)

                if new_right - new_left < min_width:
                    width = max(_bbox_width(original_bbox), min_width)
                    center_x = (orig_left + orig_right) / 2.0
                    new_left = _clamp(center_x - width / 2.0, 0.0, page_width)
                    new_right = _clamp(center_x + width / 2.0, 0.0, page_width)

                snapped_bbox = (
                    new_left,
                    new_bottom,
                    new_right,
                    new_top,
                )
                method = "line_snap"

        if snapped_bbox is None:
            snapped_bbox = tuple(original_bbox)

        metadata = {
            "adjusted_bbox": list(_round_bbox(snapped_bbox)),
            "snap_method": method,
            "target_id": target_id,
            "target_type": target_type,
        }

        if target_id is not None and target_type is not None:
            metadata.update(
                {
                    "target_bbox": list(_round_bbox(target.bbox)),
                    "target_length": round(_bbox_width(target.bbox), 2),
                }
            )

        entry.update(metadata)

        snapped.append(entry)

    return snapped


def run_snap(
    fields_json: str,
    targets_json: str,
    output_json: str,
    pdf_name: Optional[str] = None,
    text_height: float = 18.0,
    signature_height: float = 22.0,
    text_padding: float = 3.0,
    checkbox_padding: float = 0.5,
    metadata: Optional[Dict] = None,
) -> List[Dict]:
    with open(targets_json, "r") as f:
        targets_payload = json.load(f)

    page_width = targets_payload.get("page_width", 612.0)
    page_height = targets_payload.get("page_height", 792.0)
    target_regions = [
        TargetRegion(
            id=region["id"],
            type=region["type"],
            bbox=tuple(region["bbox"]),
            corners={k: tuple(v) for k, v in region["corners"].items()},
            detection_source=region.get("detection_source", "unknown"),
            metadata=region.get("metadata", {}),
        )
        for region in targets_payload.get("regions", [])
    ]

    fields = _load_fields_from_json(fields_json)
    if not fields:
        fields = _load_refinements_from_json(fields_json)
    # Normalize field representation (supports both raw exports and refinement JSONs)
    normalized_fields = []
    for field in fields:
        bbox = field.get("bbox") or field.get("adjusted_bbox")
        if not bbox or len(bbox) != 4:
            continue
        normalized_fields.append(
            {
                "name": field.get("name") or field.get("field_name"),
                "type": field.get("type") or field.get("field_type"),
                "bbox": bbox,
            }
        )

    fields = normalized_fields

    snapped = snap_fields_to_targets(
        fields,
        target_regions,
        page_width=page_width,
        page_height=page_height,
        text_height=text_height,
        signature_height=signature_height,
        text_x_padding=text_padding,
        checkbox_padding=checkbox_padding,
    )

    pdf_label = pdf_name or targets_payload.get("pdf")
    save_snapped_refinements(
        snapped,
        output_json,
        pdf_name=pdf_label,
        source_fields=fields_json,
        targets_path=targets_json,
        metadata=metadata,
        page_width=page_width,
        page_height=page_height,
    )

    return snapped


def save_snapped_refinements(
    refinements: List[Dict],
    output_path: str,
    pdf_name: str,
    source_fields: Optional[str] = None,
    targets_path: Optional[str] = None,
    metadata: Optional[Dict] = None,
    page_width: Optional[float] = None,
    page_height: Optional[float] = None,
) -> None:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    payload = {
        "pdf": pdf_name,
        "source_fields": source_fields,
        "targets": targets_path,
        "refinements": refinements,
        "metadata": metadata or {},
        "page_width": page_width,
        "page_height": page_height,
    }
    with open(output_path, "w") as f:
        json.dump(payload, f, indent=2)


def _load_targets_from_json(path: str) -> List[TargetRegion]:
    with open(path, "r") as f:
        data = json.load(f)
    return [
        TargetRegion(
            id=region["id"],
            type=region["type"],
            bbox=tuple(region["bbox"]),
            corners={k: tuple(v) for k, v in region["corners"].items()},
            detection_source=region.get("detection_source", "unknown"),
            metadata=region.get("metadata", {}),
        )
        for region in data.get("regions", [])
    ]


def save_alignment_report(report: Dict, output_path: str) -> None:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    serializable = {
        "matches": [
            {
                "field": match["field"],
                "target": match["target"],
                "corner_deltas": {
                    corner: [dx, dy]
                    for corner, (dx, dy) in match["corner_deltas"].items()
                },
                "center_distance": match["center_distance"],
                "tolerance": match["tolerance"],
                "passed": match["passed"],
            }
            for match in report["matches"]
        ],
        "unmatched_fields": report["unmatched_fields"],
        "unmatched_targets": report["unmatched_targets"],
    }

    with open(output_path, "w") as f:
        json.dump(serializable, f, indent=2)


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Extract target regions and optionally validate field alignment"
    )
    parser.add_argument("pdf", help="Path to the PDF form")
    parser.add_argument("output", help="Where to save target_regions.json")
    parser.add_argument(
        "--overlay",
        help="Optional path to save overlay image",
    )
    parser.add_argument(
        "--fields",
        help="Optional fields JSON to validate against target regions",
    )
    parser.add_argument(
        "--qa-report",
        help="Where to write QA alignment report if --fields is provided",
    )
    parser.add_argument("--page", type=int, default=0, help="Page index to analyze")

    args = parser.parse_args()

    validator = FieldGeometryValidator(args.pdf, page_index=args.page)
    regions = validator.save_target_regions(args.output, args.overlay)

    if args.fields and args.qa_report:
        fields = _load_fields_from_json(args.fields)
        report = FieldGeometryValidator.validate_field_alignment(fields, regions)
        save_alignment_report(report, args.qa_report)

    validator.close()


if __name__ == "__main__":
    main()

