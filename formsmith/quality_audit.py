#!/usr/bin/env python3
"""High-precision quality audit comparing expected vs actual field positions."""

import json
from pathlib import Path
from typing import Dict, List, Tuple


def _normalize_entry(raw: Dict) -> Dict | None:
    name = raw.get("name") or raw.get("field_name")
    if not name:
        return None

    field_type = raw.get("type") or raw.get("field_type") or "text"
    bbox = (
        raw.get("bbox")
        or raw.get("adjusted_bbox")
        or raw.get("original_bbox")
    )
    if not bbox or len(bbox) != 4:
        return None

    bbox_tuple = tuple(float(v) for v in bbox)
    return {
        "name": name,
        "type": field_type.lower(),
        "bbox": bbox_tuple,
    }


def load_fields(path: Path) -> Dict[str, Dict]:
    data = json.loads(path.read_text())

    raw_entries: List[Dict] = []
    if isinstance(data.get("fields"), list) and data["fields"]:
        raw_entries = data["fields"]
    elif isinstance(data.get("refinements"), list) and data["refinements"]:
        raw_entries = data["refinements"]

    fields: Dict[str, Dict] = {}
    for entry in raw_entries:
        normalized = _normalize_entry(entry)
        if normalized:
            fields[normalized["name"]] = normalized

    return fields


def bbox_to_tuple(entry: Dict) -> Tuple[float, float, float, float]:
    return tuple(float(v) for v in entry["bbox"])


def compute_metrics(expected: Dict, actual: Dict) -> Dict:
    exp_bbox = bbox_to_tuple(expected)
    act_bbox = bbox_to_tuple(actual)

    metrics = {
        "delta_x0": act_bbox[0] - exp_bbox[0],
        "delta_y0": act_bbox[1] - exp_bbox[1],
        "delta_x1": act_bbox[2] - exp_bbox[2],
        "delta_y1": act_bbox[3] - exp_bbox[3],
    }
    metrics["delta_width"] = (act_bbox[2] - act_bbox[0]) - (exp_bbox[2] - exp_bbox[0])
    metrics["delta_height"] = (act_bbox[3] - act_bbox[1]) - (exp_bbox[3] - exp_bbox[1])
    metrics["max_abs_delta"] = max(abs(v) for v in metrics.values())

    return metrics


def is_checkbox(field: Dict) -> bool:
    return field.get("type") == "checkbox"


def audit(
    expected_path: Path,
    actual_path: Path,
    output_json: Path,
    output_md: Path,
    tol_default: float,
    tol_checkbox: float,
) -> Dict:
    expected_fields = load_fields(expected_path)
    actual_fields = load_fields(actual_path)

    report = {
        "expected_count": len(expected_fields),
        "actual_count": len(actual_fields),
        "missing_fields": [],
        "extra_fields": [],
        "results": [],
        "status": "pending",
    }

    for name, expected in expected_fields.items():
        actual = actual_fields.get(name)
        if not actual:
            report["missing_fields"].append(name)
            continue

        metrics = compute_metrics(expected, actual)
        tolerance = tol_checkbox if is_checkbox(expected) else tol_default
        passed = metrics["max_abs_delta"] <= tolerance

        report["results"].append(
            {
                "field": name,
                "type": expected.get("type"),
                "tolerance": tolerance,
                "metrics": metrics,
                "passed": passed,
            }
        )

    for name in actual_fields:
        if name not in expected_fields:
            report["extra_fields"].append(name)

    report["results"].sort(key=lambda r: abs(r["metrics"]["max_abs_delta"]), reverse=True)

    total_expected = report["expected_count"]
    passed_count = sum(1 for r in report["results"] if r["passed"])
    # Treat missing fields as failures for pass rate computation
    pass_rate = 0.0
    if total_expected:
        pass_rate = passed_count / total_expected
    report["pass_rate"] = pass_rate

    if report["missing_fields"] or report["extra_fields"]:
        report["status"] = "coverage_mismatch"
    elif pass_rate >= 1.0:
        report["status"] = "pass"
    else:
        report["status"] = "tolerance_fail"

    output_json.write_text(json.dumps(report, indent=2))

    lines: List[str] = []
    lines.append("# Field Quality Audit")
    lines.append("")
    lines.append(f"Expected fields: {report['expected_count']}")
    lines.append(f"Actual fields: {report['actual_count']}")
    lines.append(f"Pass rate (<= tolerance): {report['pass_rate']*100:.1f}%")
    lines.append(f"Status: {report['status']}")
    lines.append("")
    if report["missing_fields"]:
        lines.append("## Missing Fields")
        lines.append("- " + "\n- ".join(report["missing_fields"]))
        lines.append("")
    if report["extra_fields"]:
        lines.append("## Extra Fields")
        lines.append("- " + "\n- ".join(report["extra_fields"]))
        lines.append("")

    lines.append("## Field Metrics (sorted by worst delta)")
    lines.append("")
    for result in report["results"]:
        metrics = result["metrics"]
        lines.append(
            f"### {result['field']} ({result['type']}) - {'PASS' if result['passed'] else 'FAIL'}"
        )
        lines.append(f"Tolerance: ±{result['tolerance']:.3f} px")
        lines.append(
            f"- Δx0: {metrics['delta_x0']:.3f}\n"
            f"- Δy0: {metrics['delta_y0']:.3f}\n"
            f"- Δx1: {metrics['delta_x1']:.3f}\n"
            f"- Δy1: {metrics['delta_y1']:.3f}\n"
            f"- Δwidth: {metrics['delta_width']:.3f}\n"
            f"- Δheight: {metrics['delta_height']:.3f}\n"
            f"- Max Δ: {metrics['max_abs_delta']:.3f}"
        )
        lines.append("")

    output_md.write_text("\n".join(lines))
    return report


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Run high-precision field audit")
    parser.add_argument("expected", help="Path to expected fields JSON")
    parser.add_argument("actual", help="Path to exported fields from PDF")
    parser.add_argument("--json", default="logs/quality_audit_report.json")
    parser.add_argument("--markdown", default="logs/quality_audit_report.md")
    parser.add_argument("--tol-default", type=float, default=1.0)
    parser.add_argument("--tol-checkbox", type=float, default=0.75)
    args = parser.parse_args()

    report = audit(
        Path(args.expected),
        Path(args.actual),
        Path(args.json),
        Path(args.markdown),
        args.tol_default,
        args.tol_checkbox,
    )
    print(
        f"Audit complete: {len(report['results'])} fields, pass rate {report['pass_rate']*100:.1f}%" \
        f", missing {len(report['missing_fields'])}, extra {len(report['extra_fields'])}"
    )


if __name__ == "__main__":
    main()
