#!/usr/bin/env python3
"""Iteratively refine PDF field positions until tolerances pass."""

import json
from datetime import datetime
from pathlib import Path
from typing import Dict

from .field_refinement import run_refinement
from .apply_refinements import apply_refinements
from .smart_pdf_analyzer import SmartPDFAnalyzer
from .quality_audit import audit
from .field_geometry_validator import run_snap

EXPECTED_FIELDS = Path("data/exports/initial_fields.json")
TARGET_REGIONS = Path("data/targets/jud_tc_petition_targets.json")
TEMPLATE_PDF = Path("jud-tc-Petition-to-Deem-Satisfied.pdf")
OUTPUT_BASE = Path("data/runs")

MAX_ITERATIONS = 5
PROMPT_COST_PER_TOKEN = 0.000003
COMPLETION_COST_PER_TOKEN = 0.000006


def ensure_output_dirs(run_dir: Path) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "logs").mkdir(exist_ok=True)
    (run_dir / "outputs").mkdir(exist_ok=True)


def export_fields_from_pdf(pdf_path: Path, output_json: Path, overlay_path: Path | None = None) -> None:
    analyzer = SmartPDFAnalyzer(str(pdf_path))
    analyzer.export_existing_fields(str(output_json), str(overlay_path) if overlay_path else None)


def compute_baseline(iteration: int, snap_output: Path, iteration_snap: Path) -> Path:
    return snap_output if iteration == 1 else iteration_snap


def regenerate_snap(
    source_fields: Path,
    output_json: Path,
    targets_json: Path,
    pdf_name: str,
    metadata: Dict | None = None,
) -> None:
    run_snap(
        fields_json=str(source_fields),
        targets_json=str(targets_json),
        output_json=str(output_json),
        pdf_name=pdf_name,
        text_height=18.0,
        signature_height=22.0,
        text_padding=4.0,
        checkbox_padding=1.0,
        metadata=metadata or {},
    )


def iterate_refinement(
    template_pdf: Path,
    target_pdf: Path,
    targets_json: Path,
    expected_fields: Path,
    run_dir: Path,
    max_iterations: int = MAX_ITERATIONS,
    pdf_label: str | None = None,
) -> Dict:
    ensure_output_dirs(run_dir)

    outputs_dir = run_dir / "outputs"
    logs_dir = run_dir / "logs"

    snap_output = outputs_dir / "snapped_fields.json"
    iteration_snap = outputs_dir / "iter_snapped_fields.json"
    refinement_json = outputs_dir / "iter_refinements.json"
    refined_pdf_fields = outputs_dir / "refined_pdf_fields.json"
    refined_pdf = outputs_dir / f"{target_pdf.stem}_refined.pdf"
    audit_json = logs_dir / "quality_audit.json"
    audit_md = logs_dir / "quality_audit.md"

    total_prompt = 0
    total_completion = 0
    total_cost = 0.0
    iterations_ran = 0
    pass_rate = 0.0
    worst_delta = None
    last_report = None
    coverage_mismatch = False
    prev_pass_rate = -1.0
    prev_worst_delta = None

    metadata = {
        "template_pdf": str(template_pdf),
        "target_pdf": str(target_pdf),
    }

    # Always seed snapped baseline from expected fields to ensure full coverage
    print("Generating snapped baseline from expected fields...")
    regenerate_snap(
        expected_fields,
        snap_output,
        targets_json,
        pdf_label or template_pdf.name,
        metadata=metadata,
    )

    # Persist baseline as iteration 0 seed for Agent B
    iteration_snap.write_text(snap_output.read_text())

    for iteration in range(1, max_iterations + 1):
        iterations_ran = iteration
        print(f"\n--- Iteration {iteration}/{max_iterations} ---")

        baseline_path = compute_baseline(iteration, snap_output, iteration_snap)

        source_pdf = refined_pdf if refined_pdf.exists() else target_pdf

        print("Running AI refinement (Agent B)...")
        refinement_result = run_refinement(
            pdf_path=str(source_pdf),
            fields_json=str(baseline_path),
            targets_json=str(targets_json),
            output_json=str(refinement_json),
            baseline_json=str(baseline_path),
        )

        token_usage = refinement_result.get("token_usage", {})
        iteration_cost = refinement_result.get("cost_usd", 0.0)
        total_prompt += token_usage.get("prompt", 0)
        total_completion += token_usage.get("completion", 0)
        total_cost += iteration_cost
        print(
            "  Tokens - prompt: {prompt}, completion: {completion}, total: {total}, cost ${cost:.6f}".format(
                prompt=token_usage.get("prompt", 0),
                completion=token_usage.get("completion", 0),
                total=token_usage.get("total", 0),
                cost=iteration_cost,
            )
        )

        print("Applying refined fields back to template...")
        apply_refinements(
            template_pdf=str(template_pdf),
            refined_json=str(refinement_json),
            output_pdf=str(refined_pdf),
        )

        print("Exporting fields from refined PDF...")
        export_fields_from_pdf(refined_pdf, refined_pdf_fields)

        print("Running quality audit...")
        report = audit(
            expected_fields,
            refined_pdf_fields,
            audit_json,
            audit_md,
            tol_default=1.0,
            tol_checkbox=0.75,
        )

        last_report = report
        pass_rate = report["pass_rate"]
        status = report.get("status")
        if status == "coverage_mismatch":
            print("Coverage mismatch detected (missing or extra fields). Halting refinement.")
            coverage_mismatch = True
            break
        worst = 0
        worst_field = None
        for result in report["results"]:
            if not result["passed"] and abs(result["metrics"]["max_abs_delta"]) > abs(worst):
                worst = result["metrics"]["max_abs_delta"]
                worst_field = result

        for result in report["results"]:
            status = "PASS" if result["passed"] else "FAIL"
            print(
                f"  {result['field']:<35} {status} maxΔ={result['metrics']['max_abs_delta']:.3f}"
            )

        if pass_rate >= 1.0:
            print("\nAll fields within tolerance. Done!")
            worst_delta = 0.0
            break

        if worst_field:
            worst_delta = worst_field["metrics"]["max_abs_delta"]
            print(
                f"Worst delta so far: {worst_field['field']} -> {worst_field['metrics']['max_abs_delta']:.3f} px"
            )
        else:
            print("No failing fields? (unexpected)")

        if prev_pass_rate >= 0 and pass_rate < prev_pass_rate - 1e-6:
            print("Pass rate regressed; stopping iterations to prevent divergence.")
            break

        if (
            prev_worst_delta is not None
            and worst_delta is not None
            and worst_delta > prev_worst_delta + 0.25
        ):
            print("Worst delta increased significantly; stopping iterations.")
            break

        prev_pass_rate = pass_rate
        if worst_delta is not None:
            prev_worst_delta = worst_delta

        print("Preparing snapped baseline for next iteration...")
        regenerate_snap(
            refined_pdf_fields,
            iteration_snap,
            targets_json,
            pdf_label or template_pdf.name,
            metadata=metadata,
        )

    else:
        print("\nReached max iterations without all fields passing.")

    if iterations_ran == 0:
        print("No iterations executed; applying snapped baseline only.")
        apply_refinements(
            template_pdf=str(template_pdf),
            refined_json=str(snap_output),
            output_pdf=str(refined_pdf),
        )
        export_fields_from_pdf(refined_pdf, refined_pdf_fields)
        last_report = audit(
            expected_fields,
            refined_pdf_fields,
            audit_json,
            audit_md,
            tol_default=1.0,
            tol_checkbox=0.75,
        )
        pass_rate = last_report["pass_rate"]
        status = last_report.get("status")
        coverage_mismatch = status == "coverage_mismatch"
        worst_delta = 0.0
        for result in last_report["results"]:
            if not result["passed"]:
                worst_delta = max(worst_delta, abs(result["metrics"]["max_abs_delta"]))

    total_tokens = total_prompt + total_completion
    if worst_delta is None:
        worst_delta = 0.0 if pass_rate >= 1.0 else None

    return {
        "run_dir": str(run_dir),
        "snapped_fields": str(snap_output),
        "refined_pdf": str(refined_pdf),
        "refined_fields": str(refined_pdf_fields),
        "audit_json": str(audit_json),
        "audit_md": str(audit_md),
        "iterations": iterations_ran,
        "pass_rate": pass_rate,
        "worst_delta": worst_delta,
        "tokens": {
            "prompt": total_prompt,
            "completion": total_completion,
            "total": total_tokens,
        },
        "cost_usd": total_cost,
        "coverage_mismatch": coverage_mismatch,
        "status": last_report.get("status") if last_report else "not_run",
    }


def main() -> None:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = OUTPUT_BASE / f"run_{timestamp}"
    pdf_label = TEMPLATE_PDF.name

    result = iterate_refinement(
        template_pdf=TEMPLATE_PDF,
        target_pdf=TEMPLATE_PDF,
        targets_json=TARGET_REGIONS,
        expected_fields=EXPECTED_FIELDS,
        run_dir=run_dir,
        pdf_label=pdf_label,
    )

    print("\n=== Run Summary ===")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
