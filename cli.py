#!/usr/bin/env python3
"""
FormSmith CLI — thin wrapper exposing the main pipeline verbs.

Usage:
    python cli.py analyze   <input.pdf> [--out OUT]
    python cli.py annotate  <input.pdf> [--out OUT]
    python cli.py detect    <input.pdf> [--patterns data/learned_patterns_v1.json] [--out OUT]
    python cli.py place     <input.pdf> --fields fields.json [--out output.pdf]
    python cli.py validate  <fillable.pdf> --targets targets.json
    python cli.py export    --fields fields.json [--yaml interview.yml] [--json fields.export.json]

Each verb is a thin call into the `formsmith` package. See README for details.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def cmd_analyze(args: argparse.Namespace) -> int:
    from formsmith.smart_pdf_analyzer import SmartPDFAnalyzer

    analyzer = SmartPDFAnalyzer(args.input)
    analysis = analyzer.analyze()
    out = Path(args.out or "analysis.json")
    out.write_text(json.dumps(analysis, indent=2, default=str))
    print(f"✅ Analysis written to {out}")
    return 0


def cmd_annotate(args: argparse.Namespace) -> int:
    from formsmith.annotate_pdf_fields import main as annotate_main

    sys.argv = ["annotate_pdf_fields", args.input]
    if args.out:
        sys.argv.append(args.out)
    return annotate_main()


def cmd_detect(args: argparse.Namespace) -> int:
    from formsmith.learned_field_detector import LearnedFieldDetector

    patterns_path = args.patterns or "formsmith/data/learned_patterns_v1.json"
    patterns = json.loads(Path(patterns_path).read_text())
    detector = LearnedFieldDetector(learned_patterns=patterns)
    result = detector.detect(args.input)
    out = Path(args.out or "detected_fields.json")
    out.write_text(json.dumps(result, indent=2, default=str))
    print(f"✅ Detected {len(result.get('fields', []))} fields → {out}")
    return 0


def cmd_place(args: argparse.Namespace) -> int:
    from formsmith.precise_field_creator import create_precise_fillable_pdf

    out = args.out or args.input.replace(".pdf", "_fillable.pdf")
    create_precise_fillable_pdf(args.input, out)
    print(f"✅ Fillable PDF written to {out}")
    return 0


def cmd_validate(args: argparse.Namespace) -> int:
    from formsmith.field_geometry_validator import FieldGeometryValidator

    validator = FieldGeometryValidator(args.input)
    targets = json.loads(Path(args.targets).read_text()) if args.targets else None
    report = validator.validate(targets)
    print(json.dumps(report, indent=2, default=str))
    return 0


def cmd_export(args: argparse.Namespace) -> int:
    from formsmith.output_formatter import export_all_formats

    fields = json.loads(Path(args.fields).read_text())
    paths = export_all_formats(
        fields,
        yaml_path=args.yaml,
        json_path=args.json,
    )
    for kind, p in paths.items():
        print(f"✅ {kind}: {p}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="formsmith")
    sub = p.add_subparsers(dest="cmd", required=True)

    a = sub.add_parser("analyze", help="Analyze PDF structure (text, lines, layout)")
    a.add_argument("input")
    a.add_argument("--out")
    a.set_defaults(func=cmd_analyze)

    an = sub.add_parser("annotate", help="Render a visual map of existing AcroForm fields")
    an.add_argument("input")
    an.add_argument("--out")
    an.set_defaults(func=cmd_annotate)

    d = sub.add_parser("detect", help="Detect fields on a blank PDF using learned patterns")
    d.add_argument("input")
    d.add_argument("--patterns")
    d.add_argument("--out")
    d.set_defaults(func=cmd_detect)

    pl = sub.add_parser("place", help="Create a fillable PDF using precise coordinates")
    pl.add_argument("input")
    pl.add_argument("--fields")
    pl.add_argument("--out")
    pl.set_defaults(func=cmd_place)

    v = sub.add_parser("validate", help="Validate field geometry against target regions")
    v.add_argument("input")
    v.add_argument("--targets")
    v.set_defaults(func=cmd_validate)

    e = sub.add_parser("export", help="Export fields to interview YAML and/or JSON")
    e.add_argument("--fields", required=True)
    e.add_argument("--yaml")
    e.add_argument("--json", dest="json")
    e.set_defaults(func=cmd_export)

    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
