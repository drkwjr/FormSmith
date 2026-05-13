# FormSmith

**Take a blank PDF form. Understand its layout. Detect where fields belong. Place them with pixel-level precision. Emit interview-engine-ready artifacts.**

FormSmith is a Python pipeline for PDF form intelligence. Point it at a blank government form, and it will analyze structure, detect form fields (text inputs, checkboxes, signatures), place fillable widgets at precise coordinates, validate the geometry, and emit JSON or Docassemble-compatible YAML so the form can be wired up to a guided interview.

It was extracted from a larger Massachusetts court forms project but is engine-agnostic up through field placement. The final emit stage produces interview YAML in a Docassemble-compatible format; if you use a different interview engine, that one module is the only Docassemble-coupled piece.

---

## Pipeline at a glance

```
analyze  →  detect  →  place  →  validate  →  emit
```

| Stage | What it does | Key modules |
|---|---|---|
| **analyze** | Extract text with coordinates; find lines, rectangles, layout patterns | `smart_pdf_analyzer.py`, `pdf_analyzer.py`, `extract_pdf_coordinates.py` |
| **detect** | Find form fields on a blank PDF (OpenCV + learned patterns + optional vision agents) | `learned_field_detector.py`, `pattern_learner.py`, `improved_field_mapper.py` |
| **place** | Write fillable widgets at precise bounding boxes | `smart_field_placer.py`, `precise_field_creator.py`, `create_fillable_pdf.py` |
| **validate** | Check geometry, overlaps, alignment against target regions | `field_geometry_validator.py`, `validate_field_set.py`, `quality_audit.py` |
| **emit** | Export to interview YAML (Docassemble-compatible) or generic JSON | `interview_yaml_generator.py`, `output_formatter.py`, `field_mapper.py` |

The **vision-agent layer** (`formsmith/agents/`) sits orthogonally — it gets called for ambiguous cases. See [Vision agents](#vision-agents-optional).

---

## Install

### Requirements

- Python 3.10+
- macOS / Linux (the GUI editor optionally needs Qt; everything else is headless)

### Quick install

```bash
git clone https://github.com/drkwjr/FormSmith.git
cd FormSmith
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### System dependencies

Most things work with just `pip install`. A few notes:

- **PyMuPDF** (`fitz`) and **pikepdf** ship binary wheels for common platforms — no system deps needed.
- **opencv-python** is used for underscore/checkbox detection on rendered pages.
- **pdfminer.six** has no system deps.
- Fonts: if you generate signature images, install a font (e.g. `BadScript-Regular.ttf`) into `/var/www/.fonts` or `/usr/share/fonts/truetype/`; otherwise FormSmith falls back to Pillow's default.

### Optional dependencies

The `requirements.txt` keeps only `openai` uncommented in the vision-provider section. Uncomment the lines for `anthropic` or `google-generativeai` if you want to use Claude or Gemini instead.

The GUI manual field editor needs `PySide6` — uncomment that line if you want it.

---

## Quickstart

```bash
# 1. See what fields exist in an already-fillable PDF (sanity check + visualization)
python cli.py annotate path/to/form.pdf --out out/annotated

# 2. Analyze structure of a blank PDF
python cli.py analyze path/to/blank_form.pdf --out out/analysis.json

# 3. Detect where fields should go using learned patterns
python cli.py detect path/to/blank_form.pdf \
    --patterns formsmith/data/learned_patterns_v1.json \
    --out out/detected.json

# 4. Create a fillable PDF
python cli.py place path/to/blank_form.pdf --fields out/detected.json --out out/fillable.pdf

# 5. Validate the result
python cli.py validate out/fillable.pdf --targets out/detected.json

# 6. Emit interview YAML
python cli.py export --fields out/detected.json --yaml out/interview.yml --json out/fields.export.json
```

Or as a library:

```python
from formsmith import (
    LearnedFieldDetector,
    FieldMapper,
    FieldCollectionExporter,
    InterviewYAMLGenerator,
)
from formsmith.smart_pdf_analyzer import SmartPDFAnalyzer
from formsmith.precise_field_creator import create_precise_fillable_pdf

# Analyze
analysis = SmartPDFAnalyzer("blank.pdf").analyze()

# Detect (using bundled learned patterns)
import json
patterns = json.load(open("formsmith/data/learned_patterns_v1.json"))
detector = LearnedFieldDetector(learned_patterns=patterns)
fields = detector.detect("blank.pdf")

# Place
create_precise_fillable_pdf("blank.pdf", "fillable.pdf")

# Emit
generator = InterviewYAMLGenerator()
yaml_text = generator.generate(fields)
open("interview.yml", "w").write(yaml_text)
```

---

## Vision agents (optional)

The vision-agent layer is the most powerful and the most expensive piece. It uses a vision-capable LLM to:

- Spot field types in image regions (`FieldSpotterAgent`)
- Read document layout — sections, columns, alignment (`LayoutAnalystAgent`)
- Provide pixel-perfect positional guidance (`PositionAdvisorAgent`)
- Validate placements and suggest move/resize/delete actions (`ValidatorAgent`)
- Break ties when other agents disagree (`RefereeAgent`)
- Learn from corrections (`LearningAgent`)
- Coordinate the whole thing (`MultiAgentOrchestrator`)

They're only invoked on ambiguous detections (typically 0.6–0.8 confidence from the deterministic detector), so cost stays bounded.

### Required: a vision-capable LLM API key

You can plug in any vision-capable model. The default wiring uses OpenAI (`gpt-4o` / `gpt-5-mini` class). To switch providers, swap the client construction in `formsmith/agents/base_agent.py` — it's about ten lines.

| Provider | Models with vision | Env var | SDK |
|---|---|---|---|
| **OpenAI** *(default)* | `gpt-4o`, `gpt-4o-mini`, `gpt-5-mini`, etc. | `OPENAI_API_KEY` | `openai` |
| **Anthropic** | Claude 3.5 Sonnet, Claude 3 Opus, Claude 4.x family | `ANTHROPIC_API_KEY` | `anthropic` |
| **Google** | Gemini 1.5 Flash / Pro, Gemini 2.x | `GOOGLE_API_KEY` | `google-generativeai` |

Set the key in a `.env` file or your shell:

```bash
# .env
OPENAI_API_KEY=sk-...
# or
ANTHROPIC_API_KEY=sk-ant-...
# or
GOOGLE_API_KEY=AIza...
```

**Note:** the agents will fail loudly if no key is configured. The non-agent pipeline (analyze → detect → place → validate → emit) runs fully without any API key — only the vision-enhancement step needs one.

---

## What's portable vs. interview-engine-specific

If you want to use parts of FormSmith without committing to the interview-engine flavor:

| Module | Coupling | Notes |
|---|---|---|
| `smart_pdf_analyzer`, `pdf_analyzer`, `extract_pdf_coordinates` | None | Pure PDF analysis. Drop-in. |
| `pattern_learner`, `learned_field_detector` | None | Reads the bundled `learned_patterns_v1.json`. |
| `smart_field_placer`, `precise_field_creator`, `create_fillable_pdf` | None | PDF widget creation via `pikepdf`. |
| `field_geometry_validator`, `quality_audit` | None | Geometry checks. |
| `agents/*` | None *(needs an LLM key)* | Replace `OpenAI()` client to switch providers. |
| `schemas.FieldDefinition` | Soft — has `interview_*` fields | The schema includes interview-adapter fields but they're optional. |
| `field_mapper.FieldMapper` | Soft | Maps PDF fields → interview variables. Replace if your engine has different conventions. |
| `interview_yaml_generator` | **Docassemble-style YAML** | The one module you'd swap out for a different interview engine. |
| `output_formatter` | Soft | Emits JSON, CSV, plus the YAML above. |

---

## Coordinate system

PDF coordinates have a **bottom-left origin**; most rendering libraries (PIL, OpenCV, browsers) use a **top-left origin**. FormSmith works in image-space (top-left) internally and flips Y when writing widgets — see the `y_flipped = page_height - y2` pattern in `create_fillable_pdf.py`.

Bounding boxes are `[x0, y0, x1, y1]` floats. Pages are 0-indexed.

---

## Agentic usage instructions

If you're an AI coding agent (Claude Code, Cursor, Copilot, etc.) tasked with adapting FormSmith into a project, here's how to get useful work done quickly.

### Quick orientation

1. **Read these first, in order:** `formsmith/__init__.py` → `formsmith/schemas.py` → this README's "Pipeline at a glance" table. That's enough to understand the surface area.
2. **The data contract is `FieldDefinition`** in `schemas.py`. Every stage reads or writes it. If you need to add a new attribute, add it there and update `to_pdf_widget_params()`, `to_yaml_field()`, and `to_attachment_field_mapping()`.
3. **The pipeline is one-directional.** Each stage's output is the next stage's input. Don't tangle them.

### When integrating into a host project

- **Start with `cli.py`** — it's the canonical example of how the stages compose. Mirror its structure rather than calling internal modules ad-hoc.
- **Don't import from `formsmith.agents` unless the user has an API key configured.** Wrap agent calls behind a feature flag or `try/except ImportError` so the host project doesn't crash on cold start.
- **Treat `learned_patterns_v1.json` as a starting point.** It was trained on Massachusetts court forms. For a new form family (e.g. military benefits forms), run `pattern_learner.py` against a few filled examples to generate a domain-specific pattern file.
- **The YAML output is Docassemble-style.** If the host project uses a different interview engine, write a new emitter and skip `interview_yaml_generator.py`. Reuse `FieldMapper` for naming conventions — it has nothing Docassemble-specific in it beyond the docstring.

### When extending the agents

- All agents subclass `VisionAgent` in `agents/base_agent.py`. Override `analyze()` (or whatever the agent's verb is) and return a dict matching the agent's pydantic schema where defined.
- **Provider swap:** if you're switching from OpenAI to Anthropic or Gemini, modify `base_agent.py`'s `__init__` to instantiate the right client, then update each `analyze()` to call that client's vision API. Image input format differs slightly between providers — Anthropic and Gemini both accept base64 image blocks similar to OpenAI.
- The `MultiAgentOrchestrator` implements a chain-of-responsibility with cost-bounded fallback. Don't bypass it for production use — directly calling individual agents wastes tokens.

### Common tasks and where to start

| You want to... | Start in... |
|---|---|
| Detect fields on a new form family | `pattern_learner.py` → generate patterns → `learned_field_detector.py` |
| Adjust where a field lands | `smart_field_placer.py` (calc) → `precise_field_creator.py` (write) |
| Add a new output format | `output_formatter.py` — extend `FieldCollectionExporter` |
| Validate a generated PDF | `field_geometry_validator.py` + `quality_audit.py` |
| Hand-correct a placement | `manual_field_editor.py` (Qt GUI) or `interactive_field_review.py` (CLI) |
| Swap the LLM provider | `agents/base_agent.py` |

### What not to do

- **Don't write through stages.** Don't have your code reach into `create_fillable_pdf` from a detection step. Use the staged outputs.
- **Don't hard-code coordinates** without going through `FieldDefinition`. The validator depends on the schema being consistent.
- **Don't strip `interview_*` attributes** from `FieldDefinition` even if you're not using an interview engine. They're cheap, nullable, and the YAML emitter needs them.

---

## Caveats

- **Trained on Massachusetts court forms.** The bundled patterns will produce sensible defaults on similar government forms (single column, labeled fields, signature lines). For very different layouts, retrain via `pattern_learner.py`.
- **The fancy agent layer needs API budget.** Each ambiguous field can cost ~$0.01–0.05 depending on the model. Disable agents entirely for cost-sensitive runs — the deterministic detector handles 80%+ of fields on well-structured forms.
- **GUI editor is optional.** `manual_field_editor.py` needs `PySide6`. The rest of the package is headless.
- **No active CI.** This is a working extraction, not a maintained library. Treat it as a starting point you fork and adapt.

---

## Attribution

Extracted from [drkwjr/docassemble-builder](https://github.com/drkwjr/docassemble-builder), a private repo for building Massachusetts court forms on Docassemble. The PDF intelligence pipeline is the genuinely portable piece, and this repo is the carve-out.

## License

MIT. See [LICENSE](LICENSE).
