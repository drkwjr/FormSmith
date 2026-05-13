#!/usr/bin/env python3
"""Shared architecture scaffolding for the manual PDF field editor.

This module defines the storage schema, loading/saving helpers, and
compatibility glue with the existing deterministic/LLM pipeline.

Future UI layers (PySide6 desktop app, headless batch scripts, etc.) can
import these helpers to provide drag-and-drop editing without rewriting
the data plumbing.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import shutil
import tempfile
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import fitz  # PyMuPDF
from PySide6 import QtCore, QtGui, QtWidgets

from .quality_audit import audit as run_quality_audit


logger = logging.getLogger(__name__)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


BoundingBox = Tuple[float, float, float, float]


@dataclass
class FieldRecord:
    """Canonical representation of an interactive field on a PDF page."""

    name: str
    field_type: str
    page: int
    bbox: BoundingBox
    source: str = "manual_editor"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def as_refinement_entry(self) -> Dict[str, Any]:
        """Return entry compatible with `apply_refinements` JSON payload."""

        x0, y0, x1, y1 = self.bbox
        adjusted = [x0, y0, x1, y1]
        return {
            "field_name": self.name,
            "field_type": self.field_type,
            "page": self.page,
            "original_bbox": adjusted,
            "adjusted_bbox": adjusted,
            "source": self.source,
            "metadata": self.metadata,
        }

    def translate(self, dx: float, dy: float) -> None:
        """Shift the field by the provided delta in PDF coordinate space."""

        x0, y0, x1, y1 = self.bbox
        self.bbox = (x0 + dx, y0 + dy, x1 + dx, y1 + dy)

    def resize(self, dw: float, dh: float) -> None:
        """Grow the field width/height while keeping top-left anchored."""

        x0, y0, x1, y1 = self.bbox
        self.bbox = (x0, y0, x1 + dw, y1 + dh)


@dataclass
class EditorState:
    """In-memory state for a PDF + field set."""

    pdf_path: Path
    fields: List[FieldRecord]
    page_count: int
    page_sizes: List[Tuple[float, float]]
    targets_path: Optional[Path] = None
    derived_from: Dict[str, Any] = field(default_factory=dict)
    refinements_path: Optional[Path] = None
    last_output_pdf: Optional[Path] = None
    attachment_order: List[str] = field(default_factory=list)

    @classmethod
    def from_sources(
        cls,
        pdf_path: Path,
        field_json: Optional[Path] = None,
        targets_path: Optional[Path] = None,
        yaml_path: Optional[Path] = None,
    ) -> "EditorState":
        """Load PDF metadata and field records from disk.

        `field_json` may be any of:
          * refinements payload (from `run_snap`, `iterate_refinement`, etc.)
          * exported fields JSON (from `smart_pdf_analyzer.export_existing_fields`)
        `yaml_path` can provide an attachment block for field name mapping.
        """

        pdf_path = pdf_path.resolve()
        if field_json:
            field_json = field_json.resolve()
        if targets_path:
            targets_path = targets_path.resolve()

        page_count, page_sizes = _get_page_metrics(pdf_path)
        fields = _load_fields(field_json) if field_json else []

        attachment_order: List[str] = []
        if yaml_path:
            yaml_path = yaml_path.resolve()
            name_map = _parse_attachment_fields(yaml_path)
            if name_map:
                attachment_order = [
                    name for name, _ in sorted(name_map.items(), key=lambda item: item[1])
                ]
                _apply_name_map(fields, name_map)

        derived_from = {
            "pdf": str(pdf_path),
            "fields": str(field_json) if field_json else None,
            "targets": str(targets_path) if targets_path else None,
            "yaml": str(yaml_path) if yaml_path else None,
        }

        return cls(
            pdf_path=pdf_path,
            fields=fields,
            page_count=page_count,
            page_sizes=page_sizes,
            targets_path=targets_path,
            derived_from=derived_from,
            refinements_path=field_json,
            last_output_pdf=None,
            attachment_order=attachment_order,
        )

    # --- convenience operations -------------------------------------------------

    def by_page(self, page: int) -> List[FieldRecord]:
        return [field for field in self.fields if field.page == page]

    def set_field(self, index: int, record: FieldRecord) -> None:
        self.fields[index] = record

    def add_field(self, record: FieldRecord) -> None:
        self.fields.append(record)

    def remove_field(self, index: int) -> FieldRecord:
        return self.fields.pop(index)

    def next_field_name(self, base: str = "field") -> str:
        existing = {field.name for field in self.fields}
        counter = 1
        candidate = f"{base}_{counter:02d}"
        while candidate in existing:
            counter += 1
            candidate = f"{base}_{counter:02d}"
        return candidate

    def attachment_usage(self) -> Dict[str, Optional[int]]:
        usage: Dict[str, Optional[int]] = {name: None for name in self.attachment_order}
        for idx, field in enumerate(self.fields):
            if field.name in usage and usage[field.name] is None:
                usage[field.name] = idx
        return usage

    def is_name_available(self, name: str, *, ignore_index: Optional[int] = None) -> bool:
        name = name.strip()
        if not name:
            return False  # Empty names are not allowed
        # PDF forms commonly have multiple fields with the same name
        # (e.g., multiple checkboxes for same choice, multiple signature fields)
        return True

    def set_field_name(self, index: int, new_name: str) -> bool:
        new_name = new_name.strip()
        if index < 0 or index >= len(self.fields):
            return False
        current = self.fields[index].name
        if new_name == current:
            return True
        if not self.is_name_available(new_name, ignore_index=index):
            return False
        self.fields[index].name = new_name
        return True

    # --- persistence -----------------------------------------------------------

    def to_refinements(self) -> Dict[str, Any]:
        return {
            "pdf": self.pdf_path.name,
            "page_count": self.page_count,
            "page_sizes": self.page_sizes,
            "refinements": [field.as_refinement_entry() for field in self.fields],
            "source": "manual_editor",
            "derived_from": self.derived_from,
        }

    def save_refinements(self, path: Path) -> None:
        payload = self.to_refinements()
        path = path.resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2))
        self.refinements_path = path

    def save_pdf(self, output_pdf: Path) -> Dict[str, Any]:
        """Regenerate a fillable PDF containing the current fields."""

        from apply_refinements import apply_refinements

        output_pdf = output_pdf.resolve()
        output_pdf.parent.mkdir(parents=True, exist_ok=True)

        original_refinements_path = self.refinements_path
        temp_json_path: Optional[Path] = None
        temp_path: Optional[Path] = None

        logger.info(
            "manual_editor.save_pdf start",
            extra={
                "pdf": str(self.pdf_path),
                "pdf_sha": _sha256(self.pdf_path),
                "output": str(output_pdf),
                "field_count": len(self.fields),
            },
        )
        
        # Log field positions for debugging
        sample_fields = []
        for i, field in enumerate(self.fields[:3]):
            sample_fields.append({
                "name": field.name,
                "type": field.field_type,
                "bbox": field.bbox,
                "page": field.page
            })
        logger.info(f"manual_editor.save_pdf sample fields: {sample_fields}")

        try:
            with tempfile.NamedTemporaryFile(
                suffix=".json", dir=str(output_pdf.parent), delete=False
            ) as temp_json_file:
                temp_json_path = Path(temp_json_file.name)
            payload = self.to_refinements()
            temp_json_path.write_text(json.dumps(payload, indent=2))
            refinements_path = temp_json_path

            sample = [
                {
                    "field_name": entry["field_name"],
                    "page": entry.get("page", 0),
                    "bbox": entry.get("adjusted_bbox"),
                }
                for entry in payload.get("refinements", [])[:3]
            ]
            logger.info(
                "manual_editor.save_pdf serialized",
                extra={
                    "temp_json": str(temp_json_path),
                    "refinements_count": len(payload.get("refinements", [])),
                    "sample": sample,
                },
            )
            with tempfile.NamedTemporaryFile(
                suffix=".pdf", dir=str(output_pdf.parent), delete=False
            ) as temp_pdf:
                temp_path = Path(temp_pdf.name)

            try:
                result = apply_refinements(
                    template_pdf=str(self.pdf_path),
                    refined_json=str(refinements_path),
                    output_pdf=str(temp_path),
                )
                shutil.move(str(temp_path), str(output_pdf))
                result["output_pdf"] = str(output_pdf)
                self.last_output_pdf = output_pdf
                
                # Verify fields were written by reopening the PDF
                doc_verify = fitz.open(str(output_pdf))
                try:
                    verified_widgets = []
                    for page_num in range(doc_verify.page_count):
                        page = doc_verify[page_num]
                        widgets = list(page.widgets())
                        verified_widgets.extend([
                            {
                                "name": w.field_name,
                                "page": page_num,
                                "rect": (w.rect.x0, w.rect.y0, w.rect.x1, w.rect.y1)
                            }
                            for w in widgets
                        ])
                    
                    result["verified_widget_count"] = len(verified_widgets)
                    result["verified_widgets_sample"] = verified_widgets[:3]
                    
                    logger.info(
                        "manual_editor.save_pdf verification",
                        extra={
                            "expected_count": len(self.fields),
                            "verified_count": len(verified_widgets),
                            "sample": verified_widgets[:3]
                        }
                    )
                    
                    if len(verified_widgets) != len(self.fields):
                        logger.warning(
                            f"Field count mismatch! Expected {len(self.fields)}, "
                            f"but found {len(verified_widgets)} in saved PDF"
                        )
                finally:
                    doc_verify.close()
                
                return result
            finally:
                if temp_path and temp_path.exists():
                    temp_path.unlink(missing_ok=True)
                if temp_json_path and temp_json_path.exists():
                    temp_json_path.unlink(missing_ok=True)
                self.refinements_path = original_refinements_path
        except Exception:
            logger.exception("Failed to serialize refinements JSON; falling back to existing path")
            temp_json_path = None
            refinements_path = self.refinements_path or None

        if refinements_path is None:
            raise RuntimeError("Unable to prepare refinements payload for PDF regeneration")

        with tempfile.NamedTemporaryFile(
            suffix=".pdf", dir=str(output_pdf.parent), delete=False
        ) as temp_pdf:
            temp_path = Path(temp_pdf.name)

        try:
            result = apply_refinements(
                template_pdf=str(self.pdf_path),
                refined_json=str(refinements_path),
                output_pdf=str(temp_path),
            )
            shutil.move(str(temp_path), str(output_pdf))
            result["output_pdf"] = str(output_pdf)
            self.last_output_pdf = output_pdf
            return result
        finally:
            if temp_path and temp_path.exists():
                temp_path.unlink(missing_ok=True)
            if temp_json_path and temp_json_path.exists():
                temp_json_path.unlink(missing_ok=True)
            self.refinements_path = original_refinements_path

    def run_quality_audit(
        self,
        targets_path: Path,
        output_dir: Path,
        tol_default: float = 1.0,
        tol_checkbox: float = 0.75,
    ) -> Dict[str, Any]:
        """Run deterministic quality audit against expected targets."""

        output_dir = output_dir.resolve()
        output_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        actual_fields_path = output_dir / f"manual_editor_actual_{timestamp}.json"
        target_json = Path(targets_path).resolve()

        # Export to fields-array format for auditing
        actual_payload = {
            "pdf": self.pdf_path.name,
            "fields": [
                {
                    "name": field.name,
                    "type": field.field_type,
                    "page": field.page,
                    "bbox": list(field.bbox),
                }
                for field in self.fields
            ],
        }
        actual_fields_path.write_text(json.dumps(actual_payload, indent=2))

        output_json = output_dir / f"quality_audit_{timestamp}.json"
        output_md = output_dir / f"quality_audit_{timestamp}.md"

        report = run_quality_audit(
            expected_path=target_json,
            actual_path=actual_fields_path,
            output_json=output_json,
            output_md=output_md,
            tol_default=tol_default,
            tol_checkbox=tol_checkbox,
        )

        report.update(
            {
                "audit_json": str(output_json),
                "audit_markdown": str(output_md),
                "actual_fields_json": str(actual_fields_path),
            }
        )
        return report


# ---------------------------------------------------------------------------
# Helpers


def _get_page_metrics(pdf_path: Path) -> Tuple[int, List[Tuple[float, float]]]:
    doc = fitz.open(pdf_path)
    try:
        sizes = [(page.rect.width, page.rect.height) for page in doc]
        return len(doc), sizes
    finally:
        doc.close()


def _load_fields(field_json: Path) -> List[FieldRecord]:
    data = json.loads(field_json.read_text())
    if isinstance(data, dict):
        if "refinements" in data:
            return list(_fields_from_refinements(data["refinements"]))
        if "fields" in data:
            return list(_fields_from_fields_array(data["fields"]))
    raise ValueError(f"Unsupported field payload format: {field_json}")


def _fields_from_refinements(refinements: Iterable[Dict[str, Any]]):
    for entry in refinements:
        name = entry.get("field_name") or entry.get("name") or "unnamed"
        field_type = entry.get("field_type") or entry.get("type") or "text"
        page = entry.get("page", 0)
        bbox = _extract_bbox(entry)
        metadata = entry.get("metadata") or {}
        yield FieldRecord(name=name, field_type=field_type, page=page, bbox=bbox, metadata=metadata)


def _fields_from_fields_array(fields: Iterable[Dict[str, Any]]):
    for entry in fields:
        name = entry.get("name") or "unnamed"
        field_type = entry.get("type") or "text"
        page = entry.get("page", 0)
        bbox = _extract_bbox(entry)
        metadata = {k: v for k, v in entry.items() if k not in {"name", "type", "page", "bbox"}}
        yield FieldRecord(name=name, field_type=field_type, page=page, bbox=bbox, metadata=metadata)


def _extract_bbox(entry: Dict[str, Any]) -> BoundingBox:
    bbox = (
        entry.get("adjusted_bbox")
        or entry.get("bbox")
        or entry.get("rect")
        or entry.get("original_bbox")
    )
    if not bbox or len(bbox) != 4:
        raise ValueError(f"Invalid bounding box in entry: {entry}")
    return tuple(float(v) for v in bbox)  # type: ignore[return-value]


# YAML helpers ---------------------------------------------------------------


def _parse_attachment_fields(yaml_path: Path) -> Dict[str, int]:
    """Parse interview attachment block to map field names to order index."""
    text = yaml_path.read_text(encoding="utf-8")
    fields_section = re.search(r"fields:\s*(?:#.*\n|\s*\n|\s*-.*\n)+", text)
    if not fields_section:
        return {}

    name_pattern = re.compile(r"-\s*['\"](?P<name>[^'\"]+)['\"]:")
    names: Dict[str, int] = {}
    for idx, match in enumerate(name_pattern.finditer(fields_section.group(0))):
        names[match.group("name")] = idx
    return names


def _apply_name_map(fields: List[FieldRecord], name_map: Dict[str, int]) -> None:
    """Rename fields to match YAML attachment order when possible."""

    unused_names = sorted(name_map.items(), key=lambda item: item[1])
    used_indices = set()

    for field in fields:
        if field.name in name_map:
            used_indices.add(name_map[field.name])

    for field in fields:
        if field.name in name_map:
            continue
        candidate = next((name for name, idx in unused_names if idx not in used_indices), None)
        if candidate:
            used_indices.add(name_map[candidate])
            field.name = candidate
        else:
            break


class FieldPropertiesPanel(QtWidgets.QDockWidget):
    """Dock widget for inspecting and editing the selected field."""

    BASE_TYPES = ["text", "checkbox", "signature", "textarea", "radio", "combobox", "date"]

    def __init__(self, state: EditorState, controller: "EditorWindow") -> None:
        super().__init__("Field Properties", controller)
        self.state = state
        self.controller = controller
        self.current_index: Optional[int] = None
        self._suppress_updates = False

        container = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(container)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)

        self.attachment_hint = QtWidgets.QLabel()
        self.attachment_hint.setWordWrap(True)
        layout.addWidget(self.attachment_hint)

        form = QtWidgets.QFormLayout()
        form.setHorizontalSpacing(12)
        form.setVerticalSpacing(8)

        self.name_combo = QtWidgets.QComboBox()
        self.name_combo.setEditable(True)
        self.name_combo.setInsertPolicy(QtWidgets.QComboBox.NoInsert)
        self.name_combo.currentIndexChanged.connect(self._on_name_index_changed)
        self.name_combo.lineEdit().editingFinished.connect(self._on_name_edit_finished)
        form.addRow("Field name", self.name_combo)

        self.name_warning = QtWidgets.QLabel()
        palette = self.name_warning.palette()
        palette.setColor(QtGui.QPalette.WindowText, QtGui.QColor("#c0392b"))
        self.name_warning.setPalette(palette)
        form.addRow("", self.name_warning)

        self.type_combo = QtWidgets.QComboBox()
        self.type_combo.currentTextChanged.connect(self._on_type_changed)
        form.addRow("Field type", self.type_combo)

        layout.addLayout(form)

        metadata_group = QtWidgets.QGroupBox("Metadata")
        metadata_layout = QtWidgets.QVBoxLayout(metadata_group)
        metadata_layout.setContentsMargins(8, 8, 8, 8)
        metadata_layout.setSpacing(6)

        self.metadata_table = QtWidgets.QTableWidget(0, 2)
        self.metadata_table.setHorizontalHeaderLabels(["Key", "Value"])
        self.metadata_table.horizontalHeader().setStretchLastSection(True)
        self.metadata_table.verticalHeader().setVisible(False)
        self.metadata_table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.metadata_table.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        self.metadata_table.setEditTriggers(QtWidgets.QAbstractItemView.AllEditTriggers)
        self.metadata_table.itemChanged.connect(self._on_metadata_changed)
        metadata_layout.addWidget(self.metadata_table)

        metadata_buttons = QtWidgets.QHBoxLayout()
        self.metadata_add_button = QtWidgets.QPushButton("Add row")
        self.metadata_add_button.clicked.connect(self._add_metadata_row)
        self.metadata_remove_button = QtWidgets.QPushButton("Remove row")
        self.metadata_remove_button.clicked.connect(self._remove_selected_metadata)
        metadata_buttons.addWidget(self.metadata_add_button)
        metadata_buttons.addWidget(self.metadata_remove_button)
        metadata_buttons.addStretch(1)
        metadata_layout.addLayout(metadata_buttons)

        layout.addWidget(metadata_group)
        layout.addStretch(1)

        self._controlled_widgets = [
            self.name_combo,
            self.type_combo,
            self.metadata_table,
            self.metadata_add_button,
            self.metadata_remove_button,
        ]

        self.setWidget(container)
        self._update_attachment_hint()
        self._set_enabled(False)

    def set_current_field(self, index: Optional[int]) -> None:
        self.current_index = index
        self._set_enabled(index is not None)
        self.name_warning.clear()

        if index is None:
            self._suppress_updates = True
            self.name_combo.clear()
            self.type_combo.clear()
            self.metadata_table.setRowCount(0)
            self._suppress_updates = False
            return

        record = self.state.fields[index]
        self._populate_name_combo(record)
        self._populate_type_combo(record)
        self._populate_metadata(record)

    def refresh_attachment_display(self) -> None:
        self._update_attachment_hint()
        if self.current_index is not None:
            self._populate_name_combo(self.state.fields[self.current_index])

    def _set_enabled(self, enabled: bool) -> None:
        for widget in self._controlled_widgets:
            widget.setEnabled(enabled)

    def _update_attachment_hint(self) -> None:
        if self.state.attachment_order:
            usage = self.state.attachment_usage()
            remaining = sum(1 for _, owner in usage.items() if owner is None)
            self.attachment_hint.setText(
                f"Loaded {len(self.state.attachment_order)} attachment names · {remaining} unused"
            )
        else:
            self.attachment_hint.setText("No interview YAML loaded; free-form naming.")

    def _populate_name_combo(self, record: FieldRecord) -> None:
        self._suppress_updates = True
        combo = self.name_combo
        combo.blockSignals(True)
        combo.clear()

        usage = self.state.attachment_usage() if self.state.attachment_order else {}
        current_name = record.name
        current_index = -1

        if self.state.attachment_order:
            for name in self.state.attachment_order:
                combo.addItem(name, name)
                model_item = combo.model().item(combo.count() - 1)
                owner = usage.get(name)
                if model_item and owner is not None and owner != self.current_index:
                    model_item.setEnabled(False)
                if name == current_name:
                    current_index = combo.count() - 1

        if current_index == -1 and current_name:
            combo.insertItem(0, current_name, current_name)
            current_index = 0

        if combo.count() == 0:
            combo.addItem(current_name or "", current_name or "")
            current_index = 0

        combo.setCurrentIndex(current_index)
        combo.setEditText(current_name)
        combo.blockSignals(False)
        self._suppress_updates = False

    def _populate_type_combo(self, record: FieldRecord) -> None:
        self._suppress_updates = True
        combo = self.type_combo
        combo.blockSignals(True)
        combo.clear()

        options = list(dict.fromkeys(self.BASE_TYPES + [f.field_type for f in self.state.fields]))
        for option in options:
            combo.addItem(option)

        index = combo.findText(record.field_type)
        if index == -1:
            combo.addItem(record.field_type)
            index = combo.count() - 1
        combo.setCurrentIndex(index)
        combo.blockSignals(False)
        self._suppress_updates = False

    def _populate_metadata(self, record: FieldRecord) -> None:
        self._suppress_updates = True
        self.metadata_table.blockSignals(True)
        self.metadata_table.setRowCount(0)
        for key, value in record.metadata.items():
            self._insert_metadata_row(str(key), value, update_state=False)
        self.metadata_table.blockSignals(False)
        self._suppress_updates = False

    def _on_name_index_changed(self, index: int) -> None:
        if self._suppress_updates or self.current_index is None:
            return
        candidate = self.name_combo.itemData(index)
        name = candidate if isinstance(candidate, str) else self.name_combo.currentText()
        self._apply_new_name(name)

    def _on_name_edit_finished(self) -> None:
        if self._suppress_updates or self.current_index is None:
            return
        self._apply_new_name(self.name_combo.currentText())

    def _apply_new_name(self, candidate: str) -> None:
        if self.current_index is None:
            return
        candidate = candidate.strip()
        if not candidate:
            self.name_warning.setText("Field name cannot be empty.")
            self._populate_name_combo(self.state.fields[self.current_index])
            return
        if not self.state.set_field_name(self.current_index, candidate):
            self.name_warning.setText("Invalid field name.")
            self._populate_name_combo(self.state.fields[self.current_index])
            return

        self.name_warning.clear()
        self.state.fields[self.current_index].name = candidate
        self.controller.on_field_properties_changed(self.current_index)
        self.controller.refresh_field_scene(self.current_index)
        self.refresh_attachment_display()

    def _on_type_changed(self, new_type: str) -> None:
        if self._suppress_updates or self.current_index is None:
            return
        record = self.state.fields[self.current_index]
        if record.field_type == new_type:
            return
        record.field_type = new_type
        self.controller.on_field_properties_changed(self.current_index)

    # Legacy metadata hooks removed; retain stubs for compatibility
    def _collect_metadata(self) -> Dict[str, Any]:
        metadata: Dict[str, Any] = {}
        for row in range(self.metadata_table.rowCount()):
            key_item = self.metadata_table.item(row, 0)
            if not key_item:
                continue
            key = key_item.text().strip()
            if not key:
                continue
            value_item = self.metadata_table.item(row, 1)
            value = value_item.text() if value_item else ""
            metadata[key] = value
        return metadata

    def _on_metadata_changed(self, _item) -> None:
        if self._suppress_updates or self.current_index is None:
            return
        self.state.fields[self.current_index].metadata = self._collect_metadata()
        self.controller.on_field_properties_changed(self.current_index)

    def _add_metadata_row(self) -> None:
        if self.current_index is None:
            return
        self._suppress_updates = True
        self.metadata_table.blockSignals(True)
        row = self.metadata_table.rowCount()
        self.metadata_table.insertRow(row)
        self.metadata_table.setItem(row, 0, QtWidgets.QTableWidgetItem(""))
        self.metadata_table.setItem(row, 1, QtWidgets.QTableWidgetItem(""))
        self.metadata_table.blockSignals(False)
        self._suppress_updates = False
        self.state.fields[self.current_index].metadata = self._collect_metadata()
        self.controller.on_field_properties_changed(self.current_index)
        self.metadata_table.setCurrentCell(row, 0)

    def _on_metadata_double_clicked(self, _item) -> None:
        return

    def _remove_selected_metadata(self) -> None:
        if self.current_index is None:
            return
        selection = self.metadata_table.selectionModel().selectedRows()
        if not selection:
            return
        self._suppress_updates = True
        self.metadata_table.blockSignals(True)
        for index in sorted(selection, key=lambda idx: idx.row(), reverse=True):
            self.metadata_table.removeRow(index.row())
        self.metadata_table.blockSignals(False)
        self._suppress_updates = False
        self.state.fields[self.current_index].metadata = self._collect_metadata()
        self.controller.on_field_properties_changed(self.current_index)

    def _insert_metadata_row(self, key: str, value: Any, *, update_state: bool = True) -> None:
        row = self.metadata_table.rowCount()
        self.metadata_table.insertRow(row)
        self.metadata_table.setItem(row, 0, QtWidgets.QTableWidgetItem(key))
        self.metadata_table.setItem(
            row, 1, QtWidgets.QTableWidgetItem("" if value is None else str(value))
        )
        if update_state and self.current_index is not None:
            self.state.fields[self.current_index].metadata = self._collect_metadata()
            self.controller.on_field_properties_changed(self.current_index)

# Coordinate helpers ---------------------------------------------------------


def pdf_to_scene_rect(bbox: BoundingBox, page_height: float) -> QtCore.QRectF:
    x0, y0, x1, y1 = bbox
    top = page_height - y1
    return QtCore.QRectF(x0, top, x1 - x0, y1 - y0)


def clamp_bbox_to_page(
    bbox: BoundingBox,
    page_width: float,
    page_height: float,
    min_width: float = 4.0,
    min_height: float = 4.0,
) -> BoundingBox:
    x0, y0, x1, y1 = bbox
    w = max(min_width, x1 - x0)
    h = max(min_height, y1 - y0)
    x0 = min(max(0.0, x0), page_width - w)
    y0 = min(max(0.0, y0), page_height - h)
    x1 = x0 + w
    y1 = y0 + h
    return (x0, y0, x1, y1)


# ---------------------------------------------------------------------------
# Resize Handles
# ---------------------------------------------------------------------------


class ResizeHandle(QtWidgets.QGraphicsRectItem):
    """Small draggable handle for resizing fields."""

    def __init__(self, corner: str, parent_field: "FieldGraphicsItem") -> None:
        """
        Args:
            corner: One of "NW", "NE", "SW", "SE"
            parent_field: The field this handle resizes
        """
        super().__init__(0, 0, 8, 8)
        self.corner = corner
        self.parent_field = parent_field
        self.setParentItem(parent_field)

        # Visual style
        self.setPen(QtGui.QPen(QtGui.QColor(255, 140, 0), 1.0))
        self.setBrush(QtGui.QBrush(QtGui.QColor(255, 255, 255)))

        # Set cursor based on corner
        if corner in ("NW", "SE"):
            self.setCursor(QtCore.Qt.SizeFDiagCursor)
        else:  # NE, SW
            self.setCursor(QtCore.Qt.SizeBDiagCursor)

        self.setFlag(QtWidgets.QGraphicsItem.ItemIsMovable, False)
        self.setFlag(QtWidgets.QGraphicsItem.ItemSendsGeometryChanges, True)
        self.setZValue(10)

        self._drag_start_pos = None
        self._drag_start_rect = None

    def mousePressEvent(self, event: QtGui.QGraphicsSceneMouseEvent) -> None:
        if event.button() == QtCore.Qt.LeftButton:
            self._drag_start_pos = event.scenePos()
            self._drag_start_rect = self.parent_field.rect()
            event.accept()
        else:
            super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QtGui.QGraphicsSceneMouseEvent) -> None:
        if self._drag_start_pos is None:
            return

        delta = event.scenePos() - self._drag_start_pos
        rect = QtCore.QRectF(self._drag_start_rect)

        # Apply delta based on corner
        if self.corner == "NW":
            new_left = rect.left() + delta.x()
            new_top = rect.top() + delta.y()
            new_width = max(4.0, rect.width() - delta.x())
            new_height = max(4.0, rect.height() - delta.y())
            rect.setLeft(new_left if new_width > 4.0 else rect.right() - 4.0)
            rect.setTop(new_top if new_height > 4.0 else rect.bottom() - 4.0)
        elif self.corner == "NE":
            new_right = rect.right() + delta.x()
            new_top = rect.top() + delta.y()
            new_width = max(4.0, rect.width() + delta.x())
            new_height = max(4.0, rect.height() - delta.y())
            rect.setRight(new_right if new_width > 4.0 else rect.left() + 4.0)
            rect.setTop(new_top if new_height > 4.0 else rect.bottom() - 4.0)
        elif self.corner == "SW":
            new_left = rect.left() + delta.x()
            new_bottom = rect.bottom() + delta.y()
            new_width = max(4.0, rect.width() - delta.x())
            new_height = max(4.0, rect.height() + delta.y())
            rect.setLeft(new_left if new_width > 4.0 else rect.right() - 4.0)
            rect.setBottom(new_bottom if new_height > 4.0 else rect.top() + 4.0)
        elif self.corner == "SE":
            new_right = rect.right() + delta.x()
            new_bottom = rect.bottom() + delta.y()
            new_width = max(4.0, rect.width() + delta.x())
            new_height = max(4.0, rect.height() + delta.y())
            rect.setRight(new_right if new_width > 4.0 else rect.left() + 4.0)
            rect.setBottom(new_bottom if new_height > 4.0 else rect.top() + 4.0)

        self.parent_field.setRect(rect)
        self.parent_field._update_handle_positions()

    def mouseReleaseEvent(self, event: QtGui.QGraphicsSceneMouseEvent) -> None:
        if event.button() == QtCore.Qt.LeftButton:
            self._drag_start_pos = None
            self._drag_start_rect = None
            # Trigger the change callback
            if self.parent_field._change_callback:
                self.parent_field._change_callback(self.parent_field)
            event.accept()
        else:
            super().mouseReleaseEvent(event)


class FieldGraphicsItem(QtWidgets.QGraphicsRectItem):
    """Graphics item representing a single PDF field."""

    def __init__(
        self,
        index: int,
        record: FieldRecord,
        page_height: float,
        change_callback,
    ) -> None:
        rect = pdf_to_scene_rect(record.bbox, page_height)
        super().__init__(QtCore.QRectF(0, 0, rect.width(), rect.height()))
        self.field_index = index
        self.visible_index = index
        self.page_height = page_height
        self._change_callback = change_callback
        self._suppress_callback = False

        self.setPos(rect.x(), rect.y())
        self.setZValue(5)
        self.setFlags(
            QtWidgets.QGraphicsItem.ItemIsSelectable
            | QtWidgets.QGraphicsItem.ItemIsMovable
            | QtWidgets.QGraphicsItem.ItemSendsGeometryChanges
        )
        self.setCursor(QtCore.Qt.OpenHandCursor)
        self._apply_selected_style(False)

        # Create resize handles
        self.handles = self._create_handles()
        self._update_handle_positions()
        self._set_handles_visible(False)

    def _apply_selected_style(self, selected: bool) -> None:
        if selected:
            pen = QtGui.QPen(QtGui.QColor(255, 140, 0))
            pen.setWidthF(2.0)
            pen.setStyle(QtCore.Qt.SolidLine)
            brush = QtGui.QBrush(QtGui.QColor(255, 140, 0, 80))
        else:
            pen = QtGui.QPen(QtGui.QColor(30, 144, 255))
            pen.setWidthF(1.5)
            pen.setStyle(QtCore.Qt.DashLine)
            brush = QtGui.QBrush(QtGui.QColor(30, 144, 255, 60))
        self.setPen(pen)
        self.setBrush(brush)

    def itemChange(self, change, value):
        if change == QtWidgets.QGraphicsItem.ItemSelectedHasChanged:
            self._apply_selected_style(bool(value))
            self._set_handles_visible(bool(value))
        elif change == QtWidgets.QGraphicsItem.ItemPositionHasChanged:
            if not self._suppress_callback and self._change_callback:
                self._change_callback(self)
        return super().itemChange(change, value)

    def mousePressEvent(self, event):
        if event.button() == QtCore.Qt.LeftButton:
            self.setCursor(QtCore.Qt.ClosedHandCursor)
        super().mousePressEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() == QtCore.Qt.LeftButton:
            self.setCursor(QtCore.Qt.OpenHandCursor)
            if not self._suppress_callback and self._change_callback:
                self._change_callback(self)
        super().mouseReleaseEvent(event)

    def grow(self, dw: float, dh: float) -> None:
        rect = self.rect()
        new_w = max(4.0, rect.width() + dw)
        new_h = max(4.0, rect.height() + dh)
        if new_w != rect.width() or new_h != rect.height():
            rect.setWidth(new_w)
            rect.setHeight(new_h)
            self.setRect(rect)
            self._update_handle_positions()
            if not self._suppress_callback and self._change_callback:
                self._change_callback(self)

    def _create_handles(self):
        """Create 4 corner resize handles."""
        return {
            "NW": ResizeHandle("NW", self),
            "NE": ResizeHandle("NE", self),
            "SW": ResizeHandle("SW", self),
            "SE": ResizeHandle("SE", self),
        }

    def _update_handle_positions(self) -> None:
        """Position handles at the corners of the field rectangle."""
        rect = self.rect()
        handle_offset = 4.0  # Half of handle size (8x8)

        self.handles["NW"].setPos(rect.left() - handle_offset, rect.top() - handle_offset)
        self.handles["NE"].setPos(rect.right() - handle_offset, rect.top() - handle_offset)
        self.handles["SW"].setPos(rect.left() - handle_offset, rect.bottom() - handle_offset)
        self.handles["SE"].setPos(rect.right() - handle_offset, rect.bottom() - handle_offset)

    def _set_handles_visible(self, visible: bool) -> None:
        """Show or hide resize handles."""
        for handle in self.handles.values():
            handle.setVisible(visible)

    def pdf_bbox(self) -> BoundingBox:
        x = self.pos().x()
        y = self.pos().y()
        w = self.rect().width()
        h = self.rect().height()
        x0 = x
        x1 = x + w
        y1 = self.page_height - y
        y0 = y1 - h
        return (x0, y0, x1, y1)

    def sync_with_record(self, record: FieldRecord, page_height: float) -> None:
        rect = pdf_to_scene_rect(record.bbox, page_height)
        self._suppress_callback = True
        self.page_height = page_height
        self.setRect(QtCore.QRectF(0, 0, rect.width(), rect.height()))
        self.setPos(rect.x(), rect.y())
        self._suppress_callback = False


# ---------------------------------------------------------------------------
# Save Dialog
# ---------------------------------------------------------------------------


@dataclass
class SaveOutputs:
    """User choices from the save dialog."""
    json_path: Optional[Path]
    pdf_path: Optional[Path]
    run_audit: bool
    audit_dir: Path
    replace_live: bool


class SaveDialog(QtWidgets.QDialog):
    """Dialog for choosing save outputs and optional quality audit."""

    def __init__(self, state: EditorState) -> None:
        super().__init__()
        self.setWindowTitle("Save Manual Editor Outputs")
        self.state = state
        self.outputs = SaveOutputs(
            json_path=None,
            pdf_path=None,
            run_audit=False,
            audit_dir=(state.pdf_path.parent / "logs" / "manual_editor").resolve(),
            replace_live=True,
        )
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QtWidgets.QVBoxLayout(self)
        layout.setSpacing(12)
        
        # Add helpful header
        header = QtWidgets.QLabel("Choose where to save your edited PDF and field data:")
        header.setWordWrap(True)
        layout.addWidget(header)

        form = QtWidgets.QFormLayout()
        form.setSpacing(8)

        # JSON output
        self.save_json_checkbox = QtWidgets.QCheckBox("Save refinements JSON")
        self.save_json_checkbox.setChecked(True)
        self.json_path_edit = QtWidgets.QLineEdit(str(self._default_json_path()))
        self.json_path_edit.textChanged.connect(self._update_path_labels)
        json_browse = QtWidgets.QPushButton("Browse…")
        json_browse.clicked.connect(self._browse_json)
        json_row = QtWidgets.QHBoxLayout()
        json_row.addWidget(self.json_path_edit)
        json_row.addWidget(json_browse)
        form.addRow(self.save_json_checkbox, json_row)
        
        # JSON absolute path label
        self.json_abs_label = QtWidgets.QLabel()
        self.json_abs_label.setWordWrap(True)
        font = self.json_abs_label.font()
        font.setPointSize(font.pointSize() - 1)
        self.json_abs_label.setFont(font)
        self.json_abs_label.setStyleSheet("color: #666;")
        form.addRow("", self.json_abs_label)

        # PDF output
        self.save_pdf_checkbox = QtWidgets.QCheckBox("Regenerate fillable PDF")
        self.save_pdf_checkbox.setChecked(True)
        self.pdf_path_edit = QtWidgets.QLineEdit(str(self._default_pdf_path()))
        self.pdf_path_edit.textChanged.connect(self._update_path_labels)
        self.replace_live_checkbox = QtWidgets.QCheckBox("Also copy to .v2.pdf (for interview testing)")
        self.replace_live_checkbox.setChecked(False)  # CHANGED: Default to False to avoid confusion
        self.replace_live_checkbox.setToolTip("Creates a copy named '{filename}.v2.pdf' for quick testing")
        pdf_browse = QtWidgets.QPushButton("Browse…")
        pdf_browse.clicked.connect(self._browse_pdf)
        pdf_row = QtWidgets.QHBoxLayout()
        pdf_row.addWidget(self.pdf_path_edit)
        pdf_row.addWidget(pdf_browse)
        pdf_col = QtWidgets.QVBoxLayout()
        pdf_col.addLayout(pdf_row)
        pdf_col.addWidget(self.replace_live_checkbox)
        form.addRow(self.save_pdf_checkbox, pdf_col)
        
        # PDF absolute path label
        self.pdf_abs_label = QtWidgets.QLabel()
        self.pdf_abs_label.setWordWrap(True)
        font = self.pdf_abs_label.font()
        font.setPointSize(font.pointSize() - 1)
        self.pdf_abs_label.setFont(font)
        self.pdf_abs_label.setStyleSheet("color: #666;")
        form.addRow("", self.pdf_abs_label)

        layout.addLayout(form)
        
        # Update labels initially
        self._update_path_labels()

        # Audit option
        self.audit_checkbox = QtWidgets.QCheckBox("Run quality audit after save")
        self.audit_checkbox.setChecked(False)
        self.audit_checkbox.setEnabled(self.state.targets_path is not None)
        layout.addWidget(self.audit_checkbox)

        audit_row = QtWidgets.QHBoxLayout()
        self.audit_dir_edit = QtWidgets.QLineEdit(str(self.outputs.audit_dir))
        audit_browse = QtWidgets.QPushButton("Browse…")
        audit_browse.clicked.connect(self._browse_audit_dir)
        audit_row.addWidget(QtWidgets.QLabel("Audit output folder:"))
        audit_row.addWidget(self.audit_dir_edit)
        audit_row.addWidget(audit_browse)
        layout.addLayout(audit_row)

        # Buttons
        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Save | QtWidgets.QDialogButtonBox.Cancel,
            QtCore.Qt.Horizontal,
            self,
        )
        buttons.accepted.connect(self._on_accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        # Wire checkbox enable/disable behaviour
        self.save_json_checkbox.toggled.connect(self.json_path_edit.setEnabled)
        self.save_pdf_checkbox.toggled.connect(self.pdf_path_edit.setEnabled)
        self.save_pdf_checkbox.toggled.connect(self.replace_live_checkbox.setEnabled)
        self.audit_checkbox.toggled.connect(self.audit_dir_edit.setEnabled)

        self.json_path_edit.setEnabled(True)
        self.pdf_path_edit.setEnabled(True)
        self.replace_live_checkbox.setEnabled(True)
        self.audit_dir_edit.setEnabled(False)

    def _default_json_path(self) -> Path:
        if self.state.refinements_path:
            return self.state.refinements_path
        return self.state.pdf_path.with_name(f"{self.state.pdf_path.stem}_manual_refinements.json")

    def _default_pdf_path(self) -> Path:
        if self.state.last_output_pdf:
            return self.state.last_output_pdf
        return self.state.pdf_path.with_name(f"{self.state.pdf_path.stem}_manual_editor.pdf")

    def _browse_json(self) -> None:
        filename, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Choose JSON output path",
            str(self._default_json_path()),
            "JSON Files (*.json)",
        )
        if filename:
            self.json_path_edit.setText(filename)

    def _browse_pdf(self) -> None:
        filename, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Choose PDF output path",
            str(self._default_pdf_path()),
            "PDF Files (*.pdf)",
        )
        if filename:
            self.pdf_path_edit.setText(filename)

    def _browse_audit_dir(self) -> None:
        directory = QtWidgets.QFileDialog.getExistingDirectory(
            self,
            "Choose audit output directory",
            str(self.outputs.audit_dir),
        )
        if directory:
            self.audit_dir_edit.setText(directory)
    
    def _update_path_labels(self) -> None:
        """Update the absolute path labels when user changes filenames."""
        try:
            json_abs = Path(self.json_path_edit.text()).resolve()
            self.json_abs_label.setText(f"→ {json_abs}")
        except Exception:
            self.json_abs_label.setText("")
        
        try:
            pdf_abs = Path(self.pdf_path_edit.text()).resolve()
            self.pdf_abs_label.setText(f"→ {pdf_abs}")
        except Exception:
            self.pdf_abs_label.setText("")

    def _on_accept(self) -> None:
        if not self.save_json_checkbox.isChecked() and not self.save_pdf_checkbox.isChecked():
            QtWidgets.QMessageBox.warning(
                self,
                "Nothing to save",
                "Select at least one output: refinements JSON or fillable PDF.",
            )
            return

        json_path: Optional[Path] = None
        pdf_path: Optional[Path] = None

        if self.save_json_checkbox.isChecked():
            json_path = Path(self.json_path_edit.text()).resolve()
            json_path.parent.mkdir(parents=True, exist_ok=True)

        if self.save_pdf_checkbox.isChecked():
            pdf_path = Path(self.pdf_path_edit.text()).resolve()
            pdf_path.parent.mkdir(parents=True, exist_ok=True)

        audit_dir = Path(self.audit_dir_edit.text()).resolve()
        if self.audit_checkbox.isChecked():
            audit_dir.mkdir(parents=True, exist_ok=True)

        self.outputs = SaveOutputs(
            json_path=json_path,
            pdf_path=pdf_path,
            run_audit=self.audit_checkbox.isChecked(),
            audit_dir=audit_dir,
            replace_live=self.replace_live_checkbox.isChecked(),
        )
        self.accept()


# ---------------------------------------------------------------------------
# Main Editor Window
# ---------------------------------------------------------------------------


class EditorWindow(QtWidgets.QMainWindow):
    """Main interactive window for manual field placement."""

    def __init__(self, state: EditorState) -> None:
        super().__init__()
        self.state = state
        self.doc = fitz.open(str(state.pdf_path))
        self.current_page = 0
        self.zoom_factor = 1.0
        self._pixmap_cache: Dict[int, QtGui.QPixmap] = {}
        self._field_items: List[FieldGraphicsItem] = []
        self._updating = False

        self.setWindowTitle(f"Manual Field Editor — {state.pdf_path.name}")
        self.resize(1100, 800)

        self._build_ui()
        self._refresh_scene()

    # ------------------------------- UI setup ---------------------------------

    def _build_ui(self) -> None:
        toolbar = QtWidgets.QToolBar("Controls")
        toolbar.setMovable(False)
        self.addToolBar(toolbar)

        prev_action = QtGui.QAction("◀", self)
        prev_action.triggered.connect(lambda: self._step_page(-1))
        toolbar.addAction(prev_action)

        next_action = QtGui.QAction("▶", self)
        next_action.triggered.connect(lambda: self._step_page(1))
        toolbar.addAction(next_action)

        toolbar.addSeparator()

        self.page_spin = QtWidgets.QSpinBox()
        self.page_spin.setMinimum(1)
        self.page_spin.setMaximum(max(1, self.state.page_count))
        self.page_spin.setValue(1)
        self.page_spin.valueChanged.connect(lambda value: self._set_page(value - 1))
        toolbar.addWidget(QtWidgets.QLabel("Page:"))
        toolbar.addWidget(self.page_spin)

        toolbar.addSeparator()

        self.zoom_combo = QtWidgets.QComboBox()
        for value in [50, 75, 100, 125, 150, 200, 300]:
            self.zoom_combo.addItem(f"{value}%")
        self.zoom_combo.setCurrentText("100%")
        self.zoom_combo.currentTextChanged.connect(self._on_zoom_changed)
        toolbar.addWidget(QtWidgets.QLabel("Zoom:"))
        toolbar.addWidget(self.zoom_combo)

        toolbar.addSeparator()

        add_text = QtGui.QAction("Add Text Field", self)
        add_text.triggered.connect(lambda: self._add_field("text"))
        toolbar.addAction(add_text)

        add_checkbox = QtGui.QAction("Add Checkbox", self)
        add_checkbox.triggered.connect(lambda: self._add_field("checkbox"))
        toolbar.addAction(add_checkbox)

        toolbar.addSeparator()

        self.delete_action = QtGui.QAction("Delete Field", self)
        self.delete_action.triggered.connect(self._delete_selected_fields)
        self.delete_action.setEnabled(False)
        toolbar.addAction(self.delete_action)

        toolbar.addSeparator()

        save_action = QtGui.QAction("Save…", self)
        save_action.triggered.connect(self._perform_save)
        toolbar.addAction(save_action)

        # Central graphics view
        central = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(central)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self.scene = QtWidgets.QGraphicsScene()
        self.scene.selectionChanged.connect(self._on_selection_changed)

        self.view = QtWidgets.QGraphicsView(self.scene)
        self.view.setRenderHints(
            QtGui.QPainter.Antialiasing | QtGui.QPainter.SmoothPixmapTransform
        )
        self.view.setDragMode(QtWidgets.QGraphicsView.RubberBandDrag)
        self.view.setViewportUpdateMode(QtWidgets.QGraphicsView.FullViewportUpdate)
        self.view.setTransformationAnchor(QtWidgets.QGraphicsView.AnchorUnderMouse)

        layout.addWidget(self.view)
        self.setCentralWidget(central)

        # Field properties dock
        self.properties_panel = FieldPropertiesPanel(self.state, self)
        self.addDockWidget(QtCore.Qt.RightDockWidgetArea, self.properties_panel)

        self.statusBar().showMessage("Ready")

    # ------------------------------ Data binding ------------------------------

    def refresh_field_scene(self, select_index: Optional[int] = None) -> None:
        self._refresh_scene(select_index=select_index)

    def on_field_properties_changed(self, index: Optional[int]) -> None:
        if index is not None:
            page_width, page_height = self.state.page_sizes[self.current_page]
            record = self.state.fields[index]
            for item in self._field_items:
                if item.field_index == index:
                    item.sync_with_record(record, page_height)
                    break
        self._update_status()

    def _update_properties_panel(self, force_index: Optional[int] = None) -> None:
        if force_index is not None:
            self.properties_panel.set_current_field(force_index)
            return
        selected = self._selected_field_items()
        if selected:
            self.properties_panel.set_current_field(selected[0].field_index)
        else:
            self.properties_panel.set_current_field(None)

    def _step_page(self, delta: int) -> None:
        new_page = max(0, min(self.state.page_count - 1, self.current_page + delta))
        self._set_page(new_page)

    def _set_page(self, index: int) -> None:
        if index == self.current_page or index < 0 or index >= self.state.page_count:
            return
        self.current_page = index
        self.page_spin.blockSignals(True)
        self.page_spin.setValue(index + 1)
        self.page_spin.blockSignals(False)
        self._refresh_scene()

    def _on_zoom_changed(self, text: str) -> None:
        try:
            value = float(text.strip("%"))
        except ValueError:
            return
        self.zoom_factor = max(0.1, value / 100.0)
        self._apply_zoom()
        self._update_status()

    def _apply_zoom(self) -> None:
        self.view.resetTransform()
        self.view.scale(self.zoom_factor, self.zoom_factor)

    def _render_page_pixmap(self, page_index: int) -> QtGui.QPixmap:
        if page_index in self._pixmap_cache:
            return self._pixmap_cache[page_index]

        page = self.doc[page_index]
        pix = page.get_pixmap()  # default 72 dpi -> matches PDF units
        fmt = QtGui.QImage.Format_RGBA8888 if pix.alpha else QtGui.QImage.Format_RGB888
        image = QtGui.QImage(pix.samples, pix.width, pix.height, pix.stride, fmt)
        image = image.copy()
        if not pix.alpha:
            image = image.convertToFormat(QtGui.QImage.Format_RGBA8888)
        pixmap = QtGui.QPixmap.fromImage(image)
        self._pixmap_cache[page_index] = pixmap
        return pixmap

    def _refresh_scene(self, select_index: Optional[int] = None) -> None:
        self._updating = True
        self.scene.clear()

        pixmap = self._render_page_pixmap(self.current_page)
        background = self.scene.addPixmap(pixmap)
        background.setZValue(-10)
        background.setPos(0, 0)

        page_width, page_height = self.state.page_sizes[self.current_page]
        self.scene.setSceneRect(0, 0, page_width, page_height)

        self._field_items = []
        current_page_records: List[Tuple[int, FieldRecord]] = [
            (idx, record)
            for idx, record in enumerate(self.state.fields)
            if record.page == self.current_page
        ]

        select_visible_index: Optional[int] = None

        for visible_index, (state_index, record) in enumerate(current_page_records):
            item = FieldGraphicsItem(state_index, record, page_height, self._on_item_changed)
            item.visible_index = visible_index
            item.field_index = state_index
            self.scene.addItem(item)
            self._field_items.append(item)

            if select_index is not None and state_index == select_index:
                item.setSelected(True)
                select_visible_index = visible_index

        if select_visible_index is not None:
            for item in self._field_items:
                if item.visible_index == select_visible_index:
                    self.view.centerOn(item)
                    break

        self._updating = False
        self._apply_zoom()
        self._update_status()
        self._update_properties_panel()

    def _on_item_changed(self, item: FieldGraphicsItem) -> None:
        if self._updating:
            return

        page_width, page_height = self.state.page_sizes[self.current_page]
        bbox = item.pdf_bbox()
        bbox = clamp_bbox_to_page(bbox, page_width, page_height)

        record = self.state.fields[item.field_index]
        record.bbox = bbox
        self.state.set_field(item.field_index, record)

        item.sync_with_record(record, page_height)
        self._update_status(record)
        self._update_properties_panel(force_index=item.field_index)

    def _perform_save(self) -> None:
        dialog = SaveDialog(self.state)
        if not dialog.exec():
            return

        outputs = dialog.outputs
        saved_files = []
        
        # Save JSON
        if outputs.json_path:
            self.state.save_refinements(outputs.json_path)
            saved_files.append(("Refinements JSON", outputs.json_path))
            logger.info(f"Saved refinements JSON to: {outputs.json_path}")

        # Save PDF
        pdf_result = None
        if outputs.pdf_path:
            pdf_result = self.state.save_pdf(outputs.pdf_path)
            saved_files.append(("Fillable PDF", outputs.pdf_path))
            logger.info(f"Saved PDF to: {outputs.pdf_path}")
            
            # Handle "replace live" copy
            if outputs.replace_live:
                live_target = self.state.pdf_path.with_name(f"{self.state.pdf_path.stem}.v2.pdf")
                if outputs.pdf_path.resolve() != live_target.resolve():
                    shutil.copy2(outputs.pdf_path, live_target)
                    self.state.last_output_pdf = live_target
                    saved_files.append(("Live Template (v2)", live_target))
                    logger.info(f"Copied to live template: {live_target}")

        # Build confirmation message
        msg_parts = ["✓ Save successful!\n"]
        msg_parts.append(f"Saved {len(self.state.fields)} fields.\n\n")
        msg_parts.append("Output files:\n")
        
        for label, path in saved_files:
            msg_parts.append(f"\n• {label}:")
            msg_parts.append(f"  {path}")
        
        if pdf_result:
            msg_parts.append(f"\n\nField count: {pdf_result.get('field_count', 0)}")
        
        # Show success dialog with option to open PDF
        success_dialog = QtWidgets.QMessageBox(self)
        success_dialog.setWindowTitle("Save Complete")
        success_dialog.setText("".join(msg_parts))
        success_dialog.setIcon(QtWidgets.QMessageBox.Information)
        
        # Add "Open PDF" button if we saved a PDF
        if outputs.pdf_path:
            open_button = success_dialog.addButton("Open PDF", QtWidgets.QMessageBox.ActionRole)
            open_folder_button = success_dialog.addButton("Open Folder", QtWidgets.QMessageBox.ActionRole)
        
        success_dialog.addButton(QtWidgets.QMessageBox.Ok)
        
        clicked_button = success_dialog.exec()
        
        # Handle button clicks
        if outputs.pdf_path:
            clicked = success_dialog.clickedButton()
            if clicked == open_button:
                import subprocess
                # Open PDF in default viewer
                subprocess.run(["open", str(outputs.pdf_path)])
                
                # Also run validation
                self._run_field_validation(outputs.pdf_path, outputs.json_path)
            elif clicked == open_folder_button:
                import subprocess
                subprocess.run(["open", "-R", str(outputs.pdf_path)])
        
        # Auto-open PDF if configured (and user clicked OK)
        if outputs.pdf_path and not clicked in [open_button, open_folder_button]:
            # Ask user if they want auto-open next time
            reply = QtWidgets.QMessageBox.question(
                self,
                "Open PDF Automatically?",
                "Would you like to automatically open saved PDFs in the future?\n\n"
                "This helps verify fields immediately.",
                QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No
            )
            if reply == QtWidgets.QMessageBox.Yes:
                import subprocess
                subprocess.run(["open", str(outputs.pdf_path)])
                self._run_field_validation(outputs.pdf_path, outputs.json_path)
        
        # Update status bar
        self.statusBar().showMessage(f"✓ Saved {len(saved_files)} file(s)")

        # Run audit if requested
        if outputs.run_audit and self.state.targets_path:
            report = self.state.run_quality_audit(
                self.state.targets_path,
                outputs.audit_dir,
            )
            QtWidgets.QMessageBox.information(
                self,
                "Audit Complete",
                (
                    "Quality audit finished.\n"
                    f"Pass rate: {report.get('pass_rate', 0)*100:.1f}%\n"
                    f"Report saved to {report.get('audit_markdown')}"
                ),
            )
        elif outputs.run_audit and not self.state.targets_path:
            QtWidgets.QMessageBox.warning(
                self,
                "Audit Skipped",
                "No target regions JSON available for this PDF.",
            )

    def _run_field_validation(self, pdf_path: Path, json_path: Optional[Path]) -> None:
        """Run comprehensive field validation and show results."""
        import subprocess
        import sys
        
        # Build validation command
        validator_path = Path(__file__).parent / "validate_pdf_fields.py"
        
        if not validator_path.exists():
            logger.warning(f"Validator not found: {validator_path}")
            return
        
        cmd = [sys.executable, str(validator_path), str(pdf_path), "--overlay"]
        
        # Add expected JSON if available
        if json_path and json_path.exists():
            cmd.insert(3, str(json_path))
        
        # Run validation in a new terminal window for visibility
        try:
            if sys.platform == "darwin":  # macOS
                # Run in Terminal.app so user can see output
                script = f'cd "{pdf_path.parent}" && {" ".join(cmd)} && echo "\\n\\nPress any key to close..." && read -n 1'
                subprocess.Popen([
                    "osascript", "-e",
                    f'tell application "Terminal" to do script "{script}"'
                ])
            else:
                # Fallback: just run it
                subprocess.run(cmd)
        except Exception as e:
            logger.error(f"Failed to run validation: {e}")
    
    def _on_selection_changed(self) -> None:
        selected = self._selected_field_items()
        selected = self._selected_field_items()
        self.delete_action.setEnabled(len(selected) > 0)
        self._update_status()
        if selected:
            self.properties_panel.set_current_field(selected[0].field_index)
        else:
            self.properties_panel.set_current_field(None)

    def _update_status(self, record: Optional[FieldRecord] = None) -> None:
        if record is None:
            selected = [
                item
                for item in self.scene.selectedItems()
                if isinstance(item, FieldGraphicsItem)
            ]
            if selected:
                record = self.state.fields[selected[0].field_index]

        if record:
            x0, y0, x1, y1 = record.bbox
            self.statusBar().showMessage(
                f"{record.name} ({record.field_type}) — x0:{x0:.1f}, y0:{y0:.1f}, w:{x1-x0:.1f}, h:{y1-y0:.1f}"
            )
        else:
            self.statusBar().showMessage(
                f"Page {self.current_page + 1}/{self.state.page_count} · "
                f"{len(self.state.by_page(self.current_page))} fields · "
                f"Zoom {int(self.zoom_factor * 100)}%"
            )
        self._update_properties_panel()

    def _add_field(self, field_type: str) -> None:
        page_width, page_height = self.state.page_sizes[self.current_page]
        default_height = 14.0 if field_type == "checkbox" else 18.0
        default_width = default_height if field_type == "checkbox" else 140.0
        margin = 36.0
        x0 = min(margin, max(0.0, page_width - default_width - margin))
        y1 = page_height - margin
        y0 = y1 - default_height
        record = FieldRecord(
            name=self.state.next_field_name(field_type),
            field_type=field_type,
            page=self.current_page,
            bbox=(x0, y0, x0 + default_width, y0 + default_height),
        )
        self.state.add_field(record)
        select_index = len(self.state.fields) - 1
        self._refresh_scene(select_index=select_index)

    def _delete_selected_fields(self) -> None:
        """Delete currently selected fields."""
        selected = self._selected_field_items()
        if not selected:
            return

        # Confirmation for multiple fields
        if len(selected) > 1:
            reply = QtWidgets.QMessageBox.question(
                self,
                "Delete Fields",
                f"Delete {len(selected)} selected fields?",
                QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
            )
            if reply != QtWidgets.QMessageBox.Yes:
                return

        # Collect indices and remove from state (in reverse order to preserve indices)
        indices_to_remove = sorted([item.field_index for item in selected], reverse=True)
        for idx in indices_to_remove:
            self.state.remove_field(idx)

        # Refresh the scene and rebuild field indices for remaining items
        self._refresh_scene()
        self.statusBar().showMessage(f"Deleted {len(selected)} field(s)")

    def _selected_field_items(self) -> List[FieldGraphicsItem]:
        return [
            item
            for item in self.scene.selectedItems()
            if isinstance(item, FieldGraphicsItem)
        ]


    # --------------------------- keyboard shortcuts ---------------------------

    def keyPressEvent(self, event: QtGui.QKeyEvent) -> None:
        if self._handle_key(event):
            return
        super().keyPressEvent(event)

    def _handle_key(self, event: QtGui.QKeyEvent) -> bool:
        selected = [
            item
            for item in self.scene.selectedItems()
            if isinstance(item, FieldGraphicsItem)
        ]
        if not selected:
            return False

        step = 1.0
        if event.modifiers() & QtCore.Qt.ControlModifier:
            step = 5.0
        if event.modifiers() & QtCore.Qt.AltModifier:
            step = 10.0

        key = event.key()
        if event.modifiers() & QtCore.Qt.ShiftModifier:
            if key == QtCore.Qt.Key_Right:
                for item in selected:
                    item.grow(step, 0.0)
                return True
            if key == QtCore.Qt.Key_Left:
                for item in selected:
                    item.grow(-step, 0.0)
                return True
            if key == QtCore.Qt.Key_Down:
                for item in selected:
                    item.grow(0.0, step)
                return True
            if key == QtCore.Qt.Key_Up:
                for item in selected:
                    item.grow(0.0, -step)
                return True
        else:
            if key == QtCore.Qt.Key_Right:
                for item in selected:
                    item.moveBy(step, 0.0)
                return True
            if key == QtCore.Qt.Key_Left:
                for item in selected:
                    item.moveBy(-step, 0.0)
                return True
            if key == QtCore.Qt.Key_Down:
                for item in selected:
                    item.moveBy(0.0, step)
                return True
            if key == QtCore.Qt.Key_Up:
                for item in selected:
                    item.moveBy(0.0, -step)
                return True

        # Delete/Backspace to remove fields
        if key in (QtCore.Qt.Key_Delete, QtCore.Qt.Key_Backspace):
            self._delete_selected_fields()
            return True

        return False

    # ------------------------------ Qt lifecycle -----------------------------

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        try:
            if self.doc is not None:
                self.doc.close()
        finally:
            super().closeEvent(event)


# ---------------------------------------------------------------------------
# CLI utility (for smoke testing prior to GUI work)


def _cli_summary(args: Any) -> None:
    pdf, fields, yaml_path = _resolve_cli_inputs(args.pdf, args.fields)
    state = EditorState.from_sources(pdf, fields, yaml_path=yaml_path)
    print(json.dumps(
        {
            "pdf": state.pdf_path.name,
            "page_count": state.page_count,
            "field_count": len(state.fields),
            "by_page": {page: len(state.by_page(page)) for page in range(state.page_count)},
        },
        indent=2,
    ))


def _cli_export(args: Any) -> None:
    pdf, fields, yaml_path = _resolve_cli_inputs(args.pdf, args.fields, require_fields=True)
    state = EditorState.from_sources(pdf, fields, yaml_path=yaml_path)
    state.save_refinements(Path(args.output))
    print(f"Saved refinements JSON to {args.output}")
    if args.output_pdf:
        state.save_pdf(Path(args.output_pdf))
        print(f"Wrote updated PDF to {args.output_pdf}")


def _resolve_cli_inputs(
    pdf_arg: Optional[str],
    fields_arg: Optional[str],
    *,
    allow_dialog: bool = False,
    require_fields: bool = False,
) -> Tuple[Optional[Path], Optional[Path], Optional[Path]]:
    """Resolve PDF, fields JSON, and optional YAML for name mapping."""

    pdf_path: Optional[Path] = None
    fields_path: Optional[Path] = None
    yaml_path: Optional[Path] = None

    if pdf_arg:
        pdf_path = Path(pdf_arg)
    elif allow_dialog:
        app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
        filename, _ = QtWidgets.QFileDialog.getOpenFileName(
            None,
            "Select PDF to Edit",
            str(Path.cwd()),
            "PDF Files (*.pdf)",
        )
        pdf_path = Path(filename) if filename else None
    else:
        raise ValueError("PDF path is required when dialogs are disabled")

    if pdf_path:
        yaml_candidate = pdf_path.with_suffix(".yml")
        if yaml_candidate.exists():
            yaml_path = yaml_candidate

    if fields_arg:
        fields_path = Path(fields_arg)
    elif require_fields:
        raise ValueError("Fields JSON is required for this operation")
    elif allow_dialog and pdf_path is not None:
        # Optionally prompt for fields JSON
        reply = QtWidgets.QMessageBox.question(
            None,
            "Load Existing Fields?",
            "Would you like to load an existing refinements/fields JSON file?",
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
        )
        if reply == QtWidgets.QMessageBox.Yes:
            filename, _ = QtWidgets.QFileDialog.getOpenFileName(
                None,
                "Select Refinements JSON",
                str(Path.cwd()),
                "JSON Files (*.json)",
            )
            fields_path = Path(filename) if filename else None

    # Optionally prompt for YAML attachment if none auto-detected
    if allow_dialog and pdf_path and yaml_path is None:
        reply = QtWidgets.QMessageBox.question(
            None,
            "Load interview YAML?",
            "Would you like to select a interview YAML file for field names?",
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
        )
        if reply == QtWidgets.QMessageBox.Yes:
            filename, _ = QtWidgets.QFileDialog.getOpenFileName(
                None,
                "Select interview YAML",
                str(Path.cwd()),
                "YAML Files (*.yml *.yaml)",
            )
            if filename:
                yaml_path = Path(filename)

    return pdf_path, fields_path, yaml_path


def _cli_gui(args: Any) -> None:
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    pdf_path, fields_path, yaml_path = _resolve_cli_inputs(args.pdf, args.fields, allow_dialog=True)
    if pdf_path is None:
        print("No PDF selected. Exiting.")
        return

    state = EditorState.from_sources(pdf_path, fields_path, yaml_path=yaml_path)
    window = EditorWindow(state)
    window.show()
    app.exec()


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Manual editor scaffold utilities")
    subparsers = parser.add_subparsers(dest="command")

    summary = subparsers.add_parser("summary", help="Print PDF + field counts")
    summary.add_argument("pdf", help="Path to PDF")
    summary.add_argument("--fields", help="Optional fields/refinements JSON")
    summary.set_defaults(func=_cli_summary)

    export = subparsers.add_parser("export", help="Reserialize fields to refinements JSON")
    export.add_argument("pdf", help="Path to PDF")
    export.add_argument("fields", help="Existing fields JSON (export/refinements)")
    export.add_argument("output", help="Where to save refinements JSON")
    export.add_argument("--output-pdf", help="Optional path to regenerate fillable PDF")
    export.set_defaults(func=_cli_export)

    gui = subparsers.add_parser("gui", help="Launch interactive manual editor UI")
    gui.add_argument("pdf", help="Path to PDF")
    gui.add_argument("--fields", help="Existing fields JSON (defaults to none)")
    gui.set_defaults(func=_cli_gui)

    args = parser.parse_args()
    if not getattr(args, "command", None):
        parser.print_help()
        return
    args.func(args)


if __name__ == "__main__":
    main()

