#!/usr/bin/env python3
"""Lightweight QPainter C3D trial viewer."""

from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np

from mocap_units import point_unit_scale_to_mm

ANGLE_LABEL_REGEX = r"(?i)(^.*angles?$|^.*_angle[s]?$|angle)"

AXIS_COLORS = {"x": "#ef4444", "y": "#22c55e", "z": "#3b82f6"}
SEGMENT_COLORS = (
    "#2563eb",
    "#dc2626",
    "#16a34a",
    "#9333ea",
    "#ea580c",
    "#0891b2",
    "#be123c",
    "#4f46e5",
)


@dataclass
class C3DMarkerData:
    labels: list[str]
    points: np.ndarray
    rate: float
    unit: str = "mm"

    @property
    def n_frames(self) -> int:
        return int(self.points.shape[2])


@dataclass
class MarkerVisualState:
    technical_labels: set[str] = field(default_factory=set)
    anatomical_labels: set[str] = field(default_factory=set)
    assigned_segments: dict[str, str] = field(default_factory=dict)
    active_labels: set[str] = field(default_factory=set)
    selected_label: str | None = None


def load_c3d_marker_data(
    path: Path, *, angle_label_regex: str = ANGLE_LABEL_REGEX
) -> C3DMarkerData:
    import ezc3d
    from compare_capture_systems import detect_angle_indices

    c3d = ezc3d.c3d(str(path))
    labels = [
        str(label).strip() for label in c3d["parameters"]["POINT"]["LABELS"]["value"]
    ]
    angle_indices = set(detect_angle_indices(c3d, labels, angle_label_regex).values())
    marker_indices = [
        index for index in range(len(labels)) if index not in angle_indices
    ]
    units = c3d["parameters"]["POINT"].get("UNITS", {}).get("value", ["mm"])
    unit = units[0] if units else "mm"
    points = np.asarray(c3d["data"]["points"][:3, marker_indices, :], dtype=float)
    points *= point_unit_scale_to_mm(str(unit))
    residuals = np.asarray(c3d["data"]["points"][3, marker_indices, :], dtype=float)
    points[:, residuals < 0] = np.nan
    rate = float(c3d["parameters"]["POINT"]["RATE"]["value"][0])
    return C3DMarkerData(
        labels=[labels[index] for index in marker_indices],
        points=points,
        rate=rate,
        unit="mm",
    )


def normalized(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm == 0:
        return vector
    return vector / norm


def rotation_matrix_from_drag(delta_x: float, delta_y: float) -> np.ndarray:
    rotvec = np.asarray([delta_y, delta_x, 0.0], dtype=float) * 0.006
    angle = float(np.linalg.norm(rotvec))
    if angle == 0:
        return np.eye(3)
    axis = rotvec / angle
    cross = np.array(
        [
            [0.0, -axis[2], axis[1]],
            [axis[2], 0.0, -axis[0]],
            [-axis[1], axis[0], 0.0],
        ]
    )
    return (
        np.eye(3) + math.sin(angle) * cross + (1.0 - math.cos(angle)) * (cross @ cross)
    )


def default_camera_matrix() -> np.ndarray:
    return normalized_camera_matrix(
        np.asarray(
            [
                [1.0, 0.0, 0.0],
                [0.0, 0.35, 0.94],
                [0.0, -0.94, 0.35],
            ],
            dtype=float,
        )
    )


def normalized_camera_matrix(matrix: np.ndarray) -> np.ndarray:
    right = normalized(np.asarray(matrix[0], dtype=float))
    up = normalized(np.asarray(matrix[1], dtype=float))
    depth = normalized(np.cross(right, up))
    up = normalized(np.cross(depth, right))
    return np.vstack((right, up, depth))


def camera_matrix_for_plane(plane: str) -> np.ndarray:
    matrices = {
        "XY": ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
        "YZ": ((0.0, 1.0, 0.0), (0.0, 0.0, 1.0), (1.0, 0.0, 0.0)),
        "XZ": ((1.0, 0.0, 0.0), (0.0, 0.0, 1.0), (0.0, -1.0, 0.0)),
        "ZX": ((0.0, 0.0, 1.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
    }
    return normalized_camera_matrix(np.asarray(matrices[plane.upper()], dtype=float))


def subject_horizontal_axis_from_pca(
    points: np.ndarray, vertical: np.ndarray
) -> np.ndarray:
    valid = points[:, np.all(np.isfinite(points), axis=0)]
    if valid.shape[1] < 3:
        return np.asarray((1.0, 0.0, 0.0), dtype=float)
    centered = valid - np.nanmean(valid, axis=1, keepdims=True)
    covariance = centered @ centered.T / max(1, valid.shape[1] - 1)
    _, vectors = np.linalg.eigh(covariance)
    for axis in vectors.T[::-1]:
        horizontal = axis - np.dot(axis, vertical) * vertical
        if np.linalg.norm(horizontal) > 1e-9:
            dominant = int(np.argmax(np.abs(horizontal)))
            if horizontal[dominant] < 0:
                horizontal = -horizontal
            return normalized(horizontal)
    return np.asarray((1.0, 0.0, 0.0), dtype=float)


def camera_matrix_for_subject_view(view: str, points: np.ndarray) -> np.ndarray:
    vertical = np.asarray((0.0, 0.0, 1.0), dtype=float)
    horizontal = subject_horizontal_axis_from_pca(points, vertical)
    forward = normalized(np.cross(vertical, horizontal))
    matrices = {
        "face": (horizontal, vertical, forward),
        "front": (horizontal, vertical, forward),
        "dos": (-horizontal, vertical, -forward),
        "back": (-horizontal, vertical, -forward),
        "cote": (forward, vertical, horizontal),
        "side": (forward, vertical, horizontal),
    }
    key = view.strip().lower()
    if key not in matrices:
        raise ValueError(f"Unknown subject view: {view}")
    return normalized_camera_matrix(np.asarray(matrices[key], dtype=float))


def project_points(
    points: np.ndarray,
    camera: np.ndarray,
    center: np.ndarray,
    scale: float,
    width: int,
    height: int,
) -> tuple[np.ndarray, np.ndarray]:
    rotated = camera @ (points - center[:, None])
    screen = np.vstack(
        (
            width / 2.0 + rotated[0] * scale,
            height / 2.0 - rotated[1] * scale,
        )
    )
    return screen, rotated[2]


def fit_center_and_scale(
    points: np.ndarray, width: int, height: int, camera: np.ndarray
) -> tuple[np.ndarray, float]:
    valid = points[:, np.all(np.isfinite(points), axis=0)]
    if valid.size == 0:
        return np.zeros(3), 1.0
    center = np.nanmean(valid, axis=1)
    rotated = camera @ (valid - center[:, None])
    span = np.nanmax(rotated[:2], axis=1) - np.nanmin(rotated[:2], axis=1)
    max_span = max(float(np.nanmax(span)), 1.0)
    scale = 0.82 * min(max(width, 1), max(height, 1)) / max_span
    return center, scale


try:
    from PySide6.QtCore import QPointF, Qt, QTimer
    from PySide6.QtGui import QAction, QColor, QPainter, QPen, QPolygonF
    from PySide6.QtWidgets import (
        QApplication,
        QCheckBox,
        QHBoxLayout,
        QLabel,
        QMenu,
        QPushButton,
        QSlider,
        QTableWidget,
        QTableWidgetItem,
        QVBoxLayout,
        QWidget,
    )

    QPAINTER_ANTIALIASING = QPainter.RenderHint.Antialiasing
    PYSIDE_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised when optional GUI dep is absent
    PYSIDE_AVAILABLE = False


if PYSIDE_AVAILABLE:

    class C3DTrialViewWidget(QWidget):
        def __init__(self, data: C3DMarkerData, parent: QWidget | None = None) -> None:
            super().__init__(parent)
            self.data = data
            self.state = MarkerVisualState(anatomical_labels=set(data.labels))
            self.frame = 0
            self.camera = default_camera_matrix()
            self.zoom = 1.0
            self.whole_body_view = True
            self._last_mouse_position: QPointF | None = None
            self._is_dragging = False
            self._redraw_pending = False
            self._redraw_timer = QTimer(self)
            self._redraw_timer.setSingleShot(True)
            self._redraw_timer.timeout.connect(self._flush_redraw)
            self.setMinimumSize(520, 420)
            self.setMouseTracking(True)

        def set_frame(self, frame: int) -> None:
            self.frame = max(0, min(int(frame), self.data.n_frames - 1))
            self.update()

        def set_selected_label(self, label: str | None) -> None:
            self.state.selected_label = label
            self.update()

        def set_whole_body_view(self, enabled: bool) -> None:
            self.whole_body_view = bool(enabled)
            self.update()

        def _request_redraw(self) -> None:
            if self._redraw_pending:
                return
            self._redraw_pending = True
            self._redraw_timer.start(33)

        def _flush_redraw(self) -> None:
            self._redraw_pending = False
            self.update()

        def reset_camera(self) -> None:
            self.camera = default_camera_matrix()
            self.zoom = 1.0
            self._is_dragging = False
            self.update()

        def set_camera_plane(self, plane: str) -> None:
            self.camera = camera_matrix_for_plane(plane)
            self.zoom = 1.0
            self.update()

        def set_subject_view(self, view: str) -> None:
            self.camera = camera_matrix_for_subject_view(view, self._current_points())
            self.zoom = 1.0
            self.update()

        def _current_points(self) -> np.ndarray:
            return self.data.points[:, :, self.frame]

        def _fit_points(self) -> np.ndarray:
            points = self._current_points()
            if self.whole_body_view:
                return points
            labels = self.state.active_labels or (
                {self.state.selected_label} if self.state.selected_label else set()
            )
            indices = [i for i, label in enumerate(self.data.labels) if label in labels]
            return points[:, indices] if indices else points

        def _marker_shape_and_color(self, label: str) -> tuple[str, QColor, int]:
            selected = label == self.state.selected_label
            if label in self.state.assigned_segments:
                segment = self.state.assigned_segments[label]
                index = abs(hash(segment)) % len(SEGMENT_COLORS)
                color = QColor(SEGMENT_COLORS[index])
                shape = "square" if label in self.state.technical_labels else "circle"
            elif label in self.state.technical_labels:
                color = QColor("#475569")
                shape = "square"
            elif label in self.state.anatomical_labels:
                color = QColor("#2563eb")
                shape = "circle"
            else:
                color = QColor("#111827")
                shape = "diamond"
            if (
                self.state.active_labels
                and label not in self.state.active_labels
                and not selected
            ):
                color = QColor("#d1d5db")
            return shape, color, 10 if selected else 6

        def paintEvent(self, _event) -> None:  # noqa: N802
            painter = QPainter(self)
            painter.fillRect(self.rect(), QColor("#ffffff"))
            painter.setRenderHint(QPAINTER_ANTIALIASING, not self._is_dragging)
            points = self._current_points()
            center, scale = fit_center_and_scale(
                self._fit_points(), self.width(), self.height(), self.camera
            )
            screen, depth = project_points(
                points,
                self.camera,
                center,
                scale * self.zoom,
                self.width(),
                self.height(),
            )
            order = np.argsort(depth)
            for index in order:
                point = points[:, index]
                if not np.all(np.isfinite(point)):
                    continue
                label = self.data.labels[index]
                shape, color, size = self._marker_shape_and_color(label)
                self._draw_marker(
                    painter,
                    QPointF(screen[0, index], screen[1, index]),
                    shape,
                    color,
                    size,
                )
            if not self._is_dragging:
                self._draw_labels(painter, screen, points)
            self._draw_axes(painter)
            self._draw_legend(painter)
            painter.end()

        def _draw_marker(
            self,
            painter: QPainter,
            center: QPointF,
            shape: str,
            color: QColor,
            size: int,
        ) -> None:
            painter.setPen(QPen(QColor("#111827") if shape == "diamond" else color, 1))
            painter.setBrush(color)
            if shape == "square":
                painter.drawRect(
                    int(center.x()) - size // 2, int(center.y()) - size // 2, size, size
                )
            elif shape == "diamond":
                radius = max(3, size // 2)
                painter.drawPolygon(
                    QPolygonF(
                        (
                            QPointF(center.x(), center.y() - radius),
                            QPointF(center.x() + radius, center.y()),
                            QPointF(center.x(), center.y() + radius),
                            QPointF(center.x() - radius, center.y()),
                        )
                    )
                )
            else:
                painter.drawEllipse(center, size / 2.0, size / 2.0)

        def _draw_labels(
            self, painter: QPainter, screen: np.ndarray, points: np.ndarray
        ) -> None:
            painter.setPen(QPen(QColor("#334155"), 1))
            for index, label in enumerate(self.data.labels):
                if not np.all(np.isfinite(points[:, index])):
                    continue
                if self.state.selected_label and label != self.state.selected_label:
                    continue
                painter.drawText(
                    QPointF(screen[0, index] + 6, screen[1, index] - 6), label
                )

        def _draw_axes(self, painter: QPainter) -> None:
            origin = QPointF(self.width() - 82, self.height() - 52)
            axes = (
                ("X", np.array((1.0, 0.0, 0.0)), "#ef4444"),
                ("Y", np.array((0.0, 1.0, 0.0)), "#22c55e"),
                ("Z", np.array((0.0, 0.0, 1.0)), "#3b82f6"),
            )
            painter.setBrush(QColor(255, 255, 255, 220))
            painter.setPen(QPen(QColor("#e5e7eb"), 1))
            painter.drawRect(int(origin.x()) - 22, int(origin.y()) - 44, 92, 76)
            for label, vector, color in axes:
                projected = self.camera @ vector
                endpoint = QPointF(
                    origin.x() + 34.0 * projected[0], origin.y() - 34.0 * projected[1]
                )
                painter.setPen(QPen(QColor(color), 3))
                painter.drawLine(origin, endpoint)
                painter.drawText(endpoint + QPointF(4, -4), label)

        def _draw_legend(self, painter: QPainter) -> None:
            x, y = 12, 18
            painter.setPen(QPen(QColor("#e5e7eb"), 1))
            painter.setBrush(QColor(255, 255, 255, 220))
            painter.drawRect(8, 8, 226, 78)
            entries = (
                ("rond anatomique", "circle"),
                ("carré technique", "square"),
                ("diamant non assigné", "diamond"),
            )
            for offset, (text, shape) in enumerate(entries):
                center = QPointF(x + 8, y + offset * 22)
                self._draw_marker(painter, center, shape, QColor("#64748b"), 7)
                painter.setPen(QPen(QColor("#111827"), 1))
                painter.drawText(x + 22, y + 5 + offset * 22, text)

        def mousePressEvent(self, event) -> None:  # noqa: N802
            if event.button() == Qt.LeftButton:
                self._last_mouse_position = event.position()
                self._is_dragging = True
                self.setCursor(Qt.ClosedHandCursor)

        def mouseMoveEvent(self, event) -> None:  # noqa: N802
            if not self._is_dragging or self._last_mouse_position is None:
                return
            position = event.position()
            delta = position - self._last_mouse_position
            self.camera = normalized_camera_matrix(
                rotation_matrix_from_drag(delta.x(), delta.y()) @ self.camera
            )
            self._last_mouse_position = position
            self._request_redraw()

        def mouseReleaseEvent(self, event) -> None:  # noqa: N802
            if event.button() == Qt.LeftButton:
                self._last_mouse_position = None
                self._is_dragging = False
                self.setCursor(Qt.OpenHandCursor)
                self.update()

        def wheelEvent(self, event) -> None:  # noqa: N802
            delta = event.angleDelta().y()
            self.zoom *= 1.12 if delta > 0 else 1 / 1.12
            self.zoom = max(0.05, min(self.zoom, 30.0))
            self.update()

        def mouseDoubleClickEvent(self, _event) -> None:  # noqa: N802
            self.reset_camera()

        def contextMenuEvent(self, event) -> None:  # noqa: N802
            menu = QMenu(self)
            for plane in ("XY", "YZ", "XZ"):
                action = QAction(plane, self)
                action.triggered.connect(
                    lambda _checked=False, p=plane: self.set_camera_plane(p)
                )
                menu.addAction(action)
            menu.addSeparator()
            for label, view in (("Face", "face"), ("Dos", "dos"), ("Côté", "cote")):
                action = QAction(label, self)
                action.triggered.connect(
                    lambda _checked=False, v=view: self.set_subject_view(v)
                )
                menu.addAction(action)
            menu.exec(event.globalPos())

    class C3DTrialViewerWindow(QWidget):
        def __init__(self, data: C3DMarkerData, c3d_path: Path) -> None:
            super().__init__()
            self.data = data
            self.c3d_path = c3d_path
            self.setWindowTitle(f"Visu 3D C3D - {c3d_path.name}")
            self.viewer = C3DTrialViewWidget(data)
            self.timer = QTimer(self)
            self.timer.timeout.connect(self._advance_frame)
            self._build_layout()

        def _build_layout(self) -> None:
            root = QVBoxLayout(self)
            content = QHBoxLayout()
            root.addLayout(content, 1)
            content.addWidget(self.viewer, 3)
            self.table = QTableWidget(len(self.data.labels), 1)
            self.table.setHorizontalHeaderLabels(["Marqueur"])
            for row, label in enumerate(self.data.labels):
                self.table.setItem(row, 0, QTableWidgetItem(label))
            self.table.itemSelectionChanged.connect(self._sync_selection)
            content.addWidget(self.table, 1)

            controls = QHBoxLayout()
            root.addLayout(controls)
            self.play_button = QPushButton("▶")
            self.play_button.setCheckable(True)
            self.play_button.toggled.connect(self._toggle_play)
            controls.addWidget(self.play_button)
            self.slider = QSlider(Qt.Horizontal)
            self.slider.setRange(0, max(0, self.data.n_frames - 1))
            self.slider.valueChanged.connect(self.viewer.set_frame)
            controls.addWidget(self.slider, 1)
            self.frame_label = QLabel(f"0 / {max(0, self.data.n_frames - 1)}")
            self.slider.valueChanged.connect(
                lambda value: self.frame_label.setText(
                    f"{value} / {max(0, self.data.n_frames - 1)}"
                )
            )
            controls.addWidget(self.frame_label)
            whole_body = QCheckBox("Whole body")
            whole_body.setChecked(True)
            whole_body.toggled.connect(self.viewer.set_whole_body_view)
            controls.addWidget(whole_body)

        def _sync_selection(self) -> None:
            items = self.table.selectedItems()
            self.viewer.set_selected_label(items[0].text() if items else None)

        def _toggle_play(self, checked: bool) -> None:
            self.play_button.setText("⏸" if checked else "▶")
            if checked:
                interval = max(1, int(1000.0 / max(self.data.rate, 1.0)))
                self.timer.start(interval)
            else:
                self.timer.stop()

        def _advance_frame(self) -> None:
            next_frame = (self.slider.value() + 1) % max(1, self.data.n_frames)
            self.slider.setValue(next_frame)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Open a lightweight C3D trial viewer.")
    parser.add_argument("c3d", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not PYSIDE_AVAILABLE:
        raise RuntimeError("PySide6 is required for the integrated C3D viewer.")
    app = QApplication.instance() or QApplication(sys.argv)
    window = C3DTrialViewerWindow(load_c3d_marker_data(args.c3d), args.c3d)
    window.resize(1100, 720)
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
