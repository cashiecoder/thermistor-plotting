from __future__ import annotations

from collections.abc import Iterable

import numpy as np
from matplotlib.axes import Axes
from matplotlib.collections import LineCollection
from matplotlib.figure import Figure

from .models import DeviceCurves

FIGURE_BG = "#111827"
AXES_BG = "#172033"
TEXT_COLOR = "#e5e7eb"
MUTED_TEXT = "#cbd5e1"
GRID_COLOR = "#6b7280"
CURVE_COLORS = (
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#17becf",
)
PANEL_SPECS = (
    ("Gate leakage", "Vgs (V)", "Ig (uA/mm)"),
    ("Output characteristic", "Vds (V)", "Id (mA/mm)"),
    ("Transfer characteristic", "Vgs (V)", "Id (mA/mm)"),
    ("Transconductance", "Vgs (V)", "gm (mS/mm)"),
)


def configure_figure(figure: Figure) -> list[Axes]:
    figure.clear()
    figure.patch.set_facecolor(FIGURE_BG)
    axes = list(figure.subplots(2, 2).flat)
    for ax in axes:
        ax.set_facecolor(AXES_BG)
        ax.grid(True, alpha=0.22, color=GRID_COLOR)
        ax.tick_params(colors=MUTED_TEXT)
        for spine in ax.spines.values():
            spine.set_color("#4b5563")
    return axes


def plot_single_device(figure: Figure, device: DeviceCurves) -> None:
    axes = configure_figure(figure)
    for ax, panel in zip(axes, device.panels, strict=True):
        for curve_index, curve in enumerate(panel.curves):
            ax.plot(curve.x, curve.y, linewidth=1.8, label=curve.label, color=_curve_color(curve_index))
        _finish_axis(ax, panel.title, panel.xlabel, panel.ylabel, show_legend=True)
    figure.suptitle(device.sensor.label, fontsize=13, fontweight="bold", color=TEXT_COLOR)
    figure.tight_layout()


def plot_all_devices(figure: Figure, devices: Iterable[DeviceCurves]) -> int:
    axes = prepare_overlay_figure(figure, "All sensors overlay")
    count = add_overlay_devices(axes, devices)
    finish_overlay_figure(figure, "All sensors overlay", count)
    return count


def prepare_overlay_figure(figure: Figure, title: str) -> list[Axes]:
    axes = configure_figure(figure)
    for ax, (panel_title, xlabel, ylabel) in zip(axes, PANEL_SPECS, strict=True):
        _finish_axis(ax, panel_title, xlabel, ylabel, show_legend=False)
    figure.suptitle(f"{title} (loading...)", fontsize=13, fontweight="bold", color=TEXT_COLOR)
    figure.tight_layout()
    return axes


def add_overlay_devices(axes: list[Axes], devices: Iterable[DeviceCurves]) -> int:
    count = 0
    segments_by_panel_curve: dict[tuple[int, int], list[np.ndarray]] = {}
    for device in devices:
        count += 1
        for panel_index, panel in enumerate(device.panels):
            for curve_index, curve in enumerate(panel.curves):
                if len(curve.x) == 0 or len(curve.y) == 0:
                    continue
                segments_by_panel_curve.setdefault((panel_index, curve_index), []).append(
                    np.column_stack((curve.x, curve.y))
                )

    for (panel_index, curve_index), segments in segments_by_panel_curve.items():
        collection = LineCollection(
            segments,
            colors=[_curve_color(curve_index)],
            linewidths=0.55,
            alpha=0.16,
        )
        axes[panel_index].add_collection(collection, autolim=True)

    for ax in axes:
        ax.autoscale_view()
    return count


def finish_overlay_figure(figure: Figure, title: str, count: int) -> None:
    figure.suptitle(f"{title} ({count} devices)", fontsize=13, fontweight="bold", color=TEXT_COLOR)
    figure.tight_layout()


def _curve_color(curve_index: int) -> str:
    return CURVE_COLORS[curve_index % len(CURVE_COLORS)]


def _finish_axis(ax: Axes, title: str, xlabel: str, ylabel: str, show_legend: bool) -> None:
    ax.set_title(title, fontsize=10, fontweight="bold", color=TEXT_COLOR)
    ax.set_xlabel(xlabel, color=MUTED_TEXT)
    ax.set_ylabel(ylabel, color=MUTED_TEXT)
    if show_legend:
        legend = ax.legend(fontsize=8, loc="best", facecolor="#0f172a", edgecolor="#475569", framealpha=0.92)
        for text in legend.get_texts():
            text.set_color(TEXT_COLOR)
