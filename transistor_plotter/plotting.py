from __future__ import annotations

import logging
from collections.abc import Iterable

import numpy as np
from matplotlib.axes import Axes
from matplotlib.collections import LineCollection
from matplotlib.figure import Figure

from .bias import parse_bias_value
from .histogram_data import (
    HISTOGRAM_DIODE_VDS,
    HISTOGRAM_TRANSFER_VDS,
    HISTOGRAM_VDS,
    HISTOGRAM_VGS,
)
from .ideal_mosfet import ideal_gate_leakage, ideal_gm, ideal_id, validate_reference_params
from .models import DeviceCurves

FIGURE_BG = "#111827"
AXES_BG = "#172033"
TEXT_COLOR = "#e5e7eb"
MUTED_TEXT = "#cbd5e1"
GRID_COLOR = "#6b7280"
REFERENCE_LABEL = "ideal MOSFET reference"
REFERENCE_NEUTRAL_COLOR = "#cbd5e1"
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


def plot_single_device(
    figure: Figure,
    device: DeviceCurves,
    *,
    show_reference: bool = False,
    reference_vth: float | None = None,
    reference_k: float | None = None,
) -> None:
    if show_reference:
        if reference_vth is None:
            raise ValueError("reference_vth must be provided when ideal reference overlay is enabled.")
        if reference_k is None:
            raise ValueError("reference_k must be provided when ideal reference overlay is enabled.")
        validate_reference_params(reference_vth, reference_k)

    axes = configure_figure(figure)
    for ax, panel in zip(axes, device.panels, strict=True):
        for curve_index, curve in enumerate(panel.curves):
            ax.plot(curve.x, curve.y, linewidth=1.8, label=curve.label, color=_curve_color(curve_index))

    if show_reference:
        _add_reference_curves(axes, device, reference_vth, reference_k)

    for ax, panel in zip(axes, device.panels, strict=True):
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


def plot_histograms(
    figure: Figure,
    gate_ig: list[float],
    transfer_id: list[float],
    output_id: list[float],
    *,
    loaded_count: int,
) -> None:
    figure.clear()
    figure.patch.set_facecolor(FIGURE_BG)
    axes = list(figure.subplots(1, 3))
    specs = [
        (
            gate_ig,
            f"Ig @ Vgs={HISTOGRAM_VGS:+.1f} V, Vds={HISTOGRAM_DIODE_VDS:+.1f} V",
            "Ig (uA/mm)",
            "#ff7f0e",
        ),
        (
            transfer_id,
            f"Id transfer @ Vgs={HISTOGRAM_VGS:+.1f} V, Vds={HISTOGRAM_TRANSFER_VDS:+.1f} V",
            "Id (mA/mm)",
            "#2ca02c",
        ),
        (
            output_id,
            f"Id output @ Vds={HISTOGRAM_VDS:+.1f} V, Vgs={HISTOGRAM_VGS:+.1f} V",
            "Id (mA/mm)",
            "#1f77b4",
        ),
    ]
    for ax, (values, title, xlabel, color) in zip(axes, specs, strict=True):
        ax.set_facecolor(AXES_BG)
        ax.grid(True, alpha=0.22, color=GRID_COLOR)
        ax.tick_params(colors=MUTED_TEXT)
        for spine in ax.spines.values():
            spine.set_color("#4b5563")
        finite_values = np.asarray(values, dtype=float)
        finite_values = finite_values[np.isfinite(finite_values)]
        if len(finite_values) > 0:
            ax.hist(finite_values, bins="auto", color=color, alpha=0.82, edgecolor="#0f172a", linewidth=0.7)
        ax.set_title(f"{title}\nn={len(finite_values):,}", fontsize=10, fontweight="bold", color=TEXT_COLOR)
        ax.set_xlabel(xlabel, color=MUTED_TEXT)
        ax.set_ylabel("Sensor count", color=MUTED_TEXT)

    figure.suptitle(f"Histogram distributions ({loaded_count:,} sensors loaded)", fontsize=13, fontweight="bold", color=TEXT_COLOR)
    figure.tight_layout()


def _curve_color(curve_index: int) -> str:
    return CURVE_COLORS[curve_index % len(CURVE_COLORS)]


def _add_reference_curves(axes: list[Axes], device: DeviceCurves, vth: float, k: float) -> None:
    _add_gate_leakage_reference(axes[0], device)
    _add_output_references(axes[1], device, vth, k)
    _add_transfer_references(axes[2], device, vth, k)
    _add_gm_references(axes[3], device, vth, k)


def _add_gate_leakage_reference(ax: Axes, device: DeviceCurves) -> None:
    if not device.diode_ig_vgs.curves:
        return
    x = device.diode_ig_vgs.curves[0].x
    ax.plot(
        x,
        ideal_gate_leakage(x),
        linestyle="--",
        linewidth=1.4,
        color=REFERENCE_NEUTRAL_COLOR,
        label=REFERENCE_LABEL,
    )


def _add_output_references(ax: Axes, device: DeviceCurves, vth: float, k: float) -> None:
    label_used = False
    for curve_index, curve in enumerate(device.iv_id_vds.curves):
        vgs = parse_bias_value(curve.label, "VGS")
        if vgs is None:
            logging.warning("Skipping ideal output reference; could not parse VGS from label %r.", curve.label)
            continue
        label = REFERENCE_LABEL if not label_used else "_nolegend_"
        label_used = True
        ax.plot(
            curve.x,
            ideal_id(vgs, curve.x, vth=vth, k=k),
            linestyle="--",
            linewidth=1.25,
            color=_curve_color(curve_index),
            label=label,
        )


def _add_transfer_references(ax: Axes, device: DeviceCurves, vth: float, k: float) -> None:
    label_used = False
    for curve_index, curve in enumerate(device.trans_id_vgs.curves):
        vds = parse_bias_value(curve.label, "VDS")
        if vds is None:
            logging.warning("Skipping ideal transfer reference; could not parse VDS from label %r.", curve.label)
            continue
        label = REFERENCE_LABEL if not label_used else "_nolegend_"
        label_used = True
        ax.plot(
            curve.x,
            ideal_id(curve.x, vds, vth=vth, k=k),
            linestyle="--",
            linewidth=1.25,
            color=_curve_color(curve_index),
            label=label,
        )


def _add_gm_references(ax: Axes, device: DeviceCurves, vth: float, k: float) -> None:
    label_used = False
    for curve_index, curve in enumerate(device.trans_gm_vgs.curves):
        vds = parse_bias_value(curve.label, "VDS")
        if vds is None:
            logging.warning("Skipping ideal gm reference; could not parse VDS from label %r.", curve.label)
            continue
        label = REFERENCE_LABEL if not label_used else "_nolegend_"
        label_used = True
        ax.plot(
            curve.x,
            ideal_gm(curve.x, vds, vth=vth, k=k),
            linestyle="--",
            linewidth=1.25,
            color=_curve_color(curve_index),
            label=label,
        )


def _finish_axis(ax: Axes, title: str, xlabel: str, ylabel: str, show_legend: bool) -> None:
    ax.set_title(title, fontsize=10, fontweight="bold", color=TEXT_COLOR)
    ax.set_xlabel(xlabel, color=MUTED_TEXT)
    ax.set_ylabel(ylabel, color=MUTED_TEXT)
    if show_legend:
        legend = ax.legend(fontsize=8, loc="best", facecolor="#0f172a", edgecolor="#475569", framealpha=0.92)
        for text in legend.get_texts():
            text.set_color(TEXT_COLOR)
