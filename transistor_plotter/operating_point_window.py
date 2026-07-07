from __future__ import annotations

from pathlib import Path

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from PyQt6.QtCore import QSize, Qt
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QStatusBar,
    QVBoxLayout,
    QWidget,
)

from .bias import parse_bias_value
from .models import SensorFiles
from .operating_point import OperatingPointFitError, fit_operating_point_model
from .plotting import AXES_BG, FIGURE_BG, GRID_COLOR, MUTED_TEXT, TEXT_COLOR
from .sqlite_cache import SQLiteCurveCache


class OperatingPointWindow(QMainWindow):
    def __init__(self, data_dir: Path, sensors: list[SensorFiles], curve_cache: SQLiteCurveCache) -> None:
        super().__init__()
        self.setWindowTitle("Operating Point DC Model")
        self.resize(1200, 760)
        self.data_dir = data_dir
        self.sensors = sensors
        self.filtered_sensors = list(sensors)
        self.curve_cache = curve_cache

        self.search_edit = QLineEdit()
        self.search_edit.setPlaceholderText("Search device ID or data group")
        self.search_edit.textChanged.connect(self.apply_filter)

        self.sensor_list = QListWidget()
        self.sensor_list.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.sensor_list.itemDoubleClicked.connect(self.plot_selected)

        self.count_label = QLabel()
        self.result_label = QLabel("Fit: not run")
        self.result_label.setObjectName("fitResultLabel")
        self.result_label.setWordWrap(True)

        self.plot_button = QPushButton("Plot Operating Point")
        self.plot_button.clicked.connect(self.plot_selected)

        self.figure = Figure(figsize=(10, 6), constrained_layout=False)
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.toolbar = NavigationToolbar(self.canvas, self)

        self.setStatusBar(QStatusBar())
        self._build_layout()
        self.apply_filter()
        self._draw_idle_state()

    def _build_layout(self) -> None:
        root = QWidget()
        main_layout = QHBoxLayout(root)
        main_layout.setContentsMargins(12, 12, 12, 12)
        main_layout.setSpacing(12)

        sidebar = QWidget()
        sidebar.setFixedWidth(330)
        sidebar_layout = QVBoxLayout(sidebar)
        sidebar_layout.setContentsMargins(0, 0, 0, 0)
        sidebar_layout.setSpacing(8)
        title = QLabel("Sensors")
        title.setObjectName("sectionTitle")
        sidebar_layout.addWidget(title)
        sidebar_layout.addWidget(self.search_edit)
        sidebar_layout.addWidget(self.count_label)
        sidebar_layout.addWidget(self.sensor_list, stretch=1)
        sidebar_layout.addWidget(self.result_label)
        sidebar_layout.addWidget(self.plot_button)

        plot_area = QWidget()
        plot_layout = QVBoxLayout(plot_area)
        plot_layout.setContentsMargins(10, 8, 10, 10)
        plot_layout.setSpacing(8)
        plot_layout.addWidget(self.toolbar)
        plot_layout.addWidget(self.canvas, stretch=1)

        main_layout.addWidget(sidebar)
        main_layout.addWidget(plot_area, stretch=1)
        self.setCentralWidget(root)

    def apply_filter(self) -> None:
        query = self.search_edit.text().strip().lower()
        self.filtered_sensors = [
            sensor
            for sensor in self.sensors
            if not query or query in sensor.device_id.lower() or query in sensor.group.lower()
        ]
        self.sensor_list.clear()
        for sensor in self.filtered_sensors:
            item = QListWidgetItem(sensor.label)
            item.setData(Qt.ItemDataRole.UserRole, sensor)
            item.setSizeHint(QSize(0, 26))
            self.sensor_list.addItem(item)
        if self.filtered_sensors:
            self.sensor_list.setCurrentRow(0)
        self.count_label.setText(f"{len(self.filtered_sensors):,} shown / {len(self.sensors):,} complete sensors")

    def plot_selected(self) -> None:
        item = self.sensor_list.currentItem()
        if item is None:
            self.statusBar().showMessage("Choose a sensor first.", 5000)
            return
        sensor = item.data(Qt.ItemDataRole.UserRole)
        try:
            device = self.curve_cache.load_device_curves(sensor)
            fit = fit_operating_point_model(device)
        except OperatingPointFitError as exc:
            QMessageBox.warning(self, "Could not fit operating point", str(exc))
            self.statusBar().showMessage(str(exc), 8000)
            return
        except Exception as exc:  # noqa: BLE001 - surface cache/data failures to the GUI.
            QMessageBox.critical(self, "Could not load sensor", str(exc))
            return

        self._plot_operating_point(sensor, device, fit)
        message = (
            f"Vt={fit.vth:.5g} V, k={fit.k:.5g} mA/mm/V^2, "
            f"Vov@0.1V={fit.vov:.5g} V, pts={fit.points_used}, RMS={fit.rms_error:.5g}"
        )
        self.result_label.setText(message)
        self.statusBar().showMessage(message, 12000)

    def _plot_operating_point(self, sensor, device, fit) -> None:
        self.figure.clear()
        self.figure.patch.set_facecolor(FIGURE_BG)
        axes = list(self.figure.subplots(1, 2).flat)
        for ax in axes:
            ax.set_facecolor(AXES_BG)
            ax.grid(True, alpha=0.22, color=GRID_COLOR)
            ax.tick_params(colors=MUTED_TEXT)
            for spine in ax.spines.values():
                spine.set_color("#4b5563")

        transfer_ax, output_ax = axes
        transfer_curve = _curve_with_label_value(device.trans_id_vgs.curves, "VDS", fit.transfer_vds)
        output_curve = _curve_with_label_value(device.iv_id_vds.curves, "VGS", fit.output_vgs)

        transfer_ax.plot(transfer_curve.x, transfer_curve.y, color="#ff7f0e", linewidth=1.8, label=transfer_curve.label)
        transfer_ax.scatter(fit.fit_vgs, fit.fit_id, s=16, color="#fbbf24", label="fit points", zorder=3)
        transfer_ax.plot(
            fit.fit_vgs,
            fit.fit_transfer_id,
            color="#ffffff",
            linestyle="--",
            linewidth=1.5,
            label="Heaviside saturation fit",
        )
        transfer_ax.set_title("Transfer fit at Vds=0.5 V", fontsize=10, fontweight="bold", color=TEXT_COLOR)
        transfer_ax.set_xlabel("Vgs (V)", color=MUTED_TEXT)
        transfer_ax.set_ylabel("Id (mA/mm)", color=MUTED_TEXT)

        output_ax.plot(output_curve.x, output_curve.y, color="#ff7f0e", linewidth=1.8, label=output_curve.label)
        if fit.output_vds.size > 0:
            output_ax.plot(
                fit.output_vds,
                fit.output_id,
                color="#ffffff",
                linestyle="--",
                linewidth=1.5,
                label="triode prediction to Vov",
            )
            output_ax.axvline(fit.vov, color="#cbd5e1", linestyle=":", linewidth=1.0, label="Vov")
        output_ax.set_title("Output prediction at Vgs=+0.1 V", fontsize=10, fontweight="bold", color=TEXT_COLOR)
        output_ax.set_xlabel("Vds (V)", color=MUTED_TEXT)
        output_ax.set_ylabel("Id (mA/mm)", color=MUTED_TEXT)

        for ax in axes:
            legend = ax.legend(fontsize=8, loc="best", facecolor="#0f172a", edgecolor="#475569", framealpha=0.92)
            for text in legend.get_texts():
                text.set_color(TEXT_COLOR)

        self.figure.suptitle(sensor.label, fontsize=13, fontweight="bold", color=TEXT_COLOR)
        self.figure.tight_layout(pad=1.35)
        self.canvas.draw_idle()

    def _draw_idle_state(self) -> None:
        self.figure.clear()
        self.figure.patch.set_facecolor(FIGURE_BG)
        ax = self.figure.add_subplot(111)
        ax.set_facecolor(AXES_BG)
        ax.text(
            0.5,
            0.5,
            "Choose a sensor and plot the operating-point model",
            ha="center",
            va="center",
            color=TEXT_COLOR,
            transform=ax.transAxes,
        )
        ax.set_axis_off()
        self.canvas.draw_idle()


def _curve_with_label_value(curves, bias_name: str, target: float):
    for curve in curves:
        value = parse_bias_value(curve.label, bias_name)
        if value is not None and abs(value - target) <= 1e-6:
            return curve
    return curves[0]
