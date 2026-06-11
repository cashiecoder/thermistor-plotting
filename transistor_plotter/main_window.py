from __future__ import annotations

import os
import sys
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from PyQt6.QtCore import QObject, QThread, Qt, pyqtSignal
from PyQt6.QtGui import QColor, QPalette
from PyQt6.QtWidgets import (
    QApplication,
    QAbstractItemView,
    QCheckBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QProgressBar,
    QSizePolicy,
    QStatusBar,
    QVBoxLayout,
    QWidget,
)

from .data_loader import discover_sensors, load_device_curves
from .models import DeviceCurves, SensorFiles
from .plotting import (
    TEXT_COLOR,
    add_overlay_devices,
    configure_figure,
    finish_overlay_figure,
    plot_single_device,
    prepare_overlay_figure,
)

# Uses roughly os.cpu_count() / CPU_CORE_DIVISOR loader threads.
# Set to 4 for about 1/4 of cores or 8 for about 1/8 of cores.
CPU_CORE_DIVISOR = 4
BULK_BATCH_SIZE = 50


class LoadAllWorker(QObject):
    progress = pyqtSignal(int, int)
    batch_ready = pyqtSignal(list)
    finished = pyqtSignal(int, int, bool)
    failed = pyqtSignal(str)

    def __init__(
        self,
        sensors: list[SensorFiles],
        batch_size: int = BULK_BATCH_SIZE,
        cpu_divisor: int = CPU_CORE_DIVISOR,
    ) -> None:
        super().__init__()
        self._sensors = sensors
        self._batch_size = batch_size
        cpu_count = os.cpu_count() or 1
        self._max_workers = max(1, cpu_count // cpu_divisor)
        self.worker_count = self._max_workers
        self._cancel_requested = False

    def cancel(self) -> None:
        self._cancel_requested = True

    def run(self) -> None:
        batch: list[DeviceCurves] = []
        loaded = 0
        total = len(self._sensors)
        next_sensor_index = 0
        pending = {}
        executor = ThreadPoolExecutor(max_workers=self._max_workers)

        def submit_until_full() -> None:
            nonlocal next_sensor_index
            target_pending = self._max_workers * 2
            while (
                not self._cancel_requested
                and next_sensor_index < total
                and len(pending) < target_pending
            ):
                sensor = self._sensors[next_sensor_index]
                next_sensor_index += 1
                pending[executor.submit(load_device_curves, sensor)] = sensor

        try:
            submit_until_full()
            while pending:
                if self._cancel_requested:
                    break
                done, _ = wait(pending, timeout=0.1, return_when=FIRST_COMPLETED)
                if not done:
                    continue
                for future in done:
                    pending.pop(future, None)
                    batch.append(future.result())
                    loaded += 1
                    if len(batch) >= self._batch_size:
                        self.batch_ready.emit(batch)
                        batch = []
                    if loaded == 1 or loaded == total or loaded % 25 == 0:
                        self.progress.emit(loaded, total)
                submit_until_full()
        except Exception as exc:  # noqa: BLE001 - show the GUI user the actual load failure.
            for future in pending:
                future.cancel()
            executor.shutdown(wait=False, cancel_futures=True)
            self.failed.emit(str(exc))
            return
        for future in pending:
            future.cancel()
        executor.shutdown(wait=not self._cancel_requested, cancel_futures=True)
        if batch:
            self.batch_ready.emit(batch)
        self.finished.emit(loaded, total, self._cancel_requested)


class MainWindow(QMainWindow):
    def __init__(self, data_dir: Path) -> None:
        super().__init__()
        self.setWindowTitle("Transistor Curve Plotter")
        self.resize(1300, 840)

        self.data_dir = data_dir
        self.sensors = discover_sensors(data_dir)
        self.filtered_sensors = list(self.sensors)
        self._worker_thread: QThread | None = None
        self._worker: LoadAllWorker | None = None
        self._bulk_axes = []
        self._bulk_plotted = 0
        self._bulk_title = ""
        self._active_bulk_button: QPushButton | None = None
        self._active_bulk_default_text = ""
        self._focused_axis_index: int | None = None
        self._axis_positions = []

        self.search_edit = QLineEdit()
        self.search_edit.setPlaceholderText("Search device ID or data group")
        self.search_edit.setToolTip("Filter the sensor list by device ID or data group.")
        self.search_edit.textChanged.connect(self.apply_filter)

        self.sensor_list = QListWidget()
        self.sensor_list.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.sensor_list.itemDoubleClicked.connect(self.plot_selected)

        self.count_label = QLabel()

        self.reference_checkbox = QCheckBox("Ideal MOSFET reference")
        self.reference_checkbox.setToolTip("Overlay dashed textbook ideal MOSFET curves using explicit Vth and k.")
        self.reference_checkbox.toggled.connect(self._set_reference_inputs_enabled)

        self.reference_vth_edit = QLineEdit()
        self.reference_vth_edit.setPlaceholderText("Vth (V)")
        self.reference_vth_edit.setToolTip("Threshold voltage in volts. Required only when ideal reference is enabled.")

        self.reference_k_edit = QLineEdit()
        self.reference_k_edit.setPlaceholderText("k (mA/mm/V^2)")
        self.reference_k_edit.setToolTip("Positive finite k in mA/mm/V^2. Required only when ideal reference is enabled.")

        self.plot_selected_button = QPushButton("Plot Selected")
        self.plot_selected_button.setToolTip("Plot the currently selected sensor as a four-panel figure.")
        self.plot_selected_button.clicked.connect(self.plot_selected)

        self.plot_all_button = QPushButton("Plot All Sensors")
        self.plot_all_button.setToolTip("Overlay every complete sensor on the four plot panels.")
        self.plot_all_button.clicked.connect(self.plot_all_sensors)

        self.plot_filtered_button = QPushButton("Plot Filtered")
        self.plot_filtered_button.setToolTip("Overlay only the sensors currently visible after the search filter.")
        self.plot_filtered_button.clicked.connect(self.plot_filtered_sensors)

        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)

        self.figure = Figure(figsize=(10, 7), constrained_layout=False)
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.toolbar = NavigationToolbar(self.canvas, self)
        self.canvas.mpl_connect("button_press_event", self._on_canvas_click)

        self.setStatusBar(QStatusBar())
        self._build_layout()
        self._set_reference_inputs_enabled(False)
        self.apply_filter()
        self._draw_empty_state()

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
        sidebar_layout.addWidget(self.reference_checkbox)
        sidebar_layout.addWidget(self.reference_vth_edit)
        sidebar_layout.addWidget(self.reference_k_edit)
        sidebar_layout.addWidget(self.plot_selected_button)
        sidebar_layout.addWidget(self.plot_filtered_button)
        sidebar_layout.addWidget(self.plot_all_button)
        sidebar_layout.addWidget(self.progress_bar)

        plot_area = QWidget()
        plot_layout = QVBoxLayout(plot_area)
        plot_layout.setContentsMargins(0, 0, 0, 0)
        plot_layout.setSpacing(6)
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
            self.sensor_list.addItem(item)

        if self.filtered_sensors:
            self.sensor_list.setCurrentRow(0)
        self.count_label.setText(f"{len(self.filtered_sensors):,} shown / {len(self.sensors):,} complete sensors")

    def plot_selected(self) -> None:
        if self._worker_thread is not None:
            self.statusBar().showMessage("Cancel the background overlay before plotting a selected sensor.", 5000)
            return
        item = self.sensor_list.currentItem()
        if item is None:
            self.statusBar().showMessage("Choose a sensor first.", 4000)
            return

        sensor = item.data(Qt.ItemDataRole.UserRole)
        try:
            device = load_device_curves(sensor)
        except Exception as exc:  # noqa: BLE001 - present spreadsheet read errors in the GUI.
            QMessageBox.critical(self, "Could not load sensor", str(exc))
            return

        try:
            show_reference, reference_vth, reference_k = self._reference_options()
            plot_single_device(
                self.figure,
                device,
                show_reference=show_reference,
                reference_vth=reference_vth,
                reference_k=reference_k,
            )
        except ValueError as exc:
            QMessageBox.warning(self, "Ideal reference parameters", str(exc))
            self.statusBar().showMessage(str(exc), 8000)
            return

        self._capture_axis_positions()
        self.canvas.draw_idle()
        self.statusBar().showMessage(f"Plotted {sensor.label}", 5000)

    def plot_filtered_sensors(self) -> None:
        if self._handle_active_bulk_button(self.plot_filtered_button):
            return
        self._plot_many(
            list(self.filtered_sensors),
            label="filtered sensors",
            title="Filtered sensors overlay",
            cancel_button=self.plot_filtered_button,
        )

    def plot_all_sensors(self) -> None:
        if self._handle_active_bulk_button(self.plot_all_button):
            return
        self._plot_many(
            list(self.sensors),
            label="all sensors",
            title="All sensors overlay",
            cancel_button=self.plot_all_button,
        )

    def _plot_many(self, sensors: list[SensorFiles], label: str, title: str, cancel_button: QPushButton) -> None:
        if not sensors:
            self.statusBar().showMessage("No sensors to plot.", 4000)
            return
        if self._worker_thread is not None:
            self.statusBar().showMessage("A plot job is already running.", 4000)
            return

        self._set_bulk_busy(True, cancel_button)
        self._focused_axis_index = None
        self._bulk_title = title
        self._bulk_plotted = 0
        self._bulk_axes = prepare_overlay_figure(self.figure, title)
        self._capture_axis_positions()
        self.canvas.draw_idle()
        self.progress_bar.setMaximum(len(sensors))
        self.progress_bar.setValue(0)
        worker_count = max(1, (os.cpu_count() or 1) // CPU_CORE_DIVISOR)
        self.statusBar().showMessage(f"Loading {len(sensors):,} {label} with {worker_count} worker threads...")

        self._worker_thread = QThread(self)
        self._worker = LoadAllWorker(sensors)
        self._worker.moveToThread(self._worker_thread)
        self._worker_thread.started.connect(self._worker.run)
        self._worker.progress.connect(self._on_worker_progress)
        self._worker.batch_ready.connect(self._on_worker_batch_ready)
        self._worker.finished.connect(self._on_worker_finished)
        self._worker.failed.connect(self._on_worker_failed)
        self._worker.finished.connect(self._worker_thread.quit)
        self._worker.failed.connect(self._worker_thread.quit)
        self._worker_thread.finished.connect(self._cleanup_worker)
        self._worker_thread.start()

    def _handle_active_bulk_button(self, button: QPushButton) -> bool:
        if self._worker_thread is None:
            return False
        if self._active_bulk_button is button:
            self._cancel_bulk_plot()
        else:
            self.statusBar().showMessage("Cancel the current background overlay before starting another.", 5000)
        return True

    def _cancel_bulk_plot(self) -> None:
        if self._worker is None:
            return
        self._worker.cancel()
        if self._active_bulk_button is not None:
            self._active_bulk_button.setText("Cancelling...")
            self._active_bulk_button.setEnabled(False)
        self.statusBar().showMessage("Cancelling background plot...")

    def _on_worker_progress(self, done: int, total: int) -> None:
        self.progress_bar.setMaximum(total)
        self.progress_bar.setValue(done)
        self.statusBar().showMessage(f"Loaded {done:,} of {total:,} sensors...")

    def _on_worker_batch_ready(self, devices: list[DeviceCurves]) -> None:
        if not self._bulk_axes:
            return
        self._bulk_plotted += add_overlay_devices(self._bulk_axes, devices)
        self.figure.suptitle(
            f"{self._bulk_title} ({self._bulk_plotted:,} devices loaded...)",
            fontsize=13,
            fontweight="bold",
            color=TEXT_COLOR,
        )
        self.canvas.draw_idle()

    def _on_worker_finished(self, loaded: int, total: int, cancelled: bool) -> None:
        if self._bulk_axes:
            if self._focused_axis_index is None:
                finish_overlay_figure(self.figure, self._bulk_title, self._bulk_plotted)
                self._capture_axis_positions()
            else:
                self.figure.suptitle(
                    f"{self._bulk_title} ({self._bulk_plotted:,} devices)",
                    fontsize=13,
                    fontweight="bold",
                    color=TEXT_COLOR,
                )
            self.canvas.draw_idle()
        if cancelled:
            self.statusBar().showMessage(f"Cancelled after loading {loaded:,} of {total:,} sensors.", 8000)
        else:
            self.statusBar().showMessage(f"Plotted {self._bulk_plotted:,} sensors.", 8000)
        self._set_bulk_busy(False)

    def _on_worker_failed(self, message: str) -> None:
        self._set_bulk_busy(False)
        QMessageBox.critical(self, "Could not plot sensors", message)

    def _cleanup_worker(self) -> None:
        if self._worker is not None:
            self._worker.deleteLater()
        if self._worker_thread is not None:
            self._worker_thread.deleteLater()
        self._worker = None
        self._worker_thread = None

    def _set_bulk_busy(self, busy: bool, cancel_button: QPushButton | None = None) -> None:
        self.progress_bar.setVisible(busy)
        self.plot_selected_button.setEnabled(not busy)
        if busy:
            self._active_bulk_button = cancel_button
            self._active_bulk_default_text = cancel_button.text() if cancel_button is not None else ""
            self.plot_filtered_button.setEnabled(cancel_button is self.plot_filtered_button)
            self.plot_all_button.setEnabled(cancel_button is self.plot_all_button)
            if cancel_button is not None:
                cancel_button.setText("Cancel")
        else:
            if self._active_bulk_button is not None:
                self._active_bulk_button.setText(self._active_bulk_default_text)
            self.plot_filtered_button.setEnabled(True)
            self.plot_all_button.setEnabled(True)
            self._active_bulk_button = None
            self._active_bulk_default_text = ""

    def _set_reference_inputs_enabled(self, enabled: bool) -> None:
        self.reference_vth_edit.setEnabled(enabled)
        self.reference_k_edit.setEnabled(enabled)

    def _reference_options(self) -> tuple[bool, float | None, float | None]:
        if not self.reference_checkbox.isChecked():
            return False, None, None
        vth_text = self.reference_vth_edit.text().strip()
        k_text = self.reference_k_edit.text().strip()
        if not vth_text or not k_text:
            raise ValueError("Enter explicit Vth and k values before enabling the ideal MOSFET reference.")
        try:
            return True, float(vth_text), float(k_text)
        except ValueError as exc:
            raise ValueError("Vth and k must be numeric values.") from exc

    def _draw_empty_state(self) -> None:
        axes = configure_figure(self.figure)
        for ax in axes:
            ax.set_xticks([])
            ax.set_yticks([])
        axes[0].text(
            0.5,
            0.5,
            "Select a sensor and plot",
            ha="center",
            va="center",
            transform=axes[0].transAxes,
            fontsize=12,
            color=TEXT_COLOR,
        )
        self.figure.tight_layout()
        self._capture_axis_positions()
        self.canvas.draw_idle()

    def _capture_axis_positions(self) -> None:
        if self._focused_axis_index is not None:
            return
        self._axis_positions = [axis.get_position().frozen() for axis in self.figure.axes]

    def _on_canvas_click(self, event) -> None:  # noqa: ANN001 - matplotlib event object.
        if getattr(self.toolbar, "mode", ""):
            return
        if not self.figure.axes:
            return
        if self._focused_axis_index is None:
            if event.inaxes is None or event.inaxes not in self.figure.axes:
                return
            self._focus_axis(self.figure.axes.index(event.inaxes))
            return
        if event.inaxes is None or self.figure.axes[self._focused_axis_index] is event.inaxes:
            self._restore_axes()

    def _focus_axis(self, axis_index: int) -> None:
        self._capture_axis_positions()
        self._focused_axis_index = axis_index
        for index, axis in enumerate(self.figure.axes):
            axis.set_visible(index == axis_index)
        self.figure.axes[axis_index].set_position([0.08, 0.1, 0.88, 0.78])
        self.canvas.draw_idle()
        self.statusBar().showMessage("Focused one panel. Click it again or click outside the plot to show all four.", 6000)

    def _restore_axes(self) -> None:
        for index, axis in enumerate(self.figure.axes):
            axis.set_visible(True)
            if index < len(self._axis_positions):
                axis.set_position(self._axis_positions[index])
        self._focused_axis_index = None
        self.canvas.draw_idle()
        self.statusBar().showMessage("Showing all four panels.", 4000)


def apply_dark_theme(app: QApplication) -> None:
    app.setStyle("Fusion")
    palette = QPalette()
    palette.setColor(QPalette.ColorRole.Window, QColor("#111827"))
    palette.setColor(QPalette.ColorRole.WindowText, QColor("#e5e7eb"))
    palette.setColor(QPalette.ColorRole.Base, QColor("#172033"))
    palette.setColor(QPalette.ColorRole.AlternateBase, QColor("#1f2937"))
    palette.setColor(QPalette.ColorRole.ToolTipBase, QColor("#0f172a"))
    palette.setColor(QPalette.ColorRole.ToolTipText, QColor("#e5e7eb"))
    palette.setColor(QPalette.ColorRole.Text, QColor("#e5e7eb"))
    palette.setColor(QPalette.ColorRole.Button, QColor("#243047"))
    palette.setColor(QPalette.ColorRole.ButtonText, QColor("#f8fafc"))
    palette.setColor(QPalette.ColorRole.Highlight, QColor("#2f81f7"))
    palette.setColor(QPalette.ColorRole.HighlightedText, QColor("#ffffff"))
    palette.setColor(QPalette.ColorRole.PlaceholderText, QColor("#94a3b8"))
    app.setPalette(palette)
    app.setStyleSheet(
        """
        QWidget {
            background: #111827;
            color: #e5e7eb;
            selection-background-color: #2f81f7;
            selection-color: #ffffff;
        }
        QMainWindow, QStatusBar {
            background: #111827;
            color: #cbd5e1;
        }
        QLabel#sectionTitle {
            color: #f8fafc;
            font-size: 18px;
            font-weight: 700;
        }
        QListWidget {
            background: #172033;
            color: #e5e7eb;
            border: 1px solid #334155;
            outline: 0;
        }
        QListWidget::item {
            padding: 6px;
        }
        QListWidget::item:selected {
            background: #2f81f7;
            color: #ffffff;
        }
        QListWidget::item:hover {
            background: #243047;
        }
        QLineEdit {
            padding: 7px;
            color: #f8fafc;
            background: #172033;
            border: 1px solid #334155;
            border-radius: 4px;
        }
        QLineEdit:focus {
            border-color: #2f81f7;
        }
        QPushButton {
            padding: 8px;
            color: #f8fafc;
            background: #243047;
            border: 1px solid #3b4a63;
            border-radius: 4px;
            font-weight: 600;
        }
        QPushButton:hover {
            background: #2f3d57;
            border-color: #4b5f80;
        }
        QPushButton:pressed {
            background: #1d4ed8;
        }
        QPushButton:disabled {
            color: #64748b;
            background: #1f2937;
            border-color: #334155;
        }
        QProgressBar {
            color: #f8fafc;
            background: #172033;
            border: 1px solid #334155;
            border-radius: 4px;
            text-align: center;
        }
        QProgressBar::chunk {
            background: #2f81f7;
            border-radius: 3px;
        }
        QToolBar {
            background: #111827;
            border: 0;
            spacing: 4px;
        }
        QToolButton {
            color: #e5e7eb;
            background: #243047;
            border: 1px solid #3b4a63;
            border-radius: 4px;
            padding: 4px;
        }
        QToolButton:hover {
            background: #2f3d57;
        }
        QMessageBox {
            background: #111827;
        }
        """
    )


def main() -> int:
    project_root = Path(__file__).resolve().parents[1]
    data_dir = project_root / "TransistorData"

    app = QApplication(sys.argv)
    apply_dark_theme(app)

    window = MainWindow(data_dir)
    window.show()
    return app.exec()
