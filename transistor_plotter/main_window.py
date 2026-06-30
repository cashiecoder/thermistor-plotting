from __future__ import annotations

import os
import sys
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from collections.abc import Callable
from functools import partial
from pathlib import Path

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from PyQt6.QtCore import QPoint, QRect, QObject, QThread, Qt, pyqtSignal
from PyQt6.QtGui import QColor, QPalette
from PyQt6.QtWidgets import (
    QApplication,
    QAbstractItemView,
    QCheckBox,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QProgressBar,
    QRubberBand,
    QSizePolicy,
    QStatusBar,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from .data_loader import discover_sensors
from .histogram_data import HistogramSample, extract_histogram_sample
from .models import DeviceCurves, SensorFiles
from .mosfet_fitting import FitError, fit_triode_eq_5_16
from .plotting import (
    TEXT_COLOR,
    add_overlay_devices,
    configure_figure,
    finish_overlay_figure,
    plot_histograms,
    plot_single_device,
    prepare_overlay_figure,
)
from .sqlite_cache import CacheMemoryError, SQLiteCurveCache

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
        loader: Callable[[SensorFiles], DeviceCurves],
        batch_size: int = BULK_BATCH_SIZE,
        cpu_divisor: int = CPU_CORE_DIVISOR,
    ) -> None:
        super().__init__()
        self._sensors = sensors
        self._loader = loader
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
                pending[executor.submit(self._loader, sensor)] = sensor

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
        executor.shutdown(wait=True, cancel_futures=True)
        if batch:
            self.batch_ready.emit(batch)
        self.finished.emit(loaded, total, self._cancel_requested)


class HistogramWorker(QObject):
    progress = pyqtSignal(int, int)
    batch_ready = pyqtSignal(list)
    finished = pyqtSignal(int, int, bool)
    failed = pyqtSignal(str)

    def __init__(
        self,
        sensors: list[SensorFiles],
        loader: Callable[[SensorFiles], DeviceCurves],
        batch_size: int = BULK_BATCH_SIZE,
        cpu_divisor: int = CPU_CORE_DIVISOR,
    ) -> None:
        super().__init__()
        self._sensors = sensors
        self._loader = loader
        self._batch_size = batch_size
        cpu_count = os.cpu_count() or 1
        self._max_workers = max(1, cpu_count // cpu_divisor)
        self._cancel_requested = False

    def cancel(self) -> None:
        self._cancel_requested = True

    def run(self) -> None:
        batch: list[HistogramSample] = []
        loaded = 0
        total = len(self._sensors)
        next_sensor_index = 0
        pending = {}
        executor = ThreadPoolExecutor(max_workers=self._max_workers)

        def load_sample(sensor: SensorFiles) -> HistogramSample:
            return extract_histogram_sample(self._loader(sensor))

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
                pending[executor.submit(load_sample, sensor)] = sensor

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
        except Exception as exc:  # noqa: BLE001 - surface data extraction failures in the GUI.
            for future in pending:
                future.cancel()
            executor.shutdown(wait=False, cancel_futures=True)
            self.failed.emit(str(exc))
            return
        for future in pending:
            future.cancel()
        executor.shutdown(wait=True, cancel_futures=True)
        if batch:
            self.batch_ready.emit(batch)
        self.finished.emit(loaded, total, self._cancel_requested)


class MainWindow(QMainWindow):
    def __init__(self, data_dir: Path, sensors: list[SensorFiles], curve_cache: SQLiteCurveCache) -> None:
        super().__init__()
        self.setWindowTitle("Transistor Curve Plotter")
        self.resize(1300, 840)

        self.data_dir = data_dir
        self.curve_cache = curve_cache
        self.sensors = sensors
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
        self._hist_thread: QThread | None = None
        self._hist_worker: HistogramWorker | None = None
        self._hist_started = False
        self._hist_loaded = 0
        self._hist_gate_ig: list[float] = []
        self._hist_transfer_id: list[float] = []
        self._hist_output_id: list[float] = []
        self._curve_home_limits: list[tuple[tuple[float, float], tuple[float, float]]] = []
        self._hist_home_limits: list[tuple[tuple[float, float], tuple[float, float]]] = []
        self._hover_canvas: FigureCanvas | None = None
        self._hover_axis_index: int | None = None
        self._interaction_canvas: FigureCanvas | None = None
        self._interaction_axis_index: int | None = None
        self._interaction_mode: str | None = None
        self._drag_start: tuple[float, float] | None = None
        self._drag_start_limits: tuple[tuple[float, float], tuple[float, float]] | None = None
        self._rubber_band: QRubberBand | None = None
        self._rubber_band_origin: QPoint | None = None
        self._plot_control_buttons: list[QPushButton] = []

        self.search_edit = QLineEdit()
        self.search_edit.setPlaceholderText("Search device ID or data group")
        self.search_edit.setToolTip("Filter the sensor list by device ID or data group.")
        self.search_edit.textChanged.connect(self.apply_filter)

        self.sensor_list = QListWidget()
        self.sensor_list.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.sensor_list.itemDoubleClicked.connect(self.plot_selected)

        self.count_label = QLabel()

        self.reference_checkbox = QCheckBox("Fit ideal MOSFET reference")
        self.reference_checkbox.setToolTip(
            "When checked, replot the selected sensor with fitted dashed ideal curves. "
            "Bulk all/filtered overlays do not use fitted references."
        )
        self.reference_checkbox.toggled.connect(self._on_reference_toggled)

        self.plot_selected_button = QPushButton("Plot Selected")
        self.plot_selected_button.setToolTip("Plot the currently selected sensor as a four-panel figure.")
        self.plot_selected_button.clicked.connect(self.plot_selected)

        self.plot_all_button = QPushButton("Plot All Sensors")
        self.plot_all_button.setToolTip("Overlay every complete sensor on the four plot panels.")
        self.plot_all_button.clicked.connect(self.plot_all_sensors)

        self.plot_filtered_button = QPushButton("Plot Filtered")
        self.plot_filtered_button.setToolTip("Overlay only the sensors currently visible after the search filter.")
        self.plot_filtered_button.clicked.connect(self.plot_filtered_sensors)

        self.start_hist_button = QPushButton("Start Histograms")
        self.start_hist_button.setToolTip("Build histogram distributions from all sensors.")
        self.start_hist_button.clicked.connect(self._start_histograms)

        self.stop_hist_button = QPushButton("Stop Histograms")
        self.stop_hist_button.setToolTip("Cancel the background histogram build.")
        self.stop_hist_button.clicked.connect(self._cancel_histograms)
        self.stop_hist_button.setVisible(False)

        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)

        self.figure = Figure(figsize=(10, 7), constrained_layout=False)
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.toolbar = NavigationToolbar(self.canvas, self)
        self.canvas.mpl_connect("button_press_event", self._on_canvas_click)
        self.canvas.mpl_connect("button_release_event", self._on_canvas_release)
        self.canvas.mpl_connect("scroll_event", self._on_scroll_zoom)
        self.canvas.mpl_connect("motion_notify_event", self._on_mouse_move)

        self.hist_figure = Figure(figsize=(10, 6), constrained_layout=False)
        self.hist_canvas = FigureCanvas(self.hist_figure)
        self.hist_canvas.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.hist_toolbar = NavigationToolbar(self.hist_canvas, self)
        self.hist_canvas.mpl_connect("button_press_event", self._on_canvas_press)
        self.hist_canvas.mpl_connect("button_release_event", self._on_canvas_release)
        self.hist_canvas.mpl_connect("scroll_event", self._on_scroll_zoom)
        self.hist_canvas.mpl_connect("motion_notify_event", self._on_mouse_move)

        self.tabs = QTabWidget()
        self.tabs.currentChanged.connect(self._on_tab_changed)

        self.setStatusBar(QStatusBar())
        self._build_layout()
        self.apply_filter()
        self._draw_empty_state()
        self._draw_histogram_idle_state()

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
        sidebar_layout.addWidget(self.plot_selected_button)
        sidebar_layout.addWidget(self.plot_filtered_button)
        sidebar_layout.addWidget(self.plot_all_button)
        sidebar_layout.addWidget(self.start_hist_button)
        sidebar_layout.addWidget(self.stop_hist_button)
        sidebar_layout.addWidget(self.progress_bar)

        curve_area = QWidget()
        curve_layout = QVBoxLayout(curve_area)
        curve_layout.setContentsMargins(0, 0, 0, 0)
        curve_layout.setSpacing(6)
        curve_layout.addWidget(self.toolbar)
        curve_layout.addWidget(self.canvas, stretch=1)
        curve_layout.addWidget(
            self._build_plot_controls(
                self.canvas,
                ("Gate leakage", "Output characteristic", "Transfer", "Transconductance"),
                columns=2,
            )
        )

        histogram_area = QWidget()
        histogram_layout = QVBoxLayout(histogram_area)
        histogram_layout.setContentsMargins(0, 0, 0, 0)
        histogram_layout.setSpacing(6)
        histogram_layout.addWidget(self.hist_toolbar)
        histogram_layout.addWidget(self.hist_canvas, stretch=1)
        histogram_layout.addWidget(
            self._build_plot_controls(
                self.hist_canvas,
                ("Gate Ig", "Transfer Id", "Output Id"),
                columns=3,
            )
        )

        self.tabs.addTab(curve_area, "Curves")
        self.tabs.addTab(histogram_area, "Histograms")

        main_layout.addWidget(sidebar)
        main_layout.addWidget(self.tabs, stretch=1)
        self.setCentralWidget(root)

    def _build_plot_controls(self, canvas: FigureCanvas, labels: tuple[str, ...], columns: int) -> QWidget:
        container = QWidget()
        grid = QGridLayout(container)
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setHorizontalSpacing(8)
        grid.setVerticalSpacing(6)

        for axis_index, label_text in enumerate(labels):
            cell = QWidget()
            cell_layout = QHBoxLayout(cell)
            cell_layout.setContentsMargins(0, 0, 0, 0)
            cell_layout.setSpacing(4)

            label = QLabel(label_text)
            label.setObjectName("plotControlLabel")
            cell_layout.addWidget(label)

            box_button = QPushButton("Box")
            box_button.setToolTip(f"Drag a zoom box on {label_text}.")
            box_button.clicked.connect(partial(self._set_plot_interaction, canvas, axis_index, "box"))
            cell_layout.addWidget(box_button)
            self._plot_control_buttons.append(box_button)

            pan_button = QPushButton("Pan")
            pan_button.setToolTip(f"Drag {label_text} to pan.")
            pan_button.clicked.connect(partial(self._set_plot_interaction, canvas, axis_index, "pan"))
            cell_layout.addWidget(pan_button)
            self._plot_control_buttons.append(pan_button)

            reset_button = QPushButton("Reset")
            reset_button.setToolTip(f"Reset zoom for {label_text}.")
            reset_button.clicked.connect(partial(self._reset_axis_zoom, canvas, axis_index))
            cell_layout.addWidget(reset_button)
            self._plot_control_buttons.append(reset_button)

            row, column = divmod(axis_index, columns)
            grid.addWidget(cell, row, column)
        return container

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
        if self._hist_thread is not None:
            self.statusBar().showMessage("Stop or finish histogram building before plotting a selected sensor.", 5000)
            return
        try:
            sensor, device = self._load_selected_device()
            result = fit_triode_eq_5_16(device) if self.reference_checkbox.isChecked() else None
            self._plot_device(sensor, device, fit_result=result)
        except LookupError as exc:
            self.statusBar().showMessage(str(exc), 5000)
            return
        except FitError as exc:
            QMessageBox.warning(self, "Could not fit ideal reference", str(exc))
            self.statusBar().showMessage(str(exc), 8000)
            return
        except Exception as exc:  # noqa: BLE001 - present spreadsheet read errors in the GUI.
            QMessageBox.critical(self, "Could not load sensor", str(exc))
            return

    def _on_reference_toggled(self, checked: bool) -> None:
        if self._worker_thread is not None:
            self.statusBar().showMessage("Cancel the background overlay before changing the fitted reference.", 5000)
            return
        if self.sensor_list.currentItem() is None:
            return
        self.statusBar().showMessage("Fitting and plotting selected sensor..." if checked else "Plotting selected sensor...")
        self.plot_selected()

    def _load_selected_device(self) -> tuple[SensorFiles, DeviceCurves]:
        item = self.sensor_list.currentItem()
        if item is None:
            raise LookupError("Choose a sensor first.")
        sensor = item.data(Qt.ItemDataRole.UserRole)
        return sensor, self._load_device_curves(sensor)

    def _load_device_curves(self, sensor: SensorFiles) -> DeviceCurves:
        return self.curve_cache.load_device_curves(sensor)

    def _plot_device(
        self,
        sensor: SensorFiles,
        device: DeviceCurves,
        *,
        fit_result,
    ) -> None:
        self._clear_plot_interaction()
        plot_single_device(
            self.figure,
            device,
            show_reference=fit_result is not None,
            reference_vth=fit_result.vth if fit_result is not None else None,
            reference_k=fit_result.k if fit_result is not None else None,
        )
        self._capture_axis_positions()
        self._remember_home_limits(self.canvas)
        self.canvas.draw_idle()
        if fit_result is None:
            self.statusBar().showMessage(f"Plotted {sensor.label}", 5000)
        else:
            self.statusBar().showMessage(
                (
                    f"{fit_result.method}: Vth={fit_result.vth:.4g} V, "
                    f"k={fit_result.k:.4g} mA/mm/V^2, points={fit_result.points_used}, "
                    f"RMS={fit_result.rms_error:.4g} mA/mm"
                ),
                12000,
            )

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
        if self._hist_thread is not None:
            self.statusBar().showMessage("Wait for histogram building to finish before starting a bulk overlay.", 5000)
            return

        self._clear_plot_interaction()
        self._set_bulk_busy(True, cancel_button)
        self._focused_axis_index = None
        self._bulk_title = title
        self._bulk_plotted = 0
        self._bulk_axes = prepare_overlay_figure(self.figure, title)
        self._capture_axis_positions()
        self._remember_home_limits(self.canvas)
        self.canvas.draw_idle()
        self.progress_bar.setMaximum(len(sensors))
        self.progress_bar.setValue(0)
        worker_count = max(1, (os.cpu_count() or 1) // CPU_CORE_DIVISOR)
        self.statusBar().showMessage(f"Loading {len(sensors):,} {label} with {worker_count} worker threads...")

        self._worker_thread = QThread(self)
        self._worker = LoadAllWorker(sensors, self._load_device_curves)
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
        self._remember_home_limits(self.canvas)
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
            self._remember_home_limits(self.canvas)
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

    def _on_tab_changed(self, index: int) -> None:
        if self.tabs.tabText(index) == "Histograms":
            self.statusBar().showMessage("Click Start Histograms to build histogram distributions.", 5000)

    def _set_plot_interaction(self, canvas: FigureCanvas, axis_index: int, mode: str) -> None:
        if self._plotting_busy():
            self.statusBar().showMessage("Zoom controls are disabled while plotting or building histograms.", 5000)
            return
        if (
            self._interaction_canvas is canvas
            and self._interaction_axis_index == axis_index
            and self._interaction_mode == mode
        ):
            self._clear_plot_interaction()
            return
        self._clear_plot_interaction()
        self._interaction_canvas = canvas
        self._interaction_axis_index = axis_index
        self._interaction_mode = mode
        self.statusBar().showMessage(
            f"{mode.title()} mode armed for one plot. Drag inside that plot to use it.",
            6000,
        )

    def _clear_plot_interaction(self) -> None:
        self._hide_rubber_band()
        self._interaction_canvas = None
        self._interaction_axis_index = None
        self._interaction_mode = None
        self._drag_start = None
        self._drag_start_limits = None

    def _reset_axis_zoom(self, canvas: FigureCanvas, axis_index: int) -> None:
        if self._plotting_busy():
            self.statusBar().showMessage("Zoom controls are disabled while plotting or building histograms.", 5000)
            return
        home_limits = self._home_limits_for_canvas(canvas)
        if axis_index >= len(canvas.figure.axes) or axis_index >= len(home_limits):
            return
        x_limits, y_limits = home_limits[axis_index]
        axis = canvas.figure.axes[axis_index]
        axis.set_xlim(x_limits)
        axis.set_ylim(y_limits)
        canvas.draw_idle()
        self.statusBar().showMessage("Reset zoom for one plot.", 4000)

    def _start_histograms(self) -> None:
        if self._hist_thread is not None:
            return
        if self._worker_thread is not None:
            self.statusBar().showMessage("Cancel the current bulk overlay before building histograms.", 5000)
            return
        self._clear_plot_interaction()
        self._set_plot_controls_enabled(False)
        self._hist_started = True
        self._hist_loaded = 0
        self._hist_gate_ig.clear()
        self._hist_transfer_id.clear()
        self._hist_output_id.clear()
        plot_histograms(
            self.hist_figure,
            self._hist_gate_ig,
            self._hist_transfer_id,
            self._hist_output_id,
            loaded_count=0,
        )
        self._remember_home_limits(self.hist_canvas)
        self.hist_canvas.draw_idle()

        self._hist_thread = QThread(self)
        self._hist_worker = HistogramWorker(self.sensors, self._load_device_curves)
        self._hist_worker.moveToThread(self._hist_thread)
        self._hist_thread.started.connect(self._hist_worker.run)
        self._hist_worker.progress.connect(self._on_hist_progress)
        self._hist_worker.batch_ready.connect(self._on_hist_batch_ready)
        self._hist_worker.finished.connect(self._on_hist_finished)
        self._hist_worker.failed.connect(self._on_hist_failed)
        self._hist_worker.finished.connect(self._hist_thread.quit)
        self._hist_worker.failed.connect(self._hist_thread.quit)
        self._hist_thread.finished.connect(self._cleanup_hist_worker)
        self._hist_thread.start()
        self.start_hist_button.setVisible(False)
        self.stop_hist_button.setText("Stop Histograms")
        self.stop_hist_button.setEnabled(True)
        self.stop_hist_button.setVisible(True)
        self.statusBar().showMessage("Building histograms from all sensors...")

    def _cancel_histograms(self) -> None:
        if self._hist_worker is None:
            return
        self._hist_worker.cancel()
        self.stop_hist_button.setText("Stopping...")
        self.stop_hist_button.setEnabled(False)
        self.statusBar().showMessage("Stopping histogram build...")

    def _on_hist_progress(self, done: int, total: int) -> None:
        self.progress_bar.setVisible(True)
        self.progress_bar.setMaximum(total)
        self.progress_bar.setValue(done)
        self.statusBar().showMessage(f"Built histogram samples for {done:,} of {total:,} sensors...")

    def _on_hist_batch_ready(self, samples: list[HistogramSample]) -> None:
        for sample in samples:
            self._hist_loaded += 1
            if sample.gate_ig is not None:
                self._hist_gate_ig.append(sample.gate_ig)
            if sample.transfer_id is not None:
                self._hist_transfer_id.append(sample.transfer_id)
            if sample.output_id is not None:
                self._hist_output_id.append(sample.output_id)
        plot_histograms(
            self.hist_figure,
            self._hist_gate_ig,
            self._hist_transfer_id,
            self._hist_output_id,
            loaded_count=self._hist_loaded,
        )
        self._remember_home_limits(self.hist_canvas)
        self.hist_canvas.draw_idle()

    def _on_hist_finished(self, loaded: int, total: int, cancelled: bool) -> None:
        self.progress_bar.setVisible(False)
        plot_histograms(
            self.hist_figure,
            self._hist_gate_ig,
            self._hist_transfer_id,
            self._hist_output_id,
            loaded_count=self._hist_loaded,
        )
        self._remember_home_limits(self.hist_canvas)
        self.hist_canvas.draw_idle()
        if cancelled:
            self._hist_started = False
            self.start_hist_button.setText("Start Histograms")
            self.statusBar().showMessage(f"Histogram build cancelled after {loaded:,} of {total:,} sensors.", 8000)
        else:
            self.start_hist_button.setText("Rebuild Histograms")
            self.statusBar().showMessage(f"Built histograms from {loaded:,} sensors.", 8000)
        self._set_plot_controls_enabled(True)
        self.start_hist_button.setEnabled(True)
        self.start_hist_button.setVisible(True)
        self.stop_hist_button.setVisible(False)

    def _on_hist_failed(self, message: str) -> None:
        self.progress_bar.setVisible(False)
        self._hist_started = False
        self._set_plot_controls_enabled(True)
        self.start_hist_button.setText("Start Histograms")
        self.start_hist_button.setEnabled(True)
        self.start_hist_button.setVisible(True)
        self.stop_hist_button.setVisible(False)
        QMessageBox.critical(self, "Could not build histograms", message)

    def _cleanup_hist_worker(self) -> None:
        if self._hist_worker is not None:
            self._hist_worker.deleteLater()
        if self._hist_thread is not None:
            self._hist_thread.deleteLater()
        self._hist_worker = None
        self._hist_thread = None

    def closeEvent(self, event) -> None:  # noqa: ANN001 - Qt event type.
        if self._worker is not None:
            self._worker.cancel()
        if self._hist_worker is not None:
            self._hist_worker.cancel()
        super().closeEvent(event)

    def _set_bulk_busy(self, busy: bool, cancel_button: QPushButton | None = None) -> None:
        self.progress_bar.setVisible(busy)
        self.plot_selected_button.setEnabled(not busy)
        self.reference_checkbox.setEnabled(not busy)
        self.start_hist_button.setEnabled(not busy)
        self._set_plot_controls_enabled(not busy)
        if busy:
            self._clear_plot_interaction()
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
        self._remember_home_limits(self.canvas)
        self.canvas.draw_idle()

    def _draw_histogram_idle_state(self) -> None:
        self.hist_figure.clear()
        self.hist_figure.patch.set_facecolor("#111827")
        ax = self.hist_figure.add_subplot(111)
        ax.set_facecolor("#172033")
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_color("#4b5563")
        ax.text(
            0.5,
            0.5,
            "Click Start Histograms",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=12,
            color=TEXT_COLOR,
        )
        self.hist_figure.tight_layout()
        self._remember_home_limits(self.hist_canvas)
        self.hist_canvas.draw_idle()

    def _on_mouse_move(self, event) -> None:  # noqa: ANN001 - matplotlib event object.
        if self._plotting_busy():
            return
        self._hover_canvas = event.canvas
        if event.inaxes is None:
            self._hover_axis_index = None
        else:
            try:
                self._hover_axis_index = event.canvas.figure.axes.index(event.inaxes)
            except ValueError:
                self._hover_axis_index = None
        self._update_custom_drag(event)

    def _on_canvas_press(self, event) -> bool:  # noqa: ANN001 - matplotlib event object.
        if self._plotting_busy():
            return False
        if event.button != 1:
            return False
        if event.canvas is not self._interaction_canvas:
            return False
        if self._interaction_axis_index is None or self._interaction_mode is None:
            return False
        if event.inaxes is None or event.xdata is None or event.ydata is None:
            return False
        if self._interaction_axis_index >= len(event.canvas.figure.axes):
            return False
        if event.inaxes is not event.canvas.figure.axes[self._interaction_axis_index]:
            return False

        self._drag_start = (event.xdata, event.ydata)
        self._drag_start_limits = (event.inaxes.get_xlim(), event.inaxes.get_ylim())
        if self._interaction_mode == "box":
            self._start_rubber_band(event)
        return True

    def _on_canvas_release(self, event) -> None:  # noqa: ANN001 - matplotlib event object.
        if self._plotting_busy():
            self._clear_plot_interaction()
            return
        if self._drag_start is None or self._interaction_mode is None:
            return
        if event.canvas is not self._interaction_canvas:
            return
        if self._interaction_axis_index is None or self._interaction_axis_index >= len(event.canvas.figure.axes):
            self._clear_plot_interaction()
            return

        axis = event.canvas.figure.axes[self._interaction_axis_index]
        should_redraw = False
        if self._interaction_mode == "box":
            x0, y0 = self._drag_start
            x1, y1 = self._event_data_point(event, axis)
            if abs(x1 - x0) > 1e-12 and abs(y1 - y0) > 1e-12:
                axis.set_xlim(min(x0, x1), max(x0, x1))
                axis.set_ylim(min(y0, y1), max(y0, y1))
                should_redraw = True

        self._hide_rubber_band()
        self._drag_start = None
        self._drag_start_limits = None
        if should_redraw:
            event.canvas.draw_idle()

    def _update_custom_drag(self, event) -> None:  # noqa: ANN001 - matplotlib event object.
        if self._plotting_busy():
            return
        if self._drag_start is None or self._interaction_mode is None:
            return
        if event.canvas is not self._interaction_canvas:
            return
        if self._interaction_axis_index is None or self._interaction_axis_index >= len(event.canvas.figure.axes):
            return

        x0, y0 = self._drag_start
        if self._interaction_mode == "box":
            self._update_rubber_band(event)
        elif (
            self._interaction_mode == "pan"
            and self._drag_start_limits is not None
            and event.inaxes is event.canvas.figure.axes[self._interaction_axis_index]
            and event.xdata is not None
            and event.ydata is not None
        ):
            axis = event.canvas.figure.axes[self._interaction_axis_index]
            (x_min, x_max), (y_min, y_max) = self._drag_start_limits
            dx = event.xdata - x0
            dy = event.ydata - y0
            axis.set_xlim(x_min - dx, x_max - dx)
            axis.set_ylim(y_min - dy, y_max - dy)
            event.canvas.draw_idle()

    def _start_rubber_band(self, event) -> None:  # noqa: ANN001 - matplotlib event object.
        origin = self._event_qpoint(event)
        self._rubber_band_origin = origin
        self._rubber_band = QRubberBand(QRubberBand.Shape.Rectangle, event.canvas)
        self._rubber_band.setGeometry(QRect(origin, origin))
        self._rubber_band.show()

    def _update_rubber_band(self, event) -> None:  # noqa: ANN001 - matplotlib event object.
        if self._rubber_band is None or self._rubber_band_origin is None:
            return
        axis = self._interaction_axis(event)
        if axis is None:
            return
        self._rubber_band.setGeometry(QRect(self._rubber_band_origin, self._event_qpoint(event, axis)).normalized())

    def _hide_rubber_band(self) -> None:
        if self._rubber_band is not None:
            self._rubber_band.hide()
            self._rubber_band.deleteLater()
        self._rubber_band = None
        self._rubber_band_origin = None

    def _interaction_axis(self, event):  # noqa: ANN001 - matplotlib event object.
        if self._interaction_axis_index is None or self._interaction_axis_index >= len(event.canvas.figure.axes):
            return None
        return event.canvas.figure.axes[self._interaction_axis_index]

    def _event_display_point(self, event, axis) -> tuple[float, float]:  # noqa: ANN001 - matplotlib objects.
        x = event.x if event.x is not None else axis.bbox.x0
        y = event.y if event.y is not None else axis.bbox.y0
        x = min(max(float(x), axis.bbox.x0), axis.bbox.x1)
        y = min(max(float(y), axis.bbox.y0), axis.bbox.y1)
        return x, y

    def _event_data_point(self, event, axis) -> tuple[float, float]:  # noqa: ANN001 - matplotlib objects.
        return tuple(axis.transData.inverted().transform(self._event_display_point(event, axis)))

    def _event_qpoint(self, event, axis=None) -> QPoint:  # noqa: ANN001 - matplotlib event object.
        if axis is None:
            x = event.x
            y = event.y
        else:
            x, y = self._event_display_point(event, axis)
        return QPoint(int(x), int(event.canvas.height() - y))

    def _on_scroll_zoom(self, event) -> None:  # noqa: ANN001 - matplotlib event object.
        if self._plotting_busy():
            return
        if event.inaxes is None or event.xdata is None or event.ydata is None:
            return
        toolbar = self._toolbar_for_canvas(event.canvas)
        if toolbar is not None and getattr(toolbar, "mode", ""):
            return

        scale = 0.82 if event.button == "up" else 1.22
        self._zoom_axis(event.inaxes, event.xdata, event.ydata, scale)
        event.canvas.draw_idle()

    def _toolbar_for_canvas(self, canvas) -> NavigationToolbar | None:  # noqa: ANN001 - matplotlib canvas object.
        if canvas is self.canvas:
            return self.toolbar
        if canvas is self.hist_canvas:
            return self.hist_toolbar
        return None

    def _plotting_busy(self) -> bool:
        return self._worker_thread is not None or self._hist_thread is not None

    def _set_plot_controls_enabled(self, enabled: bool) -> None:
        for button in self._plot_control_buttons:
            button.setEnabled(enabled)
        self.toolbar.setEnabled(enabled)
        self.hist_toolbar.setEnabled(enabled)

    def _home_limits_for_canvas(self, canvas) -> list[tuple[tuple[float, float], tuple[float, float]]]:  # noqa: ANN001
        if canvas is self.canvas:
            return self._curve_home_limits
        if canvas is self.hist_canvas:
            return self._hist_home_limits
        return []

    def _remember_home_limits(self, canvas) -> None:  # noqa: ANN001 - matplotlib canvas object.
        limits = [(axis.get_xlim(), axis.get_ylim()) for axis in canvas.figure.axes]
        if canvas is self.canvas:
            self._curve_home_limits = limits
        elif canvas is self.hist_canvas:
            self._hist_home_limits = limits

    def _zoom_axis(self, axis, x_center: float, y_center: float, scale: float) -> None:  # noqa: ANN001 - matplotlib axis.
        x_min, x_max = axis.get_xlim()
        y_min, y_max = axis.get_ylim()
        new_width = (x_max - x_min) * scale
        new_height = (y_max - y_min) * scale

        x_rel = (x_center - x_min) / (x_max - x_min) if x_max != x_min else 0.5
        y_rel = (y_center - y_min) / (y_max - y_min) if y_max != y_min else 0.5

        axis.set_xlim(x_center - new_width * x_rel, x_center + new_width * (1.0 - x_rel))
        axis.set_ylim(y_center - new_height * y_rel, y_center + new_height * (1.0 - y_rel))

    def _capture_axis_positions(self) -> None:
        if self._focused_axis_index is not None:
            return
        self._axis_positions = [axis.get_position().frozen() for axis in self.figure.axes]

    def _on_canvas_click(self, event) -> None:  # noqa: ANN001 - matplotlib event object.
        if self._on_canvas_press(event):
            return
        if getattr(self.toolbar, "mode", ""):
            return
        if self._interaction_canvas is self.canvas and self._interaction_mode is not None:
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
        QLabel#plotControlLabel {
            color: #cbd5e1;
            font-weight: 600;
            min-width: 118px;
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
        QTabWidget::pane {
            border: 1px solid #334155;
            border-radius: 4px;
            top: -1px;
        }
        QTabBar::tab {
            color: #cbd5e1;
            background: #172033;
            border: 1px solid #334155;
            padding: 8px 14px;
            margin-right: 2px;
            border-top-left-radius: 4px;
            border-top-right-radius: 4px;
        }
        QTabBar::tab:selected {
            color: #ffffff;
            background: #243047;
            border-bottom-color: #243047;
        }
        QTabBar::tab:hover {
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

    sensors = discover_sensors(data_dir)
    cache_path = project_root / "cache" / "transistor_data.sqlite3"
    try:
        curve_cache = SQLiteCurveCache.open(cache_path, data_dir, sensors)
    except CacheMemoryError as exc:
        QMessageBox.critical(None, "Not enough memory", str(exc))
        return 1
    except Exception as exc:  # noqa: BLE001 - cache preparation failures need to be visible.
        QMessageBox.critical(None, "Could not prepare SQLite cache", str(exc))
        return 1

    window = MainWindow(data_dir, sensors, curve_cache)
    window.show()
    return app.exec()
