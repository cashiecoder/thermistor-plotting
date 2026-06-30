from __future__ import annotations

import io
import os
import sqlite3
import tempfile
import time
import threading
from pathlib import Path

import numpy as np

from .data_loader import load_device_curves as load_xlsx_device_curves
from .models import Curve, CurveSet, DeviceCurves, SensorFiles

SCHEMA_VERSION = "1"
MEMORY_SAFETY_FACTOR = 2.0


class CacheMemoryError(RuntimeError):
    pass


class SQLiteCurveCache:
    def __init__(self, connection: sqlite3.Connection) -> None:
        self._connection = connection
        self._lock = threading.Lock()

    @classmethod
    def open(cls, cache_path: Path, data_dir: Path, sensors: list[SensorFiles]) -> "SQLiteCurveCache":
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        fingerprint = _source_fingerprint(data_dir)
        if not _cache_is_current(cache_path, fingerprint):
            build_cache(cache_path, sensors, fingerprint)
        _ensure_memory_available(cache_path)

        disk_connection = sqlite3.connect(cache_path)
        memory_connection = sqlite3.connect(":memory:", check_same_thread=False)
        try:
            disk_connection.backup(memory_connection)
        finally:
            disk_connection.close()
        return cls(memory_connection)

    def load_device_curves(self, sensor: SensorFiles) -> DeviceCurves:
        with self._lock:
            row = self._connection.execute(
                """
                SELECT id, diode_path, iv_path, trans_path
                FROM sensors
                WHERE group_name = ? AND device_id = ?
                """,
                (sensor.group, sensor.device_id),
            ).fetchone()
            if row is None:
                raise KeyError(f"Sensor is missing from SQLite cache: {sensor.label}")

            sensor_id, diode_path, iv_path, trans_path = row
            cached_sensor = SensorFiles(
                group=sensor.group,
                device_id=sensor.device_id,
                diode=Path(diode_path),
                iv=Path(iv_path),
                trans=Path(trans_path),
            )
            curve_sets = {
                panel: self._load_curve_set(sensor_id, panel)
                for panel in ("diode", "iv", "trans_id", "trans_gm")
            }

        return DeviceCurves(
            sensor=cached_sensor,
            diode_ig_vgs=curve_sets["diode"],
            iv_id_vds=curve_sets["iv"],
            trans_id_vgs=curve_sets["trans_id"],
            trans_gm_vgs=curve_sets["trans_gm"],
        )

    def _load_curve_set(self, sensor_id: int, panel: str) -> CurveSet:
        row = self._connection.execute(
            """
            SELECT id, title, xlabel, ylabel
            FROM curve_sets
            WHERE sensor_id = ? AND panel = ?
            """,
            (sensor_id, panel),
        ).fetchone()
        if row is None:
            return CurveSet(title="", xlabel="", ylabel="", curves=())

        curve_set_id, title, xlabel, ylabel = row
        curves = []
        for label, x_blob, y_blob in self._connection.execute(
            """
            SELECT label, x_blob, y_blob
            FROM curves
            WHERE curve_set_id = ?
            ORDER BY curve_index
            """,
            (curve_set_id,),
        ):
            curves.append(Curve(label=label, x=_array_from_blob(x_blob), y=_array_from_blob(y_blob)))
        return CurveSet(title=title, xlabel=xlabel, ylabel=ylabel, curves=tuple(curves))


def build_cache(cache_path: Path, sensors: list[SensorFiles], fingerprint: dict[str, str]) -> None:
    fd, tmp_name = tempfile.mkstemp(prefix=f"{cache_path.stem}.", suffix=".tmp", dir=cache_path.parent)
    os.close(fd)
    tmp_path = Path(tmp_name)
    try:
        connection = sqlite3.connect(tmp_path)
        try:
            _create_schema(connection)
            _insert_metadata(connection, fingerprint)
            for sensor in sensors:
                device = load_xlsx_device_curves(sensor)
                sensor_id = _insert_sensor(connection, device.sensor)
                _insert_curve_set(connection, sensor_id, "diode", device.diode_ig_vgs)
                _insert_curve_set(connection, sensor_id, "iv", device.iv_id_vds)
                _insert_curve_set(connection, sensor_id, "trans_id", device.trans_id_vgs)
                _insert_curve_set(connection, sensor_id, "trans_gm", device.trans_gm_vgs)
            connection.commit()
            connection.execute("VACUUM")
        finally:
            connection.close()
        tmp_path.replace(cache_path)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise


def _create_schema(connection: sqlite3.Connection) -> None:
    connection.executescript(
        """
        PRAGMA journal_mode = OFF;
        PRAGMA synchronous = OFF;
        CREATE TABLE metadata (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );
        CREATE TABLE sensors (
            id INTEGER PRIMARY KEY,
            group_name TEXT NOT NULL,
            device_id TEXT NOT NULL,
            diode_path TEXT NOT NULL,
            iv_path TEXT NOT NULL,
            trans_path TEXT NOT NULL,
            UNIQUE(group_name, device_id)
        );
        CREATE TABLE curve_sets (
            id INTEGER PRIMARY KEY,
            sensor_id INTEGER NOT NULL,
            panel TEXT NOT NULL,
            title TEXT NOT NULL,
            xlabel TEXT NOT NULL,
            ylabel TEXT NOT NULL,
            UNIQUE(sensor_id, panel)
        );
        CREATE TABLE curves (
            id INTEGER PRIMARY KEY,
            curve_set_id INTEGER NOT NULL,
            curve_index INTEGER NOT NULL,
            label TEXT NOT NULL,
            x_blob BLOB NOT NULL,
            y_blob BLOB NOT NULL
        );
        CREATE INDEX idx_curve_sets_sensor_panel ON curve_sets(sensor_id, panel);
        CREATE INDEX idx_curves_curve_set ON curves(curve_set_id, curve_index);
        """
    )


def _insert_metadata(connection: sqlite3.Connection, fingerprint: dict[str, str]) -> None:
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "built_at": str(time.time()),
        **fingerprint,
    }
    connection.executemany(
        "INSERT INTO metadata(key, value) VALUES(?, ?)",
        sorted(metadata.items()),
    )


def _insert_sensor(connection: sqlite3.Connection, sensor: SensorFiles) -> int:
    cursor = connection.execute(
        """
        INSERT INTO sensors(group_name, device_id, diode_path, iv_path, trans_path)
        VALUES(?, ?, ?, ?, ?)
        """,
        (sensor.group, sensor.device_id, str(sensor.diode), str(sensor.iv), str(sensor.trans)),
    )
    return int(cursor.lastrowid)


def _insert_curve_set(connection: sqlite3.Connection, sensor_id: int, panel: str, curve_set: CurveSet) -> None:
    cursor = connection.execute(
        """
        INSERT INTO curve_sets(sensor_id, panel, title, xlabel, ylabel)
        VALUES(?, ?, ?, ?, ?)
        """,
        (sensor_id, panel, curve_set.title, curve_set.xlabel, curve_set.ylabel),
    )
    curve_set_id = int(cursor.lastrowid)
    connection.executemany(
        """
        INSERT INTO curves(curve_set_id, curve_index, label, x_blob, y_blob)
        VALUES(?, ?, ?, ?, ?)
        """,
        [
            (curve_set_id, index, curve.label, _array_to_blob(curve.x), _array_to_blob(curve.y))
            for index, curve in enumerate(curve_set.curves)
        ],
    )


def _cache_is_current(cache_path: Path, fingerprint: dict[str, str]) -> bool:
    if not cache_path.exists():
        return False
    try:
        connection = sqlite3.connect(cache_path)
        try:
            rows = dict(connection.execute("SELECT key, value FROM metadata"))
        finally:
            connection.close()
    except sqlite3.Error:
        return False
    return rows.get("schema_version") == SCHEMA_VERSION and all(rows.get(key) == value for key, value in fingerprint.items())


def _source_fingerprint(data_dir: Path) -> dict[str, str]:
    count = 0
    total_size = 0
    newest_mtime_ns = 0
    for path in data_dir.rglob("*.xlsx"):
        stat = path.stat()
        count += 1
        total_size += stat.st_size
        newest_mtime_ns = max(newest_mtime_ns, stat.st_mtime_ns)
    return {
        "source_file_count": str(count),
        "source_total_size": str(total_size),
        "source_newest_mtime_ns": str(newest_mtime_ns),
    }


def _ensure_memory_available(cache_path: Path) -> None:
    required = int(cache_path.stat().st_size * MEMORY_SAFETY_FACTOR)
    available = _available_memory_bytes()
    if available is not None and available < required:
        raise CacheMemoryError(
            "Not enough free memory to load the SQLite curve cache into RAM. "
            f"Need about {_format_bytes(required)}, available {_format_bytes(available)}."
        )


def _available_memory_bytes() -> int | None:
    meminfo = Path("/proc/meminfo")
    if meminfo.exists():
        for line in meminfo.read_text(encoding="utf-8").splitlines():
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) * 1024
    try:
        pages = os.sysconf("SC_AVPHYS_PAGES")
        page_size = os.sysconf("SC_PAGE_SIZE")
    except (AttributeError, OSError, ValueError):
        return None
    return int(pages) * int(page_size)


def _array_to_blob(array: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    np.save(buffer, np.asarray(array, dtype=float), allow_pickle=False)
    return buffer.getvalue()


def _array_from_blob(blob: bytes) -> np.ndarray:
    return np.load(io.BytesIO(blob), allow_pickle=False)


def _format_bytes(value: int) -> str:
    size = float(value)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if size < 1024.0:
            return f"{size:.1f} {unit}"
        size /= 1024.0
    return f"{size:.1f} PB"
