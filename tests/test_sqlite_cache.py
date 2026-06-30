from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from openpyxl import Workbook

from transistor_plotter.data_loader import discover_sensors
from transistor_plotter.sqlite_cache import SQLiteCurveCache


class SQLiteCacheTests(unittest.TestCase):
    def test_cache_round_trips_workbook_curves_from_memory_sqlite(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data_dir = root / "TransistorData" / "GroupA"
            data_dir.mkdir(parents=True)

            _write_workbook(data_dir / "DIODE_BH0001_4F50_T_296_K.xlsx", [["VGS (V)", "VDS = 0.500 V"], [0.0, 1.0], [0.1, 3.0]])
            _write_workbook(data_dir / "IV_BH0001_4F50_T_296_K.xlsx", [["VDS (V)", "VGS = 0.100 V"], [0.0, 2.0], [0.5, 4.0]])
            _write_workbook(
                data_dir / "TRANS_BH0001_4F50_T_296_K.xlsx",
                [["VGS (V)", "VDS = 0.900 V"], [0.0, 5.0], [0.1, 7.0]],
                [["VGS (V)", "VDS = 0.900 V"], [0.0, 6.0], [0.1, 8.0]],
            )

            sensors = discover_sensors(root / "TransistorData")
            cache = SQLiteCurveCache.open(root / "cache" / "curves.sqlite3", root / "TransistorData", sensors)
            device = cache.load_device_curves(sensors[0])

            self.assertEqual(device.sensor.device_id, "BH0001")
            self.assertEqual(device.diode_ig_vgs.curves[0].label, "VDS = 0.500 V")
            self.assertEqual(device.diode_ig_vgs.curves[0].y.tolist(), [1.0, 3.0])
            self.assertEqual(device.iv_id_vds.curves[0].y.tolist(), [2.0, 4.0])
            self.assertEqual(device.trans_gm_vgs.curves[0].y.tolist(), [6.0, 8.0])


def _write_workbook(path: Path, *sheets: list[list[object]]) -> None:
    workbook = Workbook()
    for index, rows in enumerate(sheets):
        worksheet = workbook.active if index == 0 else workbook.create_sheet()
        worksheet.title = f"Sheet{index + 1}"
        for row in rows:
            worksheet.append(row)
    workbook.save(path)


if __name__ == "__main__":
    unittest.main()
