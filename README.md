# Transistor Curve Plotter

PyQt6 app for plotting the transistor measurements in `TransistorData`.

On startup, the app builds or refreshes a local SQLite cache at `cache/transistor_data.sqlite3`, then copies that database into memory for fast queries. If there is not enough available memory to load the SQLite cache safely, the app exits cleanly with a popup instead of trying to run partially loaded.

## Run

```bash
python -m venv .venv
source .venv/bin/activate
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python app.py
```

## What It Plots

For a selected sensor, the app makes one four-panel figure:

- `Ig (uA/mm)` vs `Vgs (V)` from the DIODE workbook
- `Id (mA/mm)` vs `Vds (V)` from the IV workbook
- `Id (mA/mm)` vs `Vgs (V)` from the first TRANS sheet
- `gm (mS/mm)` vs `Vgs (V)` from the second TRANS sheet

Check `Fit ideal MOSFET reference` to immediately replot the selected sensor with dashed textbook enhancement-NMOS reference curves. The fit first uses the largest-`VDS` transfer curve for a saturation-region starting estimate, then fits Sedra/Smith Eq. 5.16 on triode-region transfer points using least squares. The fitted values are reported in the status bar. Bulk `Plot All Sensors` and `Plot Filtered` overlays do not include fitted reference curves.

The sensor list is indexed from complete DIODE/IV/TRANS file triples. Use search to filter the list, `Plot Selected` for one device, `Plot Filtered` for only the sensors currently visible in the list, or `Plot All Sensors` for a full overlay on the same four axes. Bulk plots run in the background; the active bulk button becomes `Cancel` while it is loading. The background loader uses `CPU_CORE_DIVISOR` in `transistor_plotter/main_window.py`; set it to `8` for about one eighth of available cores or `4` for about one quarter.

Use the `Histograms` tab to view distributions across all sensors at the whiteboard voltage points. Opening the tab does not start processing; click `Start Histograms` when you want to build them. Histogram building runs in the background, can be cancelled with `Stop Histograms`, and can be rerun with `Rebuild Histograms`. Each histogram displays the finite sample count and RMS.

- gate leakage `Ig` at `Vgs = +0.1 V` using DIODE `VDS = 0 V` and `0.5 V`
- transfer `Id` at `Vgs = +0.1 V` using TRANS `VDS = 0.1 V`, `0.5 V`, and `0.9 V`
- output `Id` at `Vds = +0.5 V` using IV `VGS = -0.3 V`, `0.1 V`, and `0.5 V`

Click any curve plot panel to focus it. Click that focused panel again, or click outside the plot area, to return to the four-panel view. Use the mouse wheel over any curve or histogram axis to zoom that plot in and out around the cursor. Each curve and histogram panel has its own `Box`, `Pan`, and `Reset` buttons below the plot so zoom controls stay independent.
