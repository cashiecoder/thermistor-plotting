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

Check `Fit ideal MOSFET reference` to immediately replot the selected sensor with dashed textbook enhancement-NMOS reference curves. The fit uses only the bottom-left green largest-`VDS` TRANS transfer curve for one selected device. The `Fit Id range` two-handle slider chooses the current window in 10 mA/mm steps.

Reference fitting mode buttons:

- `Green Sat`: fits `Id = (1/2) * k * (Vgs - Vt)^2` on the green largest-`VDS` transfer curve inside the selected `Id` window and satisfying `Vds >= Vgs - Vt`. The code solves this by fitting `sqrt(Id) = sqrt(k/2) * Vgs - sqrt(k/2) * Vt` with least squares, then iterates the saturation-region mask because `Vt` is part of the mask.
- `Green Tri`: fits `Id = k * ((Vgs - Vt) * Vds - (1/2) * Vds^2)` on the green largest-`VDS` transfer curve inside the selected `Id` window and satisfying `Vds < Vgs - Vt`. If the green curve never satisfies `Vds < Vgs - Vt`, the app reports that there are not enough triode-region points rather than fitting the wrong region.
- `Blue Tri`: fits the same Eq. 5.16 triode model on the blue smallest-`VDS` transfer curve. This is usually the better choice when you want a triode fit from TRANS data, because low `VDS` is much more likely to satisfy `Vds < Vgs - Vt`.
- For both triode modes, with constant curve `Vds`, the code solves the linear least-squares form `Id = k * (Vgs * Vds - (1/2) * Vds^2) - k * Vt * Vds`, then iterates the triode-region mask.
- After either mode, the fitted `Vt` and `k` are used for the dashed references. The upper-right output reference curves draw only the triode segment from `Vds = 0` to `Vov = Vgs - Vt`, stopping at the overdrive voltage instead of continuing into saturation.

Bulk `Plot All Sensors` and `Plot Filtered` overlays do not include fitted reference curves.

Click `Operating Point Model` to open a separate modeling window with its own sensor sidebar. This window focuses on the normal operating point `Vds = 0.5 V`, `Vgs = +0.1 V`:

- It fits the orange TRANS transfer curve, `Id` vs `Vgs` at `Vds = 0.5 V`.
- The fitted model is `Id = [(k/2) * (Vgs - Vt)^2] * H(Vgs - Vt)`, where `H(x)=1` for `x>=0` and `H(x)=0` for `x<0`.
- For each candidate `Vt`, the code requires the operating point `Vgs = +0.1 V` to be above threshold and inside the fit region, limits the fit region to `Vgs < (Vds + Vt) / 2`, solves the best linear `k` by least squares, and keeps the `Vt`/`k` pair with the lowest squared error.
- The transfer plot shows only the measured orange points used in the fit plus the fitted curve over that same `Vgs` range.
- The output plot uses the fitted `Vt` and `k` to draw `Id = k * ((Vgs - Vt) * Vds - (1/2) * Vds^2)` for the orange `Vgs = +0.1 V` IV curve, only where `Vds < Vgs - Vt`.

The sensor list is indexed from complete DIODE/IV/TRANS file triples. Use search to filter the list, `Plot Selected` for one device, `Plot Filtered` for only the sensors currently visible in the list, or `Plot All Sensors` for a full overlay on the same four axes. Bulk plots run in the background; the active bulk button becomes `Cancel` while it is loading. The background loader uses `CPU_CORE_DIVISOR` in `transistor_plotter/main_window.py`; set it to `8` for about one eighth of available cores or `4` for about one quarter.

Use the `Histograms` tab to view distributions across all sensors at the whiteboard voltage points. Opening the tab does not start processing; click `Start Histograms` when you want to build them. Histogram building runs in the background, can be cancelled with `Stop Histograms`, and can be rerun with `Rebuild Histograms`. Each histogram displays the finite sample count and RMS. Use `Hist Scale` to toggle the histogram count axis between linear and log scale.

- gate leakage `Ig` at `Vgs = +0.1 V` using DIODE `VDS = 0 V` and `0.5 V`
- transfer `Id` at `Vgs = +0.1 V` using TRANS `VDS = 0.1 V`, `0.5 V`, and `0.9 V`
- output `Id` at `Vds = +0.5 V` using IV `VGS = -0.3 V`, `0.1 V`, and `0.5 V`

Click any curve plot panel to focus it. Click that focused panel again, or click outside the plot area, to return to the four-panel view. Use the mouse wheel over any curve or histogram axis to zoom that plot in and out around the cursor. Each curve and histogram panel has its own `Box`, `Pan`, and `Reset` buttons below the plot so zoom controls stay independent.
