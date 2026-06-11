# Transistor Curve Plotter

PyQt6 app for plotting the transistor measurements in `TransistorData`.

## Run

```bash
.venv/bin/python app.py
```

## What It Plots

For a selected sensor, the app makes one four-panel figure:

- `Ig (uA/mm)` vs `Vgs (V)` from the DIODE workbook
- `Id (mA/mm)` vs `Vds (V)` from the IV workbook
- `Id (mA/mm)` vs `Vgs (V)` from the first TRANS sheet
- `gm (mS/mm)` vs `Vgs (V)` from the second TRANS sheet

Check `Ideal MOSFET reference` and enter explicit `Vth` and `k` values to overlay dashed textbook enhancement-NMOS reference curves. The app does not estimate or default these parameters.

The sensor list is indexed from complete DIODE/IV/TRANS file triples. Use search to filter the list, `Plot Selected` for one device, `Plot Filtered` for only the sensors currently visible in the list, or `Plot All Sensors` for a full overlay on the same four axes. Bulk plots run in the background; the active bulk button becomes `Cancel` while it is loading. The background loader uses `CPU_CORE_DIVISOR` in `transistor_plotter/main_window.py`; set it to `8` for about one eighth of available cores or `4` for about one quarter.

Click any plot panel to focus it. Click that focused panel again, or click outside the plot area, to return to the four-panel view.
