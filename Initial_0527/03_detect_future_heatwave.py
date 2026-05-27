"""
03_detect_future_heatwave.py

Detect future-relative heatwave days from CMIP6 tasmax over China.

Definition:
  A heatwave day at a grid cell is a summer (JJA) day in 2041–2060
  where tasmax exceeds that grid cell's 90th percentile of summer tasmax
  computed over the same 2041–2060 period.

Input:
  data/processed/tasmax_ssp245_2041_2060_china.nc

Outputs:
  data/processed/tasmax_p90_ssp245_2041_2060_china.nc   (lat × lon)
  data/processed/heatwave_mask_ssp245_2041_2060_china.nc (time × lat × lon)
  outputs/heatwave_detection_report.txt
"""

import datetime
from pathlib import Path

import numpy as np
import xarray as xr

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT      = Path(__file__).parent.parent
PROCESSED = ROOT / "data" / "processed"
OUT       = ROOT / "outputs"

PROCESSED.mkdir(parents=True, exist_ok=True)
OUT.mkdir(parents=True, exist_ok=True)

INPUT_FILE   = PROCESSED / "tasmax_ssp245_2041_2060_china.nc"
P90_FILE     = PROCESSED / "tasmax_p90_ssp245_2041_2060_china.nc"
MASK_FILE    = PROCESSED / "heatwave_mask_ssp245_2041_2060_china.nc"
REPORT_FILE  = OUT / "heatwave_detection_report.txt"

SUMMER_MONTHS = [6, 7, 8]   # JJA
KELVIN_OFFSET = 273.15


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_coord(ds, candidates):
    """Return the first coordinate name from candidates found in ds."""
    for name in candidates:
        if name in ds.coords:
            return name
    raise KeyError(f"None of {candidates} found in coords: {list(ds.coords)}")


def _get_tasmax_var(ds):
    """
    Return the name of the tasmax variable in ds.
    Raises a descriptive error if not found.
    """
    if "tasmax" in ds.data_vars:
        return "tasmax"
    raise KeyError(
        f"Variable 'tasmax' not found. Available variables: {list(ds.data_vars)}"
    )


def _maybe_convert_to_celsius(da):
    """
    Convert a DataArray from Kelvin to Celsius if its units attribute
    indicates Kelvin.  Returns (converted_da, original_units, converted).
    """
    units = da.attrs.get("units", "")
    if units in ("K", "Kelvin", "kelvin"):
        da = da - KELVIN_OFFSET
        da.attrs["units"] = "°C"
        return da, units, True
    return da, units, False


# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------

def detect_heatwaves(input_path):
    """
    Open the clipped tasmax file, compute per-grid-cell JJA p90 over
    2041–2060, flag heatwave days, and return the outputs plus report stats.
    """
    ds = xr.open_dataset(input_path)

    # --- detect coordinate and variable names ---
    lat_name  = _get_coord(ds, ["lat", "latitude"])
    lon_name  = _get_coord(ds, ["lon", "longitude"])
    tmax_name = _get_tasmax_var(ds)

    da = ds[tmax_name]

    # --- unit conversion ---
    da, orig_units, converted = _maybe_convert_to_celsius(da)

    # --- select summer months (JJA) ---
    summer_mask = da["time.month"].isin(SUMMER_MONTHS)
    da_summer   = da.sel(time=summer_mask)

    # --- per-grid-cell 90th percentile over all summer days in 2041–2060 ---
    # Result shape: (lat, lon); drop the 'quantile' auxiliary coordinate
    # that xarray adds automatically.
    p90 = da_summer.quantile(0.90, dim="time").drop_vars("quantile")
    p90.attrs["long_name"]   = "90th percentile of summer tasmax (2041–2060)"
    p90.attrs["units"]       = da.attrs.get("units", "°C")
    p90.attrs["description"] = "JJA p90 computed over the 2041–2060 future period"

    # --- heatwave mask: 1 where tasmax > p90, 0 otherwise ---
    # Broadcasting works because p90 is (lat, lon) and da_summer is (time, lat, lon).
    mask = (da_summer > p90).astype(np.int8)
    mask.attrs["long_name"]   = "Heatwave day mask (future-relative)"
    mask.attrs["description"] = (
        "1 = summer day where tasmax > grid-cell JJA p90 for 2041–2060; 0 otherwise"
    )
    mask.attrs["flag_values"] = "0, 1"

    # --- report statistics ---
    time_all    = ds["time"]
    time_summer = da_summer["time"]
    n_summer    = int(time_summer.sizes["time"])
    n_lat       = int(da_summer.sizes[lat_name])
    n_lon       = int(da_summer.sizes[lon_name])

    n_hw_days   = int(mask.values.sum())
    total_cells = n_summer * n_lat * n_lon
    pct_hw      = 100.0 * n_hw_days / total_cells if total_cells > 0 else 0.0

    stats = {
        "input_file":       input_path.name,
        "tmax_var":         tmax_name,
        "orig_units":       orig_units,
        "converted":        converted,
        "full_time_range":  (str(time_all.values[0])[:10], str(time_all.values[-1])[:10]),
        "summer_time_range":(str(time_summer.values[0])[:10], str(time_summer.values[-1])[:10]),
        "n_summer_days":    n_summer,
        "p90_min":          float(p90.values.min()),
        "p90_mean":         float(p90.values.mean()),
        "p90_max":          float(p90.values.max()),
        "n_heatwave_days":  n_hw_days,
        "total_cells":      total_cells,
        "pct_heatwave":     pct_hw,
    }

    ds.close()
    return p90, mask, stats


# ---------------------------------------------------------------------------
# Report formatting
# ---------------------------------------------------------------------------

def format_report(stats):
    lines = []
    lines.append("Heatwave detection report")
    lines.append(f"Generated        : {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}")
    lines.append("")
    lines.append(f"Input file       : {stats['input_file']}")
    lines.append(f"tasmax variable  : {stats['tmax_var']}")
    lines.append(f"Original units   : {stats['orig_units']}")
    lines.append(
        f"Unit conversion  : {'K → °C applied' if stats['converted'] else 'none (already °C)'}"
    )
    lines.append("")
    lines.append(f"Full time range  : {stats['full_time_range'][0]} → {stats['full_time_range'][1]}")
    lines.append(
        f"Summer (JJA) range: {stats['summer_time_range'][0]} → {stats['summer_time_range'][1]}"
        f"  ({stats['n_summer_days']} days)"
    )
    lines.append("")
    lines.append("p90 threshold (°C):")
    lines.append(f"  min  : {stats['p90_min']:.2f}")
    lines.append(f"  mean : {stats['p90_mean']:.2f}")
    lines.append(f"  max  : {stats['p90_max']:.2f}")
    lines.append("")
    lines.append(f"Heatwave grid-cell-days : {stats['n_heatwave_days']:,}")
    lines.append(f"Total summer grid-cell-days : {stats['total_cells']:,}")
    lines.append(f"Heatwave fraction       : {stats['pct_heatwave']:.2f}%")
    lines.append("")
    lines.append(f"Output p90 file  : {P90_FILE.name}")
    lines.append(f"Output mask file : {MASK_FILE.name}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Input file not found: {INPUT_FILE}\n"
            "Run 02_clip_china.py first."
        )

    print(f"Opening {INPUT_FILE.name} ...")
    p90, mask, stats = detect_heatwaves(INPUT_FILE)

    print("Saving p90 threshold ...")
    p90.rename("tasmax_p90").to_dataset().to_netcdf(P90_FILE)

    print("Saving heatwave mask ...")
    mask.rename("heatwave_mask").to_dataset().to_netcdf(MASK_FILE)

    report = format_report(stats)
    print("\n" + report)

    REPORT_FILE.write_text(report, encoding="utf-8")
    print(f"\nReport saved → {REPORT_FILE.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
