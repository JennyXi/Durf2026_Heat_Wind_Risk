"""
04_extract_heatwave_wind.py

Identify hot-low-wind compound events over China.

Definition:
  A hot-low-wind event at a grid cell is a day where:
    (1) heatwave_mask == 1   (from 03_detect_future_heatwave.py)
    (2) sfcWind < 0.57 m/s

Inputs:
  data/processed/heatwave_mask_ssp245_2041_2060_china.nc
  data/processed/sfcWind_ssp245_2041_2060_china.nc

Outputs:
  data/processed/hot_low_wind_mask_ssp245_2041_2060_china.nc        (time × lat × lon)
  data/processed/hot_low_wind_probability_ssp245_2041_2060_china.nc (lat × lon)
  outputs/hot_low_wind_report.txt
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

MASK_INPUT_FILE = PROCESSED / "heatwave_mask_ssp245_2041_2060_china.nc"
WIND_INPUT_FILE = PROCESSED / "sfcWind_ssp245_2041_2060_china.nc"

HLW_MASK_FILE = PROCESSED / "hot_low_wind_mask_ssp245_2041_2060_china.nc"
HLW_PROB_FILE = PROCESSED / "hot_low_wind_probability_ssp245_2041_2060_china.nc"
REPORT_FILE   = OUT / "hot_low_wind_report.txt"

WIND_THRESHOLD = 0.57   # m/s — corridor activation threshold


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_coord(ds, candidates):
    """Return the first coordinate name from candidates found in ds."""
    for name in candidates:
        if name in ds.coords:
            return name
    raise KeyError(f"None of {candidates} found in coords: {list(ds.coords)}")


def _get_heatwave_var(ds):
    """
    Return the heatwave mask variable name.
    Accepts any variable whose name contains 'heatwave' or 'mask'.
    """
    for var in ds.data_vars:
        if "heatwave" in var.lower() or "mask" in var.lower():
            return var
    raise KeyError(
        f"No heatwave/mask variable found. Available: {list(ds.data_vars)}"
    )


def _get_wind_var(ds):
    """Return the sfcWind variable name, preferring the exact name 'sfcWind'."""
    if "sfcWind" in ds.data_vars:
        return "sfcWind"
    # Fall back to any variable whose name contains 'wind'
    for var in ds.data_vars:
        if "wind" in var.lower():
            return var
    raise KeyError(
        f"No wind variable found. Available: {list(ds.data_vars)}"
    )


def _check_spatial_match(ds_a, ds_b, lat_a, lon_a, lat_b, lon_b):
    """
    Verify that two datasets share the same lat/lon grids.
    Uses a tolerance of 1e-4 degrees to allow for floating-point rounding.
    Raises ValueError with a clear message on mismatch.
    """
    lats_a = ds_a[lat_a].values
    lats_b = ds_b[lat_b].values
    lons_a = ds_a[lon_a].values
    lons_b = ds_b[lon_b].values

    if lats_a.shape != lats_b.shape or not np.allclose(lats_a, lats_b, atol=1e-4):
        raise ValueError(
            f"Latitude mismatch: heatwave_mask has {lats_a.shape}, "
            f"sfcWind has {lats_b.shape}.\n"
            "Re-run 02_clip_china.py so both files are clipped to the same grid."
        )
    if lons_a.shape != lons_b.shape or not np.allclose(lons_a, lons_b, atol=1e-4):
        raise ValueError(
            f"Longitude mismatch: heatwave_mask has {lons_a.shape}, "
            f"sfcWind has {lons_b.shape}.\n"
            "Re-run 02_clip_china.py so both files are clipped to the same grid."
        )


def _align_time(da_mask, da_wind):
    """
    Align two DataArrays to their common time values.

    The heatwave mask contains only JJA days (from script 03), while sfcWind
    contains all days of the year.  Inner-join on time gives JJA days only.

    Returns (aligned_mask, aligned_wind, alignment_was_needed).
    """
    times_mask = da_mask["time"].values
    times_wind = da_wind["time"].values

    if np.array_equal(times_mask, times_wind):
        return da_mask, da_wind, False

    # Select the time steps present in the mask (JJA days) from the wind file.
    # xr.align with join='inner' would also work but sel is more explicit.
    common = np.intersect1d(times_mask, times_wind)
    if len(common) == 0:
        raise ValueError(
            "No overlapping time steps between heatwave_mask and sfcWind."
        )

    return da_mask.sel(time=common), da_wind.sel(time=common), True


# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------

def compute_compound_events(mask_path, wind_path):
    """
    Load inputs, align, apply compound condition, return outputs and stats.
    """
    ds_mask = xr.open_dataset(mask_path)
    ds_wind = xr.open_dataset(wind_path)

    # --- detect variable names ---
    mask_var = _get_heatwave_var(ds_mask)
    wind_var = _get_wind_var(ds_wind)

    # --- detect coordinate names ---
    lat_m = _get_coord(ds_mask, ["lat", "latitude"])
    lon_m = _get_coord(ds_mask, ["lon", "longitude"])
    lat_w = _get_coord(ds_wind, ["lat", "latitude"])
    lon_w = _get_coord(ds_wind, ["lon", "longitude"])

    # --- check wind units ---
    wind_units  = ds_wind[wind_var].attrs.get("units", "not specified")
    units_ok    = wind_units in ("m/s", "m s-1", "m s**-1")
    units_note  = (
        "OK" if units_ok
        else f"WARNING: expected m/s but found '{wind_units}' — verify before interpreting results"
    )

    # --- verify spatial grids match ---
    _check_spatial_match(ds_mask, ds_wind, lat_m, lon_m, lat_w, lon_w)

    # --- extract DataArrays ---
    da_mask = ds_mask[mask_var]
    da_wind = ds_wind[wind_var]

    # --- align time coordinates ---
    da_mask, da_wind, aligned = _align_time(da_mask, da_wind)

    time_range = (
        str(da_mask["time"].values[0])[:10],
        str(da_mask["time"].values[-1])[:10],
    )

    # --- compound event mask ---
    # Condition: heatwave day AND wind below threshold.
    # Broadcasting: da_mask and da_wind both have (time, lat, lon).
    hlw_mask = ((da_mask == 1) & (da_wind < WIND_THRESHOLD)).astype(np.int8)
    hlw_mask.attrs["long_name"]   = "Hot-low-wind compound event mask"
    hlw_mask.attrs["description"] = (
        f"1 = heatwave day (mask==1) with sfcWind < {WIND_THRESHOLD} m/s; 0 otherwise"
    )

    # --- per-grid-cell probability: hlw_days / heatwave_days ---
    n_hw_days  = da_mask.sum(dim="time")    # shape: (lat, lon)
    n_hlw_days = hlw_mask.sum(dim="time")   # shape: (lat, lon)

    # Where no heatwave days occurred at all, probability is undefined → NaN.
    probability = xr.where(n_hw_days > 0, n_hlw_days / n_hw_days, np.nan)
    probability.attrs["long_name"]   = "Hot-low-wind probability on heatwave days"
    probability.attrs["description"] = (
        "Fraction of heatwave days that are also low-wind days (sfcWind < 0.57 m/s)"
    )
    probability.attrs["units"] = "1"

    # --- summary statistics ---
    total_hw_cells  = int(da_mask.values.sum())
    total_hlw_cells = int(hlw_mask.values.sum())
    pct_hlw = 100.0 * total_hlw_cells / total_hw_cells if total_hw_cells > 0 else 0.0

    prob_vals = probability.values
    prob_vals_valid = prob_vals[~np.isnan(prob_vals)]

    stats = {
        "mask_file":        mask_path.name,
        "wind_file":        wind_path.name,
        "mask_var":         mask_var,
        "wind_var":         wind_var,
        "wind_units":       wind_units,
        "units_note":       units_note,
        "threshold":        WIND_THRESHOLD,
        "time_range":       time_range,
        "aligned":          aligned,
        "n_hw_cells":       total_hw_cells,
        "n_hlw_cells":      total_hlw_cells,
        "pct_hlw":          pct_hlw,
        "prob_min":         float(prob_vals_valid.min()) if len(prob_vals_valid) else np.nan,
        "prob_mean":        float(prob_vals_valid.mean()) if len(prob_vals_valid) else np.nan,
        "prob_max":         float(prob_vals_valid.max()) if len(prob_vals_valid) else np.nan,
    }

    ds_mask.close()
    ds_wind.close()
    return hlw_mask, probability, stats


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def format_report(stats):
    lines = []
    lines.append("Hot-low-wind compound event report")
    lines.append(f"Generated          : {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}")
    lines.append("")
    lines.append(f"Heatwave mask file : {stats['mask_file']}  (variable: {stats['mask_var']})")
    lines.append(f"Wind file          : {stats['wind_file']}  (variable: {stats['wind_var']})")
    lines.append(f"Wind units         : {stats['wind_units']}  [{stats['units_note']}]")
    lines.append(f"Wind threshold     : {stats['threshold']} m/s")
    lines.append("")
    lines.append(f"Time range         : {stats['time_range'][0]} → {stats['time_range'][1]}")
    lines.append(
        f"Time alignment     : {'needed and applied (JJA subset selected from sfcWind)' if stats['aligned'] else 'not needed'}"
    )
    lines.append("")
    lines.append(f"Heatwave grid-cell-days         : {stats['n_hw_cells']:,}")
    lines.append(f"Hot-low-wind grid-cell-days     : {stats['n_hlw_cells']:,}")
    lines.append(f"Hot-low-wind % of heatwave days : {stats['pct_hlw']:.2f}%")
    lines.append("")
    lines.append("Hot-low-wind probability per grid cell:")
    lines.append(f"  min  : {stats['prob_min']:.4f}")
    lines.append(f"  mean : {stats['prob_mean']:.4f}")
    lines.append(f"  max  : {stats['prob_max']:.4f}")
    lines.append("")
    lines.append(f"Output mask file   : {HLW_MASK_FILE.name}")
    lines.append(f"Output prob file   : {HLW_PROB_FILE.name}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    for path in (MASK_INPUT_FILE, WIND_INPUT_FILE):
        if not path.exists():
            raise FileNotFoundError(
                f"Input file not found: {path}\n"
                "Run 02_clip_china.py and 03_detect_future_heatwave.py first."
            )

    print("Computing hot-low-wind compound events ...")
    hlw_mask, probability, stats = compute_compound_events(
        MASK_INPUT_FILE, WIND_INPUT_FILE
    )

    print("Saving hot-low-wind mask ...")
    hlw_mask.rename("hot_low_wind_mask").to_dataset().to_netcdf(HLW_MASK_FILE)

    print("Saving hot-low-wind probability ...")
    probability.rename("hot_low_wind_probability").to_dataset().to_netcdf(HLW_PROB_FILE)

    report = format_report(stats)
    print("\n" + report)

    REPORT_FILE.write_text(report, encoding="utf-8")
    print(f"\nReport saved → {REPORT_FILE.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
