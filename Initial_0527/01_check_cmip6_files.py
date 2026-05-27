"""
01_check_cmip6_files.py

Inspect CMIP6 NetCDF files before any processing:
  - data/raw/tasmax_ssp245_2041_2060.nc
  - data/raw/sfcWind_ssp245_2041_2060.nc

Prints a summary to stdout and saves it to outputs/file_check_report.txt.
"""

import datetime
from pathlib import Path

import numpy as np
import xarray as xr

# ---------------------------------------------------------------------------
# Paths  (script lives in scripts/, project root is one level up)
# ---------------------------------------------------------------------------
ROOT = Path(__file__).parent.parent
RAW  = ROOT / "data" / "raw"
OUT  = ROOT / "outputs"
OUT.mkdir(exist_ok=True)

TASMAX_FILE  = RAW / "tasmax_ssp245_2041_2060.nc"
SFCWIND_FILE = RAW / "sfcWind_ssp245_2041_2060.nc"
REPORT_FILE  = OUT / "file_check_report.txt"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_coord(ds, candidates):
    """Return the first coordinate name from candidates that exists in ds."""
    for name in candidates:
        if name in ds.coords:
            return name
    raise KeyError(f"None of {candidates} found in dataset coords: {list(ds.coords)}")


def describe_dataset(ds, var_name, filename):
    """Return a formatted summary string for one dataset."""
    lines = []
    lines.append(f"=== {filename} ===")
    lines.append(f"  Variables  : {list(ds.data_vars)}")
    lines.append(f"  Dimensions : {dict(ds.dims)}")
    lines.append(f"  Coords     : {list(ds.coords)}")

    # Time range
    time = ds["time"]
    t0   = str(time.values[0])[:10]
    t1   = str(time.values[-1])[:10]
    lines.append(f"  Time range : {t0} → {t1}  ({len(time)} steps)")

    # Spatial resolution — use whichever coordinate name the file uses
    lat_name = _get_coord(ds, ["lat", "latitude"])
    lon_name = _get_coord(ds, ["lon", "longitude"])
    lats = ds[lat_name].values
    lons = ds[lon_name].values
    lat_res = round(float(np.diff(lats).mean()), 4)
    lon_res = round(float(np.diff(lons).mean()), 4)
    lines.append(f"  Resolution : {lat_res}° lat × {lon_res}° lon")

    # Longitude convention: 0–360 or −180–180
    lon_min, lon_max = float(lons.min()), float(lons.max())
    lon_fmt = "0–360" if lon_max > 180 else "−180–180"
    lines.append(f"  Lon range  : {lon_min:.2f} to {lon_max:.2f}  ({lon_fmt})")

    # Variable units and long name
    if var_name in ds:
        attrs     = ds[var_name].attrs
        units     = attrs.get("units", "not specified")
        long_name = attrs.get("long_name", "")
        lines.append(f"  Units      : {units}  ({long_name})")
    else:
        lines.append(f"  WARNING    : variable '{var_name}' not found in file")

    return "\n".join(lines)


def check_time_alignment(ds_a, ds_b, label_a, label_b):
    """Check whether the two datasets share identical time coordinates."""
    lines = []
    lines.append("=== Time alignment check ===")

    time_a = ds_a["time"].values
    time_b = ds_b["time"].values

    if len(time_a) != len(time_b):
        lines.append(
            f"  MISMATCH  : {label_a} has {len(time_a)} steps, "
            f"{label_b} has {len(time_b)} steps"
        )
        return "\n".join(lines)

    if (time_a == time_b).all():
        lines.append(f"  OK        : both files share {len(time_a)} identical time steps")
    else:
        n_diff = int((time_a != time_b).sum())
        lines.append(f"  MISMATCH  : {n_diff} of {len(time_a)} time steps differ")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    report_lines = []
    report_lines.append("CMIP6 file check report")
    report_lines.append(f"Generated : {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}")
    report_lines.append("")

    print(f"Opening {TASMAX_FILE.name} ...")
    ds_tasmax  = xr.open_dataset(TASMAX_FILE)

    print(f"Opening {SFCWIND_FILE.name} ...")
    ds_sfcwind = xr.open_dataset(SFCWIND_FILE)

    report_lines.append(describe_dataset(ds_tasmax,  "tasmax",  TASMAX_FILE.name))
    report_lines.append("")
    report_lines.append(describe_dataset(ds_sfcwind, "sfcWind", SFCWIND_FILE.name))
    report_lines.append("")
    report_lines.append(check_time_alignment(ds_tasmax, ds_sfcwind, "tasmax", "sfcWind"))

    report = "\n".join(report_lines)
    print("\n" + report)

    REPORT_FILE.write_text(report, encoding="utf-8")
    print(f"\nReport saved → {REPORT_FILE.relative_to(ROOT)}")

    ds_tasmax.close()
    ds_sfcwind.close()


if __name__ == "__main__":
    main()
