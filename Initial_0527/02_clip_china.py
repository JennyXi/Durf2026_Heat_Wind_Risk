"""
02_clip_china.py

Clip global CMIP6 NetCDF files to a China bounding box.

Inputs:
  data/raw/tasmax_ssp245_2041_2060.nc
  data/raw/sfcWind_ssp245_2041_2060.nc

Outputs:
  data/processed/tasmax_ssp245_2041_2060_china.nc
  data/processed/sfcWind_ssp245_2041_2060_china.nc
  outputs/clip_china_report.txt
"""

import datetime
from pathlib import Path

import xarray as xr

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT      = Path(__file__).parent.parent
RAW       = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
OUT       = ROOT / "outputs"

PROCESSED.mkdir(parents=True, exist_ok=True)
OUT.mkdir(parents=True, exist_ok=True)

FILE_PAIRS = [
    (RAW / "tasmax_ssp245_2041_2060.nc",  PROCESSED / "tasmax_ssp245_2041_2060_china.nc"),
    (RAW / "sfcWind_ssp245_2041_2060.nc", PROCESSED / "sfcWind_ssp245_2041_2060_china.nc"),
]

REPORT_FILE = OUT / "clip_china_report.txt"

# China bounding box (degrees)
LAT_MIN, LAT_MAX = 18.0, 54.0
LON_MIN, LON_MAX = 73.0, 136.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_coord(ds, candidates):
    """Return the first coordinate name from candidates found in ds."""
    for name in candidates:
        if name in ds.coords:
            return name
    raise KeyError(
        f"None of {candidates} found in coords: {list(ds.coords)}"
    )


def clip_file(input_path, output_path):
    """
    Open one CMIP6 file, normalise longitudes to −180–180 if needed,
    clip to the China bounding box, and write to output_path.

    Returns a stats dict used to build the report.
    """
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    ds = xr.open_dataset(input_path)

    lat_name = _get_coord(ds, ["lat", "latitude"])
    lon_name = _get_coord(ds, ["lon", "longitude"])

    lats = ds[lat_name].values
    lons = ds[lon_name].values

    orig_lat_range = (float(lats.min()), float(lats.max()))
    orig_lon_range = (float(lons.min()), float(lons.max()))
    orig_shape     = dict(ds.dims)

    # Convert 0–360 → −180–180 so the bounding box slice is unambiguous
    # and downstream scripts can always assume −180–180 convention.
    if float(lons.max()) > 180:
        ds = ds.assign_coords(
            {lon_name: ((ds[lon_name] + 180) % 360) - 180}
        )

    # Sort both axes so slice(min, max) always works, regardless of
    # whether the file stores latitude top-to-bottom or lons were reordered.
    ds = ds.sortby(lat_name).sortby(lon_name)

    # Bounding-box clip
    ds_china = ds.sel(
        {
            lat_name: slice(LAT_MIN, LAT_MAX),
            lon_name: slice(LON_MIN, LON_MAX),
        }
    )

    clipped_lats = ds_china[lat_name].values
    clipped_lons = ds_china[lon_name].values

    ds_china.to_netcdf(output_path)

    stats = {
        "input":             input_path.name,
        "output":            output_path.name,
        "orig_lat_range":    orig_lat_range,
        "orig_lon_range":    orig_lon_range,
        "orig_shape":        orig_shape,
        "clipped_lat_range": (float(clipped_lats.min()), float(clipped_lats.max())),
        "clipped_lon_range": (float(clipped_lons.min()), float(clipped_lons.max())),
        "clipped_shape":     dict(ds_china.dims),
        "variables":         list(ds_china.data_vars),
    }

    ds.close()
    ds_china.close()
    return stats


def format_report_block(stats):
    """Format one file's stats as a readable text block."""
    lines = []
    lines.append(f"  Input file  : {stats['input']}")
    lines.append(f"  Output file : {stats['output']}")
    lines.append(
        f"  Lat range   : {stats['orig_lat_range'][0]:.2f}–{stats['orig_lat_range'][1]:.2f}"
        f"  →  {stats['clipped_lat_range'][0]:.2f}–{stats['clipped_lat_range'][1]:.2f}"
    )
    lines.append(
        f"  Lon range   : {stats['orig_lon_range'][0]:.2f}–{stats['orig_lon_range'][1]:.2f}"
        f"  →  {stats['clipped_lon_range'][0]:.2f}–{stats['clipped_lon_range'][1]:.2f}"
    )
    lines.append(
        f"  Shape       : {stats['orig_shape']}"
        f"  →  {stats['clipped_shape']}"
    )
    lines.append(f"  Variables   : {stats['variables']}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    report_lines = []
    report_lines.append("Clip-to-China report")
    report_lines.append(
        f"Generated   : {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}"
    )
    report_lines.append(
        f"Bounding box: lat {LAT_MIN}–{LAT_MAX}N, lon {LON_MIN}–{LON_MAX}E"
    )
    report_lines.append("")

    for input_path, output_path in FILE_PAIRS:
        print(f"Clipping {input_path.name} ...")
        try:
            stats = clip_file(input_path, output_path)
            report_lines.append(f"[OK] {stats['input']}")
            report_lines.append(format_report_block(stats))
            print(f"  → saved {output_path.name}")
        except FileNotFoundError as e:
            msg = f"[SKIP] {e}"
            print(f"  {msg}")
            report_lines.append(msg)
        except Exception as e:
            msg = f"[ERROR] {input_path.name}: {e}"
            print(f"  {msg}")
            report_lines.append(msg)
        report_lines.append("")

    report = "\n".join(report_lines)
    REPORT_FILE.write_text(report, encoding="utf-8")
    print(f"\nReport saved → {REPORT_FILE.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
