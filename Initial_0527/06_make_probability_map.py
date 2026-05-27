"""
06_make_probability_map.py

Choropleth map of hot-low-wind probability across Chinese prefecture-level
cities during future heatwave days (CMIP6 SSP2-4.5, 2041–2060).

Input:
  outputs/city_hot_low_wind_probability_ssp245_2041_2060.gpkg

Outputs:
  outputs/maps/city_hot_low_wind_probability_map_ssp245_2041_2060.png
  outputs/maps/city_hot_low_wind_probability_map_ssp245_2041_2060.pdf
  outputs/map_probability_report.txt
"""

import datetime
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT     = Path(__file__).parent.parent
OUT      = ROOT / "outputs"
MAPS_DIR = OUT / "maps"

MAPS_DIR.mkdir(parents=True, exist_ok=True)

INPUT_FILE  = OUT / "city_hot_low_wind_probability_ssp245_2041_2060.gpkg"
OUT_PNG     = MAPS_DIR / "city_hot_low_wind_probability_map_ssp245_2041_2060.png"
OUT_PDF     = MAPS_DIR / "city_hot_low_wind_probability_map_ssp245_2041_2060.pdf"
REPORT_FILE = OUT / "map_probability_report.txt"

TITLE    = "Hot-Low-Wind Probability During Future Heatwave Days"
SUBTITLE = "CMIP6 SSP2-4.5, 2041–2060, future-relative heatwave threshold"

# Color for cities with no data (no grid-cell centres fell inside the polygon)
NO_DATA_COLOR = "#d0d0d0"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_prob_col(gdf):
    """
    Return the probability column to map.
    Prefers 'mean_probability'; falls back to the first column containing
    'probability'.  Raises a descriptive error if none is found.
    """
    if "mean_probability" in gdf.columns:
        return "mean_probability"
    for col in gdf.columns:
        if "probability" in col.lower():
            return col
    raise KeyError(
        f"No probability column found. Available columns: {list(gdf.columns)}"
    )


# ---------------------------------------------------------------------------
# Map
# ---------------------------------------------------------------------------

def make_map(gdf, prob_col):
    """
    Draw a choropleth of prob_col and return the figure.

    Cities with NaN probability are drawn in NO_DATA_COLOR so they are
    visible on the map without distorting the colorbar range.
    """
    fig, ax = plt.subplots(figsize=(12, 8))

    # Determine the valid value range for the colorbar
    valid_vals = gdf[prob_col].dropna()
    vmin = float(valid_vals.min())
    vmax = float(valid_vals.max())

    # --- plot cities with valid probability values ---
    gdf.plot(
        ax=ax,
        column=prob_col,
        cmap="YlOrRd",
        vmin=vmin,
        vmax=vmax,
        linewidth=0.2,
        edgecolor="white",
        legend=True,
        legend_kwds={
            "label":       "Probability",
            "orientation": "vertical",
            "shrink":      0.6,
            "pad":         0.02,
        },
        missing_kwds={
            "color": NO_DATA_COLOR,
            "label": "No data",
        },
    )

    # --- add a manual legend patch for the no-data colour ---
    no_data_patch = mpatches.Patch(color=NO_DATA_COLOR, label="No data")
    ax.legend(
        handles=[no_data_patch],
        loc="lower left",
        fontsize=8,
        framealpha=0.7,
    )

    # --- title and subtitle ---
    ax.set_title(TITLE, fontsize=13, fontweight="bold", pad=10)
    ax.text(
        0.5, -0.02,
        SUBTITLE,
        transform=ax.transAxes,
        ha="center",
        fontsize=9,
        color="#555555",
    )

    # --- hide axis frame, ticks, and coordinate labels ---
    ax.set_axis_off()

    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def format_report(stats):
    lines = []
    lines.append("Probability map report")
    lines.append(f"Generated          : {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}")
    lines.append("")
    lines.append(f"Input file         : {stats['input_file']}")
    lines.append(f"Probability column : {stats['prob_col']}")
    lines.append(f"City polygons      : {stats['n_cities']:,}")
    lines.append(f"Cities with data   : {stats['n_with_data']:,}")
    lines.append(f"Cities without data: {stats['n_no_data']:,}")
    lines.append("")
    lines.append("Mean probability across cities with data:")
    lines.append(f"  min  : {stats['prob_min']:.4f}")
    lines.append(f"  mean : {stats['prob_mean']:.4f}")
    lines.append(f"  max  : {stats['prob_max']:.4f}")
    lines.append("")
    lines.append(f"Output PNG : {OUT_PNG.name}")
    lines.append(f"Output PDF : {OUT_PDF.name}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Input file not found: {INPUT_FILE}\n"
            "Run 05_aggregate_to_city.py first."
        )

    # --- load ---
    print(f"Loading {INPUT_FILE.name} ...")
    gdf = gpd.read_file(INPUT_FILE)

    if "geometry" not in gdf.columns or gdf.geometry.isnull().all():
        raise ValueError("GeoPackage contains no valid geometries.")

    # --- CRS ---
    if gdf.crs is None:
        print("  WARNING: no CRS found — assuming EPSG:4326")
        gdf = gdf.set_crs("EPSG:4326")
    elif gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs("EPSG:4326")

    # --- detect probability column ---
    prob_col = _get_prob_col(gdf)
    print(f"  Mapping column: {prob_col}")

    # --- statistics for report ---
    valid_vals   = gdf[prob_col].dropna()
    n_with_data  = int(valid_vals.notna().sum()) if hasattr(valid_vals, 'notna') else len(valid_vals)
    n_no_data    = len(gdf) - n_with_data

    stats = {
        "input_file":  INPUT_FILE.name,
        "prob_col":    prob_col,
        "n_cities":    len(gdf),
        "n_with_data": n_with_data,
        "n_no_data":   n_no_data,
        "prob_min":    float(valid_vals.min()),
        "prob_mean":   float(valid_vals.mean()),
        "prob_max":    float(valid_vals.max()),
    }

    # --- draw map ---
    print("Drawing map ...")
    fig = make_map(gdf, prob_col)

    # --- save ---
    print(f"Saving PNG → {OUT_PNG.name} ...")
    fig.savefig(OUT_PNG, dpi=300, bbox_inches="tight")

    print(f"Saving PDF → {OUT_PDF.name} ...")
    fig.savefig(OUT_PDF, bbox_inches="tight")

    plt.close(fig)

    # --- report ---
    report = format_report(stats)
    print("\n" + report)
    REPORT_FILE.write_text(report, encoding="utf-8")
    print(f"\nReport saved → {REPORT_FILE.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
