"""
07_make_risk_category_map.py

Classify Chinese prefecture-level cities into hot-low-wind risk categories
and produce a categorical choropleth map.

Risk categories (by mean_probability = hot-low-wind days / heatwave days):
  Low        : 0.00 <= p < 0.25
  Moderate   : 0.25 <= p < 0.50
  High       : 0.50 <= p < 0.75
  Very High  : 0.75 <= p <= 1.00
  No data    : NaN  (no grid-cell centres fell inside the city polygon)

Input:
  outputs/city_hot_low_wind_probability_ssp245_2041_2060.gpkg

Outputs:
  outputs/maps/city_hot_low_wind_risk_category_map_ssp245_2041_2060.png
  outputs/maps/city_hot_low_wind_risk_category_map_ssp245_2041_2060.pdf
  outputs/city_hot_low_wind_risk_categories_ssp245_2041_2060.csv
  outputs/city_hot_low_wind_risk_categories_ssp245_2041_2060.gpkg
  outputs/risk_category_map_report.txt
"""

import datetime
from pathlib import Path

import geopandas as gpd
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT     = Path(__file__).parent.parent
OUT      = ROOT / "outputs"
MAPS_DIR = OUT / "maps"

MAPS_DIR.mkdir(parents=True, exist_ok=True)

INPUT_FILE  = OUT / "city_hot_low_wind_probability_ssp245_2041_2060.gpkg"
OUT_PNG     = MAPS_DIR / "city_hot_low_wind_risk_category_map_ssp245_2041_2060.png"
OUT_PDF     = MAPS_DIR / "city_hot_low_wind_risk_category_map_ssp245_2041_2060.pdf"
OUT_CSV     = OUT / "city_hot_low_wind_risk_categories_ssp245_2041_2060.csv"
OUT_GPKG    = OUT / "city_hot_low_wind_risk_categories_ssp245_2041_2060.gpkg"
REPORT_FILE = OUT / "risk_category_map_report.txt"

TITLE    = "Hot-Low-Wind Risk Category During Future Heatwave Days"
SUBTITLE = "CMIP6 SSP2-4.5, 2041–2060, future-relative heatwave threshold"

# Category order, bin edges, and colors — defined once, used everywhere.
CATEGORY_ORDER = ["Low", "Moderate", "High", "Very High", "No data"]

# Bin edges use right=False so each interval is [left, right).
# Upper edge 1.001 ensures probability == 1.00 falls into "Very High".
BINS   = [0.0, 0.25, 0.50, 0.75, 1.001]
LABELS = ["Low", "Moderate", "High", "Very High"]

CATEGORY_COLORS = {
    "Low":       "#1a9641",   # green
    "Moderate":  "#fdae61",   # yellow-orange
    "High":      "#f46d43",   # orange
    "Very High": "#d73027",   # red
    "No data":   "#d0d0d0",   # neutral gray
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_prob_col(gdf):
    """Prefer 'mean_probability'; fall back to first column with 'probability'."""
    if "mean_probability" in gdf.columns:
        return "mean_probability"
    for col in gdf.columns:
        if "probability" in col.lower():
            return col
    raise KeyError(
        f"No probability column found. Available columns: {list(gdf.columns)}"
    )


def assign_risk_categories(gdf, prob_col):
    """
    Add a 'risk_category' column to gdf using pd.cut with the fixed bins.
    NaN probability → 'No data'.
    Returns gdf with the new column (modifies in place).
    """
    # pd.cut returns NaN for values outside the bins and for input NaN.
    gdf["risk_category"] = pd.cut(
        gdf[prob_col],
        bins=BINS,
        labels=LABELS,
        right=False,
    ).astype(str)   # convert Categorical → str so NaN becomes the string 'nan'

    # Replace 'nan' (from NaN inputs) with the display label 'No data'
    gdf.loc[gdf[prob_col].isna(), "risk_category"] = "No data"
    # Restore any out-of-range values (edge case: prob < 0 or prob > 1.001)
    gdf.loc[gdf["risk_category"] == "nan", "risk_category"] = "No data"

    return gdf


# ---------------------------------------------------------------------------
# Map
# ---------------------------------------------------------------------------

def make_map(gdf):
    """
    Draw a categorical choropleth by plotting each risk category as a
    separate layer.  This gives full control over colour and legend order.
    """
    fig, ax = plt.subplots(figsize=(12, 8))

    # Plot each category in defined order so the legend matches
    for cat in CATEGORY_ORDER:
        subset = gdf[gdf["risk_category"] == cat]
        if subset.empty:
            continue
        subset.plot(
            ax=ax,
            color=CATEGORY_COLORS[cat],
            linewidth=0.2,
            edgecolor="white",
        )

    # --- legend: only show categories that appear in the data ---
    present_cats = [c for c in CATEGORY_ORDER if c in gdf["risk_category"].values]
    legend_handles = [
        mpatches.Patch(color=CATEGORY_COLORS[cat], label=cat)
        for cat in present_cats
    ]
    ax.legend(
        handles=legend_handles,
        title="Risk Category",
        title_fontsize=9,
        fontsize=8,
        loc="lower left",
        framealpha=0.8,
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

    ax.set_axis_off()
    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def format_report(stats):
    lines = []
    lines.append("Risk category map report")
    lines.append(f"Generated          : {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}")
    lines.append("")
    lines.append(f"Input file         : {stats['input_file']}")
    lines.append(f"Probability column : {stats['prob_col']}")
    lines.append(f"City polygons      : {stats['n_cities']:,}")
    lines.append("")
    lines.append("Cities per risk category:")
    for cat in CATEGORY_ORDER:
        count = stats["category_counts"].get(cat, 0)
        lines.append(f"  {cat:<12}: {count:,}")
    lines.append("")
    lines.append("Mean probability across cities with data:")
    lines.append(f"  min  : {stats['prob_min']:.4f}")
    lines.append(f"  mean : {stats['prob_mean']:.4f}")
    lines.append(f"  max  : {stats['prob_max']:.4f}")
    lines.append("")
    lines.append(f"Output PNG  : {OUT_PNG.name}")
    lines.append(f"Output PDF  : {OUT_PDF.name}")
    lines.append(f"Output CSV  : {OUT_CSV.name}")
    lines.append(f"Output GPKG : {OUT_GPKG.name}")
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
        print("  WARNING: no CRS — assuming EPSG:4326")
        gdf = gdf.set_crs("EPSG:4326")
    elif gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs("EPSG:4326")

    # --- detect probability column ---
    prob_col = _get_prob_col(gdf)
    print(f"  Using probability column: {prob_col}")

    # --- assign risk categories ---
    gdf = assign_risk_categories(gdf, prob_col)
    print("  Category counts:")
    for cat in CATEGORY_ORDER:
        count = (gdf["risk_category"] == cat).sum()
        print(f"    {cat}: {count}")

    # --- statistics ---
    valid_vals = gdf[prob_col].dropna()
    stats = {
        "input_file":      INPUT_FILE.name,
        "prob_col":        prob_col,
        "n_cities":        len(gdf),
        "category_counts": gdf["risk_category"].value_counts().to_dict(),
        "prob_min":        float(valid_vals.min()),
        "prob_mean":       float(valid_vals.mean()),
        "prob_max":        float(valid_vals.max()),
    }

    # --- save enriched data ---
    print(f"Saving CSV → {OUT_CSV.name} ...")
    gdf.drop(columns="geometry").to_csv(OUT_CSV, index=False, encoding="utf-8")

    print(f"Saving GeoPackage → {OUT_GPKG.name} ...")
    gdf.to_file(OUT_GPKG, driver="GPKG")

    # --- draw and save map ---
    print("Drawing map ...")
    fig = make_map(gdf)

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
