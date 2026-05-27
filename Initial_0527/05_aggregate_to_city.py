"""
05_aggregate_to_city.py

Aggregate CMIP6 grid-cell hot-low-wind probability to Chinese
prefecture-level city polygons.

Inputs:
  data/processed/hot_low_wind_probability_ssp245_2041_2060_china.nc
  data/shapefiles/china_prefecture_cities.gpkg

Outputs:
  outputs/city_hot_low_wind_probability_ssp245_2041_2060.csv
  outputs/city_hot_low_wind_probability_ssp245_2041_2060.gpkg
  outputs/city_aggregation_report.txt
"""

import datetime
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr
from shapely.geometry import Point

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT      = Path(__file__).parent.parent
PROCESSED = ROOT / "data" / "processed"
SHAPES    = ROOT / "data" / "shapefiles"
OUT       = ROOT / "outputs"

OUT.mkdir(parents=True, exist_ok=True)

NC_FILE      = PROCESSED / "hot_low_wind_probability_ssp245_2041_2060_china.nc"
CITIES_FILE  = SHAPES    / "china_prefecture_cities.gpkg"

OUT_CSV      = OUT / "city_hot_low_wind_probability_ssp245_2041_2060.csv"
OUT_GPKG     = OUT / "city_hot_low_wind_probability_ssp245_2041_2060.gpkg"
REPORT_FILE  = OUT / "city_aggregation_report.txt"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_coord(ds, candidates):
    """Return the first coordinate name from candidates found in ds."""
    for name in candidates:
        if name in ds.coords:
            return name
    raise KeyError(f"None of {candidates} found in coords: {list(ds.coords)}")


def _get_prob_var(ds):
    """
    Return the probability variable name.
    Prefers any variable whose name contains 'probability'.
    """
    for var in ds.data_vars:
        if "probability" in var.lower():
            return var
    raise KeyError(
        f"No probability variable found. Available: {list(ds.data_vars)}"
    )


# ---------------------------------------------------------------------------
# Load inputs
# ---------------------------------------------------------------------------

def load_probability_grid(nc_path):
    """
    Open the probability NetCDF and return a GeoDataFrame of valid grid-cell
    centres (lon, lat, probability), with CRS EPSG:4326.

    NaN cells (grid cells with no heatwave days) are dropped — they carry
    no information and would distort city-level means.
    """
    ds       = xr.open_dataset(nc_path)
    lat_name = _get_coord(ds, ["lat", "latitude"])
    lon_name = _get_coord(ds, ["lon", "longitude"])
    prob_var = _get_prob_var(ds)

    prob_da = ds[prob_var]   # shape: (lat, lon)
    lats    = ds[lat_name].values
    lons    = ds[lon_name].values

    # Build flat arrays with meshgrid (indexing='ij' → first axis is lat)
    lats_2d, lons_2d = np.meshgrid(lats, lons, indexing="ij")
    prob_2d = prob_da.values   # (lat, lon) numpy array

    flat_lats = lats_2d.ravel()
    flat_lons = lons_2d.ravel()
    flat_prob = prob_2d.ravel()

    # Drop NaN cells
    valid     = ~np.isnan(flat_prob)
    flat_lats = flat_lats[valid]
    flat_lons = flat_lons[valid]
    flat_prob = flat_prob[valid]

    geometry = [Point(lon, lat) for lon, lat in zip(flat_lons, flat_lats)]

    gdf = gpd.GeoDataFrame(
        {"lat": flat_lats, "lon": flat_lons, "probability": flat_prob},
        geometry=geometry,
        crs="EPSG:4326",
    )

    ds.close()
    return gdf, prob_var


def load_city_polygons(gpkg_path):
    """
    Open city polygons and reproject to EPSG:4326 if needed.
    Returns (cities_gdf, reprojected_flag).
    """
    cities = gpd.read_file(gpkg_path)

    reprojected = False
    if cities.crs is None:
        print("  WARNING: city shapefile has no CRS — assuming EPSG:4326")
        cities = cities.set_crs("EPSG:4326")
    elif cities.crs.to_epsg() != 4326:
        cities      = cities.to_crs("EPSG:4326")
        reprojected = True

    return cities, reprojected


# ---------------------------------------------------------------------------
# Aggregate
# ---------------------------------------------------------------------------

def aggregate_to_cities(points_gdf, cities_gdf):
    """
    Spatially join grid-cell points to city polygons, then aggregate
    probability values per city.

    Strategy:
      1. sjoin on city geometry only → get index_right (city row index) per point
      2. Group by index_right, compute stats
      3. Join stats back onto cities_gdf (preserves all city attributes + geometry)

    Returns a GeoDataFrame with one row per city.
    """
    # Use only the geometry column from cities for the join to avoid column conflicts.
    cities_geom = cities_gdf[["geometry"]]

    joined = gpd.sjoin(
        points_gdf[["probability", "geometry"]],
        cities_geom,
        how="left",
        predicate="within",
    )

    # Matched points have a valid index_right; unmatched (border/ocean cells) do not.
    matched = joined.dropna(subset=["index_right"])
    matched = matched.copy()
    matched["index_right"] = matched["index_right"].astype(int)

    # Aggregate probability statistics per city (identified by its row index)
    city_agg = (
        matched.groupby("index_right")["probability"]
        .agg(["mean", "median", "min", "max", "count"])
        .rename(
            columns={
                "mean":   "mean_probability",
                "median": "median_probability",
                "min":    "min_probability",
                "max":    "max_probability",
                "count":  "number_of_grid_points",
            }
        )
    )

    # Join aggregated stats back to cities_gdf, keeping all city attribute columns.
    # Left join ensures every city row is preserved (unmatched cities get NaN stats).
    result = cities_gdf.join(city_agg, how="left")

    return result


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def format_report(stats):
    lines = []
    lines.append("City aggregation report")
    lines.append(f"Generated             : {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}")
    lines.append("")
    lines.append(f"Probability file      : {stats['nc_file']}")
    lines.append(f"Probability variable  : {stats['prob_var']}")
    lines.append(f"City shapefile        : {stats['cities_file']}")
    lines.append(
        f"CRS reprojection      : {'yes → EPSG:4326' if stats['reprojected'] else 'not needed'}"
    )
    lines.append("")
    lines.append(f"Grid points (non-NaN) : {stats['n_grid_points']:,}")
    lines.append(f"City polygons         : {stats['n_cities']:,}")
    lines.append(f"Matched cities        : {stats['n_matched']:,}")
    lines.append(f"Unmatched cities      : {stats['n_unmatched']:,}  (no grid cell centres fall inside)")
    lines.append("")
    lines.append("Mean probability across matched cities:")
    lines.append(f"  min  : {stats['prob_min']:.4f}")
    lines.append(f"  mean : {stats['prob_mean']:.4f}")
    lines.append(f"  max  : {stats['prob_max']:.4f}")
    lines.append("")
    lines.append(f"Output CSV   : {OUT_CSV.name}")
    lines.append(f"Output GPKG  : {OUT_GPKG.name}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    for path in (NC_FILE, CITIES_FILE):
        if not path.exists():
            raise FileNotFoundError(
                f"Input file not found: {path}\n"
                "Check that earlier scripts have been run and the shapefile is in place."
            )

    # --- load probability grid ---
    print(f"Loading probability grid from {NC_FILE.name} ...")
    points_gdf, prob_var = load_probability_grid(NC_FILE)
    print(f"  {len(points_gdf):,} valid grid points")

    # --- load city polygons ---
    print(f"Loading city polygons from {CITIES_FILE.name} ...")
    cities_gdf, reprojected = load_city_polygons(CITIES_FILE)
    print(f"  {len(cities_gdf):,} city polygons  (CRS: {cities_gdf.crs.to_epsg()})")

    # --- aggregate ---
    print("Aggregating probability to city polygons ...")
    result_gdf = aggregate_to_cities(points_gdf, cities_gdf)

    # --- statistics for report ---
    matched_mask  = result_gdf["number_of_grid_points"].notna()
    n_matched     = int(matched_mask.sum())
    n_unmatched   = len(result_gdf) - n_matched
    mean_probs    = result_gdf.loc[matched_mask, "mean_probability"]

    stats = {
        "nc_file":       NC_FILE.name,
        "prob_var":      prob_var,
        "cities_file":   CITIES_FILE.name,
        "reprojected":   reprojected,
        "n_grid_points": len(points_gdf),
        "n_cities":      len(cities_gdf),
        "n_matched":     n_matched,
        "n_unmatched":   n_unmatched,
        "prob_min":      float(mean_probs.min()),
        "prob_mean":     float(mean_probs.mean()),
        "prob_max":      float(mean_probs.max()),
    }

    # --- save CSV (no geometry column) ---
    print(f"Saving CSV → {OUT_CSV.name} ...")
    result_gdf.drop(columns="geometry").to_csv(OUT_CSV, index=False, encoding="utf-8")

    # --- save GeoPackage (with geometry) ---
    print(f"Saving GeoPackage → {OUT_GPKG.name} ...")
    result_gdf.to_file(OUT_GPKG, driver="GPKG")

    # --- report ---
    report = format_report(stats)
    print("\n" + report)
    REPORT_FILE.write_text(report, encoding="utf-8")
    print(f"\nReport saved → {REPORT_FILE.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
