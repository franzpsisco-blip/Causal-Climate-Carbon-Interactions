from __future__ import annotations

import fiona
import rasterio
import warnings
import numpy as np
import pandas as pd
import xarray as xr
import geopandas as gpd
import matplotlib as mpl
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Tuple, Optional, List
from rasterio.transform import Affine
from rasterio.features import geometry_mask
from rasterio.warp import transform_geom, transform_bounds
from matplotlib.colors import TwoSlopeNorm
import matplotlib.patches as mpatches
from mpl_toolkits.axes_grid1 import make_axes_locatable
from shapely.geometry import Polygon, MultiPolygon
from shapely.ops import unary_union

try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    CARTOPY_AVAILABLE = True
except Exception:
    CARTOPY_AVAILABLE = False

warnings.filterwarnings("ignore", category=UserWarning)

# =============================================================================
# PATHS
# =============================================================================
BASE_DIR = Path(r"C:\Users\Zhou laoshi\OneDrive\Desktop\Data Amazonas-Resampled")
OUT_DIR = Path(r"C:\Users\Zhou laoshi\OneDrive\Desktop\Data Amazonas-Resampled\Graficas\0_Anomalias")
OUT_DIR.mkdir(parents=True, exist_ok=True)
AMAZON_SHP = Path(r"C:\Users\Zhou laoshi\OneDrive\Desktop\Data Amazonas-Resampled\amazonia_boundary_proposal_Eva_2005\amazonia_polygons.shp")

START, END = "2005-01", "2024-12"

# =============================================================================
# INPUT FILES
# =============================================================================
VARIABLE_FILES: Dict[str, str] = {
    "AET":       "AET.tif",
    "EVI":       "EVI.tif",
    "GPP":       "GPP.tif",
    "LST":       "LST.tif",
    "NEE":       "NEE.tif",
    "PAR":       "PAR.tif",
    "PDSI":      "PDSI.tif",
    "PREC":      "PRECIP.tif",
    "RECO":      "RECO.tif",
    "RUNOFF":    "RUNOFF.tif",
    "SIF":       "SIF.tif",
    "SM":        "SOIL_MOISTURE.tif",
    "TWSA":      "TWSA.tif",
    "VPD":       "VPD.tif",
    "BA":        "BA.tif",
    "XCH4":      "XCH4.tif",
    "XCO2":      "XCO2.tif",
    "AOD550":    "AOD550.tif",
}

# =============================================================================
# GROUPS / DOMAINS
# =============================================================================
GROUPS = {
    "Climate":        ["PREC", "PDSI", "VPD"],
    "Radiation":      ["LST", "PAR"],
    "Hydrological":   ["AET", "SM", "TWSA", "RUNOFF"],
    "Carbon-Veg":     ["EVI", "SIF", "GPP", "RECO", "NEE"],
    "Atmosphere":     ["XCH4", "XCO2", "AOD550"],
    "Fire":           ["BA"],
}

# =============================================================================
# OUTPUTS
# =============================================================================
OUTPNG_1 = OUT_DIR / "Figure_1_Maps_Anomalies_Amazon_2005-2024.png"
OUTPDF_1 = OUT_DIR / "Figure_1_Maps_Anomalies_Amazon_2005-2024.pdf"

OUTPNG_2 = OUT_DIR / "Figure_2_Correlation_Lag_Amazon_2005-2024.png"
OUTPDF_2 = OUT_DIR / "Figure_2_Correlation_Lag_Amazon_2005-2024.pdf"

# =============================================================================
# SETTINGS
# =============================================================================
SMOOTH_WINDOW = 5
MAX_LAG_MONTHS = 6
ANNOT_R_THRESH = 0.18

FONT_SCALE = 1.35
BASE_FONT   = int(12 * FONT_SCALE)
LABEL_FONT  = int(12 * FONT_SCALE)
TICK_FONT   = int(10.5 * FONT_SCALE)
LEGEND_FONT = int(11 * FONT_SCALE)
DPI = 450

PALETTE_CLIMATE      = ["#4a6fa5", "#6b8ec1", "#9bb7db"]
PALETTE_RADIATION    = ["#b2182b", "#ef8a62"]
PALETTE_HYDROLOGICAL = ["#08306b", "#2171b5", "#4292c6", "#6baed6"]
PALETTE_CARBON       = ["#0a5a2b", "#2e8b57", "#7fc97f", "#74c476", "#a1d99b"]
PALETTE_ATMOS        = ["#6a3d9a", "#8e7cc3", "#b39ddb"]
PALETTE_FIRE         = ["#2f2f2f"]

CMAP_CORR = mpl.colormaps.get_cmap("RdYlBu_r")
CMAP_MAP  = mpl.colormaps.get_cmap("RdBu_r")

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
    "font.size": BASE_FONT,
    "axes.labelsize": LABEL_FONT,
    "xtick.labelsize": TICK_FONT,
    "ytick.labelsize": TICK_FONT,
    "legend.fontsize": LEGEND_FONT,
    "figure.dpi": DPI,
    "savefig.dpi": DPI,
})

# =============================================================================
# HELPERS
# =============================================================================
def xy_edges_from_transform(w: int, h: int, tr: Affine) -> Tuple[np.ndarray, np.ndarray]:
    a = tr.a
    e = tr.e
    c = tr.c
    f = tr.f
    xc = c + a * (np.arange(w) + 0.5)
    yc = f + e * (np.arange(h) + 0.5)
    dx = np.diff(xc).mean() if w > 1 else 360.0 / max(w, 1)
    dy = np.diff(yc).mean() if h > 1 else 180.0 / max(h, 1)
    xe = np.r_[xc[0] - dx / 2, xc + dx / 2]
    ye = np.r_[yc[0] - dy / 2, yc + dy / 2]
    return xe, ye


def build_aoi_mask(aoi_path: Path, out_shape: Tuple[int, int], transform: Affine, raster_crs) -> np.ndarray:
    with fiona.open(str(aoi_path)) as src:
        if len(src) == 0:
            raise RuntimeError("AOI shapefile has no features.")
        shp_crs = src.crs_wkt or src.crs
        if shp_crs is None:
            raise RuntimeError("AOI shapefile has no CRS.")

        geoms = []
        for feat in src:
            geom = feat.get("geometry")
            if not geom:
                continue
            if str(shp_crs) != str(raster_crs):
                geom = transform_geom(shp_crs, raster_crs, geom, precision=6)
            geoms.append(geom)

    if not geoms:
        raise RuntimeError("AOI shapefile has no valid geometries.")

    mask = geometry_mask(
        geometries=geoms,
        out_shape=out_shape,
        transform=transform,
        invert=True,
        all_touched=False,
    )
    return mask.astype(bool)


def compute_aoi_extent(aoi_path: Path, raster_crs, pad_deg: float = 2.0) -> Optional[Tuple[float, float, float, float]]:
    if aoi_path is None or (not aoi_path.exists()):
        return None
    with fiona.open(str(aoi_path)) as src:
        shp_crs = src.crs_wkt or src.crs
        if shp_crs is None:
            return None
        b = src.bounds
    if str(shp_crs) != str(raster_crs):
        minx, miny, maxx, maxy = transform_bounds(shp_crs, raster_crs, *b, densify_pts=21)
    else:
        minx, miny, maxx, maxy = b
    return (minx - pad_deg, maxx + pad_deg, miny - pad_deg, maxy + pad_deg)


def add_map_frame(ax, lw=1.0, color="#1f1f1f", pad=0.008, shadow=True):
    if shadow:
        shadow_rect = mpatches.Rectangle(
            (pad + 0.006, pad - 0.004),
            1 - 2 * pad,
            1 - 2 * pad,
            transform=ax.transAxes,
            fill=False,
            linewidth=lw + 1.6,
            edgecolor="black",
            alpha=0.12,
            zorder=49
        )
        ax.add_patch(shadow_rect)

    rect = mpatches.Rectangle(
        (pad, pad),
        1 - 2 * pad,
        1 - 2 * pad,
        transform=ax.transAxes,
        fill=False,
        linewidth=lw,
        edgecolor=color,
        zorder=50,
        joinstyle="round"
    )
    ax.add_patch(rect)


def load_multiband_tif(path: Path, start_hint: str, aoi_mask: Optional[np.ndarray]) -> Tuple[xr.DataArray, np.ndarray, np.ndarray]:
    with rasterio.open(str(path)) as src:
        w, h = src.width, src.height
        tr = src.transform
        crs = src.crs
        arr = src.read(masked=True).filled(np.nan).astype(np.float32)

        if aoi_mask is not None:
            if aoi_mask.shape != (h, w):
                raise RuntimeError("AOI mask shape does not match raster shape.")
            arr[:, ~aoi_mask] = np.nan

        t = pd.date_range(start_hint, periods=arr.shape[0], freq="MS")
        xe, ye = xy_edges_from_transform(w, h, tr)
        xs = (xe[:-1] + xe[1:]) / 2.0
        ys = (ye[:-1] + ye[1:]) / 2.0

        da = xr.DataArray(
            arr,
            dims=("time", "y", "x"),
            coords={"time": t, "y": ys, "x": xs},
            attrs={"crs": str(crs), "transform": tr},
        )
        return da, xe, ye


def monthly_climatology(da: xr.DataArray) -> xr.DataArray:
    return da.groupby("time.month").mean("time", skipna=True)


def anomalies_from_climatology(da: xr.DataArray, clim: xr.DataArray) -> xr.DataArray:
    return da.groupby("time.month") - clim


def find_lag_of_max_correlation(a: pd.Series, b: pd.Series, max_lag: int = 6) -> Tuple[int, float]:
    best_lag, best_r = 0, 0.0
    for lag in range(-max_lag, max_lag + 1):
        if lag < 0:
            a_seg, b_seg = a.iloc[-lag:].values, b.iloc[: len(a) - (-lag)].values
        elif lag > 0:
            a_seg, b_seg = a.iloc[:-lag].values, b.iloc[lag:].values
        else:
            a_seg, b_seg = a.values, b.values

        m = (~np.isnan(a_seg)) & (~np.isnan(b_seg))
        if m.sum() < 6:
            continue
        r = np.corrcoef(a_seg[m], b_seg[m])[0, 1]
        if np.isnan(r):
            continue
        if abs(r) > abs(best_r):
            best_r, best_lag = r, lag
    return best_lag, best_r


def r_map_sig(y_da: xr.DataArray, x_da: xr.DataArray, min_n: int = 10):

    # asegurar mismo orden
    y = y_da.transpose("time", "y", "x").values
    x = x_da.transpose("time", "y", "x").values

    if y.shape != x.shape:
        raise RuntimeError(f"Shape mismatch: {y.shape} vs {x.shape}")

    T, H, W = y.shape

    r_out = np.full((H, W), np.nan)

    for i in range(H):
        for j in range(W):

            y_ij = y[:, i, j]
            x_ij = x[:, i, j]

            mask = (~np.isnan(y_ij)) & (~np.isnan(x_ij))
            n = mask.sum()

            if n < min_n:
                continue

            r = np.corrcoef(y_ij[mask], x_ij[mask])[0, 1]

            if np.isnan(r):
                continue

            t = r * np.sqrt((n - 2) / max(1e-12, 1 - r**2))

            if abs(t) >= 1.96:
                r_out[i, j] = r

    return r_out

def draw_aoi_outline(ax, aoi_exterior, lw=1.8, color="#d7191c", alpha=0.95, zorder=65):
    if aoi_exterior is None:
        return
    x, y = aoi_exterior.xy
    if CARTOPY_AVAILABLE:
        ax.plot(x, y, transform=ccrs.PlateCarree(), color=color, lw=lw, alpha=alpha, zorder=zorder)
    else:
        ax.plot(x, y, color=color, lw=lw, alpha=alpha, zorder=zorder)


def zscore(series: pd.Series) -> pd.Series:
    s = series.astype(float)
    mu = np.nanmean(s.values)
    sd = np.nanstd(s.values)
    if not np.isfinite(sd) or sd == 0:
        return s * 0.0
    return (s - mu) / sd


# =============================================================================
# RESOLVE FILE PATHS
# =============================================================================
if not BASE_DIR.exists():
    raise RuntimeError(f"BASE_DIR does not exist: {BASE_DIR}")

all_tifs = {p.name: p for p in BASE_DIR.rglob("*.tif")}

resolved_paths: Dict[str, Path] = {}
missing_vars = []
for nm, fn in VARIABLE_FILES.items():
    p = all_tifs.get(fn, None)
    if p is None:
        missing_vars.append((nm, fn))
    else:
        resolved_paths[nm] = p

print(f"[INFO] BASE_DIR: {BASE_DIR}")
print(f"[INFO] Found tif files (recursive): {len(all_tifs)}")
print(f"[INFO] Matched variables: {len(resolved_paths)} / {len(VARIABLE_FILES)}")
if missing_vars:
    print("[WARN] Missing files (first 10):")
    for item in missing_vars[:10]:
        print("   ", item)

if not resolved_paths:
    raise RuntimeError("No variables matched. Check BASE_DIR and VARIABLE_FILES names.")

# =============================================================================
# REFERENCE RASTER + AOI MASK/EXTENT + AOI OUTLINE
# =============================================================================
first_raster_path = next(iter(resolved_paths.values()))
with rasterio.open(str(first_raster_path)) as src0:
    ref_h, ref_w = src0.height, src0.width
    ref_tr = src0.transform
    ref_crs = src0.crs

aoi_mask = None
AOI_EXTENT = None
AOI_EXTERIOR = None

if AMAZON_SHP is not None and AMAZON_SHP.exists():
    aoi_mask = build_aoi_mask(AMAZON_SHP, (ref_h, ref_w), ref_tr, ref_crs)
    AOI_EXTENT = compute_aoi_extent(AMAZON_SHP, ref_crs, pad_deg=2.0)

    aoi = gpd.read_file(AMAZON_SHP, engine="fiona")
    if aoi.crs is None:
        raise RuntimeError("AMAZON_SHP has no CRS.")
    aoi = aoi.to_crs(ref_crs)

    geom_u = unary_union(aoi.geometry)
    if isinstance(geom_u, MultiPolygon):
        geom_u = max(list(geom_u.geoms), key=lambda g: g.area)
    if not isinstance(geom_u, Polygon):
        raise RuntimeError("AOI union did not produce a Polygon.")
    AOI_EXTERIOR = geom_u.exterior

# =============================================================================
# LOAD DATA + ANOMALIES + AOI MEAN TS
# =============================================================================
loaded: Dict[str, xr.DataArray] = {}
edges: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}

for nm, p in resolved_paths.items():
    da, xe, ye = load_multiband_tif(p, START, aoi_mask)
    loaded[nm] = da.sel(time=slice(START, END))
    edges[nm] = (xe, ye)

if not loaded:
    raise RuntimeError("No variables loaded after resolving paths.")

anom_ds: Dict[str, xr.DataArray] = {}
global_ts: Dict[str, pd.Series] = {}

for nm, da in loaded.items():
    clim = monthly_climatology(da)
    anom = anomalies_from_climatology(da, clim)
    anom_ds[nm] = anom

    ts = anom.mean(("y", "x"), skipna=True).to_pandas()
    ts.index = pd.to_datetime(ts.index)
    global_ts[nm] = ts

# =============================================================================
# SELECT 4 MOST REPRESENTATIVE DOMAINS (out of 5) FOR MAPS
# =============================================================================
TARGET_VAR = "GPP"   # ← cambia aquí la variable base del mapa

y_da = anom_ds[TARGET_VAR].sel(time=slice(START, END))
lon_edges, lat_edges = edges[TARGET_VAR]

MAP_VARS = ["SM", "VPD", "XCO2", "PDSI"]
DOMAINS_5 = MAP_VARS
domain_scores = {v: np.nan for v in MAP_VARS}
TOP4_DOMAINS = MAP_VARS
r_maps = {}
best_vars = {}

for v in MAP_VARS:
    x_da = anom_ds[v].sel(time=slice(START, END))
    rsig = r_map_sig(y_da, x_da, min_n=10)
    r_maps[v] = rsig
    best_vars[v] = v

print("[INFO] Best variable per domain (BA correlation):")
for dom in DOMAINS_5:
    print(f"  - {dom}: {best_vars[dom]} (score={domain_scores[dom]:.3f})")

# pick top-4 domains by score (ignore NaN)
scored_domains = [(dom, domain_scores[dom]) for dom in DOMAINS_5 if np.isfinite(domain_scores[dom])]
scored_domains.sort(key=lambda x: x[1], reverse=True)
TOP4_DOMAINS = [d for d, _ in scored_domains[:4]]

if len(TOP4_DOMAINS) < 4:
    missing_needed = 4 - len(TOP4_DOMAINS)
    # fill from remaining (even if NaN) to keep 4 slots, but will be blank if missing
    rest = [d for d in DOMAINS_5 if d not in TOP4_DOMAINS]
    TOP4_DOMAINS += rest[:missing_needed]

print("[INFO] Selected TOP-4 domains for maps:", TOP4_DOMAINS)

# shared color limits based on the 4 chosen maps
vals_all: List[np.ndarray] = []
for dom in TOP4_DOMAINS:
    rr = r_maps.get(dom, None)
    if rr is None:
        continue
    vals_all.append(rr[np.isfinite(rr)])

finite_all = np.concatenate(vals_all) if len(vals_all) else np.array([])
if finite_all.size:
    p2, p98 = np.percentile(finite_all, [2, 98])
    m = max(abs(p2), abs(p98), 0.2)
    m = min(m, 1.0)
    vmin_map, vmax_map = -m, m
else:
    vmin_map, vmax_map = -1.0, 1.0

# =============================================================================
# BUILD DF FOR FIGURE 2 (ALL GROUPS)
# =============================================================================
GROUPS_PRESENT = {g: [v for v in vs if v in global_ts] for g, vs in GROUPS.items()}
corr_vars = [v for vs in GROUPS_PRESENT.values() for v in vs]
df = pd.DataFrame({v: global_ts[v] for v in corr_vars}).sort_index().dropna(axis=1, how="all")

corr_df = df.corr()
vars_list = list(df.columns)
nvar = len(vars_list)

lag_mat = np.full((nvar, nvar), np.nan, dtype=float)
r_mat   = np.full((nvar, nvar), np.nan, dtype=float)

for i, vi in enumerate(vars_list):
    for j, vj in enumerate(vars_list):
        lag, r = find_lag_of_max_correlation(df[vi], df[vj], MAX_LAG_MONTHS)
        lag_mat[i, j], r_mat[i, j] = lag, r

# =============================================================================
# EXPORT CSV SUMMARIES (EXPLORATORY)
# Place this block AFTER lag_mat and r_mat have been computed
# =============================================================================

def _safe_corr(a: pd.Series, b: pd.Series) -> float:
    x = pd.concat([a, b], axis=1).dropna()
    if x.shape[0] < 6:
        return float("nan")
    return float(x.iloc[:, 0].corr(x.iloc[:, 1]))

def _pct(x: float) -> float:
    return float(100.0 * x)

def _quantiles(arr: np.ndarray, qs=(5, 50, 95)) -> Tuple[float, float, float]:
    if arr.size == 0:
        return (float("nan"), float("nan"), float("nan"))
    return tuple(float(v) for v in np.percentile(arr, qs))

domain_rows = []
bestvar_rows = []

# Map variable -> index in vars_list (for lag lookup)
var_to_idx = {v: i for i, v in enumerate(vars_list)}

for dom in DOMAINS_5:
    best_v = best_vars.get(dom, None)
    score_spatial = float(domain_scores.get(dom, np.nan))

    rmap = r_maps.get(dom, None)
    if rmap is None or (not isinstance(rmap, np.ndarray)):
        sig_percent = float("nan")
        pos_percent = float("nan")
        neg_percent = float("nan")
        r_p05, r_p50, r_p95 = (float("nan"), float("nan"), float("nan"))
        mean_r_sig = float("nan")
        mean_abs_r_sig = float("nan")
        n_sig = 0
        n_total = 0
    else:
        sig_mask = np.isfinite(rmap)
        n_total = int(rmap.size)
        n_sig = int(sig_mask.sum())

        sig_percent = _pct(n_sig / n_total) if n_total > 0 else float("nan")

        r_sig = rmap[sig_mask]
        mean_r_sig = float(np.nanmean(r_sig)) if r_sig.size else float("nan")
        mean_abs_r_sig = float(np.nanmean(np.abs(r_sig))) if r_sig.size else float("nan")

        pos_percent = _pct((r_sig > 0).sum() / max(r_sig.size, 1))
        neg_percent = _pct((r_sig < 0).sum() / max(r_sig.size, 1))

        r_p05, r_p50, r_p95 = _quantiles(r_sig, qs=(5, 50, 95))

    # Temporal correlation with BA using AOI-mean anomalies
    if best_v is not None and best_v in df.columns and "BA" in df.columns:
        temporal_r = _safe_corr(df["BA"], df[best_v])
    else:
        temporal_r = float("nan")

    # Lag of maximum |r| between BA and best variable (from lag_mat/r_mat)
    if best_v is not None and best_v in var_to_idx and "BA" in var_to_idx:
        i_ba = var_to_idx["BA"]
        j_x  = var_to_idx[best_v]
        best_lag = float(lag_mat[i_ba, j_x])   # BA vs X
        best_r   = float(r_mat[i_ba, j_x])
    else:
        best_lag = float("nan")
        best_r   = float("nan")

    domain_rows.append({
        "Domain": dom,
        "Best_variable": best_v,
        "Spatial_mean_abs_r_sig": score_spatial,     # your domain_scores
        "Significant_pixels_percent": sig_percent,
        "Positive_sig_pixels_percent": pos_percent,
        "Negative_sig_pixels_percent": neg_percent,
        "r_sig_p05": r_p05,
        "r_sig_p50": r_p50,
        "r_sig_p95": r_p95,
        "AOI_temporal_r_with_BA(best_var)": temporal_r,
        "AOI_lag_of_max_abs_r_months(BA_vs_best_var)": best_lag,
        "AOI_r_at_that_lag(BA_vs_best_var)": best_r,
        "n_sig_pixels": n_sig,
        "n_total_pixels": n_total,
    })

    # Also store a variable-level row (best variable per domain)
    bestvar_rows.append({
        "Domain": dom,
        "Variable": best_v,
        "Spatial_mean_abs_r_sig": score_spatial,
        "Significant_pixels_percent": sig_percent,
        "mean_r_sig": mean_r_sig,
        "mean_abs_r_sig": mean_abs_r_sig,
        "r_sig_p05": r_p05,
        "r_sig_p50": r_p50,
        "r_sig_p95": r_p95,
        "AOI_temporal_r_with_BA": temporal_r,
        "AOI_lag_of_max_abs_r_months(BA_vs_var)": best_lag,
        "AOI_r_at_that_lag(BA_vs_var)": best_r,
        "n_sig_pixels": n_sig,
        "n_total_pixels": n_total,
    })

domain_overview = pd.DataFrame(domain_rows).sort_values("Spatial_mean_abs_r_sig", ascending=False)
bestvar_stats = pd.DataFrame(bestvar_rows)

out1 = OUT_DIR / "Domain_overview.csv"
out2 = OUT_DIR / "BestVar_pixel_stats.csv"
domain_overview.to_csv(out1, index=False)
bestvar_stats.to_csv(out2, index=False)

print("Saved CSV:", out1)
print("Saved CSV:", out2)

corr_out = OUT_DIR / "AOI_anomaly_correlation_matrix.csv"
lag_out  = OUT_DIR / "AOI_anomaly_lag_matrix_months.csv"

pd.DataFrame(corr_df.values, index=vars_list, columns=vars_list).to_csv(corr_out)
pd.DataFrame(lag_mat, index=vars_list, columns=vars_list).to_csv(lag_out)

print("Saved CSV:", corr_out)
print("Saved CSV:", lag_out)

# =============================================================================
# FIGURE 1 — 4 MAPS (TOP4) + ANOMALIES (ALL DOMAINS + FIRE)
# =============================================================================
def plot_corr_map_no_cb(ax, r_sig: np.ndarray, title_txt: str):
    Lon, Lat = np.meshgrid(lon_edges, lat_edges)

    if CARTOPY_AVAILABLE:
        im = ax.pcolormesh(
            Lon, Lat,
            np.ma.masked_invalid(r_sig),
            transform=ccrs.PlateCarree(),
            cmap=CMAP_MAP,
            shading="auto",
            vmin=vmin_map,
            vmax=vmax_map,
        )
        ax.coastlines("110m", linewidth=0.6)
        ax.add_feature(cfeature.BORDERS.with_scale("110m"), linewidth=0.4, alpha=0.6)

        if AOI_EXTENT is not None:
            ax.set_extent(AOI_EXTENT, crs=ccrs.PlateCarree())
        else:
            ax.set_global()
    else:
        im = ax.pcolormesh(
            Lon, Lat,
            np.ma.masked_invalid(r_sig),
            cmap=CMAP_MAP,
            shading="auto",
            vmin=vmin_map,
            vmax=vmax_map,
        )

    ax.text(
        0.5, 1.02, title_txt,
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=14,
        fontweight="normal"
    )
    ax.set_axis_off()
    draw_aoi_outline(ax, AOI_EXTERIOR, lw=1.8, color="#d7191c", alpha=0.95, zorder=65)
    add_map_frame(ax, lw=1.2, pad=0.002, shadow=False)
    return im


fig1 = plt.figure(figsize=(16, 8))
gs1 = fig1.add_gridspec(1, 2, width_ratios=[1.5, 1.5], wspace=0.1)

# ---- Left: 2x2 maps (TOP-4)
gs1A = gs1[0, 0].subgridspec(2, 2, hspace=0.10, wspace=0.001)

axesA = []
for i in range(2):
    for j in range(2):
        ax = fig1.add_subplot(gs1A[i, j], projection=ccrs.Robinson()) if CARTOPY_AVAILABLE else fig1.add_subplot(gs1A[i, j])
        axesA.append(ax)

labels_maps = ["(A)", "(B)", "(C)", "(D)"]
ims = []

for ax, lab, dom in zip(axesA, labels_maps, TOP4_DOMAINS):
    v = best_vars.get(dom, None)
    rsig = r_maps.get(dom, None)

    if (v is None) or (rsig is None):
        ax.set_axis_off()
        continue

    title_txt = f"{lab} r(Δ{TARGET_VAR}, Δ{v}) — {dom}"
    im = plot_corr_map_no_cb(ax, rsig, title_txt)
    ims.append(im)

# Get exact position of LEFT column (maps block)
left_block = gs1[0, 0].get_position(fig1)

cbar_h = 0.025
cbar_y = left_block.y0 - 0.045

shrink = 0.7 # 80% of original width
new_width = left_block.width * shrink
new_x = left_block.x0 + (left_block.width - new_width) / 2

cbar_ax = fig1.add_axes([
    new_x,
    cbar_y,
    new_width,
    cbar_h
])

if ims:
    cbA = fig1.colorbar(ims[-1], cax=cbar_ax, orientation="horizontal")
    cbA.set_label("Pearson r", fontsize=15)
    cbA.ax.tick_params(labelsize=10)
else:
    cbar_ax.axis("off")

# ---- Right: anomalies (ALL domains + Fire), NOT maps
order_ts = ["Climate", "Radiation", "Hydrological", "Carbon-Veg", "Atmosphere", "Fire"]
gs1B = gs1[0, 1].subgridspec(len(order_ts), 1, hspace=0.30)
axesB = [fig1.add_subplot(gs1B[i]) for i in range(len(order_ts))]

pals = {
    "Climate":         PALETTE_CLIMATE,
    "Radiation":       PALETTE_RADIATION,
    "Hydrological":    PALETTE_HYDROLOGICAL,
    "Carbon-Veg":      PALETTE_CARBON,
    "Atmosphere":      PALETTE_ATMOS,
    "Fire":            PALETTE_FIRE,
}

legend_handles, legend_labels = [], []

for ax, gname in zip(axesB, order_ts):
    vars_here = GROUPS_PRESENT.get(gname, [])
    vars_here = [v for v in vars_here if v in df.columns]
    if not vars_here:
        ax.set_xticks([]); ax.set_yticks([])
        continue

    for k, var in enumerate(vars_here):
        col = pals[gname][k % len(pals[gname])]
        s = df[var]
        sm = s.rolling(SMOOTH_WINDOW, center=True, min_periods=1).mean()
        smz = zscore(sm)
        ln, = ax.plot(smz.index, smz.values, lw=1.7, color=col, label=f"Δ{var}")

        if f"Δ{var}" not in legend_labels:
            legend_labels.append(f"Δ{var}")
            legend_handles.append(ln)

    ax.axhline(0, color="#999999", lw=0.6, alpha=0.6)
    ax.set_ylabel("z(Δ)", fontsize=10)

    panel_letter = chr(69 + order_ts.index(gname))  # E, F, G...
    ax.set_title(f"({panel_letter}) AOI mean monthly anomalies — {gname}", loc="left", fontsize=14, pad=6)
    ax.grid(True, ls=":", lw=0.5, alpha=0.7)

    if gname != "Fire":
        ax.set_xticklabels([])
    else:
        ax.set_xlabel("YEARS", fontsize=11)

# figure-level legend
fig1.subplots_adjust(left=0.01, right=0.98, top=0.96, bottom=0.18)
leg = fig1.legend(
    legend_handles, legend_labels,
    ncol=6,
    frameon=False,
    loc="lower center",
    bbox_to_anchor=(0.72, 0.0),
    fontsize=10,
    handlelength=2.0,
    columnspacing=1.2
)

fig1.savefig(OUTPNG_1, dpi=DPI, bbox_extra_artists=[leg], bbox_inches="tight")
fig1.savefig(OUTPDF_1, dpi=DPI, bbox_extra_artists=[leg], bbox_inches="tight")
print("Saved:", OUTPNG_1, OUTPDF_1)
plt.show()

# =============================================================================
# FIGURE 2 — CORRELATION + LAGS (ALL VARIABLES PRESENT)
# =============================================================================
fig2 = plt.figure(figsize=(14,7))
gs2 = fig2.add_gridspec(1, 2, width_ratios=[1, 1], wspace=0.4)

axC = fig2.add_subplot(gs2[0, 0])
norm_c = TwoSlopeNorm(vmin=-1.0, vcenter=0.0, vmax=1.0)
imC = axC.imshow(corr_df.values, cmap=CMAP_CORR, norm=norm_c, interpolation="nearest", aspect="equal")
axC.set_title("(A) Pearson correlation (AOI mean anomalies)", loc="left", fontsize=13, pad=8)

axC.set_xticks(np.arange(nvar))
axC.set_yticks(np.arange(nvar))
axC.set_xticklabels(vars_list, rotation=35, ha="right", fontsize=10)
axC.set_yticklabels(vars_list, fontsize=10)

for k in range(nvar + 1):
    axC.axhline(k - 0.5, color="#e6e6e6", lw=0.8)
    axC.axvline(k - 0.5, color="#e6e6e6", lw=0.8)

R_ANNOT_THRESH = 0.30
for i in range(nvar):
    for j in range(nvar):
        r = corr_df.values[i, j]
        if np.isnan(r) or abs(r) < R_ANNOT_THRESH or i == j:
            continue
        axC.text(
            j, i, f"{r:.2f}",
            ha="center", va="center", fontsize=8,
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.55, boxstyle="round,pad=0.10")
        )

divC = make_axes_locatable(axC)
caxC = divC.append_axes("right", size="4.5%", pad=0.15)
cbC = fig2.colorbar(imC, cax=caxC)
cbC.set_label("Pearson r", fontsize=11)
cbC.ax.tick_params(labelsize=10)

axD = fig2.add_subplot(gs2[0, 1])
imD = axD.imshow(
    lag_mat,
    cmap=mpl.colormaps["RdYlBu"],
    vmin=-MAX_LAG_MONTHS, vmax=MAX_LAG_MONTHS,
    interpolation="nearest", aspect="equal",
)
axD.set_title("(B) Lag of maximum |r| (months)", loc="left", fontsize=13, pad=8)

axD.set_xticks(np.arange(nvar))
axD.set_yticks(np.arange(nvar))
axD.set_xticklabels(vars_list, rotation=35, ha="right", fontsize=10)
axD.set_yticklabels(vars_list, fontsize=10)

for k in range(nvar + 1):
    axD.axhline(k - 0.5, color="#e6e6e6", lw=0.8)
    axD.axvline(k - 0.5, color="#e6e6e6", lw=0.8)

for i in range(nvar):
    for j in range(nvar):
        r = r_mat[i, j]
        lag = lag_mat[i, j]
        if np.isnan(r) or np.isnan(lag) or abs(r) < ANNOT_R_THRESH or i == j:
            continue
        axD.text(
            j, i, f"{int(lag):+d}",
            ha="center", va="center", fontsize=8,
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.55, boxstyle="round,pad=0.10")
        )

divD = make_axes_locatable(axD)
caxD = divD.append_axes("right", size="4.5%", pad=0.15)
cbD = fig2.colorbar(imD, cax=caxD)
cbD.set_label("Lag (months)", fontsize=11)
cbD.ax.tick_params(labelsize=10)

fig2.tight_layout()
fig2.savefig(OUTPNG_2, dpi=DPI, bbox_inches="tight")
fig2.savefig(OUTPDF_2, dpi=DPI, bbox_inches="tight")
print("Saved:", OUTPNG_2, OUTPDF_2)
plt.show()