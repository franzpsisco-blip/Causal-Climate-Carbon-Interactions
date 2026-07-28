from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import rioxarray as rxr
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches

from scipy import ndimage
from shapely.geometry import box
import geopandas as gpd
import rasterio

import cartopy.crs as ccrs
import cartopy.feature as cfeature

try:
    from sklearn.cluster import MiniBatchKMeans
    SKLEARN_OK = True
except Exception:
    SKLEARN_OK = False

try:
    import pyEDM
    PYEDM_OK = True
except Exception:
    PYEDM_OK = False


# =============================================================================
# WARNINGS / NUMPY
# =============================================================================
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message="Degrees of freedom*")
warnings.filterwarnings("ignore", message="All-NaN slice*")
np.seterr(invalid="ignore", divide="ignore")


# =============================================================================
# PATHS
# =============================================================================
BASE_DIR = Path(r"C:\Users\Zhou laoshi\OneDrive\Desktop\Data Amazonas-Resampled")
OUT_DIR = BASE_DIR / "CCM_DOMINANCE_100AREAS_ALLSTACKS"
OUT_DIR.mkdir(exist_ok=True)

LOG_PATH = OUT_DIR / "runlog.txt"

STUDY_SHP = BASE_DIR / "amazonia_boundary_proposal_Eva_2005" / "amazonia_polygons.shp"

DEM_PATH = Path(
    r"C:\Users\Zhou laoshi\OneDrive\Documentos\WeChat Files\wxid_x58cj6pjp4q612\FileStorage\File\2026-01\SouthAmerica_DEM_SRTM_0p1deg.tif"
)


# =============================================================================
# INPUT FILES
# =============================================================================
FILES = {
    # targets
    "PDSI": "PDSI.tif",
    "GPP": "GPP.tif",
    "NEE": "NEE.tif",
    "RECO": "RECO.tif",
    "TWSA": "TWSA.tif",
    "SOIL_MOISTURE": "SOIL_MOISTURE.tif",

    # hydro
    "PRECIP": "PRECIP.tif",
    "VPD": "VPD.tif",
    "AET": "AET.tif",
    "RUNOFF": "RUNOFF.tif",

    # vegetation/carbon
    "EVI": "EVI.tif",
    "SIF": "SIF.tif",

    # fire
    "BA": "BA.tif",
    "FIRMS": "FIRMS.tif",

    # radiation / energy
    "PAR": "PAR.tif",
    "LST": "LST.tif",

    # atmospheric composition
    "XCO2": "XCO2.tif",
    "XCH4": "XCH4.tif",
    "AOD550": "AOD550.tif",
}

TARGETS = ["PDSI", "GPP", "NEE", "RECO", "TWSA", "SOIL_MOISTURE"]


# =============================================================================
# EXACTLY 100 AREAS
# =============================================================================
N_AREAS = 100
MAX_SAMPLE_PIXELS = 30000

# use all raster stacks to define the 100 areas
CLUSTER_VAR_ORDER = [
    "PDSI", "GPP", "NEE", "RECO", "TWSA", "SOIL_MOISTURE",
    "PRECIP", "VPD", "AET", "RUNOFF",
    "EVI", "SIF",
    "BA", "FIRMS",
    "PAR", "LST",
    "XCO2", "XCH4", "AOD550",
]

CLUSTER_MIN_N = 90
CLUSTER_MIN_VALID_VARS = 6

ADD_SPATIAL_COORDS = True
LAT_WEIGHT = 1.20
LON_WEIGHT = 1.20

CLUSTER_MAJORITY_PASSES = 1
CLUSTER_ISLAND_MIN_PIX = 14
CLUSTER_NBH_SIZE = 5


# =============================================================================
# CCM PARAMETERS
# =============================================================================
# fixed E to keep runtime manageable
E_FIXED = 3
LAG_MIN = 0
LAG_MAX = 6
TP = 1
MIN_T = 110
SAMPLE = 8

CCM_MIN_RHO_FINAL = 0.10
CCM_MIN_GAIN = 0.03


# =============================================================================
# FAMILIES
# =============================================================================
BASE_FAMS = ["ATM_COMP", "HYDRO", "VEG", "FIRE", "RAD"]

FAM_EXCLUDE_BY_TARGET = {
    "GPP": ["VEG"],
    "NEE": ["VEG"],
    "RECO": ["VEG"],
    "TWSA": ["HYDRO"],
    "SOIL_MOISTURE": ["HYDRO"],
}

def allowed_families_for_target(tgt: str):
    drop = set(FAM_EXCLUDE_BY_TARGET.get(tgt, []))
    return [f for f in BASE_FAMS if f not in drop]


# =============================================================================
# COLORS
# =============================================================================
PALETTE = {
    "white": "#f3f3f3",
    "coral": "#fc644c",
    "cyanblue": "#32a4b4",
    "deepteal": "#026c2c",
    "yellow": "#ffd44e",
    "violet": "#8a83a0",
}

NODATA_COLOR = PALETTE["white"]

FAMILY_ID = {"ATM_COMP": 1, "HYDRO": 2, "VEG": 3, "FIRE": 4, "RAD": 5}
FAMILY_COLOR = {
    "ATM_COMP": PALETTE["coral"],
    "HYDRO": PALETTE["cyanblue"],
    "VEG": PALETTE["deepteal"],
    "FIRE": PALETTE["yellow"],
    "RAD": PALETTE["violet"],
}
FAMILY_LABEL = {
    "ATM_COMP": "Atmospheric composition",
    "HYDRO": "Hydro",
    "VEG": "Vegetation / carbon",
    "FIRE": "Fire",
    "RAD": "Radiation / energy",
}
FAMILY_ORDER = ["ATM_COMP", "HYDRO", "VEG", "FIRE", "RAD"]


# =============================================================================
# MAP STYLE
# =============================================================================
OFF_LEFT = 4.0
OFF_OTHER = 2.0

HILLSHADE_ALPHA = 0.90
HILLSHADE_GAMMA = 0.55
MAP_ALPHA = 0.94
LAG_ALPHA = 0.96


# =============================================================================
# LOG
# =============================================================================
def log(msg: str):
    print(msg)
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(msg + "\n")


# =============================================================================
# IO / BASIC HELPERS
# =============================================================================
def must_exist(path: Path, label: str):
    if not path.exists():
        raise FileNotFoundError(f"Missing {label}: {path}")

def load_raster(path: Path, start="2000-01-01") -> xr.DataArray:
    da = rxr.open_rasterio(path, masked=True).astype("float32")

    ren = {}
    if "band" in da.dims:
        ren["band"] = "time"
    if "y" in da.dims:
        ren["y"] = "lat"
    if "x" in da.dims:
        ren["x"] = "lon"

    da = da.rename(ren).transpose("time", "lat", "lon")

    if not np.issubdtype(da.time.dtype, np.datetime64):
        da = da.assign_coords(time=pd.date_range(start, periods=da.sizes["time"], freq="MS"))

    if float(da.lat[0]) < float(da.lat[-1]):
        da = da.sortby("lat", ascending=False)

    return da

def reproject_match(da: xr.DataArray, ref: xr.DataArray) -> xr.DataArray:
    try:
        if getattr(ref.rio, "crs", None) is not None and getattr(da.rio, "crs", None) is None:
            da = da.rio.write_crs(ref.rio.crs)
    except Exception:
        pass

    try:
        return da.rio.reproject_match(ref)
    except Exception:
        return da.interp(lat=ref.lat, lon=ref.lon, method="nearest").transpose("time", "lat", "lon")

def common_time(*das) -> np.ndarray:
    das = [d for d in das if d is not None]
    t = das[0].time.values
    for d in das[1:]:
        t = np.intersect1d(t, d.time.values)
    return np.array(t, dtype="datetime64[ns]")

def strict_mask(da: xr.DataArray, min_n: int) -> xr.DataArray:
    return (np.isfinite(da)).sum("time") >= min_n


# =============================================================================
# TIME-SERIES PREP
# =============================================================================
def _lin_detrend_1d(x):
    x = np.asarray(x, float)
    m = np.isfinite(x)
    if m.sum() < 8:
        return x * 0 + np.nan
    t = np.arange(m.sum(), dtype=float)
    y = x[m]
    A = np.vstack([t, np.ones_like(t)]).T
    b, *_ = np.linalg.lstsq(A, y, rcond=None)
    yhat = b[0] * t + b[1]
    out = x * 0 + np.nan
    out[m] = y - yhat
    return out

def deseasonalize(da: xr.DataArray) -> xr.DataArray:
    det = xr.apply_ufunc(
        _lin_detrend_1d, da,
        input_core_dims=[["time"]],
        output_core_dims=[["time"]],
        vectorize=True,
        dask="parallelized",
        output_dtypes=[da.dtype],
    )
    clim = det.groupby("time.month").mean("time", skipna=True)
    return (det.groupby("time.month") - clim).astype("float32").transpose("time", "lat", "lon")

def zscore_series_xr(da: xr.DataArray) -> xr.DataArray:
    da = da.transpose("time", "lat", "lon")

    def _z(x):
        x = np.asarray(x, float)
        m = np.isfinite(x)
        if m.sum() < 8:
            return x * 0 + np.nan
        mu = np.nanmean(x[m])
        sd = np.nanstd(x[m]) + 1e-6
        out = x * 0 + np.nan
        out[m] = (x[m] - mu) / sd
        return out

    return xr.apply_ufunc(
        _z, da,
        input_core_dims=[["time"]],
        output_core_dims=[["time"]],
        vectorize=True,
        dask="parallelized",
        output_dtypes=[da.dtype],
    ).astype("float32").transpose("time", "lat", "lon")

def prep_da_for_analysis(da: xr.DataArray) -> xr.DataArray:
    return zscore_series_xr(deseasonalize(da)).astype("float32")


# =============================================================================
# SPATIAL BOUNDS / EXPORT
# =============================================================================
def latlon_bounds_from_coords(da2d: xr.DataArray):
    lon = np.asarray(da2d["lon"].values, dtype=float)
    lat = np.asarray(da2d["lat"].values, dtype=float)

    dx = float(np.nanmedian(np.abs(np.diff(lon)))) if lon.size > 1 else 0.5
    dy = float(np.nanmedian(np.abs(np.diff(lat)))) if lat.size > 1 else 0.5

    minx = float(np.nanmin(lon) - dx / 2.0)
    maxx = float(np.nanmax(lon) + dx / 2.0)
    miny = float(np.nanmin(lat) - dy / 2.0)
    maxy = float(np.nanmax(lat) + dy / 2.0)
    return minx, miny, maxx, maxy

def write_geotiff_like_ref(da2d: xr.DataArray, ref2d: xr.DataArray, out_path: Path, nodata=0):
    out = da2d.copy()
    if set(out.dims) == {"lat", "lon"}:
        out = out.rename({"lat": "y", "lon": "x"})
    out = out.rio.set_spatial_dims(x_dim="x", y_dim="y", inplace=False)

    ref = ref2d
    if set(ref.dims) == {"lat", "lon"}:
        ref = ref.rename({"lat": "y", "lon": "x"})

    try:
        if getattr(ref.rio, "crs", None) is not None:
            out = out.rio.write_crs(ref.rio.crs, inplace=False)
    except Exception:
        pass

    try:
        out = out.rio.write_transform(ref.rio.transform(), inplace=False)
    except Exception:
        try:
            out = out.rio.reproject_match(ref)
        except Exception:
            pass

    out = out.fillna(nodata)
    try:
        out.rio.write_nodata(nodata, inplace=True)
    except Exception:
        pass

    out.rio.to_raster(str(out_path))
    log(f"[GeoTIFF] {out_path}")

def classes_to_rgba(classes2d: np.ndarray, class_to_hex: dict[int, str]) -> np.ndarray:
    arr = np.asarray(classes2d)
    ny, nx = arr.shape
    rgba = np.zeros((ny, nx, 4), dtype=float)

    # full transparency for nodata / 0 / nan
    rgba[..., 3] = 0.0

    for cid, col in class_to_hex.items():
        m = np.isfinite(arr) & (arr == cid)
        if np.any(m):
            rgb = mcolors.to_rgb(col)
            rgba[m, 0] = rgb[0]
            rgba[m, 1] = rgb[1]
            rgba[m, 2] = rgb[2]
            rgba[m, 3] = MAP_ALPHA

    return rgba


# =============================================================================
# NOISE REMOVAL
# =============================================================================
def majority_filter(arr: np.ndarray, valid_mask: np.ndarray, passes: int = 1) -> np.ndarray:
    arr = arr.copy().astype(np.int16)

    def mode_nonzero(v):
        v = v.astype(np.int64)
        v = v[v > 0]
        if v.size == 0:
            return 0
        bc = np.bincount(v)
        return int(np.argmax(bc))

    for _ in range(passes):
        sm = ndimage.generic_filter(arr, mode_nonzero, size=3, mode="nearest")
        arr[valid_mask] = sm[valid_mask]
    arr[~valid_mask] = 0
    return arr

def remove_small_islands(arr: np.ndarray, valid_mask: np.ndarray, min_size: int = 20, nbh: int = 5) -> np.ndarray:
    out = arr.copy().astype(np.int16)
    out[~valid_mask] = 0

    def mode_nonzero(v):
        v = v.astype(np.int64)
        v = v[v > 0]
        if v.size == 0:
            return 0
        bc = np.bincount(v)
        return int(np.argmax(bc))

    for lab in np.unique(out):
        if lab <= 0:
            continue

        comp = (out == lab) & valid_mask
        if not np.any(comp):
            continue

        labeled, n = ndimage.label(comp, structure=np.ones((3, 3), dtype=int))
        if n == 0:
            continue

        sizes = np.bincount(labeled.ravel())
        small_ids = np.where(sizes < min_size)[0]
        small_ids = small_ids[small_ids != 0]
        if small_ids.size == 0:
            continue

        small = np.isin(labeled, small_ids)
        local_mode = ndimage.generic_filter(out, mode_nonzero, size=nbh, mode="nearest")
        out[small] = local_mode[small]

    out[~valid_mask] = 0
    return out


# =============================================================================
# BORDERS / HILLSHADE
# =============================================================================
def load_country_boundaries(ref_crs):
    try:
        import cartopy.io.shapereader as shpreader
        shp = shpreader.natural_earth("110m", "cultural", "admin_0_countries")
        return gpd.read_file(shp).to_crs(ref_crs)
    except Exception:
        return None

def load_study_outer_border_only(ref_crs):
    if not STUDY_SHP.exists():
        log(f"[WARN] Study shapefile not found: {STUDY_SHP}")
        return None

    try:
        gdf = gpd.read_file(STUDY_SHP)
    except Exception as e:
        log(f"[WARN] Could not read study shapefile: {repr(e)}")
        return None

    if gdf.crs is None:
        log("[WARN] Study shapefile has no CRS.")
        return None

    gdf = gdf.to_crs(ref_crs)
    gdf = gdf[gdf.geometry.notnull()].copy()

    try:
        gdf["geometry"] = gdf.geometry.buffer(0)
    except Exception:
        pass

    return gdf.dissolve()

def read_dem_hillshade_native():
    if not DEM_PATH.exists():
        log(f"[DEM] Not found: {DEM_PATH}")
        return None, None

    with rasterio.open(DEM_PATH) as src:
        dem = src.read(1).astype(np.float32)
        nodata = src.nodata
        if nodata is not None:
            dem = np.where(dem == nodata, np.nan, dem)

        dem = np.where(dem == 0, np.nan, dem)
        dem = np.where(np.isfinite(dem), dem, np.nan)

        left, bottom, right, top = src.bounds.left, src.bounds.bottom, src.bounds.right, src.bounds.top

        v = dem[np.isfinite(dem)]
        if v.size > 100:
            lo, hi = np.nanpercentile(v, [2, 98])
            d = (dem - lo) / max(1e-6, (hi - lo))
            d = np.clip(d, 0, 1)
        else:
            d = dem

        from matplotlib.colors import LightSource
        ls = LightSource(azdeg=315, altdeg=45)
        hs = ls.hillshade(np.where(np.isfinite(d), d, np.nan), vert_exag=1.55)
        hs = np.power(np.clip(hs, 0, 1), HILLSHADE_GAMMA)

        return hs.astype(np.float32), (left, right, bottom, top)

def get_plot_context(ref2d: xr.DataArray):
    borders = load_country_boundaries(ref2d.rio.crs)
    study_outer = load_study_outer_border_only(ref2d.rio.crs)
    hillshade, dem_extent = read_dem_hillshade_native()

    minx, miny, maxx, maxy = latlon_bounds_from_coords(ref2d)

    minx2 = minx - OFF_LEFT
    maxx2 = maxx + OFF_OTHER
    miny2 = miny - OFF_OTHER
    maxy2 = maxy + OFF_OTHER

    # IMPORTANT: do NOT clip shapefiles; just set map extent later
    return {
        "borders": borders,
        "study_outer": study_outer,
        "hillshade": hillshade,
        "dem_extent": dem_extent,
        "extent": (minx, maxx, miny, maxy),
        "extent_buf": (minx2, maxx2, miny2, maxy2),
    }


# =============================================================================
# FAMILY INDEX
# =============================================================================
def family_index(members: list[xr.DataArray], reducer="median") -> xr.DataArray | None:
    if not members:
        return None

    stack = xr.concat(members, dim="member")
    if reducer == "max":
        out = stack.max("member", skipna=True)
    else:
        out = stack.median("member", skipna=True)

    return out.astype("float32")


# =============================================================================
# CLUSTERING WITH ALL RASTERSTACKS
# =============================================================================
def make_pixel_feature_vector(x: np.ndarray) -> list[float]:
    x = x[np.isfinite(x)]
    if x.size < 20:
        return [np.nan] * 6

    feats = [
        float(np.nanmean(x)),
        float(np.nanstd(x)),
        float(np.nanquantile(x, 0.25)),
        float(np.nanquantile(x, 0.50)),
        float(np.nanquantile(x, 0.75)),
    ]

    L = 1
    if x.size > L + 5:
        a = x[:-L]
        b = x[L:]
        cc = np.corrcoef(a, b)[0, 1]
        feats.append(float(cc) if np.isfinite(cc) else 0.0)
    else:
        feats.append(0.0)

    return feats

def build_exact_100_areas(ref: xr.DataArray, cluster_das: dict[str, xr.DataArray]):
    if not SKLEARN_OK:
        raise ImportError("scikit-learn is required for clustering.")

    lat = ref.lat.values
    lon = ref.lon.values
    ny, nx = len(lat), len(lon)

    masks = []
    used = []
    for name in CLUSTER_VAR_ORDER:
        da = cluster_das.get(name, None)
        if da is None:
            continue
        masks.append(strict_mask(da, CLUSTER_MIN_N))
        used.append(name)

    if not masks:
        raise RuntimeError("No clustering variables available.")

    cnt = xr.zeros_like(ref.isel(time=0, drop=True), dtype=np.int16)
    for m in masks:
        cnt = cnt + m.astype(np.int16)

    M = cnt >= CLUSTER_MIN_VALID_VARS
    valid_px = int(M.sum())
    log(f"[CLUSTER] vars used: {used}")
    log(f"[CLUSTER] valid px before clustering: {valid_px}")

    if valid_px <= N_AREAS:
        raise RuntimeError(f"Too few valid pixels ({valid_px}) for {N_AREAS} areas.")

    valid_idx = np.argwhere(M.values.astype(bool))
    rng = np.random.default_rng(42)
    n_samp = min(MAX_SAMPLE_PIXELS, valid_idx.shape[0])
    pick = valid_idx[rng.choice(valid_idx.shape[0], size=n_samp, replace=False)]

    X = []
    for i, j in pick:
        feats = []
        for name in CLUSTER_VAR_ORDER:
            da = cluster_das.get(name, None)
            if da is None:
                continue
            feats.extend(make_pixel_feature_vector(da[:, i, j].values.astype(float)))

        if ADD_SPATIAL_COORDS:
            latv = float(ref.lat.values[i])
            lonv = float(ref.lon.values[j])

            latn = (latv - float(np.nanmin(ref.lat.values))) / max(
                1e-6, float(np.nanmax(ref.lat.values) - np.nanmin(ref.lat.values))
            )
            lonn = (lonv - float(np.nanmin(ref.lon.values))) / max(
                1e-6, float(np.nanmax(ref.lon.values) - np.nanmin(ref.lon.values))
            )

            feats.append(LAT_WEIGHT * latn)
            feats.append(LON_WEIGHT * lonn)

        X.append(feats)

    X = np.asarray(X, dtype=np.float32)
    col_med = np.nanmedian(X, axis=0)
    bad = ~np.isfinite(X)
    X[bad] = np.take(col_med, np.where(bad)[1])

    log(f"[CLUSTER] fitting MiniBatchKMeans k={N_AREAS} sample={X.shape[0]} feats={X.shape[1]}")
    km = MiniBatchKMeans(
        n_clusters=N_AREAS,
        random_state=42,
        batch_size=2048,
        n_init="auto",
        max_iter=250,
    )
    km.fit(X)

    labels = np.zeros((ny, nx), dtype=np.int16)
    batch_rows = 96

    for i0 in range(0, ny, batch_rows):
        i1 = min(ny, i0 + batch_rows)
        Mi = M.values[i0:i1, :]
        idx = np.argwhere(Mi)
        if idx.size == 0:
            continue

        Xi = []
        for ii, jj in idx:
            i = i0 + ii
            j = jj

            feats = []
            for name in CLUSTER_VAR_ORDER:
                da = cluster_das.get(name, None)
                if da is None:
                    continue
                feats.extend(make_pixel_feature_vector(da[:, i, j].values.astype(float)))

            if ADD_SPATIAL_COORDS:
                latv = float(ref.lat.values[i])
                lonv = float(ref.lon.values[j])

                latn = (latv - float(np.nanmin(ref.lat.values))) / max(
                    1e-6, float(np.nanmax(ref.lat.values) - np.nanmin(ref.lat.values))
                )
                lonn = (lonv - float(np.nanmin(ref.lon.values))) / max(
                    1e-6, float(np.nanmax(ref.lon.values) - np.nanmin(ref.lon.values))
                )

                feats.append(LAT_WEIGHT * latn)
                feats.append(LON_WEIGHT * lonn)

            Xi.append(feats)

        Xi = np.asarray(Xi, dtype=np.float32)
        bad = ~np.isfinite(Xi)
        Xi[bad] = np.take(col_med, np.where(bad)[1])

        lab = km.predict(Xi) + 1
        for n, (ii, jj) in enumerate(idx):
            labels[i0 + ii, jj] = int(lab[n])

    labels[~M.values] = 0

    labels = majority_filter(labels, M.values, passes=CLUSTER_MAJORITY_PASSES)
    labels = remove_small_islands(labels, M.values, min_size=CLUSTER_ISLAND_MIN_PIX, nbh=CLUSTER_NBH_SIZE)

    return xr.DataArray(labels, coords={"lat": lat, "lon": lon}, dims=("lat", "lon"))


# =============================================================================
# AGGREGATE BY AREA
# =============================================================================
def aggregate_by_area(da: xr.DataArray, areas: xr.DataArray, n_areas: int, reducer="median") -> pd.DataFrame:
    arr = da.values
    ar = areas.values.astype(np.int16)
    time = pd.to_datetime(da.time.values)

    out = np.full((len(time), n_areas), np.nan, dtype=np.float32)

    for c in range(1, n_areas + 1):
        m = (ar == c)
        if not np.any(m):
            continue
        X = arr[:, m]
        if reducer == "mean":
            out[:, c - 1] = np.nanmean(X, axis=1)
        else:
            out[:, c - 1] = np.nanmedian(X, axis=1)

    df = pd.DataFrame(out, columns=[f"a{c}" for c in range(1, n_areas + 1)])
    df.insert(0, "time", time)
    return df


# =============================================================================
# CCM
# =============================================================================
def make_libsizes(T: int) -> str | None:
    lib_end = min(170, T - 1)
    lib_start = max(30, lib_end - 110)
    if lib_end <= lib_start + 10:
        return None
    return f"{lib_start} {lib_end} 10"

def _align_lag(src: np.ndarray, tgt: np.ndarray, lag: int):
    if lag > 0:
        return src[:-lag], tgt[lag:]
    return src, tgt

def ccm_best_pyedm(source: np.ndarray, target: np.ndarray):
    if not PYEDM_OK:
        return None

    best = None

    for lag in range(LAG_MIN, LAG_MAX + 1):
        src, tgt = _align_lag(source, target, lag)
        df = pd.DataFrame({"cause": src, "effect": tgt}).replace([np.inf, -np.inf], np.nan).dropna()

        if len(df) < MIN_T:
            continue
        if df["cause"].std() == 0 or df["effect"].std() == 0:
            continue

        libSizes = make_libsizes(len(df))
        if libSizes is None:
            continue

        try:
            out = pyEDM.CCM(
                dataFrame=df.reset_index(drop=True),
                columns="cause",
                target="effect",
                E=E_FIXED,
                Tp=TP,
                libSizes=libSizes,
                sample=SAMPLE,
                showPlot=False
            )
        except Exception:
            continue

        cols = list(out.columns)
        rho_col = None
        if "cause:effect" in cols:
            rho_col = "cause:effect"
        elif "rho" in cols:
            rho_col = "rho"
        else:
            numeric_cols = [c for c in cols if c != "LibSize"]
            if numeric_cols:
                rho_col = numeric_cols[-1]

        if rho_col is None:
            continue

        rho = out[rho_col].values.astype(float)
        rho = rho[np.isfinite(rho)]
        if rho.size < 3:
            continue

        rho0 = float(rho[0])
        rhoF = float(rho[-1])
        gain = rhoF - rho0

        if rhoF < CCM_MIN_RHO_FINAL or gain < CCM_MIN_GAIN:
            continue

        if (best is None) or (rhoF > best[0]):
            best = (rhoF, lag, E_FIXED, rho0, gain)

    return best


# =============================================================================
# PLOTTING
# =============================================================================
def nice_ticks(vmin, vmax, step):
    start = np.floor(vmin / step) * step
    end = np.ceil(vmax / step) * step
    return np.arange(start, end + 0.001, step)

def draw_base_map(ax, plot_ctx: dict):
    minx2, maxx2, miny2, maxy2 = plot_ctx["extent_buf"]
    hs = plot_ctx["hillshade"]
    dem_extent = plot_ctx["dem_extent"]
    borders = plot_ctx["borders"]
    study_outer = plot_ctx["study_outer"]

    if hs is not None and dem_extent is not None:
        l, r, b, t = dem_extent
        hs = np.ma.masked_invalid(hs)
        ax.imshow(
            hs,
            cmap="gray",
            extent=(l, r, b, t),
            origin="upper",
            interpolation="bilinear",
            alpha=HILLSHADE_ALPHA,
            transform=ccrs.PlateCarree(),
            zorder=0
        )

    if borders is not None:
        borders.boundary.plot(ax=ax, color="black", linewidth=0.7, zorder=10)

    if study_outer is not None:
        study_outer.boundary.plot(ax=ax, color="red", linewidth=1.6, zorder=15)

    ax.set_extent([minx2, maxx2, miny2, maxy2], crs=ccrs.PlateCarree())
    ax.coastlines(color="0.35", lw=0.4)
    ax.add_feature(cfeature.BORDERS.with_scale("110m"), lw=0.2, edgecolor="0.45")

    xt = nice_ticks(minx2, maxx2, 5)
    yt = nice_ticks(miny2, maxy2, 5)
    ax.set_xticks(xt, crs=ccrs.PlateCarree())
    ax.set_yticks(yt, crs=ccrs.PlateCarree())
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.gridlines(draw_labels=False, linewidth=0.25, color="0.5", alpha=0.25, linestyle=":")

def plot_dominance_map(ax, da2d: xr.DataArray, title: str, plot_ctx: dict, legend=True):
    minx, maxx, miny, maxy = plot_ctx["extent"]
    draw_base_map(ax, plot_ctx)

    rgba = classes_to_rgba(da2d.values, {FAMILY_ID[f]: FAMILY_COLOR[f] for f in FAMILY_ORDER})
    ax.imshow(
        rgba,
        origin="upper",
        extent=(minx, maxx, miny, maxy),
        transform=ccrs.PlateCarree(),
        interpolation="nearest",
        zorder=5
    )
    ax.set_title(title, loc="left", fontsize=11)

    if legend:
        patches = []
        for fam in FAMILY_ORDER:
            cid = FAMILY_ID[fam]
            if np.any(da2d.values == cid):
                patches.append(mpatches.Patch(color=FAMILY_COLOR[fam], label=FAMILY_LABEL[fam]))
        if patches:
            ax.legend(
                handles=patches,
                title="Dominant family",
                frameon=False,
                loc="lower left",
                fontsize=7.5,
                title_fontsize=8.5
            )

def plot_lag_map(ax, lag_da: xr.DataArray, title: str, plot_ctx: dict):
    minx, maxx, miny, maxy = plot_ctx["extent"]
    draw_base_map(ax, plot_ctx)

    arr = lag_da.values.astype(float)
    arr = np.where(np.isfinite(arr), arr, np.nan)
    arr = np.where(arr < LAG_MIN, np.nan, arr)

    cmap = plt.cm.viridis.copy()
    cmap.set_bad((1, 1, 1, 0))  # transparent for NaN / nodata
    marr = np.ma.masked_invalid(arr)

    im = ax.imshow(
        marr,
        cmap=cmap,
        vmin=LAG_MIN,
        vmax=LAG_MAX,
        origin="upper",
        extent=(minx, maxx, miny, maxy),
        transform=ccrs.PlateCarree(),
        interpolation="nearest",
        alpha=LAG_ALPHA,
        zorder=5
    )
    ax.set_title(title, loc="left", fontsize=11)
    return im

def make_6panel_dominance_summary(results: list[dict], out_png: Path, out_pdf: Path, out_svg: Path, plot_ctx: dict):
    fig = plt.figure(figsize=(15.6, 9.6))
    gs = fig.add_gridspec(2, 3, wspace=0.08, hspace=0.10)
    panel_letters = ["a.", "b.", "c.", "d.", "e.", "f."]

    for i, res in enumerate(results[:6]):
        r = i // 3
        c = i % 3
        ax = fig.add_subplot(gs[r, c], projection=ccrs.PlateCarree())
        plot_dominance_map(
            ax=ax,
            da2d=res["dom_map"],
            title=f"{panel_letters[i]} {res['target']}",
            plot_ctx=plot_ctx,
            legend=(i == 0)
        )

    plt.tight_layout()
    fig.savefig(out_png, dpi=420, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_svg, bbox_inches="tight")
    plt.close(fig)

    log(f"[FIG] {out_png}")
    log(f"[FIG] {out_pdf}")
    log(f"[FIG] {out_svg}")


# =============================================================================
# MAIN
# =============================================================================
def main():
    if LOG_PATH.exists():
        LOG_PATH.unlink()

    log("=== CCM dominance using exactly 100 areas from all rasterstacks ===")
    log(f"[BASE] {BASE_DIR}")
    log(f"[OUT ] {OUT_DIR}")
    log(f"[pyEDM] {PYEDM_OK}")
    log(f"[sklearn] {SKLEARN_OK}")

    if not PYEDM_OK:
        raise ImportError("pyEDM is required for this script.")
    if not SKLEARN_OK:
        raise ImportError("scikit-learn is required for this script.")

    must_exist(BASE_DIR / FILES[TARGETS[0]], TARGETS[0])
    ref = load_raster(BASE_DIR / FILES[TARGETS[0]])
    ref2d = ref.isel(time=0, drop=True)
    plot_ctx = get_plot_context(ref2d)

    # -------------------------------------------------------------------------
    # Load all rasterstacks once
    # -------------------------------------------------------------------------
    log("[LOAD] all rasterstacks")
    data = {}
    for name in CLUSTER_VAR_ORDER:
        f = FILES.get(name, None)
        if f is None:
            data[name] = None
            continue
        p = BASE_DIR / f
        if not p.exists():
            log(f"  [MISS] {name}")
            data[name] = None
            continue
        da = reproject_match(load_raster(p), ref)
        data[name] = da
        log(f"  [OK] {name}")

    existing = [d for d in data.values() if d is not None]
    all_t = common_time(*existing)
    log(f"[TIME] common time all stacks: {len(all_t)}")

    proc = {}
    for name, da in data.items():
        if da is None:
            proc[name] = None
            continue
        proc[name] = prep_da_for_analysis(da.sel(time=all_t))
        log(f"  [PREP] {name}")

    # -------------------------------------------------------------------------
    # Build exactly 100 coherent areas
    # -------------------------------------------------------------------------
    log("[STEP] building exactly 100 coherent areas")
    areas = build_exact_100_areas(ref.sel(time=all_t), proc)

    areas_tif = OUT_DIR / "areas_k100.tif"
    write_geotiff_like_ref(areas.astype("int16"), ref2d, areas_tif, nodata=0)

    # quick area map
    areas_png = OUT_DIR / "areas_k100.png"
    fig = plt.figure(figsize=(8.1, 6.3))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
    draw_base_map(ax, plot_ctx)

    minx, maxx, miny, maxy = plot_ctx["extent"]
    cmap_areas = plt.get_cmap("tab20", N_AREAS).copy()
    cmap_areas.set_bad((1, 1, 1, 0))
    am = np.ma.array(areas.values.astype(float), mask=(areas.values <= 0))
    ax.imshow(
        am,
        cmap=cmap_areas,
        origin="upper",
        extent=(minx, maxx, miny, maxy),
        interpolation="nearest",
        transform=ccrs.PlateCarree(),
        alpha=0.92,
        zorder=5
    )
    ax.set_title("Coherent 100 areas from all rasterstacks", loc="left")
    fig.savefig(areas_png, dpi=360, bbox_inches="tight")
    plt.close(fig)
    log(f"[FIG] {areas_png}")

    # -------------------------------------------------------------------------
    # Build family indices
    # -------------------------------------------------------------------------
    fam_members = {
        "ATM_COMP": [proc["XCO2"], proc["XCH4"], proc["AOD550"]],
        "HYDRO": [proc["PRECIP"], proc["VPD"], proc["AET"], proc["RUNOFF"], proc["SOIL_MOISTURE"], proc["TWSA"]],
        "VEG": [proc["EVI"], proc["SIF"], proc["NEE"], proc["RECO"]],
        "FIRE": [proc["BA"], proc["FIRMS"]],
        "RAD": [proc["PAR"], proc["LST"]],
    }
    fam_members = {k: [d for d in v if d is not None] for k, v in fam_members.items()}

    fam_idx = {}
    for fam, members in fam_members.items():
        fam_idx[fam] = family_index(members, reducer="median")
        log(f"[FAMILY] {fam}: members={len(members)}")

    fam_df = {}
    for fam, da in fam_idx.items():
        if da is None:
            continue
        fam_df[fam] = aggregate_by_area(da, areas, N_AREAS, reducer="median")
        fam_df[fam].to_csv(OUT_DIR / f"area_timeseries_{fam}.csv", index=False)

    # -------------------------------------------------------------------------
    # Per target CCM
    # -------------------------------------------------------------------------
    panel_results = []
    summary_rows = []

    for tgt in TARGETS:
        log("")
        log("=" * 60)
        log(f"[TARGET] {tgt}")

        y = proc[tgt]
        if y is None:
            log(f"[SKIP] missing target: {tgt}")
            continue

        y_df = aggregate_by_area(y, areas, N_AREAS, reducer="median")
        y_df.to_csv(OUT_DIR / f"area_timeseries_TARGET_{tgt}.csv", index=False)

        allowed = allowed_families_for_target(tgt)
        fams_used = [f for f in allowed if f in fam_df]
        log(f"[ALLOWED] {fams_used}")

        rows = []
        area_winner = np.zeros(N_AREAS, dtype=np.int16)
        area_winner_lag = np.full(N_AREAS, np.nan, dtype=float)

        for a in range(1, N_AREAS + 1):
            y_ser = y_df[f"a{a}"].values.astype(float)
            best = None
            best_fam = None

            for fam in fams_used:
                x_ser = fam_df[fam][f"a{a}"].values.astype(float)
                res = ccm_best_pyedm(x_ser, y_ser)

                if res is None:
                    rows.append({
                        "area": a,
                        "family": fam,
                        "rhoF": np.nan,
                        "rho0": np.nan,
                        "gain": np.nan,
                        "lag": np.nan,
                        "E": np.nan,
                        "ok": 0,
                    })
                    continue

                rhoF, lag, E, rho0, gain = res
                rows.append({
                    "area": a,
                    "family": fam,
                    "rhoF": rhoF,
                    "rho0": rho0,
                    "gain": gain,
                    "lag": lag,
                    "E": E,
                    "ok": 1,
                })

                if (best is None) or (rhoF > best[0]):
                    best = res
                    best_fam = fam

            if best_fam is not None:
                area_winner[a - 1] = FAMILY_ID[best_fam]
                area_winner_lag[a - 1] = best[1]

        score_csv = OUT_DIR / f"ccm_area_scores_{tgt}.csv"
        pd.DataFrame(rows).to_csv(score_csv, index=False)
        log(f"[CSV] {score_csv}")

        # ---------------------------------------------------------------------
        # Map back to pixels
        # ---------------------------------------------------------------------
        dom = np.zeros_like(areas.values, dtype=np.int16)
        lag_map = np.full_like(areas.values, np.nan, dtype=float)
        arv = areas.values.astype(np.int16)

        for a in range(1, N_AREAS + 1):
            dom[arv == a] = area_winner[a - 1]
            lag_map[arv == a] = area_winner_lag[a - 1]

        valid_mask = arv > 0

        dom = majority_filter(dom, valid_mask, passes=1)
        dom = remove_small_islands(dom, valid_mask, min_size=30, nbh=5)

        dom_da = xr.DataArray(dom, coords=areas.coords, dims=areas.dims)
        lag_da = xr.DataArray(lag_map, coords=areas.coords, dims=areas.dims)

        dom_tif = OUT_DIR / f"dominance_{tgt}_k100.tif"
        lag_tif = OUT_DIR / f"winner_lag_{tgt}_k100.tif"
        write_geotiff_like_ref(dom_da.astype("int16"), ref2d, dom_tif, nodata=0)
        write_geotiff_like_ref(lag_da.fillna(-9999).astype("float32"), ref2d, lag_tif, nodata=-9999)

        # ---------------------------------------------------------------------
        # Figure: dominance map + lag map
        # ---------------------------------------------------------------------
        fig_path = OUT_DIR / f"dominance_lag_{tgt}_k100.png"
        fig = plt.figure(figsize=(15.0, 6.3))
        gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.0], wspace=0.10)

        ax1 = fig.add_subplot(gs[0, 0], projection=ccrs.PlateCarree())
        plot_dominance_map(
            ax=ax1,
            da2d=dom_da,
            title=f"Dominance map — {tgt}",
            plot_ctx=plot_ctx,
            legend=True
        )

        ax2 = fig.add_subplot(gs[0, 1], projection=ccrs.PlateCarree())
        im = plot_lag_map(
            ax=ax2,
            lag_da=lag_da,
            title=f"Lag of winning CCM — {tgt}",
            plot_ctx=plot_ctx
        )

        cbar = fig.colorbar(im, ax=ax2, shrink=0.82, pad=0.02)
        cbar.set_label("Lag (months)")
        cbar.set_ticks(np.arange(LAG_MIN, LAG_MAX + 1, 1))

        fig.savefig(fig_path, dpi=420, bbox_inches="tight")
        plt.close(fig)
        log(f"[FIG] {fig_path}")

        total_valid = int((dom_da.values > 0).sum())
        for fam in FAMILY_ORDER:
            cid = FAMILY_ID[fam]
            npx = int((dom_da.values == cid).sum())
            share = npx / total_valid if total_valid > 0 else np.nan
            summary_rows.append({
                "target": tgt,
                "family": fam,
                "pixels": npx,
                "share": share
            })

        panel_results.append({
            "target": tgt,
            "dom_map": dom_da
        })

    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------
    if summary_rows:
        summary_csv = OUT_DIR / "dominance_family_shares.csv"
        pd.DataFrame(summary_rows).to_csv(summary_csv, index=False)
        log(f"[CSV] {summary_csv}")

    if len(panel_results) >= 1:
        fig_png = OUT_DIR / "Figure_Dominance_6Panel_k100.png"
        fig_pdf = OUT_DIR / "Figure_Dominance_6Panel_k100.pdf"
        fig_svg = OUT_DIR / "Figure_Dominance_6Panel_k100.svg"
        make_6panel_dominance_summary(panel_results[:6], fig_png, fig_pdf, fig_svg, plot_ctx)

    log("")
    log("[DONE]")
    log(f"[OUT] {OUT_DIR}")


if __name__ == "__main__":
    main()