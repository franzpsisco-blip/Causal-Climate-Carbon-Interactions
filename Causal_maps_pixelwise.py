from __future__ import annotations

import json, logging, os, warnings
from dataclasses import dataclass, asdict
from datetime import datetime
from itertools import product
from pathlib import Path
from typing import Dict, Tuple, List

import numpy as np
import pandas as pd
import xarray as xr
import rioxarray as rxr
from joblib import Parallel, delayed
from scipy.ndimage import uniform_filter
from tigramite.data_processing import DataFrame as TGData
from tigramite.independence_tests.parcorr import ParCorr
from tigramite.independence_tests.robust_parcorr import RobustParCorr
try:
    from tigramite.independence_tests.gpdc import GPDC  # type: ignore
    _GPDC_AVAILABLE = True
except Exception as _gpdc_err:
    GPDC = None  # type: ignore
    _GPDC_AVAILABLE = False
    logging.warning("Initial GPDC import failed (%s). Will retry after linear tests.", _gpdc_err)

from tigramite.pcmci import PCMCI

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from tqdm import tqdm

# ------------------------------- Settings ------------------------------------
@dataclass
class Settings:
    DATA_DIR: Path = Path(r"D:/DATA_NATURE_SIF/ALL_DAILY_STACKED/daily/monthly/monthly 2000-2023/New folder/resampled")
    OUT_SUB: str = "pcmci_tau3_seq_ci"
    FILES: Dict[str, str] = None
    DATE_START: str = "2001-01-01"
    DATE_STOP: str = "2023-12-01"
    FREQ: str = "MS"
    TAU_MAX: int = 3
    PC_ALPHA: float = 0.05
    FDR_ALPHA: float = 0.05
    N_JOBS: int | None = 8
    SMOOTH_KERNEL: int = 2
    MAKE_FIGS: bool = True
    SAVE_SVG: bool = False
    SAVE_PDF: bool = True
    SAVE_PNG: bool = True
    MIN_VAR_FRACTION: float = 0.70
    INTERP_LIMIT: int | None = None
    MIN_TIMEPOINTS: int = 24
    SAVE_METADATA: bool = True
    PANEL_FONT: str = "serif"
    PANEL_SIZE: int = 10
    MASK_NON_SIGNIFICANT: bool = False
    MAX_CONST_STD: float = 1e-9
    def __post_init__(self):
        if self.FILES is None:
            self.FILES = {
                "XCO2": "XCO2.tif",
                "AOD": "AOD550.tif",
                "T2M": "T2M.tif",
                "SRAD": "SRAD.tif",
                "GPP": "GPP.tif",
                "NDVI": "NDVI.tif",
                "VPD": "VPD.tif",
                "PET": "PET.tif",
                "AET": "AET.tif",
                "RO": "RUNOFF.tif",
                "PR": "PR.tif",
                "PDSI": "PDSI.tif",
                "SOIL": "SOIL.tif",
                "SWE": "SWE.tif",
                "LAI": "LAI.tif",
                "DEF": "DEF.tif",
                "EDGAR": "EDGAR.tif",
            }

CFG = Settings()

# CI tests (run sequentially to limit RAM)
CI_TESTS: Dict[str, object] = {
    "ParCorr": ParCorr(significance="analytic"),
    "RobustParCorr": RobustParCorr(significance="analytic"),
}
if _GPDC_AVAILABLE:
    CI_TESTS["GPDC"] = GPDC(significance="analytic")  # type: ignore

# ------------------------------- Matplotlib ----------------------------------
plt.rcParams.update({
    "figure.dpi": 600,
    "savefig.dpi": 600,
    "axes.titlesize": 9,
    "axes.labelsize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "font.family": CFG.PANEL_FONT,
    "savefig.transparent": False,
})

# ------------------------------- Globals -------------------------------------
VAR_NAMES: List[str] = list(CFG.FILES.keys())
PAIR_LIST: List[Tuple[str, str]] = [
    (a, b) for a, b in product(VAR_NAMES, repeat=2)
    if a != b and ("GPP" in (a, b) or "XCO2" in (a, b) or "EDGAR" in (a, b))
]
DATE_INDEX = pd.date_range(CFG.DATE_START, CFG.DATE_STOP, freq=CFG.FREQ, name="time")

os.environ.update({"CPL_LOG": "NUL", "GDAL_SKIP": "ECW JP2ECW GEOR MrSID MSSQLSpatial OCI SOSI HDF5"})
warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

if CFG.N_JOBS is None:
    try:
        import multiprocessing as mp
        CFG.N_JOBS = max(1, mp.cpu_count() - 1)
    except Exception:
        CFG.N_JOBS = 1

# ------------------------------- IO ------------------------------------------
def load_dataset() -> xr.Dataset:
    ds_vars: Dict[str, xr.DataArray] = {}
    for var, fname in CFG.FILES.items():
        path = CFG.DATA_DIR / fname
        if not path.exists():
            logging.warning("Missing file for %s: %s (filled with NaNs)", var, path)
            template = next(iter(ds_vars.values())) if ds_vars else None
            if template is None:
                raise FileNotFoundError(f"First file {path} missing; cannot infer grid.")
            ds_vars[var] = xr.full_like(template, np.nan)
            continue
        da = rxr.open_rasterio(path, masked=True)
        da = (
            da.assign_coords(time=("band", DATE_INDEX))
              .swap_dims({"band": "time"})
              .drop_vars("band")
              .astype("float32")
        )
        ds_vars[var] = da
    ds = xr.Dataset(ds_vars)
    min_len = min(da.sizes['time'] for da in ds.data_vars.values())
    if min_len < len(DATE_INDEX):
        ds = ds.isel(time=slice(0, min_len))
    return ds

# ------------------------------- Preprocess ----------------------------------
def preprocess(ds: xr.Dataset) -> xr.Dataset:
    clim = ds.groupby("time.month").mean("time")
    anom = ds.groupby("time.month") - clim
    ti = xr.DataArray(np.arange(len(anom.time), dtype="float32"), dims="time", coords={"time": anom.time})
    detr = xr.Dataset()
    for v in VAR_NAMES:
        coef = anom[v].polyfit("time", 1, skipna=True)
        if 'polyfit_coefficients' not in coef:
            detr[v] = anom[v]; continue
        intercept = coef.polyfit_coefficients.sel(degree=0)
        slope = coef.polyfit_coefficients.sel(degree=1)
        detr[v] = anom[v] - (intercept + slope * ti)
    normed = xr.Dataset()
    for v in VAR_NAMES:
        mu = detr[v].mean("time", skipna=True)
        sd = detr[v].std("time", skipna=True) + 1e-6
        normed[v] = (detr[v] - mu) / sd
    return normed

# ------------------------------- Utils ---------------------------------------
def nan_smooth(arr: np.ndarray, k: int) -> np.ndarray:
    if k <= 1: return arr
    mask = np.isnan(arr)
    filled = np.where(mask, 0.0, arr)
    sm_sum = uniform_filter(filled, size=k, mode="nearest")
    sm_cnt = uniform_filter((~mask).astype(float), size=k, mode="nearest")
    return np.where(sm_cnt > 0, sm_sum / sm_cnt, np.nan)

def _corr(x: np.ndarray, y: np.ndarray) -> float:
    valid = (~np.isnan(x)) & (~np.isnan(y))
    if valid.sum() < 3: return np.nan
    xs, ys = x[valid], y[valid]
    if xs.std() < CFG.MAX_CONST_STD or ys.std() < CFG.MAX_CONST_STD: return np.nan
    return float(np.corrcoef(xs, ys)[0, 1])

def robust_scale(vals: np.ndarray, q: float = 99.0, eps: float = 1e-3) -> float:
    finite = vals[np.isfinite(vals)]
    if finite.size == 0: return eps
    return float(max(np.percentile(np.abs(finite), q), eps))

def safe_norm(v: float):
    try: return TwoSlopeNorm(-v, 0.0, v)
    except ValueError: return None

# ------------------------------- Pixel analysis -------------------------------
def _prepare_pixel_ts(row_da: xr.Dataset, j: int) -> pd.DataFrame:
    ts = row_da.isel(x=j).to_array().transpose("time", "variable").values
    df = (pd.DataFrame(ts, columns=VAR_NAMES)
          .interpolate(limit=CFG.INTERP_LIMIT, limit_direction="both")
          .ffill().bfill())
    T = len(df)
    keep = [v for v in VAR_NAMES if df[v].notna().sum() / T >= CFG.MIN_VAR_FRACTION]
    sub = df[keep]
    good = [v for v in sub.columns if sub[v].std(skipna=True) >= CFG.MAX_CONST_STD]
    return sub[good]

def _pcmci_run(arr: np.ndarray, cols: List[str], ci_test) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    pcmci = PCMCI(dataframe=TGData(arr, var_names=cols), cond_ind_test=ci_test, verbosity=0)
    res = pcmci.run_pcmci(tau_max=CFG.TAU_MAX, pc_alpha=CFG.PC_ALPHA)
    val, pmat = res["val_matrix"], res["p_matrix"]
    sig = pcmci.get_corrected_pvalues(pmat, fdr_method="fdr_bh") <= CFG.FDR_ALPHA
    return val, pmat, sig

def _analyse_pixel(df: pd.DataFrame, tests: Dict[str, object]) -> Dict[str, Dict[Tuple[str,str], Dict[str,float]]]:
    cols = list(df.columns)
    idx = {v: i for i, v in enumerate(cols)}
    arr = df.values.astype(float)
    pearson = {(c,e): (_corr(arr[:,idx[c]], arr[:,idx[e]]) if c in idx and e in idx else np.nan) for c,e in PAIR_LIST}
    if len(cols) < 2 or len(df) < max(CFG.MIN_TIMEPOINTS, CFG.TAU_MAX + 3):
        return {name:{p:{"pearson_r": pearson[p], "partial_r": np.nan, "lag": np.nan} for p in PAIR_LIST} for name in tests}
    results = {}
    for name, ci in tests.items():
        val, _, sig = _pcmci_run(arr, cols, ci)
        test_map = {}
        for c,e in PAIR_LIST:
            if c in idx and e in idx:
                ci_i, ei_i = idx[c], idx[e]
                lags = np.where(sig[ei_i, ci_i])[0]
                if lags.size:
                    best = int(lags[np.argmax(np.abs(val[ei_i, ci_i, lags]))])
                    prt = float(val[ei_i, ci_i, best])
                    test_map[(c,e)] = {"pearson_r": pearson[(c,e)], "partial_r": prt, "lag": best}
                else:
                    test_map[(c,e)] = {"pearson_r": pearson[(c,e)], "partial_r": np.nan, "lag": np.nan}
            else:
                test_map[(c,e)] = {"pearson_r": np.nan, "partial_r": np.nan, "lag": np.nan}
        results[name] = test_map
    return results

def _analyse_row(row_da: xr.Dataset, nx: int, tests: Dict[str, object]):
    buf = {t:{p:{k:[] for k in ("pearson_r","partial_r","lag")} for p in PAIR_LIST} for t in tests}
    for j in range(nx):
        df = _prepare_pixel_ts(row_da, j)
        res = _analyse_pixel(df, tests)
        for test_name, mapping in res.items():
            for p, vals in mapping.items():
                for k in ("pearson_r","partial_r","lag"):
                    buf[test_name][p][k].append(vals[k])
    return buf

def run_grid(ds: xr.Dataset, tests: Dict[str, object]):
    ny, nx = ds.sizes['y'], ds.sizes['x']
    keys = ["pearson_r","partial_r","lag"]
    store = {t:{p:{k:np.full((ny,nx),np.nan,dtype="float32") for k in keys} for p in PAIR_LIST} for t in tests}
    results = Parallel(n_jobs=CFG.N_JOBS, backend="loky", batch_size=1)(
        delayed(_analyse_row)(ds.isel(y=i), nx, tests) for i in tqdm(range(ny), desc="Rows")
    )
    for i, rowbuf in enumerate(results):
        for test_name, pairbuf in rowbuf.items():
            for p, kdict in pairbuf.items():
                for k, lst in kdict.items():
                    store[test_name][p][k][i,:] = np.array(lst, dtype="float32")
    return store

# ------------------------------- Export & Plot --------------------------------
def export(mapping: dict, ds: xr.Dataset, out_root: Path, test_name: str) -> None:
    coords = {"y": ds["y"], "x": ds["x"]}
    out_dirs = {k: out_root / test_name / k for k in ("npy","tif","figures")}
    for d in out_dirs.values(): d.mkdir(parents=True, exist_ok=True)
    tag = f"tau{CFG.TAU_MAX}"
    for (c,e), mats in mapping.items():
        for name, arr in mats.items():
            np.save(out_dirs["npy"] / f"{c}_{e}_{tag}_{name}.npy", arr)
            xr.DataArray(arr, coords=coords, dims=("y","x")).rio.write_crs("EPSG:4326").rio.to_raster(
                out_dirs["tif"] / f"{c}_{e}_{tag}_{name}.tif", compress="DEFLATE", nodata=np.nan)
    if CFG.SAVE_METADATA:
        settings_dict = asdict(CFG); settings_dict['DATA_DIR'] = str(settings_dict['DATA_DIR'])
        meta = {"generated": datetime.utcnow().isoformat()+'Z', "ci_test": test_name, "variables": VAR_NAMES,
                "pair_count": len(PAIR_LIST), "pairs": PAIR_LIST, "settings": settings_dict}
        with open(out_dirs["npy"]/"metadata.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

def _panel_label(ax, label):
    ax.text(0.01, 0.99, label, transform=ax.transAxes, ha='left', va='top',
            fontsize=CFG.PANEL_SIZE, fontweight='bold', fontfamily=CFG.PANEL_FONT)

def _colormaps_no_alpha():
    rd = plt.cm.get_cmap('RdBu_r').copy(); rd.set_bad(color='white', alpha=1.0)
    vr = plt.cm.get_cmap('viridis_r').copy(); vr.set_bad(color='white', alpha=1.0)
    return rd, vr

def plot(mapping: dict, ds: xr.Dataset, out_root: Path, test_name: str) -> None:
    lon, lat = ds["x"], ds["y"]
    vpr = robust_scale(np.concatenate([mapping[p]["pearson_r"].ravel() for p in PAIR_LIST]))
    vcr = robust_scale(np.concatenate([mapping[p]["partial_r"].ravel() for p in PAIR_LIST]))
    norm_pr, norm_cr = safe_norm(vpr), safe_norm(vcr)
    rd, vr = _colormaps_no_alpha()
    figs_dir = out_root / test_name / "figures"; figs_dir.mkdir(parents=True, exist_ok=True)
    exts: List[str] = []
    if CFG.SAVE_PNG: exts.append("png")
    if CFG.SAVE_PDF: exts.append("pdf")
    for (c,e), mats in mapping.items():
        pr = nan_smooth(mats["pearson_r"], CFG.SMOOTH_KERNEL)
        cr = nan_smooth(mats["partial_r"], CFG.SMOOTH_KERNEL)
        lag = mats["lag"]
        fig, axs = plt.subplots(1, 3, figsize=(9.0, 3.0), subplot_kw={"projection": ccrs.Robinson()})
        fig.patch.set_facecolor('white')
        ims = [
            axs[0].pcolormesh(lon, lat, pr, cmap=rd, norm=norm_pr, transform=ccrs.PlateCarree(), shading='auto', alpha=1.0),
            axs[1].pcolormesh(lon, lat, cr, cmap=rd, norm=norm_cr, transform=ccrs.PlateCarree(), shading='auto', alpha=1.0),
            axs[2].pcolormesh(lon, lat, lag, cmap=vr, vmin=0, vmax=CFG.TAU_MAX, transform=ccrs.PlateCarree(), shading='auto', alpha=1.0),
        ]
        titles = [f"{c} ↔ {e} Pearson r", f"{c} → {e} Partial r", f"{c} → {e} Lag (mo)"]
        for ax, im, ttl, label in zip(axs, ims, titles, ['a','b','c']):
            ax.coastlines(linewidth=0.3)
            ax.add_feature(cfeature.BORDERS, linewidth=0.2)
            ax.set_title(ttl, fontsize=9)
            cb = fig.colorbar(im, ax=ax, orientation='horizontal', fraction=0.046, pad=0.04)
            cb.ax.tick_params(labelsize=7)
            _panel_label(ax, label)
        for ext in exts:
            fig.savefig(figs_dir / f"{test_name}_{c}_{e}.{ext}", bbox_inches="tight", transparent=False)
        plt.close(fig)

# ------------------------------- GPDC retry -----------------------------------
def retry_gpdc():
    global GPDC, _GPDC_AVAILABLE
    if _GPDC_AVAILABLE:
        return
    try:
        from tigramite.independence_tests.gpdc import GPDC as _GPDC  # type: ignore
        GPDC = _GPDC  # type: ignore
        _GPDC_AVAILABLE = True
        logging.info("GPDC successfully imported on retry.")
    except Exception as e:
        logging.warning("GPDC retry failed (%s). Skipping.", e)

# ------------------------------- Main -----------------------------------------
def main():
    t0 = datetime.now()
    logging.info("Loading dataset …")
    ds_raw = load_dataset()
    logging.info("Preprocessing …")
    ds = preprocess(ds_raw)
    out_root = CFG.DATA_DIR / CFG.OUT_SUB
    # Run ParCorr & RobustParCorr first (linear / robust-linear)
    for base_test in ["ParCorr", "RobustParCorr"]:
        logging.info("Running CI test: %s", base_test)
        mapping = run_grid(ds, {base_test: CI_TESTS[base_test]})[base_test]
        export(mapping, ds, out_root, base_test)
        if CFG.MAKE_FIGS:
            plot(mapping, ds, out_root, base_test)
    # Try GPDC (nonlinear) afterwards
    if not _GPDC_AVAILABLE:
        retry_gpdc()
    if _GPDC_AVAILABLE:
        logging.info("Running CI test: GPDC")
        gpdc_test = GPDC(significance="analytic")  # type: ignore
        mapping = run_grid(ds, {"GPDC": gpdc_test})["GPDC"]
        export(mapping, ds, out_root, "GPDC")
        if CFG.MAKE_FIGS:
            plot(mapping, ds, out_root, "GPDC")
    logging.info("Completed in %s", datetime.now() - t0)

if __name__ == "__main__":
    main()
