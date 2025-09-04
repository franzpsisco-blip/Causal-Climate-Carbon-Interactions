#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Teleconnection network extraction with PCMCI for monthly XCO₂ grids (2001–2023).

- Monthly anomalies (remove climatology) → linear detrend → z-score
- Optional robustness: remove common mode and AR(1) prewhiten
- PCMCI tests τ∈[0..tau_max]; best-lag selection prefers τ>0 unless τ=0 clearly better
- Exports: links_all.csv (best lag, unfiltered), links.csv (FDR + |β| window),
           lag_scan.csv, lag_hist.csv, region_centroids.csv, map.png, chord.png,
           run_metadata.json
- Map legend: **fixed 0–6 months**, one entry each; line width encodes lag.
"""

from __future__ import annotations

import warnings
from pathlib import Path
import json
from typing import List, Tuple, Dict, Any

import numpy as np
import pandas as pd
import xarray as xr
import rioxarray as rxr
import rasterio.features
from rasterio.enums import Resampling
from scipy.ndimage import label as cc_label
from sklearn.decomposition import PCA
from sklearn.utils.extmath import svd_flip
from statsmodels.stats.multitest import multipletests
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
from matplotlib.lines import Line2D
from shapely.geometry import shape as shapely_shape

# Tigramite
from tigramite.data_processing import DataFrame as TGData
from tigramite.pcmci import PCMCI
try:
    from tigramite.independence_tests.gpdc import GPDC
    HAS_GPDC = True
except Exception:
    HAS_GPDC = False

from tigramite.independence_tests.parcorr import ParCorr

# Optional chord
try:
    from pycirclize import Circos
    HAS_CIRCLIZE = True
except Exception:
    HAS_CIRCLIZE = False

# Optional: silence scikit optimizer warnings
try:
    from sklearn.exceptions import ConvergenceWarning
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
except Exception:
    pass

# ----------------------------- Configuration ---------------------------------
o_dir = Path(r'D:/DATA_NATURE_SIF/XCO2_2014-2024_merged/results')
o_dir.mkdir(parents=True, exist_ok=True)

CFG: Dict[str, Any] = {
    # >>>> UPDATE THIS PATH if needed <<<<
    'tif_path': Path(r'D:/DATA_NATURE_SIF/ALL_DAILY_STACKED/daily/monthly/monthly 2000-2023/New folder/resampled/XCO2_all.tif'),
    # Outputs
    'geojson_path': o_dir / 'areas_auto.geojson',
    'out_dir': o_dir,
    # Time window (monthly)
    'date_start': '2001-01-01',
    'date_end'  : '2023-12-01',
    'freq': 'MS',
    # PCA / Areas
    'retain_var': 0.95,
    'min_pc': 3,
    'max_pc': 10,
    'round_dec': 2,
    'min_area_size': 100,
    # PCMCI test range
    'include_tau0_in_pcmci': True,
    'tau_max': 6,
    'pc_alpha': 0.2,
    'indep_test': 'parcorr',     # 'parcorr' (robust) or 'gpdc'
    # Best-lag SELECTION policy
    'exclude_tau0_from_selection': False,
    'prefer_lagged': True,
    'tau0_rel_improve': 0.5,
    'p_tie_tol': 1e-6,
    # Multiple testing / |β| window (for links.csv + map)
    'fdr_q': 0.20,
    'min_abs_beta': 0.45,
    'max_abs_beta': None,
    'reject_negative_betas': False,  # set True to plot only positive β
    # Data quality & preprocessing
    'min_ts_valid_ratio': 0.95,
    'remove_common_mode': True,
    'common_mode_method': 'mean',   # 'mean' or 'median'
    'prewhiten_ar1': True,
    # Plotting
    'map_figsize': (8, 4),
    'dpi': 600,
    'curvature': 0.20,
    'trend_units': 'ppm/month',
    'random_state': 0,
}

plt.rcParams.update({'font.family': 'serif', 'font.size': 10})

# ------------------------------- Utilities -----------------------------------
def sub(i: int) -> str:
    digits = '₀₁₂₃₄₅₆₇₈₉'
    return ''.join(digits[int(d)] for d in str(i))

def to_da(array: np.ndarray, like: xr.DataArray) -> xr.DataArray:
    return xr.DataArray(array, dims=('y', 'x'), coords={'y': like.y, 'x': like.x})

# ---------------------------- Data IO & prep ---------------------------------
def _align_dates_to_raster(n_bands: int, dates: pd.DatetimeIndex) -> pd.DatetimeIndex:
    if n_bands == len(dates):
        return dates
    n = min(n_bands, len(dates))
    print(f"[warn] Bands ({n_bands}) != dates ({len(dates)}). Clipping to {n}.")
    return dates[:n]

def load_and_align(path: Path, dates: pd.DatetimeIndex) -> xr.DataArray:
    """Load multiband GeoTIFF, reproject to WGS84 if needed, attach dates."""
    da = rxr.open_rasterio(path, masked=True).astype('float32')
    if da.rio.crs and da.rio.crs.to_epsg() != 4326:
        da = da.rio.reproject('EPSG:4326', Resampling.bilinear)
    band_dim = da.dims[0]
    if band_dim != 'time':
        da = da.rename({band_dim: 'time'})
    dates2 = _align_dates_to_raster(da.sizes['time'], dates)
    da = da.isel(time=slice(0, len(dates2))).assign_coords(time=dates2)
    return da.squeeze(drop=True)

def _safe_zscore(da: xr.DataArray) -> xr.DataArray:
    mean = da.mean('time'); std = da.std('time')
    std = xr.where(std == 0, np.nan, std)
    return (da - mean) / std

def compute_anomalies(da: xr.DataArray) -> xr.DataArray:
    """Monthly climatology anomalies → linear detrend → z-score."""
    clim = da.groupby('time.month').mean('time')
    anom = da.groupby('time.month') - clim
    t = xr.DataArray(np.arange(len(anom.time), dtype=float), dims='time', coords={'time': anom.time})
    coeff = anom.polyfit(dim='time', deg=1).polyfit_coefficients
    detr = anom - (coeff.sel(degree=0) + coeff.sel(degree=1) * t)
    return _safe_zscore(detr)

# --------------------------- PCA & region extraction --------------------------
def varimax(phi: np.ndarray, gamma: float = 1, q: int = 100, tol: float = 1e-6) -> np.ndarray:
    p, k = phi.shape; R = np.eye(k)
    for _ in range(q):
        L = phi @ R
        u, _, vh = np.linalg.svd(phi.T @ (L**3 - (gamma/p) * L * np.sum(L**2, 0)), full_matrices=False)
        Rn = u @ vh
        if np.allclose(R, Rn, atol=tol): break
        R = Rn
    return phi @ R

def extract_pcs(da: xr.DataArray) -> np.ndarray:
    """Return rotated loadings as array (y, x, k)."""
    T, Y, X = da.sizes['time'], da.sizes['y'], da.sizes['x']
    mat = da.values.reshape(T, Y * X)
    mask = np.isfinite(mat).all(axis=0)
    vmat = mat[:, mask]
    if vmat.size == 0:
        raise ValueError("All grid cells contain NaNs across time after preprocessing.")
    pca_full = PCA(random_state=CFG['random_state']).fit(vmat)
    evr = np.cumsum(pca_full.explained_variance_ratio_)
    k = int(np.clip(np.searchsorted(evr, CFG['retain_var']) + 1, CFG['min_pc'], CFG['max_pc']))
    pca = PCA(n_components=k, random_state=CFG['random_state']).fit(vmat)
    scores, comps = svd_flip(pca.transform(vmat), pca.components_)
    loadings = comps.T
    load_rot = varimax(loadings)
    full = np.full((Y * X, k), np.nan, dtype=np.float32)
    full[mask] = load_rot
    return full.reshape(Y, X, k)

def extract_areas(load3d: np.ndarray, da: xr.DataArray) -> Tuple[np.ndarray, Dict[int, int]]:
    # Use cached polygons if present
    if Path(CFG['geojson_path']).exists():
        gj = json.loads(Path(CFG['geojson_path']).read_text())
        tr = da.rio.transform()
        shapes = ((f['geometry'], int(f['properties']['area_id'])) for f in gj['features'])
        amap = rasterio.features.rasterize(
            shapes, out_shape=(da.sizes['y'], da.sizes['x']), transform=tr, fill=0, dtype='int32'
        )
        pcmap = {int(f['properties']['area_id']): int(f['properties']['component']) for f in gj['features']}
        return amap, pcmap
    Y, X, K = load3d.shape
    mode = np.nanargmax(np.abs(load3d), axis=2)
    rows = np.arange(Y)[:, None]; cols = np.arange(X)[None, :]
    dom_val = load3d[rows, cols, mode]
    qval = np.round(dom_val, CFG['round_dec'])
    amap = np.zeros((Y, X), dtype=np.int32)
    pcmap: Dict[int, int] = {}
    aid = 0
    for m in range(K):
        sel_m = (mode == m) & np.isfinite(qval)
        if not sel_m.any(): continue
        for v in np.unique(qval[sel_m]):
            if np.isnan(v): continue
            blobs, _ = cc_label((mode == m) & (qval == v))
            for b in range(1, blobs.max() + 1):
                idx = (blobs == b)
                if idx.sum() < CFG['min_area_size']: continue
                aid += 1; amap[idx] = aid; pcmap[aid] = m
    feats = []
    tr = da.rio.transform()
    for geom, val in rasterio.features.shapes(amap.astype('int32'), transform=tr):
        if val:
            feats.append({'type': 'Feature',
                          'properties': {'area_id': int(val), 'component': int(pcmap[int(val)])},
                          'geometry': geom})
    Path(CFG['geojson_path']).write_text(json.dumps({'type': 'FeatureCollection', 'features': feats}))
    return amap, pcmap

def area_timeseries(da: xr.DataArray, amap: np.ndarray) -> Tuple[np.ndarray, List[int], List[int]]:
    T = len(da.time); mat = da.values.reshape(T, -1)
    area_ids = sorted(set(amap.ravel()) - {0})
    series, kept, dropped = [], [], []
    for aid in area_ids:
        idx = (amap.ravel() == aid)
        s = np.nanmean(mat[:, idx], axis=1)
        valid_ratio = np.isfinite(s).mean()
        if valid_ratio < CFG['min_ts_valid_ratio']:
            dropped.append(aid); continue
        sm = s - np.nanmean(s); ss = np.nanstd(sm)
        if not np.isfinite(ss) or ss == 0:
            dropped.append(aid); continue
        series.append(sm / ss); kept.append(aid)
    if not series:
        raise ValueError("All areas dropped due to insufficient valid data.")
    S = np.vstack(series).T
    return S, kept, dropped

# ------------------------ Optional transforms (robust) ------------------------
def remove_common_mode(S: np.ndarray, method: str = 'mean') -> np.ndarray:
    """Regress out the scene-wide common mode per time step."""
    if method not in ('mean', 'median'): method = 'mean'
    c = np.nanmean(S, axis=1) if method == 'mean' else np.nanmedian(S, axis=1)
    c = (c - np.nanmean(c)) / (np.nanstd(c) if np.nanstd(c) else 1.0)
    C = c[:, None]; CtC = float(C.T @ C)
    if CtC == 0 or not np.isfinite(CtC): return S
    beta = (C.T @ S) / CtC
    R = S - C @ beta
    R = (R - np.nanmean(R, axis=0)) / np.nanstd(R, axis=0)
    return R

def prewhiten_ar1(S: np.ndarray) -> np.ndarray:
    """AR(1) prewhitening per column with finite-sample safeguard."""
    T, N = S.shape; W = np.empty_like(S)
    for j in range(N):
        x = S[:, j] - np.nanmean(S[:, j])
        if np.nanstd(x) == 0 or not np.all(np.isfinite(x)):
            W[:, j] = x; continue
        num = float(np.dot(x[1:], x[:-1])); den = float(np.dot(x[:-1], x[:-1]))
        phi = num / den if den != 0 else 0.0; phi = float(np.clip(phi, -0.99, 0.99))
        y = x.copy(); y[1:] = x[1:] - phi * x[:-1]
        y = (y - np.mean(y)) / (np.std(y) if np.std(y) else 1.0)
        W[:, j] = y
    return W

# --------------------------------- PCMCI -------------------------------------
def _make_indep_test():
    if str(CFG['indep_test']).lower() == 'gpdc' and HAS_GPDC:
        return GPDC(significance='analytic')
    return ParCorr(significance='analytic')

def run_pcmci_raw(S: np.ndarray) -> Tuple[np.ndarray, np.ndarray, int]:
    """Run PCMCI and return (p_matrix, val_matrix, tau_min_used)."""
    tau_min = 0 if bool(CFG['include_tau0_in_pcmci']) else 1
    indep = _make_indep_test()
    pcmci = PCMCI(dataframe=TGData(S), cond_ind_test=indep, verbosity=0)
    res = pcmci.run_pcmci(tau_min=tau_min, tau_max=int(CFG['tau_max']), pc_alpha=float(CFG['pc_alpha']))
    p = np.asarray(res['p_matrix']); b = np.asarray(res['val_matrix'])
    return p, b, tau_min

def _tie_break(k_candidates: List[int], pv: np.ndarray, bv: np.ndarray, tau_min: int) -> int:
    """Break ties: max |β|, then smaller absolute lag."""
    if len(k_candidates) == 1: return k_candidates[0]
    absb = [abs(bv[k]) for k in k_candidates]; max_abs = max(absb)
    ks = [k for k, a in zip(k_candidates, absb) if a == max_abs]
    if len(ks) == 1: return ks[0]
    ks.sort(key=lambda k: (tau_min + k))
    return ks[0]

def bestlag_table(p: np.ndarray, b: np.ndarray, tau_min_pcmci: int) -> List[Tuple[int, int, int, float, float]]:
    """
    Return [(i,j,lag,p,beta)] keeping only the **best lag** per pair.

    Selection rule:
      - If CFG['exclude_tau0_from_selection'] True → only k>0 (fallback to 0 if none).
      - Else prefer τ>0 unless τ=0 clearly dominates: p0 <= ratio * p(best τ>0).
      - Tie-breaks by larger |β|, then smaller lag.
    """
    N = p.shape[0]; rows: List[Tuple[int, int, int, float, float]] = []
    allow_tau0_tested = (tau_min_pcmci == 0)
    exclude_tau0 = bool(CFG.get('exclude_tau0_from_selection', False))
    prefer_lagged = bool(CFG.get('prefer_lagged', False))
    ratio = float(CFG.get('tau0_rel_improve', 0.5))
    tol = float(CFG.get('p_tie_tol', 1e-6))
    for i in range(N):
        for j in range(N):
            if i == j: continue
            pv = p[i, j, :].astype(float); bv = b[i, j, :].astype(float)
            finite_k = [k for k in range(pv.size) if np.isfinite(pv[k])]
            if not finite_k: continue
            cand = finite_k
            if exclude_tau0 and allow_tau0_tested:
                cand = [k for k in cand if k > 0] or [0]
            minp = min(pv[k] for k in cand)
            k_min_all = [k for k in cand if abs(pv[k] - minp) <= tol]
            k_star = _tie_break(k_min_all, pv, bv, tau_min_pcmci)
            if allow_tau0_tested and not exclude_tau0 and prefer_lagged and pv.size >= 2 and np.isfinite(pv[0]):
                finite_pos = [k for k in cand if k > 0]
                if finite_pos:
                    minp_pos = min(pv[k] for k in finite_pos)
                    k_pos_all = [k for k in finite_pos if abs(pv[k] - minp_pos) <= tol]
                    k_pos_star = _tie_break(k_pos_all, pv, bv, tau_min_pcmci)
                    if not (pv[0] <= ratio * pv[k_pos_star]):
                        k_star = k_pos_star
            lag = tau_min_pcmci + k_star
            rows.append((i, j, lag, float(pv[k_star]), float(bv[k_star])))
    return rows

def select_links_fdr_bestlag(p: np.ndarray, b: np.ndarray, tau_min_pcmci: int) -> List[Tuple[int, int, int, float]]:
    """FDR over best-lag p-values; then |β| window; optional sign guard for map/chord."""
    rows = bestlag_table(p, b, tau_min_pcmci)
    if not rows: return []
    pvals = [r[3] for r in rows]
    rej, _, _, _ = multipletests(pvals, alpha=float(CFG['fdr_q']), method='fdr_bh')
    min_b = float(CFG['min_abs_beta']) if CFG['min_abs_beta'] is not None else 0.0
    max_b = float(CFG['max_abs_beta']) if CFG.get('max_abs_beta') not in (None, '') else np.inf
    drop_neg = bool(CFG.get('reject_negative_betas', False))
    links: List[Tuple[int, int, int, float]] = []
    for (i, j, lag, pval, beta), keep in zip(rows, rej):
        ab = abs(beta)
        if keep and (ab >= min_b) and (ab <= max_b):
            if drop_neg and beta < 0: continue
            links.append((i, j, lag, beta))
    return links

# -------------------------------- Plotting -----------------------------------
def compute_trend_raster(da: xr.DataArray) -> xr.DataArray:
    t = np.arange(len(da.time))
    def slope(y: np.ndarray) -> float:
        mask = np.isfinite(y)
        if mask.sum() <= 1: return np.nan
        return np.polyfit(t[mask], y[mask], 1)[0]
    return xr.apply_ufunc(slope, da, input_core_dims=[["time"]], vectorize=True, output_dtypes=[float])

def plot_map(aids: List[int], amap: np.ndarray, da: xr.DataArray,
             links: List[Tuple[int, int, int, float]], tau_min: int, out_png: Path) -> None:
    # Trend background from original series
    raw_dates = pd.date_range(CFG['date_start'], CFG['date_end'], freq=CFG['freq'])
    raw = load_and_align(Path(CFG['tif_path']), raw_dates)
    slp = compute_trend_raster(raw)
    cond = to_da((amap > 0).astype(bool), raw)
    slp = xr.where(cond, slp, np.nan)
    vmin, vmax = np.nanpercentile(slp.values, [0.25, 99.75])
    fig, ax = plt.subplots(figsize=CFG['map_figsize'], dpi=CFG['dpi'],
                           subplot_kw={'projection': ccrs.Robinson()})
    ax.set_global(); ax.coastlines(resolution='110m', linewidth=0.25)
    ax.add_feature(cfeature.BORDERS, linewidth=0.25)
    im = ax.contourf(raw.x, raw.y, slp, 60, cmap='RdYlBu_r',
                     transform=ccrs.PlateCarree(), extend='both',
                     alpha=0.6, vmin=vmin, vmax=vmax)
    plt.colorbar(im, ax=ax, fraction=0.02, pad=0.01).set_label(f'Trend ({CFG['trend_units']})')
    # Region centers
    Lon, Lat = np.meshgrid(da.x, da.y)
    cents = {i: (float(Lon[amap == aid].mean()), float(Lat[amap == aid].mean())) for i, aid in enumerate(aids)}
    # Draw links: linewidth encodes lag, color encodes |β|
    if links:
        absbetas = [abs(b) for *_, b in links]
        vmin_b, vmax_b = (min(absbetas), max(absbetas)) if absbetas else (0, 1)
        cmap = plt.get_cmap('viridis'); norm = plt.Normalize(vmin_b, vmax_b)
        for s, t_, lag, beta in links:
            x0, y0 = cents[s]; x1, y1 = cents[t_]
            ax.annotate(
                '', xy=(x1, y1), xytext=(x0, y0), transform=ccrs.PlateCarree(),
                arrowprops={
                    'arrowstyle': '-|>',
                    'color': cmap(norm(abs(beta))),
                    'linewidth': 0.75 + 0.5 * int(lag),  # width by integer lag
                    'alpha': 0.85,
                    'connectionstyle': f'arc3,rad={CFG["curvature"]}',
                    'shrinkA': 10, 'shrinkB': 10
                }
            )
        # --------- FIXED LEGEND: exactly one entry per month 0..tau_max ----------
        handles = [Line2D([0], [0], color='gray',
                          lw=0.75 + 0.5 * m, label=f'{m} mo') for m in range(0, int(CFG['tau_max']) + 1)]
        lg = ax.legend(handles=handles, title='Lag', loc='lower left',
                       frameon=False, bbox_to_anchor=(-0.3, -0.3))
        lg.set_zorder(5)
        # ------------------------------------------------------------------------
        plt.colorbar(plt.cm.ScalarMappable(cmap=cmap, norm=norm), ax=ax,
                     orientation='horizontal', fraction=0.04, pad=0.09).set_label('|β|')
    else:
        ax.text(0.5, 0.02, 'No significant links at current thresholds',
                transform=ax.transAxes, ha='center', va='bottom', fontsize=9)
    # Draw area outlines & labels
    gj = json.loads(Path(CFG['geojson_path']).read_text())
    for feat in gj['features']:
        geom = shapely_shape(feat['geometry'])
        ax.add_geometries([geom], ccrs.PlateCarree(),
                          facecolor='none', edgecolor='black', linewidth=0.25)
    subs = [f'X{sub(i + 1)}' for i in range(len(aids))]
    for i, (x, y) in cents.items():
        ax.text(x, y, subs[i], transform=ccrs.PlateCarree(),
                ha='center', va='center', fontsize=9, fontweight='bold')
    gl = ax.gridlines(draw_labels=True, linewidth=0.2, linestyle='--', color='gray', alpha=0.5)
    gl.top_labels = False; gl.right_labels = False
    gl.xformatter, gl.yformatter = LONGITUDE_FORMATTER, LATITUDE_FORMATTER
    tail = 'τ tested ≥0' if int(tau_min) == 0 else 'τ tested ≥1'
    ax.set_xlabel('Longitude'); ax.set_ylabel('Latitude')
    ax.set_title(f'Teleconnection areas – best-lag links ({tail})')
    fig.tight_layout(); fig.savefig(out_png, dpi=CFG['dpi']); plt.close(fig)

# --------------------------------- Chord -------------------------------------
def chord_diagram(M: pd.DataFrame, out_png: Path) -> None:
    if not HAS_CIRCLIZE:
        fig = plt.figure(figsize=(4, 4), dpi=CFG['dpi'])
        plt.text(0.5, 0.5, 'pycirclize not installed', ha='center', va='center')
        fig.savefig(out_png, dpi=CFG['dpi']); plt.close(fig); return
    keep = (M.sum(axis=1) + M.sum(axis=0)) > 0
    if not keep.any():
        fig = plt.figure(figsize=(4, 4), dpi=CFG['dpi'])
        plt.text(0.5, 0.5, 'No links', ha='center', va='center')
        fig.savefig(out_png, dpi=CFG['dpi']); plt.close(fig); return
    M2 = M.loc[keep, keep].astype(float)
    subs = [f'X{sub(int(lbl[1:]))}' for lbl in M2.index]
    M2.index = subs; M2.columns = subs
    cmap = {subs[i]: plt.cm.tab20(i) for i in range(len(subs))}
    Circos.initialize_from_matrix(
        M2, space=5, cmap=cmap,
        label_kws={'size': 22},
        link_kws={'ec': 'gray', 'lw': 0.4, 'alpha': 0.7, 'direction': 1}
    ).savefig(out_png, dpi=CFG['dpi'])

# ------------------------------ Debug/metadata --------------------------------
def export_lag_scan(p: np.ndarray, b: np.ndarray, tau_min: int, labels: List[str], out_path: Path) -> None:
    """Export (src,tgt,lag,p,beta) for every tested lag."""
    rows = []
    N, L = p.shape[0], p.shape[2]
    for i in range(N):
        for j in range(N):
            if i == j: continue
            for k in range(L):
                rows.append((labels[i], labels[j], tau_min + k,
                             float(p[i, j, k]), float(b[i, j, k])))
    pd.DataFrame(rows, columns=['src', 'tgt', 'lag', 'p', 'beta']).to_csv(out_path, index=False)

def export_lag_hist(rows_best: List[Tuple[int,int,int,float,float]], out_path: Path) -> None:
    """Save histogram of selected lags."""
    if not rows_best:
        pd.DataFrame(columns=['lag','count']).to_csv(out_path, index=False); return
    lags = [lag for _,_,lag,_,_ in rows_best]
    vc = pd.Series(lags).value_counts().sort_index()
    vc.rename_axis('lag').reset_index(name='count').to_csv(out_path, index=False)

def _json_safe(x):
    try:
        if isinstance(x, (np.floating, np.integer)): return x.item()
    except Exception:
        pass
    if isinstance(x, float) and (np.isinf(x) or np.isnan(x)): return str(x)
    return x

def save_debug_outputs(S: np.ndarray, p: np.ndarray, b: np.ndarray, labels: List[str], tau_min: int) -> None:
    out_dir = Path(CFG['out_dir']); out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(S, columns=labels).rename_axis('time_index').to_csv(out_dir / 'area_timeseries_standardized.csv')
    np.save(out_dir / 'pcmci_p_matrix.npy', p)
    np.save(out_dir / 'pcmci_beta_matrix.npy', b)
    meta = {
        'date_start': CFG['date_start'], 'date_end': CFG['date_end'], 'freq': CFG['freq'],
        'tau_min_pcmci': int(tau_min), 'tau_max': CFG['tau_max'], 'pc_alpha': CFG['pc_alpha'], 'fdr_q': CFG['fdr_q'],
        'min_abs_beta': CFG['min_abs_beta'],
        'max_abs_beta': (None if CFG.get('max_abs_beta') in (None,'') else float(CFG['max_abs_beta'])),
        'min_ts_valid_ratio': CFG['min_ts_valid_ratio'],
        'retain_var': CFG['retain_var'], 'min_pc': CFG['min_pc'], 'max_pc': CFG['max_pc'],
        'round_dec': CFG['round_dec'], 'min_area_size': CFG['min_area_size'], 'random_state': CFG['random_state'],
        'include_tau0_in_pcmci': bool(CFG['include_tau0_in_pcmci']),
        'exclude_tau0_from_selection': bool(CFG['exclude_tau0_from_selection']),
        'prefer_lagged': bool(CFG['prefer_lagged']),
        'tau0_rel_improve': CFG['tau0_rel_improve'],
        'p_tie_tol': CFG['p_tie_tol'],
        'remove_common_mode': bool(CFG['remove_common_mode']),
        'common_mode_method': CFG['common_mode_method'],
        'prewhiten_ar1': bool(CFG['prewhiten_ar1']),
        'indep_test': CFG['indep_test'],
        'reject_negative_betas': bool(CFG.get('reject_negative_betas', False)),
    }
    meta = {k: _json_safe(v) for k, v in meta.items()}
    (out_dir / 'run_metadata.json').write_text(json.dumps(meta, indent=2))

# ---------------------------------- Main -------------------------------------
def main() -> None:
    dates = pd.date_range(CFG['date_start'], CFG['date_end'], freq=CFG['freq'])
    # 1) Load & preprocess anomalies
    da_raw = load_and_align(Path(CFG['tif_path']), dates)
    da = compute_anomalies(da_raw)
    # 2) Regions
    pcs = extract_pcs(da)
    amap, _ = extract_areas(pcs, da)
    # 3) Area time series (standardized)
    S, kept_ids, dropped_ids = area_timeseries(da, amap)
    # Optional robustness for XCO₂
    if bool(CFG['remove_common_mode']):
        S = remove_common_mode(S, method=str(CFG['common_mode_method']))
    if bool(CFG['prewhiten_ar1']):
        S = prewhiten_ar1(S)
    # Export centroids (kept areas)
    LonM, LatM = np.meshgrid(da.x.values, da.y.values)
    coords = [(f'X{i+1}', float(LonM[amap==aid].mean()), float(LatM[amap==aid].mean()), int(aid))
              for i, aid in enumerate(kept_ids)]
    pd.DataFrame(coords, columns=['region','lon','lat','area_id']).to_csv(Path(CFG['out_dir'])/'region_centroids.csv', index=False)
    # 4) PCMCI
    p, b, tau_min_pcmci = run_pcmci_raw(S)
    # (A) links_all.csv = best-lag only, unfiltered
    rows_all = bestlag_table(p, b, tau_min_pcmci)
    labels = [f'X{i+1}' for i in range(len(kept_ids))]
    df_all = pd.DataFrame([(labels[i], labels[j], int(lag), pval, beta)
                           for i,j,lag,pval,beta in rows_all],
                          columns=['src','tgt','lag','p','beta'])
    df_all.to_csv(Path(CFG['out_dir'])/'links_all.csv', index=False)
    # Full lag scan & histogram
    export_lag_scan(p, b, tau_min_pcmci, labels, Path(CFG['out_dir'])/'lag_scan.csv')
    export_lag_hist(rows_all, Path(CFG['out_dir'])/'lag_hist.csv')
    # (B) links.csv = best-lag with FDR + |β| window (+ optional sign guard)
    links = select_links_fdr_bestlag(p, b, tau_min_pcmci)
    df_best = pd.DataFrame([(labels[s], labels[t], int(lag), beta) for s,t,lag,beta in links],
                           columns=['src','tgt','lag','beta'])
    df_best.to_csv(Path(CFG['out_dir'])/'links.csv', index=False)
    # 5) Visualizations
    plot_map(kept_ids, amap, da, links, tau_min_pcmci, Path(CFG['out_dir'])/'map.png')
    # Chord diagram uses |β|
    Mdf = pd.DataFrame(0.0, index=labels, columns=labels)
    for s,t,lag,beta in links:
        Mdf.iat[s,t] = abs(beta)
    chord_diagram(Mdf, Path(CFG['out_dir'])/'chord.png')
    # 6) Debug & metadata
    save_debug_outputs(S, p, b, labels, tau_min_pcmci)
    # 7) Summary
    print(f"Kept areas: {len(kept_ids)} | Dropped: {len(dropped_ids)}")
    print(f"Best-lag pairs (links_all.csv): {len(rows_all)} | Filtered links (links.csv): {len(links)}")
    tail = 'τ tested ≥0' if int(tau_min_pcmci) == 0 else 'τ tested ≥1'
    print(f"PCMCI lag test range: {tail} .. {int(CFG['tau_max'])}")

if __name__ == '__main__':
    main()
