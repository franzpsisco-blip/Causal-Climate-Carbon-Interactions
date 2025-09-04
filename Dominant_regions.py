# ============================================================
# Dominant controls of XCO2 (2001–2023): ANTH | BIOS | HYDM | HYDE
# Panels a–f with per-panel legend/cbar rows, 600 dpi export.
# Robust to Cartopy versions (no outline_patch usage).
# Optional world shapefile border overlay.
# ============================================================

# --------- USER PATHS ---------
from pathlib import Path
BASE_DIR = Path(r"D:/DATA_NATURE_SIF/ALL_DAILY_STACKED/daily/monthly/monthly 2000-2023/New folder/resampled")
OUT_DIR  = Path(r"D:/DATA_NATURE_SIF/ALL_DAILY_STACKED/daily/monthly/monthly 2000-2023/New folder/resampled/figures")
PCMCI_MASK_DIR = ""  # optional folder with p_<VARNAME>_to_XCO2.tif; leave empty to skip
WORLD_SHP = Path(r"D:/World_Continents_-8398826466908339531/WGS84/world.shp")  # optional

# --------- PARAMETERS ----------
MAX_LAG_MAIN  = 6
MAX_LAG_SENS  = 3
Q_TOP         = 90
SMOOTH_MAPS   = True
GAUSS_SIGMA   = 0.8
ARID_BINS     = (0.1, 1.4, 14)

# --------- FILES ----------
FILES = {
    "XCO2":"XCO2.tif",
    "EDGAR":"EDGAR.tif",
    "GPP":"GPP.tif","NDVI":"NDVI.tif","LAI":"LAI.tif",
    "PDSI":"PDSI.tif","SOIL":"SOIL.tif","PR":"PR.tif","AET":"AET.tif","RUNOFF":"RUNOFF.tif",
    "VPD":"VPD.tif","PET":"PET.tif","T2M":"T2M.tif","SRAD":"SRAD.tif",
}

# --------- IMPORTS ----------
import os, warnings
import numpy as np, pandas as pd
import xarray as xr, rioxarray as rxr
import matplotlib as mpl, matplotlib.pyplot as plt
from scipy import ndimage as ndi
from scipy.signal import savgol_filter
warnings.filterwarnings("ignore", category=RuntimeWarning)

_HAS_CARTOPY = True
try:
    import cartopy.crs as ccrs, cartopy.feature as cfeature
    from cartopy.io import shapereader as shpreader
except Exception:
    _HAS_CARTOPY = False

# --------- STYLE ----------
mpl.rcParams.update({
    "figure.dpi": 300,
    "savefig.dpi": 600,      # final export
    "font.family": "serif", "font.size": 11.5,
    "axes.labelsize": 13, "axes.titlesize": 13,
    "xtick.labelsize": 11, "ytick.labelsize": 11,
    "legend.fontsize": 11, "axes.linewidth": 0.9,
    "figure.constrained_layout.use": False,
})

COLORS = {"ANTH": "#006d77", "BIOS": "#1b9e77", "HYDM": "#e69f00", "HYDE": "#6a51a3"}

CAT_ORDER = ["ANTH","BIOS","HYDM","HYDE"]

def cmap_categories(names):
    cmap = mpl.colors.ListedColormap([COLORS[n] for n in names])
    norm = mpl.colors.BoundaryNorm(np.arange(-0.5, len(names)+0.5, 1), len(names))
    return cmap, norm

def cmap_seq():
    cm = mpl.colormaps["viridis"].copy()
    cm.set_bad("#e6e6e6")
    return cm

# -------------- HELPERS --------------
def _time_index(): return pd.date_range("2001-01-01","2023-12-01",freq="MS")

def load_stack(path: Path) -> xr.DataArray:
    da = rxr.open_rasterio(path, masked=True)
    if da.dims[0] != "time": da = da.rename({da.dims[0]:"time"})
    n = min(da.sizes["time"], len(_time_index()))
    return da.isel(time=slice(0,n)).assign_coords(time=_time_index()[:n]).astype("float32").squeeze(drop=True)

def anomalies_detrend_z(da: xr.DataArray) -> xr.DataArray:
    clim = da.groupby("time.month").mean("time")
    anom = da.groupby("time.month") - clim
    t = xr.DataArray(np.arange(anom.sizes["time"], dtype=float), dims="time", coords={"time": anom.time})
    coeff = anom.polyfit(dim="time", deg=1).polyfit_coefficients
    detr = anom - (coeff.sel(degree=0) + coeff.sel(degree=1)*t)
    m = detr.mean("time"); s = detr.std("time"); s = xr.where(s==0, np.nan, s)
    return (detr - m) / s

def corr_nan(x,y):
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum()<3: return np.nan
    xs, ys = x[m], y[m]
    sx, sy = xs.std(), ys.std()
    if sx==0 or sy==0: return np.nan
    return float(np.corrcoef(xs,ys)[0,1])

def best_lag_corr(driver: xr.DataArray, target: xr.DataArray, max_lag: int):
    T,Y,X = driver.sizes["time"], driver.sizes["y"], driver.sizes["x"]
    drv = driver.values.reshape(T,-1); tgt = target.values.reshape(T,-1)
    r = np.full(drv.shape[1], np.nan, dtype="float32")
    k = np.full(drv.shape[1], np.nan, dtype="float32")
    for j in range(drv.shape[1]):
        x = drv[:,j]; y = tgt[:,j]
        if not (np.isfinite(x).any() and np.isfinite(y).any()): continue
        r_star = np.nan; k_star = np.nan
        for lag in range(max_lag+1):
            if T-lag<3: continue
            rr = corr_nan(x[:T-lag], y[lag:])
            if np.isnan(rr): continue
            if np.isnan(r_star) or abs(rr)>abs(r_star):
                r_star, k_star = rr, lag
        r[j], k[j] = r_star, k_star
    r_da = xr.DataArray(r.reshape(Y,X), coords={"y":driver.y,"x":driver.x}, dims=("y","x"))
    k_da = xr.DataArray(k.reshape(Y,X), coords={"y":driver.y,"x":driver.x}, dims=("y","x"))
    return r_da, k_da

def coslat_weights(y,x):
    lat_rad = np.deg2rad(y.values)[:,None]
    w = np.cos(lat_rad)*np.ones((y.size,x.size), dtype="float32")
    return xr.DataArray(w, coords={"y":y,"x":x}, dims=("y","x"))

def lon_profile(absr, w):
    lon = absr["x"].values; A = np.abs(absr.values); W = w.values
    num = np.nansum(A*W, axis=0); den = np.nansum(W*np.isfinite(A), axis=0)
    prof = np.divide(num, den, out=np.full_like(num, np.nan), where=den>0)
    return lon, prof

def minmax01(arr, qlo=5, qhi=95):
    lo, hi = np.nanpercentile(arr, [qlo,qhi])
    return np.clip((arr-lo)/(hi-lo+1e-9), 0, 1)

def gaussian_nan(a, sigma):
    mask = np.isfinite(a).astype(float)
    num = ndi.gaussian_filter(np.nan_to_num(a, nan=0.0)*mask, sigma=sigma)
    den = ndi.gaussian_filter(mask, sigma=sigma)
    return np.divide(num, den, out=np.full_like(a,np.nan,dtype=float), where=den>0)

def savgol_safe(y, win=19, poly=2):
    win = max(5, min(win, (len(y)//2)*2-1))
    if win <= poly: return y
    try: return savgol_filter(y, win, poly, mode="interp")
    except: return y

def xr_max_all(das):
    out = das[0]
    for d in das[1:]:
        out = xr.ufuncs.maximum(out, d)
    return out

def top_mask_exact(score_arr, q, valid_mask):
    arr = score_arr.copy()
    arr[~valid_mask] = np.nan
    thr = np.nanpercentile(arr, q)
    return (arr >= thr) & valid_mask, thr

# -------------- MAIN --------------
OUT_DIR.mkdir(parents=True, exist_ok=True)
paths = {k: BASE_DIR/v for k,v in FILES.items()}
for req in ["XCO2","EDGAR","GPP","NDVI","LAI","PDSI","SOIL","PR","AET","RUNOFF","VPD","PET","T2M","SRAD"]:
    if not paths[req].exists(): raise FileNotFoundError(f"Missing {req}: {paths[req]}")

X   = load_stack(paths["XCO2"])
drv = {nm: load_stack(paths[nm]) for nm in FILES if nm!="XCO2" and paths[nm].exists()}
Xz   = anomalies_detrend_z(X)
drvz = {k: anomalies_detrend_z(v) for k,v in drv.items()}

r_k_main = {k: best_lag_corr(da, Xz, MAX_LAG_MAIN) for k,da in drvz.items()}
r_k_sens = {k: best_lag_corr(da, Xz, MAX_LAG_SENS) for k,da in drvz.items()}

def _mask_from(dirn, fname, templ):
    if not dirn: return xr.ones_like(templ)
    p = Path(dirn)/fname
    if not p.exists(): return xr.ones_like(templ)
    da = rxr.open_rasterio(p, masked=True).squeeze(drop=True).astype("float32")
    try: M = da.rio.reproject_match(templ, resampling=0)
    except: M = da
    return xr.where(M<0.10, 1.0, np.nan)

sig = {k: _mask_from(PCMCI_MASK_DIR, f"p_{k}_to_XCO2.tif", r_k_main[k][0]) for k in drvz.keys()}

def _abs_r(name): return np.abs(r_k_main[name][0])

ANTH = _abs_r("EDGAR") * sig.get("EDGAR",1.0)

BIOS = xr_max_all([_abs_r("GPP"), _abs_r("NDVI"), _abs_r("LAI")]) * xr_max_all([sig.get("GPP",1.0), sig.get("NDVI",1.0), sig.get("LAI",1.0)])

HYDM = xr_max_all([_abs_r("PDSI"), _abs_r("SOIL"), _abs_r("PR"), _abs_r("AET"), _abs_r("RUNOFF")]) * \
       xr_max_all([sig.get("PDSI",1.0), sig.get("SOIL",1.0), sig.get("PR",1.0), sig.get("AET",1.0), sig.get("RUNOFF",1.0)])

HYDE = xr_max_all([_abs_r("VPD"), _abs_r("PET"), _abs_r("T2M"), _abs_r("SRAD")]) * \
       xr_max_all([sig.get("VPD",1.0), sig.get("PET",1.0), sig.get("T2M",1.0), sig.get("SRAD",1.0)])

GROUPS = {"ANTH":ANTH, "BIOS":BIOS, "HYDM":HYDM, "HYDE":HYDE}

SCORES = {k: xr.apply_ufunc(minmax01, v) for k,v in GROUPS.items()}
DomScore = None
for k in CAT_ORDER:
    DomScore = SCORES[k] if DomScore is None else xr.ufuncs.maximum(DomScore, SCORES[k])

stack = xr.concat([GROUPS[k] for k in CAT_ORDER], dim="cls")
all_nan = xr.ufuncs.isnan(stack).all("cls")
cat = stack.fillna(-1.0).argmax("cls").astype("float32").where(~all_nan)

land_mask = np.zeros_like(cat.values, dtype=bool)
for nm, da in drv.items():
    land_mask |= np.isfinite(da.isel(time=0).values) if "time" in da.dims else np.isfinite(da.values)

def apply_land_mask_float(da):
    arr = da.values.astype("float32", copy=True); arr[~land_mask] = np.nan
    return xr.DataArray(arr, coords=da.coords, dims=da.dims)

cat_m      = apply_land_mask_float(cat)
DomScore_m = apply_land_mask_float(DomScore)

def _abs_r2(name): return np.abs(r_k_sens[name][0])

GROUPS2 = {
    "ANTH":_abs_r2("EDGAR"),
    "BIOS":xr_max_all([_abs_r2("GPP"), _abs_r2("NDVI"), _abs_r2("LAI")]),
    "HYDM":xr_max_all([_abs_r2("PDSI"), _abs_r2("SOIL"), _abs_r2("PR"), _abs_r2("AET"), _abs_r2("RUNOFF")]),
    "HYDE":xr_max_all([_abs_r2("VPD"), _abs_r2("PET"), _abs_r2("T2M"), _abs_r2("SRAD")]),
}

DomScore2 = None
for k in CAT_ORDER:
    s = xr.apply_ufunc(minmax01, GROUPS2[k])
    DomScore2 = s if DomScore2 is None else xr.ufuncs.maximum(DomScore2, s)

valid = np.isfinite(DomScore.values) & np.isfinite(DomScore2.values)
Aq,_ = top_mask_exact(DomScore.values, Q_TOP, valid)
Bq,_ = top_mask_exact(DomScore2.values, Q_TOP, valid)
agree = np.zeros_like(DomScore.values, dtype="float32")
agree[Aq & ~Bq] = 1; agree[~Aq & Bq] = 2; agree[Aq & Bq] = 3
agree_m = apply_land_mask_float(xr.DataArray(agree, coords={"y":X.y,"x":X.x}, dims=("y","x")))

w = coslat_weights(X.y, X.x); W=w.values; tot=np.nansum(W)
shares = {k: np.nansum(W[cat_m.values==i])/tot*100.0 for i,k in enumerate(CAT_ORDER)}

member_map = {
    "ANTH":["EDGAR"],
    "BIOS":["GPP","NDVI","LAI"],
    "HYDM":["PDSI","SOIL","PR","AET","RUNOFF"],
    "HYDE":["VPD","PET","T2M","SRAD"],
}

def best_lag_for_group(members, class_mask):
    Rs = [np.abs(r_k_main[m][0]) for m in members]
    Rmax = xr_max_all(Rs)
    kstar = xr.full_like(Rmax, np.nan)
    for Ri, m in zip(Rs, members):
        ki = r_k_main[m][1]
        kstar = xr.where((Ri==Rmax) & np.isfinite(Ri), ki, kstar)
    return kstar.where(class_mask)

best_lag_by_class={}
for i,k in enumerate(CAT_ORDER):
    mask = xr.DataArray((cat.values==i), coords=cat.coords, dims=cat.dims)
    best_lag_by_class[k] = best_lag_for_group(member_map[k], mask).values

lon_profiles={}
for i,k in enumerate(CAT_ORDER):
    Rval = xr_max_all([np.abs(r_k_main[m][0]) for m in member_map[k]])
    lon, prof = lon_profile(Rval.where(cat==i), w)
    lon_profiles[k]=(lon, savgol_safe(prof, win=19, poly=2))

AET_mean = drv["AET"].mean("time").values
PET_mean = drv["PET"].mean("time").values
aridity  = np.divide(AET_mean, PET_mean, out=np.full_like(AET_mean,np.nan), where=(PET_mean>0))
arid_bins    = np.linspace(*ARID_BINS)
arid_centers = 0.5*(arid_bins[:-1] + arid_bins[1:])
cl_fracs = {k:[] for k in CAT_ORDER}
for b0,b1 in zip(arid_bins[:-1], arid_bins[1:]):
    sel = (aridity>=b0) & (aridity<b1) & land_mask & np.isfinite(cat_m.values)
    denom = np.nansum(sel)
    if denom < 1:
        for k in CAT_ORDER: cl_fracs[k].append(np.nan)
        continue
    for i,k in enumerate(CAT_ORDER):
        cl_fracs[k].append(np.nansum(sel & (cat_m.values==i))/denom*100.0)

# ===================== FIGURE (a–f) ======================
fig = plt.figure(figsize=(17.6, 10.0))
G = fig.add_gridspec(2, 3, height_ratios=[1.05, 1.00], hspace=0.22, wspace=0.25)

from matplotlib.patches import Patch
handles = [Patch(facecolor=COLORS[k], edgecolor="none", label=k) for k in CAT_ORDER]

# helper: draw map (no outline_patch); optional shapefile border overlay
def draw_map(subspec, data, cmap, norm=None, title_text=""):
    ocean_face, land_edge = "#f4f7fb", "0.35"
    if _HAS_CARTOPY:
        proj = ccrs.Robinson()
        ax = fig.add_subplot(subspec, projection=proj)
        ax.set_global(); ax.set_facecolor(ocean_face)
        ax.add_feature(cfeature.LAND, facecolor="#f0f0f0", zorder=0)
        ax.coastlines(color=land_edge, linewidth=0.45, zorder=1)
        ax.add_feature(cfeature.BORDERS, linewidth=0.25, edgecolor="0.4", zorder=1)
        # optional world shapefile border
        try:
            if WORLD_SHP.exists():
                geoms = list(shpreader.Reader(str(WORLD_SHP)).geometries())
                ax.add_geometries(geoms, ccrs.PlateCarree(),
                                  facecolor="none", edgecolor="0.25",
                                  linewidth=0.6, zorder=2)
        except Exception:
            pass
        im = ax.pcolormesh(data["x"], data["y"], data, cmap=cmap, norm=norm,
                           transform=ccrs.PlateCarree(), shading="auto", zorder=3)
    else:
        ax = fig.add_subplot(subspec)
        im = ax.imshow(data.values, origin="upper", cmap=cmap, norm=norm, aspect="auto")
        for s in ax.spines.values(): s.set_visible(False)
        ax.set_xticks([]); ax.set_yticks([])
    if title_text: ax.set_title(title_text, pad=6)
    return ax, im

# (a)
gs_a = G[0,0].subgridspec(2,1, height_ratios=[15,1], hspace=0.06)
cmap_cat, norm_cat = cmap_categories(CAT_ORDER)
axA, imA = draw_map(gs_a[0,0], cat_m, cmap_cat, norm_cat, "a. Dominant class")
axAleg = fig.add_subplot(gs_a[1,0]); axAleg.axis("off")
axAleg.legend(handles=handles, loc="center", ncol=4, frameon=False)

# --------- Panel (b) : Dominance score with world shapefile border ---------
from pathlib import Path

gs_b = G[0,1].subgridspec(2,1, height_ratios=[15,1.6], hspace=0.08)

# display smoothing only
ds_disp = gaussian_nan(DomScore_m.values, GAUSS_SIGMA) if SMOOTH_MAPS else DomScore_m.values
dom_for_plot = xr.DataArray(ds_disp, coords=DomScore_m.coords, dims=DomScore_m.dims)

# main map
axB, imB = draw_map(
    gs_b[0,0],
    dom_for_plot,
    cmap_seq(),              # sequential cmap
    None,
    "b. Dominance score"
)

# add world shapefile border on top (if Cartopy is available and file exists)
if _HAS_CARTOPY:
    try:
        from cartopy.io import shapereader as shpreader
        world_shp = Path(r"D:/World_Continents_-8398826466908339531/WGS84/world.shp")
        if world_shp.exists():
            reader = shpreader.Reader(str(world_shp))
            axB.add_geometries(
                reader.geometries(),
                ccrs.PlateCarree(),
                facecolor="none",
                edgecolor="0.20",   # dark gray border
                linewidth=0.6,
                zorder=4            # above the raster
            )
    except Exception:
        pass  # keep plotting even if shapefile loading fails

# legend/colorbar strip
caxB = fig.add_subplot(gs_b[1,0])
cbarB = fig.colorbar(imB, cax=caxB, orientation="horizontal")
cbarB.set_label("Normalized dominance score")

# --------- Panel (c) ---------
gs_c = G[0,2].subgridspec(2,1, height_ratios=[15,1.6], hspace=0.08)

# Simplified 2-color colormap (gray = no agreement, green = agreement)
cmap_agree = mpl.colors.ListedColormap(["#d9d9d9", "#2ca02c"])
norm_agree = mpl.colors.BoundaryNorm([-0.5, 0.5, 1.5], 2)

axC, imC = draw_map(gs_c[0,0], agree_m.where(agree_m > 0), cmap_agree, norm_agree, "c. Agreement")

# Legend instead of colorbar (clearer for binary map)
axCleg = fig.add_subplot(gs_c[1,0]); axCleg.axis("off")
axCleg.legend(
    handles=[
        Patch(facecolor="#d9d9d9", edgecolor="none", label="No agreement"),
        Patch(facecolor="#2ca02c", edgecolor="none", label="Agreement"),
    ],
    loc="center", ncol=2, frameon=False
)

# (d)
gs_d = G[1,0].subgridspec(2,1, height_ratios=[15,2], hspace=0.25)  # was [15,1.2], hspace=0.06
axD = fig.add_subplot(gs_d[0,0])
for k in CAT_ORDER:
    lon, prof = lon_profiles[k]
    axD.plot(lon, np.abs(prof), lw=2, color=COLORS[k], label=k)

axD.set_xlabel("Longitude (°)"); axD.set_ylabel("Area-weighted |r*|")
axD.set_title("d. Longitudinal profiles", pad=4); axD.grid(ls=":", lw=0.6)
axDleg = fig.add_subplot(gs_d[1,0]); axDleg.axis("off")
axDleg.legend(handles=handles, loc="center", ncol=4, frameon=False)

# (e)
gs_e = G[1,1].subgridspec(2,1, height_ratios=[15,2], hspace=0.25)  # same fix
axE = fig.add_subplot(gs_e[0,0])
lag_bins = np.arange(0, MAX_LAG_MAIN+2) - 0.5; lag_centers = np.arange(0, MAX_LAG_MAIN+1)
width = 0.85 / len(CAT_ORDER)
for i,k in enumerate(CAT_ORDER):
    arr = best_lag_by_class[k]
    hist,_ = np.histogram(arr[np.isfinite(arr)], bins=lag_bins)
    axE.bar(lag_centers + (i - (len(CAT_ORDER)-1)/2)*width, hist, width=width, color=COLORS[k])

axE.set_xlabel("Best lag (months)"); axE.set_ylabel("Pixel count")
axE.set_title("e. Best lag distribution", pad=4)
axEleg = fig.add_subplot(gs_e[1,0]); axEleg.axis("off")
axEleg.legend(handles=handles, loc="center", ncol=4, frameon=False)

# (f)
gs_f = G[1,2].subgridspec(2,1, height_ratios=[15,2], hspace=0.25)  # same fix
axF = fig.add_subplot(gs_f[0,0])
for k in CAT_ORDER:
    axF.plot(arid_centers, np.asarray(cl_fracs[k], float), marker="o", lw=1.8, color=COLORS[k], label=k)

axF.set_xlabel("Evaporative regime (AET/PET)"); axF.set_ylabel("Class fraction (%)")
axF.set_title("f. Aridity coupling", pad=4); axF.grid(ls=":", lw=0.6)
axFleg = fig.add_subplot(gs_f[1,0]); axFleg.axis("off")
axFleg.legend(handles=handles, loc="center", ncol=4, frameon=False)

# Save
OUT_NAME = "dominant_areas_WITH_ANTH_noCRYO_panels_a_to_f_NO_TITLE_600dpi.png"
fig.savefig(OUT_DIR / OUT_NAME, bbox_inches="tight")
plt.close(fig)
print("[done] Figure:", OUT_DIR / OUT_NAME)
