



from __future__ import annotations
import argparse
from pathlib import Path
from typing import List, Dict

import numpy as np
import pandas as pd
import xarray as xr
import rioxarray as rxr
import shapefile
from shapely.geometry import shape, mapping
from shapely.ops import unary_union

import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from scipy import stats

from tigramite import data_processing as pp
from tigramite.pcmci import PCMCI
from tigramite.independence_tests.parcorr import ParCorr
from tigramite.independence_tests.gpdc import GPDC
from tigramite.independence_tests.robust_parcorr import RobustParCorr

# CONFIGURATION
BASE_DIR   = Path("D:/DATA_NATURE_SIF/ALL_DAILY_STACKED/daily/monthly")
OUT_DIR    = BASE_DIR / "pcmci_results"
FIG_DIR    = OUT_DIR  / "figures"
DATES      = pd.date_range("2014-01-01", "2024-03-01", freq="MS")
TAU_MAX    = 3
PC_ALPHA   = 0.2
FDR_ALPHA  = 0.2
RHO_THRESH = 0.2

# Plot parameters
mpl.rcParams.update({
    "font.family"    : "serif",
    "font.size"      : 10,
    "axes.labelsize" : 10,
    "axes.titlesize" : 12,
    "legend.fontsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
})
FIG_NET = (5, 5)
FIG_SUM = (4, 3)
ALPHA_NODE = 0.2

# VARIABLES & CLASSES
FILES = {
    "XCO2": "XCO2.tif", "AOD": "AOD.tif",
    "T2M": "T2M.tif",   "SP": "SP.tif",  "SR": "SR.tif",
    "GPP": "GPP.tif",   "NDVI": "NDVI.tif",
    "VT":  "VT.tif",    "VPD": "VPD.tif",
    "PET": "PET.tif",   "AET": "AET.tif",
    "RO":  "RO.tif",    "PREC": "PRECIP.tif", "PDSI": "PDSI.tif",
}
CLASSES = {
    "Atmosphere":["XCO2","AOD"],
    "Meteorology":["T2M","SP","SR"],
    "Vegetation":["GPP","NDVI","VT","VPD"],
    "Water":["RO","PREC","PDSI","PET","AET"]
}
CLASS_COLS = {
    "Atmosphere": "dimgray",
    "Meteorology":"steelblue",
    "Vegetation":"forestgreen",
    "Water":     "royalblue",
    "Other":     "lightgrey",
}

# CONTINENT GEOMETRIES
WORLD_SHP = Path("D:/World_Continents_-8398826466908339531/WGS84/world.shp")
reader    = shapefile.Reader(str(WORLD_SHP))
FIELDS    = [f[0] for f in reader.fields[1:]]
regions: List[str] = []
geoms: Dict[str,List] = {}
for rec, shp_r in zip(reader.records(), reader.shapes()):
    cont = dict(zip(FIELDS, rec)).get("CONTINENT")
    if cont and cont != "Antarctica":
        regions.append(cont)
        geoms.setdefault(cont, []).append(shape(shp_r.__geo_interface__))

regions = sorted(set(regions)) + ["Global"]
ALL_GEOM = unary_union([g for lst in geoms.values() for g in lst])

# HELPER FUNCTIONS
def monthly_anomaly(ts: pd.Series) -> pd.Series:
    ts   = ts.astype(float).ffill().bfill()
    clim = ts.groupby(ts.index.month).transform("mean")
    anom = ts - clim
    m, b = np.polyfit(np.arange(len(anom)), anom, 1)
    detr = anom - (m*np.arange(len(anom)) + b)
    sd   = detr.std(ddof=0)
    if sd > 0:
        return (detr - detr.mean()) / sd
    else:
        return pd.Series(np.nan, index=ts.index)

def vclass(var: str) -> str:
    for cls, vals in CLASSES.items():
        if var in vals:
            return cls
    return "Other"

# LOAD RASTERS
def load_rasters() -> Dict[str, rxr.DataArray]:
    rasters: Dict[str, rxr.DataArray] = {}
    for var, fn in FILES.items():
        fp = BASE_DIR / fn
        if not fp.exists():
            continue
        da = rxr.open_rasterio(fp, masked=True).squeeze()
        # ensure CRS and spatial dims are set
        if da.rio.crs is None:
            da = da.rio.write_crs("EPSG:4326")
        da.rio.set_spatial_dims(x_dim="x", y_dim="y", inplace=True)
        da = da.rename({"band": "time"}).assign_coords(time=DATES)
        # per-variable coverage mask (≥50% non-null)
        cov = da.notnull().mean("time") >= 0.5
        da  = da.where(cov)
        # skip rasters that became empty
        if da.sizes.get("x", 1) == 0 or da.sizes.get("y", 1) == 0:
            continue
        rasters[var] = da
    return rasters

# DRAW NETWORK (unchanged)
def draw_network(edges: pd.DataFrame, region: str, method: str) -> None:
    if edges.empty:
        return
    G = nx.DiGraph()
    for _, r in edges.iterrows():
        G.add_edge(r.source, r.target, lag=r.lag, rho=r.rho)
    nodes = sorted(set(edges.source) | set(edges.target))
    order = []
    for cls in ["Atmosphere","Meteorology","Vegetation","Water"]:
        order += [v for v in CLASSES[cls] if v in nodes]
    order += [v for v in nodes if v not in order]
    theta = np.linspace(0, 2*np.pi, len(order), endpoint=False)
    pos   = {v:(np.cos(t), np.sin(t)) for v, t in zip(order, theta)}
    cmap, norm = mpl.cm.RdYlBu_r, mpl.colors.Normalize(vmin=-1, vmax=1)
    bins, widths = [0,0.2,0.4,0.8,1], [0.5,1,2,4]
    styles = {
        0:"solid", 1:(0,(1,1)), 2:"dashed", 3:"dotted",
        4:"dashdot", 5:(0,(3,1,1,1)), 6:(0,(5,1))
    }
    fig, ax = plt.subplots(figsize=FIG_NET, subplot_kw={"aspect":"equal"})
    ax.axis('off')
    for lag, st in styles.items():
        eds = [(u,v) for u,v,d in G.edges(data=True) if d['lag']==lag]
        if not eds:
            continue
        cols = [cmap(norm(G[u][v]['rho'])) for u,v in eds]
        wds  = [widths[np.digitize(abs(G[u][v]['rho']), bins)-1] for u,v in eds]
        nx.draw_networkx_edges(
            G, pos, edgelist=eds, edge_color=cols,
            width=wds, style=st, alpha=0.8,
            arrowsize=14, arrowstyle='-|>', connectionstyle='arc3,rad=0.2'
        )
    node_cols = [mpl.colors.to_rgba(CLASS_COLS[vclass(v)], ALPHA_NODE) for v in order]
    nx.draw_networkx_nodes(G, pos, nodelist=order,
                           node_color=node_cols, edgecolors='k', node_size=1100)
    nx.draw_networkx_labels(G, pos, font_size=8)
    sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cb = fig.colorbar(sm, ax=ax, shrink=0.3, pad=0.01, anchor=(1,1))
    cb.ax.tick_params(labelsize=10)
    cb.set_label('ρ', fontsize=10)
    lag_lines = [
        mpl.lines.Line2D([0],[0], color='grey', lw=2, ls=styles[l], label=f'{l} mo')
        for l in styles
    ]
    rho_lines = [
        mpl.lines.Line2D([0],[0], color='grey', lw=w, label=f'|ρ|≥{bins[i]:.2f}')
        for i,w in enumerate(widths)
    ]
    cls_lines = [
        mpl.lines.Line2D(
            [0],[0], marker='o', markersize=7, linestyle='None',
            markerfacecolor=mpl.colors.to_rgba(c, ALPHA_NODE),
            markeredgecolor='k', label=k
        )
        for k,c in CLASS_COLS.items() if k != 'Other'
    ]
    ax.legend(handles=rho_lines, title='|ρ| bins',
              loc='lower center', bbox_to_anchor=(0.5,-0.12),
              ncol=len(rho_lines), frameon=False)
    ax.add_artist(ax.legend(handles=cls_lines, title='Class',
                            loc='lower right', bbox_to_anchor=(1.28,-0.1),
                            frameon=False))
    ax.add_artist(ax.legend(handles=lag_lines, title='Lag',
                            loc='upper right', bbox_to_anchor=(1.28,0.7),
                            handlelength=3, frameon=False))
    ax.set_title(f"{region} PCMCI (τ≤{TAU_MAX})", pad=12)
    outdir = FIG_DIR / method
    outdir.mkdir(parents=True, exist_ok=True)
    plt.tight_layout(pad=0.6)
    plt.savefig(outdir / f"network_{region}_{method}.png", dpi=400)
    plt.close()

# RUN PCMCI PER REGION
def run_pcmci(region: str,
              rasters: Dict[str, rxr.DataArray],
              test,
              method: str,
              summary: List[pd.DataFrame],
              draw_net: bool) -> None:
    geom = ALL_GEOM if region == 'Global' else unary_union(geoms[region])
    df = pd.DataFrame(index=DATES)
    for var, da in rasters.items():
        # clip by region if not global
        if region == 'Global':
            sub = da
        else:
            sub = da.rio.clip([mapping(geom)], da.rio.crs, drop=False)
        if {'x','y'} - set(sub.dims) or sub.size == 0:
            continue
        # area‑weighted spatial mean
        w = np.cos(np.deg2rad(sub['y']))   # dims = ('y',)
        # broadcast to full grid (y, x)
        w2 = xr.DataArray(
            np.broadcast_to(w.values[:, None], (w.sizes['y'], sub.sizes['x'])),
            coords={'y': sub['y'], 'x': sub['x']},
            dims=('y','x')
        )
        ts_arr = (sub * w2).sum(dim=('y','x'), skipna=True)
        denom  = w2.sum(dim=('y','x'))
        if denom == 0:
            continue
        s = (ts_arr / denom).to_series()
        if s.notna().sum() < TAU_MAX + 2:
            continue
        df[var] = monthly_anomaly(s)
    df.dropna(axis=1, how='all', inplace=True)
    df.dropna(inplace=True)
    if df.shape[1] < 2 or len(df) < TAU_MAX + 2:
        return
    pcmci = PCMCI(
        dataframe=pp.DataFrame(df.values,
                               datatime=np.arange(len(df)),
                               var_names=list(df.columns)),
        cond_ind_test=test
    )
    res = pcmci.run_pcmci(tau_max=TAU_MAX, pc_alpha=PC_ALPHA)
    val = res['val_matrix']
    pm  = res['p_matrix']
    sig = pcmci.get_corrected_pvalues(pm, fdr_method='fdr_bh') <= FDR_ALPHA
    mask = sig & (np.abs(val) >= RHO_THRESH)
    links = [
        {'source': s, 'target': t, 'lag': int(l), 'rho': float(val[i,j,l])}
        for i,s in enumerate(df.columns)
        for j,t in enumerate(df.columns)
        if s != t
        for l in np.where(mask[i,j])[0]
    ]
    od = OUT_DIR / method
    od.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(links).to_csv(od / f"edges_{region}_{method}.csv", index=False)
    best: Dict[tuple,str] = {}
    for e in links:
        k = (e['source'], e['target'])
        if k not in best or abs(e['rho']) > abs(best[k]['rho']):
            best[k] = e
    df_best = pd.DataFrame(best.values())
    if not df_best.empty:
        summary.append(pd.DataFrame({
            'variable': pd.concat([df_best.source, df_best.target]),
            'abs_rho' : df_best.rho.abs().repeat(2).values,
            'method'  : method
        }))
        if draw_net:
            draw_network(df_best, region, method)

# SUMMARY BOX-PLOT
def summary_plot(rows: List[pd.DataFrame]) -> None:
    if not rows:
        return
    df = pd.concat(rows, ignore_index=True)
    order = [
        'AET','AOD','GPP','NDVI','PDSI','PET','PREC',
        'RO','SIF','SP','SR','T2M','VPD','VT','XCO2'
    ]
    df['variable'] = pd.Categorical(df['variable'], categories=order, ordered=True)
    palette = {'ParCorr':'#1f77b4','GPDC':'#ff7f0e','RobustParCorr':'#2ca02c'}
    sns.set_style('white')
    fig, ax = plt.subplots(figsize=FIG_SUM)
    sns.boxplot(
        data=df, x='variable', y='abs_rho', hue='method',
        palette=palette, showcaps=False, showfliers=False,
        medianprops={'linewidth':0.5},
        whiskerprops={'linewidth':0.5},
        ax=ax
    )
    ax.set_ylim(0,1)
    ax.set_xlabel('Variable')
    ax.set_ylabel('Partial correlation, |ρ|')
    ax.tick_params(axis='x', rotation=90)
    ax.legend(title='Method', frameon=False, ncol=3, loc='lower right')
    sns.despine(trim=True)
    outfn = FIG_DIR / 'partial_correlation_boxplot.png'
    fig.tight_layout(pad=0.25)
    fig.savefig(outfn, dpi=600)
    plt.close()

# ENTRY POINT
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--no-network', action='store_true',
                        help='skip network diagrams')
    args = parser.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    rasters = load_rasters()
    tests = {
        'ParCorr'        : ParCorr(significance='analytic'),
        'GPDC'           : GPDC(significance='analytic'),
        'RobustParCorr'  : RobustParCorr(significance='analytic')
    }
    summary: List[pd.DataFrame] = []
    for method, test in tests.items():
        for region in regions:
            run_pcmci(region, rasters, test, method,
                      summary, draw_net=not args.no_network)
    summary_plot(summary)
    print('Done.')

if __name__ == '__main__':
    main()

