from pathlib import Path
import os

import matplotlib.pyplot as plt
import scanpy as sc
from matplotlib.colors import LinearSegmentedColormap

import helper.plot_helper as ph


# =========================
# Config: change these for subcluster analyses
# =========================
PROJECT_DIR = Path("/mnt/d/lxk/project/jiangshucai20260506")
if not PROJECT_DIR.exists():
    PROJECT_DIR = Path("D:/lxk/project/jiangshucai20260506")
FIG_NAME = "CAFs"
H5AD_PATH = PROJECT_DIR / "h5ad" / "CAFs.h5ad"

CELLTYPE_KEY = "celltype"
UMAP_KEYS = ["celltype", "group", "sample"]
DENSITY_KEYS = ["group", "sample"]
COUNT_KEYS = ["group", "sample"]
EXPR_LAYER = "normalize"

OUT_DIR = PROJECT_DIR / "fig" / FIG_NAME
UMAP_DIR = OUT_DIR / "umap"
MARKER_DIR = OUT_DIR / "marker"
COUNT_DIR = OUT_DIR / "count"
DENSITY_DIR = OUT_DIR / "density"

marker_genes_dict = {
    "TREM2+ TAM": ["APOC1", "TREM2", "STAB1", "LTC4S", "PLTP", "CX3CR1", "CEBPA"],
    "Cycling myeloid": ["TOP2A", "TYMS", "STMN1", "HMGB2", "PCNA", "SMC4", "CKS1B"],
    "LYVE1+ resident TAM": ["CD163", "SELENOP", "F13A1", "LYVE1", "MRC1", "SIGLEC1", "KLF4"],
    "SPP1+ lipid TAM": ["GPNMB", "SPP1", "FABP5", "CD36", "LPL", "PLIN2", "LGALS3", "CTSL"],
    "HLA-high APC": ["HLA-DPB1", "HLA-DQB1", "HLA-DQA1", "HLA-DRB5", "CD74", "PLCG2"],
    "MMP9+ osteoclast-like TAM": ["MMP9", "ACP5", "CTSK", "SIGLEC15", "ATP6V0D2", "ANPEP", "DCSTAMP"],
    "FCN1+ monocyte": ["FCN1", "S100A8", "S100A9", "VCAN", "LYZ", "S100A10", "EREG", "AREG"],
    "Inflammatory TAM": ["CCL4L2", "CCL3L1", "CCL4", "CCL3", "IL1B", "TNF", "CCL2", "CH25H"],
    "LRMDA+ migratory TAM": ["LRMDA", "DOCK4", "ARHGAP24", "FRMD4A", "ELMO1", "NAV3", "KCNQ3", "SYNDIG1"],
}

marker_cmap = LinearSegmentedColormap.from_list(
    "marker_cmap",
    ["#F2F2F2", "#D8C7DD", "#9B6EA8", "#4A2C63"],
)


def make_dirs():
    for path in [UMAP_DIR, MARKER_DIR, COUNT_DIR, DENSITY_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def set_plot_style():
    sc.settings.set_figure_params(
        dpi=300,
        dpi_save=600,
        frameon=True,
        vector_friendly=True,
        format="pdf",
    )
    plt.rcParams["axes.grid"] = False
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["ps.fonttype"] = 42


def read_h5ad_safely(path):
    try:
        return sc.read_h5ad(path)
    except Exception as err:
        if "/uns/log1p" not in str(err) and "encoding_type='null'" not in str(err):
            raise

        import shutil
        import h5py

        fixed_path = path.with_name(path.stem + "_fixed_log1p.h5ad")
        if not fixed_path.exists():
            shutil.copy(path, fixed_path)
            with h5py.File(fixed_path, "r+") as handle:
                if "/uns/log1p/base" in handle:
                    del handle["/uns/log1p/base"]
        return sc.read_h5ad(fixed_path)


def save_scanpy_axes(axes, path):
    if isinstance(axes, list):
        fig = axes[0].figure
    else:
        fig = axes.figure
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_umaps(adata):
    for key in UMAP_KEYS:
        if key not in adata.obs and key not in adata.var_names:
            print(f"Skip UMAP: {key!r} not found")
            continue

        ax = sc.pl.umap(
            adata,
            color=key,
            legend_loc="right margin",
            frameon=True,
            size=3,
            show=False,
            title=key,
        )
        save_scanpy_axes(ax, UMAP_DIR / f"{FIG_NAME}_umap_{key}.pdf")


def plot_markers(adata):
    layer = EXPR_LAYER if EXPR_LAYER in adata.layers else None

    fig, ax, stat = ph.plot_marker_dotplot(
        adata,
        marker_genes_dict,
        groupby=CELLTYPE_KEY,
        layer=layer,
        scale="minmax",
        cmap=marker_cmap,
        title="Canonical markers",
        save=MARKER_DIR / f"{FIG_NAME}_marker_dotplot.pdf",
    )
    plt.close(fig)
    stat.to_csv(MARKER_DIR / f"{FIG_NAME}_marker_dotplot_values.csv", index=False)

    fig, ax, mat = ph.plot_marker_heatmap(
        adata,
        marker_genes_dict,
        groupby=CELLTYPE_KEY,
        layer=layer,
        scale="minmax",
        cmap=marker_cmap,
        title="Canonical markers",
        save=MARKER_DIR / f"{FIG_NAME}_marker_heatmap.pdf",
    )
    plt.close(fig)
    mat.to_csv(MARKER_DIR / f"{FIG_NAME}_marker_heatmap_values.csv")


def plot_counts(adata):
    for key in COUNT_KEYS:
        if key not in adata.obs:
            print(f"Skip counts: {key!r} not found")
            continue

        order = None
        if key == "group":
            order = [x for x in ["LT", "LZ"] if x in set(adata.obs[key].astype(str))]

        fig, ax, table = ph.plot_cell_counts(
            adata,
            groupby=key,
            celltype_key=CELLTYPE_KEY,
            normalize=True,
            order=order,
            title=f"Cell composition by {key}",
            save=COUNT_DIR / f"{FIG_NAME}_cell_composition_bar_{key}.pdf",
        )
        plt.close(fig)
        table.to_csv(COUNT_DIR / f"{FIG_NAME}_cell_composition_bar_{key}.csv")

        fig, axes, props = ph.plot_cell_pies(
            adata,
            groupby=key,
            celltype_key=CELLTYPE_KEY,
            order=order,
            donut=False,
            min_pct=1,
            show_pct=True,
            pct_min=5,
            ncols=3 if key == "sample" else 2,
            title=f"Cell composition by {key}",
            save=COUNT_DIR / f"{FIG_NAME}_cell_composition_pie_{key}.pdf",
        )
        plt.close(fig)
        props.to_csv(COUNT_DIR / f"{FIG_NAME}_cell_composition_pie_{key}.csv")


def plot_densities(adata):
    for key in DENSITY_KEYS:
        if key not in adata.obs:
            print(f"Skip density: {key!r} not found")
            continue

        sc.tl.embedding_density(adata, basis="umap", groupby=key)
        axes = sc.pl.embedding_density(
            adata,
            basis="umap",
            groupby=key,
            bg_dotsize=5,
            fg_dotsize=5,
            color_map="GnBu",
            show=False,
        )
        save_scanpy_axes(axes, DENSITY_DIR / f"{FIG_NAME}_density_{key}.pdf")


def main():
    os.chdir(PROJECT_DIR)
    make_dirs()
    set_plot_style()

    adata = read_h5ad_safely(H5AD_PATH)

    plot_umaps(adata)
    plot_markers(adata)
    plot_counts(adata)
    plot_densities(adata)

    print(f"Figures saved under: {OUT_DIR}")


if __name__ == "__main__":
    main()
