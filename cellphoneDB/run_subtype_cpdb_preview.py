from pathlib import Path
import os

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
import matplotlib.pyplot as plt
import omicverse as ov


PROJECT = Path(__file__).resolve().parents[1]
CPDB_ZIP = PROJECT / "cellphoneDB" / "cellphonedb.zip"
OUT_ROOT = PROJECT / "fig" / "cellphoneDB" / "LT_subtype_preview"
CPDB_OUT = PROJECT / "cellphoneDB" / "results" / "LT_subtype_preview"

GROUP = "LT"
GROUP_KEY = "group"
CELLTYPE_KEY = "celltype"
MAX_CELLS_PER_TYPE = 250
ITERATIONS = 50
RANDOM_STATE = 20260611
DRY_RUN = os.environ.get("CPDB_DRY_RUN", "0") == "1"

INPUTS = [
    PROJECT / "h5ad" / "sft.h5ad",
    PROJECT / "h5ad" / "myeloid.h5ad",
    PROJECT / "h5ad" / "T_NK.h5ad",
    PROJECT / "h5ad" / "CAFs.h5ad",
]

KEEP_CELLTYPES = [
    # SFT tumor subtypes
    "Reactive SFT",
    "ECM-remodeling SFT",
    "COL17A1+ SFT",
    "MEST+ SFT",
    "NGFR+ SFT",
    "Cycling SFT",
    # Myeloid subtypes
    "TREM2+ TAM",
    "SPP1+ lipid TAM",
    "Inflammatory TAM",
    "LYVE1+ resident TAM",
    "MMP9+ osteoclast-like TAM",
    "HLA-high APC",
    "FCN1+ monocyte",
    # T/NK subtypes
    "FGFBP2+ cytotoxic NK",
    "GZMK+ CD8 T",
    # CAFs, if the CAF object has a single retained label
    "CAFs",
]

COLORS = {
    "Reactive SFT": "#4C5F8F",
    "ECM-remodeling SFT": "#A66C9A",
    "COL17A1+ SFT": "#B7794B",
    "MEST+ SFT": "#6FA083",
    "NGFR+ SFT": "#B95F62",
    "Cycling SFT": "#5BA8A0",
    "TREM2+ TAM": "#4C5F8F",
    "SPP1+ lipid TAM": "#B95F62",
    "Inflammatory TAM": "#D8B56D",
    "LYVE1+ resident TAM": "#6FA083",
    "MMP9+ osteoclast-like TAM": "#5BA8A0",
    "HLA-high APC": "#A66C9A",
    "FCN1+ monocyte": "#8DA6BF",
    "FGFBP2+ cytotoxic NK": "#6FA083",
    "GZMK+ CD8 T": "#9BAE6A",
    "CAFs": "#8DA6BF",
}

STORY_KEYWORDS = [
    "Collagen",
    "Fibronectin",
    "Laminin",
    "Osteopontin",
    "Transforming growth factor",
    "Chemokines",
    "Interleukin",
    "VEGF",
    "PDGF",
    "MIF",
    "Integrin",
    "Notch",
    "Complement",
    "HLA",
]


def mkdir(path):
    Path(path).mkdir(parents=True, exist_ok=True)
    return Path(path)


def sample_names(obs, max_cells_per_type, seed):
    rng = np.random.default_rng(seed)
    names = []
    for celltype, idx in obs.groupby(CELLTYPE_KEY, observed=True).groups.items():
        idx = np.array(list(idx))
        if len(idx) > max_cells_per_type:
            idx = rng.choice(idx, size=max_cells_per_type, replace=False)
        names.extend(idx.tolist())
    return names


def load_one(path):
    print(f"Reading backed: {path.name}")
    b = sc.read_h5ad(path, backed="r")
    obs = b.obs.copy()
    if CELLTYPE_KEY not in obs.columns:
        raise ValueError(f"{path} has no obs['{CELLTYPE_KEY}']")
    if GROUP_KEY not in obs.columns:
        raise ValueError(f"{path} has no obs['{GROUP_KEY}']")

    obs[CELLTYPE_KEY] = obs[CELLTYPE_KEY].astype(str)
    obs[GROUP_KEY] = obs[GROUP_KEY].astype(str)
    obs = obs[(obs[GROUP_KEY] == GROUP) & obs[CELLTYPE_KEY].isin(KEEP_CELLTYPES)].copy()
    if obs.empty:
        print(f"  no retained cells in {path.name}")
        b.file.close()
        return None

    names = sample_names(obs, MAX_CELLS_PER_TYPE, RANDOM_STATE)
    x = b[names, :].to_memory()
    b.file.close()
    x.obs["source_h5ad"] = path.stem
    print(f"  loaded {x.n_obs} cells")
    print(x.obs[CELLTYPE_KEY].astype(str).value_counts())
    return x


def savefig(name):
    plt.savefig(OUT_ROOT / f"{name}.pdf", dpi=500, bbox_inches="tight")
    plt.savefig(OUT_ROOT / f"{name}.png", dpi=350, bbox_inches="tight")
    plt.close("all")


def choose_story_pathways(comm_adata, n=8):
    if "classification" not in comm_adata.var.columns:
        return []
    items = (
        comm_adata.var["classification"]
        .fillna("")
        .astype(str)
        .replace("", np.nan)
        .dropna()
        .unique()
        .tolist()
    )
    picked = []
    for keyword in STORY_KEYWORDS:
        for item in items:
            if keyword.lower() in item.lower() and item not in picked:
                picked.append(item)
    return picked[:n]


def plot_pathway_panel(adata_plot, comm_adata, pathway, color_dict):
    safe = pathway.replace("/", "_").replace("\\", "_").replace(":", "").replace(" ", "_")
    pathway_dir = mkdir(OUT_ROOT / "story_pathways" / safe)
    plotters = [
        (
            "01_sankey",
            lambda: ov.pl.ccc_stat_plot(
                adata_plot,
                plot_type="sankey",
                display_by="interaction",
                signaling=[pathway],
                palette=color_dict,
                top_n=10,
                figsize=(8, 6),
                show=False,
            ),
        ),
        (
            "02_lr_contribution",
            lambda: ov.pl.ccc_stat_plot(
                adata_plot,
                plot_type="lr_contribution",
                signaling=pathway,
                figsize=(9, 4.8),
                show=False,
            ),
        ),
        (
            "03_chord",
            lambda: ov.pl.ccc_network_plot(
                comm_adata,
                plot_type="chord",
                signaling=pathway,
                palette=color_dict,
                normalize_to_sender=True,
                figsize=(6, 6),
                show=False,
            ),
        ),
        (
            "04_dot_heatmap",
            lambda: ov.pl.ccc_heatmap(
                adata_plot,
                plot_type="dot",
                sender_use=list(color_dict.keys()),
                receiver_use=list(color_dict.keys()),
                display_by="interaction",
                signaling=pathway,
                top_n=12,
                transpose=True,
                cmap="mako",
                figsize=(12, 7),
                top_anno="cell",
                left_anno="cell",
                show=False,
            ),
        ),
    ]
    for name, func in plotters:
        try:
            func()
            plt.savefig(pathway_dir / f"{name}.pdf", dpi=500, bbox_inches="tight")
            plt.savefig(pathway_dir / f"{name}.png", dpi=350, bbox_inches="tight")
            plt.close("all")
        except Exception as exc:
            print(f"[skip] {pathway} {name}: {type(exc).__name__}: {exc}")


def main():
    mkdir(OUT_ROOT)
    mkdir(CPDB_OUT)

    adatas = [x for x in (load_one(p) for p in INPUTS) if x is not None]
    if not adatas:
        raise RuntimeError("No cells retained for CPDB preview.")

    adata = ad.concat(adatas, join="inner", merge="same")
    adata.obs_names_make_unique()
    adata.obs[CELLTYPE_KEY] = adata.obs[CELLTYPE_KEY].astype(str)

    present = [x for x in KEEP_CELLTYPES if x in set(adata.obs[CELLTYPE_KEY])]
    adata.obs[CELLTYPE_KEY] = pd.Categorical(adata.obs[CELLTYPE_KEY], categories=present)
    adata = adata[~adata.obs[CELLTYPE_KEY].isna()].copy()
    adata.uns[f"{CELLTYPE_KEY}_colors"] = [COLORS.get(x, "#999999") for x in adata.obs[CELLTYPE_KEY].cat.categories]
    color_dict = dict(zip(adata.obs[CELLTYPE_KEY].cat.categories, adata.uns[f"{CELLTYPE_KEY}_colors"]))

    print("Final CPDB preview shape:", adata.shape)
    counts = adata.obs[CELLTYPE_KEY].value_counts()
    print(counts)
    counts.to_csv(OUT_ROOT / "input_cell_counts.csv")
    adata.write_h5ad(CPDB_OUT / "LT_subtype_preview_input.h5ad")
    if DRY_RUN:
        print("Dry run finished before CellPhoneDB.")
        return

    ov.plot_set(font_path="Arial")
    cpdb_results, adata_cpdb = ov.single.run_cellphonedb_v5(
        adata,
        cpdb_file_path=str(CPDB_ZIP),
        celltype_key=CELLTYPE_KEY,
        min_cell_fraction=0.005,
        min_genes=200,
        min_cells=3,
        iterations=ITERATIONS,
        threshold=0.1,
        pvalue=0.05,
        threads=10,
        output_dir=str(CPDB_OUT),
        cleanup_temp=True,
    )

    adata_plot = adata if "cpdb_results" in adata.uns else adata_cpdb
    comm_adata = (
        ov.single.extract_comm_adata(adata, result_uns_key="cpdb_results")
        if "cpdb_results" in adata.uns
        else adata_cpdb
    )
    comm_adata.write_h5ad(CPDB_OUT / "LT_subtype_preview_comm_adata.h5ad")

    if "classification" in comm_adata.var.columns:
        pd.Series(sorted(comm_adata.var["classification"].fillna("").astype(str).unique())).to_csv(
            OUT_ROOT / "available_pathways.csv", index=False, header=["pathway"]
        )

    try:
        ov.pl.ccc_heatmap(
            adata_plot,
            plot_type="dot",
            display_by="aggregation",
            cmap="YlGnBu",
            figsize=(8, 6),
            show=False,
        )
        savefig("00_overall_dot_heatmap")
    except Exception as exc:
        print(f"[skip] overall heatmap: {type(exc).__name__}: {exc}")

    try:
        ov.pl.ccc_stat_plot(adata_plot, plot_type="scatter", figsize=(5.5, 5.5), show=False)
        savefig("01_incoming_outgoing_scatter")
    except Exception as exc:
        print(f"[skip] scatter: {type(exc).__name__}: {exc}")

    story_pathways = choose_story_pathways(comm_adata, n=8)
    pd.Series(story_pathways).to_csv(OUT_ROOT / "selected_story_pathways.csv", index=False, header=["pathway"])
    print("Selected story pathways:", story_pathways)
    for pathway in story_pathways:
        plot_pathway_panel(adata_plot, comm_adata, pathway, color_dict)

    print(f"Done. Figures: {OUT_ROOT}")
    print(f"Raw CPDB results: {CPDB_OUT}")


if __name__ == "__main__":
    main()
