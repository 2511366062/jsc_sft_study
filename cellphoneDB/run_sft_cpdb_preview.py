import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt
import omicverse as ov


PROJECT = Path(__file__).resolve().parents[1]
H5AD = PROJECT / "h5ad" / "rna_0.h5ad"
CPDB_ZIP = PROJECT / "cellphoneDB" / "cellphonedb.zip"
OUT_ROOT = PROJECT / "fig" / "cellphoneDB" / "LT_major_preview"
CPDB_OUT = PROJECT / "cellphoneDB" / "results" / "LT_major_preview"

GROUP = "LT"
CELLTYPE_KEY = "celltype"
GROUP_KEY = "group"
SAMPLE_KEY = "sample"

CELLTYPES = [
    "SFT_Tumor",
    "Myeloid",
    "CAFs",
    "Endo",
    "Pericyte/SMC",
    "T/NK",
]

CELLTYPE_COLORS = {
    "SFT_Tumor": "#4C5F8F",
    "Myeloid": "#B95F62",
    "CAFs": "#8DA6BF",
    "Endo": "#A66C9A",
    "Pericyte/SMC": "#B7794B",
    "T/NK": "#6FA083",
}

MAX_CELLS_PER_TYPE = 900
RANDOM_STATE = 20260611

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
]


def mkdir(path):
    Path(path).mkdir(parents=True, exist_ok=True)
    return Path(path)


def downsample_indices(obs, key, max_per_group, seed):
    rng = np.random.default_rng(seed)
    keep = []
    for value, idx in obs.groupby(key, observed=True).groups.items():
        idx = np.array(list(idx))
        if len(idx) > max_per_group:
            idx = rng.choice(idx, size=max_per_group, replace=False)
        keep.extend(idx.tolist())
    return keep


def savefig(name):
    plt.savefig(OUT_ROOT / f"{name}.pdf", dpi=500, bbox_inches="tight")
    plt.savefig(OUT_ROOT / f"{name}.png", dpi=350, bbox_inches="tight")
    plt.close("all")


def choose_story_pathways(comm_adata, n=8):
    if "classification" not in comm_adata.var.columns:
        return []
    classifications = (
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
        for item in classifications:
            if keyword.lower() in item.lower() and item not in picked:
                picked.append(item)
    return picked[:n]


def plot_pathway_panel(adata_plot, comm_adata, pathway, color_dict):
    safe = (
        pathway.replace("/", "_")
        .replace("\\", "_")
        .replace(":", "")
        .replace(" ", "_")
    )
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
                figsize=(11, 6.5),
                top_anno="cell",
                left_anno="cell",
                show=False,
            ),
        ),
    ]

    for name, func in plotters:
        try:
            fig, ax = func()
            plt.savefig(pathway_dir / f"{name}.pdf", dpi=500, bbox_inches="tight")
            plt.savefig(pathway_dir / f"{name}.png", dpi=350, bbox_inches="tight")
            plt.close("all")
        except Exception as exc:
            print(f"[skip] {pathway} {name}: {type(exc).__name__}: {exc}")


def main():
    mkdir(OUT_ROOT)
    mkdir(CPDB_OUT)

    print(f"Reading {H5AD}")
    adata = sc.read_h5ad(H5AD)
    print("full shape:", adata.shape)
    print("obs columns:", list(adata.obs.columns))

    adata.obs[CELLTYPE_KEY] = adata.obs[CELLTYPE_KEY].astype(str)
    adata.obs[GROUP_KEY] = adata.obs[GROUP_KEY].astype(str)

    mask = (adata.obs[GROUP_KEY] == GROUP) & adata.obs[CELLTYPE_KEY].isin(CELLTYPES)
    adata = adata[mask].copy()
    print("subset shape before downsample:", adata.shape)
    print(adata.obs[CELLTYPE_KEY].value_counts())

    keep_names = downsample_indices(adata.obs, CELLTYPE_KEY, MAX_CELLS_PER_TYPE, RANDOM_STATE)
    adata = adata[keep_names].copy()
    adata.obs[CELLTYPE_KEY] = pd.Categorical(
        adata.obs[CELLTYPE_KEY],
        categories=[x for x in CELLTYPES if x in set(adata.obs[CELLTYPE_KEY])],
        ordered=False,
    )
    adata.uns[f"{CELLTYPE_KEY}_colors"] = [
        CELLTYPE_COLORS[x] for x in adata.obs[CELLTYPE_KEY].cat.categories
    ]
    color_dict = dict(zip(adata.obs[CELLTYPE_KEY].cat.categories, adata.uns[f"{CELLTYPE_KEY}_colors"]))

    print("subset shape after downsample:", adata.shape)
    print(adata.obs[CELLTYPE_KEY].value_counts())
    adata.write_h5ad(CPDB_OUT / "LT_major_preview_input.h5ad")

    ov.plot_set(font_path="Arial")
    cpdb_results, adata_cpdb = ov.single.run_cellphonedb_v5(
        adata,
        cpdb_file_path=str(CPDB_ZIP),
        celltype_key=CELLTYPE_KEY,
        min_cell_fraction=0.005,
        min_genes=200,
        min_cells=3,
        iterations=100,
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
    comm_adata.write_h5ad(CPDB_OUT / "LT_major_preview_comm_adata.h5ad")

    if "classification" in comm_adata.var.columns:
        pd.Series(
            sorted(comm_adata.var["classification"].fillna("").astype(str).unique())
        ).to_csv(OUT_ROOT / "available_pathways.csv", index=False, header=["pathway"])

    try:
        fig, ax = ov.pl.ccc_heatmap(
            adata_plot,
            plot_type="dot",
            display_by="aggregation",
            cmap="YlGnBu",
            figsize=(7, 5),
            show=False,
        )
        savefig("00_overall_dot_heatmap")
    except Exception as exc:
        print(f"[skip] overall heatmap: {type(exc).__name__}: {exc}")

    try:
        fig, ax = ov.pl.ccc_network_plot(
            comm_adata,
            plot_type="individual_outgoing",
            palette=color_dict,
            figsize=(8, 8),
            show=False,
        )
        savefig("01_outgoing_network")
    except Exception as exc:
        print(f"[skip] outgoing network: {type(exc).__name__}: {exc}")

    try:
        fig, ax = ov.pl.ccc_stat_plot(
            adata_plot,
            plot_type="scatter",
            figsize=(5.5, 5.5),
            show=False,
        )
        savefig("02_incoming_outgoing_scatter")
    except Exception as exc:
        print(f"[skip] scatter: {type(exc).__name__}: {exc}")

    story_pathways = choose_story_pathways(comm_adata, n=8)
    pd.Series(story_pathways).to_csv(OUT_ROOT / "selected_story_pathways.csv", index=False, header=["pathway"])
    print("selected story pathways:", story_pathways)

    for pathway in story_pathways:
        plot_pathway_panel(adata_plot, comm_adata, pathway, color_dict)

    print(f"Done. Figures: {OUT_ROOT}")
    print(f"Raw CPDB results: {CPDB_OUT}")


if __name__ == "__main__":
    main()
