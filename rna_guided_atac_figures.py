"""Generate RNA-guided scATAC figures only.

RNA results are used as prior gene sets. Every figure visualizes ATAC gene
activity, ATAC embedding, ATAC sample effects, or RNA-ATAC concordance.
No RNA-only figures are produced.
"""

from __future__ import annotations

import re
import shutil
import sys
from pathlib import Path

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from scipy import sparse


ROOT = Path("/mnt/d/lxk/project/jiangshucai20260506")
H5AD = ROOT / "h5ad"
DEG = ROOT / "DEG/deseq2_results"
FIG_DEG = ROOT / "fig/DEG/DESeq2"
TIME = ROOT / "fig/time"
CPDB = ROOT / "cpdb_results"
OLD = ROOT / "atac/final_pdf"
OUT = ROOT / "atac/rna_guided_atac_pdf_final"
TAB = ROOT / "atac/rna_guided_atac_tables_final"
OUT.mkdir(parents=True, exist_ok=True)
TAB.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(ROOT / "helper"))
from plot_helper import setMyStyle  # noqa: E402


ATAC_FILE = H5AD / "atac_gene_matrix_annotated.h5ad"
RNA_FILES = {
    "Myeloid": H5AD / "myeloid.h5ad",
    "T_NK": H5AD / "T_NK.h5ad",
    "SFT": H5AD / "sft.h5ad",
}
RNA_TO_ATAC = {
    "Myeloid": "Myeloid",
    "myeloid": "Myeloid",
    "T_NK": "T/NK",
    "SFT": "SFT_Tumor",
    "sft": "SFT_Tumor",
}
SAMPLE_ORDER = ["LT-d", "LT-e", "LT-f", "LT-g", "LZ-d", "LZ-f", "LZ-g"]
GROUP_COLORS = {"LT": "#3568A8", "LZ": "#B63C45"}
SAMPLE_COLORS = dict(zip(SAMPLE_ORDER, sns.color_palette("Set2", 7)))
CMAP = LinearSegmentedColormap.from_list(
    "cellnature", ["#17365D", "#E8EEF3", "#F4D6C8", "#A6222A"]
)

CORE_PDFS = [
    "01_QC_fragment_TSSE.pdf",
    "02_QC_sample_summary.pdf",
    "03_ATAC_UMAP_landscape.pdf",
    "04_ATAC_preliminary_composition.pdf",
    "09_ATAC_broad_marker_activity.pdf",
    "10_LT_vs_LZ_differential_gene_activity.pdf",
    "11_celltype_LT_LZ_activity_heatmap.pdf",
    "12_selected_accessible_tile_annotation.pdf",
    "13_candidate_TF_activity_proxy.pdf",
    "15_scATAC_summary_dashboard.pdf",
]


def setup_style() -> None:
    setMyStyle()
    sns.set_theme(style="white", context="paper")
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 8,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "axes.linewidth": 0.8,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "legend.fontsize": 7,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.+-]+", "_", value).strip("_")


def save_pdf(fig: plt.Figure, name: str) -> None:
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def clean_axes(ax: plt.Axes) -> None:
    ax.grid(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def copy_core() -> None:
    for index, filename in enumerate(CORE_PDFS, start=1):
        source = OLD / filename
        target = OUT / f"{index:02d}_core_{filename[3:]}"
        shutil.copy2(source, target)


def read_atac() -> ad.AnnData:
    atac = sc.read_h5ad(ATAC_FILE)
    atac.obs["sample_clean"] = (
        atac.obs["sample"].astype(str).str.replace(r"^at\s+", "", regex=True)
    )
    atac.obs["group"] = np.where(
        atac.obs["sample_clean"].str.startswith("LT"), "LT", "LZ"
    )
    sc.pp.normalize_total(atac, target_sum=10_000)
    sc.pp.log1p(atac)
    return atac


def mean_vector(matrix) -> np.ndarray:
    return np.asarray(matrix.mean(axis=0)).ravel()


def module_score(atac: ad.AnnData, genes: list[str]) -> tuple[np.ndarray, list[str]]:
    present = list(dict.fromkeys(g for g in genes if g in atac.var_names))
    if not present:
        return np.full(atac.n_obs, np.nan), []
    matrix = atac[:, present].X
    score = np.asarray(matrix.mean(axis=1)).ravel()
    return score, present


def aggregate_score(
    atac: ad.AnnData,
    score: np.ndarray,
    broad_type: str,
) -> pd.DataFrame:
    obs = atac.obs[["sample_clean", "group", "celltype"]].copy()
    obs["score"] = score
    obs = obs[obs["celltype"].astype(str) == broad_type]
    return (
        obs.groupby(["sample_clean", "group"], observed=True)["score"]
        .agg(["mean", "median", "std", "count"])
        .reset_index()
    )


def program_figure(
    atac: ad.AnnData,
    genes: list[str],
    broad_type: str,
    title: str,
    name: str,
) -> pd.DataFrame | None:
    score, present = module_score(atac, genes)
    if len(present) < 2:
        return None
    mask = atac.obs["celltype"].astype(str).to_numpy() == broad_type
    umap = np.asarray(atac.obsm["X_umap"])
    values = score[mask]
    lo, hi = np.nanpercentile(values, [2, 98])
    summary = aggregate_score(atac, score, broad_type)

    fig, axes = plt.subplots(
        1, 3, figsize=(10.2, 3.15),
        gridspec_kw={"width_ratios": [1.15, 1.15, 0.85]},
    )
    axes[0].scatter(
        umap[:, 0], umap[:, 1], s=0.25, color="#E5E5E5",
        rasterized=True, linewidths=0,
    )
    points = axes[0].scatter(
        umap[mask, 0], umap[mask, 1], c=values, s=0.6,
        cmap=CMAP, vmin=lo, vmax=hi, rasterized=True, linewidths=0,
    )
    fig.colorbar(points, ax=axes[0], fraction=0.045, pad=0.02, label="ATAC activity")
    axes[0].set_title(f"{broad_type} cells")
    axes[0].set_xticks([])
    axes[0].set_yticks([])
    axes[0].set_xlabel("UMAP1")
    axes[0].set_ylabel("UMAP2")
    clean_axes(axes[0])

    sample_summary = summary.set_index("sample_clean").reindex(SAMPLE_ORDER)
    axes[1].bar(
        np.arange(len(SAMPLE_ORDER)),
        sample_summary["mean"],
        color=[SAMPLE_COLORS[x] for x in SAMPLE_ORDER],
        width=0.72,
    )
    axes[1].errorbar(
        np.arange(len(SAMPLE_ORDER)),
        sample_summary["mean"],
        yerr=sample_summary["std"] / np.sqrt(sample_summary["count"]),
        fmt="none", ecolor="#333333", elinewidth=0.7, capsize=2,
    )
    axes[1].set_xticks(np.arange(len(SAMPLE_ORDER)))
    axes[1].set_xticklabels(SAMPLE_ORDER)
    axes[1].set_xlabel("")
    axes[1].set_ylabel("Mean ATAC activity")
    axes[1].tick_params(axis="x", rotation=45)
    axes[1].set_title("Sample-level activity")
    clean_axes(axes[1])

    grouped = summary.groupby("group", observed=True)["mean"].agg(["mean", "std"])
    axes[2].bar(
        ["LT", "LZ"], grouped.reindex(["LT", "LZ"])["mean"],
        yerr=grouped.reindex(["LT", "LZ"])["std"],
        color=[GROUP_COLORS["LT"], GROUP_COLORS["LZ"]],
        width=0.62, capsize=3,
    )
    axes[2].set_ylabel("Mean sample activity")
    axes[2].set_title("Group summary")
    clean_axes(axes[2])

    fig.suptitle(
        f"{title}\nRNA-guided program in scATAC ({len(present)} genes)",
        x=0.02, ha="left", fontweight="bold",
    )
    fig.tight_layout()
    save_pdf(fig, name)
    summary["program"] = title
    summary["genes"] = ";".join(present)
    return summary


def cluster_to_celltype(rna: ad.AnnData) -> dict[str, str]:
    table = pd.crosstab(
        rna.obs["clusters"].astype(str), rna.obs["celltype"].astype(str)
    )
    return table.idxmax(axis=1).to_dict()


def read_rna_marker_sets(n_genes: int = 12) -> dict[str, dict[str, list[str]]]:
    all_sets: dict[str, dict[str, list[str]]] = {}
    for lineage, path in RNA_FILES.items():
        rna = sc.read_h5ad(path, backed="r")
        mapping = cluster_to_celltype(rna)
        names = rna.uns["rank_genes_groups"]["names"]
        sets = {}
        for cluster in names.dtype.names:
            genes = []
            for gene in names[cluster]:
                gene = str(gene)
                if gene.startswith(("MT-", "RPS", "RPL")):
                    continue
                if gene not in genes:
                    genes.append(gene)
                if len(genes) >= n_genes:
                    break
            sets[mapping.get(str(cluster), str(cluster))] = genes
        all_sets[lineage] = sets
        rna.file.close()
    return all_sets


def subtype_program_figures(
    atac: ad.AnnData,
    marker_sets: dict[str, dict[str, list[str]]],
    start_index: int,
) -> int:
    index = start_index
    all_summaries = []
    for lineage, sets in marker_sets.items():
        broad = RNA_TO_ATAC[lineage]
        for subtype, genes in sets.items():
            summary = program_figure(
                atac, genes, broad,
                f"{subtype} marker program",
                f"{index:02d}_ATAC_RNA_marker_{safe_name(lineage)}_{safe_name(subtype)}",
            )
            if summary is not None:
                summary["lineage"] = lineage
                summary["subtype"] = subtype
                all_summaries.append(summary)
                index += 1

    combined = pd.concat(all_summaries, ignore_index=True)
    combined.to_csv(TAB / "01_RNA_marker_program_ATAC_scores.csv", index=False)

    for lineage in marker_sets:
        data = combined[combined["lineage"] == lineage]
        sample_matrix = data.pivot(index="subtype", columns="sample_clean", values="mean")
        sample_matrix = sample_matrix.reindex(columns=SAMPLE_ORDER)
        z = sample_matrix.sub(sample_matrix.mean(axis=1), axis=0).div(
            sample_matrix.std(axis=1).replace(0, 1), axis=0
        )
        fig, ax = plt.subplots(figsize=(6.1, max(2.8, 0.38 * len(z))))
        sns.heatmap(
            z, cmap=CMAP, center=0, vmin=-2, vmax=2, linewidths=0.3,
            linecolor="white", cbar_kws={"label": "Row z-score", "shrink": 0.72},
            ax=ax,
        )
        ax.set_xlabel("ATAC sample")
        ax.set_ylabel("RNA-defined subtype program")
        ax.set_title(f"{lineage} RNA programs across scATAC samples", fontweight="bold")
        ax.tick_params(axis="x", rotation=45)
        ax.tick_params(axis="y", rotation=0)
        fig.tight_layout()
        save_pdf(fig, f"{index:02d}_ATAC_{safe_name(lineage)}_program_sample_heatmap")
        index += 1

        group_matrix = (
            data.groupby(["subtype", "group"], observed=True)["mean"]
            .mean().unstack().reindex(columns=["LT", "LZ"])
        )
        group_delta = pd.DataFrame(
            {"LT_minus_LZ": group_matrix["LT"] - group_matrix["LZ"]}
        ).sort_values("LT_minus_LZ")
        fig, ax = plt.subplots(figsize=(4.7, max(2.8, 0.4 * len(group_delta))))
        colors = np.where(
            group_delta["LT_minus_LZ"] > 0,
            GROUP_COLORS["LT"], GROUP_COLORS["LZ"],
        )
        ax.barh(group_delta.index, group_delta["LT_minus_LZ"], color=colors)
        ax.axvline(0, color="#333333", lw=0.7)
        ax.set_xlabel("ATAC program activity (LT - LZ)")
        ax.set_title(f"{lineage} RNA programs in scATAC", fontweight="bold")
        clean_axes(ax)
        fig.tight_layout()
        save_pdf(fig, f"{index:02d}_ATAC_{safe_name(lineage)}_program_LT_LZ")
        index += 1
    return index


def read_deg_jobs() -> list[tuple[str, str, Path]]:
    jobs = []
    for path in sorted(DEG.glob("*__DESeq2.csv")):
        parts = path.stem.replace("__DESeq2", "").split("__")
        jobs.append((parts[0], parts[1].replace("_", " "), path))
    return jobs


def sample_gene_activity(
    atac: ad.AnnData,
    genes: list[str],
    broad_type: str,
) -> pd.DataFrame:
    present = [g for g in genes if g in atac.var_names]
    rows = []
    broad_mask = atac.obs["celltype"].astype(str).to_numpy() == broad_type
    for sample in SAMPLE_ORDER:
        mask = broad_mask & (atac.obs["sample_clean"].to_numpy() == sample)
        if mask.sum() == 0:
            continue
        values = mean_vector(atac[mask, present].X)
        rows.append(pd.Series(values, index=present, name=sample))
    return pd.DataFrame(rows)


def deg_validation_figures(atac: ad.AnnData, start_index: int) -> int:
    index = start_index
    records = []
    for lineage, subtype, path in read_deg_jobs():
        broad = RNA_TO_ATAC[lineage]
        deg = pd.read_csv(path).dropna(subset=["log2FoldChange"])
        deg = deg[deg["gene"].isin(atac.var_names)]
        significant = deg[deg["padj"].fillna(1) < 0.05]
        source = significant if len(significant) >= 8 else deg.nsmallest(100, "pvalue")
        lz = source.nlargest(8, "log2FoldChange")
        lt = source.nsmallest(8, "log2FoldChange")
        selected = pd.concat([lt, lz]).drop_duplicates("gene")
        activity = sample_gene_activity(atac, selected["gene"].tolist(), broad)
        if activity.empty:
            continue
        z = activity.sub(activity.mean(axis=0), axis=1).div(
            activity.std(axis=0).replace(0, 1), axis=1
        )
        atac_effect = (
            activity.loc[[x for x in activity.index if x.startswith("LZ")]].mean()
            - activity.loc[[x for x in activity.index if x.startswith("LT")]].mean()
        )
        compare = selected.set_index("gene")[["log2FoldChange", "padj"]].join(
            atac_effect.rename("ATAC_LZ_minus_LT")
        ).dropna()
        concordance = np.corrcoef(
            compare["log2FoldChange"], compare["ATAC_LZ_minus_LT"]
        )[0, 1] if len(compare) > 2 else np.nan

        fig, axes = plt.subplots(1, 3, figsize=(11.4, 3.6),
                                 gridspec_kw={"width_ratios": [1.5, 0.9, 1]})
        sns.heatmap(
            z.T, cmap=CMAP, center=0, vmin=-2, vmax=2, linewidths=0.25,
            linecolor="white", cbar_kws={"label": "Gene-wise z-score", "shrink": 0.65},
            ax=axes[0],
        )
        axes[0].set_xlabel("ATAC sample")
        axes[0].set_ylabel("RNA DEG")
        axes[0].tick_params(axis="x", rotation=45)
        axes[0].tick_params(axis="y", rotation=0)
        axes[0].set_title("ATAC activity of RNA DEGs")

        axes[1].scatter(
            compare["log2FoldChange"], compare["ATAC_LZ_minus_LT"],
            c=np.where(compare["log2FoldChange"] > 0, GROUP_COLORS["LZ"], GROUP_COLORS["LT"]),
            s=25, edgecolor="white", linewidth=0.4,
        )
        for gene, row in compare.iterrows():
            axes[1].text(row["log2FoldChange"], row["ATAC_LZ_minus_LT"], gene, fontsize=5.5)
        axes[1].axhline(0, color="#777777", lw=0.5)
        axes[1].axvline(0, color="#777777", lw=0.5)
        axes[1].set_xlabel("RNA log2FC (LZ/LT)")
        axes[1].set_ylabel("ATAC activity (LZ - LT)")
        axes[1].set_title(f"RNA-ATAC concordance\nr={concordance:.2f}")
        clean_axes(axes[1])

        lz_genes = lz["gene"].tolist()
        lt_genes = lt["gene"].tolist()
        lz_score, _ = module_score(atac, lz_genes)
        lt_score, _ = module_score(atac, lt_genes)
        delta = lz_score - lt_score
        summary = aggregate_score(atac, delta, broad)
        sns.barplot(
            data=summary, x="sample_clean", y="mean", hue="group",
            order=SAMPLE_ORDER, palette=GROUP_COLORS, dodge=False, ax=axes[2],
        )
        axes[2].axhline(0, color="#555555", lw=0.6)
        axes[2].set_xlabel("")
        axes[2].set_ylabel("LZ-up minus LT-up activity")
        axes[2].tick_params(axis="x", rotation=45)
        axes[2].legend(frameon=False, title="")
        axes[2].set_title("ATAC directional score")
        clean_axes(axes[2])

        fig.suptitle(
            f"{subtype}: RNA DESeq2-guided scATAC validation",
            x=0.02, ha="left", fontweight="bold",
        )
        fig.tight_layout()
        save_pdf(
            fig,
            f"{index:02d}_ATAC_DEG_validation_{safe_name(lineage)}_{safe_name(subtype)}",
        )
        compare["lineage"] = lineage
        compare["subtype"] = subtype
        compare["concordance"] = concordance
        records.append(compare.reset_index())
        index += 1
    pd.concat(records, ignore_index=True).to_csv(
        TAB / "02_RNA_DEG_ATAC_concordance.csv", index=False
    )
    return index


def driver_files() -> list[Path]:
    return sorted(TIME.glob("**/*_filtered_drivers.csv"))


def pseudotime_figures(atac: ad.AnnData, start_index: int) -> int:
    index = start_index
    rows = []
    for path in driver_files():
        state = path.stem.replace("_filtered_drivers", "")
        lineage = "Myeloid" if "myeloid" in str(path).lower() else "SFT"
        broad = RNA_TO_ATAC[lineage]
        data = pd.read_csv(path, index_col=0)
        genes = data.index.astype(str).tolist()[:20]
        summary = program_figure(
            atac, genes, broad,
            f"{state} CellRank driver program",
            f"{index:02d}_ATAC_CellRank_driver_{safe_name(lineage)}_{safe_name(state)}",
        )
        if summary is not None:
            summary["lineage"] = lineage
            summary["state"] = state
            rows.append(summary)
            index += 1
    if rows:
        pd.concat(rows, ignore_index=True).to_csv(
            TAB / "03_CellRank_driver_ATAC_scores.csv", index=False
        )
    return index


def select_cpdb_interactions(
    path: Path,
    immune_keywords: list[str],
    n: int = 5,
) -> pd.DataFrame:
    data = pd.read_csv(path, sep="\t", low_memory=False)
    pair_cols = []
    for col in data.columns:
        if "|" not in col:
            continue
        sender, receiver = col.split("|", 1)
        sender_sft = "SFT" in sender
        receiver_sft = "SFT" in receiver
        sender_immune = any(x in sender for x in immune_keywords)
        receiver_immune = any(x in receiver for x in immune_keywords)
        if (sender_sft and receiver_immune) or (sender_immune and receiver_sft):
            pair_cols.append(col)
    if not pair_cols:
        return pd.DataFrame()
    numeric = data[pair_cols].apply(pd.to_numeric, errors="coerce")
    data["cross_score"] = numeric.max(axis=1, skipna=True).fillna(0)
    data["best_pair_col"] = numeric.idxmax(axis=1)
    data = data[
        data["gene_a"].notna() & data["gene_b"].notna()
        & (data["gene_a"].astype(str) != "")
        & (data["gene_b"].astype(str) != "")
    ]
    data = data.sort_values(["cross_score", "rank"], ascending=[False, True])
    data = data.drop_duplicates(["gene_a", "gene_b"])
    return data.head(n)


def communication_figure(
    atac: ad.AnnData,
    sender_gene: str,
    receiver_gene: str,
    sender_broad: str,
    receiver_broad: str,
    classification: str,
    pair: str,
    name: str,
) -> bool:
    if sender_gene not in atac.var_names or receiver_gene not in atac.var_names:
        return False
    umap = np.asarray(atac.obsm["X_umap"])
    sender_mask = atac.obs["celltype"].astype(str).to_numpy() == sender_broad
    receiver_mask = atac.obs["celltype"].astype(str).to_numpy() == receiver_broad
    sender_values = np.asarray(atac[:, sender_gene].X.toarray()).ravel()
    receiver_values = np.asarray(atac[:, receiver_gene].X.toarray()).ravel()

    fig, axes = plt.subplots(1, 3, figsize=(10.4, 3.15))
    for ax, mask, values, gene, title in [
        (
            axes[0], sender_mask, sender_values, sender_gene,
            f"{sender_broad} sender activity",
        ),
        (
            axes[1], receiver_mask, receiver_values, receiver_gene,
            f"{receiver_broad} receiver activity",
        ),
    ]:
        ax.scatter(umap[:, 0], umap[:, 1], s=0.25, color="#E5E5E5",
                   rasterized=True, linewidths=0)
        lo, hi = np.percentile(values[mask], [2, 98])
        pts = ax.scatter(
            umap[mask, 0], umap[mask, 1], c=values[mask], s=0.6,
            cmap=CMAP, vmin=lo, vmax=hi, rasterized=True, linewidths=0,
        )
        fig.colorbar(pts, ax=ax, fraction=0.045, pad=0.02, label=gene)
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel("UMAP1")
        ax.set_ylabel("UMAP2")
        clean_axes(ax)

    rows = []
    for sample in SAMPLE_ORDER:
        sm = atac.obs["sample_clean"].to_numpy() == sample
        rows.append({
            "sample": sample,
            "group": "LT" if sample.startswith("LT") else "LZ",
            f"{sender_gene}_{sender_broad}": np.mean(
                sender_values[sm & sender_mask]
            ),
            f"{receiver_gene}_{receiver_broad}": np.mean(
                receiver_values[sm & receiver_mask]
            ),
        })
    summary = pd.DataFrame(rows)
    plot = summary.set_index("sample")[
        [
            f"{sender_gene}_{sender_broad}",
            f"{receiver_gene}_{receiver_broad}",
        ]
    ]
    plot = plot.sub(plot.mean()).div(plot.std().replace(0, 1))
    sns.heatmap(
        plot.T, cmap=CMAP, center=0, vmin=-2, vmax=2, linewidths=0.3,
        linecolor="white", cbar_kws={"label": "Row z-score", "shrink": 0.65},
        ax=axes[2],
    )
    axes[2].set_xlabel("ATAC sample")
    axes[2].set_ylabel("")
    axes[2].tick_params(axis="x", rotation=45)
    axes[2].tick_params(axis="y", rotation=0)
    axes[2].set_title("Sender-receiver accessibility")
    fig.suptitle(
        f"{pair} | {classification}\n"
        "CellPhoneDB-guided ligand-receptor activity in scATAC",
        x=0.02, ha="left", fontweight="bold",
    )
    fig.tight_layout()
    save_pdf(fig, name)
    return True


def communication_figures(atac: ad.AnnData, start_index: int) -> int:
    index = start_index
    configurations = [
        (
            CPDB / "statistical_analysis_significant_means_06_11_2026_010945.txt",
            ["TAM", "myeloid", "monocyte", "APC"], "Myeloid",
        ),
        (
            CPDB / "statistical_analysis_significant_means_06_11_2026_013039.txt",
            ["NK", "CD8", "CD4", "Treg", "T/NK"], "T/NK",
        ),
    ]
    selected_rows = []
    for path, keywords, broad in configurations:
        selected = select_cpdb_interactions(path, keywords, n=12)
        made = 0
        for _, row in selected.iterrows():
            sender_label, receiver_label = str(row["best_pair_col"]).split("|", 1)
            if "SFT" in sender_label:
                sender_broad, receiver_broad = "SFT_Tumor", broad
            else:
                sender_broad, receiver_broad = broad, "SFT_Tumor"
            success = communication_figure(
                atac,
                str(row["gene_a"]), str(row["gene_b"]),
                sender_broad, receiver_broad,
                str(row.get("classification", "")),
                str(row["interacting_pair"]),
                f"{index:02d}_ATAC_LR_{safe_name(broad)}_{safe_name(str(row['interacting_pair']))}",
            )
            if success:
                selected_rows.append(row)
                index += 1
                made += 1
            if made >= 5:
                break
    if selected_rows:
        pd.DataFrame(selected_rows).to_csv(
            TAB / "04_CellPhoneDB_selected_ATAC_interactions.csv", index=False
        )
    return index


def write_manifest() -> None:
    files = sorted(OUT.glob("*.pdf"))
    pd.DataFrame({
        "figure": [p.name for p in files],
        "size_kb": [round(p.stat().st_size / 1024, 1) for p in files],
    }).to_csv(TAB / "00_figure_manifest.csv", index=False)
    lines = [
        "# RNA-guided scATAC figure index",
        "",
        f"Total PDF figures: {len(files)}",
        "",
        "All figures visualize scATAC results. RNA analyses are used only to define",
        "marker, DEG, pseudotime-driver, and ligand-receptor gene sets.",
        "",
    ]
    lines.extend(f"- `{p.name}`" for p in files)
    (OUT / "README_figure_index.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    setup_style()
    print("Copying core ATAC figures")
    copy_core()
    print("Reading compact ATAC gene activity matrix")
    atac = read_atac()
    print("RNA subtype marker programs -> ATAC")
    marker_sets = read_rna_marker_sets()
    index = subtype_program_figures(atac, marker_sets, 11)
    print("RNA DESeq2 results -> ATAC")
    index = deg_validation_figures(atac, index)
    print("CellRank drivers -> ATAC")
    index = pseudotime_figures(atac, index)
    print("CellPhoneDB interactions -> ATAC")
    communication_figures(atac, index)
    write_manifest()
    print(f"Finished: {OUT}")


if __name__ == "__main__":
    main()
