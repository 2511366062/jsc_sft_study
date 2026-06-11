def _get_obs_order(adata, key, values):
    if hasattr(adata.obs[key], "cat"):
        return [x for x in adata.obs[key].cat.categories.astype(str) if x in values]
    return list(values)


def _get_category_colors(adata, key, categories, palette=None):
    default_palette = [
        "#4C5F8F", "#B7794B", "#6FA083", "#B95F62", "#A66C9A", "#5BA8A0",
        "#8DA6BF", "#D8B56D", "#9BAE6A", "#7E6BB2", "#8B6A5B", "#C58A54",
        "#6D8BA6", "#C7B46A", "#7FAD8B", "#B7A6C9", "#A77C73", "#9A9A9A",
    ]

    if palette is not None:
        if isinstance(palette, dict):
            return {
                cat: palette.get(cat, default_palette[i % len(default_palette)])
                for i, cat in enumerate(categories)
            }
        return {cat: palette[i % len(palette)] for i, cat in enumerate(categories)}

    uns_key = f"{key}_colors"
    if uns_key in adata.uns and hasattr(adata.obs[key], "cat"):
        obs_categories = [str(x) for x in adata.obs[key].cat.categories]
        uns_colors = list(adata.uns[uns_key])
        color_map = {
            cat: uns_colors[i]
            for i, cat in enumerate(obs_categories)
            if i < len(uns_colors)
        }
        return {
            cat: color_map.get(cat, default_palette[i % len(default_palette)])
            for i, cat in enumerate(categories)
        }

    return {cat: default_palette[i % len(default_palette)] for i, cat in enumerate(categories)}


def plot_umap_panel(
    adata,
    color=("celltype", "sample", "group"),
    save=None,
    dpi=600,
    show=None,
    **kwargs,
):
    """
    Plot a Scanpy UMAP panel, for example celltype / sample / group.

    Optional parameters:
    - color: obs keys or genes to plot on UMAP.
    - save: output path. If None, only displays the figure.
    - dpi: save dpi when save is provided.
    - show: passed to scanpy. If save is provided, defaults to False.
    - **kwargs: any extra sc.pl.umap parameters, such as ncols, wspace,
      legend_loc, frameon, size, title, palette, vmax, vmin.
    """
    import matplotlib.pyplot as plt
    import scanpy as sc

    if show is None:
        show = False if save else None

    axes = sc.pl.umap(
        adata,
        color=list(color) if isinstance(color, (tuple, list)) else color,
        show=show,
        **kwargs,
    )

    if save:
        if isinstance(axes, list):
            fig = axes[0].figure
        elif axes is not None:
            fig = axes.figure
        else:
            fig = plt.gcf()
        fig.savefig(save, dpi=dpi, bbox_inches="tight")

    return axes


def cell_count_table(adata, groupby="sample", celltype_key="celltype", order=None, celltype_order=None):
    import pandas as pd

    if groupby not in adata.obs:
        raise KeyError(f"{groupby!r} was not found in adata.obs")
    if celltype_key not in adata.obs:
        raise KeyError(f"{celltype_key!r} was not found in adata.obs")

    obs = adata.obs[[groupby, celltype_key]].copy().dropna()
    obs[groupby] = obs[groupby].astype(str)
    obs[celltype_key] = obs[celltype_key].astype(str)

    counts = pd.crosstab(obs[groupby], obs[celltype_key])

    if order is None:
        order = _get_obs_order(adata, groupby, counts.index)
    counts = counts.reindex(order)

    if celltype_order is None:
        celltype_order = _get_obs_order(adata, celltype_key, counts.columns)
        if not celltype_order:
            celltype_order = list(counts.sum(axis=0).sort_values(ascending=False).index)
    counts = counts.reindex(columns=celltype_order, fill_value=0)
    return counts


def plot_cell_counts(
    adata,
    groupby="sample",
    celltype_key="celltype",
    normalize=True,
    order=None,
    celltype_order=None,
    palette=None,
    figsize=None,
    title=None,
    ylabel=None,
    legend_title=None,
    legend_bbox=(1.02, 1),
    save=None,
    dpi=600,
    ax=None,
):
    """
    Plot publication-style cell number/composition bars from an AnnData object.

    Colors are read from ``adata.uns[f"{celltype_key}_colors"]`` by default,
    so bar colors stay consistent with Scanpy UMAP colors.
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter

    counts = cell_count_table(
        adata,
        groupby=groupby,
        celltype_key=celltype_key,
        order=order,
        celltype_order=celltype_order,
    )

    if normalize:
        table = counts.div(counts.sum(axis=1).replace(0, np.nan), axis=0) * 100
        default_ylabel = "Cell proportion (%)"
    else:
        table = counts
        default_ylabel = "Number of cells"

    colors = _get_category_colors(adata, celltype_key, list(table.columns), palette=palette)

    if ax is None:
        if figsize is None:
            figsize = (max(4.2, 0.58 * len(table.index) + 2.4), 4.2)
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    bottom = np.zeros(len(table.index), dtype=float)
    x = np.arange(len(table.index))
    for ct in table.columns:
        values = table[ct].to_numpy(dtype=float)
        ax.bar(
            x,
            values,
            bottom=bottom,
            color=colors[ct],
            edgecolor="white",
            linewidth=0.35,
            width=0.78,
            label=ct,
        )
        bottom += values

    ax.set_xticks(x)
    ax.set_xticklabels(table.index, rotation=45 if len(table.index) > 3 else 0, ha="right")
    ax.set_xlabel("")
    ax.set_ylabel(ylabel or default_ylabel)
    if title:
        ax.set_title(title, pad=8)

    ax.grid(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.8)
    ax.spines["bottom"].set_linewidth(0.8)
    ax.tick_params(axis="both", which="major", length=3.5, width=0.8, labelsize=10)

    if normalize:
        ax.set_ylim(0, 100)
        ax.yaxis.set_major_formatter(PercentFormatter(xmax=100))
    else:
        ax.set_ylim(0, bottom.max() * 1.08 if len(bottom) else 1)

    ax.legend(
        title=legend_title or celltype_key,
        bbox_to_anchor=legend_bbox,
        loc="upper left",
        frameon=False,
        borderaxespad=0,
        handlelength=1.1,
        handleheight=0.9,
        labelspacing=0.35,
        fontsize=9,
        title_fontsize=10,
    )

    fig.tight_layout()
    if save:
        fig.savefig(save, dpi=dpi, bbox_inches="tight")

    return fig, ax, table


def plot_cell_pies(
    adata,
    groupby="sample",
    celltype_key="celltype",
    order=None,
    celltype_order=None,
    palette=None,
    min_pct=0,
    donut=True,
    shadow=False,
    show_pct=False,
    pct_min=5,
    ncols=3,
    figsize=None,
    title=None,
    legend=True,
    legend_title=None,
    save=None,
    dpi=600,
):
    """
    Plot one pie/donut chart per sample or group.

    Colors are read from ``adata.uns[f"{celltype_key}_colors"]`` by default.
    Set ``donut=False`` for solid pie charts. ``shadow=True`` gives a subtle
    pseudo-3D look, but flat pies are recommended for publication figures.
    """
    import math
    import numpy as np
    import matplotlib.pyplot as plt

    counts = cell_count_table(
        adata,
        groupby=groupby,
        celltype_key=celltype_key,
        order=order,
        celltype_order=celltype_order,
    )
    props = counts.div(counts.sum(axis=1).replace(0, np.nan), axis=0) * 100
    colors = _get_category_colors(adata, celltype_key, list(props.columns), palette=palette)

    groups = list(props.index)
    ncols = min(ncols, max(1, len(groups)))
    nrows = math.ceil(len(groups) / ncols)
    if figsize is None:
        figsize = (3.0 * ncols + (1.4 if legend else 0), 2.8 * nrows)
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)
    flat_axes = axes.ravel()

    for ax, group in zip(flat_axes, groups):
        values = props.loc[group]
        shown = values[values >= min_pct]
        shown = shown[shown > 0]
        wedgeprops = {"linewidth": 0.8, "edgecolor": "white"}
        if donut:
            wedgeprops["width"] = 0.42

        def autopct(pct):
            return f"{pct:.0f}%" if show_pct and pct >= pct_min else ""

        ax.pie(
            shown.values,
            colors=[colors[x] for x in shown.index],
            startangle=90,
            counterclock=False,
            wedgeprops=wedgeprops,
            shadow=shadow,
            autopct=autopct if show_pct else None,
            pctdistance=0.72 if donut else 0.68,
            textprops={"fontsize": 8, "color": "white", "weight": "bold"},
        )
        ax.set_title(f"{group}\nn={int(counts.loc[group].sum())}", fontsize=10, pad=4)
        ax.set_aspect("equal")

    for ax in flat_axes[len(groups):]:
        ax.axis("off")

    if title:
        fig.suptitle(title, y=1.02, fontsize=12)

    if legend:
        handles = [
            plt.Line2D(
                [0],
                [0],
                marker="o",
                color="none",
                markerfacecolor=colors[ct],
                markeredgecolor="none",
                markersize=7,
                label=ct,
            )
            for ct in props.columns
        ]
        fig.legend(
            handles=handles,
            title=legend_title or celltype_key,
            loc="center left",
            bbox_to_anchor=(1.0, 0.5),
            frameon=False,
            fontsize=9,
            title_fontsize=10,
        )

    fig.tight_layout()
    if save:
        fig.savefig(save, dpi=dpi, bbox_inches="tight")

    return fig, axes, props


def _marker_table(
    adata,
    marker_genes_dict,
    groupby="celltype",
    layer=None,
    use_raw=False,
    group_order=None,
    scale="zscore",
):
    import numpy as np
    import pandas as pd
    from scipy import sparse

    if groupby not in adata.obs:
        raise KeyError(f"{groupby!r} was not found in adata.obs")

    marker_items = []
    for panel, panel_genes in marker_genes_dict.items():
        for gene in panel_genes:
            marker_items.append({"gene": gene, "marker_group": panel})

    if use_raw:
        if adata.raw is None:
            raise ValueError("adata.raw is None, but use_raw=True was requested")
        var_names = adata.raw.var_names
    else:
        var_names = adata.var_names

    present_items = [item for item in marker_items if item["gene"] in var_names]
    unique_genes = []
    for item in present_items:
        if item["gene"] not in unique_genes:
            unique_genes.append(item["gene"])

    if use_raw:
        X = adata.raw[:, unique_genes].X
    elif layer is None:
        X = adata[:, unique_genes].X
    else:
        X = adata[:, unique_genes].layers[layer]

    missing = [item["gene"] for item in marker_items if item["gene"] not in var_names]
    if missing:
        print("Missing genes:", ", ".join(dict.fromkeys(missing)))
    if not present_items:
        raise ValueError("None of the requested marker genes were found in adata")

    obs_groups = adata.obs[groupby].astype(str)
    if group_order is None:
        group_order = _get_obs_order(adata, groupby, obs_groups.unique())
    group_order = [g for g in group_order if g in set(obs_groups)]

    rows = []
    for group in group_order:
        mask = (obs_groups == group).to_numpy()
        Xg = X[mask]
        if sparse.issparse(Xg):
            avg = np.asarray(Xg.mean(axis=0)).ravel()
            pct = np.asarray((Xg > 0).mean(axis=0)).ravel() * 100
        else:
            avg = np.asarray(Xg).mean(axis=0)
            pct = (np.asarray(Xg) > 0).mean(axis=0) * 100
        stats = {
            gene: (float(avg_expr), float(pct_expr))
            for gene, avg_expr, pct_expr in zip(unique_genes, avg, pct)
        }
        for pos, item in enumerate(present_items):
            gene = item["gene"]
            avg_expr, pct_expr = stats[gene]
            rows.append(
                {
                    "group": group,
                    "gene": gene,
                    "gene_label": gene,
                    "marker_pos": pos,
                    "marker_group": item["marker_group"],
                    "avg_expr": avg_expr,
                    "pct_expr": pct_expr,
                }
            )

    table = pd.DataFrame(rows)
    if scale == "zscore":
        mean_by_gene = table.groupby("marker_pos")["avg_expr"].transform("mean")
        std_by_gene = table.groupby("marker_pos")["avg_expr"].transform("std").replace(0, np.nan)
        table["plot_expr"] = ((table["avg_expr"] - mean_by_gene) / std_by_gene).fillna(0).clip(-2, 2)
    elif scale == "minmax":
        min_by_gene = table.groupby("marker_pos")["avg_expr"].transform("min")
        max_by_gene = table.groupby("marker_pos")["avg_expr"].transform("max")
        denom = (max_by_gene - min_by_gene).replace(0, np.nan)
        table["plot_expr"] = ((table["avg_expr"] - min_by_gene) / denom).fillna(0)
    elif scale == "raw":
        table["plot_expr"] = table["avg_expr"]
    else:
        raise ValueError("scale must be one of: 'zscore', 'minmax', 'raw'")

    table["scaled_expr"] = table["plot_expr"]
    gene_labels = [item["gene"] for item in present_items]
    return table, gene_labels, group_order


def plot_marker_dotplot(
    adata,
    marker_genes_dict,          # dict: {"celltype": ["gene1", "gene2"]}; duplicate genes are allowed
    groupby="celltype",         # obs column used as rows, usually "celltype" or "clusters"
    layer=None,                 # expression layer to use; e.g. "normalize"; None uses adata.X
    use_raw=False,              # True uses adata.raw instead of adata.X/layers
    group_order=None,           # optional row order; e.g. ["SFT_Tumor", "Myeloid", ...]
    scale="zscore",             # "zscore", "minmax", or "raw" for dot color
    cmap="RdBu_r",              # matplotlib/seaborn colormap or custom LinearSegmentedColormap
    size_range=(12, 190),       # min/max dot area for 0-100% expressed cells
    figsize=None,               # optional figure size; auto-computed if None
    title=None,                 # optional plot title
    show_marker_group_bar=True, # draw colored bars above marker groups
    marker_group_colors=None,   # optional dict/list for marker group bars; defaults to adata.uns colors
    save=None,                  # optional output path, e.g. "./fig/marker_dotplot.pdf"
    dpi=600,                    # save resolution
):
    """
    Draw a polished marker bubble plot.

    Parameters
    ----------
    adata
        AnnData object.
    marker_genes_dict
        Marker dictionary. Keys are marker groups/cell types, values are gene
        lists. Duplicate genes are allowed, so the same gene can appear under
        different marker groups.
    groupby
        Column in ``adata.obs`` used as groups on the y-axis.
    layer
        Layer used for expression values. Use ``layer="normalize"`` if your
        log-normalized matrix is stored there. If None, use ``adata.X``.
    use_raw
        If True, use ``adata.raw``. This overrides ``layer``.
    group_order
        Optional order of groups on the y-axis.
    scale
        Dot color transformation:
        ``"zscore"`` = gene-wise centered z-score, can be negative;
        ``"minmax"`` = gene-wise 0-1 scaling, no negative values;
        ``"raw"`` = mean expression without scaling.
    cmap
        Colormap for dot color. Can be a string such as ``"mako"`` or a custom
        matplotlib colormap object.
    size_range
        Dot area range for percentage of cells expressing each gene.
    figsize
        Figure size. If None, estimated from number of genes and groups.
    title
        Optional title.
    show_marker_group_bar
        Whether to show colored bars above marker gene groups.
    marker_group_colors
        Optional colors for the marker group bars. If None, colors are read
        from ``adata.uns[f"{groupby}_colors"]`` when available.
    save
        Optional output path.
    dpi
        Save resolution.

    Returns
    -------
    fig, ax, table
        Matplotlib figure/axes and a long-format table containing avg_expr,
        pct_expr, and plot_expr.
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    table, genes, groups = _marker_table(
        adata,
        marker_genes_dict,
        groupby=groupby,
        layer=layer,
        use_raw=use_raw,
        group_order=group_order,
        scale=scale,
    )

    gene_pos = {i: i for i in range(len(genes))}
    group_pos = {g: i for i, g in enumerate(groups)}
    x = table["marker_pos"].map(gene_pos).to_numpy()
    y = table["group"].map(group_pos).to_numpy()
    pct = table["pct_expr"].to_numpy()
    sizes = size_range[0] + (pct / 100.0) * (size_range[1] - size_range[0])

    if figsize is None:
        figsize = (max(7, 0.34 * len(genes) + 2.2), max(3.2, 0.38 * len(groups) + 1.5))
    fig, ax = plt.subplots(figsize=figsize)

    sc = ax.scatter(
        x,
        y,
        c=table["plot_expr"],
        s=sizes,
        cmap=cmap,
        vmin=-2 if scale == "zscore" else (0 if scale == "minmax" else None),
        vmax=2 if scale == "zscore" else (1 if scale == "minmax" else None),
        edgecolor="#F7F7F7",
        linewidth=0.35,
    )

    ax.set_xticks(range(len(genes)))
    ax.set_xticklabels(genes, rotation=45, ha="right", fontsize=9)
    ax.set_xlim(-0.55, len(genes) - 0.45)
    ax.set_yticks(range(len(groups)))
    ax.set_yticklabels(groups, fontsize=10)
    ax.invert_yaxis()
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.grid(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.8)
    ax.spines["bottom"].set_linewidth(0.8)
    ax.tick_params(length=3.5, width=0.8)

    marker_groups = [k for k, v in marker_genes_dict.items() if any(g in genes for g in v)]
    bar_colors = _get_category_colors(
        adata,
        groupby,
        marker_groups,
        palette=marker_group_colors,
    )

    boundaries = []
    start = 0
    for marker_group, panel_genes in marker_genes_dict.items():
        present = [g for g in panel_genes if g in genes]
        if not present:
            continue
        end = start + len(present) - 1
        center = (start + end) / 2
        if show_marker_group_bar:
            ax.add_patch(
                Rectangle(
                    (start - 0.45, 1.015),
                    len(present) - 0.1,
                    0.045,
                    transform=ax.get_xaxis_transform(),
                    facecolor=bar_colors[marker_group],
                    edgecolor="none",
                    clip_on=False,
                )
            )
        ax.text(
            center,
            1.075 if show_marker_group_bar else 1.025,
            marker_group,
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="bottom",
            fontsize=9,
            color="#333333",
            clip_on=False,
        )
        boundaries.append(end + 0.5)
        start = end + 1
    for b in boundaries[:-1]:
        ax.axvline(b, color="#D0D0D0", linewidth=0.7, zorder=0)

    if title:
        ax.set_title(title, pad=30 if show_marker_group_bar else 18)

    cbar = fig.colorbar(sc, ax=ax, fraction=0.025, pad=0.018)
    cbar_label = {
        "zscore": "Scaled mean expression",
        "minmax": "Relative mean expression",
        "raw": "Mean expression",
    }[scale]
    cbar.set_label(cbar_label, fontsize=9)
    cbar.ax.tick_params(labelsize=8, length=2.5)

    legend_pcts = [25, 50, 75]
    handles = [
        ax.scatter([], [], s=size_range[0] + (p / 100) * (size_range[1] - size_range[0]),
                   color="#8A8A8A", edgecolor="#F7F7F7", linewidth=0.35)
        for p in legend_pcts
    ]
    ax.legend(
        handles,
        [f"{p}%" for p in legend_pcts],
        title="Expressed cells",
        loc="upper left",
        bbox_to_anchor=(1.05, 0.58),
        frameon=False,
        fontsize=8,
        title_fontsize=9,
        labelspacing=0.8,
    )

    fig.tight_layout()
    if save:
        fig.savefig(save, dpi=dpi, bbox_inches="tight")
    return fig, ax, table


def plot_marker_heatmap(
    adata,
    marker_genes_dict,          # dict: {"celltype": ["gene1", "gene2"]}; duplicate genes are allowed
    groupby="celltype",         # obs column used as rows, usually "celltype" or "clusters"
    layer=None,                 # expression layer to use; e.g. "normalize"; None uses adata.X
    use_raw=False,              # True uses adata.raw instead of adata.X/layers
    group_order=None,           # optional row order; e.g. ["SFT_Tumor", "Myeloid", ...]
    scale="zscore",             # "zscore", "minmax", or "raw" for heatmap color
    cmap="vlag",                # matplotlib/seaborn colormap or custom LinearSegmentedColormap
    figsize=None,               # optional figure size; auto-computed if None
    title=None,                 # optional plot title
    show_marker_group_bar=True, # draw colored bars above marker groups
    marker_group_colors=None,   # optional dict/list for marker group bars; defaults to adata.uns colors
    save=None,                  # optional output path, e.g. "./fig/marker_heatmap.pdf"
    dpi=600,                    # save resolution
):
    """
    Draw a gene-wise scaled marker heatmap from an AnnData object.

    Parameters are the same as ``plot_marker_dotplot`` except that color is
    displayed as a heatmap and no expression-percentage dot size is shown.

    Returns
    -------
    fig, ax, mat
        Matplotlib figure/axes and the plotted matrix.
    """
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    from matplotlib.patches import Rectangle

    table, genes, groups = _marker_table(
        adata,
        marker_genes_dict,
        groupby=groupby,
        layer=layer,
        use_raw=use_raw,
        group_order=group_order,
        scale=scale,
    )
    mat = table.pivot(index="group", columns="marker_pos", values="plot_expr")
    mat = mat.reindex(index=groups, columns=range(len(genes)))
    mat.columns = genes

    if figsize is None:
        figsize = (max(7, 0.32 * len(genes) + 2.0), max(3.2, 0.36 * len(groups) + 1.3))
    fig, ax = plt.subplots(figsize=figsize)

    sns.heatmap(
        mat,
        ax=ax,
        cmap=cmap,
        vmin=-2 if scale == "zscore" else (0 if scale == "minmax" else None),
        vmax=2 if scale == "zscore" else (1 if scale == "minmax" else None),
        center=0 if scale == "zscore" else None,
        linewidths=0.25,
        linecolor="#F2F2F2",
        cbar_kws={
            "label": {
                "zscore": "Scaled mean expression",
                "minmax": "Relative mean expression",
                "raw": "Mean expression",
            }[scale],
            "shrink": 0.72,
        },
    )
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=10)
    if title:
        ax.set_title(title, pad=12)

    marker_groups = [k for k, v in marker_genes_dict.items() if any(g in genes for g in v)]
    bar_colors = _get_category_colors(
        adata,
        groupby,
        marker_groups,
        palette=marker_group_colors,
    )

    start = 0
    for marker_group, panel_genes in marker_genes_dict.items():
        present = [g for g in panel_genes if g in genes]
        if not present:
            continue
        end = start + len(present)
        ax.vlines(end, *ax.get_ylim(), colors="#D0D0D0", linewidth=0.7)
        if show_marker_group_bar:
            ax.add_patch(
                Rectangle(
                    (start, 1.015),
                    len(present),
                    0.045,
                    transform=ax.get_xaxis_transform(),
                    facecolor=bar_colors[marker_group],
                    edgecolor="none",
                    clip_on=False,
                )
            )
        ax.text(
            (start + end) / 2,
            1.075 if show_marker_group_bar else 1.025,
            marker_group,
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="bottom",
            fontsize=9,
            color="#333333",
            clip_on=False,
        )
        start = end

    fig.tight_layout()
    if save:
        fig.savefig(save, dpi=dpi, bbox_inches="tight")
    return fig, ax, mat



def setGrid(isNeed):
    import matplotlib as mpl, seaborn as sns
    import scanpy as sc
    import matplotlib.pyplot as plt

    plt.style.use('default')  # 取消自定义样式
    plt.rcParams['pdf.fonttype'] = 42  # Type 42 (TrueType) 字体，可编辑
    plt.rcParams['ps.fonttype'] = 42  # PostScript 字体也设置为 Type 42
    sc.settings.set_figure_params(dpi=300, dpi_save=600)
    if not isNeed:
        sns.set_theme(style="white", rc={"axes.grid": False})

import scipy.sparse as sp
def issparse(adata):
    is_sparse = sp.issparse(adata.X)
    print(f"adata.X 是稀疏矩阵吗？: {is_sparse}")
    #adata.X = sp.csr_matrix(adata.X) #转化为稀疏矩阵
    return is_sparse



def myStyle():
    import matplotlib
    import warnings
    import scanpy as sc
    warnings.filterwarnings("ignore")
    sc.settings.n_jobs = 30
    # vector_friendly: 细胞量巨大（比如十几万细胞）时，它会自动把密密麻麻的“散点”转换为像素位图
    # scanpy=True:重置为 Scanpy 官方推荐的配色体系和样式底包
    sc.set_figure_params(scanpy=True, dpi=500, dpi_save=500, frameon=False, vector_friendly=True, figsize=(10, 10),
                         format='png')
    matplotlib.rcParams['pdf.fonttype'] = 42
    matplotlib.rcParams["axes.grid"] = False


def setMyStyle(bottom = True,left = True,top = False,right = False):
    import matplotlib
    import warnings
    import scanpy as sc

    warnings.filterwarnings("ignore")
    sc.settings.n_jobs = 30

    sc.set_figure_params(
        scanpy=True,
        dpi=500,
        dpi_save=500,
        frameon=True,          # 保留坐标轴和刻度
        vector_friendly=True,
        #figsize=(10, 10),
        format="png"
    )

    matplotlib.rcParams["axes.grid"] = False
    matplotlib.rcParams["pdf.fonttype"] = 42
    matplotlib.rcParams["ps.fonttype"] = 42

    # 保留左/下刻度线，去掉上/右刻度线
    matplotlib.rcParams["xtick.bottom"] = bottom
    matplotlib.rcParams["ytick.left"] = left
    matplotlib.rcParams["xtick.top"] = top
    matplotlib.rcParams["ytick.right"] = right

    # 刻度线样式
    matplotlib.rcParams["xtick.major.size"] = 4
    matplotlib.rcParams["ytick.major.size"] = 4
    matplotlib.rcParams["xtick.major.width"] = 1
    matplotlib.rcParams["ytick.major.width"] = 1


import os

def mydir(path: str) -> str:
    # exist_ok=True: 如果目录已存在，不会抛出异常（即什么都不做）
    # 并且它会自动创建多级父目录（类似 Linux 的 mkdir -p）
    os.makedirs(path, exist_ok=True)
    return path