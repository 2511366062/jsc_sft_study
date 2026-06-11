# LT Tumor-Centered Cell-Cell Communication Analysis

## Analysis Scope

This analysis focuses on ligand-receptor communication within the tumor tissue compartment (LT). The main biological question is whether SFT tumor cells form distinct immune and inflammatory niches with myeloid cells and T/NK cells.

Two tumor-centered communication modules are prioritized:

```text
LT SFT tumor <-> LT myeloid
LT SFT tumor <-> LT T/NK
```

Cross-compartment interactions such as `LZ immune cells <-> LT tumor cells` are not used as the main analysis because CellPhoneDB infers potential ligand-receptor interactions among cells assumed to coexist in the same local microenvironment. Therefore, LT-internal interactions are more biologically interpretable for tumor niche communication.

## Software And Database

The analysis is performed using:

```text
omicverse 2.1.2
CellPhoneDB v5 workflow through omicverse
CellPhoneDB database: cellphoneDB/cellphonedb.zip
```

Recommended running parameters for exploratory analysis:

```python
cpdb_results, adata_cpdb = ov.single.run_cellphonedb_v5(
    adata,
    cpdb_file_path="./cellphoneDB/cellphonedb.zip",
    celltype_key="celltype",
    min_cell_fraction=0.005,
    min_genes=200,
    min_cells=3,
    iterations=50,
    threshold=0.1,
    pvalue=0.05,
    threads=10,
    output_dir="./cellphoneDB/results/LT_tumor_centered",
    cleanup_temp=True,
)
```

For final figures, `iterations=1000` can be used after the target cell groups and pathways are fixed.

## Algorithm Principle

CellPhoneDB infers statistically enriched ligand-receptor interactions between annotated cell groups.

The core steps are:

1. Ligands and receptors are matched against a curated ligand-receptor database.
2. For each pair of cell types, the average expression of ligand and receptor genes is calculated.
3. Interactions are retained when ligand and receptor expression pass expression thresholds in the corresponding sender and receiver cell groups.
4. Cell labels are permuted repeatedly to estimate whether an observed interaction score is higher than expected by chance.
5. Significant interactions are reported using empirical permutation-derived P values.
6. Interactions are grouped into pathway-like classifications, such as `Signaling by Chemokines`, `Adhesion by Collagen/Integrin`, and `Signaling by HLA`.

Important interpretation:

```text
CellPhoneDB reports inferred ligand-receptor communication potential, not direct physical cell contact.
```

## LT SFT Tumor And LT Myeloid Communication

### Biological Rationale

The LT tumor-myeloid module is used to connect the myeloid DEG results with tumor-centered immune remodeling. Previous DEG and subtype analyses suggested that LT myeloid cells are associated with:

```text
ECM remodeling
collagen organization
chemotaxis and leukocyte migration
inflammatory activation
TAM survival or polarization
antigen presentation
immune regulation
```

Therefore, selected pathways should emphasize stromal remodeling, inflammatory recruitment, TAM activation, antigen presentation, and immune suppression.

### Selected Pathways

```python
tumor_myeloid_story_pathways = [
    "Adhesion by Collagen/Integrin",
    "Adhesion by Fibronectin",
    "Signaling by Fibronectin",
    "Adhesion by Osteopontin",
    "Adhesion by Thrombospondin",
    "Signaling by Chemokines",
    "Signaling by Interleukin",
    "Signaling by Tumor necrosis factor",
    "Signaling by Colony-Stimulating factor",
    "Signaling by Complement",
    "Signaling by HLA",
    "Signaling by Pro-MHC",
    "Signaling by Galectin",
    "Signaling by Transforming growth factor",
    "Signaling by MIF",
]
```

### Why These Pathways Were Selected

| Pathway group | Selected pathways | Biological interpretation |
|---|---|---|
| ECM remodeling | `Adhesion by Collagen/Integrin`, `Adhesion by Fibronectin`, `Signaling by Fibronectin`, `Adhesion by Thrombospondin` | Links tumor and TAM states to extracellular matrix remodeling, collagen organization, adhesion, and stromal niche formation. |
| SPP1-associated TAM niche | `Adhesion by Osteopontin` | Captures SPP1-related tumor-myeloid interaction, consistent with SPP1-positive TAM biology and tissue remodeling. |
| Inflammatory recruitment | `Signaling by Chemokines`, `Signaling by Interleukin`, `Signaling by Tumor necrosis factor` | Connects chemotaxis, inflammatory cytokines, and leukocyte migration signals observed in DEG/enrichment results. |
| TAM survival and activation | `Signaling by Colony-Stimulating factor`, `Signaling by Complement`, `Signaling by MIF` | Represents myeloid maintenance, complement-associated inflammatory activation, and MIF-mediated tumor-myeloid crosstalk. |
| Antigen presentation | `Signaling by HLA`, `Signaling by Pro-MHC` | Supports HLA-high/APC-like myeloid communication and antigen presentation programs. |
| Immune regulation and fibrosis | `Signaling by Galectin`, `Signaling by Transforming growth factor` | Captures immunoregulatory and fibrosis-associated tumor-myeloid signaling. |

## LT SFT Tumor And LT T/NK Communication

### Biological Rationale

The LT tumor-T/NK module is used to evaluate whether tumor cells communicate with cytotoxic and inflammatory lymphocytes. This analysis should focus less on broad ECM remodeling and more on:

```text
T/NK recruitment
immune cell adhesion
antigen presentation
interferon and inflammatory signaling
immune checkpoint-like regulation
tumor-derived immunosuppressive signals
```

### Selected Pathways

```python
tnk_tumor_story_pathways = [
    "Signaling by Chemokines",
    "Signaling by Lymphotactin",
    "Adhesion by ICAM",
    "Adhesion by VCAM",
    "Signaling by Selectin",
    "Signaling by HLA",
    "Signaling by Pro-MHC",
    "Signaling by Interferon",
    "Signaling by Interleukin",
    "Signaling by Tumor necrosis factor",
    "Signaling by Complement",
    "Signaling by Galectin",
    "Signaling by Poliovirus receptor",
    "Signaling by Transforming growth factor",
    "Adhesion by Osteopontin",
]
```

### Why These Pathways Were Selected

| Pathway group | Selected pathways | Biological interpretation |
|---|---|---|
| Lymphocyte recruitment | `Signaling by Chemokines`, `Signaling by Lymphotactin` | Captures CCL/CXCL and XCL-like signals involved in T/NK trafficking toward tumor regions. |
| Immune cell adhesion | `Adhesion by ICAM`, `Adhesion by VCAM`, `Signaling by Selectin` | Represents immune adhesion, rolling, and retention signals that may mediate tumor-lymphocyte contact. |
| Antigen presentation | `Signaling by HLA`, `Signaling by Pro-MHC` | Links tumor or APC-like signaling to T cell recognition and antigen presentation-related DEG programs. |
| Cytotoxic and inflammatory response | `Signaling by Interferon`, `Signaling by Interleukin`, `Signaling by Tumor necrosis factor`, `Signaling by Complement` | Connects tumor-T/NK communication with interferon response, inflammatory cytokine signaling, and complement-related immune activation. |
| Immune regulation | `Signaling by Galectin`, `Signaling by Poliovirus receptor` | Captures checkpoint-like or suppressive interactions, including Galectin-related and PVR/TIGIT/CD226-like axes. |
| Tumor suppressive niche | `Signaling by Transforming growth factor`, `Adhesion by Osteopontin` | Represents TGF-beta and SPP1-associated immunosuppressive or remodeling signals that may dampen cytotoxic immunity. |

## Suggested Figure Strategy

For each module, recommended plots are:

```text
1. Overall dot heatmap for all selected LT cell groups
2. Pathway-specific dot heatmap for selected story pathways
3. Chord or network plot for key pathways
4. Ligand-receptor contribution plot for top interactions
5. Sankey plot for sender-receiver directionality
```

Recommended figure order:

```text
LT tumor <-> myeloid communication
  -> ECM/SPP1/TGF/MIF/chemokine/HLA-centered story

LT tumor <-> T/NK communication
  -> chemokine/HLA/interferon/checkpoint-like/TGF-centered story
```

## Interpretation Notes

The communication results should be integrated with DEG and enrichment analysis rather than interpreted alone.

Recommended wording:

```text
CellPhoneDB analysis suggested enhanced ligand-receptor communication potential between SFT tumor cells and immune cell subsets in the LT tumor microenvironment.
```

Avoid overstatement:

```text
These results indicate inferred communication potential, not direct spatial contact or experimentally validated signaling.
```

