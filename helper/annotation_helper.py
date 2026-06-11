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