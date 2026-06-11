import sys

print(sys.executable)
for module_name in [
    "scanpy",
    "anndata",
    "omicverse",
    "cellphonedb",
    "numpy",
    "pandas",
    "matplotlib",
    "seaborn",
]:
    try:
        module = __import__(module_name)
        print(module_name, "OK", getattr(module, "__version__", ""))
    except Exception as exc:
        print(module_name, "ERR", type(exc).__name__, str(exc))
