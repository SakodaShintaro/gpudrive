# 変換

```bash
pip install nbconvert
```

```bash
python -m nbconvert --to python 01_scenario_loading.ipynb
```

# 環境

```bash
root@d5abbcd02f40:/workspace# python3 tutorial/01_scenario_loading.py 
RuntimeError: module compiled against ABI version 0x1000009 but this version of numpy is 0x2000000
/usr/local/lib/python3.11/dist-packages/pygame/pkgdata.py:25: UserWarning: pkg_resources is deprecated as an API. See https://setuptools.pypa.io/en/latest/pkg_resources.html. The pkg_resources package is slated for removal as early as 2025-11-30. Refrain from using this package or pin to Setuptools<81.
  from pkg_resources import resource_stream, resource_exists
RuntimeError: module compiled against ABI version 0x1000009 but this version of numpy is 0x2000000
Segmentation fault (core dumped)
```

というエラーが出たため、

```bash
pip uninstall numpy -y
pip install "numpy<2.0"
```

により解消した。
