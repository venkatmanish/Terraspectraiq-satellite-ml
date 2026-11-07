# Data provenance

`raw/` holds the three original files of the **Statlog (Landsat Satellite)** dataset,
unmodified, as distributed by the UCI Machine Learning Repository (dataset id 146).

| File | Rows | SHA-256 |
|---|---|---|
| `sat.trn` | 4,435 | `e896dc88a960fa2404160fc4c3cb3dc53fcf4afd80ba920bf2d261bd42d12613` |
| `sat.tst` | 2,000 | `4b9167b8a92baafafed7c8809aef86d0683a5e685c19d98d119fa6974f0f2479` |
| `sat.doc` | — | `07f0312277b3bc4600548a0b75a848d72f1c88d9de416711b05cc2535d961dac` |

`terraspectra/data.py` checks the checksums of `sat.trn` and `sat.tst` every time it loads them.

**Re-download**

```bash
curl -L -o statlog.zip "https://archive.ics.uci.edu/static/public/146/statlog+landsat+satellite.zip"
unzip statlog.zip sat.trn sat.tst sat.doc -d data/raw/
```

**Source and licence.** Srinivasan, A. (1993). *Statlog (Landsat Satellite)* [Dataset].
UCI Machine Learning Repository. https://doi.org/10.24432/C55887 — licensed under
Creative Commons Attribution 4.0 (CC BY 4.0), which permits redistribution with attribution.

**Format.** Space-separated integers, 37 per line: 36 spectral values (4 Landsat MSS bands ×
9 pixels of a 3×3 neighbourhood, pixel-major, read left-to-right then top-to-bottom) and the
land-cover class of the centre pixel (1, 2, 3, 4, 5, 7; there is no class 6).
