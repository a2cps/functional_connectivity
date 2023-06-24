from pathlib import Path

import pandas as pd
from nilearn import datasets as nilearn_datasets

data_dir = Path("src/functional_connectivity/data")

for dimension in [64, 128, 256, 512, 1024]:
    for mm in [2, 3]:
        nilearn_datasets.fetch_atlas_difumo(
            dimension=dimension,
            resolution_mm=mm,
            data_dir=data_dir,
            legacy_format=False,
        )

for n in [400]:
    for networks in [7, 17]:
        nilearn_datasets.fetch_atlas_schaefer_2018(
            n_rois=n, resolution_mm=2, yeo_networks=networks, data_dir=data_dir
        )


rois: pd.DataFrame = nilearn_datasets.fetch_coords_power_2011(
    legacy_format=False
).rois
rois.query("not roi in [127, 183, 184, 185, 243, 244, 245, 246]", inplace=True)
rois.rename(columns={"roi": "region"}).to_csv(
    data_dir / "power2011.csv", index=False
)
