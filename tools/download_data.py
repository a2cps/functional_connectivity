from nilearn import datasets

data_dir = "/Users/psadil/git/a2cps/postfmriprep/src/postfmriprep/data"

for dimension in [64, 128, 256, 512, 1024]:
    for mm in [2, 3]:
        datasets.fetch_atlas_difumo(
            dimension=dimension,
            resolution_mm=mm,
            data_dir=data_dir,
            legacy_format=False,
        )

for n in [400]:
    for networks in [7, 17]:
        datasets.fetch_atlas_schaefer_2018(
            n_rois=n, resolution_mm=2, yeo_networks=networks, data_dir=data_dir
        )
