from pathlib import Path

import nibabel as nb
import pandas as pd
import prefect
from nilearn import image

from functional_connectivity import utils


@prefect.task
@utils.cache_nii
def clean_img(
    img: Path,
    confounds_file: Path,
    mask_img: Path | None = None,
    high_pass: float | None = None,
    low_pass: float | None = None,
    detrend: bool = False,
) -> nb.nifti1.Nifti1Image:
    confounds = pd.read_parquet(confounds_file)
    n_tr = confounds.shape[0]
    nii: nb.nifti1.Nifti1Image = nb.loadsave.load(img).slicer[:, :, :, -n_tr:]  # type: ignore
    out: nb.nifti1.Nifti1Image = image.clean_img(
        nii,
        high_pass=high_pass,
        low_pass=low_pass,
        t_r=utils.get_tr(nii),
        standardize=False,
        detrend=detrend,
        mask_img=mask_img,
        clean__standardize_confounds="zscore_sample",
    )  # type: ignore
    return out
