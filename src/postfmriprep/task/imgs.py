import tempfile
import typing
from pathlib import Path

import nibabel as nb
import numpy as np
import prefect

from postfmriprep import utils


@prefect.task
@utils.cache_nii
def correct_bias(
    img: Path,
    mask: Path,
    s: int | None = None,
    b: float | str | None = None,
    c: int | str | None = None,
) -> nb.Nifti1Image:
    import os

    image = nb.load(img)
    avg = nb.Nifti1Image(
        dataobj=image.get_fdata().mean(-1),
        affine=image.affine,
        header=image.header,
    )
    args = ""
    if s:
        args += f" -s {s}"
    if b:
        args += f" -b [ {b} ]"
    if c:
        args += f" -c [ {c} ]"

    with tempfile.NamedTemporaryFile(suffix=".nii.gz") as f:
        avg.to_filename(f.name)
        with tempfile.NamedTemporaryFile(suffix=".nii.gz") as o:
            with tempfile.NamedTemporaryFile(suffix=".nii.gz") as biasfile:
                os.system(
                    f"N4BiasFieldCorrection -i {f.name} -o [ {o.name},{biasfile.name} ] -x {mask} {args}"
                )
                bias_field: np.ndarray = nb.load(biasfile.name).get_fdata()

    debiased = image.get_fdata()
    for tr in range(debiased.shape[-1]):
        debiased[:, :, :, tr] /= bias_field

    return nb.Nifti1Image(
        dataobj=debiased, affine=image.affine, header=image.header
    )


@prefect.task()
@utils.cache_nii
def clean_img(
    img: Path,
    mask: Path,
    confounds_file: Path | None = None,
    high_pass: float | None = None,
    low_pass: float | None = None,
    detrend: bool = False,
    standardize: bool = False,
    fwhm: float
    | np.ndarray
    | tuple[float]
    | list[float]
    | typing.Literal["fast"]
    | None = None,
    winsorize: bool = False,
    to_percentchange: bool = False,
    n_non_steady_state_tr: int = 0,
) -> nb.Nifti1Image:
    import pandas as pd
    from nilearn import image

    if confounds_file:
        confounds = pd.read_parquet(confounds_file)
    else:
        confounds = None
    nii: nb.Nifti1Image = nb.load(img).slicer[:, :, :, n_non_steady_state_tr:]

    assert len(nii.shape) == 4

    if detrend:
        nii = utils.detrend(nii, mask=mask)
    if winsorize:
        nii = _winsorize(nii)
    if to_percentchange:
        nii = _to_local_percent_change(nii)

    # note that this relies on default behavior for standardizing confounds when passed to image.clean
    nii_clean: nb.Nifti1Image = image.clean_img(
        imgs=nii,
        high_pass=high_pass,
        low_pass=low_pass,
        t_r=utils.get_tr(nii),
        confounds=confounds,
        standardize=standardize,
        detrend=False,
        mask_img=mask,
    )  # type: ignore

    if fwhm:
        nii_smoothed: nb.Nifti1Image = image.smooth_img(nii_clean, fwhm=fwhm)  # type: ignore
        return nii_smoothed
    else:
        return nii_clean


def _to_local_percent_change(
    img: nb.Nifti1Image, fwhm: float = 16
) -> nb.Nifti1Image:
    from nibabel import processing

    avg = nb.Nifti1Image(
        dataobj=img.get_fdata().mean(-1), affine=img.affine, header=img.header
    )
    smoothed = processing.smooth_image(avg, fwhm=fwhm)
    pc = np.asarray(img.get_fdata().copy())
    for tr in range(img.shape[-1]):
        pc[:, :, :, tr] -= avg.get_fdata()
        pc[:, :, :, tr] /= smoothed.get_fdata()
    pc *= 100
    pc += 100

    return nb.Nifti1Image(dataobj=pc, affine=img.affine, header=img.header)


def _winsorize(img: nb.Nifti1Image, std: float = 3) -> nb.Nifti1Image:
    # from scipy.stats import mstats
    # from scipy import stats

    ms = img.get_fdata().mean(axis=-1, keepdims=True)
    stds = img.get_fdata().std(axis=-1, ddof=1, keepdims=True)

    Z = np.abs((img.get_fdata() - ms) / stds)
    # Z = np.abs(stats.zscore(img.get_fdata(), axis=-1, ddof=1))
    if (Z > std).mean() > 0.01:
        raise ValueError("We're removing more than 1% of values!")

    replacements = ms + std * stds * np.sign(img.get_fdata() - ms)
    winsorized = img.get_fdata().copy()
    winsorized[Z > std] = replacements[Z > std]

    # winsorized = mstats.winsorize(img.get_fdata(), limits=[lower, upper], axis=-1)

    return nb.Nifti1Image(
        dataobj=winsorized, affine=img.affine, header=img.header
    )
