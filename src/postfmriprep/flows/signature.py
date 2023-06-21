from pathlib import Path

import ancpbids
import nibabel as nb
import numpy as np
import polars as pl
import prefect
import pydantic
from nibabel import processing
from nilearn import _utils, masking
from pydantic.dataclasses import dataclass

from postfmriprep import utils
from postfmriprep.task import imgs
from postfmriprep.task import utils as task_utils


@dataclass(frozen=True)
class Func3d:
    label: str
    path: pydantic.FilePath
    dtype: str


def _make_mask(img) -> nb.Nifti1Image:
    return nb.Nifti1Image(
        dataobj=np.ones(img.shape[:3], dtype=np.uint8), affine=img.affine
    )


def _to_df3d(img, label: str, dtype="f8") -> pl.DataFrame:
    i = _utils.check_niimg(img)
    out = masking.apply_mask(
        i,
        _make_mask(i),
        dtype=dtype,
    )
    return pl.DataFrame(
        {"voxel": np.arange(out.shape[0], dtype=np.uint32), label: out}
    )


@prefect.task()
@utils.cache_dataframe
def to_df4d(img: Path, dtype: str = "f8") -> pl.DataFrame:
    i = _utils.check_niimg(img)
    out = masking.apply_mask(
        i,
        _make_mask(i),
        dtype=dtype,
    )
    d = (
        pl.DataFrame(out, schema=[str(x) for x in range(out.shape[1])])
        .with_columns(
            pl.Series(
                "t",
                (np.arange(out.shape[0]) * 800).astype(np.uint32),
            )
        )
        .melt(id_vars=["t"], value_name="signal", variable_name="voxel")
        .with_columns(pl.col("voxel").cast(pl.UInt32()))
    )

    return d


@prefect.task()
@utils.cache_dataframe
def to_parquet3d(
    fmriprep_func3ds: list[Func3d], func3ds: list[Func3d] | None = None
) -> pl.DataFrame:
    assert len(fmriprep_func3ds)

    d = _to_df3d(
        fmriprep_func3ds[0].path,
        label=fmriprep_func3ds[0].label,
        dtype=fmriprep_func3ds[0].dtype,
    )
    for func3d in fmriprep_func3ds[1:]:
        d = d.join(
            _to_df3d(
                func3d.path,
                label=func3d.label,
                dtype=func3d.dtype,
            ),
            on="voxel",
        )
    target = nb.load(fmriprep_func3ds[0].path)
    if func3ds:
        for func3d in func3ds:
            img: nb.Nifti1Image = nb.load(func3d.path)
            d = d.join(
                _to_df3d(
                    processing.resample_from_to(img, target, order=1),
                    label=func3d.label,
                    dtype=func3d.dtype,
                ),
                on="voxel",
            )
    return d


@prefect.task()
def _gather_to_resample(
    extra: list[Func3d],
    layout: ancpbids.BIDSLayout,
    sub: str,
    ses: str,
    space: str = "MNI152NLin2009cAsym",
) -> list[Func3d]:
    filters = {"sub": sub, "ses": ses, "space": space}
    GM = layout.get(label="GM", return_type="filename", **filters)[0]
    WM = layout.get(label="WM", return_type="filename", **filters)[0]
    CSF = layout.get(label="CSF", return_type="filename", **filters)[0]
    return [
        Func3d(
            label="GM",
            dtype="f8",
            path=Path(str(GM)),
        ),
        Func3d(
            label="CSF",
            dtype="f8",
            path=Path(str(CSF)),
        ),
        Func3d(
            label="WM",
            dtype="f8",
            path=Path(str(WM)),
        ),
    ] + extra


@prefect.task()
def _get_in_space(
    layout: ancpbids.BIDSLayout,
    sub: str,
    ses: str,
    task: str,
    run: str,
    space: str = "MNI152NLin2009cAsym",
) -> list[Func3d]:
    filters = {
        "sub": sub,
        "ses": ses,
        "space": space,
        "task": task,
        "run": run,
        "extension": ".nii.gz",
    }
    mask = layout.get(desc="brain", return_type="filename", **filters)[0]
    aparcaseg = layout.get(desc="aparcaseg", return_type="filename", **filters)[
        0
    ]
    aseg = layout.get(
        desc="aseg",
        return_type="filename",
        **filters,
    )[0]
    return [
        Func3d(
            label="brain",
            dtype="?",
            path=Path(str(mask)),
        ),
        Func3d(
            label="aparcaseg",
            dtype="uint16",
            path=Path(str(aparcaseg)),
        ),
        Func3d(
            label="aseg",
            dtype="uint8",
            path=Path(str(aseg)),
        ),
    ]


@prefect.flow()
def signature_flow(
    subdirs: list[Path],
    outdirs: list[Path],
    high_pass: float | None = None,
    low_pass: float | None = 0.1,
    n_non_steady_state_tr: int = 12,
    detrend: bool = True,
    fwhm: float | None = None,
    winsorize: bool = True,
    space: str = "MNI152NLin2009cAsym",
) -> None:
    if len(outdirs) == 1:
        output_dirs = outdirs * len(subdirs)
    elif len(outdirs) > 1 and (not (len(outdirs) == len(subdirs))):
        raise AssertionError
    else:
        output_dirs = outdirs

    for subdir, out in zip(subdirs, output_dirs):
        layout = ancpbids.BIDSLayout(str(subdir))
        for sub in layout.get_subjects():
            for ses in layout.get_sessions(sub=sub):
                func3ds = _gather_to_resample.submit(
                    extra=[
                        Func3d(
                            path=utils.get_nps_mask("group"),
                            label="NPS",
                            dtype="f8",
                        ),
                        Func3d(
                            path=utils.get_nps_mask("negative"),
                            label="NPSneg",
                            dtype="f8",
                        ),
                        Func3d(
                            path=utils.get_nps_mask("positive"),
                            label="NPSpos",
                            dtype="f8",
                        ),
                    ],
                    layout=layout,
                    sub=str(sub),
                    ses=str(ses),
                    space=space,
                )

                for task in layout.get_tasks(sub=sub, ses=ses):
                    for run in layout.get_runs(sub=sub, ses=ses, task=task):
                        confounds = task_utils.update_confounds.submit(
                            out
                            / "signature-confounds"
                            / f"sub={sub}"
                            / f"ses={ses}"
                            / f"task={task}"
                            / f"run={run}"
                            / "part-0.parquet",
                            confounds=task_utils._get(
                                layout=layout,
                                filters={
                                    "sub": str(sub),
                                    "ses": str(ses),
                                    "task": str(task),
                                    "run": str(run),
                                    "desc": "confounds",
                                    "extension": ".tsv",
                                },
                            ),
                            n_non_steady_state_tr=n_non_steady_state_tr,
                        )
                        preproc = task_utils._get(
                            layout=layout,
                            filters={
                                "sub": str(sub),
                                "ses": str(ses),
                                "task": str(task),
                                "run": str(run),
                                "space": space,
                                "desc": "preproc",
                                "extension": ".nii.gz",
                            },
                        )

                        i = imgs.clean_img.submit(
                            out / "signature-cleaned" / preproc.name,
                            img=preproc,  # type: ignore
                            mask=task_utils._get(
                                layout=layout,
                                filters={
                                    "sub": str(sub),
                                    "ses": str(ses),
                                    "task": str(task),
                                    "run": str(run),
                                    "desc": "brain",
                                    "extension": ".nii.gz",
                                },
                            ),
                            confounds_file=confounds,  # type: ignore
                            high_pass=high_pass,
                            low_pass=low_pass,
                            detrend=detrend,
                            fwhm=fwhm,
                            winsorize=winsorize,
                            to_percentchange=False,
                            n_non_steady_state_tr=n_non_steady_state_tr,
                        )
                        to_df4d.submit(
                            out
                            / "signature-cleaned-ds"
                            / f"sub={sub}"
                            / f"ses={ses}"
                            / f"task={task}"
                            / f"run={run}"
                            / "part-0.parquet",
                            img=i,  # type: ignore
                            dtype="f8",
                        )

                        fmriprep_func3ds = _get_in_space.submit(
                            layout=layout,
                            sub=str(sub),
                            ses=str(ses),
                            task=str(task),
                            run=str(run),
                            space=space,
                        )
                        to_parquet3d.submit(
                            out
                            / "signature-cleaned-ds3d"
                            / f"sub={sub}"
                            / f"ses={ses}"
                            / f"task={task}"
                            / f"run={run}"
                            / "part-0.parquet",
                            func3ds=func3ds,  # type: ignore
                            fmriprep_func3ds=fmriprep_func3ds,  # type: ignore
                        )
