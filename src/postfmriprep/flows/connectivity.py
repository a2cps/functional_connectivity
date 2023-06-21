from pathlib import Path

import ancpbids
import nibabel as nb
import numpy as np
import pandas as pd
import prefect
from nilearn import maskers
from nilearn.connectome import ConnectivityMeasure
from pydantic.dataclasses import dataclass
from sklearn import covariance

from postfmriprep import utils
from postfmriprep.task import compcor
from postfmriprep.task import utils as task_utils


@dataclass(frozen=True)
class Coordinate:
    label: str
    seed: tuple[int, int, int]


def df_to_coordinates(dataframe: pd.DataFrame) -> frozenset[Coordinate]:
    coordinates = set()
    for row in dataframe.itertuples():
        coordinates.add(Coordinate(label=row.label, seed=(row.x, row.y, row.z)))

    return frozenset(coordinates)


@prefect.task
def get_baliki_coordinates() -> frozenset[Coordinate]:
    return frozenset(
        {
            Coordinate(label="mPFC", seed=(2, 52, -2)),
            Coordinate(label="rNAc", seed=(10, 12, -8)),
            Coordinate(label="rInsula", seed=(40, -6, -2)),
            Coordinate(label="S1/M1", seed=(-32, -34, 66)),
        },
    )


def get_power_coordinates() -> frozenset[Coordinate]:
    from nilearn import datasets

    rois: pd.DataFrame = datasets.fetch_coords_power_2011(
        legacy_format=False
    ).rois
    rois.query(
        "not roi in [127, 183, 184, 185, 243, 244, 245, 246]", inplace=True
    )
    rois.rename(columns={"roi": "label"}, inplace=True)
    return df_to_coordinates(rois)


@prefect.task
@utils.cache_dataframe
def spheres_connectivity(
    img: Path,
    confounds_file: Path,
    coordinates: frozenset[Coordinate],
    radius: int = 5,  # " ... defined as 10-mm spheres centered ..."
    high_pass: float | None = None,
    low_pass: float | None = None,
    detrend: bool = False,
) -> pd.DataFrame:
    """
    for confounds,
    - Friston24,
    - top 5 principal components
    """
    confounds = pd.read_parquet(confounds_file)

    n_tr = confounds.shape[0]
    nii: nb.Nifti1Image = nb.load(img).slicer[:, :, :, -n_tr:]
    masker = maskers.NiftiSpheresMasker(
        seeds=[x.seed for x in coordinates],
        radius=radius,
        high_pass=high_pass,
        low_pass=low_pass,
        t_r=utils.get_tr(nii),
        standardize=False,
        standardize_confounds=True,
        detrend=detrend,
    )
    # confounds are already sliced
    time_series = masker.fit_transform(nii, confounds=confounds)
    connectivity_measure = ConnectivityMeasure(
        cov_estimator=covariance.EmpiricalCovariance(store_precision=False),  # type: ignore
        kind="correlation",
    )
    correlation_matrix = connectivity_measure.fit_transform([time_series]).squeeze()  # type: ignore
    df = utils._mat_to_df(correlation_matrix, [x.label for x in coordinates])
    df["connectivity"] = np.arctanh(df["connectivity"])
    return df


@prefect.task
@utils.cache_dataframe
def get_labels_connectivity(
    img: Path,
    confounds_file: Path,
    labels_img: Path,
    high_pass: float | None = None,
    low_pass: float | None = None,
    detrend: bool = False,
) -> pd.DataFrame:
    """
    for confounds,
    - Friston24,
    - top 5 principal components
    """
    confounds = pd.read_parquet(confounds_file)

    n_tr = confounds.shape[0]
    nii: nb.Nifti1Image = nb.load(img).slicer[:, :, :, -n_tr:]
    masker = maskers.NiftiLabelsMasker(
        labels_img=labels_img,
        high_pass=high_pass,
        low_pass=low_pass,
        t_r=utils.get_tr(nii),
        standardize=False,
        standardize_confounds=True,
        detrend=detrend,
        resampling_target="data",
    )
    # confounds are already sliced
    time_series = masker.fit_transform(nii, confounds=confounds)
    connectivity_measure = ConnectivityMeasure(
        cov_estimator=covariance.EmpiricalCovariance(store_precision=False),  # type: ignore
        kind="correlation",
    )
    correlation_matrix: np.ndarray = connectivity_measure.fit_transform(
        [time_series]
    ).squeeze()  # type: ignore
    df = utils._mat_to_df(
        correlation_matrix,
        [str(x + 1) for x in range(correlation_matrix.shape[0])],
    ).assign(
        img=utils.img_stem(img),
        confounds="+".join([str(x) for x in confounds.columns.values]),
    )
    return df


def _get_probseg(layout, sub, ses, space) -> list[Path]:
    return [
        task_utils._get.fn(
            layout=layout,
            filters={
                "sub": str(sub),
                "ses": str(ses),
                "space": str(space),
                "label": label,
                "suffix": "probseg",
                "extension": ".nii.gz",
            },
        )
        for label in ["GM", "WM", "CSF"]
    ]


@prefect.flow
def connectivity_flow(
    subdirs: list[Path],
    outdirs: list[Path],
    high_pass: float | None = 0.01,
    low_pass: float | None = 0.1,
    n_non_steady_state_tr: int = 12,
    detrend: bool = False,
    space: str = "MNI152NLin2009cAsym",
) -> None:
    if len(outdirs) == 1:
        output_dirs = outdirs * len(subdirs)
    elif len(outdirs) > 1 and (not (len(outdirs) == len(subdirs))):
        raise AssertionError
    else:
        output_dirs = outdirs

    baliki_coordinates = get_baliki_coordinates.submit()

    for subdir, out in zip(subdirs, output_dirs):
        layout = ancpbids.BIDSLayout(str(subdir))
        for sub in layout.get_subjects():
            for ses in layout.get_sessions(sub=sub):
                probseg = _get_probseg(
                    layout=layout, sub=sub, ses=ses, space=space
                )
                for task in layout.get_tasks(sub=sub, ses=ses):
                    for run in layout.get_runs(sub=sub, ses=ses, task=task):
                        i = task_utils._get(
                            layout=layout,
                            filters={
                                "sub": str(sub),
                                "ses": str(ses),
                                "task": str(task),
                                "run": str(run),
                                "space": str(space),
                                "desc": "preproc",
                                "suffix": "bold",
                                "extension": ".nii.gz",
                            },
                        )
                        acompcor = compcor.do_compcor.submit(
                            out
                            / "acompcor"
                            / f"sub={sub}/ses={ses}/task={task}/run={run}/space={space}"
                            / "part-0.parquet",
                            img=i,
                            boldref=task_utils._get(
                                layout=layout,
                                filters={
                                    "sub": str(sub),
                                    "ses": str(ses),
                                    "task": str(task),
                                    "run": str(run),
                                    "space": str(space),
                                    "suffix": "boldref",
                                    "extension": ".nii.gz",
                                },
                            ),
                            probseg=probseg,
                            high_pass=high_pass,
                            low_pass=low_pass,
                            n_non_steady_state_tr=n_non_steady_state_tr,
                            detrend=detrend,
                        )

                        confounds = task_utils.update_confounds.submit(
                            out
                            / "connectivity-confounds"
                            / f"sub={sub}"
                            / f"ses={ses}"
                            / f"task={task}"
                            / f"run={run}"
                            / "part-0.parquet",
                            acompcor_file=acompcor,  # type: ignore
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
                            label="WM+CSF",
                            n_non_steady_state_tr=n_non_steady_state_tr,
                        )

                        spheres_connectivity.submit(
                            out
                            / "connectivity-dmn"
                            / f"sub={sub}/ses={ses}/task={task}/run={run}/space={space}"
                            / "part-0.parquet",
                            img=i,
                            coordinates=baliki_coordinates,  # type: ignore
                            confounds_file=confounds,  # type: ignore
                            high_pass=high_pass,
                            low_pass=low_pass,
                            detrend=detrend,
                        )

                        get_labels_connectivity.submit(
                            out
                            / "connectivity-labels"
                            / f"sub={sub}/ses={ses}/task={task}/run={run}/space={space}"
                            / "part-0.parquet",
                            img=i,
                            confounds_file=confounds,  # type: ignore
                            labels_img=utils.get_fan_atlas_file(),
                            high_pass=high_pass,
                            low_pass=low_pass,
                            detrend=detrend,
                        )
