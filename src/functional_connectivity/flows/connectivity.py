import tempfile
from pathlib import Path
from typing import Iterable

import ancpbids
import nibabel as nb
import numpy as np
import pandas as pd
import prefect
from nilearn import maskers, signal
from nilearn.connectome import ConnectivityMeasure, connectivity_matrices
from pydantic.dataclasses import dataclass
from sklearn import covariance

from functional_connectivity import datasets, utils
from functional_connectivity.task import compcor, img
from functional_connectivity.task import utils as task_utils


@dataclass(frozen=True)
class Coordinate:
    region: int
    seed: tuple[int, int, int]


def df_to_coordinates(dataframe: pd.DataFrame) -> frozenset[Coordinate]:
    coordinates = set()
    for row in dataframe.itertuples():
        coordinates.add(
            Coordinate(region=row.region, seed=(row.x, row.y, row.z))
        )

    return frozenset(coordinates)


def get_baliki_coordinates() -> frozenset[Coordinate]:
    coordinates = datasets.get_baliki_lut()
    return df_to_coordinates(coordinates)


def get_power_coordinates() -> frozenset[Coordinate]:
    coordinates = datasets.get_coordinates_power2011()
    return df_to_coordinates(coordinates)


def _update_labels(
    resampled_labels_img: nb.Nifti1Image, labels: pd.DataFrame
) -> pd.DataFrame:
    resampled_labels = np.unique(
        np.asarray(resampled_labels_img.dataobj, dtype=int)
    )
    out = labels[labels["region"].isin(resampled_labels)]
    # - 1 is for background value
    if not (len(resampled_labels) - 1) == out.shape[0]:
        msg = "we appear to have lost labels. this should not be possible."
        raise AssertionError(msg)

    return out


def _do_connectivity(
    time_series: np.ndarray,
    labels: Iterable[int],
    estimator: covariance.EmpiricalCovariance,
) -> pd.DataFrame:
    # need to do standardization manually due to ddof issue
    # https://github.com/nilearn/nilearn/issues/3406
    connectivity_measure = ConnectivityMeasure(
        cov_estimator=estimator(store_precision=False), kind="covariance"
    )
    covariance_matrix: np.ndarray = connectivity_measure.fit_transform(
        [
            signal._standardize(
                time_series, detrend=False, standardize="zscore_sample"
            )
        ]
    ).squeeze()  # type: ignore
    return utils._mat_to_df(
        connectivity_matrices.cov_to_corr(covariance_matrix), labels=labels
    )


@prefect.task
@utils.cache_dataframe
def get_coordinates_connectivity(
    img: Path,
    coordinates: frozenset[Coordinate],
    estimator: covariance.EmpiricalCovariance,
    radius: int = 5,  # " ... defined as 10-mm spheres centered ..."
) -> pd.DataFrame:
    masker = maskers.NiftiSpheresMasker(
        seeds=[x.seed for x in coordinates],
        radius=radius,
        standardize=False,
        detrend=False,
    )
    time_series: np.ndarray = masker.fit_transform(img)

    return _do_connectivity(
        time_series=time_series,
        labels=[x.region for x in coordinates],
        estimator=estimator,
    )


@prefect.task
@utils.cache_dataframe
def get_maps_connectivity(
    img: Path,
    maps: datasets.Labels,
    estimator: covariance.EmpiricalCovariance,
) -> pd.DataFrame:
    masker = maskers.NiftiMapsMasker(
        maps_img=maps.labels_img,
        standardize=False,
        detrend=False,
        resampling_target="data",
    )
    time_series: np.ndarray = masker.fit_transform(img)

    return _do_connectivity(
        time_series=time_series,
        labels=maps.labels["region"].to_list(),
        estimator=estimator,
    )


@prefect.task
@utils.cache_dataframe
def get_labels_connectivity(
    img: Path,
    labels: datasets.Labels,
    estimator: covariance.EmpiricalCovariance,
) -> pd.DataFrame:
    masker = maskers.NiftiLabelsMasker(
        labels_img=labels.labels_img,
        standardize=False,
        detrend=False,
        resampling_target="data",
    )
    # need to fit here in case of loss of labels
    time_series: np.ndarray = masker.fit_transform(img)
    labels_lookup = _update_labels(masker._resampled_labels_img_, labels.labels)

    return _do_connectivity(
        time_series=time_series,
        labels=labels_lookup["region"].to_list(),
        estimator=estimator,
    )


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


def _get_labels() -> dict[str, datasets.Labels]:
    out = {}
    for n in [400]:
        for networks in [7, 17]:
            out.update(
                {
                    f"schaefer_nrois-{n}_resolution-2_networks-{networks}": datasets.get_atlas_schaefer_2018(
                        n_rois=n, resolution_mm=2, yeo_networks=networks
                    )
                }
            )
    for resolution in [2]:
        out.update(
            {
                f"fan_resolution-{resolution}": datasets.get_fan_atlas(
                    resolution=resolution
                )
            }
        )
    out.update(
        {
            f"gordon_space-mni_resolution-{resolution}": datasets.get_atlas_gordon_2016(
                resolution_mm=resolution, space="MNI"
            )
        }
    )
    return out


def _get_maps() -> dict[str, datasets.Labels]:
    out = {}
    for dimension in [64, 128, 256, 512, 1024]:
        for mm in [2]:
            out.update(
                {
                    f"difumo_dimension-{dimension}_resolution-{mm}mm": datasets.get_difumo(
                        dimension=dimension, resolution_mm=mm
                    )
                }
            )
    return out


def _get_coordinates() -> dict[str, Coordinate]:
    return {"dmn": get_baliki_coordinates(), "power": get_power_coordinates()}


def _get_estimators() -> dict[str, covariance.EmpiricalCovariance]:
    return {
        "empirical": covariance.EmpiricalCovariance,
        "oracle_approximating_shrinkage": covariance.OAS,
        "leodit_wolf": covariance.LedoitWolf,
        "minimum_covariance_determinant": covariance.MinCovDet,
    }


@prefect.flow
def connectivity_flow(
    subdirs: list[Path],
    outdirs: list[Path],
    high_pass: float | None = 0.01,
    low_pass: float | None = 0.1,
    n_non_steady_state_tr: int = 12,
    space: str = "MNI152NLin2009cAsym",
) -> None:
    if len(outdirs) == 1:
        output_dirs = outdirs * len(subdirs)
    elif len(outdirs) > 1 and (not (len(outdirs) == len(subdirs))):
        raise AssertionError
    else:
        output_dirs = outdirs

    labels = _get_labels()
    maps = _get_maps()
    coordinates = _get_coordinates()
    estimators = _get_estimators()

    for subdir, out in zip(subdirs, output_dirs):
        layout = ancpbids.BIDSLayout(str(subdir))
        with tempfile.TemporaryDirectory() as _tmpdir:
            tmpdir = Path(_tmpdir)
            connectivity_parts = []
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
                                / f"sub={sub}"
                                / f"ses={ses}"
                                / f"task={task}"
                                / f"run={run}"
                                / f"space={space}"
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
                            )

                            confounds = task_utils.update_confounds.submit(
                                out
                                / "connectivity-confounds"
                                / f"sub={sub}"
                                / f"ses={ses}"
                                / f"task={task}"
                                / f"run={run}"
                                / "part-0.parquet",
                                acompcor_file=acompcor,
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

                            cleaned = img.clean_img.submit(
                                out
                                / "connectivity-cleaned"
                                / f"sub-{sub}_ses-{ses}_task-{task}_run-{run}_desc-preproc_bold.nii.gz",
                                img=i,
                                confounds_file=confounds,
                                low_pass=low_pass,
                                high_pass=high_pass,
                                detrend=False,
                            )

                            for e, estimator in estimators.items():
                                for atlas, label in labels.items():
                                    connectivity_parts.append(
                                        get_labels_connectivity.submit(
                                            tmpdir
                                            / "connectivity"
                                            / f"sub={sub}"
                                            / f"ses={ses}"
                                            / f"task={task}"
                                            / f"run={run}"
                                            / f"space={space}"
                                            / f"atlas={atlas}"
                                            / f"estimator={e}"
                                            / "part-0.parquet",
                                            img=cleaned,
                                            labels=label,
                                            estimator=estimator,
                                        )
                                    )
                                for atlas, m in maps.items():
                                    connectivity_parts.append(
                                        get_maps_connectivity.submit(
                                            tmpdir
                                            / "connectivity"
                                            / f"sub={sub}"
                                            / f"ses={ses}"
                                            / f"task={task}"
                                            / f"run={run}"
                                            / f"space={space}"
                                            / f"atlas={atlas}"
                                            / f"estimator={e}"
                                            / "part-0.parquet",
                                            img=cleaned,
                                            maps=m,
                                            estimator=estimator,
                                        )
                                    )
                                for key, value in coordinates.items():
                                    connectivity_parts.append(
                                        get_coordinates_connectivity.submit(
                                            tmpdir
                                            / "connectivity"
                                            / f"sub={sub}"
                                            / f"ses={ses}"
                                            / f"task={task}"
                                            / f"run={run}"
                                            / f"space={space}"
                                            / f"atlas={key}"
                                            / f"estimator={e}"
                                            / "part-0.parquet",
                                            img=cleaned,
                                            coordinates=value,
                                            estimator=estimator,
                                        )
                                    )

            task_utils.merge_parquet.submit(
                files=connectivity_parts,
                outdir=out / "connectivity",
                partition_cols=["sub", "ses", "task"],
            )
