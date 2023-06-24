from importlib import resources
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import pydantic
from pydantic.dataclasses import dataclass


@dataclass(frozen=True, config={"arbitrary_types_allowed": True})
class NilearnBunch:
    maps: pydantic.FilePath
    labels: pd.DataFrame


def get_mpfc_mask() -> Path:
    """Return mPFC mask produced from Smallwood et al. replication.

    Returns:
        Path: Path to mPFC mask.
    """
    with resources.path(
        "postfmriprep.data", "smallwood_mpfc_MNI152_1p5.nii.gz"
    ) as f:
        mpfc = f
    return mpfc


def get_rs2_labels() -> Path:
    with resources.path("postfmriprep.data", "TD_label.nii") as f:
        labels = f
    return labels


def get_fan_atlas_file(resolution: Literal["2mm", "3mm"] = "2mm") -> Path:
    """Return file from ToPS model (https://doi.org/10.1038/s41591-020-1142-7)

    Returns:
        Path: Path to atlas
    """
    with resources.path(
        "postfmriprep.data", f"Fan_et_al_atlas_r279_MNI_{resolution}.nii.gz"
    ) as f:
        atlas = f
    return atlas


def get_power2011_coordinates_file() -> Path:
    """Return file for volumetric atlas from Power et al. 2011 (https://doi.org/10.1016/j.neuron.2011.09.006)

    Returns:
        Path: Path to atlas
    """
    with resources.path("postfmriprep.data", "power2011.tsv") as f:
        atlas = f
    return atlas


def get_power2011_coordinates() -> pd.DataFrame:
    """Return dataframe volumetric atlas from Power et al. 2011 (https://doi.org/10.1016/j.neuron.2011.09.006)

    Returns:
        dataframe of coordinates
    """
    return pd.read_csv(
        get_power2011_coordinates_file(),
        delim_whitespace=True,
        index_col="ROI",
        dtype={"x": np.float16, "y": np.int16, "z": np.int16},
    )


def get_mni6gray_mask() -> Path:
    with resources.path("postfmriprep.data", "MNI152_T1_6mm_gray.nii.gz") as f:
        out = f
    return out


def get_difumo(
    dimension: Literal[64, 128, 256, 512, 1024], resolution_mm: Literal[2, 3]
) -> NilearnBunch:
    with resources.path(
        f"postfmriprep.data.difumo_atlases.{dimension}.{resolution_mm}mm",
        "maps.nii.gz",
    ) as f:
        maps = f
    with resources.path(
        f"postfmriprep.data.difumo_atlases.{dimension}",
        f"labels_{dimension}_dictionary.csv",
    ) as f:
        labels = pd.read_csv(f)
    return NilearnBunch(maps=maps, labels=labels)


def get_atlas_schaefer_2018(
    n_rois: Literal[400],
    resolution_mm: Literal[2],
    yeo_networks: Literal[7, 17],
):
    with resources.path(
        "postfmriprep.data.schaefer_2018",
        f"Schaefer2018_{n_rois}Parcels_{yeo_networks}Networks_order_FSLMNI152_{resolution_mm}mm.nii.gz",
    ) as f:
        maps = f
    with resources.path(
        "postfmriprep.data.schaefer_2018",
        f"Schaefer2018_{n_rois}Parcels_{yeo_networks}Networks_order.txt",
    ) as f:
        labels_ = np.genfromtxt(
            f, usecols=1, dtype="S", delimiter="\t", encoding=None
        )
    # labels_ = np.insert(labels_wo_bg, 0, "Background")
    labels = pd.DataFrame.from_dict({"label": labels_}).reset_index(
        names="region"
    )
    labels = labels.assign(region=labels["region"] + 1)

    return NilearnBunch(maps=maps, labels=labels)


def get_atlas_baliki() -> pd.DataFrame:
    with resources.path("postfmriprep.data", "dmn.csv") as f:
        coordinates = pd.read_csv(f)
    return coordinates
