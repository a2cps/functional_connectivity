from importlib import resources
from typing import Literal

import pandas as pd
import pydantic
from pydantic.dataclasses import dataclass


@dataclass(frozen=True, config={"arbitrary_types_allowed": True})
class Labels:
    labels_img: pydantic.FilePath
    labels: pd.DataFrame


def get_fan_atlas_lut() -> pd.DataFrame:
    with resources.as_file(
        resources.files("functional_connectivity.data").joinpath("fan.csv")
    ) as f:
        labels = pd.read_csv(f)
    return labels


def get_fan_atlas(resolution: Literal[2, 3] = 2) -> Labels:
    """Return file from ToPS model (https://doi.org/10.1038/s41591-020-1142-7)"""
    with resources.as_file(
        resources.files("functional_connectivity.data").joinpath(
            f"Fan_et_al_atlas_r279_MNI_{resolution}mm.nii.gz"
        )
    ) as f:
        atlas = f
    labels = get_fan_atlas_lut()
    return Labels(labels_img=atlas, labels=labels)


def get_coordinates_power2011() -> pd.DataFrame:
    """Return dataframe volumetric atlas from Power et al. 2011 (https://doi.org/10.1016/j.neuron.2011.09.006)

    Returns:
        dataframe of coordinates
    """
    with resources.as_file(
        resources.files("functional_connectivity.data").joinpath(
            "power2011.csv"
        )
    ) as f:
        coordinates = pd.read_csv(f)

    return coordinates


def get_difumo_lut(dimension: Literal[64, 128, 256, 512, 1024]) -> pd.DataFrame:
    with resources.as_file(
        resources.files(
            f"functional_connectivity.data.difumo_atlases.{dimension}"
        ).joinpath(f"labels_{dimension}_dictionary.csv")
    ) as f:
        labels = pd.read_csv(f)

    return labels


def get_difumo(
    dimension: Literal[64, 128, 256, 512, 1024], resolution_mm: Literal[2, 3]
) -> Labels:
    with resources.as_file(
        resources.files(
            f"functional_connectivity.data.difumo_atlases.{dimension}.{resolution_mm}mm"
        ).joinpath("maps.nii.gz")
    ) as f:
        maps = f
    labels = get_difumo_lut(dimension=dimension).rename(
        columns={"Component": "region"}
    )
    return Labels(labels_img=maps, labels=labels)


def get_atlas_schaefer_2018_lut(
    n_rois: Literal[400],
    yeo_networks: Literal[7, 17],
) -> pd.DataFrame:
    with resources.as_file(
        resources.files("functional_connectivity.data.schaefer_2018").joinpath(
            f"Schaefer2018_{n_rois}Parcels_{yeo_networks}Networks_order.txt"
        )
    ) as f:
        labels = pd.read_csv(
            f,
            delim_whitespace=True,
            names=["region", "label", "r", "g", "b", "a"],
        )
    return labels


def get_atlas_schaefer_2018(
    n_rois: Literal[400],
    resolution_mm: Literal[2],
    yeo_networks: Literal[7, 17],
) -> Labels:
    with resources.as_file(
        resources.files("functional_connectivity.data.schaefer_2018").joinpath(
            f"Schaefer2018_{n_rois}Parcels_{yeo_networks}Networks_order_FSLMNI152_{resolution_mm}mm.nii.gz"
        ),
    ) as f:
        maps = f
    labels = get_atlas_schaefer_2018_lut(
        n_rois=n_rois, yeo_networks=yeo_networks
    )

    return Labels(labels_img=maps, labels=labels)


def get_baliki_lut() -> pd.DataFrame:
    with resources.as_file(
        resources.files("functional_connectivity.data").joinpath("baliki.csv")
    ) as f:
        coordinates = pd.read_csv(f)
    return coordinates


def get_gordon_2016_lut() -> pd.DataFrame:
    with resources.as_file(
        resources.files("functional_connectivity.data.gordon_2016").joinpath(
            "Parcels.tsv"
        )
    ) as f:
        lut = pd.read_csv(f, delim_whitespace=True)

    return lut


def get_atlas_gordon_2016(
    resolution_mm: Literal[1, 2, 3], space: Literal["MNI", "711-2b"] = "MNI"
) -> Labels:
    with resources.as_file(
        resources.files("functional_connectivity.data.gordon_2016").joinpath(
            f"Parcels_{space}_{resolution_mm}{resolution_mm}{resolution_mm}.nii.gz"
        )
    ) as f:
        labels_img = f
    labels = get_gordon_2016_lut()

    return Labels(labels_img=labels_img, labels=labels)
