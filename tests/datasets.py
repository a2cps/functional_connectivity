# simple test for loading each dataset

import pandas as pd

from functional_connectivity import datasets


def test_get_fan_atlas_lut():
    assert isinstance(datasets.get_fan_atlas_lut(), pd.DataFrame)


def test_get_fan_atlas():
    tests = []
    for r in [2, 3]:
        tests.append(
            isinstance(datasets.get_fan_atlas(resolution=r), datasets.Labels)
        )

    assert all(tests)


def test_get_gordon_2016_lut():
    assert isinstance(datasets.get_gordon_2016_lut(), pd.DataFrame)


def test_get_atlas_gordon_2016():
    tests = []
    for r in [1, 2, 3]:
        for s in ["MNI", "711-2b"]:
            tests.append(
                isinstance(
                    datasets.get_atlas_gordon_2016(resolution_mm=r, space=s),
                    datasets.Labels,
                )
            )

    assert all(tests)


def test_get_baliki_lut():
    assert isinstance(datasets.get_baliki_lut(), pd.DataFrame)


def test_get_coordinates_power2011():
    assert isinstance(datasets.get_coordinates_power2011(), pd.DataFrame)


def test_get_difumo_lut():
    tests = []
    for d in [64, 128, 256, 512, 1024]:
        tests.append(
            isinstance(datasets.get_difumo_lut(dimension=d), pd.DataFrame)
        )
    assert all(tests)


def test_get_atlas_schaefer_2018_lut():
    tests = []
    for n_rois in [400]:
        for yeo_networks in [7, 17]:
            tests.append(
                isinstance(
                    datasets.get_atlas_schaefer_2018_lut(
                        n_rois=n_rois, yeo_networks=yeo_networks
                    ),
                    pd.DataFrame,
                )
            )
    assert all(tests)


def test_get_atlas_schaefer_2018():
    tests = []
    for n_rois in [400]:
        for yeo_networks in [7, 17]:
            for r in [2]:
                tests.append(
                    isinstance(
                        datasets.get_atlas_schaefer_2018(
                            n_rois=n_rois,
                            yeo_networks=yeo_networks,
                            resolution_mm=r,
                        ),
                        datasets.Labels,
                    )
                )
    assert all(tests)
