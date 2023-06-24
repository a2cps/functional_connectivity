import tempfile
import typing
from pathlib import Path

import ancpbids
import pandas as pd
import prefect

from postfmriprep import utils


def write_tsv(dataframe: pd.DataFrame, filename: Path | None = None) -> Path:
    if filename:
        written = filename
        dataframe.to_csv(filename, index=False, sep="\t")
    else:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".tsv") as f:
            dataframe.to_csv(f, index=False, sep="\t")
            written = Path(f.name)
    return written


@prefect.task
def write_parquet(
    dataframe: pd.DataFrame, filename: Path | None = None
) -> Path:
    dataframe.columns = dataframe.columns.astype(str)
    if filename:
        written = filename
        dataframe.to_parquet(filename, index=False)
    else:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".parquet") as f:
            dataframe.to_parquet(f, index=False)
            written = Path(f.name)
    return written


@prefect.task
@utils.cache_dataframe
def update_confounds(
    confounds: Path,
    n_non_steady_state_tr: int = 0,
    acompcor_file: Path | None = None,
    usecols: list[str] = [
        "trans_x",
        "trans_x_derivative1",
        "trans_x_power2",
        "trans_x_derivative1_power2",
        "trans_y",
        "trans_y_derivative1",
        "trans_y_power2",
        "trans_y_derivative1_power2",
        "trans_z",
        "trans_z_derivative1",
        "trans_z_power2",
        "trans_z_derivative1_power2",
        "rot_x",
        "rot_x_derivative1",
        "rot_x_power2",
        "rot_x_derivative1_power2",
        "rot_y",
        "rot_y_derivative1",
        "rot_y_power2",
        "rot_y_derivative1_power2",
        "rot_z",
        "rot_z_derivative1",
        "rot_z_power2",
        "rot_z_derivative1_power2",
    ],
    label: typing.Literal["CSF", "WM", "WM+CSF"] | None = "WM+CSF",
    extra: Path | None = None,
    standardize: bool = True,
) -> pd.DataFrame:
    confounds_df = pd.read_csv(
        confounds, delim_whitespace=True, usecols=usecols
    )
    n_tr = confounds_df.shape[0] - n_non_steady_state_tr
    components_df = confounds_df.iloc[-n_tr:, :].reset_index(drop=True)
    if extra:
        extra_cols = pd.read_parquet(extra)
        components_df = pd.concat([components_df, extra_cols], axis=1)
    if acompcor_file and label:
        acompcor = pd.read_parquet(
            acompcor_file, columns=["component", "tr", "value", "label"]
        )
        components = (
            acompcor.query("label==@label and component < 5")
            .drop("label", axis=1)
            .pivot(index="tr", columns=["component"], values="value")
        )
        out = pd.concat([components_df, components], axis=1)
    else:
        out = components_df
    out.columns = out.columns.astype(str)
    if standardize:
        out = (out - out.mean()) / out.std(ddof=1)
    return out


@prefect.task()
def _get(
    layout: ancpbids.BIDSLayout,
    filters: dict[str, str],
) -> Path:
    file = layout.get(
        return_type="filename",
        **filters,
    )
    if not len(file) == 1:
        raise ValueError(
            f"Expected that only 1 file would be retreived but saw {file=}; {filters=}"
        )
    return Path(str(file[0]))
