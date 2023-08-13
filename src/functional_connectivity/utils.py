import logging
import tempfile
from pathlib import Path
from typing import Callable, Concatenate, ParamSpec, Sequence, TypeVar

import nibabel as nb
import numpy as np
import pandas as pd


def img_stem(img: Path) -> str:
    return img.name.removesuffix(".gz").removesuffix(".nii")


def exclude_to_index(n_non_steady_state_tr: int, n_tr: int) -> np.ndarray:
    return np.array(list(range(n_non_steady_state_tr, n_tr)))


def get_tr(nii: nb.nifti1.Nifti1Image) -> float:
    return nii.header.get("pixdim")[4]  # type: ignore


P = ParamSpec("P")
R = TypeVar("R")


def cache_nii(
    f: Callable[P, nb.nifti1.Nifti1Image]
) -> Callable[Concatenate[Path, P], Path]:
    def wrapper(_filename: Path, *args: P.args, **kwargs: P.kwargs) -> Path:
        if _filename.exists():
            logging.info(f"found cached {_filename}")
        else:
            out = f(*args, **kwargs)
            parent = _filename.parent
            if not parent.exists():
                parent.mkdir(parents=True)
            out.to_filename(_filename)
        return _filename

    # otherwise logging won't name of wrapped function
    # NOTE: unsure why @functools.wraps(f) doesn't work.
    # ends up complaining about the signature
    for attr in ("__name__", "__qualname__"):
        try:
            value = getattr(f, attr)
        except AttributeError:
            pass
        else:
            setattr(wrapper, attr, value)

    return wrapper


def cache_dataframe(
    f: Callable[P, pd.DataFrame]
) -> Callable[Concatenate[Path | None, P], Path]:
    def wrapper(
        _filename: Path | None, *args: P.args, **kwargs: P.kwargs
    ) -> Path:
        if _filename and _filename.exists():
            logging.warning(f"found cached {_filename}")
            outfile = _filename
        else:
            out = f(*args, **kwargs)
            if _filename:
                parent = _filename.parent
                if not parent.exists():
                    parent.mkdir(parents=True)
                outfile = _filename
            else:
                outfile = Path(tempfile.mkstemp(suffix=".parquet")[1])
            if isinstance(out, pd.DataFrame):
                out.columns = out.columns.astype(str)
                out.to_parquet(path=outfile, write_statistics=True)
            else:
                out.write_parquet(outfile, statistics=True)
        return outfile

    # otherwise logging won't name of wrapped function
    # NOTE: unsure why @functools.wraps(f) doesn't work.
    # ends up complaining about the signature
    for attr in ("__name__", "__qualname__"):
        try:
            value = getattr(f, attr)
        except AttributeError:
            pass
        else:
            setattr(wrapper, attr, value)

    return wrapper


def _mat_to_df(cormat: np.ndarray, labels: Sequence[int]) -> pd.DataFrame:
    source = []
    target = []
    connectivity = []
    for xi, x in enumerate(labels):
        for yi, y in enumerate(labels):
            if yi <= xi:
                continue
            else:
                source.append(x)
                target.append(y)
                connectivity.append(cormat[xi, yi])

    return pd.DataFrame.from_dict(
        {"source": source, "target": target, "connectivity": connectivity}
    )


def get_poly_design(n: int, degree: int) -> np.ndarray:
    x = np.arange(n)
    x = x - np.mean(x, axis=0)
    X = np.vander(x, degree, increasing=True)  # noqa: N806
    q, r = np.linalg.qr(X)

    z = np.diag(np.diag(r))
    raw = np.dot(q, z)

    norm2 = np.sum(raw**2, axis=0)
    Z = raw / np.sqrt(norm2)  # noqa: N806
    return Z


def detrend(img: nb.nifti1.Nifti1Image, mask: Path) -> nb.nifti1.Nifti1Image:
    from nilearn import masking

    Y = masking.apply_mask(img, mask_img=mask)  # noqa: N806

    resid = _detrend(Y=Y)
    # Put results back into Niimg-like object
    return masking.unmask(resid, mask)  # type: ignore


def _detrend(Y: np.ndarray) -> np.ndarray:  # noqa: N803
    X = get_poly_design(Y.shape[0], degree=3)  # noqa: N806
    beta = np.linalg.pinv(X).dot(Y)
    return Y - np.dot(X[:, 1:], beta[1:, :])
