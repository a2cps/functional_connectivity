from pathlib import Path

import click
from distributed.deploy import subprocess

from functional_connectivity import task_runners
from functional_connectivity.flows.connectivity import connectivity_flow


def _main(
    fmriprep_subdirs: list[Path],
    output_dirs: list[Path],
    n_workers: int = 1,
    threads_per_worker: int = 1,
) -> None:
    connectivity_flow.with_options(
        task_runner=task_runners.DaskTaskRunner(
            cluster_class=subprocess.SubprocessCluster,
            cluster_kwargs={
                "n_workers": n_workers,
                "threads_per_worker": threads_per_worker,
                "dashboard_address": None,
            },
        )
    )(subdirs=fmriprep_subdirs, outdirs=output_dirs, return_state=True)


@click.command(context_settings={"ignore_unknown_options": True})
@click.argument(
    "fmriprep-dir",
    type=click.Path(
        exists=True,
        file_okay=False,
        dir_okay=True,
        resolve_path=True,
        path_type=Path,
    ),
)
@click.option(
    "--output-dir",
    default="out",
    type=click.Path(
        exists=False,
        file_okay=False,
        dir_okay=True,
        resolve_path=True,
        path_type=Path,
    ),
)
@click.option("--sub-limit", type=int, default=None)
@click.option("--n-workers", type=int, default=1)
@click.option("--threads-per-worker", type=int, default=1)
def main(
    fmriprep_dir: Path,
    output_dir: Path = Path("out"),
    sub_limit: int | None = None,
    n_workers: int = 1,
    threads_per_worker: int = 1,
) -> None:
    fmriprep_subdirs = [
        x for x in fmriprep_dir.glob("sub*") if Path(x).is_dir()
    ][:sub_limit]
    _main(
        fmriprep_subdirs=fmriprep_subdirs,
        output_dirs=[output_dir],
        n_workers=n_workers,
        threads_per_worker=threads_per_worker,
    )
