import contextlib
import functools
import inspect
import typing
from uuid import UUID

import distributed
from distributed.deploy import cluster
from prefect import states, task_runners
from prefect.client.schemas import objects
from prefect.context import FlowRunContext
from prefect.futures import PrefectFuture
from prefect.task_runners import TaskConcurrencyType
from prefect.utilities.collections import visit_collection
from prefect.utilities.importtools import (
    from_qualified_name,
    to_qualified_name,
)

R = typing.TypeVar("R")


class DaskTaskRunner(task_runners.BaseTaskRunner):
    """
    A parallel task_runner that submits tasks to the `dask.distributed` scheduler.
    By default a temporary `distributed.LocalCluster` is created (and
    subsequently torn down) within the `start()` contextmanager. To use a
    different cluster class (e.g.
    [`dask_kubernetes.KubeCluster`](https://kubernetes.dask.org/)), you can
    specify `cluster_class`/`cluster_kwargs`.

    Alternatively, if you already have a dask cluster running, you can provide
    the cluster object via the `cluster` kwarg or the address of the scheduler
    via the `address` kwarg.
    !!! warning "Multiprocessing safety"
        Note that, because the `DaskTaskRunner` uses multiprocessing, calls to flows
        in scripts must be guarded with `if __name__ == "__main__":` or warnings will
        be displayed.

    Args:
        cluster (distributed.deploy.Cluster, optional): Currently running dask cluster;
            if one is not provider (or specified via `address` kwarg), a temporary
            cluster will be created in `DaskTaskRunner.start()`. Defaults to `None`.
        address (string, optional): Address of a currently running dask
            scheduler. Defaults to `None`.
        cluster_class (string or callable, optional): The cluster class to use
            when creating a temporary dask cluster. Can be either the full
            class name (e.g. `"distributed.LocalCluster"`), or the class itself.
        cluster_kwargs (dict, optional): Additional kwargs to pass to the
            `cluster_class` when creating a temporary dask cluster.
        adapt_kwargs (dict, optional): Additional kwargs to pass to `cluster.adapt`
            when creating a temporary dask cluster. Note that adaptive scaling
            is only enabled if `adapt_kwargs` are provided.
        client_kwargs (dict, optional): Additional kwargs to use when creating a
            [`dask.distributed.Client`](https://distributed.dask.org/en/latest/api.html#client).

    Examples:
        Using a temporary local dask cluster:
        ```python
        from prefect import flow
        from prefect_dask.task_runners import DaskTaskRunner

        @flow(task_runner=DaskTaskRunner)
        def my_flow():
            ...
        ```

        Using a temporary cluster running elsewhere. Any Dask cluster class should
        work, here we use [dask-cloudprovider](https://cloudprovider.dask.org):
        ```python
        DaskTaskRunner(
            cluster_class="dask_cloudprovider.FargateCluster",
            cluster_kwargs={
                "image": "prefecthq/prefect:latest",
                "n_workers": 5,
            },
        )
        ```

        Connecting to an existing dask cluster:
        ```python
        DaskTaskRunner(address="192.0.2.255:8786")
        ```
    """

    def __init__(
        self,
        cluster: cluster.Cluster | None = None,
        address: str | None = None,
        cluster_class: typing.Callable | None = None,
        cluster_kwargs: dict | None = None,
        adapt_kwargs: dict | None = None,
        client_kwargs: dict | None = None,
    ):
        # Validate settings and infer defaults
        if address:
            if cluster or cluster_class or cluster_kwargs or adapt_kwargs:
                msg = (
                    "Cannot specify `address` and "
                    "`cluster`/`cluster_class`/`cluster_kwargs`/`adapt_kwargs`"
                )
                raise ValueError(msg)
        elif cluster:
            if cluster_class or cluster_kwargs:
                msg = "Cannot specify `cluster` and `cluster_class`/`cluster_kwargs`"
                raise ValueError(msg)
            if not cluster.asynchronous:
                msg = (
                    "The cluster must have `asynchronous=True` to be "
                    "used with `DaskTaskRunner`."
                )
                raise ValueError(msg)
        elif isinstance(cluster_class, str):
            cluster_class = from_qualified_name(cluster_class)

        # Create a copies of incoming kwargs since we may mutate them
        cluster_kwargs = cluster_kwargs.copy() if cluster_kwargs else {}
        adapt_kwargs = adapt_kwargs.copy() if adapt_kwargs else {}
        client_kwargs = client_kwargs.copy() if client_kwargs else {}

        # Update kwargs defaults
        client_kwargs.setdefault("set_as_default", False)

        # The user cannot specify async/sync themselves
        if "asynchronous" in client_kwargs:
            msg = (
                "`client_kwargs` cannot set `asynchronous`. "
                "This option is managed by Prefect."
            )
            raise ValueError(msg)
        if "asynchronous" in cluster_kwargs:
            msg = (
                "`cluster_kwargs` cannot set `asynchronous`. "
                "This option is managed by Prefect."
            )
            raise ValueError(msg)

        # Store settings
        self.address = address
        self.cluster_class = cluster_class
        self.cluster_kwargs = cluster_kwargs
        self.adapt_kwargs = adapt_kwargs
        self.client_kwargs = client_kwargs

        # Runtime attributes
        self._client: distributed.Client | None = None
        self._cluster: typing.Optional["cluster.Cluster"] = cluster
        self._dask_futures: dict[UUID, distributed.Future] = {}

        super().__init__()

    @property
    def concurrency_type(self) -> TaskConcurrencyType:
        return (
            TaskConcurrencyType.PARALLEL
            if self.cluster_kwargs.get("processes")
            else TaskConcurrencyType.CONCURRENT
        )

    def duplicate(self):
        """
        Create a new instance of the task runner with the same settings.
        """
        return type(self)(
            address=self.address,
            cluster_class=self.cluster_class,
            cluster_kwargs=self.cluster_kwargs,
            adapt_kwargs=self.adapt_kwargs,
            client_kwargs=self.client_kwargs,
        )

    def __eq__(self, other: "DaskTaskRunner") -> bool:
        """
        Check if an instance has the same settings as this task runner.
        """
        if type(self) == type(other):
            return (
                self.address == other.address
                and self.cluster_class == other.cluster_class
                and self.cluster_kwargs == other.cluster_kwargs
                and self.adapt_kwargs == other.adapt_kwargs
                and self.client_kwargs == other.client_kwargs
            )
        else:
            return NotImplemented

    async def submit(
        self,
        key: UUID,
        call: functools.partial[typing.Awaitable[objects.State[R]]],
    ) -> None:
        if not self._started:
            msg = "The task runner must be started before submitting work."
            raise RuntimeError(msg)
        if not self._client:
            msg = "Dask client missing during task submission"
            raise RuntimeError(msg)

        # unpack the upstream call in order to cast Prefect futures to Dask futures
        # where possible to optimize Dask task scheduling
        call_kwargs = self._optimize_futures(call.keywords)

        if "task_run" in call_kwargs:
            if flow_context := FlowRunContext.get():
                task_run = call_kwargs["task_run"]
                # Dask displays the text up to the first '-' as the name; the task run key
                # should include the task run name for readability in the Dask console.
                # For cases where the task run fails and reruns for a retried flow run,
                # the flow run count is included so that the new key will not match
                # the failed run's key, therefore not retrieving from the Dask cache.
                if not flow_context.flow_run:
                    msg = "No run_count"
                    raise RuntimeError(msg)
                flow_run = flow_context.flow_run
                dask_key = (
                    f"{task_run.name}-{task_run.id.hex}-{flow_run.run_count}"
                )
            else:
                msg = "Should be in flow"
                raise RuntimeError(msg)
        else:
            dask_key = str(key)

        self._dask_futures[key] = self._client.submit(
            call.func,
            *call.args,
            key=dask_key,
            # Dask defaults to treating functions are pure, but we set this here for
            # explicit expectations. If this task run is submitted to Dask twice, the
            # result of the first run should be returned. Subsequent runs would return
            # `Abort` exceptions if they were submitted again.
            pure=True,
            **call_kwargs,
        )

    def _get_dask_future(self, key: UUID) -> distributed.Future:
        """
        Retrieve the dask future corresponding to a Prefect future.
        The Dask future is for the `run_fn`, which should return a `State`.
        """
        return self._dask_futures[key]

    def _optimize_futures(self, expr) -> dict[str, typing.Any]:
        def visit_fn(expr):
            if isinstance(expr, PrefectFuture):
                dask_future = self._dask_futures.get(expr.key)
                if dask_future is not None:
                    return dask_future
            # Fallback to return the expression unaltered
            return expr

        return visit_collection(expr, visit_fn=visit_fn, return_data=True)

    async def wait(
        self, key: UUID, timeout: float | None = None
    ) -> objects.State:
        try:
            future = self._get_dask_future(key)
            result = await future.result(timeout=timeout)
            # future.release()
            return result
        except distributed.TimeoutError:
            return states.Crashed(
                message="Wow! This is a totally unknown crash."
            )
        except BaseException as exc:
            return await states.exception_to_crashed_state(exc)

    async def _start(self, exit_stack: contextlib.AsyncExitStack):
        """
        Start the task runner and prep for context exit.
        - Creates a cluster if an external address is not set.
        - Creates a client to connect to the cluster.
        - Pushes a call to wait for all running futures to complete on exit.
        """

        if self._cluster:
            self.logger.info(
                f"Connecting to existing Dask cluster {self._cluster}"
            )
            connect_to = self._cluster
            if self.adapt_kwargs:
                self._cluster.adapt(**self.adapt_kwargs)
        elif self.address:
            self.logger.info(
                f"Connecting to an existing Dask cluster at {self.address}"
            )
            connect_to = self.address
        else:
            self.cluster_class = self.cluster_class or distributed.LocalCluster

            self.logger.info(
                f"Creating a new Dask cluster with "
                f"`{to_qualified_name(self.cluster_class)}`"
            )
            connect_to = self._cluster = await exit_stack.enter_async_context(
                self.cluster_class(asynchronous=True, **self.cluster_kwargs)
            )
            if self.adapt_kwargs:
                adapt_response = self._cluster.adapt(**self.adapt_kwargs)
                if inspect.isawaitable(adapt_response):
                    await adapt_response

        self._client = await exit_stack.enter_async_context(
            distributed.Client(
                connect_to, asynchronous=True, **self.client_kwargs
            )
        )

        if not self._client:
            msg = "Failed to start dask client"
            raise RuntimeError(msg)

        if self._client.dashboard_link:
            self.logger.info(
                f"The Dask dashboard is available at {self._client.dashboard_link}",
            )

    def __getstate__(self):
        """
        Allow the `DaskTaskRunner` to be serialized by dropping
        the `distributed.Client`, which contains locks.
        Must be deserialized on a dask worker.
        """
        data = self.__dict__.copy()
        data.update({k: None for k in ("_client", "_cluster")})
        return data

    def __setstate__(self, data: dict):
        """
        Restore the `distributed.Client` by loading the client on a dask worker.
        """
        self.__dict__.update(data)
        self._client = distributed.get_client()
