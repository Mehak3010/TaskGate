"""Docker sandbox: build, run, exec, copy, teardown.

Design rules (section 11 of the brief):

* agent containers run as a non-root user, with CPU / memory / pid limits, and
  with networking off unless the task explicitly allows it;
* the host Docker socket is never mounted into a container;
* every container and image this module creates is registered for cleanup, and
  teardown runs on the normal path, on an exception and on Ctrl-C.

If the Docker daemon is not reachable the whole module degrades to a no-op via
:func:`docker_available`, and the pipeline falls back to the static audit plus
mock calibration.
"""

from __future__ import annotations

import atexit
import os
import signal
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

from ..models import Task

#: Docker's default non-root user cannot exist before the image user does, so we
#: create one in the generated runner image.
DEFAULT_AGENT_USER = "taskgate"

#: Dockerfile for a thin runtime image layered on top of the task's own image. A
#: task image is built as root and has no ``taskgate`` user, so starting a
#: container with ``user="taskgate"`` fails with "unable to find user taskgate:
#: no matching entries in passwd file" - the whole dynamic audit then errors out
#: silently. This layer creates the user and hands it a writable ``/app``.
_RUNNER_DOCKERFILE = """\
FROM {base}
USER root
RUN (command -v useradd >/dev/null 2>&1 && useradd -m -s /bin/bash {user}) \\
    || (command -v adduser >/dev/null 2>&1 && adduser -D -s /bin/bash {user}) \\
    || true
RUN mkdir -p /app /home/{user} && chown -R {user} /app /home/{user} || true
"""

#: How long a single endpoint probe may block before we move on. The Docker SDK's
#: own default is 60s, which makes a dead pipe look like a hang.
PING_TIMEOUT_S = 5.0

_ACTIVE = threading.local()


class DockerUnavailable(RuntimeError):
    """Raised when the Docker daemon cannot be reached."""


# --------------------------------------------------------------------------
# availability
# --------------------------------------------------------------------------
#: Docker Desktop on Windows serves the engine on this named pipe. The Docker
#: Python SDK's ``from_env()`` ignores ``DOCKER_CONTEXT`` and, when the ``default``
#: context points at ``\\.\pipe\docker_engine``, that pipe is dead - the SDK then
#: blocks for its full 60s timeout and TaskGate silently degrades to the static
#: audit while claiming the daemon is fine. Probing the engine's real endpoint
#: first, with a short timeout, is what makes the dynamic audit actually run.
_WINDOWS_DESKTOP_PIPE = "npipe:////./pipe/dockerDesktopLinuxEngine"
_WINDOWS_DEFAULT_PIPE = "npipe:////./pipe/docker_engine"


def _candidate_endpoints() -> list[str | None]:
    """Ordered Docker endpoints to probe; ``None`` means "let the SDK resolve it"."""
    candidates: list[str | None] = []
    host = os.environ.get("DOCKER_HOST")
    if host:
        candidates.append(host)
    if os.name == "nt":
        candidates.append(_WINDOWS_DESKTOP_PIPE)
        candidates.append(_WINDOWS_DEFAULT_PIPE)
    candidates.append(None)

    seen: set[str | None] = set()
    ordered: list[str | None] = []
    for endpoint in candidates:
        if endpoint not in seen:
            seen.add(endpoint)
            ordered.append(endpoint)
    return ordered


def _ping_fast(client, timeout_s: float = PING_TIMEOUT_S) -> bool:
    """Probe an endpoint with a short timeout, then restore the client's own.

    The timeout lives on the shared ``APIClient``, so setting it at construction
    would cap *every* call - builds and long ``exec``s included - at a few
    seconds and make them look like failures. We therefore shorten it only for
    the probe.
    """
    api = client.api
    original = api.timeout
    try:
        api.timeout = timeout_s
        return bool(client.ping())
    finally:
        api.timeout = original


def _client_or_none():
    try:
        import docker  # noqa: PLC0415 - imported lazily so the module loads without docker
    except Exception:  # noqa: BLE001
        return None
    for endpoint in _candidate_endpoints():
        try:
            if endpoint is None:
                client = docker.from_env()
            else:
                client = docker.DockerClient(base_url=endpoint)
            _ping_fast(client)
            return client
        except Exception:  # noqa: BLE001 - try the next endpoint
            continue
    return None


_DOCKER_AVAILABLE: bool | None = None


def docker_available(refresh: bool = False) -> bool:
    """Cached check for a reachable Docker daemon."""
    global _DOCKER_AVAILABLE
    if _DOCKER_AVAILABLE is None or refresh:
        _DOCKER_AVAILABLE = _client_or_none() is not None
    return _DOCKER_AVAILABLE


# --------------------------------------------------------------------------
# cleanup registry
# --------------------------------------------------------------------------
class _CleanupRegistry:
    """Tracks live containers/images so Ctrl-C never leaks a container."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._containers: set[str] = set()
        self._images: set[str] = set()
        self._installed = False

    def install(self) -> None:
        if self._installed:
            return
        self._installed = True
        atexit.register(self.teardown_all)
        try:
            signal.signal(signal.SIGINT, self._handle_sigint)
        except (ValueError, OSError):  # non-main thread or unsupported platform
            pass

    def _handle_sigint(self, signum, frame):  # noqa: ANN001
        self.teardown_all()
        raise KeyboardInterrupt

    def add_container(self, container_id: str) -> None:
        with self._lock:
            self._containers.add(container_id)

    def add_image(self, tag: str) -> None:
        with self._lock:
            self._images.add(tag)

    def forget_container(self, container_id: str) -> None:
        with self._lock:
            self._containers.discard(container_id)

    def forget_image(self, tag: str) -> None:
        with self._lock:
            self._images.discard(tag)

    def teardown_all(self) -> None:
        with self._lock:
            containers = list(self._containers)
            images = list(self._images)
            self._containers.clear()
            self._images.clear()
        client = _client_or_none()
        if client is None:
            return
        for container_id in containers:
            try:
                client.containers.get(container_id).remove(force=True, v=True)
            except Exception:  # noqa: BLE001 - cleanup must never raise
                pass
        for tag in images:
            try:
                client.images.remove(tag, force=True)
            except Exception:  # noqa: BLE001
                pass


REGISTRY = _CleanupRegistry()


# --------------------------------------------------------------------------
# results
# --------------------------------------------------------------------------
@dataclass
class RunResult:
    exit_code: int
    output: str = ""
    duration_s: float = 0.0
    timed_out: bool = False

    @property
    def ok(self) -> bool:
        return self.exit_code == 0 and not self.timed_out


@dataclass
class ContainerSpec:
    cpus: float = 2.0
    mem_limit: str = "2g"
    pids_limit: int = 256
    network_disabled: bool = True
    read_only: bool = False
    agent_user: str = DEFAULT_AGENT_USER
    #: Extra writable paths when ``read_only`` is on.
    tmpfs: dict[str, str] = field(default_factory=lambda: {"/tmp": "size=256m"})


def image_tag(task: Task) -> str:
    safe = "".join(c if c.isalnum() or c in "._-" else "-" for c in task.task_id).lower()
    return f"taskgate/{safe}:latest"


def runner_tag(task: Task) -> str:
    """Tag of the derived image that adds the non-root agent user."""
    base, _, _ = image_tag(task).partition(":")
    return f"{base}-runner:latest"


# --------------------------------------------------------------------------
# sandbox
# --------------------------------------------------------------------------
class Sandbox:
    """One task's container lifecycle.

    Usable as a context manager so callers cannot forget teardown:

        with Sandbox(task) as box:
            box.build()
            box.start()
            box.exec("bash /solution/solve.sh")
    """

    def __init__(
        self,
        task: Task,
        *,
        spec: ContainerSpec | None = None,
        client=None,
        rebuild: bool = False,
    ) -> None:
        self.task = task
        self.spec = spec or ContainerSpec()
        self.tag = image_tag(task)
        self.runner_tag = runner_tag(task)
        self.rebuild = rebuild
        self._client = client
        self._container = None
        self._built = False

    # -- client ------------------------------------------------------------
    @property
    def client(self):
        if self._client is None:
            self._client = _client_or_none()
        if self._client is None:
            raise DockerUnavailable("Docker daemon is not reachable")
        return self._client

    # -- build -------------------------------------------------------------
    def build(self, labels: dict[str, str] | None = None) -> RunResult:
        started = time.perf_counter()
        dockerfile = self.task.dockerfile_path
        if dockerfile is None:
            raise DockerUnavailable(f"{self.task.task_id}: no Dockerfile")
        if not self.rebuild:
            try:
                self.client.images.get(self.tag)
                self.client.images.get(self.runner_tag)
                self._built = True
                return RunResult(
                    0, f"images {self.tag}, {self.runner_tag} already present",
                    time.perf_counter() - started,
                )
            except Exception:  # noqa: BLE001 - not present yet
                pass
        REGISTRY.install()
        log = ""
        try:
            _image, logs = self.client.images.build(
                path=str(self.task.path),
                dockerfile=dockerfile.name,
                tag=self.tag,
                rm=True,
                labels={"taskgate.task": self.task.task_id, **(labels or {})},
            )
            log = "\n".join(entry.get("stream", "") for entry in logs if isinstance(entry, dict))
        except Exception as exc:  # noqa: BLE001 - build errors are findings, not crashes
            return RunResult(1, f"{type(exc).__name__}: {exc}", time.perf_counter() - started)
        REGISTRY.add_image(self.tag)

        # Second, thin layer: the task image has no ``taskgate`` user, so add one
        # (and make /app writable) before any container runs as that user.
        try:
            import io

            runner = _RUNNER_DOCKERFILE.format(base=self.tag, user=self.spec.agent_user)
            _image, logs = self.client.images.build(
                fileobj=io.BytesIO(runner.encode("utf-8")),
                tag=self.runner_tag,
                rm=True,
                labels={"taskgate.task": self.task.task_id, "taskgate.role": "runner"},
            )
            log += "\n".join(entry.get("stream", "") for entry in logs if isinstance(entry, dict))
        except Exception as exc:  # noqa: BLE001
            return RunResult(
                1,
                f"runner image build failed: {type(exc).__name__}: {exc}",
                time.perf_counter() - started,
            )
        REGISTRY.add_image(self.runner_tag)
        self._built = True
        return RunResult(0, log.strip() or f"built {self.runner_tag}", time.perf_counter() - started)

    def installed_packages(self) -> set[str]:
        """The package set of the built image, for the reproducibility check."""
        result = self.run_one_off(
            ["sh", "-lc", "python -m pip freeze 2>/dev/null || true"],
            image=self.tag,
        )
        return {line.strip() for line in result.output.splitlines() if line.strip()}

    # -- container ---------------------------------------------------------
    def start(self):
        REGISTRY.install()
        kwargs: dict = {
            "image": self.runner_tag,
            "command": ["sleep", "infinity"],
            "working_dir": "/app",
            "user": self.spec.agent_user,
            "mem_limit": self.spec.mem_limit,
            "nano_cpus": int(self.spec.cpus * 1_000_000_000),
            "pids_limit": self.spec.pids_limit,
            "network_disabled": self.spec.network_disabled,
            "read_only": self.spec.read_only,
            "auto_remove": False,
        }
        if self.spec.read_only and self.spec.tmpfs:
            kwargs["tmpfs"] = self.spec.tmpfs
        # Create first, register, then start: ``containers.run`` leaves an orphan
        # behind if the start step fails (e.g. an image without the agent user),
        # and an unregistered orphan is invisible to teardown. Creating then
        # starting lets us clean up on failure instead of leaking.
        container = self.client.containers.create(**kwargs)
        self._container = container
        REGISTRY.add_container(container.id)
        try:
            container.start()
        except Exception:  # noqa: BLE001 - surface the error, but never leak
            try:
                container.remove(force=True, v=True)
            except Exception:  # noqa: BLE001
                pass
            REGISTRY.forget_container(container.id)
            self._container = None
            raise
        return self._container

    @property
    def container(self):
        if self._container is None:
            raise RuntimeError("container not started")
        return self._container

    def exec(self, command: str | list[str], *, timeout_s: float | None = None, workdir: str | None = None) -> RunResult:
        argv = command if isinstance(command, list) else ["sh", "-lc", command]
        started = time.perf_counter()
        try:
            exit_code, output = self.container.exec_run(
                argv,
                workdir=workdir or "/app",
                demux=False,
            )
        except Exception as exc:  # noqa: BLE001
            return RunResult(1, f"{type(exc).__name__}: {exc}", time.perf_counter() - started)
        text = output.decode("utf-8", "replace") if isinstance(output, (bytes, bytearray)) else str(output)
        return RunResult(int(exit_code), text, time.perf_counter() - started)

    def run_one_off(
        self,
        argv: list[str],
        *,
        image: str | None = None,
        network_disabled: bool | None = None,
        timeout_s: float | None = None,
    ) -> RunResult:
        """Run a command in a throwaway container (no agent user, no limits)."""
        started = time.perf_counter()
        try:
            output = self.client.containers.run(
                image or self.tag,
                argv,
                remove=True,
                network_disabled=(
                    self.spec.network_disabled if network_disabled is None else network_disabled
                ),
                stderr=True,
            )
        except Exception as exc:  # noqa: BLE001 - docker raises on non-zero exit
            text = getattr(exc, "output", b"") or str(exc).encode()
            text = text.decode("utf-8", "replace") if isinstance(text, (bytes, bytearray)) else str(text)
            code = getattr(getattr(exc, "response", None), "status_code", 1) or 1
            return RunResult(int(code) if str(code).isdigit() else 1, text, time.perf_counter() - started)
        text = output.decode("utf-8", "replace") if isinstance(output, (bytes, bytearray)) else str(output)
        return RunResult(0, text, time.perf_counter() - started)

    def copy_in(self, host_path: Path, container_path: str) -> bool:
        """Copy a host file or directory into the running container.

        Container paths are POSIX, so they are split with :mod:`posixpath`. Using
        ``pathlib`` here is wrong on Windows: ``str(Path("/solution").parent)`` is
        ``"\\"``, and ``put_archive("\\", ...)`` targets a directory that does not
        exist inside the container - the copy then fails, the solution and tests
        are never staged, and *every* reference run "fails" (a false D001).
        """
        try:
            import io
            import posixpath
            import tarfile

            container_path = container_path.rstrip("/") or "/"
            buffer = io.BytesIO()
            with tarfile.open(fileobj=buffer, mode="w") as tar:
                tar.add(str(host_path), arcname=posixpath.basename(container_path))
            buffer.seek(0)
            parent = posixpath.dirname(container_path) or "/"
            self.container.put_archive(parent, buffer.getvalue())
            return True
        except Exception:  # noqa: BLE001
            return False

    def read_file(self, container_path: str) -> str | None:
        try:
            stream, _ = self.container.get_archive(container_path)
            import io
            import tarfile

            buffer = io.BytesIO(b"".join(stream))
            with tarfile.open(fileobj=buffer) as tar:
                member = next(iter(tar.getmembers()), None)
                if member is None:
                    return None
                extracted = tar.extractfile(member)
                return extracted.read().decode("utf-8", "replace") if extracted else None
        except Exception:  # noqa: BLE001
            return None

    # -- teardown ----------------------------------------------------------
    def remove(self, *, keep_image: bool = False) -> None:
        if self._container is not None:
            try:
                self._container.remove(force=True, v=True)
            except Exception:  # noqa: BLE001
                pass
            REGISTRY.forget_container(self._container.id)
            self._container = None
        if not keep_image and self._built:
            for tag in (self.runner_tag, self.tag):
                try:
                    self.client.images.remove(tag, force=True)
                except Exception:  # noqa: BLE001
                    pass
                REGISTRY.forget_image(tag)
            self._built = False

    def __enter__(self) -> "Sandbox":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.remove()


def cleanup_all() -> None:
    """Public hook so the CLI can force a full teardown (e.g. on Ctrl-C)."""
    REGISTRY.teardown_all()
