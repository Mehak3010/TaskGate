"""Docker daemon detection tests.

Regression guard for the bug found in Phase 2: on Windows Docker Desktop serves
the engine on ``npipe:////./pipe/dockerDesktopLinuxEngine`` while the Python
SDK's ``from_env()`` resolves the ``default`` context to ``\\\\.\\pipe\\docker_engine``.
That pipe is dead, so ``from_env().ping()`` blocked for the SDK's full 60s
timeout and TaskGate silently fell back to the static audit while reporting the
daemon as unreachable. :func:`taskgate.runner.sandbox._candidate_endpoints`
probes the engine's real endpoint first, with a short timeout.
"""

from __future__ import annotations

import os

import pytest

from taskgate.runner.sandbox import (
    _WINDOWS_DEFAULT_PIPE,
    _WINDOWS_DESKTOP_PIPE,
    _candidate_endpoints,
    _client_or_none,
    docker_available,
)


def _requires_docker() -> bool:
    return _client_or_none() is not None


docker_only = pytest.mark.skipif(
    not _requires_docker(),
    reason="needs a reachable Docker daemon (mark with @pytest.mark.docker)",
)


def test_candidate_endpoints_always_end_with_sdk_resolution():
    """``None`` must be last so the SDK/context is the final fallback."""
    assert _candidate_endpoints()[-1] is None


def test_candidate_endpoints_honour_docker_host(monkeypatch):
    monkeypatch.setenv("DOCKER_HOST", "tcp://127.0.0.1:2375")
    endpoints = _candidate_endpoints()
    assert endpoints[0] == "tcp://127.0.0.1:2375"


def test_candidate_endpoints_are_deduplicated(monkeypatch):
    """A DOCKER_HOST that duplicates a known pipe must not be probed twice."""
    monkeypatch.setenv("DOCKER_HOST", _WINDOWS_DEFAULT_PIPE)
    endpoints = _candidate_endpoints()
    assert len(endpoints) == len(set(endpoints))


@pytest.mark.skipif(os.name != "nt", reason="Windows-only engine endpoint")
def test_candidate_endpoints_probe_docker_desktop_pipe_first(monkeypatch):
    """The regression: the Desktop engine endpoint is probed before the SDK's own."""
    monkeypatch.delenv("DOCKER_HOST", raising=False)
    endpoints = _candidate_endpoints()
    assert endpoints[0] == _WINDOWS_DESKTOP_PIPE
    assert _WINDOWS_DESKTOP_PIPE in endpoints
    assert _WINDOWS_DEFAULT_PIPE in endpoints


@pytest.mark.docker
@docker_only
def test_docker_available_is_true_when_daemon_reachable():
    assert docker_available(refresh=True) is True


@pytest.mark.docker
@docker_only
def test_client_or_none_returns_a_working_client():
    client = _client_or_none()
    assert client is not None
    # ping() succeeds and the server reports a version.
    assert client.ping() is True
    assert client.version().get("Version")


@pytest.mark.docker
@docker_only
def test_detected_client_keeps_a_normal_timeout():
    """Regression: the fast ping must not cripple the client's real timeout.

    An earlier fix set the 5s probe timeout on the whole client, which made
    image builds and long ``exec`` calls time out (so every reference solution
    "failed"). The short timeout must apply to the ping only.
    """
    client = _client_or_none()
    assert client is not None
    assert client.api.timeout >= 30

