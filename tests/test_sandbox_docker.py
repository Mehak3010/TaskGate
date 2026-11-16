"""Docker sandbox lifecycle tests (Phase 2 regression guards).

Two bugs found while verifying the Docker path:

* the sandbox built a task's image but never created the non-root ``taskgate``
  user, so every container start failed with "unable to find user taskgate: no
  matching entries in passwd file" and the dynamic audit silently produced
  nothing. The fix layers a thin runner image that adds the user.
* ``containers.run()`` leaves an orphan container behind when the start step
  fails, and the orphan was never registered for teardown. ``start()`` now
  creates, registers and only then starts.

These tests need a reachable daemon, so they are marked ``docker`` and skipped
otherwise.
"""

from __future__ import annotations

import pytest

from taskgate.runner.sandbox import (
    DEFAULT_AGENT_USER,
    ContainerSpec,
    Sandbox,
    _client_or_none,
    image_tag,
    runner_tag,
)

_SIMPLE_DOCKERFILE = (
    'FROM python:3.11-slim\nWORKDIR /app\nCMD ["sleep", "infinity"]\n'
)


def _requires_docker() -> bool:
    return _client_or_none() is not None


docker_only = pytest.mark.skipif(
    not _requires_docker(),
    reason="needs a reachable Docker daemon (mark with @pytest.mark.docker)",
)


def test_runner_tag_derives_from_image_tag(make_task):
    """The runner tag must be a distinct, stable layer on top of the base tag."""
    task = make_task(task_yaml="id: t07_network_dep\n", task_id="t07_network_dep")
    assert image_tag(task) == "taskgate/t07_network_dep:latest"
    assert runner_tag(task) == "taskgate/t07_network_dep-runner:latest"


@pytest.mark.docker
@docker_only
def test_runner_image_provides_the_non_root_agent_user(make_task):
    """Regression: the container must actually start as the agent user."""
    task = make_task(files={"Dockerfile": _SIMPLE_DOCKERFILE}, task_id="t_user_probe")
    box = Sandbox(task, rebuild=True)
    try:
        assert box.build().ok
        box.start()
        result = box.exec("id -un")
        assert result.exit_code == 0, result.output
        assert result.output.strip() == DEFAULT_AGENT_USER
    finally:
        box.remove()


@pytest.mark.docker
@docker_only
def test_agent_user_is_not_root(make_task):
    """The dynamic audit must never run as root (brief section 11)."""
    task = make_task(files={"Dockerfile": _SIMPLE_DOCKERFILE}, task_id="t_root_probe")
    box = Sandbox(task, rebuild=True)
    try:
        assert box.build().ok
        box.start()
        uid = box.exec("id -u")
        assert uid.exit_code == 0
        assert uid.output.strip() != "0"
    finally:
        box.remove()


@pytest.mark.docker
@docker_only
def test_failed_start_does_not_leak_a_container(make_task):
    """Regression: a start failure must not leave an unregistered container."""
    task = make_task(files={"Dockerfile": _SIMPLE_DOCKERFILE}, task_id="t_leak_probe")
    box = Sandbox(task, rebuild=True, spec=ContainerSpec(agent_user=DEFAULT_AGENT_USER))
    try:
        assert box.build().ok
        # Point at the userless base image so start() must fail.
        box.runner_tag = box.tag
        client = box.client
        before = len(client.containers.list(all=True))
        with pytest.raises(Exception):
            box.start()
        after = len(client.containers.list(all=True))
        assert after == before
    finally:
        box.remove()
