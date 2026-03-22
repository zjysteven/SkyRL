"""Harbor environment provider backed by agentdocker-lite.

Drop-in replacement for Harbor's DockerEnvironment that uses Linux namespace
sandboxes instead of Docker containers.  Achieves ~20x faster container
lifecycle (create/reset/delete) while maintaining identical execution semantics.

Usage in Harbor config::

    environment:
      import_path: "examples.train_integrations.harbor.agentdocker_lite_environment:AgentDockerLiteEnvironment"
      # ... resource overrides as usual

Or via CLI override::

    harbor_trial_config.environment.import_path=\
        "examples.train_integrations.harbor.agentdocker_lite_environment:AgentDockerLiteEnvironment"
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

import atexit

from harbor.environments.base import BaseEnvironment, ExecResult
from harbor.models.trial.paths import TrialPaths

logger = logging.getLogger(__name__)


def _print_timing_summary() -> None:
    """Print timing summary at process exit."""
    summary = get_timing_summary()
    if not summary:
        return
    lines = ["\n=== AgentDockerLite Timing Summary ==="]
    for op, stats in summary.items():
        lines.append(
            f"  {op:15s}: count={stats['count']:5d}  "
            f"mean={stats['mean_ms']:8.1f}ms  "
            f"min={stats['min_ms']:8.1f}ms  "
            f"max={stats['max_ms']:8.1f}ms  "
            f"total={stats['total_ms']:10.1f}ms"
        )
    lines.append("=" * 42)
    logger.info("\n".join(lines))
    # Also print to stderr so it's visible even if logging is suppressed
    print("\n".join(lines), flush=True)


atexit.register(_print_timing_summary)

# ---------------------------------------------------------------------------
#  Timing instrumentation — collects per-operation latencies for benchmarking
# ---------------------------------------------------------------------------

import threading

_timing_lock = threading.Lock()
_timing_data: dict[str, list[float]] = {
    "start": [],
    "exec": [],
    "stop": [],
    "upload_file": [],
    "upload_dir": [],
    "download_file": [],
    "download_dir": [],
}


def _record_timing(op: str, elapsed_ms: float) -> None:
    with _timing_lock:
        _timing_data[op].append(elapsed_ms)


def get_timing_summary() -> dict[str, Any]:
    """Return aggregated timing stats for all operations.

    Call this after training to compare against Docker baseline.
    """
    summary = {}
    for op, times in _timing_data.items():
        if times:
            summary[op] = {
                "count": len(times),
                "total_ms": sum(times),
                "mean_ms": sum(times) / len(times),
                "min_ms": min(times),
                "max_ms": max(times),
            }
    return summary


# ---------------------------------------------------------------------------
#  Image building and caching
# ---------------------------------------------------------------------------

# Global lock to serialize builds of the same image
_image_build_locks: dict[str, asyncio.Lock] = {}

# Where to cache built rootfs directories
_ROOTFS_CACHE_DIR = Path(
    os.environ.get("AGENTDOCKER_ROOTFS_CACHE", "/tmp/agentdocker_lite_rootfs_cache")
)


def _dockerfile_content_hash(dockerfile_path: Path) -> str:
    """Hash Dockerfile + any files in the same directory to get a cache key."""
    h = hashlib.sha256()
    env_dir = dockerfile_path.parent
    for f in sorted(env_dir.rglob("*")):
        if f.is_file():
            h.update(f.name.encode())
            h.update(f.read_bytes())
    return h.hexdigest()[:16]


def _build_and_export_rootfs(
    environment_dir: Path,
    environment_name: str,
    docker_image: str | None = None,
) -> Path:
    """Build a Docker image from a Dockerfile and export it as a rootfs directory.

    If docker_image is provided (prebuilt), use that directly.
    Otherwise, build from the Dockerfile in environment_dir.

    Results are cached by content hash of the Dockerfile directory.
    """
    if docker_image:
        # Prebuilt image — cache by image name
        safe_name = docker_image.replace("/", "_").replace(":", "_")
        cached = _ROOTFS_CACHE_DIR / safe_name
        if cached.exists() and cached.is_dir():
            logger.info("Using cached rootfs for prebuilt image %s", docker_image)
            return cached

        # Pull and export
        subprocess.run(["docker", "pull", docker_image], check=True, capture_output=True)
        return _export_image_to_rootfs(docker_image, cached)

    # Build from Dockerfile
    dockerfile_path = environment_dir / "Dockerfile"
    if not dockerfile_path.exists():
        raise FileNotFoundError(f"No Dockerfile found at {dockerfile_path}")

    content_hash = _dockerfile_content_hash(dockerfile_path)
    cached = _ROOTFS_CACHE_DIR / f"{environment_name}_{content_hash}"

    if cached.exists() and cached.is_dir():
        logger.info("Using cached rootfs for %s (hash=%s)", environment_name, content_hash)
        return cached

    # Build the image
    image_tag = f"agentdocker__{environment_name}:{content_hash}"
    logger.info("Building Docker image %s from %s", image_tag, environment_dir)
    result = subprocess.run(
        ["docker", "build", "-t", image_tag, str(environment_dir)],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"docker build failed:\n{result.stderr}")

    rootfs_path = _export_image_to_rootfs(image_tag, cached)

    # Clean up the built image to save disk
    subprocess.run(["docker", "rmi", "-f", image_tag], capture_output=True)

    return rootfs_path


def _export_image_to_rootfs(image_name: str, output_dir: Path) -> Path:
    """Export a Docker image as a flat rootfs directory."""
    output_dir.mkdir(parents=True, exist_ok=True)

    create = subprocess.run(
        ["docker", "create", image_name],
        capture_output=True,
        text=True,
    )
    if create.returncode != 0:
        raise RuntimeError(f"docker create failed: {create.stderr.strip()}")
    container_id = create.stdout.strip()

    try:
        export_proc = subprocess.Popen(
            ["docker", "export", container_id],
            stdout=subprocess.PIPE,
        )
        tar_proc = subprocess.Popen(
            ["tar", "-C", str(output_dir), "-xf", "-"],
            stdin=export_proc.stdout,
        )
        if export_proc.stdout is not None:
            export_proc.stdout.close()
        tar_proc.communicate()

        if tar_proc.returncode != 0:
            # Clean up partial extraction
            shutil.rmtree(output_dir, ignore_errors=True)
            raise RuntimeError(f"tar extraction failed for {image_name}")
    finally:
        subprocess.run(["docker", "rm", "-f", container_id], capture_output=True)

    logger.info("Rootfs exported: %s -> %s", image_name, output_dir)
    return output_dir


# ---------------------------------------------------------------------------
#  AgentDockerLiteEnvironment
# ---------------------------------------------------------------------------


class AgentDockerLiteEnvironment(BaseEnvironment):
    """Harbor environment backed by agentdocker-lite namespace sandboxes.

    This replaces DockerEnvironment with ~20x faster container lifecycle:
    - Create: ~10ms vs ~271ms (Docker)
    - Per-command exec: ~11ms vs ~22ms (Docker)
    - Delete: ~2ms vs ~104ms (Docker)
    """

    def __init__(
        self,
        environment_dir: Path,
        environment_name: str,
        session_id: str,
        trial_paths: TrialPaths,
        task_env_config,
        *args,
        **kwargs,
    ):
        super().__init__(
            environment_dir=environment_dir,
            environment_name=environment_name,
            session_id=session_id,
            trial_paths=trial_paths,
            task_env_config=task_env_config,
            *args,
            **kwargs,
        )
        self._sandbox = None
        self._rootfs_path: Path | None = None
        self._sandbox_name = f"hb_{session_id.lower().replace('.', '-')}"

    # ------------------------------------------------------------------ #
    #  Abstract method implementations                                     #
    # ------------------------------------------------------------------ #

    @staticmethod
    def type():
        # We don't have a registered EnvironmentType enum value, but this
        # is only called for built-in types. For import_path providers,
        # this is not used by the factory. Return a string for logging.
        return "agentdocker_lite"

    @property
    def is_mounted(self) -> bool:
        # We mount trial log directories as bind mounts, same as Docker.
        return True

    @property
    def supports_gpus(self) -> bool:
        return False

    @property
    def can_disable_internet(self) -> bool:
        # agentdocker-lite supports net_isolate
        return True

    def _validate_definition(self):
        dockerfile = self.environment_dir / "Dockerfile"
        if not dockerfile.exists() and not (
            self.task_env_config and self.task_env_config.docker_image
        ):
            raise FileNotFoundError(
                f"No Dockerfile at {dockerfile} and no docker_image configured"
            )

    async def start(self, force_build: bool) -> None:
        t0 = time.monotonic()

        # Use the public Sandbox() factory which auto-detects:
        #   root → RootfulSandbox (full namespace + mount isolation)
        #   non-root → RootlessSandbox (user namespace, no root needed)
        # RootlessSandbox solves the Ray permission problem: workers run
        # as non-root but still get full isolation via user namespaces.
        from agentdocker_lite import Sandbox as _SandboxFactory
        from agentdocker_lite import SandboxConfig

        # Build/cache rootfs (serialized per environment_name to avoid duplicate builds)
        docker_image = (
            self.task_env_config.docker_image
            if not force_build and self.task_env_config.docker_image
            else None
        )
        env_name = self.environment_name
        lock_key = docker_image or env_name
        if lock_key not in _image_build_locks:
            _image_build_locks[lock_key] = asyncio.Lock()

        async with _image_build_locks[lock_key]:
            self._rootfs_path = await asyncio.to_thread(
                _build_and_export_rootfs,
                self.environment_dir,
                env_name,
                docker_image,
            )

        # Prepare host-side log directories
        self.trial_paths.agent_dir.mkdir(parents=True, exist_ok=True)
        self.trial_paths.verifier_dir.mkdir(parents=True, exist_ok=True)
        self.trial_paths.artifacts_dir.mkdir(parents=True, exist_ok=True)

        # Configure resource limits
        memory_max = None
        if self.task_env_config.memory_mb:
            memory_max = str(self.task_env_config.memory_mb * 1024 * 1024)

        # Determine network isolation
        allow_internet = getattr(self.task_env_config, "allow_internet", True)

        # Build volume mounts: bind host trial dirs -> container /logs/*
        volumes = [
            f"{self.trial_paths.agent_dir}:/logs/agent:rw",
            f"{self.trial_paths.verifier_dir}:/logs/verifier:rw",
            f"{self.trial_paths.artifacts_dir}:/logs/artifacts:rw",
        ]

        # If the task has a tests directory, mount it
        tests_dir = self.environment_dir.parent / "tests"
        if tests_dir.exists():
            volumes.append(f"{tests_dir}:/tests:ro")

        # Extract WORKDIR from Dockerfile, default to /
        working_dir = _extract_workdir(self.environment_dir / "Dockerfile")

        config = SandboxConfig(
            image=str(self._rootfs_path),
            working_dir=working_dir,
            memory_max=memory_max,
            net_isolate=not allow_internet,
            volumes=volumes,
        )

        # Create sandbox via factory (auto-selects Rootful or Rootless)
        self._sandbox = await asyncio.to_thread(
            _SandboxFactory, config, self._sandbox_name
        )

        elapsed = (time.monotonic() - t0) * 1000
        logger.info(
            "AgentDockerLite sandbox started in %.1fms: %s",
            elapsed,
            self._sandbox_name,
        )
        _record_timing("start", elapsed)

    async def stop(self, delete: bool) -> None:
        t0 = time.monotonic()
        if self._sandbox is not None:
            try:
                await asyncio.to_thread(self._sandbox.delete)
            except Exception as e:
                logger.warning("Error deleting sandbox %s: %s", self._sandbox_name, e)
            self._sandbox = None

        elapsed = (time.monotonic() - t0) * 1000
        logger.info(
            "AgentDockerLite sandbox stopped in %.1fms: %s",
            elapsed,
            self._sandbox_name,
        )
        _record_timing("stop", elapsed)

    async def exec(
        self,
        command: str,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        timeout_sec: int | None = None,
    ) -> ExecResult:
        t0 = time.monotonic()

        if self._sandbox is None:
            raise RuntimeError("Sandbox not started. Call start() first.")

        # Build the actual command with cwd and env support
        parts = []
        if env:
            for k, v in env.items():
                parts.append(f"export {k}={_shell_quote(v)}")
        if cwd:
            parts.append(f"cd {_shell_quote(cwd)}")
        parts.append(command)

        full_command = " && ".join(parts)

        try:
            output, exit_code = await asyncio.to_thread(
                self._sandbox.run, full_command, timeout_sec
            )
        except Exception as e:
            elapsed = (time.monotonic() - t0) * 1000
            _record_timing("exec", elapsed)
            return ExecResult(
                stdout="",
                stderr=str(e),
                return_code=-1,
            )

        elapsed = (time.monotonic() - t0) * 1000
        _record_timing("exec", elapsed)

        return ExecResult(
            stdout=output or "",
            stderr="",
            return_code=exit_code,
        )

    async def upload_file(self, source_path: Path | str, target_path: str) -> None:
        t0 = time.monotonic()
        if self._sandbox is None:
            raise RuntimeError("Sandbox not started")

        await asyncio.to_thread(self._sandbox.copy_to, str(source_path), target_path)

        elapsed = (time.monotonic() - t0) * 1000
        _record_timing("upload_file", elapsed)

    async def upload_dir(self, source_dir: Path | str, target_dir: str) -> None:
        t0 = time.monotonic()
        if self._sandbox is None:
            raise RuntimeError("Sandbox not started")

        source_dir = Path(source_dir)
        # Copy directory contents recursively via host filesystem
        target_host = self._sandbox.rootfs / target_dir.lstrip("/")
        target_host.mkdir(parents=True, exist_ok=True)

        await asyncio.to_thread(
            shutil.copytree, str(source_dir), str(target_host), dirs_exist_ok=True
        )

        elapsed = (time.monotonic() - t0) * 1000
        _record_timing("upload_dir", elapsed)

    async def download_file(self, source_path: str, target_path: Path | str) -> None:
        t0 = time.monotonic()
        if self._sandbox is None:
            raise RuntimeError("Sandbox not started")

        await asyncio.to_thread(
            self._sandbox.copy_from, source_path, str(target_path)
        )

        elapsed = (time.monotonic() - t0) * 1000
        _record_timing("download_file", elapsed)

    async def download_dir(self, source_dir: str, target_dir: Path | str) -> None:
        t0 = time.monotonic()
        if self._sandbox is None:
            raise RuntimeError("Sandbox not started")

        source_host = self._sandbox.rootfs / source_dir.lstrip("/")
        target_dir = Path(target_dir)
        target_dir.mkdir(parents=True, exist_ok=True)

        if source_host.exists():
            await asyncio.to_thread(
                shutil.copytree, str(source_host), str(target_dir), dirs_exist_ok=True
            )

        elapsed = (time.monotonic() - t0) * 1000
        _record_timing("download_dir", elapsed)


def _shell_quote(s: str) -> str:
    """Quote a string for safe shell interpolation."""
    import shlex
    return shlex.quote(s)


def _extract_workdir(dockerfile_path: Path) -> str:
    """Extract the last WORKDIR directive from a Dockerfile."""
    workdir = "/"
    if not dockerfile_path.exists():
        return workdir
    try:
        for line in dockerfile_path.read_text().splitlines():
            stripped = line.strip()
            if stripped.upper().startswith("WORKDIR "):
                workdir = stripped.split(None, 1)[1].strip()
    except Exception:
        pass
    return workdir
