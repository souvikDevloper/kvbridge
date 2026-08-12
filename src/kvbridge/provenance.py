"""Best-effort immutable code identity for experiment manifests."""

from __future__ import annotations

import os
import subprocess
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path


def code_revision(repository: str | Path | None = None) -> str:
    """Return an explicit revision or the checked-out Git commit, else ``unknown``."""
    explicit = os.environ.get("KVBRIDGE_CODE_REVISION")
    if explicit:
        return explicit
    root = Path(repository) if repository is not None else Path(__file__).parents[2]
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    return result.stdout.strip() or "unknown"


def package_versions(names: tuple[str, ...]) -> dict[str, str]:
    """Resolve installed distribution versions without importing optional packages."""
    resolved: dict[str, str] = {}
    for name in names:
        try:
            resolved[name] = version(name)
        except PackageNotFoundError:
            resolved[name] = "not-installed"
    return resolved
