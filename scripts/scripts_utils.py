"""Utilities for scripts."""

__copyright__ = """
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""

from enum import Enum
import fcntl
import hashlib
import os
from pathlib import Path
import platform
import shutil
import time
from typing import Optional
import urllib.request

_URL = "https://github.com/bazelbuild/buildtools/releases/download/5.1.0/"

"""Version SHAs.

Gather shas with:
    for f in buildozer buildifier; do
        echo \"$f\": {
        for v in darwin-amd64 darwin-arm64 linux-amd64 linux-arm64 \
            windows-amd64.exe
        do
            echo "\"$v\": \"$(wget -q -O - https://github.com/bazelbuild/buildtools/releases/download/5.1.0/$f-$v | sha256sum | cut -d ' ' -f1)\", # noqa: E501"
        done
        echo },
    done
"""
_VERSION_SHAS = {
    "buildozer": {
        "darwin-amd64": "294f4d0790f4dba18c9b7617f57563e07c2c7e529a8915bcbc49170dc3c08eb9",  # noqa: E501
        "darwin-arm64": "57f8d90fac6b111bd0859b97847d3db2ce71419f44588b0e91250892037cf638",  # noqa: E501
        "linux-amd64": "7346ce1396dfa9344a5183c8e3e6329f067699d71c4391bd28317391228666bf",  # noqa: E501
        "linux-arm64": "0b08e384709ec4d4f5320bf31510d2cefe8f9e425a6565b31db06b2398ff9dc4",  # noqa: E501
        "windows-amd64.exe": "d62bc159729fad9500fd20c7375e0ab53695376f1e358737af74bc1f03fb196b",  # noqa: E501
    },
    "buildifier": {
        "darwin-amd64": "c9378d9f4293fc38ec54a08fbc74e7a9d28914dae6891334401e59f38f6e65dc",  # noqa: E501
        "darwin-arm64": "745feb5ea96cb6ff39a76b2821c57591fd70b528325562486d47b5d08900e2e4",  # noqa: E501
        "linux-amd64": "52bf6b102cb4f88464e197caac06d69793fa2b05f5ad50a7e7bf6fbd656648a3",  # noqa: E501
        "linux-arm64": "917d599dbb040e63ae7a7e1adb710d2057811902fdc9e35cce925ebfd966eeb8",  # noqa: E501
        "windows-amd64.exe": "2f039125e2fbef4c804e43dc11c71866cf444306ac6d0f5e38c592854458f425",  # noqa: E501
    },
}


class Release(Enum):
    BUILDOZER = "buildozer"
    BUILDIFIER = "buildifier"


def chdir_repo_root() -> None:
    """Change the working directory to the repository root.

    This is done so that scripts run from a consistent directory.
    """
    os.chdir(Path(__file__).parent.parent)


def _get_hash(file: Path) -> str:
    """Returns the sha256 of a file."""
    digest = hashlib.sha256()
    with file.open("rb") as f:
        while True:
            chunk = f.read(1024 * 64)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _download(url: str, local_path: Path) -> Optional[int]:
    """Downloads the URL to the path. Returns an HTTP error code on failure."""
    with urllib.request.urlopen(url) as response:
        if response.code != 200:
            return int(response.code)
        with local_path.open("wb") as f:
            shutil.copyfileobj(response, f)
    return None


def get_release(release: Release) -> str:
    """Install a file to carbon-lang's cache.

    release: The release to cache.
    """
    cache_dir = Path.home().joinpath(".cache", "carbon-lang-scripts")
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Translate platform information into Bazel's release form.
    machine = platform.machine()
    if machine == "x86_64":
        machine = "amd64"
    version = f"{platform.system().lower()}-{machine}"

    # Get ready to add .exe for Windows.
    ext = ""
    if platform.system() == "Windows":
        ext = ".exe"

    # Ensure the platform is supported, and grab its hash.
    if version not in _VERSION_SHAS[release.value]:
        # If this because a platform support issue, we may need to print errors.
        exit(f"No {release.value} release available for platform: {version}")
    want_hash = _VERSION_SHAS[release.value][version]

    # Hold a lock while checksumming and downloading the path. Otherwise,
    # parallel runs by pre-commit may conflict with one another with
    # simultaneous downloads.
    with open(cache_dir.joinpath(f"{release.value}.lock"), "w") as lock_file:
        fcntl.lockf(lock_file.fileno(), fcntl.LOCK_EX)

        # Check if there's a cached file that can be used.
        local_path = cache_dir.joinpath(f"{release.value}{ext}")
        if local_path.is_file() and want_hash == _get_hash(local_path):
            return str(local_path)

        # Download the file.
        url = f"{_URL}/{release.value}-{version}{ext}"
        retries = 5
        while True:
            err = _download(url, local_path)
            if err is None:
                break
            retries -= 1
            if retries == 0:
                exit(
                    f"Failed to download {release.value}-{version}: HTTP {err}."
                )
            time.sleep(1)
        local_path.chmod(0o755)

        # Verify the downloaded hash.
        found_hash = _get_hash(local_path)
        if want_hash != found_hash:
            exit(
                f"Downloaded {release.value}-{version} but found sha256 "
                f"{found_hash} ({local_path.stat().st_size} bytes), wanted "
                f"{want_hash}"
            )

    return str(local_path)


def locate_bazel() -> str:
    """Returns the bazel command.

    We use the `BAZEL` environment variable if present. If not, then we try to
    use `bazelisk` and then `bazel`.
    """
    bazel = os.environ.get("BAZEL")
    if bazel:
        return bazel

    for cmd in ("bazelisk", "bazel"):
        target = shutil.which(cmd)
        if target:
            return target

    exit("Unable to run Bazel")
