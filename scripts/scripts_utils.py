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
from typing import Dict, Optional
import urllib.request

_BAZEL_TOOLS_URL = (
    "https://github.com/bazelbuild/buildtools/releases/download/v6.3.3/"
)

"""Version SHAs.

Gather shas with:
    for f in buildozer buildifier; do
        echo \"$f\": {
        for v in darwin-amd64 darwin-arm64 linux-amd64 linux-arm64 \
            windows-amd64.exe
        do
            echo "\"$v\": \"$(wget -q -O - https://github.com/bazelbuild/buildtools/releases/download/v6.3.3/$f-$v | sha256sum | cut -d ' ' -f1)\", # noqa: E501"
        done
        echo },
    done
"""
_BAZEL_TOOLS_VERSION_SHAS = {
    "buildozer": {
        "darwin-amd64": "9b0bbecb3745250e5ad5a9c36da456699cb55e52999451c3c74047d2b1f0085f",  # noqa: E501
        "darwin-arm64": "085928dd4deffa1a7fd38c66c4475e37326b2d4942408e8e3d993953ae4c626c",  # noqa: E501
        "linux-amd64": "1dcdc668d7c775e5bca2d43ac37e036468ca4d139a78fe48ae207d41411c5100",  # noqa: E501
        "linux-arm64": "94b96d6a3c52d6ef416f0eb96c8a9fe7f6a0757f0458cc8cf190dfc4a5c2d8e7",  # noqa: E501
        "windows-amd64.exe": "fc1c4f5de391ec6d66f2119c5bd6131d572ae35e92ddffe720e42b619ab158e0",  # noqa: E501
    },
    "buildifier": {
        "darwin-amd64": "3c36a3217bd793815a907a8e5bf81c291e2d35d73c6073914640a5f42e65f73f",  # noqa: E501
        "darwin-arm64": "9bb366432d515814766afcf6f9010294c13876686fbbe585d5d6b4ff0ca3e982",  # noqa: E501
        "linux-amd64": "42f798ec532c58e34401985043e660cb19d5ae994e108d19298c7d229547ffca",  # noqa: E501
        "linux-arm64": "6a03a1cf525045cb686fc67cd5d64cface5092ebefca3c4c93fb6e97c64e07db",  # noqa: E501
        "windows-amd64.exe": "2761bebc7392d47c2862c43d85201d93efa57249ed09405fd82708867caa787b",  # noqa: E501
    },
}

_TARGET_DETERMINATOR_URL = "https://github.com/bazel-contrib/target-determinator/releases/download/v0.25.0/"  # noqa: E501

"""Version SHAs.

Gather shas with:
    for v in darwin.amd64 darwin.arm64 linux.amd64 linux.arm64 \
        windows.amd64.exe
    do
        echo "\"$v\": \"$(wget -q -O - https://github.com/bazel-contrib/target-determinator/releases/download/v0.25.0/target-determinator.$v | sha256sum | cut -d ' ' -f1)\", # noqa: E501"
    done
"""
_TARGET_DETERMINATOR_SHAS = {
    "darwin.amd64": "8c7245603dede429b978e214ca327c3f3d686a1bc712c1298fca0396a0f25f23",  # noqa: E501
    "darwin.arm64": "8f975b471c4a51d32781b757e1ece9700221bfd4c0ea507c18fa382360d1111f",  # noqa: E501
    "linux.amd64": "c8a09143e9fe6eccc4b27a6be92c5929e5a78034a8d0b4c43dbed4ee539ec903",  # noqa: E501
    "linux.arm64": "f34618c885d239d77a31f594daf73a67c1133ab4a0376d37a29dbe8d1d2b0b90",  # noqa: E501
    "windows.amd64.exe": "e14fd75e33d193f579505cf3e641e07025904fc027686e13e154ba8e10ac0f58",  # noqa: E501
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


def _get_cached_binary(name: str, url: str, want_hash: str) -> str:
    cache_dir = Path.home().joinpath(".cache", "carbon-lang-scripts")
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Hold a lock while checksumming and downloading the path. Otherwise,
    # parallel runs by pre-commit may conflict with one another with
    # simultaneous downloads.
    with open(cache_dir.joinpath(f"{name}.lock"), "w") as lock_file:
        fcntl.lockf(lock_file.fileno(), fcntl.LOCK_EX)

        # Check if there's a cached file that can be used.
        local_path = cache_dir.joinpath(name)
        if local_path.is_file() and want_hash == _get_hash(local_path):
            return str(local_path)

        # Download the file.
        retries = 5
        while True:
            err = _download(url, local_path)
            if err is None:
                break
            retries -= 1
            if retries == 0:
                exit(f"Failed to download {url}: HTTP {err}.")
            time.sleep(1)
        local_path.chmod(0o755)

        # Verify the downloaded hash.
        found_hash = _get_hash(local_path)
        if want_hash != found_hash:
            exit(
                f"Downloaded {url} but found sha256 "
                f"{found_hash} ({local_path.stat().st_size} bytes), wanted "
                f"{want_hash}"
            )

    return str(local_path)


def _get_machine() -> str:
    machine = platform.machine()
    if machine == "x86_64":
        machine = "amd64"
    elif machine == "aarch64":
        machine = "arm64"
    return machine


def _get_platform_ext() -> str:
    if platform.system() == "Windows":
        return ".exe"
    else:
        return ""


def _select_hash(hashes: Dict[str, str], version: str) -> str:
    # Ensure the platform version is supported and has a hash.
    if version not in hashes:
        # If this because a platform support issue, we may need to print errors.
        exit(f"No release available for platform: {version}")
    return hashes[version]


def get_target_determinator() -> str:
    """Install the Bazel target-determinator tool to carbon-lang's cache."""
    # Translate platform information into this tool's release binary form.
    version = f"{platform.system().lower()}.{_get_machine()}"
    ext = _get_platform_ext()
    url = f"{_TARGET_DETERMINATOR_URL}/target-determinator.{version}{ext}"
    want_hash = _select_hash(_TARGET_DETERMINATOR_SHAS, version)

    return _get_cached_binary(f"target-determinator{ext}", url, want_hash)


def get_release(release: Release) -> str:
    """Install a Bazel-released tool to carbon-lang's cache.

    release: The release to cache.
    """
    # Translate platform information into Bazel's release form.
    version = f"{platform.system().lower()}-{_get_machine()}"
    ext = _get_platform_ext()
    url = f"{_BAZEL_TOOLS_URL}/{release.value}-{version}{ext}"
    want_hash = _select_hash(_BAZEL_TOOLS_VERSION_SHAS[release.value], version)

    return _get_cached_binary(f"{release.value}{ext}", url, want_hash)


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
