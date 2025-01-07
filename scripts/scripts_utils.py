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
import tempfile
import time
from typing import NamedTuple, Optional
import urllib.request


# The tools we track releases for.
class Release(Enum):
    BAZELISK = "bazelisk"
    BUILDIFIER = "buildifier"
    BUILDOZER = "buildozer"
    TARGET_DETERMINATOR = "target-determinator"


class ReleaseInfo(NamedTuple):
    # The base URL for downloads. Should include the version.
    url: str
    # The separator in a binary's name, either `-` or `.`.
    separator: str


_BAZEL_TOOLS_URL = (
    "https://github.com/bazelbuild/buildtools/releases/download/v7.3.1/"
)

# Structured information per release tool.
_RELEASES = {
    Release.BAZELISK: ReleaseInfo(
        "https://github.com/bazelbuild/bazelisk/releases/download/v1.25.0/", "-"
    ),
    Release.BUILDIFIER: ReleaseInfo(_BAZEL_TOOLS_URL, "-"),
    Release.BUILDOZER: ReleaseInfo(_BAZEL_TOOLS_URL, "-"),
    Release.TARGET_DETERMINATOR: ReleaseInfo(
        "https://github.com/bazel-contrib/target-determinator/releases/download/v0.30.0/",  # noqa: E501
        ".",
    ),
}


# Shas for the tools.
#
# To update, change the version in a tool's URL and use
# `calculate_release_shas.py`. This is maintained separate from _RELEASES just
# to make copy-paste updates simpler.
_RELEASE_SHAS = {
    Release.BAZELISK: {
        "darwin-amd64": "0af019eeb642fa70744419d02aa32df55e6e7a084105d49fb26801a660aa56d3",  # noqa: E501
        "darwin-arm64": "b13dd89c6ecd90944ca3539f5a2c715a18f69b7458878c471a902a8e482ceb4b",  # noqa: E501
        "linux-amd64": "fd8fdff418a1758887520fa42da7e6ae39aefc788cf5e7f7bb8db6934d279fc4",  # noqa: E501
        "linux-arm64": "4c8d966e40ac2c4efcc7f1a5a5cceef2c0a2f16b957e791fa7a867cce31e8fcb",  # noqa: E501
        "windows-amd64.exe": "641a3dfebd717703675f912917735c44b45cf6300bfdfb924537f3cfbffcdd92",  # noqa: E501
    },
    Release.BUILDIFIER: {
        "darwin-amd64": "375f823103d01620aaec20a0c29c6cbca99f4fd0725ae30b93655c6704f44d71",  # noqa: E501
        "darwin-arm64": "5a6afc6ac7a09f5455ba0b89bd99d5ae23b4174dc5dc9d6c0ed5ce8caac3f813",  # noqa: E501
        "linux-amd64": "5474cc5128a74e806783d54081f581662c4be8ae65022f557e9281ed5dc88009",  # noqa: E501
        "linux-arm64": "0bf86c4bfffaf4f08eed77bde5b2082e4ae5039a11e2e8b03984c173c34a561c",  # noqa: E501
        "windows-amd64.exe": "370cd576075ad29930a82f5de132f1a1de4084c784a82514bd4da80c85acf4a8",  # noqa: E501
    },
    Release.BUILDOZER: {
        "darwin-amd64": "854c9583efc166602276802658cef3f224d60898cfaa60630b33d328db3b0de2",  # noqa: E501
        "darwin-arm64": "31b1bfe20d7d5444be217af78f94c5c43799cdf847c6ce69794b7bf3319c5364",  # noqa: E501
        "linux-amd64": "3305e287b3fcc68b9a35fd8515ee617452cd4e018f9e6886b6c7cdbcba8710d4",  # noqa: E501
        "linux-arm64": "0b5a2a717ac4fc911e1fec8d92af71dbb4fe95b10e5213da0cc3d56cea64a328",  # noqa: E501
        "windows-amd64.exe": "58d41ce53257c5594c9bc86d769f580909269f68de114297f46284fbb9023dcf",  # noqa: E501
    },
    Release.TARGET_DETERMINATOR: {
        "darwin.amd64": "faa79bed4f3b516e64532beafabf4c340aa0f67c52770d9e74782b8d32033b8c",  # noqa: E501
        "darwin.arm64": "0536158d45ac9d59e5fb5fd9f061a407f28e9e3d9d95ccaa4832583ed4cf13b8",  # noqa: E501
        "linux.amd64": "2c75ff991eb5fe46d6df3f6266cac8fb8f3abf4037be486e4c5dc871bd6c4d64",  # noqa: E501
        "linux.arm64": "5ec31448c87a972583e0213028e83fefe7f566ccc9d64412e333ed12390573c5",  # noqa: E501
        "windows.amd64.exe": "11449d0388deae5b04372be35e4b480f3beeffa8b954e707d60ebd15d3ebe428",  # noqa: E501
    },
}


def chdir_repo_root() -> None:
    """Change the working directory to the repository root.

    This is done so that scripts run from a consistent directory.
    """
    os.chdir(Path(__file__).parents[1])


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
    """Returns the path to the cached binary.

    If the matching version is already cached, returns it. Otherwise, downloads
    from the URL and verifies the hash matches.
    """
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


def _select_hash(hashes: dict[str, str], version: str) -> str:
    # Ensure the platform version is supported and has a hash.
    if version not in hashes:
        # If this because a platform support issue, we may need to print errors.
        exit(f"No release available for platform: {version}")
    return hashes[version]


def get_release(release: Release) -> str:
    """Install a tool to carbon-lang's cache and return its path.

    release: The release to cache.
    """
    info = _RELEASES[release]
    shas = _RELEASE_SHAS[release]

    # Translate platform information into Bazel's release form.
    ext = _get_platform_ext()
    platform_label = (
        f"{platform.system().lower()}{info.separator}{_get_machine()}{ext}"
    )
    url = f"{info.url}/{release.value}{info.separator}{platform_label}"
    want_hash = _select_hash(shas, platform_label)

    return _get_cached_binary(f"{release.value}{ext}", url, want_hash)


def calculate_release_shas() -> None:
    """Prints sha information for tracked tool releases."""
    print("_RELEASE_SHAS = {")
    for release, info in _RELEASES.items():
        shas = _RELEASE_SHAS[release]

        print(f"  {release}: {{")
        for platform_label in shas.keys():
            url = f"{info.url}/{release.value}{info.separator}{platform_label}"
            with tempfile.NamedTemporaryFile() as f:
                path = Path(f.name)
                _download(url, path)
                hash = _get_hash(path)
            print(f'    "{platform_label}": "{hash}",  # noqa: E501')
        print("  },")
    print("}")


def locate_bazel() -> str:
    """Returns the bazel command.

    In order, try:
    1. The `BAZEL` environment variable.
    2. `bazelisk`
    3. `bazel`
    4. `run_bazelisk.py`
    """
    bazel = os.environ.get("BAZEL")
    if bazel:
        return bazel

    for cmd in ("bazelisk", "bazel"):
        target = shutil.which(cmd)
        if target:
            return target

    return str(Path(__file__).parent / "run_bazelisk.py")
