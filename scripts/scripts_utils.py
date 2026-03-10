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
    "https://github.com/bazelbuild/buildtools/releases/download/v8.5.1/"
)

# Structured information per release tool.
_RELEASES = {
    Release.BAZELISK: ReleaseInfo(
        "https://github.com/bazelbuild/bazelisk/releases/download/v1.28.1/", "-"
    ),
    Release.BUILDIFIER: ReleaseInfo(_BAZEL_TOOLS_URL, "-"),
    Release.BUILDOZER: ReleaseInfo(_BAZEL_TOOLS_URL, "-"),
    Release.TARGET_DETERMINATOR: ReleaseInfo(
        "https://github.com/bazel-contrib/target-determinator/releases/download/v0.32.0/",  # noqa: E501
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
        "darwin-amd64": "023225736cea5dc88f2b0807d5b1af4eb0f69a4ed45e3994b2c18c263bc80e48",  # noqa: E501
        "darwin-arm64": "dea3f3f5de2dbc5e269e0132cdd369d5efe738f7b973d5d4eb2b4f7055a97b39",  # noqa: E501
        "linux-amd64": "22e7d3a188699982f661cf4687137ee52d1f24fec1ec893d91a6c4d791a75de8",  # noqa: E501
        "linux-arm64": "8ded44b58a0d9425a4178af26cf17693feac3b87bdcfef0a2a0898fcd1afc9f2",  # noqa: E501
        "windows-amd64.exe": "b9d65a1f7c2d7af885a96a4fd5aa36b40fb41816d30944390569eef908bdc954",  # noqa: E501
    },
    Release.BUILDIFIER: {
        "darwin-amd64": "31de189e1a3fe53aa9e8c8f74a0309c325274ad19793393919e1ca65163ca1a4",  # noqa: E501
        "darwin-arm64": "62836a9667fa0db309b0d91e840f0a3f2813a9c8ea3e44b9cd58187c90bc88ba",  # noqa: E501
        "linux-amd64": "887377fc64d23a850f4d18a077b5db05b19913f4b99b270d193f3c7334b5a9a7",  # noqa: E501
        "linux-arm64": "947bf6700d708026b2057b09bea09abbc3cafc15d9ecea35bb3885c4b09ccd04",  # noqa: E501
        "windows-amd64.exe": "f4ecb9c73de2bc38b845d4ee27668f6248c4813a6647db4b4931a7556052e4e1",  # noqa: E501
    },
    Release.BUILDOZER: {
        "darwin-amd64": "b85b9ad59c1543999a5d8bc8bee6e42b9f025be3ff520bc2d090213698850b43",  # noqa: E501
        "darwin-arm64": "d0cf2f6e11031d62bfd4584e46eb6bb708a883ff948be76538b34b83de833262",  # noqa: E501
        "linux-amd64": "2b745ca2ad41f1e01673fb59ac50af6b45ca26105c1d20fad64c3d05a95522f5",  # noqa: E501
        "linux-arm64": "87ee1d2d81d08ccae8f9147fc58503967c85878279e892f2990912412feef1a1",  # noqa: E501
        "windows-amd64.exe": "e177155c2c8ef41569791de34f13077cefe3e5623f9f02e099347232bc028901",  # noqa: E501
    },
    Release.TARGET_DETERMINATOR: {
        "darwin.amd64": "289c61f8f4553a29d6ad2fbf1779a83180e7504c278196851becfd3f4163f6f4",  # noqa: E501
        "darwin.arm64": "6e688292b43f99f7b76d0af0fc32ac1eec8c110571a323c7e30bcfc9ac41275c",  # noqa: E501
        "linux.amd64": "792ed4c6f53aad60e8255686c272f28d3c039eaaef03518dbbde33519713802a",  # noqa: E501
        "linux.arm64": "339d0b7d3c72734f435d67e9b66936261c58c02be4ce64bdb76c85f1683e8084",  # noqa: E501
        "windows.amd64.exe": "457aba640e737edaf06d3922fa33914109dbe6382af951dfe6a99dbdf50c714c",  # noqa: E501
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
