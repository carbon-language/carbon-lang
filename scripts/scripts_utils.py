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
    "https://github.com/bazelbuild/buildtools/releases/download/v8.2.0/"
)

# Structured information per release tool.
_RELEASES = {
    Release.BAZELISK: ReleaseInfo(
        "https://github.com/bazelbuild/bazelisk/releases/download/v1.26.0/", "-"
    ),
    Release.BUILDIFIER: ReleaseInfo(_BAZEL_TOOLS_URL, "-"),
    Release.BUILDOZER: ReleaseInfo(_BAZEL_TOOLS_URL, "-"),
    Release.TARGET_DETERMINATOR: ReleaseInfo(
        "https://github.com/bazel-contrib/target-determinator/releases/download/v0.30.3/",  # noqa: E501
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
        "darwin-amd64": "5c77f33f91dd3df119d192175100cb5b50302eb7ee37859cbab79e10a76ccce8",  # noqa: E501
        "darwin-arm64": "d1ca9911cc19e1f17483f93956908334f2b7f3dd13f20853417b68fc3c3eb370",  # noqa: E501
        "linux-amd64": "6539c12842ad76966f3d493e8f80d67caa84ec4a000e220d5459833c967c12bc",  # noqa: E501
        "linux-arm64": "54f85ef4c23393f835252cc882e5fea596e8ef3c4c2056b059f8067cd19f0351",  # noqa: E501
        "windows-amd64.exe": "023734f33ed6b9c6d65468fe20bb2c5fb32473ccb8aca2fc5bf1521e61ce1622",  # noqa: E501
    },
    Release.BUILDIFIER: {
        "darwin-amd64": "309b3c3bfcc4b1533d5f7f796adbd266235cfb6f01450f3e37423527d209a309",  # noqa: E501
        "darwin-arm64": "e08381a3ed1d59c0a17d1cee1d4e7684c6ce1fc3b5cfa1bd92a5fe978b38b47d",  # noqa: E501
        "linux-amd64": "3e79e6c0401b5f36f8df4dfc686127255d25c7eddc9599b8779b97b7ef4cdda7",  # noqa: E501
        "linux-arm64": "c624a833bfa64d3a457ef0235eef0dbda03694768aab33f717a7ffd3f803d272",  # noqa: E501
        "windows-amd64.exe": "a27fcf7521414f8214787989dcfb2ac7d3f7c28b56e44384e5fa06109953c2f1",  # noqa: E501
    },
    Release.BUILDOZER: {
        "darwin-amd64": "b7bd7189a9d4de22c10fd94b7d1d77c68712db9bdd27150187bc677e8c22960e",  # noqa: E501
        "darwin-arm64": "781527c5337dadba5a0611c01409c669852b73b72458650cc7c5f31473f7ae3f",  # noqa: E501
        "linux-amd64": "0e54770aa6148384d1edde39ef20e10d2c57e8c09dd42f525e100f51b0b77ae1",  # noqa: E501
        "linux-arm64": "a9f38f2781de41526ce934866cb79b8b5b59871c96853dc5a1aee26f4c5976bb",  # noqa: E501
        "windows-amd64.exe": "8ce5a9a064b01551ffb8d441fa9ef4dd42c9eeeed6bc71a89f917b3474fd65f6",  # noqa: E501
    },
    Release.TARGET_DETERMINATOR: {
        "darwin.amd64": "04adf78f763e622467181669fdf275e01edc1ec3d79940e78040127a15b7c8b2",  # noqa: E501
        "darwin.arm64": "f59ee18404577a704bc1399907c35b546fd66ffd5a1e145e7955a3d3e57a2a13",  # noqa: E501
        "linux.amd64": "6eaa8921e6c614c309536af3dc7ca23f52e5ced30b9032e6443bbe0d41a8ae33",  # noqa: E501
        "linux.arm64": "1c7216426d4e2ca63b912fe2be2ab8f3f9ccbe2aefa174e2a22e7f19f5f36065",  # noqa: E501
        "windows.amd64.exe": "53d377274c40b1a0e37db96c20fa4b701d1e5e2650af14517c49e170b2564736",  # noqa: E501
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
