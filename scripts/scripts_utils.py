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

_URL = "https://github.com/bazelbuild/buildtools/releases/download/4.2.5/"

"""Version SHAs.

Gather shas with:
    for f in buildozer buildifier; do
        echo \"$f\": {
        for v in darwin-amd64 darwin-arm64 linux-amd64 linux-arm64 \
            windows-amd64.exe
        do
            echo "\"$v\": \"$(wget -q -O - https://github.com/bazelbuild/buildtools/releases/download/4.2.5/$f-$v | sha256sum | cut -d ' ' -f1)\", # noqa: E501"
        done
        echo },
    done
"""
_VERSION_SHAS = {
    "buildozer": {
        "darwin-amd64": "3fe671620e6cb7d2386f9da09c1de8de88b02b9dd9275cdecd8b9e417f74df1b",  # noqa: E501
        "darwin-arm64": "ff4d297023fe3e0fd14113c78f04cef55289ca5bfe5e45a916be738b948dc743",  # noqa: E501
        "linux-amd64": "e8e39b71c52318a9030dd9fcb9bbfd968d0e03e59268c60b489e6e6fc1595d7b",  # noqa: E501
        "linux-arm64": "96227142969540def1d23a9e8225524173390d23f3d7fd56ce9c4436953f02fc",  # noqa: E501
        "windows-amd64.exe": "2a9a7176cbd3b2f0ef989502128efbafd3b156ddabae93b9c979cd4017ffa300",  # noqa: E501
    },
    "buildifier": {
        "darwin-amd64": "757f246040aceb2c9550d02ef5d1f22d3ef1ff53405fe76ef4c6239ef1ea2cc1",  # noqa: E501
        "darwin-arm64": "4cf02e051f6cda18765935cb6e77cc938cf8b405064589a50fe9582f82c7edaf",  # noqa: E501
        "linux-amd64": "f94e71b22925aff76ce01a49e1c6c6d31f521bbbccff047b81f2ea01fd01a945",  # noqa: E501
        "linux-arm64": "2113d79e45efb51e2b3013c8737cb66cadae3fd89bd7e820438cb06201e50874",  # noqa: E501
        "windows-amd64.exe": "4185a40d3154cacbe8b79f570b94e2c6f74fc9e317362b7d028c2e6c94edf9ba",  # noqa: E501
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
