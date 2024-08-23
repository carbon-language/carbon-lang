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
    "https://github.com/bazelbuild/buildtools/releases/download/v7.1.2/"
)

# Structured information per release tool.
_RELEASES = {
    Release.BAZELISK: ReleaseInfo(
        "https://github.com/bazelbuild/bazelisk/releases/download/v1.20.0/", "-"
    ),
    Release.BUILDIFIER: ReleaseInfo(_BAZEL_TOOLS_URL, "-"),
    Release.BUILDOZER: ReleaseInfo(_BAZEL_TOOLS_URL, "-"),
    Release.TARGET_DETERMINATOR: ReleaseInfo(
        "https://github.com/bazel-contrib/target-determinator/releases/download/v0.27.0/",  # noqa: E501
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
        "darwin-amd64": "51a6228d51704c656df9fceacad18d64f265b973905b3efdcf8a504b687545bf",  # noqa: E501
        "darwin-arm64": "29753341c0ddc35931fb240e247fbba0b83ef81bccc2433dd075363ec02a67a6",  # noqa: E501
        "linux-amd64": "d9af1fa808c0529753c3befda75123236a711d971d3485a390507122148773a3",  # noqa: E501
        "linux-arm64": "467ec3821aca5e278c8570b7c25e0dfc1a061d2873be89e4a266aaf488148426",  # noqa: E501
        "windows-amd64.exe": "4175ce7ef4b552fb17e93ce49a245679dc26a35cf2fbc7c3146daca6ffc7a81e",  # noqa: E501
    },
    Release.BUILDIFIER: {
        "darwin-amd64": "687c49c318fb655970cf716eed3c7bfc9caeea4f2931a2fd36593c458de0c537",  # noqa: E501
        "darwin-arm64": "d0909b645496608fd6dfc67f95d9d3b01d90736d7b8c8ec41e802cb0b7ceae7c",  # noqa: E501
        "linux-amd64": "28285fe7e39ed23dc1a3a525dfcdccbc96c0034ff1d4277905d2672a71b38f13",  # noqa: E501
        "linux-arm64": "c22a44eee37b8927167ee6ee67573303f4e31171e7ec3a8ea021a6a660040437",  # noqa: E501
        "windows-amd64.exe": "a8331515019d8d3e01baa1c76fda19e8e6e3e05532d4b0bce759bd759d0cafb7",  # noqa: E501
    },
    Release.BUILDOZER: {
        "darwin-amd64": "90da5cf4f7db73007977a8c6bec23fa7022265978187e1da8df5edc91daf6ee1",  # noqa: E501
        "darwin-arm64": "bedff301bc51f04da46d2c8900c1753032ea88485af375a9f1b7bed0915558e0",  # noqa: E501
        "linux-amd64": "8d5c459ab21b411b8be059a8bdf59f0d3eabf9dff943d5eccb80e36e525cc09d",  # noqa: E501
        "linux-arm64": "a00d1790e8c92c5022d83e345d6629506836d73c23c5338d5f777589bfaed02d",  # noqa: E501
        "windows-amd64.exe": "3a650e10f07787760889d7e5694924d881265ae2384499fd59ada7c39c02366e",  # noqa: E501
    },
    Release.TARGET_DETERMINATOR: {
        "darwin.amd64": "f3ef5abce3499926534237ffa183f54139c4760e376813973b35f8cfa5eb50cf",  # noqa: E501
        "darwin.arm64": "17ee63f8f34c4f61907cf963ce81463b3be5b0a67b068beb02ab9a8cf7fb13d5",  # noqa: E501
        "linux.amd64": "65000bba3a5eb1713d93b1e08e33b6fbe5787535664bbc1ba2f4166b0d26d0a1",  # noqa: E501
        "linux.arm64": "99146eef911873f8dbba722214d4c382ebbeab52b0e030e89314db85b70c8558",  # noqa: E501
        "windows.amd64.exe": "b59a8122a5b72517c8488a638afb8fc9c78da2eaa3c6e7ecb9638052a3ebc3ee",  # noqa: E501
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
