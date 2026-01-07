#!/usr/bin/env python3

"""Updates example module file to use the nightly toolchain release.

This script computes the most recent nightly Carbon toolchain release, and
updates the example module file with an `archive_override` pointing at it.

Usage:
  # Within the `examples/bazel` directory:
  ./update_module_to_nightly.py

For more details about using the Carbon toolchain with Bazel, see the
documentation in `examples/bazel/MODULE.bazel`.
"""

__copyright__ = """
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""

import datetime
import re
import os
import sys
import base64
import urllib.request
import urllib.error
import json

MODULE_NAME = "carbon_toolchain"
MODULE_FILENAME = "MODULE.bazel"
DEP_PATTERN = re.compile(
    rf'bazel_dep\s*\(\s*name\s*=\s*"{MODULE_NAME}".*?\)', re.DOTALL
)
OVERRIDE_PATTERN = re.compile(
    rf'archive_override\s*\(\s*module_name\s*=\s*"{MODULE_NAME}".*?\)',
    re.DOTALL,
)

# The nightly build starts at 2am UTC, and we give it up to 4 hours to complete.
BUFFER_HOURS = 6

RELEASES_URL = (
    "https://github.com/carbon-language/carbon-lang/releases/download"
)
API_URL = (
    "https://api.github.com/repos/carbon-language/carbon-lang/releases/tags"
)


def log(msg: str) -> None:
    print(f"[update_module_to_nightly] {msg}", file=sys.stderr)


def compute_version() -> str:
    now_utc = datetime.datetime.now(datetime.timezone.utc)
    target_time = now_utc - datetime.timedelta(hours=BUFFER_HOURS)
    date = target_time.strftime("%Y.%m.%d")

    return f"0.0.0-0.nightly.{date}"


def get_digest(version: str, filename: str) -> str:
    url = f"{API_URL}/v{version}"
    headers = {
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
        # GitHub API requires a User-Agent, urllib doesn't send one by default
        "User-Agent": "python-urllib",
    }
    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req) as response:
            release_data = json.load(response)
    except urllib.error.HTTPError as e:
        log(f"Error: Unable to find `v{version}`: {e.code}: {e.reason}")
        sys.exit(1)

    assets = release_data.get("assets", [])
    for asset in assets:
        name = str(asset.get("name"))
        if name != filename:
            continue

        digest = str(asset.get("digest"))
        if not digest.startswith("sha256:"):
            log(f"Error: found invalid digest for `{filename}`: `{digest}`")
            sys.exit(1)

        # Re-encode from the GitHub format to Bazel.
        digest = (
            "sha256-"
            + base64.b64encode(bytes.fromhex(digest[len("sha256:") :])).decode()
        )
        return digest

    log(f"Error: unable to find a digest for `{filename}`")
    sys.exit(1)


def generate_override(version: str) -> str:
    basename = f"carbon_toolchain-{version}"
    digest = get_digest(version, f"{basename}.tar.gz")
    return (
        f"archive_override(\n"
        f'    module_name = "{MODULE_NAME}",\n'
        f'    integrity = "{digest}",\n'
        f'    strip_prefix = "{basename}/lib/carbon",\n'
        f'    urls = ["{RELEASES_URL}/v{version}/{basename}.tar.gz"],\n'
        f")"
    )


def main() -> None:
    if not os.path.exists(MODULE_FILENAME):
        log(f"Error: `{MODULE_FILENAME}` not found in current directory.")
        sys.exit(1)

    with open(MODULE_FILENAME, "r") as f:
        content = f.read()

    # 1. Verification (Check if dependency exists)
    dep_match = DEP_PATTERN.search(content)
    if not dep_match:
        log(
            f"Error: `bazel_dep` for `{MODULE_NAME}` not found in "
            f"`{MODULE_FILENAME}`."
        )
        sys.exit(1)

    version = compute_version()
    new_block = generate_override(version)

    (new_content, count) = OVERRIDE_PATTERN.subn(new_block, content)
    if count > 0:
        log("Existing override found, replacing with a fresh one")
    else:
        log("No existing override found, inserting one")
        new_content = (
            content[: dep_match.end()]
            + "\n\n"
            + new_block
            + content[dep_match.end() :]
        )

    with open(MODULE_FILENAME, "w") as f:
        f.write(new_content)

    log(f"Successfully updated `{MODULE_FILENAME}` to version `{version}`")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(1)
