#!/usr/bin/env python3

"""Queries latest module versions from MODULE.bazel."""

__copyright__ = """
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""

import json
import re
import subprocess
import urllib.error
import urllib.request


def _query_bazel_deps(module_text: str) -> None:
    """Query BCR for `bazel_dep` rule versions."""
    bazel_dep_re = re.compile(
        r'bazel_dep\([^)]*name\s*=\s*"([^"]+)"[^)]*version\s*='
    )
    packages = sorted([m[1] for m in bazel_dep_re.finditer(module_text)])
    print("- BCR:")
    for pkg in packages:
        try:
            url = f"https://bcr.bazel.build/modules/{pkg}/metadata.json"
            with urllib.request.urlopen(url) as response:
                data = json.loads(response.read().decode())
                versions = data.get("versions", [])
                print(f"  - {pkg}: {versions[-1] if versions else 'Unknown'}")
        except Exception as e:
            print(f"  - {pkg}: Error {e}")


def _query_archive_overrides(module_text: str) -> None:
    """Query Git for `archive_override` rule versions."""
    archive_re = re.compile(
        r"archive_override\(\s*"
        r'module_name\s*=\s*"([^"]+)".*?'
        r'urls\s*=\s*\["[^"]*?github\.com/([^/]+/[^/]+?)/archive',
        re.DOTALL,
    )
    github_tags = sorted(
        [
            (m[1], f"https://github.com/{m[2]}.git")
            for m in archive_re.finditer(module_text)
        ]
    )

    print("- GitHub Tag:")
    for pkg, url in github_tags:
        try:
            output = subprocess.check_output(
                ["git", "ls-remote", "--tags", url], text=True
            )
            # Find the latest tag by sorting them (excluding the ^{}
            # dereferenced tags).
            tags = [
                line.split("/")[-1]
                for line in output.splitlines()
                if not line.endswith("^{}")
            ]

            # Simple version sort (git-style) using split by common delimiters.
            tags.sort(
                key=lambda tag: [
                    int(x) if x.isdigit() else x
                    for x in re.split(r"(\d+)", tag)
                ]
            )
            latest_tag = tags[-1] if tags else "Unknown"
            print(f"  - {pkg}: {latest_tag}")
        except Exception as e:
            print(f"  - {pkg}: Error {e}")


def _query_git_overrides(module_text: str) -> None:
    """Query GitHub for `git_override` rule versions."""
    git_re = re.compile(
        r"git_override\(\s*"
        r'module_name\s*=\s*"([^"]+)".*?'
        r'remote\s*=\s*"([^"]+)"',
        re.DOTALL,
    )
    git_repos = sorted([(m[1], m[2]) for m in git_re.finditer(module_text)])
    print("- Git HEAD:")
    for pkg, url in git_repos:
        try:
            output = subprocess.check_output(
                ["git", "ls-remote", url, "HEAD"], text=True
            )
            commit = output.split()[0]
            print(f"  - {pkg}: {commit}")
        except Exception as e:
            print(f"  - {pkg}: Error {e}")


def main() -> None:
    with open("MODULE.bazel", "r") as f:
        module_text = f.read()

    _query_bazel_deps(module_text)
    _query_archive_overrides(module_text)
    _query_git_overrides(module_text)


if __name__ == "__main__":
    main()
