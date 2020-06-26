#!/usr/bin/env python3

"""Prepares a new proposal file and PR."""

__copyright__ = """
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""

import os
import re
import shlex
import shutil
import subprocess
import sys

_USAGE = """Usage:
  ./new-proposal.py <title>

Generates a branch and PR for a new proposal with the specified title.
"""

_PROMPT = """This will:
  - Create and switch to a new branch named '%s'.
  - Create a new proposal titled '%s'.
  - Create a PR for the proposal.

Continue? (Y/n) """


def _FillTemplate(template_path, title, pr_num):
    """Fills out template TODO fields."""
    with open(template_path) as template_file:
        content = template_file.read()
    content = re.sub(r"^# TODO\n", "# %s\n" % title, content)
    content = re.sub(
        r"(https://github.com/[^/]+/[^/]+/pull/)####",
        r"\g<1>%d" % pr_num,
        content,
    )
    content = re.sub(r"## TODO(?:.|\n)*(## Problem)", r"\1", content)
    return content


def _Run(argv, check=True):
    """Runs a command."""
    cmd = " ".join([shlex.quote(x) for x in argv])
    print("\n+ RUNNING: %s" % cmd, file=sys.stderr)
    p = subprocess.run(argv)
    if check and p.returncode != 0:
        sys.exit("ERROR: Command failed: %s" % cmd)


def _RunPRCreate(argv):
    """Runs a command and returns the PR#."""
    cmd = " ".join([shlex.quote(x) for x in argv])
    print("\n+ RUNNING: %s" % cmd, file=sys.stderr)
    p = subprocess.Popen(argv, stdout=subprocess.PIPE)
    out, _ = p.communicate()
    out = out.decode("utf-8")
    print(out, end="")
    if p.returncode != 0:
        sys.exit("ERROR: Command failed: %s" % cmd)
    match = re.search(
        r"^https://github.com/[^/]+/[^/]+/pull/(\d+)$", out, re.MULTILINE
    )
    if not match:
        sys.exit("ERROR: Failed to find PR# in output.")
    return int(match[1])


if __name__ == "__main__":
    # Require an argument.
    if len(sys.argv) != 2:
        sys.exit(_USAGE)
    title = sys.argv[1]

    # Verify git and gh are available.
    git_bin = shutil.which("git")
    if not git_bin:
        sys.exit("ERROR: Missing `git` CLI.")
    gh_bin = shutil.which("gh")
    if not gh_bin:
        sys.exit("ERROR: Missing `gh` CLI.")
    precommit_bin = shutil.which("pre-commit")
    if not precommit_bin:
        sys.exit("ERROR: Missing `pre-commit` CLI.")

    # Ensure a good working directory.
    proposals_dir = os.path.realpath(
        os.path.join(os.path.dirname(__file__), "../../proposals")
    )
    os.chdir(proposals_dir)

    # Verify there are no uncommitted changes.
    p = subprocess.run([git_bin, "diff-index", "--quiet", "HEAD", "--"])
    if p.returncode != 0:
        sys.exit("ERROR: There are uncommitted changes in your git repo.")

    # Only use the first 20 chars of the title for branch names.
    branch = "proposal-%s" % (title.lower().replace(" ", "-")[0:20])

    # Prompt before proceeding.
    response = "?"
    while response not in ("y", "n", ""):
        response = input(_PROMPT % (branch, title)).lower()
    if response == "n":
        sys.exit("ERROR: Cancelled")

    # Create a proposal branch.
    _Run([git_bin, "checkout", "-b", branch, "master"])
    _Run([git_bin, "push", "-u", "origin", branch])

    # Copy template.md to a temp file.
    template_path = os.path.join(proposals_dir, "template.md")
    temp_path = os.path.join(proposals_dir, "new-proposal.tmp")
    shutil.copyfile(template_path, temp_path)
    _Run([git_bin, "add", temp_path])
    _Run([git_bin, "commit", "-m", "Creating new proposal: %s" % title])

    # Create a PR with WIP+proposal labels.
    _Run([git_bin, "push"])
    pr_num = _RunPRCreate(
        [
            gh_bin,
            "pr",
            "create",
            "--label",
            "WIP,proposal",
            "--title",
            title,
            "--body",
            "",
        ]
    )

    # Remove the temp file, create p####.md, and fill in PR information.
    os.remove(temp_path)
    final_path = os.path.join(proposals_dir, "p%04d.md" % pr_num)
    content = _FillTemplate(template_path, title, pr_num)
    with open(final_path, "w") as final_file:
        final_file.write(content)
    _Run([git_bin, "add", temp_path, final_path])
    _Run([precommit_bin, "run"], check=False)  # Needs a ToC update.
    _Run([git_bin, "add", final_path])
    _Run([git_bin, "commit", "-m", "Filling out template with PR %d" % pr_num])

    # Push the PR update.
    _Run([git_bin, "push"])

    print(
        "\nCreated PR %d for %s. Make changes to:\n  %s"
        % (pr_num, title, final_path)
    )
