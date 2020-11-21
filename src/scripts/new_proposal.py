#!/usr/bin/env python3

"""Prepares a new proposal file and PR."""

__copyright__ = """
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""

import argparse
import os
import re
import shlex
import shutil
import subprocess
import sys

_PROMPT = """This will:
  - Create and switch to a new branch named '%s'.
  - Create a new proposal titled '%s'.
  - Create a PR for the proposal.

Continue? (Y/n) """


def _exit(error):
    """Wraps sys.exit for testing."""
    sys.exit(error)


def _parse_args(args=None):
    """Parses command-line arguments and flags."""
    parser = argparse.ArgumentParser(
        description="Generates a branch and PR for a new proposal with the "
        "specified title."
    )
    parser.add_argument(
        "title",
        metavar="TITLE",
        help="The title of the proposal.",
    )
    parser.add_argument(
        "--branch",
        metavar="BRANCH",
        help="The name of the branch. Automatically generated from the title "
        "by default.",
    )
    return parser.parse_args(args=args)


def _calculate_branch(parsed_args):
    """Returns the branch name."""
    if parsed_args.branch:
        return parsed_args.branch
    # Only use the first 20 chars of the title for branch names.
    return "proposal-%s" % (parsed_args.title.lower().replace(" ", "-")[0:20])


def _find_tool(tool):
    """Checks if a tool is present."""
    tool_path = shutil.which(tool)
    if not tool_path:
        _exit("ERROR: Missing the '%s' command-line tool." % tool)
    return tool_path


def _fill_template(template_path, title, pr_num):
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


def _get_proposals_dir():
    """Returns the path to the proposals directory."""
    return os.path.realpath(
        os.path.join(os.path.dirname(__file__), "../../proposals")
    )


def _run(argv, check=True):
    """Runs a command."""
    cmd = " ".join([shlex.quote(x) for x in argv])
    print("\n+ RUNNING: %s" % cmd, file=sys.stderr)
    p = subprocess.run(argv)
    if check and p.returncode != 0:
        _exit("ERROR: Command failed: %s" % cmd)


def _run_pr_create(argv):
    """Runs a command and returns the PR#."""
    cmd = " ".join([shlex.quote(x) for x in argv])
    print("\n+ RUNNING: %s" % cmd, file=sys.stderr)
    p = subprocess.Popen(argv, stdout=subprocess.PIPE)
    out, _ = p.communicate()
    out = out.decode("utf-8")
    print(out, end="")
    if p.returncode != 0:
        _exit("ERROR: Command failed: %s" % cmd)
    match = re.search(
        r"^https://github.com/[^/]+/[^/]+/pull/(\d+)$", out, re.MULTILINE
    )
    if not match:
        _exit("ERROR: Failed to find PR# in output.")
    return int(match[1])


def main():
    parsed_args = _parse_args()
    title = parsed_args.title
    branch = _calculate_branch(parsed_args)

    # Verify tools are available.
    git_bin = _find_tool("git")
    gh_bin = _find_tool("gh")
    precommit_bin = _find_tool("pre-commit")

    # Ensure a good working directory.
    proposals_dir = _get_proposals_dir()
    os.chdir(proposals_dir)

    # Verify there are no uncommitted changes.
    p = subprocess.run([git_bin, "diff-index", "--quiet", "HEAD", "--"])
    if p.returncode != 0:
        _exit("ERROR: There are uncommitted changes in your git repo.")

    # Prompt before proceeding.
    response = "?"
    while response not in ("y", "n", ""):
        response = input(_PROMPT % (branch, title)).lower()
    if response == "n":
        _exit("ERROR: Cancelled")

    # Create a proposal branch.
    _run([git_bin, "checkout", "-b", branch, "trunk"])
    _run([git_bin, "push", "-u", "origin", branch])

    # Copy template.md to a temp file.
    template_path = os.path.join(proposals_dir, "template.md")
    temp_path = os.path.join(proposals_dir, "new-proposal.tmp.md")
    shutil.copyfile(template_path, temp_path)
    _run([git_bin, "add", temp_path])
    _run([git_bin, "commit", "-m", "Creating new proposal: %s" % title])

    # Create a PR with WIP+proposal labels.
    _run([git_bin, "push"])
    pr_num = _run_pr_create(
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
    content = _fill_template(template_path, title, pr_num)
    with open(final_path, "w") as final_file:
        final_file.write(content)
    _run([git_bin, "add", temp_path, final_path])
    _run([precommit_bin, "run"], check=False)  # Needs a ToC update.
    _run([git_bin, "add", final_path, os.path.join(proposals_dir, "README.md")])
    _run(
        [
            git_bin,
            "commit",
            "--amend",
            "-m",
            "Filling out template with PR %d" % pr_num,
        ]
    )

    # Push the PR update.
    _run([git_bin, "push", "--force-with-lease"])

    print(
        "\nCreated PR %d for %s. Make changes to:\n  %s"
        % (pr_num, title, final_path)
    )


if __name__ == "__main__":
    main()
