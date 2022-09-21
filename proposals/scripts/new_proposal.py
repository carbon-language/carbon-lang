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
from typing import List, Optional

_PROMPT = """This will:
  - Create and switch to a new branch named '%s'.
  - Create a new proposal titled '%s'.
  - Create a PR for the proposal.

Continue? (Y/n) """

_LINK_TEMPLATE = """Proposal links (add links as proposal evolves):

-   Evolution links:
    -   [Proposal PR](https://github.com/carbon-language/carbon-lang/pull/%s)
    -   `[RFC topic](TODO)`
    -   `[Decision topic](TODO)`
    -   `[Decision PR](TODO)`
    -   `[Announcement](TODO)`
-   Related links (optional):
    -   `[Idea topic](TODO)`
    -   `[TODO](TODO)`
"""


def _parse_args(args: Optional[List[str]] = None) -> argparse.Namespace:
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
    parser.add_argument(
        "--proposals-dir",
        metavar="PROPOSALS_DIR",
        help="The proposals directory, mainly for testing cross-repository. "
        "Automatically found by default.",
    )
    parser.add_argument(
        "--branch-start-point",
        metavar="BRANCH_START_POINT",
        default="trunk",
        type=str,
        help="The starting point for the new branch.",
    )
    return parser.parse_args(args=args)


def _calculate_branch(parsed_args: argparse.Namespace) -> str:
    """Returns the branch name."""
    if parsed_args.branch:
        assert isinstance(parsed_args.branch, str)
        return parsed_args.branch
    # Only use the first 20 chars of the title for branch names.
    return "proposal-%s" % (parsed_args.title.lower().replace(" ", "-")[0:20])


def _find_tool(tool: str) -> str:
    """Checks if a tool is present."""
    tool_path = shutil.which(tool)
    if not tool_path:
        exit("ERROR: Missing the '%s' command-line tool." % tool)
    return tool_path


def _fill_template(template_path: str, title: str, pr_num: int) -> str:
    """Fills out template TODO fields."""
    with open(template_path) as template_file:
        content = template_file.read()
    content = re.sub(r"^# TODO\n", "# %s\n" % title, content)
    content = re.sub(
        r"(https://github.com/[^/]+/[^/]+/pull/)####",
        r"\g<1>%d" % pr_num,
        content,
    )
    content = re.sub(r"\n## TODO(?:.|\n)*?(\n## )", r"\1", content)
    return content


def _get_proposals_dir(parsed_args: argparse.Namespace) -> str:
    """Returns the path to the proposals directory."""
    if parsed_args.proposals_dir:
        assert isinstance(parsed_args.proposals_dir, str)
        return parsed_args.proposals_dir
    return os.path.realpath(
        os.path.join(os.path.dirname(__file__), "../../proposals")
    )


def _run(
    argv: List[str], check: bool = True, get_stdout: bool = False
) -> Optional[str]:
    """Runs a command."""
    cmd = " ".join([shlex.quote(x) for x in argv])
    print("\n+ RUNNING: %s" % cmd, file=sys.stderr)

    stdout_pipe = None
    if get_stdout:
        stdout_pipe = subprocess.PIPE

    p = subprocess.Popen(argv, stdout=stdout_pipe)
    stdout, _ = p.communicate()
    if get_stdout:
        out = stdout.decode("utf-8")
        print(out, end="")
    if check and p.returncode != 0:
        exit("ERROR: Command failed: %s" % cmd)
    if get_stdout:
        return out
    return None


def _run_pr_create(argv: List[str]) -> int:
    """Runs a command and returns the PR#."""
    out = _run(argv, get_stdout=True)
    assert out is not None
    match = re.search(
        r"^https://github.com/[^/]+/[^/]+/pull/(\d+)$", out, re.MULTILINE
    )
    if not match:
        exit("ERROR: Failed to find PR# in output.")
    return int(match.group(1))


def main() -> None:
    parsed_args = _parse_args()
    title = parsed_args.title
    branch = _calculate_branch(parsed_args)

    # Verify tools are available.
    gh_bin = _find_tool("gh")
    git_bin = _find_tool("git")
    precommit_bin = _find_tool("pre-commit")

    # Ensure a good working directory.
    proposals_dir = _get_proposals_dir(parsed_args)
    os.chdir(proposals_dir)

    # Verify there are no uncommitted changes.
    p = subprocess.run([git_bin, "diff-index", "--quiet", "HEAD", "--"])
    if p.returncode != 0:
        exit("ERROR: There are uncommitted changes in your git repo.")

    # Prompt before proceeding.
    response = "?"
    while response not in ("y", "n", ""):
        response = input(_PROMPT % (branch, title)).lower()
    if response == "n":
        exit("ERROR: Cancelled")

    # Create a proposal branch.
    _run(
        [git_bin, "switch", "--create", branch, parsed_args.branch_start_point]
    )
    _run([git_bin, "push", "-u", "origin", branch])

    # Copy template.md to a temp file.
    template_path = os.path.join(proposals_dir, "scripts/template.md")
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
            "--draft",
            "--label",
            "proposal",
            "--label",
            "proposal draft",
            "--repo",
            "carbon-language/carbon-lang",
            "--title",
            title,
            "--body",
            "TODO: add summary and links here",
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
