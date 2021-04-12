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
    parser.add_argument(
        "--proposals-dir",
        metavar="PROPOSALS_DIR",
        help="The proposals directory, mainly for testing cross-repository. "
        "Automatically found by default.",
    )
    parser.add_argument(
        "--start-point",
        metavar="START_POINT",
        default="trunk",
        type=str,
        help="The starting point for the new branch.",
    )
    parser.add_argument(
        "--dry-run",
        metavar="DRY_RUN",
        default=-1,
        type=int,
        help="Set to a PR# to print but don't execute commands.",
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


def _get_proposals_dir(parsed_args):
    """Returns the path to the proposals directory."""
    if parsed_args.proposals_dir:
        return parsed_args.proposals_dir
    return os.path.realpath(
        os.path.join(os.path.dirname(__file__), "../../proposals")
    )


def _run(argv, dry_run, check=True, get_stdout=False):
    """Runs a command."""
    cmd = " ".join([shlex.quote(x) for x in argv])
    print("\n+ RUNNING: %s" % cmd, file=sys.stderr)

    if dry_run > 0:
        return

    stdout_pipe = None
    if get_stdout:
        stdout_pipe = subprocess.PIPE

    p = subprocess.Popen(argv, stdout=stdout_pipe)
    stdout, _ = p.communicate()
    if get_stdout:
        out = stdout.decode("utf-8")
        print(out, end="")
    if check and p.returncode != 0:
        _exit("ERROR: Command failed: %s" % cmd)
    if get_stdout:
        return out


def _run_pr_create(argv, dry_run):
    """Runs a command and returns the PR#."""
    out = _run(argv, dry_run=dry_run, get_stdout=True)
    if dry_run > 0:
        return dry_run
    match = re.search(
        r"^https://github.com/[^/]+/[^/]+/pull/(\d+)$", out, re.MULTILINE
    )
    if not match:
        _exit("ERROR: Failed to find PR# in output.")
    return int(match[1])


def main():
    parsed_args = _parse_args()
    title = parsed_args.title
    dry_run = parsed_args.dry_run
    branch = _calculate_branch(parsed_args)

    # Verify tools are available.
    gh_bin = _find_tool("gh")
    git_bin = _find_tool("git")
    precommit_bin = _find_tool("pre-commit")

    # Ensure a good working directory.
    proposals_dir = _get_proposals_dir(parsed_args)
    os.chdir(proposals_dir)

    # Verify there are no uncommitted changes.
    if dry_run <= 0:
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
    _run(
        [git_bin, "switch", "--create", branch, parsed_args.start_point],
        dry_run=dry_run,
    )
    _run([git_bin, "push", "-u", "origin", branch], dry_run=dry_run)

    # Copy template.md to a temp file.
    template_path = os.path.join(proposals_dir, "template.md")
    temp_path = os.path.join(proposals_dir, "new-proposal.tmp.md")
    if dry_run > 0:
        _run(["cp", template_path, temp_path], dry_run=dry_run)
    else:
        shutil.copyfile(template_path, temp_path)
    _run([git_bin, "add", temp_path], dry_run=dry_run)
    _run(
        [git_bin, "commit", "-m", "Creating new proposal: %s" % title],
        dry_run=dry_run,
    )

    # Create a PR with WIP+proposal labels.
    _run([git_bin, "push"], dry_run=dry_run)
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
        ],
        dry_run=dry_run,
    )

    # Add links.
    _run(
        [
            gh_bin,
            "pr",
            "comment",
            str(pr_num),
            "--repo",
            "carbon-language/carbon-lang",
            "--body",
            _LINK_TEMPLATE % pr_num,
        ],
        dry_run=dry_run,
    )

    # Remove the temp file, create p####.md, and fill in PR information.
    if dry_run > 0:
        _run(["rm", temp_path], dry_run=dry_run)
    else:
        os.remove(temp_path)
    final_path = os.path.join(proposals_dir, "p%04d.md" % pr_num)
    content = _fill_template(template_path, title, pr_num)
    if dry_run > 0:
        _run(["cat", content, ">", final_path], dry_run=dry_run)
    else:
        with open(final_path, "w") as final_file:
            final_file.write(content)
    _run([git_bin, "add", temp_path, final_path], dry_run=dry_run)
    # Needs a ToC update.
    _run([precommit_bin, "run"], dry_run=dry_run, check=False)
    _run(
        [git_bin, "add", final_path, os.path.join(proposals_dir, "README.md")],
        dry_run=dry_run,
    )
    _run(
        [
            git_bin,
            "commit",
            "--amend",
            "-m",
            "Filling out template with PR %d" % pr_num,
        ],
        dry_run=dry_run,
    )

    # Push the PR update.
    _run([git_bin, "push", "--force-with-lease"], dry_run=dry_run)

    print(
        "\nCreated PR %d for %s. Make changes to:\n  %s"
        % (pr_num, title, final_path)
    )


if __name__ == "__main__":
    main()
