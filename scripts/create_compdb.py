#!/usr/bin/env python3

"""Create a compilation database for Clang tools like `clangd`.

If you want `clangd` to be able to index this project, run this script from
the workspace root to generate a rich compilation database. After the first
run, you should only need to run it if you encounter `clangd` problems, or if
you want `clangd` to build an up-to-date index of the entire project. Note
that in the latter case you may need to manually clear and rebuild clangd's
index after running this script.

Note that this script will build generated files in the Carbon project and
otherwise touch the Bazel build. It works to do the minimum amount necessary.
Once setup, generally subsequent builds, even of small parts of the project,
different configurations, or that hit errors won't disrupt things. But, if
you do hit errors, you can get things back to a good state by fixing the
build of generated files and re-running this script.
"""

__copyright__ = """
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""

import argparse
import json
import subprocess
import sys
from typing import Any, Dict

import scripts_utils


def _build_generated_files(
    bazel: str, logtostderr: bool, dump_files: bool
) -> None:
    print("Building the generated files so that tools can find them...")

    # Collect the generated file labels. Include some rules which generate
    # files but aren't classified as "generated file".
    kinds_query = (
        "filter("
        ' ".*\\.(h|hpp|hxx|cpp|cc|c|cxx|def|inc|s|S)$",'
        ' kind("(.*generate.*|manifest_as_cpp)",'
        # tree_sitter is excluded here because it causes the query to failure on
        # `@platforms`.
        "      deps(//... except //utils/tree_sitter/...))"
        ")"
    )
    log_to = None
    if not logtostderr:
        log_to = subprocess.DEVNULL
    generated_file_labels = subprocess.check_output(
        [bazel, "query", "--keep_going", "--output=label", kinds_query],
        stderr=log_to,
        encoding="utf-8",
    ).splitlines()
    if dump_files:
        for f in sorted(generated_file_labels):
            print(f)
        sys.exit(0)
    print(f"Found {len(generated_file_labels)} generated files...", flush=True)

    # Directly build these labels so that indexing can find them. Allow this to
    # fail in case there are build errors in the client, and just warn the user
    # that they may be missing generated files.
    subprocess.check_call(
        [bazel, "build", "--keep_going", "--remote_download_outputs=toplevel"]
        + generated_file_labels
        # We also need the Bazel C++ runfiles that aren't "generated", but are
        # not linked into place until built.
        + ["@bazel_tools//tools/cpp/runfiles:runfiles"]
    )


def _get_config_for_entry(entry: Dict[str, Any]) -> str:
    """Returns the configuration for a compile command entry."""
    arguments = entry.get("arguments")

    # Only handle files where the object file argument is easily found as
    # the last argument, which matches the expected structure from Bazel.
    if not arguments or len(arguments) < 2 or arguments[-2] != "-o":
        return "unknown"
    obj_file = arguments[-1]

    # The configuration is the name of the subdirectory of `bazel-out`.
    if not obj_file.startswith("bazel-out/"):
        return "unknown"
    return str(obj_file.split("/")[1])


def _filter_compilation_database(file_path: str) -> None:
    """Filters out duplicate exec-config entries from the database."""
    print("Filtering out duplicate exec-configuration entries...")
    try:
        with open(file_path, "r") as f:
            commands = json.load(f)
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
        sys.exit(1)
    except json.JSONDecodeError:
        print(f"Error: The file '{file_path}' is not a valid JSON file.")
        sys.exit(1)

    # We want to skip compiles that were in the "exec" configuration for tools.
    # Because we generate compile commands for every bazel cc_* target in the
    # main configuration, even if only used by tools, their sources should be
    # covered and the exec configuration would simply be a duplicate.
    #
    # Detecting this based on the `-exec-` string in the configuration name of
    # the directory is a bit of a hack, but even using the `--notool_deps`
    # argument, Bazel seems to sometimes include this configuration in the query
    # that produces the compilation database.
    filtered_commands = [
        entry
        for entry in commands
        if "-exec-" not in _get_config_for_entry(entry)
    ]

    with open(file_path, "w") as f:
        # Use indent=4 for a human-readable, pretty-printed output file
        json.dump(filtered_commands, f, indent=4)
    print(
        "Filtered out "
        f"{len(commands) - len(filtered_commands)} "
        "duplicate entries..."
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        allow_abbrev=False,
    )
    parser.add_argument(
        "--alsologtostderr",
        action="store_true",
        help="Prints subcommand errors to stderr (default: False)",
    )
    parser.add_argument(
        "--dump-files",
        action="store_true",
        help="Dumps the full list of generated files (default: False)",
    )

    args = parser.parse_args()
    scripts_utils.chdir_repo_root()
    bazel = scripts_utils.locate_bazel()

    _build_generated_files(bazel, args.alsologtostderr, args.dump_files)

    print(
        "Generating compile_commands.json (may take a few minutes)...",
        flush=True,
    )
    subprocess.run(
        [
            bazel,
            "run",
            "@hedron_compile_commands//:refresh_all",
            "--",
            "--notool_deps",
        ]
    )

    _filter_compilation_database("compile_commands.json")


if __name__ == "__main__":
    main()
