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

import subprocess

import scripts_utils


def _build_generated_files(bazel: str) -> None:
    print("Building the generated files so that tools can find them...")

    # Collect the generated file labels. Include some rules which generate
    # files but aren't classified as "generated file".
    kinds_query = (
        "filter("
        ' ".*\\.(h|cpp|cc|c|cxx|def|inc)$",'
        ' kind("generated file", deps(//...))'
        ")"
        " union "
        'kind("cc_proto_library", deps(//...))'
    )
    generated_file_labels = subprocess.check_output(
        [bazel, "query", "--keep_going", "--output=label", kinds_query],
        stderr=subprocess.DEVNULL,
        encoding="utf-8",
    ).splitlines()
    print(f"Found {len(generated_file_labels)} generated files...")

    # Directly build these labels so that indexing can find them. Allow this to
    # fail in case there are build errors in the client, and just warn the user
    # that they may be missing generated files.
    subprocess.check_call(
        [bazel, "build", "--keep_going"] + generated_file_labels
    )


def main() -> None:
    scripts_utils.chdir_repo_root()
    bazel = scripts_utils.locate_bazel()

    _build_generated_files(bazel)

    print("Generating compile_commands.json (may take a few minutes)...")
    subprocess.run([bazel, "run", "@hedron_compile_commands//:refresh_all"])


if __name__ == "__main__":
    main()
