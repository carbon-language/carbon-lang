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

import json
import re
import subprocess
import sys
from pathlib import Path

import scripts_utils

scripts_utils.chdir_repo_root()
directory = Path.cwd()

# We use the `BAZEL` environment variable if present. If not, then we try to
# use `bazelisk` and then `bazel`.
bazel = scripts_utils.locate_bazel()

# Load compiler flags. We do this first in order to fail fast if not run from
# the workspace root.
print("Reading the arguments to use...")
try:
    with open("compile_flags.txt") as flag_file:
        arguments = [line.strip() for line in flag_file]
except FileNotFoundError:
    sys.exit(Path(sys.argv[0]).name + " must be run from the project root")

# Prepend the `clang` executable path to the arguments that looks into our
# downloaded Clang toolchain.
arguments = ["clang++"] + arguments

print("Building compilation database...")

# Find all of the C++ source files that we expect to compile cleanly as
# stand-alone files. This is a bit simpler than scraping the actual compile
# actions and allows us to directly index header-only libraries easily and
# pro-actively index the specific headers in the project.
source_files_query = subprocess.check_output(
    [
        bazel,
        "query",
        "--keep_going",
        "--output=location",
        # Workaround for https://github.com/bazelbuild/bazel/issues/8900
        "--incompatible_display_source_file_location",
        'filter(".*\\.(h|cpp|cc|c|cxx)$", kind("source file", deps(//...)))',
    ],
    stderr=subprocess.DEVNULL,
    universal_newlines=True,
)
source_files = [
    Path(line.split(":")[0]) for line in source_files_query.splitlines()
]

# Filter into the Carbon source files that we'll find directly in the
# workspace, and LLVM source files that need to be mapped through the merged
# LLVM tree in Bazel's execution root.
carbon_files = [
    f.relative_to(directory)
    for f in source_files
    if f.parts[: len(directory.parts)] == directory.parts
]
llvm_files = [
    Path("bazel-execroot/external").joinpath(
        *f.parts[f.parts.index("llvm-project") :]
    )
    for f in source_files
    if "llvm-project" in f.parts
]
print(
    "Found %d Carbon source files and %d LLVM source files..."
    % (len(carbon_files), len(llvm_files))
)

# Now collect the generated file labels.
# cc_proto_library generates files, but they aren't seen with "generated file".
generated_file_labels = subprocess.check_output(
    [
        bazel,
        "query",
        "--keep_going",
        "--output=label",
        (
            'filter(".*\\.(h|cpp|cc|c|cxx|def|inc)$",'
            'kind("generated file", deps(//...)))'
            " union "
            'kind("cc_proto_library", deps(//...))'
        ),
    ],
    stderr=subprocess.DEVNULL,
    universal_newlines=True,
).splitlines()
print("Found %d generated files..." % (len(generated_file_labels),))

# Directly build these labels so that indexing can find them. Allow this to
# fail in case there are build errors in the client, and just warn the user
# that they may be missing generated files.
print("Building the generated files so that tools can find them...")
subprocess.check_call([bazel, "build", "--keep_going"] + generated_file_labels)

# Also build some specific targets that depend on external packages so those are
# fetched and linked into the Bazel execution root. We try to use cheap files
# where possible, but in some cases need to create a virtual include directory.
subprocess.check_call(
    [
        bazel,
        "build",
        "--keep_going",
        "@llvm-project//llvm:LICENSE.TXT",
        "@com_google_absl//:LICENSE",
        "@com_google_googletest//:LICENSE",
        "@com_googlesource_code_re2//:LICENSE",
        "@com_github_google_benchmark//:benchmark",
        "@com_google_libprotobuf_mutator//:LICENSE",
        "@com_google_protobuf//:any_proto",
    ]
)


# Manually translate the label to a user friendly path into the Bazel output
# symlinks.
def _label_to_path(s: str) -> Path:
    # Map external repositories to their part of the output tree.
    s = re.sub(r"^@([^/]+)//", r"bazel-bin/external/\1/", s)
    # Map this repository to the root of the output tree.
    s = s if not s.startswith("//") else "bazel-bin/" + s[len("//") :]
    # Replace the colon used to mark the package name with a slash.
    s = s.replace(":", "/")
    # Convert to a native path.
    return Path(s)


generated_files = [_label_to_path(label) for label in generated_file_labels]

# Generate compile_commands.json with an entry for each C++ input.
entries = [
    {
        "directory": str(directory),
        "file": str(f),
        "arguments": arguments + [str(f)],
    }
    for f in carbon_files + llvm_files + generated_files
]
with open("compile_commands.json", "w") as json_file:
    json.dump(entries, json_file, indent=2)
