#!/usr/bin/env python3

"""Set up workspace for clangd.

If you want clangd to be able to index this project, run this script
from the workspace root to generate the data that clangd needs. After
the first run, you should only need to run it if you encounter clangd
problems, or if you want clangd to build an up-to-date index of the
entire project. Note that in the latter case you may need to manually
clear and rebuild clangd's index after running this script.
"""

__copyright__ = """
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""

import itertools
import json
import os
import re
import subprocess
import sys

# Load compiler flags. We do this first in order to fail fast if not run from
# the workspace root.
try:
    with open("compile_flags.txt") as flag_file:
        command = "clang " + " ".join([line.strip() for line in flag_file])
except FileNotFoundError:
    sys.exit(
        os.path.basename(sys.argv[0]) + " must be run from the project root"
    )

directory = os.getcwd()

print("Building compilation database...")

# Identify all inputs to all C++ compilation actions in this workspace.
# Header files seem to be omitted, presumably because they're not declared
# as inputs in the BUILD file, but we don't need them for this purpose.
actions = subprocess.run(
    ["bazel", "aquery", 'mnemonic("CppCompile", "//...:*")'],
    capture_output=True,
    check=True,
    text=True,
).stdout
input_lists = [
    re.split(r",\s*", input_list)
    for input_list in re.findall(r"Inputs: \[([^\]]*)\]", actions)
]
inputs = set(itertools.chain.from_iterable(input_lists))

# Generate compile_commands.json with an entry for each C++ input.
entries = []
for input in inputs:
    # Bazel considers these to be inputs, but they're not C++ source files.
    if (
        input == "external/bazel_tools/tools/cpp/grep-includes.sh"
        or "_middlemen" in input
    ):
        continue
    # Generated files are located in a bin directory that depends on the
    # target architecture and build mode, e.g. bazel-out/k8-fastbuild/bin.
    # bazel-bin is a symlink to the most recently-written bin directory,
    # which makes it more durable for our purposes.
    input = re.sub(r"bazel-out/[^/]*/bin/", "bazel-bin/", input)
    entries.append(
        {
            "directory": directory,
            "file": input,
            "command": command + " " + os.path.join(directory, input),
        }
    )

with open("compile_commands.json", "w") as json_file:
    json.dump(entries, json_file, indent=2)

print("Building generated C++ files...")

# Identify all generated files that are transitive dependencies of C++ Bazel
# rules in this workspace, and build them, so that they're available to clangd.
generated_files = subprocess.run(
    [
        "bazel",
        "query",
        'kind("generated file", deps(kind("cc_.*", "//...:*")))',
    ],
    capture_output=True,
    check=True,
    text=True,
).stdout.splitlines()
subprocess.run(["bazel", "build"] + generated_files, check=True)
