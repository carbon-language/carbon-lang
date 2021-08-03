#!/bin/bash -eux
# Part of the Carbon Language project, under the Apache License v2.0 with LLVM
# Exceptions. See /LICENSE for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Ensure the working directory is the script's directory.
cd "$(dirname "$0")"

# Update the `llvm-bazel` project first by just pulling from its HEAD.
git submodule update --remote llvm-bazel

# Find the current LLVM commit in the `llvm-bazel` project.
llvm_commit=$(cd llvm-bazel; git submodule status third_party/llvm-project | awk '/-[0-9a-f]+ / { print substr($1, 2) }')

# Fetch and checkout this commit of LLVM
cd llvm-project
git fetch
git checkout $llvm_commit
cd ..
