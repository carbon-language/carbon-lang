#!/bin/bash -eux
#
# Part of the Carbon Language project, under the Apache License v2.0 with LLVM
# Exceptions. See /LICENSE for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# Runs an example migration of woff2 C++ code.

# cd to the carbon-lang root.
cd "$(dirname "$0")/../../.."

EXAMPLE=third_party/examples/woff2

# Remove any previous conversion. Each time this is run, it should demonstrate
# on a fresh copy of woff2.
rm -rf "${EXAMPLE}/carbon/"

# Initialize the carbon directory with C++ code only.
mkdir -p "${EXAMPLE}/carbon/"
for x in LICENSE include src; do
  cp -R "${EXAMPLE}/original/${x}" "${EXAMPLE}/carbon/${x}"
done

# Copy files into the carbon directory to simplify the setup.
cp "${EXAMPLE}/BUILD.original" \
  "${EXAMPLE}/carbon/BUILD"
cp "${EXAMPLE}/WORKSPACE.original" \
  "${EXAMPLE}/carbon/WORKSPACE"
cp "${EXAMPLE}/compile_flags.carbon.txt" \
  "${EXAMPLE}/carbon/compile_flags.txt"

# Kludge for adding LLVM include paths into the compile flags.
# TODO: Find better solution.
for x in $(
    clang++ -Wp,-v -xc++ -stdlib=libc++ - -fsyntax-only < /dev/null 2>&1 |
    grep /llvm/); do
  echo "-isystem" >> "${EXAMPLE}/carbon/compile_flags.txt"
  echo "${x}" >> "${EXAMPLE}/carbon/compile_flags.txt"
done

# Run the migration tool.
bazel build -c opt //migrate_cpp
# Not sure why, but execution of cpp_refactoring fails while saving refactorings
# if not in the directory. Ideally shouldn't be required, passing the path to
# migrate_cpp should work.
#pushd "${EXAMPLE}/carbon"
#../../../../bazel-bin/migrate_cpp/migrate_cpp .
pushd "${EXAMPLE}/carbon/include"
../../../../../bazel-bin/migrate_cpp/migrate_cpp .
popd

# Don't save the compile flags.
rm "${EXAMPLE}/carbon/compile_flags.txt"
