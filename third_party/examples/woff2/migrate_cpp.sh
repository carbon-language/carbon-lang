#!/bin/bash -eux
#
# Part of the Carbon Language project, under the Apache License v2.0 with LLVM
# Exceptions. See /LICENSE for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# Runs an example migration of woff2 C++ code.

# cd to the carbon-lang root.
cd "$(dirname "$0")/../../.."

EXAMPLE="${PWD}/third_party/examples/woff2"

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

# Kludge for adding LLVM include paths into the compile flags.
# TODO: Find better solution.
COMPILE_FLAGS=($(cat "${EXAMPLE}/compile_flags.carbon.txt" | sed 's/"/\\"/g'))
for x in $(
    clang++ -Wp,-v -xc++ -stdlib=libc++ - -fsyntax-only < /dev/null 2>&1 |
    grep /llvm/); do
  COMPILE_FLAGS+=("-isystem")
  COMPILE_FLAGS+=("${x}")
done

# Construct a compilation database for use by run-clang-tidy.py.
COMPDB="${EXAMPLE}/carbon/compile_commands.json"
echo "[" > "${COMPDB}"
for f in $(find "${EXAMPLE}/carbon" -regex ".*\.\(cc\|h\)"); do
  echo "{ \"file\": \"$(realpath --relative-to "${EXAMPLE}/carbon" ${f})\"," >> "${COMPDB}"
  echo "  \"directory\": \"${EXAMPLE}/carbon\"," >> "${COMPDB}"
  echo "  \"arguments\": [" >> "${COMPDB}"
  echo "    \"clang++\"," >> "${COMPDB}"
  for index in ${!COMPILE_FLAGS[@]}; do
    echo "    \"${COMPILE_FLAGS[$index]}\"," >> "${COMPDB}"
  done
  echo "    \"${f}\"" >> "${COMPDB}"
  echo "  ]" >> "${COMPDB}"
  echo "}," >> "${COMPDB}"
done
# Remove the last comma, for JSON syntax correctness.
sed -i '$ s/,$//' "${COMPDB}"
echo "]" >> "${EXAMPLE}/carbon/compile_commands.json"

# Run the migration tool.
bazel build -c opt //migrate_cpp
# Not sure why, but execution of cpp_refactoring fails while saving refactorings
# if not in the directory. Ideally shouldn't be required, passing the path to
# migrate_cpp should work.
pushd "${EXAMPLE}/carbon"
../../../../bazel-bin/migrate_cpp/migrate_cpp .
popd

# Don't save the compile flags.
rm "${EXAMPLE}/carbon/compile_commands.json"
