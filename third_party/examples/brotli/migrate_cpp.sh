#!/bin/bash -eux
#
# Part of the Carbon Language project, under the Apache License v2.0 with LLVM
# Exceptions. See /LICENSE for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# Runs an example migration of the Brotli C++ code.

# cd to the carbon-lang root.
cd "$(dirname "$0")/../../.."

EXAMPLE=third_party/examples/brotli

# Remove any previous conversion.
rm -rf "${EXAMPLE}/carbonized/"

# Initialize the converted direction, excluding non-C++ code.
rsync -a \
  "${EXAMPLE}/original/" \
  "${EXAMPLE}/carbonized/" \
  --exclude csharp \
  --exclude go \
  --exclude java \
  --exclude js \
  --exclude python \
  --exclude research
