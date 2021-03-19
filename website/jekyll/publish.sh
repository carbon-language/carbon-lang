#!/bin/bash -eu
#
# Part of the Carbon Language project, under the Apache License v2.0 with LLVM
# Exceptions. See /LICENSE for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# Syntax:
#   bazel run :publish

if [[ $# != 0 ]]; then
  echo "Expected 0 arguments, got $#" >&2
  exit 1
fi

# Make a temp directory for the site.
export TMP_SITE="$(mktemp -d)"
function cleanup {
  echo "Removing ${TMP_SITE}"
  rm -rf "${TMP_SITE}"
}
trap cleanup EXIT

# Extract the built site.
tar -xof website/jekyll/jekyll_site.tgz -C "${TMP_SITE}"

gsutil -m -u carbon-language cp -R \
  "${TMP_SITE}"/jekyll_site/* gs://www.carbon-lang.dev/
