#!/bin/bash -eu
#
# Part of the Carbon Language project, under the Apache License v2.0 with LLVM
# Exceptions. See /LICENSE for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# Syntax:
#   bazel run :serve

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

# Copy jekyll configs.
cp website/jekyll/Gemfile website/jekyll/Gemfile.lock "${TMP_SITE}"
sed "s:/tmp/:${TMP_SITE}/:" website/jekyll/_config.yml > \
  "${TMP_SITE}/_config.yml"

# Extract the built site.
tar -xof website/jekyll/jekyll_site.tgz -C "${TMP_SITE}"

# Stub the site dir for jekyll; needed even though this shouldn't rebuild.
mkdir "${TMP_SITE}/site"

pushd "${TMP_SITE}"
# Specify hostname to override the default of 127.0.0.1.
bundle exec jekyll server --host "$(hostname)" --skip-initial-build
popd
