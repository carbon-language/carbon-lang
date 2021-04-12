#!/bin/bash -eux
#
# Part of the Carbon Language project, under the Apache License v2.0 with LLVM
# Exceptions. See /LICENSE for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# Syntax:
#   bazel build :jekyll_site.tgz

GEN_SIDEBAR_HTML="$1"
OUT_TGZ="$2"

if [[ $# != 2 ]]; then
  echo "Expected 2 arguments, got $#" >&2
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

# Create a copy of the site for commands, mixing in generated files.
cp -RL website/jekyll/theme/ "${TMP_SITE}"/site/
mkdir "${TMP_SITE}"/site/licenses
mv "${TMP_SITE}"/site/LICENSE "${TMP_SITE}"/site/licenses/LICENSE_theme.txt
cp website/jekyll/licenses.md "${TMP_SITE}"/site/
cp -RL docs/ "${TMP_SITE}"/site/docs/
cp -RL proposals/ "${TMP_SITE}"/site/proposals/
cp "${GEN_SIDEBAR_HTML}" "${TMP_SITE}"/site/_includes/gen_sidebar.html
# Avoid producing errors when trying to copy a directory.
for f in * .*; do
  if [[ -f "$f" ]]; then
    cp "$f" "${TMP_SITE}/site/"
  fi
done

pushd "${TMP_SITE}"
bundle exec jekyll build --quiet
popd
tar -czf "${OUT_TGZ}" -C "${TMP_SITE}" jekyll_site
