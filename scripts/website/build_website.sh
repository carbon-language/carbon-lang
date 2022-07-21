#!/bin/bash -eu
#
# Part of the Carbon Language project, under the Apache License v2.0 with LLVM
# Exceptions. See /LICENSE for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# Converts markdown files under docs/ into HTML under docs/dist/. Run in the repository root directory.
# Serve the built docs locally using:
# python3 -m http.server --bind 127.0.0.1 --directory docs/dist

cd docs

# shellcheck disable=SC2044
for markdown_file in $(find . -name "*.md"); do
    html_file="dist/${markdown_file%.*}.html"
    html_file="${html_file/README.html/index.html}"
    mkdir -p "$(dirname "$html_file")"
    pandoc --from=gfm --to=html5 --lua-filter=../scripts/website/convert_md_links.lua "$markdown_file" -o "$html_file"
done
