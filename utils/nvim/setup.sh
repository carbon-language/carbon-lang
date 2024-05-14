#!/bin/sh

# Part of the Carbon Language project, under the Apache License v2.0 with LLVM
# Exceptions. See /LICENSE for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

set -xeuo pipefail

ROOT="$(git rev-parse --show-toplevel)"

mkdir -p ~/.config/nvim/{lua,parser,queries}

# add highlight queries
ln -sTf "$PWD/utils/treesitter/queries" ~/.config/nvim/queries/carbon

# add carbon.lua
ln -sf "$PWD/utils/nvim/carbon.lua" ~/.config/nvim/lua/carbon.lua

# load carbon.lua on startup
grep 'require "carbon"' ~/.config/nvim/init.lua || echo 'require "carbon"' >> ~/.config/nvim/init.lua

# build treesitter
cd utils/treesitter
tree-sitter generate
clang -o ~/.config/nvim/parser/carbon.so -shared src/parser.c src/scanner.c -I ./src -Os -fPIC
