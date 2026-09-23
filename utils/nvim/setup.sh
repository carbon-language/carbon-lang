#!/bin/sh

# Part of the Carbon Language project, under the Apache License v2.0 with LLVM
# Exceptions. See /LICENSE for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

set -euo pipefail
 
DIR="$(dirname -- "$(readlink -f -- "$0")")"
ROOT="$(git -C "$DIR" rev-parse --show-toplevel)"

mkdir -p ~/.config/nvim/{lua,parser,queries}

# add highlight queries
[ -e ~/.config/nvim/queries/carbon ] && unlink ~/.config/nvim/queries/carbon
ln -sf "$ROOT/utils/tree_sitter/queries" ~/.config/nvim/queries/carbon

# add carbon.lua
[ -e ~/.config/nvim/lua/carbon.lua ] && unlink ~/.config/nvim/lua/carbon.lua
ln -sf "$ROOT/utils/nvim/carbon.lua" ~/.config/nvim/lua/carbon.lua

# load carbon.lua on startup
grep 'require "carbon"' ~/.config/nvim/init.lua >/dev/null || echo 'require "carbon"' >> ~/.config/nvim/init.lua

# build tree_sitter
cd utils/tree_sitter
tree-sitter generate
clang -o ~/.config/nvim/parser/carbon.so -shared src/parser.c src/scanner.c -I ./src -Os -fPIC
