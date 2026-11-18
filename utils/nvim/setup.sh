#!/bin/sh

# Part of the Carbon Language project, under the Apache License v2.0 with LLVM
# Exceptions. See /LICENSE for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

set -euo pipefail

if ! command -v tree-sitter >/dev/null 2>&1; then
  echo "tree-sitter command not found" >&2
  echo "please run npm install -g tree-sitter-cli" >&2
  exit 1
fi

DIR="$(dirname -- "$(readlink -f -- "$0")")"
ROOT="$(git -C "$DIR" rev-parse --show-toplevel)"

mkdir -p ~/.config/nvim/{lua,parser,queries}

echo "Linking carbon queries and carbon.lua to neovim's configuration..." >&2
[ -e ~/.config/nvim/queries/carbon ] && unlink ~/.config/nvim/queries/carbon
ln -sf "$ROOT/utils/tree_sitter/queries" ~/.config/nvim/queries/carbon
[ -e ~/.config/nvim/lua/carbon.lua ] && unlink ~/.config/nvim/lua/carbon.lua
ln -sf "$ROOT/utils/nvim/carbon.lua" ~/.config/nvim/lua/carbon.lua

# load carbon.lua on startup
echo "Adding \`require \"carbon\"\` to init.lua..." >&2
grep 'require "carbon"' ~/.config/nvim/init.lua >/dev/null || echo 'require "carbon"' >> ~/.config/nvim/init.lua

# build tree_sitter
echo "Building and copying in tree-sitter binary..." >&2
(cd "$ROOT" && bazel build //utils/tree_sitter:parser_shared -c opt --action_env=PATH)
cp "$ROOT/bazel-bin/utils/tree_sitter/carbon.so" ~/.config/nvim/parser/carbon.so
# bazel builds read-only binaries
chmod +w ~/.config/nvim/parser/carbon.so
