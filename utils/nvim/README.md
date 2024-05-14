<!--
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

# Neovim plugin for Carbon

Treesitter based syntax highlighting and language server client for Neovim.

This requires neovim >= 0.9 and
[nvim-lspconfig](https://github.com/neovim/nvim-lspconfig) to be installed.

1. Run `bazel build language_server` in project root.
2. Run `utils/nvim/setup.sh`.
3. Start nvim in carbon-lang root folder and open a carbon file.
4. View document symbols. If you have telescope.nvim installed, you can use
   `:Telescope lsp_document_symbols`
