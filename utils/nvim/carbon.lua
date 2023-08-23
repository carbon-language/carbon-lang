-- Part of the Carbon Language project, under the Apache License v2.0 with LLVM
-- Exceptions. See /LICENSE for license information.
-- SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

vim.filetype.add({
  extension = {
    carbon = 'carbon',
  },
})

vim.treesitter.language.add("carbon")

-- LSP
local lsp = require 'lspconfig'
local configs = require 'lspconfig.configs'
local util = require 'lspconfig.util'

-- Check if the config is already defined (useful when reloading this file)
if not configs.carbon then
  configs.carbon = {
    default_config = {
      cmd = { "./bazel-bin/language_server/language_server" },
      filetypes = { "carbon" },
      root_dir = util.find_git_ancestor,
    }
  }
end
lsp.carbon.setup {}
