-- Part of the Carbon Language project, under the Apache License v2.0 with LLVM
-- Exceptions. See /LICENSE for license information.
-- SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

vim.filetype.add({
  extension = {
    carbon = 'carbon',
  },
})
vim.treesitter.language.add("carbon")
vim.api.nvim_create_autocmd('FileType', {
  pattern = { 'carbon' },
  callback = function() vim.treesitter.start() end,
})

-- LSP
function find_git_ancestor(startpath)
  return vim.fs.dirname(vim.fs.find('.git', { path = startpath, upward = true })[1])
end
local root = find_git_ancestor(vim.env.PWD);

vim.lsp.config("carbon", {
  cmd = { root .. "/bazel-bin/toolchain/carbon", "language-server" },
  filetypes = { "carbon" },
})
vim.lsp.enable("carbon")
