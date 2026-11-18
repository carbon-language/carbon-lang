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

toolchain_path = nil
if root and vim.uv.fs_stat(root .. "/bazel/carbon_rules") then
  -- when we're in the carbon repo, use the toolchain from there
  -- (note that this won't update if one `cd`s into the carbon repo)
  toolchain_path = root .. "/bazel-bin/toolchain/carbon"
elseif vim.fn.executable("carbon") then
  -- otherwise, check if it's on the path
  toolchain_path = "carbon"
end

if toolchain_path then
  vim.lsp.config("carbon", {
    cmd = { toolchain_path, "language-server" },
    filetypes = { "carbon" },
  })
  vim.lsp.enable("carbon")
end
