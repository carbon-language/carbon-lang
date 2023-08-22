# Developer Utilities

<!--
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

This directory collects tools and utilities that may be useful to both Carbon
developers and developers writing Carbon code.

## Editor Support

-   [VSCode](./vscode/README.md)
-   [Neovim](./nvim/README.md)
-   [Vim](./vim/README.md)
-   [IntelliJ](./textmate/README.md#intellij)
-   [Atom](./textmate/README.md#atom)

### Other Editors

Any editor that supports Language server protocol and/or tree-sitter is
supported. The editor just needs to be configured manually.
`bazel build language_server` produces the language server binary.
`utils/treesitter` contains the treesitter grammar.
