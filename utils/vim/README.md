# Carbon Syntax Highlighting for Vim & Neovim

<!--
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

For Carbon developers using Vim or Neovim, this plugin provides syntax
highlighting for .carbon files found throughout `explorer/testdata`

## Repository and contributing

This code is developed as
[part](https://github.com/carbon-language/carbon-lang/tree/trunk/utils/vim) of
the [Carbon Language](https://github.com/carbon-language/carbon-lang) project.
Everything is then automatically mirrored into a dedicated
[repository](https://github.com/carbon-language/vim-carbon-lang).

If you would like to contribute, please follow the normal
[Carbon contributing guide](https://github.com/carbon-language/carbon-lang/blob/trunk/CONTRIBUTING.md)
and submit pull requests to the main repository.

## Manual Installation

### Vim Users

From the current directory `utils/vim`, please run the following commands to
install the syntax file.

```
mkdir -p ~/.vim/syntax && cp syntax/carbon.vim ~/.vim/syntax/
mkdir -p ~/.vim/ftdetect && cp ftdetect/carbon.vim ~/.vim/ftdetect/
```

### Neovim Users

Instead of copying to the `~/.vim` directory, please use the `~/.config/nvim`
directory, or your custom Neovim root directory.

```
mkdir -p ~/.config/nvim/syntax && cp syntax/carbon.vim ~/.config/nvim/syntax/
mkdir -p ~/.config/nvim/ftdetect && cp ftdetect/carbon.vim ~/.config/nvim/ftdetect/
```
