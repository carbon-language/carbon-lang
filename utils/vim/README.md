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

## Installation

In case you don't have a preferred plugin manager, and you're using `Vim 8.0` version or 
higher, consider use its built-in package management:

```bash
# See ':help packages' to more information
$ git clone https://github.com/carbon-language/vim-carbon-lang ~/.vim/pack/vendor/start/vim-carbon-lang
```

Otherwise, basic usage examples for some popular ones are listing below:

- [**Vim Plug**](https://github.com/junegunn/vim-plug)
```vim
call plug#begin()
    Plug 'carbon-language/vim-carbon-lang'
call plug#end()
```

- [**Packer**](https://github.com/wbthomason/packer.nvim) (Only Neovim)
```lua
return require('packer').startup(function(use)
    use 'carbon-language/vim-carbon-lang'
end)
```
