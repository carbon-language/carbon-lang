# Tree-sitter grammar for Carbon

<!--
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

Tree-sitter is currently used for syntax highlighting in supported editors.

## Development

> TODO: Ideally, we should compile the grammar as part of `bazel build`.
> However, we want to avoid requiring `tree-sitter` being pre-installed, so that
> may involve substantial work. Instead, we commit the generated files.

To install tree-sitter, run:

```
npm install -g tree-sitter-cli
```

Then, after modifying `grammar.js`, manual updates are required:

```
tree-sitter generate
```

This will autogenerate files for commit:

-   `src/parser.c`
-   `src/tree_sitter/parser.h`

## Editor Installation

### Helix

1. Install
   [tree-sitter](https://tree-sitter.github.io/tree-sitter/creating-parsers#installation)
   and Nodejs.
2. Install [Helix](https://docs.helix-editor.com/install.html).
3. Run `./helix.sh`

### Emacs

TODO
