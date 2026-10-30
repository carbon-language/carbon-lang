# Tree-sitter grammar for Carbon

<!--
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

Tree-sitter is currently used for syntax highlighting in supported editors.


## Development

tree-sitter is only built by upstream for Windows, macOS, and Linux, so if
you're using a BSD you cannot currently build the tree-sitter stuff. If you
are so inclined, it is very reasonable that one could build tree-sitter in
our bazel project, no one has yet done that work.

```bash
bazel build //utils/tree_sitter:parser_shared
bazel test //utils/tree_sitter:string_tests
```

## Editor Installation

### Helix

1.  Install
    [tree-sitter](https://tree-sitter.github.io/tree-sitter/creating-parsers#installation)
    and Nodejs.
2.  Install [Helix](https://docs.helix-editor.com/install.html).
3.  Run `./helix.sh`

### Emacs

TODO
