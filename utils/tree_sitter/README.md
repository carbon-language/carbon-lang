# Tree-sitter grammar for Carbon

<!--
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

Tree-sitter is currently used for syntax highlighting in supported editors.

## Development

We use a non-hermetic tree-sitter invocation, so it must be installed locally.
To install tree-sitter, run:

```
npm install -g tree-sitter-cli
```

### Building and Testing

To build and test changes to the grammar using Bazel, provide specific flags to
allow the build to access the system `tree-sitter` binary and environment.

```bash
bazel test //utils/tree_sitter:string_tests \
  --strategy=Genrule=local \
  --action_env=PATH \
  --action_env=HOME
```

-   `--strategy=Genrule=local`: Disables sandboxing for the genrule that runs
    `tree-sitter generate`, allowing it to find the system-installed binary.
-   `--action_env=PATH`: Passes your current `PATH` to the build actions,
    ensuring `tree-sitter` can be found. If `tree-sitter` is not in your default
    PATH, you can specify it explicitly, for example:
    `--action_env=PATH=/path/to/tree-sitter/bin:$PATH`.
-   `--action_env=HOME`: Passes your `HOME` environment variable, which
    `tree-sitter` may need to locate its configuration or cache.

## Editor Installation

### Helix

1. Install
   [tree-sitter](https://tree-sitter.github.io/tree-sitter/creating-parsers#installation)
   and Nodejs.
2. Install [Helix](https://docs.helix-editor.com/install.html).
3. Run `./helix.sh`

### Emacs

TODO
