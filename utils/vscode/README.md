<!--
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

# VS Code Extension for Carbon

Currently only contains basic syntax highlighting.

## Installing

1. Install Node JS.
2. To generate VS Code extension file (.vsix).

```shell
npm install && npm run package
```

3. Install the extension

```shell
code --install-extension out/carbon.vsix
```

## Development

1. `bazel build language_server` in project root.
2. Open utils/vscode folder in VS Code.
3. Launch the extension using Run command (F5).
4. In the opened window, open the carbon-lang repository as folder.
5. Open a carbon file.
6. Open code outline (Ctrl+Shift+O).

To update dependencies:

```shell
npm update
```
