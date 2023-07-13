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

1. Open utils/vscode folder in VS Code.
2. Launch the extension using Run command (F5).
