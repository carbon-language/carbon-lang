# Extension development

<!--
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

Currently only contains basic syntax highlighting.

## Releases

This assumes NodeJS is installed, along with `vsce` (using
`npm install -g vsce`).

1.  `npm install && vsce publish`

## Local installation

This assumes NodeJS is installed, along with `vsce` (using
`npm install -g vsce`).

1.  `npm install && vsce package -o carbon.vsix && realpath carbon.vsix`
    -   This installs dependencies, builds the VSIX file, and prints the path
        for installation.
2.  Install the plugin:
    -   If you're using VS Code locally, run
        `npm install && vsce package -o carbon.vsix && code --install-extension carbon.vsix`
    -   If you're using VS Code's remote mode:
        1.  In vscode, open the
            [command palette](https://code.visualstudio.com/docs/getstarted/userinterface#_command-palette)
            and select "Extensions: Install from VSIX...".
        2.  Enter the path printed by the above command.

## Development

1.  `bazel build //toolchain` in project root.
2.  Open utils/vscode folder in VS Code.
3.  Launch the extension using Run command (F5).
4.  In the opened window, open the carbon-lang repository as folder.
5.  Open a carbon file.
6.  Open code outline (Ctrl+Shift+O).

## Debugging output

1.  Go to the "Output" panel.
2.  In the top right, there is a dropdown; select "Carbon Language Server".

## Updating dependencies

To update dependencies, run `npm update`.
