# Extension development

<!--
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

## Tool setup

NodeJS is required to build the extension. You will also need to install `vsce`:

```
npm install -g vsce
```

## Common operations

-   Build and install:

    -   Locally:

        ```
        npm install && vsce package -o carbon.vsix && code --install-extension carbon.vsix
        ```

    -   From a remote SSH host using VS Code Server:

        ```
        npm install && vsce package -o carbon.vsix && ~/.vscode-server/cli/servers/Stable-*/server/bin/code-server --install-extension carbon.vsix
        ```

    -   Using the UI:

        1. `npm install && vsce package -o carbon.vsix && realpath carbon.vsix`
            - This installs dependencies, builds the VSIX file, and prints the
              path.
        2. Open the
           [command palette](https://code.visualstudio.com/docs/getstarted/userinterface#_command-palette)
           and select "Extensions: Install from VSIX...".
        3. Enter the path printed by the above command.

-   Build and publish the release using the website:

    1. `npm install && vsce package -o carbon.vsix && realpath carbon.vsix`
    2. Go to https://marketplace.visualstudio.com/manage/publishers/carbon-lang
    3. Next to the extension name, click the "..." and select "Update".
    4. Select the `carbon.vsix` file.

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
