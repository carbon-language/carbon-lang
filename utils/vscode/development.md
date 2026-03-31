# Extension development

<!--
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

## Tool setup

NodeJS is required to build the extension. You will also need to install `vsce`
and `ovsx`:

```
npm install -g @vscode/vsce ovsx
```

This installs `vsce` and `ovsx` to `/usr/local/bin`. Ensure that
`/usr/local/bin` is in your `$PATH` environment variable to use it.

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

-   Build and publish the release to the VSCode Marketplace using the website:

    1. `npm install && vsce package -o carbon.vsix && realpath carbon.vsix`
    2. Go to https://marketplace.visualstudio.com/manage/publishers/carbon-lang
        - We use `infra-role@carbon-lang.dev` for publishing; the GitHub account
          `CarbonInfraBot` can also be used for login. Contact leads if you
          require access.
    3. Next to the extension name, click the "..." and select "Update".
    4. Select the `carbon.vsix` file.

-   Build and publish the release to the Open VSX Registry by following the
    [Open VSX documentation for publishing extensions](https://github.com/EclipseFdn/open-vsx.org/wiki/Publishing-Extensions).

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
