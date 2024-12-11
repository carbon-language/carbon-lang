# Carbon Language

<!--
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

This extension provides support for the
[Carbon Language](https://github.com/carbon-language/carbon-lang).

This extension is currently experimental, and being developed alongside Carbon.

## Quickstart

1. Install `carbon` to a local path.
    - The extension will look in `bazel-bin` if by default, for the output of
      `bazel build //toolchain`. However, it will be more reliable to take a
      [release](https://github.com/carbon-language/carbon-lang/releases) and put
      it in a directory.
2. Install the
   [Carbon Language extension](https://marketplace.visualstudio.com/items?itemName=carbon-lang.carbon-vscode).
3. Configure the installed path to `carbon`.

## Configuration

The configuration is under `carbon-vscode.*`. At present, the only configuration
is the path to `carbon`. This looks like:

```
"carbon.carbonPath": "/path/to/carbon"
```

## Communication

See Carbon's
[collaboration systems](https://github.com/carbon-language/carbon-lang/blob/trunk/CONTRIBUTING.md#collaboration-systems).
Asking questions on
[GitHub Discussions](https://github.com/carbon-language/carbon-lang/discussions)
will work, but we're most active on [Discord](https://discord.gg/ZjVdShJDAs) and
have the #editor-integrations channel there.

## Documentation

Carbon currently only has project-level documentation. See the
[GitHub repository](https://github.com/carbon-language/carbon-lang).
