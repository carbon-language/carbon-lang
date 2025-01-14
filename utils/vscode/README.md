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

1. Download and install a `carbon`
   [release](https://github.com/carbon-language/carbon-lang/releases).
    - By default, the extension will look for `carbon` under `./bazel-bin`. This
      is for developers actively working on Carbon and running VS Code inside a
      [carbon-lang](https://github.com/carbon-language/carbon-lang) clone.
2. Install the
   [Carbon Language extension](https://marketplace.visualstudio.com/items?itemName=carbon-lang.carbon-vscode).
3. Configure the installed path to `carbon`.

## Configuration

The configuration is under `carbon.*`. At present, the only configuration is the
path to the `carbon` binary. This looks like:

```
"carbon.carbonPath": "/path/to/carbon"
```

## Communication

See Carbon's
[collaboration systems](https://github.com/carbon-language/carbon-lang/blob/trunk/CONTRIBUTING.md#collaboration-systems).
We're most active on [Discord](https://discord.gg/ZjVdShJDAs) and have a
#editor-integrations channel. We'll also respond to questions on
[GitHub Discussions](https://github.com/carbon-language/carbon-lang/discussions).

## Documentation

Carbon currently only has project-level documentation. See the
[GitHub repository](https://github.com/carbon-language/carbon-lang).
