---
name: Bazel usage
description:
    Instructions that **MUST** be followed when using Bazel or Bazelisk to
    build, test, and debug in the Carbon repository.
---

# Bazel usage

<!--
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

This skill documents how best to use Bazel when building, testing, or
manipulating the Carbon repository's Bazel in any way.

## Bazel wrappers

Carbon uses Bazel for its build system. To ensure consistent versions, the
project uses Bazelisk.

> [!IMPORTANT] Always use `bazelisk` whenever you want to run Bazel. Never run
> `bazel` directly in the Carbon project. Anything you want to do with `bazel`
> can be done with the `bazelisk` command instead.

-   **Bazelisk**: Try to use `bazelisk` in your existing `$PATH` if available.
    -   **`run_bazelisk.py`**: If `bazelisk` isn't available, use
        `./scripts/run_bazelisk.py` to run bazelisk without it being installed.

## Essential commands

### Building

-   **Build all**: `bazelisk build //...`
-   **Build toolchain**: `bazelisk build //toolchain/...`
-   **Build specific target**: `bazelisk build //toolchain:carbon`

### Testing

-   **Test all**: `bazelisk test //...:all`
-   **Test toolchain**: `bazelisk test //toolchain/...`
-   **Test examples**: `bazelisk test //examples/...`

> [!TIP] Running all of the tests can be slow, so try to narrowly test the
> immediately relevant parts of the project first, and only expand coverage as
> necessary to be confident in the changes.

> [!TIP] For specialized instructions on testing and developing the Carbon
> toolchain, consult these skills:
>
> -   [Toolchain tests](/.agents/skills/toolchain_tests/SKILL.md): For
>     authoring, structuring, and running `file_test` tests.
> -   [Toolchain development](/.agents/skills/toolchain_development/SKILL.md):
>     For architecture, essential commands, and debugging the toolchain.

### Running binaries built by Bazel

> [!IMPORTANT] Always manually run binaries built by Bazel using the
> `bazelisk run` command. Never run the binary directly from `bazel-bin/`.

You can run the Carbon driver or command line directly via Bazel:

-   `bazelisk run //toolchain -- compile --phase=parse toolchain/parse/testdata/basics/empty.carbon`

## Advanced configurations

### AddressSanitizer (ASan)

To enable ASan for local testing:

-   Pass `--config=asan`: `bazelisk test --config=asan //...`

## Common pitfalls and troubleshooting

### `bazel clean`

Changes to packages installed on your system (like changing LLVM versions or
installing `libc++`) may not be noticed by Bazel.

-   Run `bazelisk clean` to force cached state to be rebuilt when environment
    changes occur.
