---
name: Code style
description:
    Instructions for code formatting and style guidelines in the Carbon
    toolchain.
---

# Code style

<!--
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

## License

-   **Licenses**: All Carbon files outside of `third_party/` should have a
    license following
    [CONTRIBUTING license instructions](/CONTRIBUTING.md#license).

## Formatting

-   **C++**: Always check `clang-format` on C++ files.
-   **Carbon**: The toolchain's `format` command doesn't work well right now.
    Instead, try to format Carbon code based on other Carbon files and the C++
    style.
-   **Markdown**: Use `pre-commit run prettier --files <file.md>` to format
    markdown files correctly.

## Style Guides

Carbon's toolchain uses LLVM-style C++ with some specific conventions.

-   **Style Guide**: Follow the
    [Carbon C++ Project Style Guide](/docs/project/cpp_style_guide.md).
-   **Markdown style**: Follow the
    [Google developer documentation style guide](https://developers.google.com/style).
