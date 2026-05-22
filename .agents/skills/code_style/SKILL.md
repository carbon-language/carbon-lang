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

-   **Bazel**: Use `pre-commit run buildifier --files <file.bzl>` to format
    Bazel files.
-   **C++**: Use `pre-commit run clang-format --files <file.cpp>` to format C++
    files.
-   **Carbon**: The toolchain's `format` command doesn't work well right now.
    Instead, try to format Carbon code based on other Carbon files and the C++
    style.
-   **Markdown**: Use `pre-commit run prettier --files <file.md>` to format
    markdown files.
-   **Python**: Use `pre-commit run black --files <file.py>` to format Python
    files.

## Style Guides

-   **C++ style**: Follow the
    [Carbon C++ Project Style Guide](/docs/project/cpp_style_guide.md).
-   **Markdown style**: Follow the
    [Google developer documentation style guide](https://developers.google.com/style).
-   **Python style**: Follow the [PEP 8](https://peps.python.org/pep-0008/)
    style guide.
    -   Wrap code and comments to 80 columns.
    -   Run `pre-commit run flake8 --files <file.py>` to check Python style.
