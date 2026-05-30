# Gemini & AI Assistant Guide for Carbon

<!--
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

This document provides high-density technical context for AI assistants
contributing to the Carbon Language project.

## General instructions

-   **Communication**: Be concise, professional, and technical. Use GitHub-style
    markdown.
-   **Verification**: Always run relevant tests.

## Project structure

-   **[`common/`](common/)**: Common C++ utilities used across the project.
-   **[`core/`](core/)**: The Carbon standard library (Core).
-   **[`docs/`](docs/)**: Project documentation, design, and style guides.
-   **[`examples/`](examples/)**: Example Carbon programs and code snippets.
-   **[`proposals/`](proposals/)**: Evolution proposals.
-   **[`testing/`](testing/)**: Testing utilities and infrastructure.
-   **[`toolchain/`](toolchain/)**: The C++ implementation of the compiler
    (Toolchain).

## Bazel usage

> [!IMPORTANT] Always use `bazelisk` instead of `bazel` for all commands in the
> Carbon project. Refer to the
> [Bazel usage skill](/.agents/skills/bazel/SKILL.md) for detailed instructions.
