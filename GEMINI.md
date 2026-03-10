# Gemini & AI assistant guide for Carbon

<!--
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

This document provides high-density technical context for AI assistants (and
humans!) contributing to the Carbon Language project. If you are an AI
assistant, **read this first** to avoid common pitfalls.

## Table of contents

-   [General instructions](#general-instructions)
-   [Project structure](#project-structure)
-   [Toolchain development](#toolchain-development)

## General instructions

-   **Communication**: Be concise, professional, and technical. Use GitHub-style
    markdown.
-   **Verification**: Always run relevant tests.
-   **Tool usage**: Use web search for any research outside the immediate
    codebase or KIs.

## Project structure

-   **[`common/`](common/)**: Common C++ utilities used across the project.
-   **[`core/`](core/)**: The Carbon standard library (Core).
-   **[`docs/`](docs/)**: Project documentation, design, and style guides.
-   **[`examples/`](examples/)**: Example Carbon programs and code snippets.
-   **[`proposals/`](proposals/)**: Evolution proposals.
-   **[`testing/`](testing/)**: Testing utilities and infrastructure.
-   **[`toolchain/`](toolchain/)**: The C++ implementation of the compiler
    (Toolchain).

## Tool usage

See the "Tool usage" skill for instructions on what tools to use in the
carbon-lang project.

## Code style

See the "Code style" skill for instructions on formatting, style guides, and
code conventions to follow.

## Toolchain development

See the "Toolchain Development" skill for instructions on architecture,
building, testing, debugging, C++ patterns, and common pitfalls.
