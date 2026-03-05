# Gemini & AI Assistant Guide for Carbon

<!--
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

This document provides high-density technical context for AI assistants (and
humans!) contributing to the Carbon Language project. If you are an AI
assistant, **read this first** to avoid common pitfalls.

## Table of Contents

-   [Project Structure](#project-structure)
-   [Building and Testing](#building-and-testing)
-   [Debugging and Diagnostics](#debugging-and-diagnostics)
-   [C++ Coding Patterns](#c-coding-patterns)
-   [Common Pitfalls](#common-pitfalls)

## Project Structure

-   **`toolchain/`**: The C++ implementation of the compiler (Toolchain).
    -   `toolchain/check/`: Semantic analysis (SemIR generation).
    -   `toolchain/parse/`: Parsing (Token -> Parse Tree).
    -   `toolchain/lex/`: Lexing (Source -> Tokens).
    -   `toolchain/sem_ir/`: Semantic Intermediate Representation (SemIR)
        definitions.
    -   `toolchain/lower/`: Lowering to LLVM IR.
-   **`proposals/`**: Evolution proposals.

> **Note**: The **`explorer`** codebase (a prototype interpreter) has been moved
> to its own repository. You may see references to it in old proposals or
> documentation, but it is not part of the active `toolchain` development in
> this repository.

## Building and Testing

We use [Bazel](https://bazel.build/).

### Essential Commands

-   **Test everything**: `bazel test //...`
-   **Test specific target**: `bazel test //toolchain/check:check_test`
-   **Build toolchain**: `bazel build //toolchain/...`

### Updating Test Data

Carbon tests often use `file_test` (for example,
`//toolchain/testing/file_test`). If you change compiler behavior, you likely
need to update expected test outputs. **Do not manually edit thousands of lines
of expected output.** Use the script:

```bash
./toolchain/autoupdate_testdata.py
# Or for a specific file:
./toolchain/autoupdate_testdata.py toolchain/check/testdata/my_test.carbon
```

### Pre-commit

Running `pre-commit` is mandatory.

```bash
pre-commit run -a
```

## Debugging and Diagnostics

-   **Printing to stderr**: Use `llvm::errs() << "debug info\n";` or
    `std::cerr`.
    -   Avoid `std::cout` (it may interfere with tool output).
-   **SemIR Stringification**:
    -   SemIR objects often have a `Print` method or `operator<<`.
    -   `inst.Print(llvm::errs())`
-   **Debugging Crashes**:
    -   Bazel sandboxing can hide artifacts. Use `--sandbox_debug` if needed,
        but often running the binary directly from `bazel-bin/` is easier for
        debugging.

## C++ Coding Patterns

Carbon's toolchain uses LLVM-style C++ with some specific conventions.

### Error Handling

-   **No Exceptions**: We do not use C++ exceptions.
-   **`ErrorOr<T>`**: Return `ErrorOr<T>` for fallible operations.
    -   Check with `if (auto result = Function(); result) { Use(*result); }`
-   **`llvm::Expected<T>`**: Similar to `ErrorOr`, used when interfacing with
    LLVM.

### Casting (LLVM Style)

-   Use `llvm::cast<T>(obj)` (checked, asserts on failure).
-   Use `llvm::dyn_cast<T>(obj)` (returns null on failure).
-   Use `llvm::isa<T>(obj)` (boolean check).
-   **Avoid** `dynamic_cast` and standard RTTI.

### Data Structures

-   Prefer LLVM ADTs: `llvm::SmallVector`, `llvm::StringRef`, `llvm::DenseMap`.
-   `StringRef` is a view; be careful with lifetimes.

## Common Pitfalls

1.  **Legacy `explorer` references**: The `explorer` prototype has been moved.
    Ignore references to it in proposals or old docs; focus on `toolchain`.
2.  **Manually updating test files**: Always check if `autoupdate_testdata.py`
    can do it for you.
3.  **Using `std::string` unnecessarily**: Prefer `llvm::StringRef` for
    arguments.
4.  **Header Includes**: We use specific include orders (often enforced by
    `clang-format`).
