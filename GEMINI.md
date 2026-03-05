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
-   [Toolchain architecture](#toolchain-architecture)
-   [Building and testing](#building-and-testing)
-   [Debugging and diagnostics](#debugging-and-diagnostics)
-   [C++ coding patterns](#c-coding-patterns)
-   [Common pitfalls](#common-pitfalls)

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
    -   [`base/`](toolchain/base/): Base infrastructure and common utilities.
    -   [`check/`](toolchain/check/): Semantic analysis (SemIR generation).
    -   [`lex/`](toolchain/lex/): Lexing (Source -> Tokens).
    -   [`lower/`](toolchain/lower/): Lowering to LLVM IR.
    -   [`parse/`](toolchain/parse/): Parsing (Token -> Parse Tree).
    -   [`sem_ir/`](toolchain/sem_ir/): Semantic Intermediate Representation
        (SemIR) definitions.

## Toolchain architecture

-   **Documentation**: Refer to [`toolchain/docs`](toolchain/docs) for detailed
    architecture design and patterns.
    -   Refer to [Toolchain Idioms](toolchain/docs/idioms.md) for a
        comprehensive list of patterns (for example, `ValueStore`, formatting
        `.def` files, struct reflection) used throughout the implementation.
-   **Phases**: Lex -> Parse -> Check -> Lower.
-   **Definitions**: Many kinds (tokens, parse nodes, SemIR instructions) are
    defined in `.def` files and expanded by way of macros.
-   **Handlers**:
    -   Parser: `Handle<StateName>` in `parse/handle_*.cpp`.
    -   Checker: `HandleParseNode` in `check/handle_*.cpp`.
    -   Lowering: `HandleInst` in `lower/handle_*.cpp`.
-   **Iteration**: Prefer iterative algorithms over recursive ones to prevent
    stack exhaustion on complex codebases.

## Building and testing

AI assistants should use `bazelisk` instead of `bazel` for build and test
commands, because some AI editors won't see `bazel` aliases.

### Essential commands

-   **Test everything**: `bazelisk test //...`
-   **Test specific target**: `bazelisk test //toolchain/testing:file_test`
-   **Test specific file**:
    `bazelisk test //toolchain/testing:file_test --test_arg=--file_tests=<path_to_carbon_file>`
-   **Build toolchain**: `bazelisk build //toolchain/...`

### Updating test data

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

Running `pre-commit` is mandatory. To run it on all files:

```bash
pre-commit run -a
```

To validate a specific list of files:

```bash
pre-commit run --files <files>
```

### Formatting

-   **C++**: Always check `clang-format` on C++ files.
-   **Carbon**: The toolchain's `format` command doesn't work well right now.
    Instead, try to format Carbon code based on other Carbon files and the C++
    style.
-   **Markdown**: Use `pre-commit run prettier --files <file.md>` to format
    markdown files correctly.

## Debugging and diagnostics

-   **Printing to stderr**: Use `llvm::errs() << "debug info\n";`.
    -   Avoid `std::cout` (it may interfere with tool output).
-   **SemIR Stringification**:
    -   SemIR objects often have a `Print` method or `operator<<`.
    -   `inst.Print(llvm::errs())`
-   **Debugging Crashes**:
    -   Bazel sandboxing can hide artifacts. Use `--sandbox_debug` if needed,
        but often running the binary directly from `bazel-bin/` is easier for
        debugging.

## C++ coding patterns

Carbon's toolchain uses LLVM-style C++ with some specific conventions.

-   **Style Guide**: Follow the
    [Carbon C++ Project Style Guide](docs/project/cpp_style_guide.md).
-   **Markdown style**: Follow the
    [Google developer documentation style guide](https://developers.google.com/style).

### Error handling

-   **No exceptions**: Do not use C++ exceptions.
-   **`ErrorOr<T>`**: Return `ErrorOr<T>` for fallible operations.
    -   Check with `if (auto result = Function(); result) { Use(*result); }`
-   **`llvm::Expected<T>`**: Similar to `ErrorOr`, used when interfacing with
    LLVM.

### Casting (LLVM style)

-   Use `llvm::cast<T>(obj)` (checked, asserts on failure).
-   Use `llvm::dyn_cast<T>(obj)` (returns null on failure).
-   Use `llvm::isa<T>(obj)` (boolean check).
-   **Avoid** `dynamic_cast` and standard RTTI.

### Data structures

-   Prefer APIs in `common/` and `toolchain/base/` over LLVM ADTs. For example,
    use `Map` instead of `llvm::DenseMap`.
-   If no Carbon API exists, prefer LLVM ADTs over standard library ones (for
    example `llvm::SmallVector`, `llvm::StringRef`).
-   `StringRef` is a view; be careful with lifetimes.

## Common pitfalls

1.  **Legacy `explorer` references**: The `explorer` prototype has been moved.
    Ignore references to it in proposals or old docs; focus on `toolchain`.
2.  **Manually updating test files**: Always check if `autoupdate_testdata.py`
    can do it for you.
3.  **Using `std::string` unnecessarily**: Prefer `llvm::StringRef` for
    arguments.
4.  **Header includes**: Use specific include orders (often enforced by
    `clang-format`).
5.  **Parse node order**: Semantics processes parse nodes in post-order; ensure
    your parser transitions support this.
