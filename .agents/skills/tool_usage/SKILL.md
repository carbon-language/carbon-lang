---
name: Tool usage
description:
    Instructions for AI assistants on what tools to use in the carbon-lang
    project.
---

# Tool usage

<!--
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

## Bazelisk and Bazel

We use `bazelisk` for build and test.

**IMPORTANT**: AI assistants use `bazelisk` instead of `bazel`.

## Pre-commit

Running `pre-commit` is mandatory. To run it on all files:

```bash
pre-commit run -a
```

To validate a specific list of files:

```bash
pre-commit run --files <files>
```

## Command line tools restrictions

AI assistants **MUST NOT** use legacy or generic UNIX shell search/edit commands
when specialized environment tools exist.

-   **DO NOT USE**: `cat`, `less`, `grep`, `sed`, or other shell utilities for
    viewing, searching, or modifying files.
-   **DO NOT USE**: `patch` to write and apply patch files.
-   **DO NOT USE**: Writing custom scripts in other languages to circumvent this
    limitation.
-   **DO USE**: High-fidelity semantic API tools:
    -   Viewing: Use `view_file` instead of `cat` / `less`.
    -   Searching: Use `grep_search` / `find_by_name` instead of `grep` /
        `find`.
    -   Modifying: Use `replace_file_content`, `multi_replace_file_content`, or
        `write_to_file` instead of `sed` / `patch` / `python` edits.

You may only write and run temporary programs to modify source code if no
semantic tool is applicable or when performing complex, systematic transforms
across many codebase directories simultaneously.

## Temporary files management

Temporary files and scratchpad test scripts created by the assistant during
analysis, experiments, or debugging:

-   **MUST** reside within the `tmp/` subdirectory under the workspace root.
-   **MUST** be periodically cleaned out and deleted before ending your turn to
    preserve a clean git workspace.
