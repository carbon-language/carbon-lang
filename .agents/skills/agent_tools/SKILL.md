---
name: Agent tools
description:
    Guidelines and restrictions on shell commands and file manipulation tools
    for AI assistants.
---

# Agent tools

<!--
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

AI assistants working on the Carbon repository **MUST NOT** use legacy or
generic UNIX shell search/edit commands when specialized environment tools
exist.

## Command line tools restrictions

-   **DO NOT USE**: `cat`, `less`, `grep`, `sed`, or other shell utilities for
    viewing, searching, or modifying files.
-   **DO NOT USE**: `patch` to write and apply patch files.
-   **DO NOT USE**: Writing custom scripts in other languages to circumvent this
    limitation.
-   **DO USE**: High-fidelity semantic API tools:
    -   **Viewing**: Use `view_file` instead of `cat` / `less`.
    -   **Searching**: Use `grep_search` / `find_by_name` instead of `grep` /
        `find`.
    -   **Modifying**: Use `replace_file_content`, `multi_replace_file_content`,
        or `write_to_file` instead of `sed` / `patch` / `python` edits.

You may only write and run temporary programs to modify source code if no
semantic tool is applicable or when performing complex, systematic transforms
across many codebase directories simultaneously.

## Temporary files management

Temporary files and scratchpad test scripts created by the assistant during
analysis, experiments, or debugging:

-   **MUST** reside within the `tmp/` subdirectory under the workspace root.
-   **MUST** be periodically cleaned out and deleted before ending your turn to
    preserve a clean git workspace.
