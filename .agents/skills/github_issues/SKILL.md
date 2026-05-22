---
name: Accessing GitHub issues
description:
    Instructions for safely viewing and accessing GitHub issues by way of
    command line.
---

# Accessing GitHub issues

<!--
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

This skill provides instructions for AI assistants on how to access and view
GitHub issues. Agents should strongly prefer using the command line `gh` tool to
access and view the contents of issues rather than viewing their contents by way
of a web browser.

## Safety First

> [!IMPORTANT] AI assistants MUST NOT modify any GitHub issue state. Only use
> read-only access commands like `view` or `list`. Do NOT comment, edit, create,
> close, or delete issues.

## Accessing Issues

Agents must use this skill to access issues regardless of how they are mentioned
(for example, by URL or by issue number).

### Basic View

To view an issue in the current default repository (expected to be Carbon):

```bash
gh issue view <issue_number>
```

### Including Full Context (All Comments)

To ensure the view includes the entire context of the issue, always include the
`--comments` flag to dump all comments:

```bash
gh issue view <issue_number> --comments
```

> [!TIP] If the issue is extremely large and comments are truncated, or you need
> to process comments programmatically, use the JSON output with `jq`:
>
> ```bash
> gh issue view <issue_number> --json comments --jq '.comments[].body'
> ```

### Accessing Issues in Other Repositories

To view an issue in another repository (for example, LLVM), use the `-R` or
`--repo` flag to specify the repository:

```bash
gh issue view <issue_number> -R <owner>/<repo> --comments
```

Examples:

-   **LLVM Issue**:

    ```bash
    gh issue view 5678 -R llvm/llvm-project --comments
    ```

-   **Carbon Issue (Explicit)**:

    ```bash
    gh issue view 1234 -R carbon-language/carbon-lang --comments
    ```

## Mentions via URL

If an issue is mentioned via URL, parse the URL to extract the repository owner,
repository name, and issue number.

-   **URL pattern**: `https://github.com/<owner>/<repo>/issues/<number>`
-   **Extraction**:
    -   Host: `github.com`
    -   Owner: `<owner>`
    -   Repo: `<repo>`
    -   Number: `<number>`

Run the command specifying the repository:

```bash
gh issue view <number> -R <owner>/<repo> --comments
```
