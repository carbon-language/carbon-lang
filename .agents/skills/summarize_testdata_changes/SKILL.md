<!--
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

---

name: Summarize testdata changes description: Instructions for summarizing
changes to Carbon testdata files (`toolchain/*/testdata`).

---

# Summarize testdata changes

This skill provides instructions for creating a comprehensive report summarizing
changes to Carbon testdata files (`toolchain/*/testdata`) and associating them
with related code changes.

## Goals

Produce a report that:

1.  Summarizes code changes outside of testdata.
2.  Groups similar testdata changes together, listing all affected files for
    each group. **Every change to testdata must be represented by at least one
    group. This includes changes to CHECK lines.**
3.  Provides detailed breakdowns of test input changes and diagnostic output
    changes in the corresponding group. **Every single change to inputs or to
    STDERR checks must be explicitly mentioned in the group, with either an
    inline diff or a link to the file.**

## Process

### 1. Identify Changes

Use your VCS (Git or Jujutsu) or query Github to identify changes. For large
changes, it is recommended to use the included helper script to extract test
input changes.

#### For Git Users:

-   **Summarize code changes**: `git diff --stat -- ':!toolchain/*/testdata'`
    -   To see content of non-testdata changes:
        `git diff -- ':!toolchain/*/testdata'`
-   **Identify testdata changes**: `git diff --name-only 'toolchain/*/testdata'`

#### For Jujutsu (jj) Users:

-   **Summarize code changes**:
    `jj --no-pager diff --stat '~toolchain/*/testdata'`
    -   Note: Quoting the fileset `'~toolchain/*/testdata'` is critical if it
        contains wildcards.
    -   To see content of non-testdata changes, use `--git` to get standard
        unified diff format: `jj --no-pager diff --git '~toolchain/*/testdata'`
-   **Identify testdata changes**:
    `jj --no-pager diff --name-only 'toolchain/*/testdata'`

#### For Github Pull Requests:

-   **Summarize code changes**: `gh pr diff`
-   **Identify testdata changes**:
    `gh pr diff --name-only | grep '^toolchain/.*/testdata'`

#### Handling Specific Revisions:

If you are summarizing changes in a specific revision (for example, `@-`) or
pull request (for example, #1234), add `-r <rev>` or `<pr_number>` to the
commands:

-   `git diff <rev>^ <rev> ...` (or use `git show <rev>`)
-   `jj --no-pager diff -r <rev> ...`
-   `gh pr diff <pr_number>`

### 2. Extract Test Input Changes (Recommended)

To easily identify changes, use the included Python helper script to extract all
text additions and removals from the diff, categorized by Input, STDERR, and
STDOUT changes. This script reads a unified diff from stdin.

```bash
# For Git:
git diff -- 'toolchain/*/testdata' | python3 .agents/skills/summarize_testdata_changes/scripts/parse_diff.py

# For Jujutsu (jj):
jj diff --git 'toolchain/*/testdata' | python3 .agents/skills/summarize_testdata_changes/scripts/parse_diff.py

# For a specific revision with jj:
jj diff -r @- --git 'toolchain/*/testdata' | python3 .agents/skills/summarize_testdata_changes/scripts/parse_diff.py

# For a specific PR with Github:
gh pr diff 1234 | python3 .agents/skills/summarize_testdata_changes/scripts/parse_diff.py
```

### 3. Identify Patterns and Produce a List of Groups

-   Read the diff and produce a list of groups of changes that share a common
    theme or cause (for example, "Updated expected output for integer literals",
    "Added tests for new keyword").
-   **CRITICAL**: _Every single change_ in the testdata diff must be represented
    by at least one group. Do not ignore changes to `CHECK` lines.
    -   If it's not clear what group a change belongs to, create a new group for
        it.
-   For each group:
    -   Provide a brief description of the group.
    -   (Optional) Briefly note if the group appears to be an intended or
        unintended consequence of the code changes.
-   Divide the groups into sections:
    -   Test Changes: Changes to test inputs (lines not prefixed with
        `// CHECK`), along with diagnostic output changes where relevant
    -   Diagnostic Changes: Changes to diagnostic output (lines prefixed with
        `// CHECK:STDERR`) with no corresponding changes to test inputs
    -   [Output Type] Changes: Changes to STDOUT (lines prefixed with `// CHECK:STDOUT`)
        -   Create one section for each relevant kind of test. For example,
            parser tests should typically be in a "Parse Tree Changes" section,
            check tests should typically be in a "SemIR Changes" section, and
            lower tests should typically be in an "LLVM IR Changes" section.

### 4. Improve Grouping

-   Read the list of groups and check to see if any of them should be combined
    or split apart. If needed, do so.

### 5. Assign Changes to Groups

-   Read the diff again, and then for _each_ change in the diff:
    -   Add the change to the appropriate group (or, rarely, groups).
    -   **CRITICAL**: _Every single change_ in the testdata diff must be
        represented by at least one group. Do not ignore changes to `CHECK`
        lines.
    -   If the change affects _test inputs_ (lines not prefixed with `// CHECK`)
        or _diagnostic output_ (lines prefixed with `// CHECK:STDERR`):
        -   List the file within the group. Don't just give one or a few
            examples. Include every file.
        -   Provide an inline diff if the change is small.
        -   Provide a link to the file if the change is large.
    -   Otherwise, if the change only affects _STDOUT_ (lines prefixed with
        `// CHECK:STDOUT`):
        -   Ensure the group contains a representative example that matches the
            current change.
        -   The representative example should be an inline diff of the change.
    -   **CRITICAL**: _Every single change_ to test inputs and diagnostic
        outputs in the files being summarized must be explicitly listed in at
        least one group. Do not skip changes, even if they are similar to
        changes you've already seen, and do not just give examples.

### 6. Validation

As a final validation step:

-   Read through the testdata diff again.
-   Ensure that every change in the diff is reflected by at least one group in
    the report.

## Report Template

Use the following template for the generated report:

```markdown
# `testdata` Change Summary

## Code Changes

[One paragraph summarizing changes outside of testdata.]

## Test Changes

### [Group Name]

[Description of the group.]

[Change 1: diff context OR link]

[Change 2: diff context OR link]

...

## Diagnostic Changes

### [Group Name]

[Description of the group.]

[Change 1: diff context OR link]

[Change 2: diff context OR link]

...

## [Output Type] Changes

### [File Path]

[Description of the group.]

[Example diff context]

Changes of this kind were found in [Number] files. Examples: [List of files]

...
```

Skip sections that would be empty.
