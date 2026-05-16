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

## Prek

Running `prek` is mandatory. To run it on all files:

```bash
prek run -a
```

To validate a specific list of files:

```bash
prek run --files <files>
```
