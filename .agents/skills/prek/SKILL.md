---
name: Prek
description:
    Instructions for running prek, the Carbon pre-submit/style/lint checker,
    that *MUST* be run before submitting an change.
---

# Prek

<!--
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

`prek` is the Carbon pre-submit, style, and lint checker. Running it is
mandatory before submitting any changes.

## Running prek

To run `prek` on all files:

```bash
prek run -a
```

To validate a specific list of files:

```bash
prek run --files <files>
```

## Running prek in a Jujutsu (jj) workspace

If you are working in a Jujutsu workspace, running `prek` directly will fail
because it expects a standard Git repository structure. Instead, use the helper
script:

```bash
./scripts/jj_prek.sh
```

This script runs `prek` on all files that have changed between `trunk` and your
current Jujutsu `@` change.

## Prek dependency errors

> [!TIP] If `prek` fails with an error about resolving dependencies or security
> policy, you may be running in a restricted environment where the
> special-purpose `gpkg` tool is required. Prefix the command with `gpkg`, for
> example: `gpkg prek run -a` or `gpkg ./scripts/jj_prek.sh`.
