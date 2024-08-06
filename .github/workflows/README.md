# Workflows

<!--
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

## Hardening

Workflows are hardened using
[Step Security tool](https://app.stepsecurity.io/secureworkflow). Findings for
the "Harden Runner" steps are
[available online](https://app.stepsecurity.io/github/carbon-language/carbon-lang/actions/runs).

### Allowed endpoints

Most jobs only have a few endpoints, but due to tools which do downloads, a few
have significantly more. These are:

-   pre_commit.yaml (Bazel, pre-commit)
-   nightly_release.yaml (Bazel)
-   tests.yaml (Bazel)

When updating one of these, consider updating all of them.

We try to keep `allowed-endpoints` with one per line. Prettier wants to wrap
them, which we fix this with `prettier-ignore`.

## Testing

We keep around an `action-test` branch in carbon-lang, which can be used to test
triggers with `push:` configurations. For example:

```
on:
  push:
    branches: [action-test]
```
