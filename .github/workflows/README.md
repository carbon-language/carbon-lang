# Testing workflows

<!--
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

We keep around an `action-test` branch in carbon-lang, which can be used to test
triggers with `push:` configurations. For example:

```
on:
  push:
    branches: [action-test]
```
