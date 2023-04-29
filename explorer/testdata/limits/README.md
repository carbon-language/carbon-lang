# Limit tests

<!--
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

These tests check for various limit conditions (such as an infinite loop). The
tests collectively disable autoupdate so that tracing isn't enabled, because
tracing creates substantial additional overhead.
