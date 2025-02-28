# Advent of Code 2024

<!--
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

This directory contains sample solutions written in Carbon for
[Advent of Code 2024](https://adventofcode.com/2024/).

The Carbon toolchain is in a very early state, so these samples frequently need
to work around missing functionality and are not reflective of expected Carbon
style and idioms. Instead, the purpose of these examples are to test the current
state of the toolchain against larger code examples than those that are present
in the toolchain's own tests, to find bugs in the toolchain, and to drive
feature development in the toolchain by presenting somewhat realistic testcases.

If one of these examples stops building after a change to the toolchain, please:

-   Make sure that the build break is an expected consequence of the change.
-   Update the `BUILD` file to exclude that example.
-   File an issue and assign it to @zygoloid.
