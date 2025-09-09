# Principle: File concatenation should introduce compiler errors

<!--
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

<!-- toc -->

## Table of contents

-   [Background](#background)
-   [Principle](#principle)
-   [Applications of these principles](#applications-of-these-principles)

<!-- tocstop -->

## Background

In C++, it is a common practice to concatenate files together in order to
recreate and distribute a bug repro or to build a piece of C++ code in an
environment without the requirement of a build system. This allows code taken
from any build environment to be easily compiled by a developer in any other
environment, through a single compiler invocation.

The reason this works is largely historical, as #include performs a textual
inclusion, but it can be maintained without textual includes as well.

## Principle

Concatenating any two Carbon files together will not change the meaning of the
code in either file. It will be required to order the concatenation so that
dependencies are ordered first, in order to maintain the information
accumulation principle. There is always an ordering that allows any Carbon
project to be fully concatenated into a single Carbon source file and thus built
with a single compiler invocation.

## Applications of these principles

Carbon uses syntax to denote the start of a library or package, which functions
at the top of a file or in the middle of a file in the same way. Language rules
may change behaviour and/or diagnostics based on the relationship of code
between libraries or packages, but not between files.
