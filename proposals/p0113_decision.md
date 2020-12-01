# Decision for: Add a C++ style guide

<!--
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

Proposal accepted on 2020-10-27

Affirming:

-   [austern](https://github.com/austern)
-   [chandlerc](https://github.com/chandlerc)
-   [geoffromer](https://github.com/geoffromer)
-   [gribozavr](https://github.com/gribozavr)
-   [josh11b](https://github.com/josh11b)
-   [noncombatant](https://github.com/noncombatant)
-   [zygoloid](https://github.com/zygoloid)

Abstaining:

-   [tituswinters](https://github.com/tituswinters)

## Rationale

A common, established style guide will allow us to focus on the more important
aspects of coding and code review. Once we're familiar with the rules, their
consistent application will result in a more readable codebase, and rules that
can largely be automated by clang-format will result in a more efficient
development process.

This particular ruleset attempts to align with our expected lexical rules for
the Carbon language (for example, comment syntax and capitalization rules),
which will also mean that we can use a largely consistent style between the
aspects of the toolchain implemented in C++ and the aspects implemented in
Carbon.
