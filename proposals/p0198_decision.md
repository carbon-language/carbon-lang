# Decision for: Comments

<!--
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

Proposal accepted on 2020-02-19

Affirming:

-   [austern](https://github.com/austern)
-   [chandlerc](https://github.com/chandlerc)
-   [geoffromer](https://github.com/geoffromer)
-   [gribozavr](https://github.com/gribozavr)
-   [josh11b](https://github.com/josh11b)
-   [zygoloid](https://github.com/zygoloid)

Abstaining:

-   [noncombatant](https://github.com/noncombatant)
-   [tituswinters](https://github.com/tituswinters)

## Rationale

-   Some comment syntax is necessary to support software evolution, readable and
    understandable code, and many other goals of Carbon.
-   A single, simple, and consistent comment style supports Carbon's goal of
    easy to read and understand code, and fast development tools.
-   The experiment of restricting comments to be the only non-whitespace text on
    a line supports Carbon's goal of software evolution.
-   The careful open lexical space left supports Carbon's goal of language
    evolution.
-   The use of `//` as the primary syntax marking comments supports
    interoperability with C++-trained programmers and codebases by avoiding
    unnecessary and unhelpful churn of comment syntax.
