# Decision for: String literals

<!--
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

Proposal accepted on 2021-02-23

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

This proposal supports the goal of making Carbon code
[easy to read, understand, and write](/docs/project/goals.md#code-that-is-easy-to-read-understand-and-write),
by ensuring that essentially every kind of string content can be represented in
a Carbon string literal, in a way that is natural, toolable, and easy to read:

-   Multi-line strings are supported by multi-line string literals, and the
    rules for stripping leading indentation enhance readability by allowing
    those literals to avoid visually disrupting the indentation structure of the
    code.
-   Strings that make extensive use of `\` and `"` are supported by raw string
    literals.
-   Treating raw versus ordinary and single-line versus multi-line as orthogonal
    allows Carbon to support all 4 combinations while keeping the language
    simple.
-   The handling of `\#` within raw string literals makes it possible to use
    escape sequences within raw string literals when necessary, for example to
    embed arbitrary byte values or Unicode data. This ensures that the
    programmer is never prevented from using a raw string literal, or forced to
    assemble a single logical string by concatenating ordinary and raw literals
    (with the negligible and fixable exception of strings like
    `"\\################"`, as noted in the proposal).
-   "File type indicators" make it easier for tooling to understand the contents
    of literals, in order to provide features such as syntax highlighting,
    automated formatting, and potentially even certain kinds of static analysis,
    for code that's embedded in string literals.
-   Support for non-Unicode strings by way of `\x` ensures "support for software
    outside the primary use case".
-   Avoids unnecessary invention, following Rust and particularly Swift.
