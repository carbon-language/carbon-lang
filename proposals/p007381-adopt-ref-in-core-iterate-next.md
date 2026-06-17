# Adopt `ref` in `Core.Iterate.Next`

<!--
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

[Pull request](https://github.com/carbon-language/carbon-lang/pull/7381)

<!-- toc -->

## Table of contents

-   [Abstract](#abstract)
-   [Problem](#problem)
-   [Background](#background)
-   [Proposal](#proposal)
-   [Rationale](#rationale)
-   [Alternatives considered](#alternatives-considered)

<!-- tocstop -->

## Abstract

Change the `cursor` parameter in `Core.Iterate.Next` to be a `ref` parameter
instead of a pointer parameter.

## Problem

According to [p001885](p001885-for-statement-and-user-types.md),
`Core.Iterate.Next` is declared as follows:

```carbon
  fn Next[self: Self](cursor: CursorType*) -> Optional(ElementType);
```

Aside from the old `self` syntax, this uses pass-by-pointer to pass a mutable
reference. This stopped being idiomatic Carbon when `ref` patterns were added,
but we still require it when interacting with this core language feature.

## Background

Proposal [p001885](p001885-for-statement-and-user-types.md) added the `Iterate`
interface.

Proposal [p002006](p002006-values-variables-pointers-and-references.md) affirmed
that Carbon does not have reference _types_, but does have reference
_expressions_.

Proposal [p005434](p005434-ref-parameters-arguments-returns-and-val-returns.md)
added support for `ref` _patterns_ to match reference expressions. These
replaced pass-by-pointer as the idiomatic way to pass a handle to a mutable
object into a function.

## Proposal

Change the declaration of `Core.Iterate.Next` to:

```carbon
  fn Next(self, ref cursor: CursorType) -> Optional(ElementType);
```

## Rationale

-   [Code that is easy to read, understand, and write](/docs/project/goals.md#code-that-is-easy-to-read-understand-and-write)
    -   Makes the `Iterate` interface, and user-defined `impl`s of it, more
        consistent with the rest of the language.

## Alternatives considered

None.
