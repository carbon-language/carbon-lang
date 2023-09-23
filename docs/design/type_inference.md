# Type inference

<!--
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

<!-- toc -->

## Table of contents

-   [Overview](#overview)
-   [Open questions](#open-questions)
    -   [Inferring a variable type from literals](#inferring-a-variable-type-from-literals)
-   [Alternatives considered](#alternatives-considered)
-   [References](#references)

<!-- tocstop -->

## Overview

[Type inference](https://en.wikipedia.org/wiki/Type_inference) occurs in Carbon
when the `auto` keyword is used. This may occur in
[variable declarations](values.md#binding-patterns-and-local-variables-with-let-and-var)
or [function declarations](functions.md).

At present, type inference is very simple: given the expression which generates
the value to be used for type inference, the inferred type is the precise type
of that expression. For example, the inferred type for `auto` in
`fn Foo(x: i64) -> auto { return x; }` is `i64`.

Type inference is currently supported for [function return types](functions.md)
and
[declared variable types](values.md#binding-patterns-and-local-variables-with-let-and-var).

## Open questions

### Inferring a variable type from literals

Using the type on the right side for `var y: auto = 1` currently results in a
constant `IntLiteral(1)` value, whereas most languages would suggest a variable
integer type, such as `i64`. Carbon might also make it an error. Although type
inference currently only addresses `auto` for variables and function return
types, this is something that will be considered as part of type inference in
general, because it also affects checked generics, templates, lambdas, and
return types.

## Alternatives considered

-   [Use `_` instead of `auto`](/proposals/p0851.md#use-_-instead-of-auto)

## References

-   Proposal
    [#851: auto keyword for vars](https://github.com/carbon-language/carbon-lang/pull/851)
