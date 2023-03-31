# Values, variables, and value categories

<!--
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

<!-- toc -->

## Table of contents

-   [Value categories](#value-categories)
-   [Binding patterns and local variables with `let` and `var`](#binding-patterns-and-local-variables-with-let-and-var)
    -   [Pattern match control flow](#pattern-match-control-flow)
    -   [Pattern matching in local variables](#pattern-matching-in-local-variables)
-   [Open questions](#open-questions)
    -   [Slice or array nested value pattern matching](#slice-or-array-nested-value-pattern-matching)
    -   [Generic/template pattern matching](#generictemplate-pattern-matching)
    -   [Pattern matching as function overload resolution](#pattern-matching-as-function-overload-resolution)

<!-- tocstop -->

## Value categories

Every value in Carbon has a
[value category](<https://en.wikipedia.org/wiki/Value_(computer_science)#lrvalue>)
that is either an L-value or an R-value.

_L-values_ are _located values_. They represent _storage_ and have a stable
address. They are in principle mutable, although their type's API may limit the
mutating operations available.

_R-values_ are _readonly values_. They cannot be mutated in any way and may not
have storage or a stable address.

## Binding patterns and local variables with `let` and `var`

[_Binding patterns_](/docs/design/README.md#binding-patterns) produce named
R-values by default. This is the desired default for many pattern contexts,
especially function parameters. R-values are a good model for "input" function
parameters which are the dominant and default style of function parameters:

```carbon
fn Sum(x: i32, y: i32) -> i32 {
  // `x` and `y` are R-values here. We can read them, but not modify.
  return x + y;
}
```

A pattern can be introduced with the `var` keyword to create a _variable
pattern_. This creates an L-value including the necessary storage. Every binding
pattern name introduced within a variable pattern is also an L-value. The
initializer for a variable pattern is directly used to initialize the storage.

```carbon
fn Consume(var x: SomeData) {
  // We can mutate and use the local `x` L-value here.
}
```

### Local variables

A local binding pattern can be introduced with either the `let` or `var`
keyword. The `let` introducer begins a readonly pattern just like the default
patterns in other contexts. The `var` introduce works exactly the same as
introducing the pattern inside of a `let` binding with `var` -- there's just no
need for the outer `let`.

-   `let` _identifier_`:` _< expression |_ `auto` _>_ `=` _value_`;`
-   `var` _identifier_`:` _< expression |_ `auto` _> [_ `=` _value ]_`;`

These are just simple examples of binding patterns used directly in local
declarations. Local `let` and `var` declarations build on Carbon's general
[pattern matching](/docs/design/pattern_matching.md) design, with `var`
declarations implicitly starting off within a `var` pattern while `let`
declarations introduce patterns that work the same as function parameters and
others with bindings that are R-values by default.
