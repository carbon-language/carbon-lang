# Function return type inference

<!--
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

[Pull request](https://github.com/carbon-language/carbon-lang/pull/826)

<!-- toc -->

## Table of contents

-   [Problem](#problem)
-   [Background](#background)
    -   [Lambdas](#lambdas)
    -   [`auto` keyword](#auto-keyword)
    -   [`const` qualification](#const-qualification)
-   [Proposal](#proposal)
-   [Details](#details)
    -   [Explicit return required](#explicit-return-required)
    -   [Executable semantics changes](#executable-semantics-changes)
    -   [Disallow direct recursion](#disallow-direct-recursion)
        -   [Indirect recursion](#indirect-recursion)
-   [Rationale based on Carbon's goals](#rationale-based-on-carbons-goals)
-   [Open questions](#open-questions)
    -   [Multiple returns](#multiple-returns)
    -   [`const` qualification](#const-qualification-1)
-   [Alternatives considered](#alternatives-considered)
    -   [Only allow `auto` return types if parameters are generic](#only-allow-auto-return-types-if-parameters-are-generic)
    -   [Provide alternate function syntax for concise return type inference](#provide-alternate-function-syntax-for-concise-return-type-inference)
    -   [Allow separate declaration and definition](#allow-separate-declaration-and-definition)

<!-- tocstop -->

## Problem

Should there be a shorter way of declaring a function? This embodies two
questions:

-   Should there be a way for a declarer to ask that the return type of a
    function be inferred from returned values?
-   Should there be an alternate function syntax that provides briefer function
    declarations for simple cases of return type inference?

## Background

Under
[Proposal #438: Add statement syntax for function declarations](https://github.com/carbon-language/carbon-lang/pull/438),
the syntax approved was:

> `fn` _identifier_ `(` _args_ `)` _[_ `->` _expression ]_ `{` _statements_ `}`

Executable semantics currently supports the syntax:

> `fn` _identifier_ `(` _args_ `) =>` _expression_

In C++, there is similar automatic type inference for return types, although the
use of `=>` reflects syntax tentatively discussed for matching use.

### Lambdas

Lambdas are also under discussion for Carbon: however, a different syntax is
likely to arise. This proposal does not try to address lambda syntax, although
it is possible that a decision on lambda syntax may lead to an alternate syntax
for declaring functions in order to maintain syntax parity.

### `auto` keyword

This is the first proposal to formally include a use of `auto`; although
[`var` statement #339](p0339.md) mentions it in an alternative, the proposal did
not explicitly propose the keyword. However, the `var x: auto` use-case is
expected in Carbon, and so uses of `auto` here can be considered in that
context.

Note the `auto` keyword name and behavior can generally be considered consistent
with C++.

### `const` qualification

C++ initially used the matching return type for lambdas. However, it switched to
its `auto` variable rules, which are defined in terms of template argument
deduction and discard `const` qualifiers. This yields subtle behavioral
differences.

Note that `const` is not yet defined in Carbon.

## Proposal

Support automatic type inference in Carbon with a new `auto` keyword, as in:

> `fn` _identifier_ `(` _args_ `) -> auto {` _statements_ `}`

For now, only one return statement is allowed: we will avoid defining rules for
handling differing return types.

Separate declarations and definitions are not supported with `auto` return types
because the declaration must have a known return type.

## Details

### Explicit return required

Functions using `-> auto` must return an expression; using the implicit return
or `return;` is invalid. This is consistent with the
[design for returning empty tuples](/docs/design/control_flow/return.md#returning-empty-tuples).

### Executable semantics changes

Executable semantics will remove the `=>` function syntax.

In practice, this means the current executable semantics syntax:

```
fn Add(x: i32, y: i32) => x + y;
```

Becomes:

```
fn Add(x: i32, y: i32) -> auto { return x + y; }
```

### Disallow direct recursion

Direct recursive calls can create complexities in inferring the return type. For
example:

```
// In the return statement.
fn Factorial(x: i32) -> auto {
    return (if x == 1 then x else x * Factorial(x - 1));
}

// Before the return statement, but affecting the return type.
fn Factorial(x: i32) -> auto {
    var x: auto = (if x == 1 then x else x * Factorial(x - 1));
    return x;
}
```

As a consequence, direct recursion is rejected: that is, a function with an
`auto` return type may not call itself.

#### Indirect recursion

Indirect recursion will typically look like:

```
fn ExplicitReturn() -> i32;
fn AutoReturn() -> auto { return ExplicitReturn(); }
fn ExplicitReturn() -> i32 { return AutoReturn(); }
```

This is valid because the return type of `AutoReturn()` can be calculated to be
`i32` from the forward declaration of `ExplicitReturn`.

Due to how name lookup works, there is no risk of indirect recursion causing
problems for typing:

-   A similar separate declaration of an `auto` return is
    [disallowed](#proposal).
-   Without a separate declaration, name lookup would fail. That is, the below
    has a name lookup error at `return B();`:

    ```
    fn A() -> auto { return B(); }
    fn B() -> auto { return A(); }
    ```

As a consequence, no specific rules about indirect recursion are needed, unlike
direct recursion.

## Rationale based on Carbon's goals

-   [Code that is easy to read, understand, and write](/docs/project/goals.md#code-that-is-easy-to-read-understand-and-write)

    -   We are avoiding providing an alternative function syntax in order to
        provide users less syntax they'll need to understand.
    -   As a practical measure, there are likely to be cases in generic code
        that are more feasible to write.

-   [Interoperability with and migration from existing C++ code](/docs/project/goals.md#interoperability-with-and-migration-from-existing-c-code)

    -   The intent is to have `auto` as a return type work similarly to C++'s
        type inference, in order to ease migration.

## Open questions

### Multiple returns

It's likely that we should support `auto` on functions with multiple returns.
This proposal declines to address that use-case because of the complexities
which arise when return types may differ. Instead, the use-case is left for
future proposals to handle.

### `const` qualification

As noted in background, [`const` qualification](#const-qualification) has seen
changes in C++. While we should work to choose a story that can remain
long-term, this will likely be revisited when rules around deduction for
templates are addressed.

## Alternatives considered

### Only allow `auto` return types if parameters are generic

`auto` return types are likely to be a negative impact to readability, because a
reader must read the function body in order to determine the return type. We
could detect whether parameters used in determining an `auto` return type are
generic, and only allow `auto` to be used when they are.

Advantages:

-   Limits the readability impact of `auto`.
    -   `auto` could not be used where the return type is clearly deterministic,
        and thus easy to write. In these cases, readers would never need to look
        at the function body to determine the return.
    -   `auto` could be used where the return type may be difficult to easily
        write as a consequence of potential generic parameters.

Disadvantages:

-   Increases the rule set around use of `auto` in return types.
-   Breaks C++ compatibility.

The decision to allow `auto` in more cases is primarily for C++ compatibility.

### Provide alternate function syntax for concise return type inference

Executable semantics currently demonstrates an alternate function syntax with:

```
fn Add(x: i32, y: i32) => x + y;
```

This is a more concise syntax that builds upon return type inference. We could
keep that, or come up with some other syntax.

Advantages:

-   For short functions, provides a more succinct manner of defining the
    function.
-   `=>` use echoes tentative matching syntax.

Disadvantages:

-   Creates an additional syntax which can be used for equivalent behavior.
-   Overlaps with lambda use-cases, but lambdas will likely end up looking
    different; that is, we should not assume the syntax will converge.
    -   We may in particular take a more sharply divergent syntax for lambdas,
        in order to allow for significant brevity in typing, which may not make
        sense to support for function declarations.
-   Limits use of `=>` for other purposes.
    -   To the extent that `=>` might primarily be replacing `-> auto`, assuming
        Carbon adopted Rust-style
        [block expression returns](https://doc.rust-lang.org/reference/expressions/block-expr.html),
        this offers limited value for the additional token use.

This proposal suggests not adding an inference-focused alternate function syntax
at this time; while consistent with tentative matching syntax, the split from
function syntax is significant. The
[one way principle](/docs/project/principles/one_way.md) applies here.

If an alternate function syntax is added, it should be done in tandem with a
lambda proposal in order to ensure consistency in syntax.

### Allow separate declaration and definition

The return type must be possible to determine from the declaration. As a
consequence, the declaration and definition of a function returning `auto`
cannot be significantly separated: a caller must be able to see the definition.
Thus, although a separate declaration would be insufficient for a call, we could
allow using `auto` with a separate declaration and definition when the caller
can see _both_.

This example would be valid because `CallAdd` can see the `Add` definition:

```
fn Add(x: i32, y: i32) -> auto;

fn Add(x: i32, y: i32) -> auto { return x + y; }

fn CallAdd() -> i32 { return Add(1, 2); }
```

However, the following example would be invalid because `CallAdd` can only see
the `Add` declaration (even though the definition may be in the same file), and
it would need the definition to determine the return type:

```
fn Add(x: i32, y: i32) -> auto;

fn CallAdd() -> i32 { return Add(1, 2); }

fn Add(x: i32, y: i32) -> auto { return x + y; }
```

Advantages:

-   Separating the declaration and definition could be used in files to cluster
    brief declarations, then put definitions below.
    -   A particularly common use-case of this might be in class declarations,
        where a user might want to offer a declaration and define it out of line
        where it doesn't interrupt the class's API.
    -   It may be preferred to only use `auto` return types with short
        functions, which limits this advantage as a short function could easily
        be inlined.

Disadvantages:

-   Cannot be used to break call cycles, which is the usual use of separate
    declarations and definitions.
    -   A separate declaration still cannot be called until after the definition
        is provided.
    -   This may be particularly confusing to humans.

The decision is to not support separate declarations and definitions with `auto`
return types due to the limited utility.
