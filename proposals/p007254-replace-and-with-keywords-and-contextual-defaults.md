# Replace `:!` and `:?` with keywords and contextual defaults

<!--
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

[Pull request](https://github.com/carbon-language/carbon-lang/pull/7254)

<!-- toc -->

## Table of contents

-   [Abstract](#abstract)
-   [Problem](#problem)
-   [Background](#background)
-   [Proposal](#proposal)
-   [Details](#details)
    -   [Phase Keywords and Contextual Defaults](#phase-keywords-and-contextual-defaults)
        -   [Contextual Defaults](#contextual-defaults)
    -   [Associated Constants](#associated-constants)
    -   [Extended Types](#extended-types)
-   [Rationale](#rationale)
-   [Alternatives considered](#alternatives-considered)
    -   [Keep the `:!` syntax](#keep-the--syntax)
    -   [Use other keywords (for example, `static`)](#use-other-keywords-for-example-static)
    -   [Erased model for associated constants](#erased-model-for-associated-constants)
    -   [Context-sensitive defaults based on parameter type](#context-sensitive-defaults-based-on-parameter-type)
    -   [Allow redundant phase keywords](#allow-redundant-phase-keywords)
    -   [Use `exprtype` and `expr` keywords](#use-exprtype-and-expr-keywords)

<!-- tocstop -->

## Abstract

This proposal removes the `:!` syntax for generics and templates in favor of
keywords (`generic`, `template`, `runtime`) and contextual defaults for phase.
It also replaces `:?` with `fwd` and introduces `exttype` for extended types.

## Problem

The `:!` syntax for generics and templates has several issues:

-   It doesn't work well for controlling the phase for functions.
-   The connection between generics/templates and `!` is tenuous and not an
    effective mnemonic.
-   It is very inventive syntax with little familiarity from other languages.
-   It makes Carbon code using generics start to look like ASCII-art due to
    dense punctuation.
-   It is in tension with more compelling use cases for `!`, such as for
    operations that are required to succeed or terminate (for example,
    unwrapping optionals).

## Background

These issues were discussed in leads issue #6932, and a direction was decided to
move away from punctuation and towards keywords and contextual defaults.

## Proposal

We propose to:

1.  Remove `:!` syntax for generics and templates.
2.  Introduce contextual defaults for phase:
    -   Parameters to compile-time entities (interfaces, impls, classes) are
        `generic` by default.
    -   Deduced function parameters are `generic` by default.
    -   Explicit function parameters are `runtime` by default.
3.  Allow overriding defaults with keywords `template`, `generic`, and
    `runtime`.
4.  Disallow keywords that match the contextual default to ensure consistency.
5.  Replace `:?` with a binding modifier `fwd` and use `exttype` as the analog
    to `type` but for extended types that cover what were previously considered
    _forms_.

## Details

### Phase Keywords and Contextual Defaults

The keywords `generic`, `template`, and `runtime` are used to specify the phase
of parameters.

#### Contextual Defaults

-   **Compile-time entities**: Parameters to entities like `interface`, `impl`,
    and `class` are `generic` by default.

    ```carbon
    interface I(T: type) { ... } // T is generic
    ```

    They can be marked as `template`:

    ```carbon
    class C(template T: type) { ... } // T is template
    ```

-   **Deduced function parameters**: Parameters in `[]` for functions default to
    `generic`.

    ```carbon
    fn F[T: type](arg: T); // T is generic
    ```

    They can be marked as `template`:

    ```carbon
    fn F[template T: type](arg: T); // T is template
    ```

    If we ever add deduced runtime parameters, they will use `runtime`:

    ```carbon
    fn F[runtime T: type](arg: T); // T is runtime
    ```

-   **Explicit function parameters**: Parameters in `()` for functions default
    to `runtime`.
    ```carbon
    fn F(arg: i32); // arg is runtime
    ```
    They can be marked as `generic` or `template`:
    ```carbon
    fn F(generic T: type, arg: T); // T is generic
    ```

Keywords matching the contextual default are **disallowed** to avoid confusion
and ensure consistency.

### Associated Constants

Associated constants in interfaces and impls require no extra keywords. Their
meaning is guided by the context.

The model of an interface is essentially a class whose "evaluation time" is
inherently the symbolic compile-time phase. As a consequence, its "evaluation
time" fields act as generic constants, and it wouldn't make sense to put an
additional phase constraint on them.

### Extended Types

This proposal replaces the concept of "forms" (as described in
[`/docs/design/values.md`](/docs/design/values.md)) with **extended types**.

The term "forms" was originally used to generalize types to include expression
category, phase, and value. However, this terminology was found to conflict
confusingly with the concept of "unformed state". To resolve this, we move to a
model where these are considered "extended types", connecting them more directly
to the type system while preserving `type` for standard object types.

Under this new design:

-   `exttype` is the keyword for the type of extended types (analogous to `type`
    for object types).
-   The previous `:?` syntax for deduced form bindings is replaced by a binding
    modifier `fwd`. This modifier causes the right-hand-side of the binding's
    `:` to be converted to `exttype` rather than `type`.
-   `fwd` can also be used in return types (for example, `-> fwd T`) to forward
    the extended type information.

This approach allows us to reclaim high-value punctuation like `?` for other
uses (such as optional types) while providing a more explicit and less
punctuated syntax for advanced generic programming.

Example:

```carbon
fn F[fwd T: exttype](arg: T) -> fwd T;
```

## Rationale

This proposal advances the following Carbon goals:

-   **Code that is easy to read, understand, and write**: By removing dense
    punctuation and using keywords, the code becomes more readable and less like
    ASCII-art. Contextual defaults reduce verbosity.
-   **Software and language evolution**: Reclaiming `!` and `?` opens up syntax
    space for other high-value features like optional unwrapping.

## Alternatives considered

### Keep the `:!` syntax

One alternative was to retain the existing punctuation-based syntax where `:!`
is used to denote generic and template parameters.

-   **Advantages**:
    -   Maintains continuity with the previously established design.
    -   Is very concise, requiring no keywords.
-   **Disadvantages**:
    -   The syntax makes code look like "ASCII-art" due to the high density of
        punctuation.
    -   The connection between `!` and generics is not an effective mnemonic.
    -   It blocks other potential uses for `!`, such as for operations that are
        required to succeed or terminate (for example, unwrapping optionals).
    -   It does not scale well to controlling the phase of function parameters.
-   **Decision**: This alternative was rejected because the disadvantages in
    readability and extensibility outweigh the benefit of conciseness. The leads
    decided to move towards keywords and contextual defaults.

### Use other keywords (for example, `static`)

Another alternative considered was using different keywords to specify the
phase, such as `static` or `static val`.

-   **Advantages**:
    -   `static` is a familiar term in many languages (like C++) for things
        decided before runtime.
    -   Could allow for a syntax where the declaration itself contains all
        information, independent of context.
-   **Disadvantages**:
    -   `static` is heavily overloaded in C++ and has many different meanings
        (storage duration, class members, etc.), which could cause confusion for
        users coming from C++.
    -   Other keywords like `comptime` or `symbolic` were also considered but
        found to be less accessible or less fitting as mnemonics than `generic`.
-   **Decision**: This alternative was rejected because the chosen keywords
    (`generic`, `template`, `runtime`) are more specific to the phase concept in
    Carbon and avoid the overloading issues of `static`.

### Erased model for associated constants

For associated constants in interfaces, an alternative was to use an "erased"
model to arrive at names instead of the proposed model.

-   **Advantages**: (Details are sparse in the issue, but it was raised as a
    conceptual alternative).
-   **Disadvantages**: It was found to be less comfortable than the model where
    context guides the meaning.
-   **Decision**: This alternative was rejected in favor of the model where
    interfaces are treated essentially as classes evaluated at the symbolic
    compile-time phase. In this model, fields naturally act as generic constants
    without requiring extra keywords, which was preferred for its simplicity and
    fit with the rest of the design.

### Context-sensitive defaults based on parameter type

One alternative suggested was to make explicit function parameters default to
`generic` if they cannot be represented at runtime (such as types).

-   **Advantages**:
    -   Allows omitting keywords in more cases, such as `fn F(T: type, arg: T)`.
-   **Disadvantages**:
    -   Adds complexity to the defaulting rules.
    -   Tricky for types like integers that can be used in both runtime and
        compile-time contexts (for example, array sizes).
-   **Decision**: Rejected due to the added complexity and edge cases with types
    used in both phases.

### Allow redundant phase keywords

Another alternative was to allow keywords matching the contextual default to be
used optionally (for example, allowing `generic` in a context where it is the
default).

-   **Advantages**:
    -   Simpler mental model for beginners who might want to be explicit
        everywhere.
    -   Allows ignoring the rule until a linter enforces it.
-   **Disadvantages**:
    -   Creates multiple ways to say the same thing.
    -   Can confuse readers wondering why a default was made explicit.
-   **Decision**: Rejected to ensure consistency and avoid confusion, following
    the pattern used elsewhere in Carbon (for example, not allowing redundant
    `public`).

### Use `exprtype` and `expr` keywords

One alternative considered for replacing "forms" was to use the terminology
"expression types" with `exprtype` as the bottom type and `expr` as the binding
modifier.

-   **Advantages**:
    -   Maintains progressive disclosure by keeping `type` as the primary term
        for object types and qualifying it as `exprtype` for expression types.
    -   Connects directly to the concept of expression metadata.
-   **Disadvantages**:
    -   It has a slightly awkward construction where the narrower term ("type")
        is the base term, and the broader term ("expression type") is qualified.
    -   It confusingly implies that it refers to the _type of the expression_,
        while we want that use of the term "type" to not include the extended
        information.
    -   It also implies with `expr` on a binding that the expression itself is
        bound and captured, rather than being evaluated first. Hard to explain
        that this matches the _evaluated_ expression.
-   **Decision**: This alternative was rejected in favor of the **Extended
    Types** model. The team preferred "extended types" as the terminology anchor
    (yielding `exttype`). For the binding modifier, `fwd` was chosen because it
    connects to the use case of forwarding extended type information (similar to
    C++ `std::forward`) and fits well as a three-letter keyword similar to
    `ref`, `var`, and `val`.
