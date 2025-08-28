# Principle: The signature is the contract

<!--
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

<!-- toc -->

## Table of contents

-   [Principle](#principle)
-   [Applications of this principle](#applications-of-this-principle)
-   [Background](#background)
-   [Benefits and rationale](#benefits-and-rationale)
-   [Exceptions](#exceptions)

<!-- tocstop -->

## Principle

The declaration of an entity is authoritative for all typechecking information
about the aspects of the entity described by the entity's declaration (not
including things like the members of the entity). An entity's declaration should
generally be sufficient to typecheck uses of the entity, for uses that do not
reference any members of that entity. Information provided in a declaration is
not overridden or changed by its definition.

## Applications of this principle

-   Parameters and return types are fully specified in a function declaration,
    so Carbon can generally typecheck calls to a function given only its
    declaration. This allows the definition of the function to be separately
    compiled in the `impl` file, assuming its body is not needed for code
    generation.
    -   This extends to class definitions only requiring declarations of their
        member functions.
-   Constraints can be implied only within a single declaration, see
    [the alternative considered in proposal #818](/proposals/p0818.md#implied-constraints-across-declarations).
-   The types and constraints on generic parameters to a type declaration (like
    a `class` or `interface`) are specified in the declaration, not inferred
    from the uses in they body of the definition.
-   The type of a local variable is fixed by its declaration, not inferred from
    later uses. This follows
    [the (proposed) "unidirectional type propagation" principle](https://github.com/carbon-language/carbon-lang/pull/103).

## Background

-   This is aligned with the
    ["Low context-sensitivity" principle](low_context_sensitivity.md): a
    function signature should be understandable without having to locate, read,
    and understand its definition.
-   This is in support of the
    ["Information accumulation" principle](information_accumulation.md), which
    motivates having separate forward declarations in addition to definitions.
-   [Rust's Golden Rule](https://steveklabnik.com/writing/rusts-golden-rule/):
    states a similar rule for Rust.

## Benefits and rationale

The more you can do with the declaration of an entity without its definition,
the more opportunities there are to move the definition to a file that is
compiled separately, improving build parallelism.

This principle supports
[evolution of software written in Carbon](/docs/project/goals.md#software-and-language-evolution)
by separating the contract of an API from its implementation. This reduces the
amount of code that must be considered to understand the contract, and separates
changes that affect the contract from other changes. In this way, we can evolve
implementations without accidentally breaking its callers.

This principle supports encapsulation, allowing isolation of implementation
details separate from API.

This principle encourages the declaration of APIs to make their contract
explicit, contributing to self documentation. Favoring explicit contracts over
inferred requirements avoids creating problems that the compiler has to solve.
Particularly concerning is if requirements would have to be propagated multiple
steps, iterating until a fixed point is found. This can lead to longer compile
times and a more complex compiler implementation, hurting both the compiler
development and developer understanding of what the compiler does.

The ability to use an entity with only its declaration allows breaking of
dependency cycles, by using forward declarations with fewer dependencies than
the full definition. This is instead of forward references, which are counter to
the ["information accumulation" principle](information_accumulation.md).

Allowing a definition to override aspects of the corresponding declaration would
open the door to inconsistency. This is something that can happen in C++
(potentially leading to ODR violations):

-   The behaviour of `&x` where `x` is an lvalue of an incomplete class type,
    where you can't do the lookup for a member named `operator&`, see
    [this note on C++ unary operators](https://eel.is/c++draft/expr#unary.op-5).
-   The MSVC ABI has different member pointer representations for different
    kinds of classes, but you can use a pointer to member before the class is
    defined.

## Exceptions

The main exception to this principle is when there is an explicit opt-in as part
of the signature or declaration to look in the body of the definition for
specific information that would normally be in the signature. Examples:

-   The `auto` keyword may be used to indicate that the return type of a
    function is determined by the `return` statements in the body of the
    function.
-   The `template` keyword may be used to indicate that the constraints on a
    generic parameter are not complete, and instead it will be determined by the
    uses of that parameter in the body of the function.

Both cases require the definition to be visible to the caller in order to type
check.

There is also the exception that there can be non-member information about an
entity in the body of an entity's definition that may be needed to type check
uses. For example, an interface or named constraint may have requirements that
affect type checking of uses.
