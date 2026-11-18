# Name lookup

<!--
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

<!-- toc -->

## Table of contents

-   [Overview](#overview)
    -   [Unqualified name lookup](#unqualified-name-lookup)
    -   [Name lookup for common, standard types](#name-lookup-for-common-standard-types)
-   [Open questions](#open-questions)
    -   [Shadowing](#shadowing)
-   [References](#references)

<!-- tocstop -->

## Overview

Names are always introduced into some scope which defines where they can be
referenced. Many of these scopes are themselves named. For example, types are
named, and name scopes can be defined directly with `namespace`, see
["Code and name organization"](code_and_name_organization/README.md#namespaces).

### Unqualified name lookup

Unqualified name lookup in Carbon searches the current and enclosing scopes up
to the top-level file scope. Names found include:

-   Locally declared names in the current scope or any enclosing scope, where
    the declaration must precede the reference.
    -   When an out-of-line declaration or definition is qualified by a scope or
        nested scopes (such as `fn Foo.Bar(...)`), the scopes nominated by the
        qualifier (`Foo` and `Foo.Bar`) act as enclosing scopes for unqualified
        name lookup within that declaration's signature and body. This applies
        to namespaces, classes, and interfaces.
-   Names from the current package that are visible by being imported (such as
    with `import library "..."`), including the implicit import of the API file
    from an `impl` file.
-   The implicit "prelude" of importing and aliasing the fundamentals of the
    standard library.

Note that a package's name is not injected into the scope of its files, and
imports within a single package do not specify the package name. Symbols from
other packages must be imported and accessed through their package namespace.

In declarative scopes (such as namespaces, classes, and interfaces), if
unqualified name lookup is performed for a name and fails to find it, that name
is said to be _poisoned_ in that scope. It is an error if a poisoned name is
later introduced in that scope, as that would alter the meaning of earlier
lookups. Within a library, poisoned names persist from the API file to
implementation files.

In sequential scopes (such as function bodies), redeclaring an entity is
disallowed.

### Name lookup for common, standard types

The Carbon standard library is in the `Core` package. A subset of this package,
called the "prelude", is implicitly imported in every file, so the package name
`Core` is always available.

Some keywords and type literals, such as `bool` and `i32`, are aliases for
entities in the prelude. Similarly, some of the Carbon language syntax, such as
operators and `for` loops, is defined in terms of interfaces in the prelude.

## Open questions

### Shadowing

Name shadowing has been discussed in proposals
[#3763: Matching redeclarations](/proposals/p003763-matching-redeclarations.md#use-package-wide-name-poisoning).
One rule under consideration is:

> Always look in all enclosing scopes, and diagnose an ambiguity if an
> unqualified name is found in more than one enclosing scope.

This is aligned with the
[name shadowing discussion in the "low context-sensitivity" principle](/docs/project/principles/low_context_sensitivity.md#name-shadowing).

## References

-   Proposal
    [#107: Code and name organization](https://github.com/carbon-language/carbon-lang/pull/107)
-   Proposal
    [#2287: Allow unqualified name lookup](https://github.com/carbon-language/carbon-lang/pull/2287)
-   Proposal
    [#2550: Simplified package declaration for the `Main` package](https://github.com/carbon-language/carbon-lang/pull/2550)
-   Proposal
    [#3763: Matching redeclarations](https://github.com/carbon-language/carbon-lang/pull/3763)
