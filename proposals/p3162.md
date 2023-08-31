# Reduce ambiguity in terminology

<!--
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

[Pull request](https://github.com/carbon-language/carbon-lang/pull/3162)

<!-- toc -->

## Table of contents

-   [Abstract](#abstract)
-   [Problem](#problem)
-   [Background](#background)
-   [Proposal](#proposal)
-   [Details](#details)
-   [Rationale](#rationale)
-   [Alternatives considered](#alternatives-considered)

<!-- tocstop -->

## Abstract

Change terminology away from terms that are ambiguous:

-   Reserve "generic type" for types with (compile-time) parameters, like
    `Vector` in `Vector(T:! type)`. Don't use that term to refer to `T`, as it
    would with [#2360](/proposals/p2360.md#terminology).
-   Use the term "compile-time" instead of "constant" to mean "template or
    symbolic." Expand the term "constant" to include values, such as from `let`
    bindings.

## Problem

Right now, the term "generic type" has two meanings. In this example:

```
class Vector(T:! type);
```

Both `Vector` and `T` could be called a "generic type." It would be much less
confusing if one of those two would have a different name.

Similarly, "constant" can currently mean multiple things:

-   "evaluation at compile time," as in "constant evaluation"
-   "not variable," as in `let` instead of `var`
-   "non-mutating view," as in `const T*`

In practice, this has resulted in confusion. For example, the term "constant
bindings" doesn't include all `let` bindings, even though they are not variable
bindings.

## Background

The two meanings of "generic type" come from:

-   Proposal [#2360](/proposals/p2360.md#terminology) defines a generic type to
    be a type or facet introduced by a `:!` binding, such as in a generic
    parameter or associated constant.
-   Other uses of the term generic, such as in generic function, mean a language
    construct with a compile-time parameter (as in
    [Rust](https://doc.rust-lang.org/rust-by-example/generics.html)). This is
    the usage in the broader programming language community, and includes
    calling parameterized types "generic types" (as in
    [Java](https://docs.oracle.com/javase/tutorial/java/generics/types.html),
    [.NET](https://learn.microsoft.com/en-us/dotnet/standard/generics/#terminology),
    [Swift](https://docs.swift.org/swift-book/documentation/the-swift-programming-language/generics/#Generic-Types)).

Issue
[#1391: New name for "constant" value phase](https://github.com/carbon-language/carbon-lang/issues/1391)
implemented in proposal
[#2964: Expression phase terminology](https://github.com/carbon-language/carbon-lang/pull/2964),
expanded the term "constant" from referring to just template constants to also
include symbolic constants from checked generics. Since then proposal
[#2006: Values, variables, pointers, and references](https://github.com/carbon-language/carbon-lang/pull/2006)
introduced the `const` modifier on types, providing a read-only view.

## Proposal

We make these changes:

-   Reserve "generic type" for types with (compile-time) parameters, like
    `Vector` in `Vector(T:! type)`. Don't use that term to refer to `T`, as it
    would with
    [#2360](https://github.com/carbon-language/carbon-lang/blob/trunk/proposals/p2360.md#terminology).
-   Expand "constant binding" to include all `let` bindings.
-   Use the term "compile-time binding" to refer to the collection of template
    and symbolic bindings, such as from generic parameters and associated
    constants.
-   Similarly "compile-time constants" to refer to template and symbolic
    constants, as opposed to just "constants."
-   The term "compile-time parameter" may be used instead of "generic
    parameter." For now, both terms will be used, but in the future it might be
    clearer to only use "generic" to mean "has compile-time parameters."
-   Only use "symbolic binding" and "template binding" not "symbolic constant
    binding" nor "template constant binding."
-   Where applicable, switch from talking about parameters to bindings, since
    almost everything that applies to compile-time parameters also applies to
    compile-time bindings. For example, see
    ["binding patterns"](/docs/design/generics/terminology.md#bindings) and
    ["facet binding"](/docs/design/generics/terminology.md#facet-binding) in the
    generics terminology doc.

## Details

The following design documents have been updated in this proposal to reflect
these changes:

-   [Language design overview](/docs/design/README.md)
-   [Generics terminology](/docs/design/generics/terminology.md)
-   [Member access expressions](/docs/design/expressions/member_access.md)

Some of these changes have already been implemented in:

-   [PR #3048: Update Generics terminology document](https://github.com/carbon-language/carbon-lang/pull/3048)
-   [PR #3061: Update generics overview](https://github.com/carbon-language/carbon-lang/pull/3061)

## Rationale

This proposal advances these goals of Carbon:

-   [Language tools and ecosystem](/docs/project/goals.md#language-tools-and-ecosystem):
    Having clear and unambiguous terminology is important for making a precise
    language specification and as well as other design and developer
    documentation.
-   [Code that is easy to read, understand, and write](/docs/project/goals.md#code-that-is-easy-to-read-understand-and-write):
    particularly the sub-goal that "The behavior and semantics of code should be
    clearly and simply specified whenever possible."

## Alternatives considered

The main alternative considered was the status quo. This was discussed:

-   [starting 2023-Jul-14 in #naming](https://discord.com/channels/655572317891461132/963846118964350976/1129542605538074777)
-   [2023-Jul-27 in open discussion](https://docs.google.com/document/d/1gnJBTfY81fZYvI_QXjwKk1uQHYBNHGqRLI2BS_cYYNQ/edit?resourcekey=0-ql1Q1WvTcDvhycf8LbA9DQ#heading=h.3tki3lncihf)
-   [2023-Aug-28 in #naming](https://discord.com/channels/655572317891461132/963846118964350976/1145790947083423777)

During those discussions, we also considered "symbolic type" instead of "generic
type". This did not work out, though, since it conflicted with the term
"symbolic" used as an expression phase. We considered other possibilities:
"figurative type", "computed type", "hole type", "open type", and "placeholder
type" (though that might be better applied to `auto`).

It also came up that we did not want to use the term "generic binding" to mean a
compile-time binding, because that term would be better applied to a
parameterized binding, see
[#naming on 2023-Jul-31](https://discord.com/channels/655572317891461132/963846118964350976/1135704682128494712).
Those are not currently supported in Carbon, but are something we are likely to
add. For example,
[Rust has generic associated types](https://rust-lang.github.io/generic-associated-types-initiative/explainer/motivation.html)
as of [v1.65](https://blog.rust-lang.org/2022/10/28/gats-stabilization.html).
