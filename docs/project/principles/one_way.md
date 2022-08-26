# Principle: Prefer providing only one way to do a given thing

<!--
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

<!-- toc -->

## Table of contents

-   [Background](#background)
-   [Principle](#principle)
-   [Applications of this principle](#applications-of-this-principle)
-   [Caveats](#caveats)
    -   [Specialized syntax](#specialized-syntax)
    -   [Non-obvious alternatives](#non-obvious-alternatives)
    -   [In evolution](#in-evolution)
-   [Alternatives considered](#alternatives-considered)

<!-- tocstop -->

## Background

It's common in programming languages to provide multiple, similar ways of doing
the same thing. Sometimes this reflects the legacy of a language, and
difficulties in evolving in ways that would require changes to
developer-authored code, thereby retaining backwards compatibility. Other times
it reflects a desire to provide both verbose and concise versions of the same
syntax. We are concerned with both forms.

We also are cautious about creating alternatives that may give rise to a
[paradox of choice](https://en.wikipedia.org/wiki/The_Paradox_of_Choice),
wherein options are similar enough that developers actively spend time analyzing
trade-offs, and the time spent that way outweighs the potential benefits of a
correct choice.

Where multiple, similar implementation options exist, it can sometimes give rise
to style guidelines to indicate a preferential choice; sometimes because one
option is objectively better, but sometimes because making a choice is better
than not making one. Even with a style guide, developers may diverge in style by
accident or intent, choosing different coding patterns simply because either
option works. It can also become an issue as developers move between an
organization that they need to learn a new style guide, and relearn habits.

A couple examples of this in other languages are:

-   In Perl,
    ["There is more than one way to do it."](https://en.wikipedia.org/wiki/There%27s_more_than_one_way_to_do_it)
-   In Python,
    ["There should be one -- and preferably only one -- obvious way to do it."](https://www.python.org/dev/peps/pep-0020/)

## Principle

In Carbon, we will prefer providing only one way to do a given thing. That is,
given a syntax scenario where multiple design options are available, we will
tend to provide _one_ option rather than providing several and letting users
choose. This echoes Python's principle.

Minimizing choices serves several goals:

-   [Language tools](/docs/project/goals.md#language-tools-and-ecosystem) should
    be easier to write and maintain with the lower language complexity implied
    by less duplication of functionality.
-   [Software and language evolution](/docs/project/goals.md#software-and-language-evolution)
    processes should find it easier to both consider existing syntax and avoid
    creation of new syntax conflicts.
-   [Understandability of code](/docs/project/goals.md#code-that-is-easy-to-read-understand-and-write)
    should be promoted if developers have less syntax they need to understand.
    This can be expected to improve code quality and productivity so long as the
    resulting code structures aren't overly complicated.

By minimizing the overlap of language features, we hope to make work easier for
both Carbon's maintainers and developers.

## Applications of this principle

We can observe the application of this principle by comparing several language
features to C++. There, improving understandability is frequently the primary
motivation:

-   Where C++ allows logical operators to be written with either symbols (for
    example, `&&`) or text (for example, `and`), Carbon will only support one
    form (in this case, [text](/proposals/p0680.md)).
-   Where C++ allows hexadecimal numeric literals to be either lowercase
    (`0xaa`) or uppercase (`0xAA`), and with `x` optionally uppercase as well,
    Carbon will only allow the [`0xAA` casing](/proposals/p0143.md).
-   Where C++ provides both `struct` and `class` with the only difference is
    access control defaults, Carbon will only provide one (`class`, albeit with
    default public visibility diverging from C++).

However, sometimes language tools are the primary motivation. For example, where
C++ allows braces to be omitted for single-statement control flow blocks, Carbon
will [require braces](/proposals/p0623.md). This offers a syntax simplification
that should allow for better error detection.

## Caveats

### Specialized syntax

Sometimes overlap will occur because a specialized syntax offers particular
benefits, typically as a matter of convenience for either a common use-case or a
particularly complex and important use-case. Some examples of why and where this
occurs are:

-   For [performance](/docs/project/goals.md#performance-critical-software), it
    may at times be necessary to provide a specialized syntax that better
    supports optimization than a generic syntax.
-   For
    [understandability of code](/docs/project/goals.md#code-that-is-easy-to-read-understand-and-write),
    there may be times that a particular use-case is common enough that
    simplifying its syntax provides substantial benefit.
    -   For example, `for (var x: auto in list)` could typically be written with
        as a `while` loop, but range-based for loops are considered to improve
        understandability. However, C++'s `for (;;)` syntax is sufficiently
        close to `while` that we expect to use `while` to address the
        corresponding use-cases.
-   For
    [migration and interoperability](/docs/project/goals.md#interoperability-with-and-migration-from-existing-c-code),
    it may be pragmatic to provide both an ideal way of doing things for new
    Carbon code, and a separate approach that is more C++-compatible for
    migration.
    -   For example, consider generics and templates: generics are considered to
        be the preferred form for new code, but templates are considered a
        necessity for migration of C++ code. This is not an evolution situation
        because we do not anticipate ever removing templates.

### Non-obvious alternatives

Echoing Python, there may be non-obvious alternative ways of doing a given
thing, such as using `while (condition) { DoSomething(); break; }` in place of
`if (condition) { DoSomething(); }`. As a more complex example, lambdas could be
implemented using other code constructs; this would require significantly more
code and hinder understandability.

This kind of overlap may exist, but will hopefully be considered sufficiently
non-idiomatic that examples won't be common in code. If a choice would not
likely be based mainly on coding styles, it's likely sufficiently distinct that
this principle won't apply.

### In evolution

For [evolution](/docs/project/goals.md#software-and-language-evolution), it will
often be necessary to temporarily provide an "old" and "new" way of doing things
simultaneously.

For example, if renaming a language feature, it may be appropriate to provide
the same functionality under two identifiers. However, one should be marked as
deprecated and eventually removed. We should be cautious of adding new,
overlapping features without a plan to remove the corresponding legacy version.

## Alternatives considered

-   [Provide multiple ways of doing a given thing](/proposals/p0829.md#provide-multiple-ways-of-doing-a-given-thing)
