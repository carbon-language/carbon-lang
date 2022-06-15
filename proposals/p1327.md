# Generics: `impl forall`

<!--
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

[Pull request](https://github.com/carbon-language/carbon-lang/pull/1327)

<!-- toc -->

## Table of contents

-   [Problem](#problem)
-   [Background](#background)
-   [Proposal](#proposal)
-   [Rationale](#rationale)
-   [Alternatives considered](#alternatives-considered)

<!-- tocstop -->

## Problem

We have an ambiguity in the grammar between declaring a parameterized impl and
an impl for an array type.

```
// Parameterized impl
external impl [T:! Printable] Vector(T) as Printable { ... }
// Impl for an array type
external impl [i32; 5] as Printable { ... }
```

When the parser sees `impl [`, it doesn't know which kind of impl declaration it
is parsing without more lookahead than we plan to support. For the same reason,
these declarations are easy for humans to confuse.

## Background

Parameterized impls introduced in
[#920: Generic parameterized impls (details 5)](https://github.com/carbon-language/carbon-lang/pull/920).

Array syntax has not been finalized, but the leading contender is `[i32; 5]`,
similar to
[Rust](https://doc.rust-lang.org/rust-by-example/primitives/array.html). This is
what is currently provisionally implemented in Explorer. Some other contenders
also start with `[` and have the same problem.

This problem was discussed and resolved in question-for-leads issue
[#1192: Parameterized impl syntax](https://github.com/carbon-language/carbon-lang/issues/1192).

## Proposal

This proposal implements the decision in
[#1192](https://github.com/carbon-language/carbon-lang/issues/1192) to write
parameterized impls using this syntax:

> `impl forall [`_generic parameters_`]` _type_ `as` _constraint_ ...

and to remove the option to includes bindings in the _type_ `as` _constraint_
part of the declaration.

This PR includes the changes to
[the generics details design doc](/docs/design/generics/details.md).

## Rationale

This decision favored approaches that did not require more lookahead. This is to
simplify compiler and tool development and to make it easier for humans to read
the code, in support of these goals:

-   [Language tools and ecosystem](/docs/project/goals.md#language-tools-and-ecosystem)
-   [Code that is easy to read, understand, and write](/docs/project/goals.md#code-that-is-easy-to-read-understand-and-write),
    particularly "Excellent ergonomics", "Support tooling at every layer of the
    development experience, including IDEs", and "Design features to be simple
    to implement."

## Alternatives considered

@zygoloid listed some options for addressing this problem
[in the #syntax channel on Discord](https://discord.com/channels/655572317891461132/709488742942900284/963191891334168628):

> Summary of options for implicit parameters / arrays ambiguity discussed so
> far:
>
> 1. Just make it work as-is: `impl [a; b]` parses as an array type,
>    `impl [a, b]` parses as an implicit parameter. Theoretically this is
>    unambiguous given that a `;` is required inside the `[`...`]` in the former
>    and disallowed in the latter. Concerns: it's likely to be visually
>    ambiguous.
> 2. Add mandatory parentheses: `impl [T:! Type] (Vector(T) as Container)`.
>    Concerns: it's hard to avoid requiring them in cases that don't start with
>    a `[` if we want an unambiguous grammar. Requiring them always would impose
>    a small ergonomic hit.
> 3. Add an introducer keyword for implicit parameters:
>    `impl where [T:! Type] Vector(T) as Container`. Unambiguous. Concerns:
>    still some visual ambiguity due to reuse of `[`...`]`, concern over whether
>    we'd uniformly use this syntax (`fn F where [T:! Type](x: T)`) or have
>    non-uniform syntax for implicit parameters.
> 4. Use a different syntax for array types in general:
>    `impl Array(T) as Container` or `impl Array[N] as Container`. Concerns: may
>    want a first-class syntax here, especially if (per @geoffromer 's variadics
>    work, we want some special behavior for a deduced bound), and there's a
>    strong convention to use `[`...`]` for this. The latter syntax is messy
>    because of our types-as-expressions approach, but we could imagine
>    providing a `impl Type as Indexable where .Result = Type` to construct
>    array types. `T[]` might be a special case of some kind.
> 5. Use a different syntax for implicit parameters in general:
>    `impl<T:! Type> Vector(T) as Container`. Concerns: we don't have many
>    delimiter options unless we start using multi-character delimiters; `()`,
>    `[]`, and `{}` are all used for types, leaving `<>` as the only remaining
>    bracket. Use of `<>` as brackets as a long history but not a good one. ...
> 6. Remove the implicit parameter list from impls and force them to be
>    introduced where they're first used: `impl Vector(T:! Type) as Container`.
>    Concerns: harms readability in some cases, eg
>    `impl Optional(T:! As(U:! Type)) as As(Optional(U))` versus
>    `impl [U:! Type, T:! As(U)] Optional(T) as As(Optional(U))`.
> 7. Move the implicit parameter list before the impl keyword, perhaps with an
>    introducer: `generic [T:! Type] impl Vector(T) as Container`. Concerns:
>    increases verbosity; would be inconsistent if we put everything but me
>    there, and surprising if we put me there. Also not clear what a good
>    keyword is, given that the existence of deduced parameters isn't the same
>    as an entity being generic.

Ultimately we adopted approach 3, but changed to the new keyword `forall` to
avoid overloading the meaning of a keyword used for something else.
