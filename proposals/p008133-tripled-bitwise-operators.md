# Tripled bitwise operators

<!--
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

[Pull request](https://github.com/carbon-language/carbon-lang/pull/8133)

<!-- toc -->

## Table of contents

-   [Abstract](#abstract)
-   [Problem](#problem)
-   [Background](#background)
-   [Proposal](#proposal)
-   [Future work](#future-work)
-   [Rationale](#rationale)
-   [Alternatives considered](#alternatives-considered)
    -   [Other symbol sequences](#other-symbol-sequences)
    -   [Keyword operators](#keyword-operators)
    -   [`<<` and `>>` for C++ streaming operator interop](#-and--for-c-streaming-operator-interop)

<!-- tocstop -->

## Abstract

Change Carbon's bitwise and bit-shift operators to use tripled symbol characters
(`&&&`, `|||`, `^^^`, `~~~`, `<<<`, `>>>`) and corresponding compound assignment
operators (`&&&=`, `|||=`, `^^^=`, `<<<=`, `>>>=`).

This frees up `^` for memory safety annotations and `|` for potential future
uses such as sum types, restores distinct spellings for bitwise complement
(`~~~`) and bitwise xor (`^^^`), and avoids overloading infix `&` for both
"bitwise and" and "facet type intersection".

## Problem

We want to free up the `^` operator for use in Carbon's memory safety
annotations. `^` is currently being used as the "bit flip" operator: `^x` is the
bitwise complement (negation) of `x` and `x ^ y` is the bitwise exclusive or
(xor) of `x` and `y`.

We in general prefer operators to only have one meaning. Currently infix `&`
means either "bitwise and" or "facet type intersection" depending on its
operands. If we were to change the spelling of the bitwise operators, it would
avoid overloading infix `&` and free up `|`, perhaps for use in sum types.

Bitwise operations are not common in the code we surveyed, making it hard to
justify using so many short symbol operators. This is why we previously combined
bitwise complement and xor into `^` -- we wanted to use `~` for "move from".

## Background

Carbon's current bitwise and shift operators were decided in
[proposal #1191: "Bitwise and shift operators"](https://github.com/carbon-language/carbon-lang/pull/1191).
The [background from #1191](p001191-bitwise-and-shift-operators.md#background)
is still relevant.

Some other languages use other spellings of the bitwise operators. Several (for
example, Ada, Clojure, Erlang, Forth, Kotlin, Maple, Mathematica, MATLAB,
Pascal) use keywords or named functions instead of symbols. In addition:

-   [F#](https://learn.microsoft.com/en-us/dotnet/fsharp/language-reference/symbol-and-operator-reference/bitwise-operators)
    and [Elixir](https://elixir.hexdocs.pm/1.12/Bitwise.html) use `&&&`, `|||`,
    `^^^`, `~~~`, `<<<`, `>>>` (though Elixir deprecated `^^^` in 1.12 in favor
    of `bxor/2` due to a precedence issue in Elixir's operator table).
-   [Haskell](https://rosettacode.org/wiki/Bitwise_operations#Haskell) uses
    `.&.`, `.|.` for bitwise and and or.

## Proposal

We switch to using the symbol characters from the C/C++ bitwise and shift
operators, each repeated three times like F#. This restores using different
operators for bitwise xor and bitwise complement.

Bitwise operators:

-   `&&&`: bitwise and, infix
-   `|||`: bitwise or, infix
-   `^^^`: bitwise xor, infix
-   `~~~`: bitwise complement, prefix unary

Shift operators:

-   `<<<`: bit shift left, infix
-   `>>>`: bit shift right, infix

Assignment statements:

-   `x &&&= y;` meaning `x = x &&& y;`
-   `x |||= y;` meaning `x = x ||| y;`
-   `x ^^^= y;` meaning `x = x ^^^ y;`
-   `x <<<= y;` meaning `x = x <<< y;`
-   `x >>>= y;` meaning `x = x >>> y;`

None of the semantics, operator interfaces, precedences, or associativities are
changed; `~~~` remains non-repeating, so `~~~(~~~x)` still requires parentheses.
`|`, `^`, `<<`, `>>`, `&=`, `|=`, `^=`, `<<=`, and `>>=` are removed as symbolic
tokens.

Infix `&` remains as an operator on facet types with the same precedence (and
for now is not customizable), and prefix unary `&` remains the address-of
operator.

## Future work

Using three symbol characters for bitwise operators opens the door to adding
more bitwise operators, such as bit rotations.

We may want to allow overloading infix `&`, giving it an operator interface for
customization for user types.

## Rationale

-   [Code that is easy to read, understand, and write](/docs/project/goals.md#code-that-is-easy-to-read-understand-and-write):
    Having distinct operators for distinct operations increases clarity.
-   [Practical safety and testing mechanisms](/docs/project/goals.md#practical-safety-and-testing-mechanisms):
    We are freeing up `^` for use in safety annotations.
-   [Interoperability with and migration from existing C++ code](/docs/project/goals.md#interoperability-with-and-migration-from-existing-c-code):
    We are using the same symbol characters as C++, each repeated three times.
    By switching to tripled symbols, we can restore the correspondence with C++
    `~` for bitwise complement.

## Alternatives considered

### Other symbol sequences

Other multi-character symbol sequences were considered:

-   `\&`, `\|`, ...
-   `%&`, `%|`, ..., using `%` as a prefix modifier (similar to how Swift
    prepends `&` or Zig appends `%` for wrapping arithmetic)
-   `.&.`, `.|.`, ..., as used by Haskell
-   `/\`, `\/`, `(+)`, ...

These were brought up in the
[#syntax channel in Discord starting on 2026-11-09](https://discord.com/channels/655572317891461132/709488742942900284/1569455484187049987)
and
[open discussion on 2026-11-12](https://docs.google.com/document/d/1mjllGO3ZCL4qGt9uJHUtcxKoHAGEY7Y999ie4EtBWB8/edit?pli=1&tab=t.uvyw2c4zrz9e#heading=h.8iwpeulxc8hd).
These did not appeal visually.

### Keyword operators

We decided we wanted bitwise operators to be symbolic instead of keywords
primarily for two reasons:

-   We want to be consistent with our other "numeric calculation" operators.
    None of our arithmetic or comparison operators use keywords; keywords are
    used for logical and control-flow operators (`and`, `or`, `not`,
    `if`-`then`-`else`), casts (`as`), and type or reference operators (`const`,
    `partial`, `where`, `ref`).
-   In code that does use bitwise operators, it tends to use them a lot. Such
    code tends to be the low-level and/or high-performance code that Carbon
    wants to support.

This rationale has not really changed since it was
[considered in proposal #1191](p001191-bitwise-and-shift-operators.md#use-a-multi-character-spelling).
While we are changing from the C++ operators, we hope that using the same symbol
characters (repeated three times) will make the transition easier.

### `<<` and `>>` for C++ streaming operator interop

We considered keeping `<<` and `>>` in order to make interop with C++ streaming
I/O (`cout <<`...) more natural. These operators would be left-associative like
the C++ operators, and be rewritten into new C++ compatibility operator
interfaces, which would only be implemented for C++ types with `operator<<` and
`operator>>` functions that return a reference to the left argument's type.

We decided against this for a few reasons:

-   We can define a variadic function for streaming as an alternative
    workaround.
-   There was an opportunity for confusion during the transition. For example,
    migrated code might use `<<` or `>>` and have it use the C++ operator
    overload when bit shifting was intended. The heuristic of restricting to
    reference-returning functions was intended to limit this.
-   There was a minor concern that JavaScript already makes a different
    distinction between `>>` and `>>>` (arithmetic versus logical shift).
-   There was a dislike of having Carbon operators that exist solely for
    interop, and can never be used in a fully native Carbon application.

This was discussed in
[open discussion on 2026-11-12](https://docs.google.com/document/d/1mjllGO3ZCL4qGt9uJHUtcxKoHAGEY7Y999ie4EtBWB8/edit?pli=1&tab=t.uvyw2c4zrz9e#heading=h.8iwpeulxc8hd)
and
[#syntax in Discord on 2026-11-13](https://discord.com/channels/655572317891461132/709488742942900284/1570832973425803345).
