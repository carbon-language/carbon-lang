# Tuples and tuple indexing

<!--
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

[Pull request](https://github.com/carbon-language/carbon-lang/pull/3646)

<!-- toc -->

## Table of contents

-   [Abstract](#abstract)
-   [Problem](#problem)
-   [Background](#background)
-   [Proposal](#proposal)
-   [Details](#details)
    -   [Lexing](#lexing)
    -   [Indexes as names](#indexes-as-names)
    -   [Precedence](#precedence)
    -   [Expression operand](#expression-operand)
    -   [Bounds](#bounds)
    -   [Tuple slicing](#tuple-slicing)
-   [Rationale](#rationale)
-   [Alternatives considered](#alternatives-considered)
    -   [Alternative lexing rule](#alternative-lexing-rule)
    -   [Decimal indexing restriction](#decimal-indexing-restriction)
    -   [Square bracket notation](#square-bracket-notation)
    -   [Negative indexing from the end of the tuple](#negative-indexing-from-the-end-of-the-tuple)
    -   [Trailing commas](#trailing-commas)

<!-- tocstop -->

## Abstract

Add support for extracting elements of a tuple by their numerical index.

Also formally add the well-established basic syntactic and semantic rules for
tuples, for which we have had leads issues but no proposal, into the design.

## Problem

Currently, the only way to access the elements of a tuple is through pattern
matching. While this handles many cases well, it is sometimes desirable to
access an element of a tuple more succinctly, especially in cases where only a
single element's value is needed.

## Background

In Python, tuple indexing is performed using square brackets:

```python
tup = (1, 2, 3)
# Prints 2.
print(tup[1])
```

In C++, `std::pair` is indexed using `.first` and `.second`, and `std::tuple` is
indexed using `std::get<I>`.

In Rust and Swift, a tuple is indexed using `.N`, where `N` is a decimal integer
literal.

-   Rust disallows digit separators and base prefixes in `N`, but allows certain
    literal suffixes
    [for historical reasons](https://github.com/rust-lang/rust/issues/60210).
-   Swift disallows digit separators and base prefixes in `N`. `swiftc` allows
    leading `0` digits, although this appears to be an unintentional consequence
    of `llvm::StringRef::getAsInteger` allowing them.

The current Carbon documentation suggests using `tuple[i]` for tuple indexing,
but this has not been the subject of an approved proposal.

## Proposal

Formally, we have not yet approved a proposal that says that Carbon has tuple
types, although we have approved several proposals that explicitly include
support for tuples. So, this proposal does that: tuples exist in Carbon, and are
product types with unnamed positional elements.

This proposal also updates the design to match other decisions that have been
made in leads issues but not captured by a proposal, specifically:

-   Leads issue #2191 (one-tuples and one-tuple syntax), despite being focused
    on one-tuples, established the syntax for tuples of all arities.
-   Leads issue #710 established rules for assignment, comparison, and implicit
    conversion of tuples. These operations are performed elementwise, with
    relational comparisons being performed lexicographically.

Finally, the main intent of this proposal is to add support for indexing tuples,
using the following syntaxes:

-   `.` _N_, where _N_ is an integer literal, and
-   `.` `(` _expr_ `)`, where _expr_ is a template constant of integer type.

For pointers to tuples, `->` _N_ and `->` `(` _expr_ `)` are also supported.

## Details

### Lexing

Multi-level tuple indexing will result in constructs such as
`tuple_of_tuples.1.2`. It's important that these are lexed as two tuple indexing
operations, not as `tuple_of_tuples` `.` `1.2`, as it would be under the current
lexical rules, so a new rule is introduced:

-   When a `.` or `->` token is followed immediately by a digit, it is lexed as
    a `.` or `->` token followed by an integer literal, never a real literal.

Note that this results in lexing being slightly contextual: the rule to lex a
token after a `.` or `->` is different from the rule to lex a token in any other
context. However, there is an alternative equivalent formulation of the rule
that is not context-sensitive: that `.integer` is treated as a single lexeme
that produces two tokens, and likewise for `->integer`.

### Indexes as names

The elements of a tuple are treated as if they had decimal integers as their
names: `.0`, `.1`, and so on. It is an error to use a different spelling of that
integer in a simple member access, because that spelling would not match the
element name. For example, `(1, 2).0x0` is invalid, as is `large_tuple.1_2`.
These spellings can be used as an [expression operand](#expression-operand) as
described below: `(1, 2).(0x0)` and `large_tuple.(1_2)` are both valid.

### Precedence

The `.` _N_ syntax has the same precedence as postfix member access syntax, `.`
_name_, and can be combined in the same expression: `a.0.x.1` is valid.

The `.` `(` _expr_ `)` syntax is not new in this proposal, and continues to have
the same precedence as `.` _name_.

### Expression operand

In the `.` `(` _expr_ `)` syntax, if the first operand is a tuple and the second
operand is a constant of any integer type, the result is the corresponding tuple
element, as if specified by a decimal integer literal. This rule is built into
the language; the `.` `(` ... `)` notion is not currently overloadable.

### Bounds

If the tuple index is not between 0 and one less than the number of elements in
the tuple, inclusive, the indexing is invalid.

### Tuple slicing

The current skeleton design suggests using `tuple[a .. b]` to slice tuples. For
example, `tuple[0 .. 2]` could be used to extract the first two elements of a
tuple. Tuple slicing support is not covered by this proposal, but could be added
in the future with syntax such as `tuple.(0 .. 2)`. However, note that there is
a risk that this syntax may lead to an incorrect theory about how Carbon works:
namely, that `tuple.__` gives an element whereas `tuple.(__)` gives a tuple.

## Rationale

Goals:

-   [Language tools and ecosystem](/docs/project/goals.md#language-tools-and-ecosystem)
    -   The lexing rule is relatively simple to implement. Tools such as syntax
        highlighters can treat `.i` as a distinct kind of token rather than
        implementing any kind of context-sensitive lexing.
-   [Software and language evolution](/docs/project/goals.md#software-and-language-evolution)
    -   Consistent use of tuple field indexes can be used to support code that
        adds new tuple elements over time.
-   [Code that is easy to read, understand, and write](/docs/project/goals.md#code-that-is-easy-to-read-understand-and-write)
    -   This feature allows tuple access to be written more concisely than
        pattern matching would allow.
    -   Lexing `.1.2` as four tokens rather than two avoids a surprise that
        would make chained member access hard to write.
    -   For simple member access, requiring a decimal integer with no digit
        separators allows the member access to be treated as an element name,
        making the indexing easier to understand.
-   [Interoperability with and migration from existing C++ code](/docs/project/goals.md#interoperability-with-and-migration-from-existing-c-code)
    -   This feature provides a migration syntax for existing use of `.first`,
        `.second`, and `std::get<I>`. The permission to use expressions rather
        than only literals supports migration of `std::get<expression>`.

Principles:

-   [Low context sensitivity](/docs/project/principles/low_context_sensitivity.md).
    -   We look only at the character immediately before a numeric literal to
        determine whether it is lexed as a tuple index that stops before the
        next `.` or as a general numeric literal.

## Alternatives considered

### Alternative lexing rule

We could lex `.0`, `.1`, ... as a single token rather than as separate `.` and
`0`, `1`, ... tokens. This would somewhat simplify the lexing rules, because
they would no longer be contextual. We choose to not do this because:

-   This would be inconsistent with our handling of `struct.fieldname`.
-   Either `tuple . 0` would be invalid, unlike `struct.fieldname`, or it would
    need to use a distinct grammar production from `tuple.0`.

We could lex an integer literal when the previous token is `.`, regardless of
whether the literal follows the `.` immediately. For example, we could treat

```carbon
let n: i32 = ((1, 2, 3), 4) . 0.1;
```

as tuple indexing, rather than as a tuple followed by a `.` and a real literal.
This is what Swift does. We choose to not do this because:

-   The `0.1` literal in this case looks like a real literal, not tuple
    indexing, so this would likely cause surprise for readers.
-   This would make the context-sensitive lexing be non-local. The chosen rule
    can be interpreted as lexing `.[0-9]*` as a single lexeme, but forming two
    tokens from it, whereas this alternative rule would be much more firmly a
    context-sensitive lexing rule.

We could get a similar result in other ways:

-   We could allowing a real literal after a `.`, and split it into a pair of
    member accesses when needed. This is
    [what `rustc` does](https://github.com/rust-lang/rust/pull/71322).
-   We could lex a real literal as three tokens: an integer token, a `.` token,
    and a suffix token, and merge them back together in the parser. This is
    [what `intellij` does](https://github.com/intellij-rust/intellij-rust/commit/f82f6cd68567e574bf1e30f5e0d263ee15d1d36e)
    when parsing Rust.

Note that these approaches are not entirely equivalent to each other. In Rust,
for example, the difference is observable in proc macros. Also, using any kind
of token merging or splitting approach would result in the token stream not
matching the interpretation of the program, which is problematic for tooling.
For example, many common Rust syntax highlighters do not properly highlight
chained tuple indexing.

### Decimal indexing restriction

Carbon follows Rust and Swift in restricting tuple indexes to being decimal
integers:

```carbon
// OK
let a: i32 = (1, 2, 3).0;

// Error, invalid index for tuple element.
let b: i32 = (1, 2, 3).0x0;
```

This restriction introduces an inconsistency between `.0x0` and `.(0x0)`, and we
could easily drop it. However, the restriction allows us to consider `.0`, `.1`,
and so on to simply be the names of the tuple elements, analogous to struct
field names, and there isn't a clear utility for permitting a base prefix or a
digit separator in a tuple index.

### Square bracket notation

Instead of `tuple.0` and `tuple.(IndexConstant)`, we could use `tuple[0]` and
`tuple[IndexConstant]`. This would result in more consistent syntax for indexing
with a constant versus with an expression, but would make accessing an element
of a tuple less consistent with accessing an element of a struct. We expect
tuple access with a non-literal index to be a rare operation, so the consistency
with that syntax seems to have lower value.

Also, the use of `.` notation aims to convey the intent of the developer better:
we intend `x[n]` notation to be used primarily for _homogenous_ indexing,
whereas `.` notation is used for _heterogenous_ access. This also reflects the
difference in phase: tuple indexing requires a constant index in the same way
that struct member access requires a constant name, whereas array or container
indexing would typically be expected to permit a runtime index.

The `.N` notation can also be extended to perform member indexing into a struct
or class, at least the latter of which would not be reasonable to support with
`[]` notation. However, such support is not part of this proposal.

Use of `[]` notation has the advantage of reducing visual ambiguity for cases
such as `O.0`, `l.0`, and `Z.0`, which might be visually confused with `0.0`,
`1.0`, and `2.0`, respectively. However, we're not aware of this being a problem
in practice in Rust or Swift, which use this notation, and the same problem
exists even without the `.0` suffix: `F(O, l, Z)` may resemble `F(0, 1, 2)`.

### Negative indexing from the end of the tuple

We could support `tuple.-1`, or perhaps `tuple.(-1)`, as a notation for "the
last element of the tuple", as used for example in Python. We choose not to
support this at this time because such notation can be confusing and has awkward
edge cases. An off-by-one error, or an attempt to access a one-past-the-start
element, will sometimes be accepted and silently do the wrong thing.

If a future proposal introduces tuple slicing, it should revisit this question,
because this kind of indexing from the end is often desirable when forming a
slice. The possibility of using a different notation for this operation should
be considered, such as `tuple.(.size - 1)`.

### Trailing commas

Carbon permits optional trailing commas in tuples, with mandatory trailing
commas for one-tuples. Alternatives to this choice were considered in
[leads issue #2191](https://github.com/carbon-language/carbon-lang/issues/2191).
