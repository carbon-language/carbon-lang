# `char` redesign

<!--
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

[Pull request](https://github.com/carbon-language/carbon-lang/pull/6710)

<!-- toc -->

## Table of contents

-   [Abstract](#abstract)
-   [Problem](#problem)
-   [Background](#background)
-   [Proposal](#proposal)
-   [Details](#details)
    -   [Add a `char` type literal](#add-a-char-type-literal)
        -   [Escape sequences](#escape-sequences)
    -   [Add a `Core.CharLiteral` type for character literals](#add-a-corecharliteral-type-for-character-literals)
    -   [Operators](#operators)
        -   [Conversion operators](#conversion-operators)
        -   [Comparison operators](#comparison-operators)
        -   [Arithmetic operators](#arithmetic-operators)
            -   [`char` integer parameters](#char-integer-parameters)
            -   [Overflow semantics](#overflow-semantics)
            -   [Preferring i32 returns](#preferring-i32-returns)
    -   [Revoke and replace proposal #1964: Character Literals](#revoke-and-replace-proposal-1964-character-literals)
-   [Rationale](#rationale)
-   [Future work](#future-work)
-   [Alternatives considered](#alternatives-considered)
    -   [Align `char` fully with C++, or make it fully valid](#align-char-fully-with-c-or-make-it-fully-valid)
    -   [Raw character literals](#raw-character-literals)
    -   [Disallow hex escape sequences in character literals](#disallow-hex-escape-sequences-in-character-literals)
    -   [Allow grapheme clusters in character literals](#allow-grapheme-clusters-in-character-literals)
    -   [Reuse string literal syntax for character literals](#reuse-string-literal-syntax-for-character-literals)
        -   [Treat single-character string literals as a third "text literal" type](#treat-single-character-string-literals-as-a-third-text-literal-type)

<!-- tocstop -->

## Abstract

-   Add a `char` type literal mapping to `Core.Char` and equivalent to C++'s
    `char`.
    -   8 bits, unsigned, treated as a single UTF-8
        [code unit](https://en.wikipedia.org/wiki/Character_encoding#Code_unit).
-   Add a `Core.CharLiteral` type for character literals, similar to
    `Core.IntLiteral`.
-   Allow operations for `char` and `Core.CharLiteral` which reinforce the
    "character" concept, versus an integer value.
-   Revokes and replaces
    [#1964: Character Literals](https://github.com/carbon-language/carbon-lang/pull/1964).

## Problem

`char` is an important type due to its common use in C++ code. However, the
related proposal
[#1964: Character Literals](https://github.com/carbon-language/carbon-lang/pull/1964)
has several issues, including:

-   Lacks a decision for `char` handling; it is not mentioned in proposal #1964.
    -   Similarly, decides there are character literals, but more detail is
        needed for implementation.
-   Type literal naming no longer reflects naming consensus.
    -   `Char8` seems potentially more equivalent to `std::char8_t` instead of
        `char`, and for interop purposes these are slightly different types.
        Similar applies to `Char16` and `Char32`.
    -   As a design direction, we have been lowercasing type literals (such as
        `u8`).
-   Conflicting statements about behavior.
    -   For example, "Rationale" states that `var b: u8 = 'a' + 1` would be
        supported, while "Operations" states that `+` is returning a character
        literal (not a `u8`).
    -   For character literals, states "Escape sequences which would result in
        non-UTF-8 encodings or more than one code point are not included."
        However, it goes on to say that `let smiley: Char16 = '\u{1F600}'` is
        valid even though `1F600` would require multiple code units in both
        UTF-8 and UTF-16.
-   Unclear that it gives us a good UTF plan.
    -   Does not decide what a single character in a Carbon string is.
    -   No consideration regarding interop with the `std::char32_t` family of
        types or [ICU](https://github.com/unicode-org/icu) compatibility.

In other words, it's likely we want something similar to `Char32`, but it may be
named something like `Core.Char32` and have slightly different type behaviors
than decided in #1964. On the other hand, we need something compatible with the
C++ `char` in order to proceed with basic C++ interop, and #1964 doesn't provide
that.

## Background

-   [Proposal #1964: Character Literals](https://github.com/carbon-language/carbon-lang/pull/1964)
    is fundamental, and a lot of the underlying thoughts still apply. In
    particular, we still want character types to be distinct from numeric types.
-   [Proposal #199: String literals](https://github.com/carbon-language/carbon-lang/pull/199)
    is important because we want character and string literals to have mirrored
    escaping concepts.
-   [Proposal #5448: Carbon &lt;-> C++ Interop: Primitive Types](https://github.com/carbon-language/carbon-lang/pull/5448)
    left the question of character type mappings open. This proposal aims to
    answer it for `char`.
-   [Issue #5903: Built-in character type questions](https://github.com/carbon-language/carbon-lang/issues/5903)
    addressed type questions.
-   [Issue #5922: Built-in character operators](https://github.com/carbon-language/carbon-lang/issues/5922)
    addressed operators.

## Proposal

The way `char` will work is:

-   Add a `char` type literal.
    -   Carbon's `str` type will use `char` for elements.
    -   For interop, map Carbon's `char` to C++'s `char`.
-   Add a `Core.CharLiteral` type for character literals, similar to
    `Core.IntLiteral`.
-   Provide operators which are consistent with the character concept.

This proposal additionally revokes and replaces proposal #1964, rather than
trying to define which parts we are keeping and which are changing.

## Details

### Add a `char` type literal

`char` is intended to offer a basic construct for Carbon's strings that is both
compatible with UTF-8, and has high fidelity with C++ strings.

In support of that, important notes are:

-   `char` itself will be a
    [type literal](https://github.com/carbon-language/carbon-lang/blob/trunk/docs/design/lexical_conventions/words.md#type-literals).
-   `char` notionally represents a UTF-8 code unit.
    -   It can contain invalid code units, as long as it remains 8 bits. We do
        not assume runtime validation.
-   `char` will be backed by `Core.Char`, in the prelude.
    -   `Core.Char` will adapt `u8`.
-   C++ interoperability will transparently map `char` and `Cpp.char` on API
    boundaries.
    -   When used with Carbon, C++ `char` will be unsigned by default
        (`-funsigned-char`). A program can switch back to signed
        (`-fno-unsigned-char`), and Carbon will maintain interoperability but
        bits will be interpreted differently in each language.

#### Escape sequences

Escape sequences are the same as for a string literal. Only one escape sequence
may be provided in a character literal.

### Add a `Core.CharLiteral` type for character literals

`Core.CharLiteral` is the type of a character literal, similar to how
`Core.IntLiteral` is the type of integer literals. It abstractly represents a
single Unicode code point. This gives us a compile-time structure for characters
that can be typed and referred to in programs.

Semantics of a character literal will be equivalent to a
[simple string literal](https://github.com/carbon-language/carbon-lang/blob/trunk/docs/design/lexical_conventions/string_literals.md#simple-and-block-string-literals),
except that:

-   A character literal has a validated Unicode code point value.
-   The enclosing character is `'`.
-   The contents are precisely one character or escape sequence.
    -   The `\x` escape sequence is limited to values up to `7F`, where the
        UTF-8 code unit and Unicode code point values are identical.

An important detail of the character literal type is it gives us a way to track
constant values at compile time. For example, `'a' + 1` has a constant value of
`b`. This means we can diagnose uses of character literals that don't represent
a valid Unicode code point, such as `'a' + 0xFFFFFF`.

### Operators

The goal of provided operators is to provide a set of operators which map to
common operations a user would want to do. It is a non-goal to support use of
`char` as an arbitrary byte or integer: developers should use `u8` for that.

In general, `char` and `Core.CharLiteral` operators are intended to be mirrors
of each other.

#### Conversion operators

-   `char`
    -   `ImplicitAs`: None
    -   `ExplicitAs`: To/from `u8`, plus the set of `ImplicitAs` for `u8`.
        -   For example, `u8` has `ImplicitAs` to `u16`, so `char` has
            `ExplicitAs` to `u16`.
-   `Core.CharLiteral`
    -   `ImplicitAs`: to `char` only
    -   `ExplicitAs`: To/from the set of `ImplicitAs` for `i32` and `u32`.
        -   For example, `i32` has `ImplicitAs` to `i64`, so `Core.CharLiteral`
            has `ExplicitAs` to `i64`.
        -   For example, `i64` does not have `ImplicitAs` to `i32`; conversion
            requires two casts, `((i64_val as i32) as Core.CharLiteral)`.

Casting from a `char` to a `Core.CharLiteral` is not supported.

See also
[implicit numeric conversions](https://github.com/carbon-language/carbon-lang/blob/trunk/docs/design/expressions/implicit_conversions.md#data-types).

#### Comparison operators

-   `char`
    -   `EqWith` and `OrderedWith` when both operands are `char`.
    -   `ImplicitAs` should allow substituting one operand with
        `Core.CharLiteral`.
-   `Core.CharLiteral`
    -   `EqWith` and `OrderedWith` when operands are `Core.CharLiteral`.

#### Arithmetic operators

-   `char`
    -   `AddWith`: `char + &lt;integer> -> char` (with reversible operands)
        -   Equivalent to `(((char as i16) + &lt;integer>) as u8) as char)`
    -   `SubWith`:
        -   `char - &lt;integer> -> char` (non-reversible operands)
            -   Equivalent to `(((char as i16) - &lt;integer>) as u8) as char)`
        -   `char - char -> i32`
            -   Equivalent to `(lhs as i32) - (rhs as i32)`.
            -   `ImplicitAs` should allow substituting one operand with
                `Core.CharLiteral`.
-   `Core.CharLiteral`
    -   `AddWith`: `Core.CharLiteral + &lt;integer> -> Core.CharLiteral` (with
        reversible operands)
    -   `SubWith`:
        -   `Core.CharLiteral - &lt;integer> -> Core.CharLiteral`
            (non-reversible operands)
        -   `Core.CharLiteral - Core.CharLiteral -> i32`
            -   Provides a unicode code point delta.

##### `char` integer parameters

Arbitrary integers are supported for most of these operations. For example, we
want to allow addition of negative numbers, even though the representation of
`char` is unsigned, without requiring additional casts.

##### Overflow semantics

Operations will use error overflow semantics,
[similar to signed integers](https://github.com/carbon-language/carbon-lang/blob/trunk/docs/design/expressions/arithmetic.md#overflow-and-other-error-conditions).
For example, `(('a' as char) + 500)` is invalid code because it causes `char`
overflow. That's why conversions are to signed values (for example,
`char as i16`).

##### Preferring i32 returns

In arithmetic, `i32` returns are preferred for deltas because they should be
valid for unicode code points. Even though `char` is only 8-bits, using `i32`
for returns there too creates consistency with `Core.CharLiteral`.

### Revoke and replace proposal #1964: Character Literals

This revokes proposal #1964 for simplicity. Rather than trying to detail which
decisions still apply and which don't, this proposal is acting from an
assumption that all decisions there no longer apply. We can still benefit by
pointing towards the rationale in explicitly maintaining decisions, but we want
to go through that step.

## Rationale

-   [Performance-critical software](https://github.com/carbon-language/carbon-lang/blob/trunk/docs/project/goals.md#performance-critical-software)
    -   The intent is that Carbon's main string type privileges UTF-8 over other
        potential encodings. A `char` represents a single code unit within that,
        and is consequently efficient to access. It can also be invalid, meaning
        we don't guarantee performing runtime validation for users (avoiding
        performance overhead), and that users might be able to use it for other
        encodings.
-   [Software and language evolution](https://github.com/carbon-language/carbon-lang/blob/trunk/docs/project/goals.md#software-and-language-evolution)
    -   `Core.CharLiteral` is designed as a Unicode code point, and even though
        this design doesn't include a way to use values over `7F`, we anticipate
        those will be added in the future. It's being provided as a building
        block for more elaborate Unicode functionality, including both UTF-16
        and UTF-32, even as we prioritize UTF-8.
-   [Code that is easy to read, understand, and write](https://github.com/carbon-language/carbon-lang/blob/trunk/docs/project/goals.md#code-that-is-easy-to-read-understand-and-write)
    -   Character literal syntax mirrors string literal syntax. The main
        divergence is `\x80` and higher similar escapes, which are not supported
        due to potentially ambiguous behavior, still in furtherance of this
        goal.
-   [Practical safety and testing mechanisms](https://github.com/carbon-language/carbon-lang/blob/trunk/docs/project/goals.md#practical-safety-and-testing-mechanisms)
    -   Restricting the set of operators valid for `char` gives us a way to do
        different sorts of validation that can be more character-oriented than
        if we treated it as an arbitrary byte.
    -   Treating `Core.CharLiteral` as a valid Unicode character allows us to
        provide static checking for some operations, such as `'a' + 1` resulting
        in another valid Unicode code point; more is also transitively possible,
        including involving `char`.
-   [Interoperability with and migration from existing C++ code](https://github.com/carbon-language/carbon-lang/blob/trunk/docs/project/goals.md#interoperability-with-and-migration-from-existing-c-code)
    -   Modeling `char` as a UTF-8 code unit creates behavior which is very
        similar to C++, but still shifts towards a more character-oriented
        approach. We do expect some migration friction as a consequence (as
        use-cases might need either more casts, or to switch to a byte type).

## Future work

There's still significant future work, including:

-   `signed char`, `unsigned char`
-   `std::char8_t`, `std::char16_t`, `std::char32_t`
-   UTF-16 and UTF-32 support

It should not be assumed that there's any restriction on the designs of those
features, particularly no restrictions from #1964.

## Alternatives considered

### Align `char` fully with C++, or make it fully valid

Alternatives were discussed in
[zygoloid's comment on #5903](https://github.com/carbon-language/carbon-lang/issues/5903#issuecomment-3494068591).

The comment notes that three options were proposed:

1.  `char` is fully aligned with C++.

    There is no universal convention for what the value in a `char` means, and
    the numerical encoding of Unicode characters into `char` sequences might
    even be platform-dependent. For example, we might use some code page on
    Windows, EBCDIC on some IBM targets, and probably UTF-8 everywhere else.
    Likely the encoding would match what a character literal in C++ code would
    do for that target. Even when the target normally uses UTF-8, it would be
    reasonable to use an array of `char` as the type of the output buffer when
    transcoding from UTF-8 to some other encoding, and generally an encoded text
    buffer (in any encoding) would typically be represented as an array of
    `char`. It might also be reasonable to use an array of `char` for things
    that aren't necessarily text, such as file contents.

2.  `char` models a UTF-8 code unit, although it may not necessarily be valid,
    and may appear in a sequence that is not a valid UTF-8 encoding.

    As with the first option, `char` can represent an integer in [0, 255], although
    it is not an integer type. Higher-level abstractions would likely (eventually)
    be provided to represent different views of the code unit sequence as (for example)
    a sequence of code points or a sequence of graphemes, but the fundamental model
    exposes the encoding. Functions taking `char` or `char` sequences would assume
    UTF-8 encoding, and would need to consider how to handle invalid `char`s and
    invalid `char` sequences.

3.  Use a foundation that enforces Unicode string validity, for some definition
    of "Unicode string validity".

    The `char` type is a Unicode character. Strings would notionally be a
    sequence of Unicode characters, possibly also maintaining some higher-level
    string invariants. String indexing, if it exists, would likely treat the
    string as a sequence of Unicode characters. String invariants would be
    enforced by type conversion into the string type rather than within the
    string operations, and certain classes of invalid strings would be
    unrepresentable.

Rationale as evaluated are:

-   **Privilege UTF-8 over other encodings:** UTF-8 is
    [typically the best choice](https://utf8everywhere.org/) for representing
    text, even when targeting languages where characters are 3 bytes in UTF-8
    but 2 in UTF-16, and even on Windows where the system APIs typically operate
    primarily in UTF-16 or UCS-2. We should create affordances that encourage
    use of UTF-8 (such as having the `char` type be conventionally UTF-8).
    -   Our overall goal to support (only) modern environments and a general
        desire for consistency and portability argues against supporting
        non-Unicode encodings for character types.
    -   Having _some_ convention for the meaning of the value of a `char` seems
        important, and the lack of one in C++ has been a notable problem over
        time, leading to the addition of `char8_t` et al, which have not been
        entirely satisfactory solutions due to the existing widespread usage of
        plain `char`.
-   **Do not privilege any particular meaning of "validity":** There are many
    different ways in which you can view a sequence of UTF-8 code units as being
    valid or invalid. For example: Can a string start with a combining
    character? Can it have mismatched LRE/RLE/PDF characters in it? Can it be
    unnormalized, or must it be in NFC, or in NFD? Can it contain unassigned
    Unicode characters? Can it contain PUA characters? Can it contain
    non-characters? Picking any set of answers to these questions as being our
    canonical notion of "validity" is somewhat arbitrary.
-   **Do not privilege any particular level for accessing elements of the string
    other than code units:** There are many different layers of abstraction at
    which you can interpret the contents of a string. The atoms that users want
    to interact with, such as glyphs or grapheme clusters in rendering, or
    combining characters when editing or performing substring searches, aren't
    in one-to-one correspondence with Unicode characters any more than they're
    in one-to-one correspondence with UTF-8 code units. So it's not clear that
    privileging Unicode-character-oriented access (or indeed any of the other
    higher-level Unicode views) is appropriate. However, code units are in
    direct correspondence with bytes of memory, which is directly relevant for
    low-level operations, so there is a reason to provide direct access to
    byte-level / code-unit-level operations.
    -   If string indexing operates on Unicode characters, it would either be
        non-constant-time or would require not storing strings as just a
        sequence of UTF-8. Having a constant-time indexing operation on strings
        seems very important (especially for interop and for meeting C++
        developers where they are), even though a lot of the desired
        functionality (perhaps all of it) can be provided with iterator- or
        cursor-like machinery instead.
-   **Enforcing validity is problematic for existing API structures:** Requiring
    strings to be valid UTF-8 presents difficulties when moving text into or out
    of other sources. For example, when reading text from a validly-encoded
    UTF-8 file into a text buffer, one would need to deal with a read that ends
    in the middle of an encoding of a character. I don't know how Rust deals
    with this, but it seems like it would create significant impedance mismatch
    with C-like buffered I/O utilities. Similarly, when interoperating with C++,
    it would create friction if our string representation requires strings to be
    valid UTF-8 encodings.
-   **We can allow additional invariants without requiring them:** For a
    known-to-be-valid UTF-8 sequence, a higher-level abstraction can be built,
    and similarly, yet-higher-level abstractions can be built for whatever other
    invariants we want to enforce. So using option 2 rather than option 3 as our
    foundation doesn't prevent enforcing invariants in the type system (but nor
    does it encourage doing so).

This proposal is choosing option 2, that `char` models a UTF-8 code unit without
validation. In some sense, option 2 is still "fully aligned with C++", but with
C++'s `char8_t` rather than with C++'s `char`.

### Raw character literals

[Raw string literals](https://github.com/carbon-language/carbon-lang/blob/trunk/docs/design/lexical_conventions/string_literals.md#raw-string-literals)
use a `#` prefix. There's limited use for this in character literals;
technically, `'\\'` could instead be `#'\'#`, but that's longer and extra
characters may prove distracting. Raw string literals are more useful when
there's a longer character sequence, whereas character literals have one
character by definition. For simplicity, character literals won't have raw
syntax.

### Disallow hex escape sequences in character literals

A `\x##` escape sequence abstractly represents a UTF-8 code unit. Whereas values
over 7F are valid in string literals (allowing arbitrary byte values), these are
disallowed in character literals because we want a more validated Unicode
behavior. Developers could instead rely on `\u` escapes for `\x`.

It can still be useful to allow `\x` escapes for low-range values because some
developers will still need to specify
[ANSI escapes](https://en.wikipedia.org/wiki/ANSI_escape_code). Carbon
[drops support for some escape sequences](https://github.com/carbon-language/carbon-lang/blob/trunk/docs/design/lexical_conventions/string_literals.md#escape-sequences),
such as `\a`, and specifically advises `\x` as an alternative for developers
that need it. Requiring `\a` -> `\x07` -> `\u{07}` is incrementally more verbose
syntax, and developers may be confused why `"\x1B"` is allowed for strings but
`'\u{1B}'` is required for characters.

Values over 7F are ambiguous between an arbitrary byte value and a Unicode code
point, and so should be invalid. However, where both interpretations are
identical for UTF-8 (values up to and including 7F), we will allow `\x` escape
sequences.

### Allow grapheme clusters in character literals

This proposal carries forward the decision in #1964
[to not support grapheme clusters](https://github.com/carbon-language/carbon-lang/pull/1964/files#diff-192d5568d8c1d15e68abe0c46cc52cc0b375a372d1dad8d2154d09f8b29666c5R340)
in character literals.

### Reuse string literal syntax for character literals

Instead of using single quotes (for example, `'a'`), we could use string literal
syntax with a conversion (for example, `"a" as char`) for character literals.
This was proposed because it would free up the single quote for other,
unspecified syntax uses.

For background, character literals are common in C++. For example, in
SourceGraph search statistics (some of these are in comments -- a search
limitation):

-   `'(.|\\.)'`:
    [46.2 million](https://sourcegraph.com/search?q=context:global+lang:c%2B%2B+count:50000000+/%27%28.%7C%5C%5C.%29%27/&patternType=keyword&sm=0)
-   `<<`:
    [over 100 million](https://sourcegraph.com/search?q=context:global+lang:c%2B%2B+count:100000000+/+%3C%3C+/&patternType=keyword&sm=0)
-   `>>`:
    [10.4 million](https://sourcegraph.com/search?q=context:global+lang:c%2B%2B+count:50000000+/+%3E%3E+/&patternType=keyword&sm=0)
-   `%`:
    [5.3 million](https://sourcegraph.com/search?q=context:global+lang:c%2B%2B+count:10000000+/+%25+/&patternType=keyword&sm=0)

This creates several disadvantages for removing character literals in Carbon:

-   **Migrating C++ developers to Carbon:** The frequency of use can be expected
    to have trained developers to expect single quotes to be used for
    characters, especially the C++ developers that Carbon is targeting.
    Repurposing them would create a friction for C++ developers to need to
    understand the different meanings of the same syntax in each of C++ and
    Carbon, something Carbon prefers to avoid.

-   **Increased runtime error risks:** Runtime errors could take the form of
    simple increased overhead, such as converting a string literal to a `str`
    then to a `char`. However, they could also be more insidious, such as doing
    `[0]` on a string literal and not validating that the string is exactly one
    character (this would also likely return a null byte for `""[0]`). By having
    a character literal type, Carbon encourages developers to stay within guide
    rails that make it easier to get compile-time behavior and program
    validation.

-   **Block string literal use:** We already have another use for single quotes
    in Carbon:
    [block string literals](/docs/design/lexical_conventions/string_literals.md).
    The syntax may need to change along with removing character literals, to
    make room for other uses of single quotes.

    -   If retained, it would constrain uses of single quotes. For example, a
        unary operator syntax has overlap (that is, if `'a` and `''a` are valid,
        then `'''a` is ambiguous).

    -   The choice of single quotes in proposal
        [#1360: Change raw string literal syntax](https://github.com/carbon-language/carbon-lang/pull/1360)
        was made accounting for single quotes in character literals, and that
        commonality would be lost.

-   **Tooling:** The prevalence of single quotes being used for either strings
    or characters also affects their treatment in tools not specialized to
    Carbon: they expect them to be used for strings. For example, Rust's use of
    single quotes for lifetime annotations has been observed to break
    language-agnostic syntax highlighting.

While a compelling proposal for a different use of single quotes may come up in
the future, freeing up the character for other purposes is insufficient to
justify a different syntax for character literals.

#### Treat single-character string literals as a third "text literal" type

A related alternative with the same goal of eliminating single quotes for
character literals is that, rather than requiring single-character string
literals be explicitly converted to `char`, they could instead have a third type
of text literal. This would implicitly cast to either `str` or `char`.

This approach would lead to three literal types: `StrLiteral`, `CharLiteral`,
and `TextLiteral`. The distinction of `CharLiteral` is important because we
still want to support arithmetic on character literals, such as `'a' + 1` (which
we would not want to be allowed for `StrLiteral`).

The existence of a third type would be important for generic code, even when not
trying to use character literals. For example:

```carbon
  fn StoreValue[U:! type](ref a: Optional(U), b: U) {
    a = b;
  }

  fn StrLogic[T:! type](a: T) {
    var x: Optional(T) = a;
    StoreValue(x, "str");
  }

  fn F() {
    StrLogic("a");
  }
```

Here, `T` is deduced to be `TextLiteral`. However, `U` has no valid value: it's
passed `Optional(TextLiteral)`, while `"str"` is a `StrLiteral` (which should
not be convertible to `TextLiteral`). As a consequence, this code is invalid,
even though the same code would be valid if there were not `TextLiteral` type.

Advantages:

-   Avoids an explicit cast.

Disadvantages:

-   Shares most of the disadvantages of the primary explicit conversion
    approach.
    -   This includes the risk that developers will write `"..."[0]` instead of
        `"..." as char` when they need a character, although the frequency may
        be reduced.
-   Having additional types in common literals could lead to programmer errors
    in deducing generic types, as described above.
-   Implicit casts cause more operator ambiguity.
    -   How are operators that have different meanings for string and character
        literals handled, such as `Cpp.std.cout <<` or `<=>`?
    -   In Carbon, we'd probably still want string operators to work; for
        example, `"a" + "b" => "ab"`, and that can be compile-time. Is `"a" + 1`
        a pointer to the null byte as it is in C++ (similar to `&("a"[1])`), a
        character addition (`'a' + 1 => 'b'`), or does it require an explicit
        cast in order to ensure behavior is deliberate?
