# Character literals

<!--
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

[Pull request](https://github.com/carbon-language/carbon-lang/pull/1964)

<!-- toc -->

## Table of contents

-   [Abstract](#abstract)
-   [Problem](#problem)
-   [Background](#background)
-   [Proposal](#proposal)
-   [Details](#details)
    -   [Types](#types)
    -   [Operations](#operations)
-   [Rationale](#rationale)
-   [Alternatives considered](#alternatives-considered)
    -   [No distinct character types](#no-distinct-character-types)
    -   [No distinct character literal](#no-distinct-character-literal)
    -   [Supporting prefix declarations](#supporting-prefix-declarations)
    -   [Allowing numeric escape sequences](#allowing-numeric-escape-sequences)
    -   [Supporting formulations of grapheme clusters and non-code-point code-units](#supporting-formulations-of-grapheme-clusters-and-non-code-point-code-units)
-   [Future Work](#future-work)
    -   [UTF code unit types proposal](#utf-code-unit-types-proposal)

<!-- tocstop -->

## Abstract

This proposal specifies lexical rules for constant characters in Carbon:

Put character literals in single quotes, like `'a'`. Character literals work
like numeric literals:

-   Every different literal value has its own type.
-   The literal itself doesn't have a bit width as a consequence. Instead,
    variables use explicitly sized character types and character literals can be
    converted to these types when representable.
-   A character literal must contain exactly one code point.

Follows the plan from open design idea
[#1934: Character Literals](https://github.com/carbon-language/carbon-lang/issues/1934).

## Problem

Carbon currently has no lexical syntax for character literals, and only provides
string literals and numeric literals. We wish to provide a distinct lexical
syntax for character literals versus string literals.

The advantage of having an explicit character type fundamentally comes down to
characters being represented as integers whereas strings are represented as
buffers. This will allow characters to have different operations, and be more
familiar to use. For example:

```
if (c >= 'A' and c <= 'Z') {
    c += 'a' - 'A';
}
```

The example above shows how we would be able to use operations similar to
integers. Being able to use the comparison operations and supporting arithmetic
operations provides an intuitive approach to using characters. This allows us to
remove unnecessary logic of type conversion and other control flow logic, that
is needed to work with a single element string. See [Rationale](#rationale) for
more examples showing more appropriate use of characters over using strings.

## Background

Character Literals by definition is a type of literal in programming for the
representation of a single character's value within the source code of a
computer program. Character literals between languages have some minor nuances
but are fundamentally designed for the same purpose. Languages that have a
dedicated character data type generally include character literals, for example
C++, Java, Swift to name a few. Whereas other languages that lack distinct
character type, like Python use strings of length one to serve the same purpose
a character data type. For more information see
[Character Literals Wiki](https://en.wikipedia.org/wiki/Character_literal),
[Character Literals DBpedia](https://dbpedia.org/page/Character_literal)

## Proposal

Put character literals in single quotes, like `'a'`. Character literals work
like numeric literals:

-   Every different literal value has its own type.
-   The literal itself doesn't have a bit width as a consequence. Instead,
    variables use explicitly sized character types and character literals can be
    converted to these types when representable. Follows the plan from #1934.
-   A character literal will model single Unicode code points that have a single
    concrete numerical representation. We will not be supporting other
    formulations like code unit sequences or grapheme clusters as these will be
    modeled with normal string literals.

## Details

-   A character literal is a sequence enclosed with single quotes delimiter ('),
    of UTF-8 code units that must be a valid encoding. This matches
    [the UTF-8 encoding of Carbon source files](https://github.com/carbon-language/carbon-lang/blob/trunk/proposals/p0142.md#character-encoding).
-   A character literal must encode exactly one code point.
-   It supports addition and subtraction, [as described below](#operations).
-   Character literals will support the relevant subset of the backslash (`\`)
    escape sequences in string literals, including `\t`, `\n`, `\r`, `\"`, `\'`,
    `\\`, `\0`, and `\u{HHHH...}`. See
    [String Literals: Escape sequence](https://github.com/carbon-language/carbon-lang/blob/trunk/proposals/p0199.md#escape-sequences).
    -   Escape sequences which would result in non-UTF-8 encodings or more than
        one code point are not included.
    -   The escape of an embedded newline is also excluded as it isn't expected
        to be relevant for character literals.

We will not support:

-   character literals that don't contain exactly one Unicode code point;
-   multi-line literals;
-   "raw" literals (using #'x'#);
-   `\x` escape sequences;
-   character literals with a single quote (`'`) or back-slash (`\`), except as
    part of an escape sequence;
-   empty character literals (`''`);
-   a backslash followed by an (unescaped) newline;
-   ASCII control codes (0...31), including whitespace characters other than
    word space (tab, line feed, carriage return, form feed, and vertical tab),
    except when specified with an escape sequence.

### Types

For the time being, Carbon will support three character types: `Char8`,
`Char16`, and `Char32`. These types are capable of representing both code units
and code points. It’s important to note that the support for different
UTF-encoding code unit types will be addressed in a separate proposal. Please
refer to the [UTF code unit types proposal](#utf-code-unit-types-proposal)for
more information on that topic.

In Carbon, the type `CharN` represents a character, where `N` corresponds to the
bit size of the character type (`8`, `16`, or `32`). We will only allow
character literals that map directly to a complete value of a code point. Here
are examples of character literals for each specific type:

-   `Char8`: The character literal consists of a single Unicode code point that
    can be represented within 8 bits. For example:

`let allowed: Char8 = ‘a’ `

In this example, the character literal `’a’` corresponds to the Unicode code
point `97`, which is within the valid range of `Char8` since `97` is less than
or equal to `0x7F`.

-   `Char16`: The character literal represents a Unicode code point that can be
    represented within 16 bits. Here’s an example:

`let smiley: Char16 = ‘\u{1F600}’`

The character literal `’\u{1F600}’` represents the smiley face emoji, which has
the Unicode code point `128512`. Since `128512` can be represented within 16
bits, it can be assigned to a variable of type `Char16`.

-   `Char32`: This character type allows the representation of Unicode code
    points within 32 bits. Here’s an example:

`let musicalNote: Char32 = ‘🎵’`

In this case, the character literal `’🎵’` corresponds to the musical note emoji
with the Unicode code point `127925`. Since `127925` falls within the range that
can be represented by `Char32`, it can be assigned to a variable of type
`Char32`.

By restricting character literals to those that can be directly mapped to code
points within the specific character types, we ensure accurate representation
and compatibility with the chosen character encoding scheme.

### Operations

Character literals representing a single code point support the following
operators:

-   Comparison: `<`, `>`, `<=`, `>=` `==`
-   Plus: `+`. This doesn't concatenate, but allows numerically adjusting the
    value:
    -   Only one operand may be a character literal, the other must be an
        integer literal.
    -   The result is the character literal whose numeric value is the sum of
        numeric value of the operands. If that sum is not a valid Unicode code
        point, it is an error.
-   Subtract: `-`. This will subtract the value of the two characters, or a
    character followed by an integer literal:
    -   If the `-` is used between two character literals, the result will be an
        integer constant. For example, `'z' - 'a'` is equivalent to `25`.
    -   If the `-` is used between a character literal followed by a integer
        literal, this will produce a character constant. For example `'z' - 4`
        is equivalent to `'v'`.
    -   If the `-` is used between a integer literal followed by a character
        literal `100 - 'a'`, this will be rejected unless the integer is cast to
        a character.

There is intentionally no implicit conversion from character literals to integer
types, but explicit conversions are permitted between character literals and
integer types. Carbon will separate the integer types from character types
entirely.

## Rationale

This proposal supports the goal of making Carbon code
[easy to read, understand, and write](/docs/project/goals.md#code-that-is-easy-to-read-understand-and-write).
Adding support for a specific character literal supports clean, readable,
concise use and is a much more familiar concept that will make it easier to
adopt Carbon coming from other languages. Have a distinct character literal will
also allow us support useful operations designed to manipulate the literal's
value. When working with an explicit character type we can use operators that
have unique behavior, for example say we wanted to advance a character to the
next literal. In other languages the `+` operator is often used for
concatenation, so using a `String` will produce a type error: `"a" + 1`. However
with a character literal, we can support operations for these use cases:

```
var b: u8;

b = 'a' + 1;
b + 1 == 'c';
```

See [Operations](#operations) and
[No Distinct Character Literal](#no-distinct-character-literal) for more
information.

Further, this design follows other standards set in place by previous proposals.
For example following the
[String Literals: Escaping Sequence](https://github.com/carbon-language/carbon-lang/blob/trunk/proposals/p0199.md#escape-sequences-1)
and representing characters as integers with the behaviour inline with
[Integer Literals](https://github.com/carbon-language/carbon-lang/blob/trunk/proposals/p0143.md).

This also supports our goal for
[Interoperability with and migration from existing C++ code](/docs/project/goals.md#interoperability-with-and-migration-from-existing-c-code)
by ensuring that every kind of character literal that exists in C++ can be
represented in a Carbon character literal. This is done in a way that is natural
to adopt, understand, easy to read by having explicit character types mapped to
the C++ character types and the correct associated encoding.

Finally, the choice to use Unicode and UTF-8 by default reflects the Carbon goal
to prioritize
[modern OS platforms, hardware architectures, and environments](/docs/project/goals.md#modern-os-platforms-hardware-architectures-and-environments).
This reflects the
[growing adoption of UTF-8](https://en.wikipedia.org/wiki/UTF-8#Adoption).

## Alternatives considered

### No distinct character types

Unlike C++, Carbon will separate the integer and the character types. We
considered using `u8`, `u16`, and `u32` instead of `Char8`, `Char16`, and
`Char32` to reduce the number of different types users needed to be aware of and
convert between. We decided against it because it came with a number of
disadvantages:

-   `u8`, `u16`, and `u32` have the wrong arithmetic semantics: we don't want
    wrapping, and many `uN` operations, like multiplication, division, and
    shift, are not meaningful on code units. There may be rare cases where you
    want to use those operations, such as if you're implementing a conversion to
    or from code units. But in those rare cases it would be reasonable for the
    user to convert to an integer type to perform that operation and convert
    back when done.
-   Some operations want to be able to tell the difference between values that
    are intended to be UTF-8 instead of having no specified encoding.
-   Some operations want to be able to know that they've been given text rather
    than random bytes of data. For example, `Print(0x41 as u8)` would be
    expected to print `"65"` while `Print('\u{41}')` and `Print(0x41 as Char8)`
    would be expected to print `"A"`.
-   It's useful for developers to document the intended meaning of a value, and
    using a distinct type is one way to do that.

See [UTF code unit types proposal](#utf-code-unit-types-proposal) for more
information about UTF encoding types for a future proposal.

### No distinct character literal

In principle, a character literal can be represented by reusing string literals
similar to how Python handles character literals, however this would prevent
performing operations on characters as integers. For example, the `+` operator
on strings is used for concatenation, but `+` on a character would change its
value.

```
// `digit` must be in the range 0..9.
fn DigitToChar(digit: i32) -> Char8 {
  return '0' + digit;
}
```

Furthermore, many properties of Unicode characters are defined on ranges of code
points, motivating supporting comparison operators on code points.

```
fn IsDingBatCodePoint(c: Char32) -> bool {
  return c >= '\u{2700}' and c <= '\u{27BF}';
}
```

### Supporting prefix declarations

No support is proposed for prefix declarations like `u`, `U`, or `L`. In
practice they are used to specify the character literal types and their encoding
in languages like C and C++. There are a several benefits to omitting prefix
declarations; improved readablitly, simplifying how a character's type is
determined, and how we are encoding character literals. When declaring a
character literal, the type is based on the contents of the character so that
`var c: u8 = 'a'` is a valid character that can be converted to `u8`, in order
to support prefix declarations we would need to extend our type system to have
other explicit type checks like in C++; a UTF-16 `u'`, UTF-32 `U'`, and wide
characters `L'`. This would be more familiar for individuals coming to Carbon
from a C++ background, and simplify our approach for C++ Interoperability. At
the cost of diverge from existing standards, for example
[Proposal 142](https://github.com/carbon-language/carbon-lang/blob/trunk/proposals/p0142.md#character-encoding)
states all of Carbon source code should be UTF-8 encoded. Prefix declarations
would detract the readability of the character literals and increase the
complexity of character literal [Types](#types).

### Allowing numeric escape sequences

This proposal does not support numeric escape sequences using `\x`. This
simplifies the design of character types and literals, making them only
represent code points and not code units. However this does come with the
disadvantage of less consistency of character literals with string literals,
since they now accept different escape sequences. We don't want to remove
numeric escape sequence from string literals, so we can support string use cases
like representing invalid encodings.

This approach has the additional concern that if character literals don't
support numeric escape sequences, developers may choose to use numeric literals
instead, at a cost of type-safety and readability. For example, it isn't clear
in `var first_digit: Char8 = 0;` whether `0` is supposed to be a `NUL` character
or the encoding of the `'0'` character (48). We addressed this concern, and type
safety concerns about distinguishing numbers and characters, by making the
integer to character conversions explicit.

### Supporting formulations of grapheme clusters and non-code-point code-units

Rather than explicitly limiting characters literals to a more integer-like
representation of a single Unicode code point, we could represent characters
literal formulations of grapheme clusters and non-code-point code units. What
humans tend to think of as a "character" corresponds to a "grapheme cluster."
The encoding of a grapheme cluster can be arbitrarily long and complex, which
would sacrifice the ability to perform integer operations. If we wanted to add
support for other character formulations, we would need to use separate
spellings to represent a small set of operations that are today expressed with
integer-based math on C++'s character literals. This includes things like
converting an integer between 0 and 9 into the corresponding digit character, or
computing the difference between two digits/two other characters. For these
reasons, we have decided to start out by representing character literals as
single Unicode code points following a more integer-like model. However this
topic should be revisited if we find that there is a significant need for the
additional functionality and attendant complexity for these other character
formulations.

## Future Work

### UTF code unit types proposal

There have been several ideas and discussions around how we would like to handle
UTF code units. This section will hopefully provide some guidance for a future
proposal when the topic is revisited for how we would like to build out
encoding/decoding for character literals.

We will have the types `Char8`, `Char16`, and `Char32` representing code units
in UTF-8, UTF-16, and UTF-32, but we will not support all code units, but only
those which map directly to the complete value of a code point. However,
character literals will use their own types distinct from these:

-   We will support value preserving implicit conversions from character
    literals to code point or code unit types. In particular, a character
    literal converts to a `Char8` UTF-8 code unit if it is less than or equal to
    0x7F, and `Char16` UTF-16 code unit if it is less than or equal to 0xFFFF.
-   Conversions from string or character literals to a non-value-preserving
    encoding must be explicit.
-   Conversions from string literals to Unicode strings are implicit, even
    though the numeric values of the encoding may change.

We can see whether the particular literal is represented in the variable's type
by only looking at the types.

```
let allowed: Char8 = 'a';
```

The above is allowed because the type of `'a'` is the character literal
consisting of the single Unicode code point 97, which can be converted to
`Char8` since 97 is less than or equal to 0x7F.

```
let error1: Char8 = '😃';
let error2: Char8 = 'AB';
```

However these should produce errors. The type of `'😃'` is the character literal
consisting of the single Unicode code point `0x1F603`, which is greater than
0x7F. The type of `'AB'` is a character literal that is a sequence of two
Unicode code points, which has no conversion to a type that only handles a
single UTF-8 code unit.

All of `'\n'`, and `'\u{A}'` represent the same character and so have the same
type. However, explicitly converting this character literal to another character
set might result in a character with a different value, but that still
represents the newline character.
