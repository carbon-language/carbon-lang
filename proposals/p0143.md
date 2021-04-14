# Numeric literals

<!--
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

[Pull request](https://github.com/carbon-language/carbon-lang/pull/143)

<!-- toc -->

## Table of contents

-   [Problem](#problem)
-   [Background](#background)
-   [Proposal](#proposal)
-   [Details](#details)
    -   [Integer literals](#integer-literals)
    -   [Real number literals](#real-number-literals)
        -   [Ties](#ties)
    -   [Digit separators](#digit-separators)
        -   [Open question: digit separator placement](#open-question-digit-separator-placement)
-   [Alternatives considered](#alternatives-considered)
    -   [Integer bases](#integer-bases)
        -   [Octal literals](#octal-literals)
        -   [Decimal literals](#decimal-literals)
        -   [Case sensitivity](#case-sensitivity)
    -   [Real number syntax](#real-number-syntax)
    -   [Digit separator syntax](#digit-separator-syntax)
-   [Rationale](#rationale)
    -   [Painter rationale](#painter-rationale)
    -   [Open questions](#open-questions)

<!-- tocstop -->

## Problem

This proposal specifies lexical rules for numeric constants in Carbon.

## Background

We wish to cover literals for two categories of types:

-   Integer types, that can represent some (typically contiguous) subset of the
    integers, ℤ.
-   Real number types, that can represent some
    [discrete](https://en.wikipedia.org/wiki/Isolated_point) subset of the real
    numbers, ℝ. (Typically only rational numbers can be represented, but that
    doesn't matter for our purposes.)

Real number types may include additional values (infinities and NaN values). We
do not provide a notation to express such values.

In C++, the following syntaxes are used:

-   Integer literals
    -   `12345` (decimal)
    -   `0x1FE` (hexadecimal)
    -   `0123` (octal)
    -   `0b1010` (binary)
-   Real number literals
    -   Decimal
        -   `123.`
        -   `.123`
        -   `123.456`
        -   `123.e456` (= 123 \* 10<sup>456</sup>)
        -   `.123e456`
        -   `123.456e789`
        -   `123e456` (no decimal point)
        -   Any of the above with a `+` or `-` after `e`.
    -   Hexadecimal
        -   `0x123.p456` (= 123<sub>16</sub> \* 2<sup>456</sup>)
        -   `0x.123p456`
        -   `0x123.456p789`
        -   `0x123p456` (no hexadecimal point)
        -   Any of the above with a `+` or `-` after `p`.
-   Digit separators (`'`) may appear between any two digits
-   An optional suffix defines the type
    -   `U` (`unsigned`) and `L` (`long`) or `LL` (`long long`) for integers
        (order-independent, but `LUL` disallowed)
    -   `F` (`float`) or `L` (`long double`) for real numbers
-   User-defined literals may have custom suffixes, starting with `_` for
    non-standard-library literals.

C++ numeric literals are case-insensitive, except in the suffix of a
user-defined literal. Negative numbers are formed by applying a unary `-`
operator to a non-negative literal.

The type of a literal in C++ depends primarily on its syntax and its suffix.
However, for integer literals, the type also depends on the value; the language
rules attempt to pick a type large enough to fit the value. An `unsigned` type
is always used if a `U` suffix is present, is never used for a decimal literal
without a `U` suffix, and otherwise may or may not be used depending on whether
the value happens to fit into an unsigned type but not into a signed type of the
same width.

Other languages use somewhat different rules, but the broad lexical structure
above -- an optional prefix for the base, a value, an optional exponent, and an
optional suffix -- is common across a large number of languages.

## Proposal

We allow these syntaxes:

-   Integer literals
    -   `12345` (decimal)
    -   `0x1FE` (hexadecimal)
    -   `0b1010` (binary)
-   Real number literals
    -   `123.456` (digits on both sides of the `.`)
    -   `123.456e789` (optional `+` or `-` after the `e`)
    -   `0x1.2p123` (optional `+` or `-` after the `p`)
-   Digit separators (`_`) may be used, but only in conventional locations

Note that real number literals always contain a `.` with digits on both sides,
and integer literals never contain a `.`.

Literals are case-sensitive.

No support is proposed for literals with type suffixes, but without prejudice:
this proposal proposes neither the inclusion nor the absence of such literals.

## Details

### Integer literals

Decimal integers are written as a non-zero decimal digit followed by zero or
more additional decimal digits, or as a single `0`.

Integers in other bases are written as a `0` followed by a base specifier
character, followed by a sequence of digits in the corresponding base. The
available base specifiers and corresponding bases are:

| Base specifier | Base | Digits                   |
| -------------- | ---- | ------------------------ |
| `b`            | 2    | `0` and `1`              |
| `x`            | 16   | `0` ... `9`, `A` ... `F` |

The above table is case-sensitive. For example, `0b1` and `0x1A` are valid, and
`0B1`, `0X1A`, and `0x1a` are invalid.

A zero at the start of a literal can never be followed by another digit: either
the literal is `0`, the `0` begins a base specifier, or the next character is a
decimal point (see below).

### Real number literals

Real numbers are written as a decimal or hexadecimal integer followed by a
period (`.`) followed by a sequence of one or more decimal or hexadecimal
digits, respectively. A digit is required on each side of the period. `0.` and
`.3` are both invalid.

A real number can be followed by an exponent character, an optional `+` or `-`
(defaulting to `+` if absent), and a character sequence matching the grammar of
a decimal integer with some value _N_. For a decimal real number, the exponent
character is `e`, and the effect is to multiply the given value by
10<sup>&plusmn;_N_</sup>. For a hexadecimal real number, the exponent character
is `p`, and the effect is to multiply the given value by
2<sup>&plusmn;_N_</sup>. The exponent suffix is optional for both decimal and
hexadecimal real numbers.

Note that a decimal integer followed by `e` is not a real number literal. For
example, `3e10` is not a valid literal.

When a real number literal is interpreted as a value of a real number type, its
value is the representable real number closest to the value of the literal. In
the case of a [tie](#ties), the conversion to the real number type is invalid.

The decimal real number syntax allows for any decimal fraction to be expressed
-- that is, any number of the form _a_ x 10<sup>-_b_</sup>, where _a_ is an
integer and _b_ is a non-negative integer. Because the decimal fractions are
dense in the reals and the set of values of the real number type is assumed to
be discrete, every value of the real number type can be expressed as a real
number literal. However, for certain applications, directly expressing the
intended real number representation may be more convenient than producing a
decimal equivalent that is known to convert to the intended value. Hexadecimal
real number literals are provided in order to permit values of binary floating
or fixed point real number types to be expressed directly.

#### Ties

As described above, a real number literal that lies exactly between two
representable values for its target type is invalid. Such ties are extremely
unlikely to occur by accident: for example, when interpreting a literal as
`Float64`, `1.` would need to be followed by exactly 53 decimal digits (followed
by zero or more `0`s) to land exactly half-way between two representable values,
and the probability of `1.` followed by a random 53-digit sequence resulting in
such a tie is one in 5<sup>53</sup>, or about
0.000000000000000000000000000000000009%. For `Float32`, it's about
0.000000000000001%, and even for a typical `Float16` implementation with 10
fractional bits, it's around 0.00001%.

Ties are much easier to express as hexadecimal floating-point literals: for
example, `0x1.0000_0000_0000_08p+0` is exactly half way between `1.0` and the
smallest `Float64` value greater than `1.0`, which is `0x1.0000_0000_0000_1p+0`.

Whether written in decimal or hexadecimal, a tie provides very strong evidence
that the developer intended to express a precise floating-point value, and
provided one bit too much precision (or one bit too little, depending on whether
they expected some rounding to occur), so rejecting the literal seems like a
better option than accepting it and making an arbitrary choice between the two
possible values.

### Digit separators

If digit separators (`_`) are included in literals, they must meet the
respective condition:

-   For decimal integers, the digit separators shall occur every three digits
    starting from the right. For example, `2_147_483_648`.
-   For hexadecimal integers, the digit separators shall occur every four digits
    starting from the right. For example, `0x7FFF_FFFF`.
-   For real number literals, digit separators can appear in the decimal and
    hexadecimal integer portions (prior to the period and after the optional `e`
    or mandatory `p`) as described in the previous bullets. For example,
    `2_147.483648e12_345` or `0x1_00CA.FEF00Dp+24`
-   For binary literals, digit separators can appear between any two digits. For
    example, `0b1_000_101_11`.

#### Open question: digit separator placement

**2020-09-15: core team meeting selected Alternative 0**

As an alternative to the rule proposed above, we could consider different
restrictions on where digit separators can appear:

**Alternative 0:** as presented above.

**Alternative 1:** allow any digit groupings (for example, `123_4567_89`).

Pro:

-   Simpler, more flexible rule, that may allow some groupings that are
    conventional in a specific domain. For example, `var Date: d = 01_12_1983;`,
    or `var Int64: time_in_microseconds = 123456_000000;`.
-   Culturally agnostic. For example, the Indian convention for digit separators
    would group the last three digits, and then every two digits before that
    (1,23,45,678 could be written `1_23_45_678`).

Con:

-   Less self-checking that numeric literals are interpreted the way that the
    author intends.

**Alternative 2:** as above, but additionally require binary digits to be
grouped in 4s.

Pro:

-   More enforcement that digit grouping is conventional.

Con:

-   No clear, established rule for how to group binary digits. In some cases, 8
    digit groups may be more conventional.
-   When used to express literals involving bit-fields, arbitrary grouping may
    be desirable. For example:

    ```carbon
    var Float32: flt_max =
      BitCast(Float32, 0b0_11111110_11111111111111111111111);
    ```

**Alternative 3:** allow any regular grouping.

Pro:

-   Can be applied uniformly to all bases.

Con:

-   Provides no assistance for decimal numbers with a single digit separator.
-   Does not allow binary literals to express an intent to initialize irregular
    bit-fields.

## Alternatives considered

There are a number of different design choices we could make, as divergences
from the above proposal. Those choices, along with the arguments that led to
choosing the proposed design rather than each alternative, are presented below.

### Integer bases

#### Octal literals

No support is proposed for octal literals. In practice, their appearance in C
and C++ code in a sample corpus consisted of (in decreasing order of commonality
and excluding `0` literals):

-   file permissions,
-   cases where decimal was clearly intended (`CivilDay(2020, 04, 01)`), and
-   (in _distant_ third place) anything else.

The number of intentional uses of octal literals, other than in file
permissions, was negligible. We considered the following alternatives:

**Baseline:** This proposal suggests that we do not support octal literals.
Octal literals are rare and mostly obsolescent. File permissions can be
supported in some other way.

**Alternative 1:** Follow C and C++, and use `0` as the base prefix for octal.

Pro:

-   More similar to C++ and other languages.

Con:

-   Subtle and error-prone rule: for example, left-padding with zeroes for
    alignment changes the meaning of literals.

**Alternative 2:** Use `0o` as the base prefix for octal.

Pro:

-   Unlikely to be misinterpreted as decimal.
-   Follows several other languages (for example, Python).

Con:

-   Additional language complexity.

If we decide we want to introduce octal literals at a later date, use of
alternative 2 is suggested.

#### Decimal literals

**We could permit leading `0`s in decimal integers (and in floating-point
numbers).**

Pro:

-   We would allow leading `0`s to be used to align columns of numbers.

Con:

-   The same literal could be valid but have a different value in C++ and
    Carbon.

**We could add an (optional) base specifier `0d` for decimal integers.**

Pro:

-   Uniform treatment of all bases. Left-padding with `0` could be achieved by
    using `0d000123`.

Con:

-   No evidence of need for this functionality.

**We could permit an `e` in decimal literals to express large powers of 10.**

Pro:

-   Many uses of (eg) `1e6` in our sample C++ corpus intend to form an integer
    literal instead of a floating-point literal.

Con:

-   Would violate the expectations of many C++ programmers used to `e`
    indicating a floating-point constant.

We suggest that this syntax is not added at this point. However, it should be
reconsidered at a later date, once developers are used the requirement that real
literals always contain a period.

#### Case sensitivity

**We could make base specifiers case-insensitive.**

Pro:

-   More similar to C++.

Con:

-   `0B1` is easily mistaken for `081`
-   `0B1` can be confused with `0xB1`
-   `0O17` is easily mistaken for `0017`
-   Allowing more than one way to write literals will lead to style divergence.

**We could make the digit sequence in hexadecimal integers case-insensitive.**

Pro:

-   More similar to C++.
-   Some developers will be more comfortable writing hexadecimal digits in
    lowercase. Some tools, such as `md5`, will print lowercase.

Con:

-   Allowing more than one way to write literals will lead to style divergence.
-   Lowercase hexadecimal digits are less visually distinct from the `x` base
    specifier (for example, the digit sequence is more visually distinct in
    `0xAC` than in `0xac`).

**We could require the digit sequence in hexadecimal integers to be written
using lowercase letters `a`..`f`.**

Pro:

-   Some developers will be more comfortable writing hexadecimal digits in
    lowercase. Some tools, such as `md5`, will print lowercase.
-   `B` and `D` are more likely to be confused with `8` and `0` than `b` and `d`
    are.

Con:

-   Some developers will be more comfortable writing hexadecimal digits in
    uppercase. Some tools will print uppercase.
-   Lowercase hexadecimal digits are less visually distinct from the `x` base
    specifier (for example, the digit sequence is more visually distinct in
    `0xAC` than in `0xac`).

### Real number syntax

**We could allow real numbers with no digits on one side of the period (`3.` or
`.5`).**

Pro:

-   More similar to C++.
-   Allows numbers to be expressed more tersely.

Con:

-   Gives meaning to `tup.0` syntax that may be useful for indexing tuples.
-   Gives meaning to `0.ToString()` syntax that may be useful for performing
    member access on literals.
-   May harm readability by making the difference between an integer literal and
    a real number literal less significant.
-   Allowing more than one way to write literals will lead to style divergence.

See also the section on
[floating-point literals](https://google.github.io/styleguide/cppguide.html#Floating_Literals)
in the Google style guide, which argues for the same rule.

**We could allow a real number with no `e` or `p` to omit a period (`1e100`).**

Pro:

-   More similar to C++.
-   Allows numbers to be expressed more tersely.

Con:

-   Assuming that such numbers are integers rather than real numbers is a common
    error in C++.

**We could allow the `e` or `p` to be written in uppercase.**

Pro:

-   More similar to C++.
-   Most calculators use `E`, to avoid confusion with the constant `e`.

Con:

-   Allowing more than one way to write literals will lead to style divergence.
-   `E` may be confused with a hexadecimal digit.

**We could require a `p` in a hexadecimal real number literal.**

Pro:

-   More similar to C++.
-   When explicitly writing a bit-pattern for a floating-point type, it's
    reasonable to always include the exponent value.

Con:

-   Less consistent.
-   Makes hexadecimal floating-point values even more expert-only.

**We could arbitrarily pick one of the two values when a real number is exactly
half-way between two representable values.**

Pro:

-   More similar to C++.
-   Would accept more cases, and it's likely that either of the two possible
    values would be acceptable in practice.

Con:

-   Would either need to specify which option is chosen or, following C++,
    accept that programs using such literals have non-portable semantics.
-   Numbers specified to the exact level of precision required to form a tie are
    a strong signal that the programmer intended to specify a particular value.

### Digit separator syntax

**2020-09-15: core team meeting chose to forward digit separator to painter**

**2020-10-05: painter selected Alternative 2: `_` as digit separator**

There are various different characters we could attempt to use as a digit
separator. The options we considered are:

**Alternative 0:** `'` as a digit separator.

Pro:

-   Follows C++ syntax.
-   Used in several (mostly European) writing conventions.

Con:

-   `'` is also likely to be used to introduce character literals.

**Alternative 1:** `,` as a digit separator.

Pro:

-   More similar to how numbers are written in English text and many other
    cultures.

Con:

-   Commas are expected to widely be used in Carbon programs for other purposes,
    where there may be digits on both sides of the comma. For example, there
    could be readability problems if `f(1, 234)` called `f` with two arguments
    but `f(1,234)` called `f` with a single argument.
-   Comma is interpreted as a decimal point in the conventions of many cultures.
-   Unprecedented in common programming languages.

**Alternative 2:** `_` as a digit separator.

Pro:

-   Follows convention of C#, Java, JavaScript, Python, D, Ruby, Rust, Swift,
    ...
-   Culturally agnostic, because it doesn't match any common human writing
    convention.

Con:

-   Underscore is not used as a digit grouping separator in any common human
    writing convention.

**Alternative 3:** whitespace as a digit separator.

Pro:

-   Used and understood by many cultures.
-   Never interpreted as a decimal point instead of a grouping separator.
-   Also usable to the right of a decimal point.

Con:

-   Omitted separators in lists of numbers may result in distinct numbers being
    spliced together. For example, `f(1, 23, 4 567)` may be interpreted as three
    separate numerical arguments instead of four arguments with a missing comma.
-   Unprecedented in other programming languages.

**Alternative 4:** `.` as digit separator, `,` as decimal point.

Pro:

-   More familiar to cultures that write numbers this way.

Con:

-   As with `,` as a digit separator, `,` as a decimal point is problematic.
-   This usage is unfamiliar and would be surprising to programmers; programmers
    from cultures where `,` is the decimal point in regular writing are likely
    already accustomed to using `.` as the decimal point in programming
    environments, and the converse is not true.

**Alternative 5:** No digit separator syntax.

Pro:

-   Simpler language rules.
-   More consistent source syntax, as there is no choice as to whether to use
    digit separators or not.

Con:

-   Harms the readability of long literals.

## Rationale

The proposal provides a syntax that is sufficiently close to that used both by
C++ and many other languages to be very familiar. However, it selects a
reasonably minimal subset of the syntaxes. This minimal approach provides
benefits directly in line with both the simplicity and readability goals of
Carbon:

-   Reduces unnecessary choices for programmers.
-   Simplifies the syntax rules of the language.
-   Improves consistency of written Carbon code.

That said, it still provides sufficient variations to address important use
cases for the goal of not leaving room for a lower level language:

-   Hexadecimal and binary integer literals.
-   Scientific notation floating point literals.
-   Hexadecimal (scientific) floating point literals.

### Painter rationale

The primary aesthetic benefit of `'` to the painter is consistency with C++.
However, its rare usage in C++ at this point reduces this advantage to a very
small one, while there is broad convergence amongst other languages around `_`.
The choice here has no risk of significant meaning or building up patterns of
reading for users that might be disrupted by the change, and so it seems
reasonable to simply converge with other languages to end up in the less
surprising and more conventional syntax space.

### Open questions

Placement restrictions of digit separators:

-   The core team had consensus for the proposed restricted placement rules.

Use `_` or `'` as the digit separator character:

-   The core team deferred this decision to the painter.
-   The painter selected `_`.
