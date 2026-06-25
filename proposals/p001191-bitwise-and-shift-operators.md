# Bitwise and shift operators

<!--
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

[Pull request](https://github.com/carbon-language/carbon-lang/pull/1191)

<!-- toc -->

## Table of contents

-   [Problem](#problem)
-   [Background](#background)
    -   [Overflow in shift operators](#overflow-in-shift-operators)
-   [Proposal](#proposal)
-   [Details](#details)
-   [Rationale](#rationale)
-   [Alternatives considered](#alternatives-considered)
    -   [Use different symbols for bitwise operators](#use-different-symbols-for-bitwise-operators)
        -   [Use a multi-character spelling](#use-a-multi-character-spelling)
        -   [Don't provide an xor operator](#dont-provide-an-xor-operator)
        -   [Use `~`, or some other symbol, for complement](#use--or-some-other-symbol-for-complement)
    -   [Provide different operators for arithmetic and logical shifts](#provide-different-operators-for-arithmetic-and-logical-shifts)
    -   [Provide rotate operators](#provide-rotate-operators)
    -   [Guarantee behavior of large shifts](#guarantee-behavior-of-large-shifts)
    -   [Support shifting a constant by a variable](#support-shifting-a-constant-by-a-variable)
    -   [Converting complements to unsigned types](#converting-complements-to-unsigned-types)

<!-- tocstop -->

## Problem

Carbon needs operations for working with bit representations of values.

## Background

C++ provides four bitwise operations for Boolean algebra: complement (`~`), and
(`&`), or (`|`), and xor (`^`). These are all useful in bit-manipulation code
(although `^` is used substantially less than the others). In addition, C++
provides two bit-shift operators `<<` and `>>` that can perform three different
operations: left shift, arithmetic right shift, and logical right shift. The
meaning of `>>` is determined by the signedness of the first operand.

C and Swift use the same set of operators as C++. Rust uses most of the same
operators, but uses `!` instead of `~` for complement, unifying it with the
logical not operator, which is spelled `not` in Carbon and as `!` in Rust and
C++. Go uses most of the same operators as C++, but uses unary prefix `^`
instead of `~` for complement, mirroring binary `^` for xor.

In addition to the operators provided by C and C++, bit-rotate operators are
present in some languages, and a short notation for them may be convenient.
Finally, there are other non-degenerate binary Boolean operations with no common
operator symbol:

-   The "implies" operator (equivalent to `~a | b`).
-   The "implied by" operator (equivalent to `a | ~b`).
-   The complement of each of the other operators (NAND, NOR, XNOR, "does not
    imply", "is not implied by").

### Overflow in shift operators

The behavior of shift operators in C++ had a turbulent past. The behavior of
shift operators has always been undefined if the right operand is out of range
-- not between zero inclusive and the bit-width of the left operator exclusive
-- but other restrictions have varied:

-   Unsigned left shift has never had any restrictions on the first operand.
-   For signed left shift:
    -   In C++98, the result was fully unspecified.
    -   In C++11, the result was specified only if the first operand was
        non-negative and the result fit into the destination type -- that is, if
        no 1 bit is shifted into the sign bit.
    -   In C++14, the result was specified only if the first operand was
        non-negative and the result fit into the unsigned type corresponding to
        the destination type -- that is, if no 1 bit is shifted out of the sign
        bit.
    -   In C++20 onwards, there are no restrictions beyond a range restriction
        on the right operand, and the result is otherwise always specified, even
        if the left operand is negative.
-   Unsigned right shift has never had any restrictions on the first operand.
-   For signed right shift:
    -   In C++17 and earlier, if the left operand is negative, the result is
        implementation-defined.
    -   In C++20 onwards, the result is always specified, even if the left
        operand is negative.

There is a clear trend towards defining more cases, following two's complement
rules.

## Proposal

Use the same operator set as C++, but replace `~` with unary prefix `^`.

Define the behavior for all cases of `<<` and `>>` except where the right
operand is either negative or is greater than or equal to the bit-width of the
left operand.

## Details

See changes to the design, and in particular
[the new section on bitwise and shift operators](/docs/design/expressions/bitwise.md).

## Rationale

-   [Performance-critical software](/docs/project/goals.md#performance-critical-software)
    -   Bit operations are important low-level primitives. Providing operators
        for them is important in order to allow low-level high-performance code
        to be written elegantly in Carbon.
    -   By not defining semantics for `<<` and `>>` when the right-hand operand
        is out of range, we can directly use hardware instructions for these
        operations whose behavior in these cases vary between architectures.
-   [Code that is easy to read, understand, and write](/docs/project/goals.md#code-that-is-easy-to-read-understand-and-write)
    -   Using operator notation rather than function call notation for bitwise
        operators improves readability in code making heavy use of these
        operations.
-   [Practical safety and testing mechanisms](/docs/project/goals.md#practical-safety-and-testing-mechanisms)
    -   Carbon follows C++ in treating `<<` and `>>` as programming errors when
        the right hand operand is out of range, but Carbon guarantees that such
        errors will not directly result in unbounded misbehavior in hardened
        builds.
-   [Modern OS platforms, hardware architectures, and environments](/docs/project/goals.md#modern-os-platforms-hardware-architectures-and-environments)
    -   All hardware architectures we care to support are natively two's
        complement architectures, and that assumption allows us to define the
        semantics of shift operators in the way that makes the most sense for
        such architectures.
    -   Our bitwise operations make no assumptions about the endianness of the
        hardware architecture, although the shift operators make the most sense
        on a little-endian or big-endian architecture, which are the only
        endiannesses we expect to see in modern hardware platforms.
-   [Interoperability with and migration from existing C++ code](/docs/project/goals.md#interoperability-with-and-migration-from-existing-c-code)
    -   The same set of operators is provided as in C++, making it easy for
        programmers and programs to migrate, with the exception that the `~`
        operator is mechanically replaced with `^`. This change is expected to
        be easy for programmers to accommodate, especially given that Rust's
        choice to replace `~` with `!` does not seem to be a source of sustained
        complaints.
    -   The extensibility support reflects the full scope of operator
        overloading in C++, permitting separate overloading of each of the
        bitwise operators with custom return types. This should allow smooth
        interoperability with C++ overloaded operators.

## Alternatives considered

### Use different symbols for bitwise operators

The operator syntax for bitwise operators was decided in
[#545](https://github.com/carbon-language/carbon-lang/issues/545). Some of the
specific alternatives considered are discussed below.

#### Use a multi-character spelling

We considered using various multi-character spellings for the bitwise and, or,
xor, and complement operators:

-   `&:`, `|:`, `^:`, `~:`
-   `.&.`, `.|.`, `.^.`, `.~.`
-   `.*.`, `.+.`, `.!=.`, `.!.`
-   `/\`, `\/`, `(+)`, `-|`
-   `bitand`, `bitor`, `bitxor`, `compl`

The advantage of switching to such a set of operators is that this would free up
the single-character `&`, `|`, `^`, and `~` tokens for other uses that may occur
more frequently in Carbon programs. We have some candidate uses for these
operators already:

-   `&` is used for combining interfaces and as a unary address-of operator.
-   `|` may be useful for sum types, for alternatives in patterns, or as another
    kind of bracket as in
    [Ruby's lambda notation](https://ruby-doc.org/docs/ruby-doc-bundle/Manual/man-1.4/syntax.html#iter).
-   `~` may be useful as a destructive move notation.
-   `^` may be useful as a postfix pointer dereference operator.

Other motivations for switching to a different set of spellings include:

-   There are some concerns that `<` and `<<` are visually similar, analogous to
    `&` and `&&`.
-   Carbon has moved away from `&&` and other punctuation based _logical_
    operators and towards keywords like `and`. Bitwise operators could similarly
    switch to keywords like `bitand`.

However, moving substantially away from the C++ operator set comes with a set of
concerns:

-   There are strong established expectations and intuitions about these
    operators and their spellings among C++ practitioners.
-   In some of the code that uses these operators, they are used a lot, and a
    more cumbersome operator may consequently cause an outsized detriment on
    readability.
-   These operations are used particularly in the area of low-level,
    high-performance code, which is an area for which we want Carbon to be
    especially appealing. Using short operators for these operations
    demonstrates our commitment to providing good support for such code.
-   Even if we didn't use these operators as bit operators, we would still want
    to exercise caution when using them for some other purpose to avoid
    surprising C++ developers.
-   While some visual similarity exists such as between `<` and `<<`, the
    contexts in which these operators are used are sufficiently different to
    avoid any serious concerns.
-   The primary motivation of using `and` instead of `&&` doesn't apply for
    bitwise operators: the _logical_ operator represents _control flow_.
    Separating logical and bitwise "and" operations more visibly seems
    especially important because of this control flow semantic difference.
    Without any control flow and with the keywords being significantly longer
    for bitwise operations, the above considerations were the dominant ones that
    led us to stick with familiar `&` and `|` spellings.

#### Don't provide an xor operator

We considered omitting the `^` operator, providing this functionality in some
other way, such as a named function or an `xor` keyword, while keeping the `&`
and `|` symbols for bitwise operations. We could take a similar approach for the
complement operation, such as by using a `compl` keyword. The primary motivation
was to avoid spending two precious operator characters on relatively uncommon
operations. However, we did not want to apply the same change to `&` and `|`,
and it seemed important to maintain consistency between the three binary bitwise
operators from C++.

Using `^` for both operations provides some of the benefits here, allowing us to
reclaim `~`, without introducing the inconsistency that would result from using
keywords.

#### Use `~`, or some other symbol, for complement

We could follow C++ and use `~` as the complement operator. However, using `~`
for this purpose spends a precious operator character on a relatively uncommon
operation, and `~` is often visually confusible with `-`, creating the potential
for readability problems. Also, in C++, `~` is overloaded to also refer to
destruction, and we may want to use the same notation for destruction or
destructive move semantics in Carbon.

We found `^` to be a satisfying alternative with a good rationale and mnemonic:
`^` is a bit-flipping operator -- `a ^ b` flips the bits in `a` that are
specified in `b` -- and complement is an operator that flips _all_ the bits.
`^a` is equivalent to `a ^ n`, where `n` is the all-one-bits value in the type
of `a`.

We also considered using `!` for complement, like Rust does. Unlike in Rust,
this would not be a generalization of `!` on `bool`, because we use `not` for
`bool` negation, and repurposing `!` in this way compared to C++ seemed
confusing.

### Provide different operators for arithmetic and logical shifts

We could provide different operators for arithmetic right shift and logical
right shift. This might allow programmers to better express their intent.
However, it seems unnecessary, as using the type of the left operand is a
strategy that doesn't appear to have caused significant problems in practice in
the languages that have followed it.

Basing the kind of shift on the signedness of the left operand also follows from
viewing a negative number as having an infinite number of leading 1 bits, which
is the underlying mathematical model behind the two's complement representation.

### Provide rotate operators

We could provide bitwise rotation operators. However, there doesn't seem to be a
sufficient need to justify adding another operator symbol for this purpose.

### Guarantee behavior of large shifts

Logically, the behavior of shifts is meaningful for all values of the second
operand:

-   A shift by an amount greater than or equal to the bit-width of the first
    operand will shift out all of the original bits, producing a result where
    all value bits are the same.
-   A shift in one direction by a negative amount is treated as a shift in the
    opposite direction by the negation of that amount.

Put another way, we can view the bits of the first operand as an N-bit window
into an infinite sequence of bits, with infinitely many leading sign bits (all
zeroes for an unsigned value) and infinitely many trailing zero bits after a
notional binary point, and a shift moves that window around. Or equivalently, a
shift is always a multiplication by 2<sup>N</sup> followed by rounding and
wrapping.

We could provide the correct result for all shifts, regardless of the magnitude
of the second operand. This is the approach taken by Python, except that Python
rejects negative shift counts. The primary reason we do not do this is lack of
hardware support. For example, x86 does not have an instruction to perform this
operation. Rather, x86 provides shift instructions that mask off all bits of the
second operand except for the bottom 5 or 6, meaning that a left shift of a
64-bit operand by 64 will return the operand unchanged rather than producing
zero.

We could instead provide x86-like behavior, guaranteeing to consider only the
lowest `N` bits of the second operand when the first operand is an `iN` or `uN`.
This is the approach taken by Java for its 32-bit `int` type and 64-bit `long`
type, where the second operand is taken modulo 32 or 64, respectively, and in
JavaScript, where operands of bitwise and bit-shift operators are treated as
32-bit integers and the second operand of a shift is taken modulo 32. This
approach would provide an operation that can be implemented by a single
instruction on x86 platforms when `N` is 32 or 64, and for all smaller types and
for all other platforms the operation can be implemented with two instructions:
a mask and a shift. For larger types, single-instruction support may not be
available, but nonetheless the performance will be close to optimal, requiring
at most one additional mask. There is still some performance cost in some cases,
but the primary reason we do not do this is the same reason we choose to not
define signed integer overflow: this masked result is unlikely to be the value
that the developer actually wanted.

Instead of the above options, Carbon treats a second operand that is not in the
interval [0, N) as a programming error, just like signed integer overflow:

-   Debugging builds can detect and report this error without the risk of false
    positives.
-   Performance builds can optimize on the basis that this situation will not
    occur, and can in particular use the dedicated x86 instructions that ignore
    the high order bits of the second operand.
-   Optimized builds guarantee that either the programming error results in
    program termination or that _some_ value is produced, and moreover that said
    value is the result of applying _some_ mathematical shift to the input. For
    example, it's valid for an `i32` shift to be implemented by an x86 64-bit
    shift that will produce 0 if the second operand is in [32, 63) but that will
    treat a second operand of, say, 64 or -64 the same as 0.

### Support shifting a constant by a variable

We considered various ways to support

```
var a: i32 = ...;
var b: i32 = 1 << a;
var c: i32 = 1234 >> a;
```

with no explicit type specified for the first operand of a bit-shift operator.
We considered the following options:

-   Use the type of the second operand as the result type. This would be
    surprising, because the type of the second operand doesn't otherwise
    influence the result type of a built-in bit-shift operator.
-   Use some suitable integer type that can fit the first operand. However, this
    is unlikely to do the right thing for a left-shift, and will frequently pick
    either a type that's too large, resulting in the program being rejected due
    to narrowing, or a type that's too small, resulting in a program that has
    undefined behavior due to the second operand being too large. We could apply
    this approach only for right shifts, but it was deemed too inconsistent to
    use different rules for left and right shifts.
-   We could find a way to defer picking the type in which the operation is
    performed until later. For example, we could treat `1 << a` as a value of a
    new type that carries its left-hand operand as a type parameter and its
    right-hand operand as runtime state, and allow that type to be converted in
    the same way as its integer constant. However, this would introduce
    substantial complexity: reasonable and expected uses such as
    ```
    var mask: u32 = (1 << a) - 1;
    ```
    would require a second new type for a shifted value plus an offset, and
    general support would require a facility analogous to
    [expression templates](https://en.wikipedia.org/wiki/Expression_templates).
    Further, this facility would allow implicit conversions that notionally
    overflow, such as would happen in the above example when `a` is greater
    than 32.

In the absence of a good approach, we disallow such conversions for now. The
above example can be written as:

```
var a: i32 = ...;
var b: i32 = (1 as i32) << a;
var c: i32 = (1234 as i32) >> a;
```

### Converting complements to unsigned types

We view an integer constant has having infinitely many high-order sign bits
followed by some number of lower-order value bits. As a consequence, the
complement of a positive integer constant is negative. As a result, some
important forms of initialization use a negative integer constant initializer
for an unsigned type:

```
// Initializer here is the integer value -8.
var mask: u32 = ^7;
```

We considered some options for handling this:

-   We could allow negative integer constants to convert to unsigned types if
    doing so only discards sign bits. This violates the "semantics-preserving"
    rule for implicit conversions.
-   We could change our model of integer constants to distinguish between
    "masks" -- numbers with infinitely many 1 bits preceding the value bits that
    are nonetheless not considered to be negative. This was considered to
    introduce too much complexity.
-   We could allow conversions to unsigned types from signed types and negative
    constants in general, or at least in cases where the signed operand is no
    wider than the unsigned type, and perform wrapping. The latter option seems
    plausible, but we don't have sufficient motivation for it, and were worried
    about a risk of bugs from allowing an implicit conversion at runtime that
    converts a negative value to an unsigned type.
-   We could reject such initializations, with an explicit conversion required
    to convert such values to unsigned types. This seems to present unacceptable
    ergonomics for code performing bit-manipulation.

On balance, our preferred option was to permit implicit conversions from
negative literals to unsigned types so long as we only discard sign bits.
