# Operator tokens

<!--
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

[Pull request](https://github.com/carbon-language/carbon-lang/pull/601)

<!-- toc -->

## Table of contents

-   [Problem](#problem)
-   [Background](#background)
-   [Proposal](#proposal)
-   [Details](#details)
    -   [Two kinds of operator tokens](#two-kinds-of-operator-tokens)
    -   [Symbolic token list](#symbolic-token-list)
    -   [Whitespace](#whitespace)
-   [Rationale based on Carbon's goals](#rationale-based-on-carbons-goals)
-   [Alternatives considered](#alternatives-considered)

<!-- tocstop -->

## Problem

Carbon needs a set of tokens to represent operators.

## Background

Some languages have a fixed set of operator tokens. For example:

-   [C++ operators](https://eel.is/c++draft/lex.operators)
    -   The keyword operators `and`, `or`, etc. are lexical synonyms for
        corresponding symbolic operators `&&`, `||`, etc.
-   [Rust operators](https://doc.rust-lang.org/book/appendix-02-operators.html)

Other languages have extensible rules for defining operators, including the
facility for a developer to define operators that aren't part of the base
language. For example:

-   [Swift operator rules](https://docs.swift.org/swift-book/ReferenceManual/LexicalStructure.html#ID418)
-   [Haskell operator rules](https://www.haskell.org/onlinereport/haskell2010/haskellch2.html#dx7-18008)

Operators tokens can be formed by various rules, for example:

-   At each lexing step, form the longest known operator token possible from the
    remaining character sequence. For example, in C++, `a += b` is 3 tokens and
    `a =+ b` is four tokens, because there are `+`, `=`, and `+=` operators, but
    there is no `=+` operator. This approach is sometimes known as "max munch".
-   At each lexing step, treat the longest sequence of operator-like characters
    possible as an operator. The program is invalid if there is no such
    operator. For example, in a C++-like language using this approach, `a =+ b`
    would be invalid instead of meaning `a = (+b)`.
-   Use semantic information to determine how to split a sequence of operator
    characters into one or more operators, for example based on the types of the
    operands.

## Proposal

Carbon has a fixed set of tokens that represent operators, defined by the
language specification. Developers cannot define new tokens to represent new
operators; there may be facilities to overload operators, but that is outside
the scope of this proposal. There are two kinds of tokens that represent
operators:

-   _Symbolic tokens_ consist of one or more symbol characters. In particular,
    such a token contains no characters that are valid in identifiers, no quote
    characters, and no whitespace.
-   _Keywords_ follow the lexical rules for words.

Symbolic tokens are lexed using a "max munch" rule: at each lexing step, the
longest symbolic token defined by the language specification that appears
starting at the current input position is lexed, if any.

Not all uses of symbolic tokens within the Carbon grammar will be as operators.
For example, we will have `(` and `)` tokens that serve to delimit various
grammar productions, and we may not want to consider `.` to be an operator,
because its right "operand" is not an expression.

When a symbolic token is used as an operator, we use the presence or absence of
whitespace around the symbolic token to determine its fixity, in the same way we
expect a human reader to recognize them. For example, we want `a* - 4` to treat
the `*` as a unary operator and the `-` as a binary operator, while `a * -4`
results in the reverse. This largely requires whitespace on only one side of a
unary operator and on both sides of a binary operator. However, we'd also like
to support binary operators where a lack of whitespace reflects precedence such
as`2*x*x + 3*x + 1` where doing so is straightforward. The rules we use to
achieve this are:

-   There can be no whitespace between a unary operator and its operand.
-   The whitespace around a binary operator must be consistent: either there is
    whitespace on both sides or on neither side.
-   If there is whitespace on neither side of a binary operator, the token
    before the operator must be an identifier, a literal, or any kind of closing
    bracket (for example, `)`, `]`, or `}`), and the token after the operator
    must be an identifier, a literal, or any kind of opening bracket (for
    example, `(`, `[`, or `{`).

This proposal includes an initial set of symbolic tokens covering only the
grammar productions that have been approved so far. This list should be extended
by proposals that use additional symbolic tokens.

## Details

### Two kinds of operator tokens

Two kinds of operator tokens are proposed. These two kinds are intended for
different uses, not as alternate spellings of the same functionality:

-   Symbolic tokens are intended to be used for widely-recognized operators,
    such as the mathematical operators `+`, `*`, `<`, and so on.
    -   Symbolic tokens used as operators would generally be expected to also be
        meaningful for some user-defined types, and should be candidates for
        being made overloadable once we support operator overloading.
-   Keywords are intended to be used for cases such as the following:
    -   Operators that perform flow control, such as `and`, `or`, `throw`,
        `yield`, and operators closely connected to these, such as `not`. It is
        important that these stand out from other operators as they have action
        that goes beyond evaluating their operands and computing a value.
    -   Operators that are rare and that we do not want to spend our finite
        symbolic token budget on, such as perhaps xor or bit rotate.
    -   Operators with very low precedence, and perhaps certain operators with
        very high precedence.
    -   Special-purpose operators for which there is no conventional established
        symbol and for which we do not want to invent one, such as `as`.

The example operators in this section are included only to motivate the two
kinds of operator token; those specific operators are not proposed as part of
this proposal.

### Symbolic token list

The following is the initial list of symbolic tokens recognized in a Carbon
source file:

|     |      |      |     |     |     |
| --- | ---- | ---- | --- | --- | --- |
| `(` | `)`  | `{`  | `}` | `[` | `]` |
| `,` | `.`  | `;`  | `:` | `*` | `&` |
| `=` | `->` | `=>` |     |     |     |

This list is expected to grow over time as more symbolic tokens are required by
language proposals.

### Whitespace

We wish to support the use of the same symbolic token as a prefix operator, an
infix operator, and a postfix operator, in some cases. In particular, we have
[decided in #523](https://github.com/carbon-language/carbon-lang/issues/523)
that the `*` operator should support all three uses; this operator will be
introduced in a future proposal. In order to support such usage, we want a rule
that allows us to simply and unambiguously parse operators that might have all
three fixities.

For example, given the expression `a * - b`, there are two possible parses:

-   As `a * (- b)`, multiplying `a` by the negation of `b`.
-   As `(a *) - b`, subtracting `b` from the pointer type `a *`.

Our chosen rule to distinguish such cases is to consider the presence or absence
of whitespace, as we think this strikes a good balance between simplicity and
expressiveness for the programmer and simplicity and good support for error
recovery in the implementation. `a * -b` uses the first interpretation, `a* - b`
uses the second interpretation, and other combinations (`a*-b`, `a *- b`,
`a* -b`, `a * - b`, `a*- b`, `a *-b`) are rejected as errors.

In general, we require whitespace to be present or absent around the operator to
indicate its fixity, as this is a cue that a human reader would use to
understand the code: binary operators have whitespace on both sides, and unary
operators lack whitespace between the operator and its operand. We also make
allowance for omitting the whitespace around a binary operator in cases where it
aids readability to do so, such as in expressions like `2*x*x + 3*x + 1`: for an
operator with whitespace on neither side, if the token immediately before the
operator indicates it is the end of an operand, and the token immediately after
the operator indicates it is the beginning of an operand, the operator is
treated as binary.

We define the set of tokens that constitutes the beginning or end of an operand
as:

-   Identifiers, as in `x*x + y*y`.
-   Literals, as in `3*x + 4*y` or `"foo"+s`.
-   Brackets of any kind, facing away from the operator, as in `f()*(n + 3)` or
    `args[3]*{.real=4, .imag=1}`.

For error recovery purposes, this rule functions best if no expression context
can be preceded by a token that looks like the end of an operand and no
expression context can be followed by a token that looks like the start of an
operand. One known exception to this is in function definitions:

```
fn F(p: Int *) -> Int * { return p; }
```

Both occurrences of `Int *` here are erroneous. The first is easy to detect and
diagnose, but the second is more challenging, if `{...}` is a valid expression
form. We expect to be able to easily distinguish between code blocks starting
with `{` and expressions starting with `{` for all cases other than `{}`.
However, the code block `{}` is not a reasonable body for a function with a
return type, so we expect errors involving a combination of misplaced whitespace
and `{}` to be rare, and we should be able to recover well from the remaining
cases.

From the perspective of token formation, the whitespace rule means that there
are four _variants_ of each symbolic token:

-   A symbolic token with whitespace on both sides is a _binary_ variant of the
    token.
-   A symbolic token with whitespace on neither side, where the preceding token
    is an identifier, literal, or closing bracket, and the following token is an
    identifier, literal, or `(`, is also a _binary_ variant of the token.
-   A symbolic token with whitespace on neither side that does not satisfy the
    preceding rule is a _unary_ variant of the token.
-   A symbolic token with whitespace on the left side only is a _prefix_ variant
    of the token.
-   A symbolic token with whitespace on the right side only is a _postfix_
    variant of the token.

When used in non-operator contexts, any variant of a symbolic token is
acceptable. When used in operator contexts, only a binary variant of a token can
be used as a binary operator, only a prefix or unary variant of a token can be
used as a prefix operator, and only a postfix or unary variant of a token can be
used as a postfix operator.

This whitespace rule has been
[implemented in the Carbon toolchain](https://github.com/carbon-language/carbon-lang/pull/576)
for all operators by tracking the presence or absence of trailing whitespace as
part of a token, and
[in executable semantics](https://github.com/carbon-language/carbon-lang/commit/04d3a885ae01a779aadb19f51ec7a5a12ffe295c)
for the `*` operator by forming four different token variants as described
above.

The choice to disallow whitespace between a unary operator and its operand is
_experimental_.

## Rationale based on Carbon's goals

-   Software and language evolution

    -   By not allowing user-defined operators, we reduce the possibility that
        operators added to the language later will conflict with existing uses
        in programs. Due to the use of a max munch rule, we might add an
        operator that causes existing code to be interpreted differently, but
        such problems will be easy to detect and resolve, because we know the
        operator set in advance.

-   Code that is easy to read, understand, and write

    -   The fixed operator set means that developers don't need to understand an
        unbounded and extensible number of operators and precedence rules. The
        fixed operator set encourages functionality that does not correspond to
        a well-known operator symbol to be exposed by way of a named operation
        instead of a symbol, improving readability among developers not familiar
        with a codebase.
    -   Requiring whitespace to be used consistently around operators reduces
        the possibility for confusing formatting.
    -   Permitting whitespace on either both sides of a binary operator or on
        neither side allows expressions such as `2*x*x + 3*x + 1` to use the
        absence of whitespace to improve readability. Because the language
        officially sanctions both choices, the formatting tool can be expected
        to preserve the user's choice.
    -   The choice to lex the longest known symbolic token rather than the
        longest sequence of symbolic characters makes it easier to write
        expressions involving a series of prefix or postfix operators, such as
        `x = -*p;`.

-   Interoperability with and migration from existing C++ code

    -   The fixed operator set makes a mapping between Carbon operators and C++
        operators easier, by avoiding any desire to map arbitrary user-defined
        Carbon operators into a C++ form.
    -   The choice of a fixed operator set and a "max munch" rule will be
        familiar to C++ developers, as it is the same approach taken by C++.
    -   The whitespace rule permits the `*` operator to be used for all of
        multiplication, dereference, and pointer type formation, as in C++,
        while still permitting Carbon to treat type expressions as expressions.

## Alternatives considered

We could lex the longest sequence of symbolic characters rather than lexing only
the longest known operator.

Advantages:

-   Adding new operators could be done without any change to the lexing rules.
-   If unknown operators are rejected, adding new operators would carry no risk
    of changing the meaning of existing valid code.

Disadvantages:

-   Sequences of prefix or postfix operators would require parentheses or
    whitespace. For example, `Int**` would lex as `Int` followed by a single
    `**` token, and `**p` would lex as a single `**` token followed by `p`, if
    there is no `**` operator. While we could define `**`, `***`, and so on as
    operators, doing so would add complexity and inconsistency to the language
    rules.

We could support an extensible operator set, giving the developer the option to
add new operators.

Advantages:

-   This would increase expressivity, especially for embedded domain-specific
    languages.

Disadvantages:

-   This would harm readability, at least for those unfamiliar with the code
    using the operators.
-   This could harm our ability to evolve the language, by admitting the
    possibility of a custom operator colliding with a newly-introduced standard
    operator, although this risk could be reduced by providing a separate
    lexical syntax for custom operators.
-   We would need to either lex the longest sequence of symbolic characters we
    can, which has the same disadvantage discussed for that approach above, or
    use a more sophisticated rule to determine how to split operators -- perhaps
    based on what operator overloads are in scope -- increasing complexity.

We could apply different whitespace restrictions or no whitespace restrictions.
See [#520](https://github.com/carbon-language/carbon-lang/issues/520) for
discussion of the alternatives and the leads decision.

We could require whitespace around a binary operator followed by `[` or `{`. In
particular, for examples such as:

```
fn F() -> Int*{ return Null; }
var n: Int = pointer_to_array^[i];
```

... this would allow us to form a unary operator instead of a binary operator,
which is likely to be more in line with the developer's expectations.

Advantages:

-   Room to add a postfix `^` dereference operator, or similarly any other
    postfix operator producing an array, without creating surprises for pointers
    to arrays.
-   Allows the whitespace before the `{` of a function body to be consistently
    omitted if desired.

Disadvantages:

-   The rule would be more complex, and would be asymmetric: we must allow
    closing square brackets before unspaced binary operators to permit things
    like `arr[i]*3`.
-   Would interact badly with expression forms that begin with a `[` or `{`, for
    example `Time.Now()+{.seconds = 3}` or `names+["Lrrr"]`.
