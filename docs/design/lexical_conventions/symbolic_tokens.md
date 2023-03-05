<!-- omit in toc -->

# Symbolic Tokens

<!--
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

<!-- toc -->
<!-- omit in toc -->

## Table of contents

-   Overview
-   Details
-   Alternatives considered
-   References

<!-- tocstop -->

## Overview

Symbolic tokens are a set of tokens used to represent operators. Operators are
one use of symbolic tokens, but they are also used in patterns `:`, declarations
(`->` to indicate return type, `,` to separate parameters), statements (`;`, `=`, and so
on), and other places (`,` to separate function call arguments).

Some languages have a fixed set of symbolic tokens, For example:
[C++ operators](https://eel.is/c++draft/lex.operators) and
[rust operators](https://doc.rust-lang.org/book/appendix-02-operators.html).
While some others have extensible rules for defining operators, including the
facility for a developer to define operators that aren't part of the
baselanguage. For example:
[Swift operator rules](https://docs.swift.org/swift-book/ReferenceManual/LexicalStructure.html#ID418),
[Haskell operator rules](https://www.haskell.org/onlinereport/haskell2010/haskellch2.html#dx7-18008).

Carbon has a fixed set of tokens that represent operators, defined by the
language specification. Developers cannot define new tokens to represent new
operators.

Symbolic tokens are lexed using a "max munch" rule: at each lexing step, the
longest symbolic token defined by the language specification that appears
starting at the current input position is lexed, if any.

Not all uses of symbolic tokens within the Carbon grammar will be treated as
operators. For example, `(` and `)` tokens serves to delimit various grammar
productions, and we may not want to consider `.` to be an operator, because its
right "operand" is not an expression.

The presence or absence of whitespace around the symbolic token is used to
determine its fixity, in the same way we expect a human reader to recognize
them. For example, we want `a* - 4` to treat the `*` as a unary operator and the
`-` as a binary operator, while `a * -4` treats `*` as a mathematical operation
and `-` as the negative sign. Hence we can say that the whitespaces plays a
really important role here, and we use some rules to avoid confusion:

-   There can be no whitespace between a unary operator and its operand.
-   The whitespace around a binary operator must be consistent: either there is
    whitespace on both sides or on neither side.
-   If there is whitespace on neither side of a binary operator, the token
    before the operator must be an identifier, a literal, or any kind of closing
    bracket (for example, `)`, `]`, or `}`), and the token after the operator
    must be an identifier, a literal, or any kind of opening bracket (for
    example, `(`, `[`, or `{`).

## Details

Symbolic tokens are intended to be used for widely-recognized operators, such as
the mathematical operators `+`, `*`, `<`, and so on. Those used as operators
would generally be expected to also be meaningful for some user-defined types,
and should be candidates for being made overloadable once we support operator
overloading.

### Symbolic token list

The following is the initial list of symbolic tokens recognized in a Carbon
source file:

| Token | Explanation                                                                                                |
| ----- | ---------------------------------------------------------------------------------------------------------- |
| `*`   | Indirection, multiplication, and forming pointers                                                          |
| `&`   | Address-of or Bitwise AND                                                                                  |
| `=`   | Assignment                                                                                                 |
| `->`  | Return type and `p->x` equivalent to `(*p).x` (in C++)                                                     |
| `=>`  | Match syntax                                                                                               |
| `[]`  | Subscript                                                                                                  |
| `()`  | Function call and function declaration                                                                     |
| `{}`  | Struct literals, blocks of control flow statements and the bodies of definitions (classes, functions, etc) |
| `,`   | Separate arguments in a function call, elements of a tuple, or parameters of a function declaration        |
| `.`   | Member access                                                                                              |
| `:`   | Scope                                                                                                      |

This list is expected to grow over time as more symbolic tokens are required by
language proposals.

Note: The above list only covers up to
[#601](https://github.com/carbon-language/carbon-lang/pull/601) and more have
been added since that are not reflected here.

### Whitespace

to support the use of the same symbolic token as a prefix operator, an infix
operator, and a postfix operator (in some cases) we want a rule that allows us
to simply and unambiguously parse operators that might have all three fixities.

For example, given the expression `a * - b`, there are two possible parses:

-   As `a * (- b)`, multiplying `a` by the negation of `b`.
-   As `(a *) - b`, subtracting `b` from the pointer type `a *`.

The chosen rule to distinguish such cases is to consider the presence or absence
of whitespace, as it strikes a good balance between simplicity and
expressiveness for the programmer and simplicity and good support for error
recovery in the implementation. Hence `a * -b` uses the first interpretation,
`a* - b` uses the second interpretation, and other combinations (`a*-b`,
`a *- b`, `a* -b`, `a * - b`, `a*- b`, `a *-b`) are rejected as errors.

We require whitespace to be present or absent around the operator to indicate
its fixity, as this is a cue that a human reader would use to understand the
code: binary operators have whitespace on both sides, and unary operators lack
whitespace between the operator and its operand.

But in some cases omitting the whitespace around a binary operator aids
readability, such as in expressions like `2*x*x + 3*x + 1`, hence we have an
allowance in such cases. In this case the operator with whitespace on neither
side, if the token immediately before the operator indicates it is the end of an
operand, and the token immediately after the operator indicates it is the
beginning of an operand, the operator is treated as binary.

The defined set of tokens that constitutes the beginning or end of an operand
are:

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

## Alternatives considered

-   Lexing the longest sequence of symbolic characters rather than lexing only
    the longest known operator.

    Advantages - Adding new operators could be done without any change to the
    lexing rules. - If unknown operators are rejected, adding new operators
    would carry no risk of changing the meaning of existing valid code.

    Disadvantages:

    -   Sequences of prefix or postfix operators would require parentheses or
        whitespace. For example, `Int**` would lex as `Int` followed by a single
        `**` token, and `**p` would lex as a single `**` token followed by `p`,
        if there is no `**` operator. While we could define `**`, `***`, and so
        on as operators, doing so would add complexity and inconsistency to the
        language rules.

-   Supporting an extensible operator set, giving the developer the option to
    add new operators. Advantages:

    -   This would increase expressivity, especially for embedded
        domain-specific languages.

    Disadvantages:

    -   This would harm readability, at least for those unfamiliar with the code
        using the operators.
    -   This could harm our ability to evolve the language, by admitting the
        possibility of a custom operator colliding with a newly-introduced
        standard operator, although this risk could be reduced by providing a
        separate lexical syntax for custom operators.
    -   We would need to either lex the longest sequence of symbolic characters
        we can, which has the same disadvantage discussed for that approach
        above, or use a more sophisticated rule to determine how to split
        operators -- perhaps based on what operator overloads are in scope --
        increasing complexity.

-   We could apply different whitespace restrictions or no whitespace
    restrictions
    ([#520](https://github.com/carbon-language/carbon-lang/issues/520)). We
    could require whitespace around a binary operator followed by `[` or `{`. In
    particular, for examples such as:

    ```
    fn F() -> Int*{ return Null; }
    var n: Int = pointer_to_array^[i];
    ```

    This would allow us to form a unary operator instead of a binary operator,
    which is likely to be more in line with the developer's expectations.

    Advantages:

    -   Room to add a postfix `^` dereference operator, or similarly any other
        postfix operator producing an array, without creating surprises for
        pointers to arrays.
    -   Allows the whitespace before the `{` of a function body to be
        consistently omitted if desired.

    Disadvantages:

    -   The rule would be more complex, and would be asymmetric: we must allow
        closing square brackets before unspaced binary operators to permit
        things like `arr[i]*3`.
    -   Would interact badly with expression forms that begin with a `[` or `{`,
        for example `Time.Now()+{.seconds = 3}` or `names+["Lrrr"]`.

## References

-   Proposal
    [#601: Symbolic tokens](https://github.com/carbon-language/carbon-lang/pull/601)
