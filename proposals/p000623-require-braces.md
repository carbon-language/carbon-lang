# Require braces

<!--
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

[Pull request](https://github.com/carbon-language/carbon-lang/pull/623)

<!-- toc -->

## Table of contents

-   [Problem](#problem)
-   [Background](#background)
    -   [Consistency](#consistency)
    -   [C++ style rules](#c-style-rules)
    -   [`goto fail`](#goto-fail)
-   [Proposal](#proposal)
-   [Rationale based on Carbon's goals](#rationale-based-on-carbons-goals)
-   [Alternatives considered](#alternatives-considered)
    -   [Optional braces](#optional-braces)
    -   [Optional parentheses](#optional-parentheses)
    -   [`elif`](#elif)

<!-- tocstop -->

## Problem

In C++, braces are often optional, such as in:

```
for (Shape x : shapes)
  Draw(x);
```

Carbon adopted this design choice by default in proposals
[#285](/proposals/p0285.md), [#340](/proposals/p0340.md), and
[#353](/proposals/p0353.md). But should we keep it?

## Background

### Consistency

We were adopting the C++ syntax primarily for easy consistency with C++. For
other languages:

-   Some modern languages require braces, including Go, Rust, and Swift. Note
    some which require braces make parentheses optional.
-   Kotlin is an example modern language which does not require braces.
-   Older languages tend to make braces optional, including Java, JavaScript,
    and C#.

### C++ style rules

There is a
[CERT rule](https://wiki.sei.cmu.edu/confluence/display/c/EXP19-C.+Use+braces+for+the+body+of+an+if%2C+for%2C+or+while+statement)
for when to use braces.

### `goto fail`

Apple had an infamous `goto fail` bug:

-   https://www.imperialviolet.org/2014/02/22/applebug.html
-   https://dwheeler.com/essays/apple-goto-fail.html

## Proposal

Carbon should require braces when they might otherwise be optional. We should
continue to require parentheses. This currently comes up in `if`/`else`,
`while`, and `for`.

We will allow `else if` as a special structure for repeated `if` statements due
to their frequency. For example, `if (...) { ... } else if (...) { ... }`
behaves identically to `if (...) { ... } else { if (...) { ... } }`.

## Rationale based on Carbon's goals

-   **Software and language evolution**: Reducing parsing ambiguity of curly
    braces is important for expanding the syntactic options for Carbon.

-   **Code that is easy to read, understand, and write**: We have a preference
    to give only one way to do things, and optional braces are inconsistent with
    that. It's also easier to understand and parse code that uses braces, and
    defends against the possibility of `goto fail`-style bugs, whether
    accidental or malicious.

-   **Interoperability with and migration from existing C++ code**: While C++
    does make braces optional in related situations, we believe this isn't
    fundamental to migration or familiarity for C++ developers, so this goal is
    not meaningfully impacted by this change.

## Alternatives considered

### Optional braces

We could instead keep braces optional, such as:

```carbon
if (x)
  return y;
```

Advantages:

-   Consistent with C++.
-   Allows for greater brevity in code, such as `if (!success) return error;`.

Disadvantages:

-   Gives two ways of doing the same thing.
    -   Style guides will make choices, creating more contextual coding style.
    -   Some community members opined that requiring braces was the only
        solution which is both appealing and easy to automate formatting for.
        -   A contrary opinion is that single-line if statements without braces
            are more appealing, and not significantly more difficult to
            automate.
-   More complex and harder to parse.

    -   Nested `if` statements can have unclear `else` bindings. For example:

        ```carbon
        if (x)
        if (y)
          DoIfY();
        else
          DoIfNotY();
        else
          DoIfNotX();
        ```

    -   If Carbon were to reuse `if` and `else` keywords for a ternary operator,
        that could omit braces in order to avoid ambiguity. For example,
        `int x = if y then 3 else 7;`.

-   Developers are known to make mistakes adding statements to conditionals
    missing braces, keeping consistent indentation, and missing the incorrect
    behavior due to cognitive load. For example:

    ```carbon
    if (x)
      return y;
    ```

    ->

    ```carbon
    if (x)
      print("Returning y");
      return y;
    ```

    -   Developers commonly want to add debugging statements to conditionals,
        and missing braces can lead to these kinds of errors.
    -   It's possible that, over time, adding braces to add debug statements
        then removing the debug statements without removing statements will lead
        to braces remaining when braces are optional.
        -   This can be addressed by style guides and automated formatting that
            inserts or removes braces as appropriate, cleaning up after
            temporary edits.
        -   Some people feel the churn of adding braces and later removing them
            is a significant toil that can be avoided by always adding them,
            such as [here](https://wiki.c2.com/?AlwaysUseBracesOnIfThen).
    -   It can more easily lead to bugs like `goto fail;`.
        -   However, as the blog post points out, one can have incorrect
            indentation with braces too.

### Optional parentheses

Languages which make braces required can make parentheses optional, or even
disallow them, such as:

```carbon
if x {
  return y;
}
```

Advantages:

-   Exchanges verbosity in syntax, requiring `{}` but removing `()`.
    -   Particularly useful in that many conditionals have `{}` regardless of
        optionality, but all conditionals could in theory remove `()`.
-   Cross-language consistency with languages that require `{}`.

Disadvantages:

-   Visual inconsistency with C++.
-   Parentheses make parsing less likely to encounter ambiguity.
-   Optional parentheses lead to two ways of doing the same thing, although
    disallowing them entirely could address this.

### `elif`

We could make `else if` a single token, such as `elseif` or `elif`.

Advantages:

-   Can parse as a single token.

Disadvantages:

-   Visual inconsistency with C++.
    -   `else if` also appears in some languages that require braces, such as
        Go, Rust, and Swift.
