# Assignment statements

<!--
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

[Pull request](https://github.com/carbon-language/carbon-lang/pull/2511)

<!-- toc -->

## Table of contents

-   [Abstract](#abstract)
-   [Problem](#problem)
-   [Background](#background)
-   [Proposal](#proposal)
-   [Details](#details)
-   [Rationale](#rationale)
-   [Alternatives considered](#alternatives-considered)
    -   [Allow assignment as a subexpression](#allow-assignment-as-a-subexpression)
    -   [Allow chained assignment](#allow-chained-assignment)
    -   [Do not provide increment and decrement](#do-not-provide-increment-and-decrement)
    -   [Treat increment as syntactic sugar for adding `1`](#treat-increment-as-syntactic-sugar-for-adding-1)
    -   [Define `$` in terms of `$=`](#define--in-terms-of-)
    -   [Do not allow overloading the behavior of `=`](#do-not-allow-overloading-the-behavior-of-)
    -   [Treat the left hand side of `=` as a pattern](#treat-the-left-hand-side-of--as-a-pattern)
    -   [Different names for interfaces](#different-names-for-interfaces)

<!-- tocstop -->

## Abstract

Assignment is permitted only as a complete statement, not as a subexpression.
Assignment and compound assignment syntax follow C++ in all other respects.
Pre-increment is provided. Post-increment is not. Uses of all of these operators
are translated into calls to interface members.

## Problem

Assignment is a cornerstone of imperative programming. Carbon does not currently
have an approved proposal describing the syntax and interpretation of
assignment.

## Background

In C-family languages, there are three kinds of assignment-like operators:

-   Simple assignment: `variable = value`
-   Compound assignment: `variable $= value`, for some binary operators `$`,
    meaning `variable = variable $ value`, except that `variable` is evaluated
    only once.
-   Increment and decrement: `++variable` and `--variable` meaning
    `variable += 1` and `variable -= 1`, and `variable++` and `variable--`
    acting similarly but returning the prior value of `variable`.

These operators behave mostly like other binary operators, and in particular the
above expression forms can be used as subexpressions of other expressions.

Chained assignment is supported, and associates from right to left: `a = b = 1`
assigns `1` to `b`, then assigns the result of that assignment to `a`. In C++,
that result is typically an lvalue referring to `b`, but in C, it is an rvalue
containing the result of converting `1` to the type of `b`. Note that all other
operators in C and C++ associate in the other direction.

These operators have a collection of known issues, including:

-   Confusion between assignment and comparison, for example in constructs such
    as `if (variable = 3) { ... }`. This is sufficiently rife that a de facto
    compiler-warning-enforced convention has arisen of using additional
    parentheses for the rare cases when assignment is intended:
    `if ((variable = 3))`.
-   Risk of unsequenced modification and access to the same variable, resulting
    in undefined behavior. For example, `n = a + n++;` has undefined behavior in
    C and C++ for this reason, because the act of incrementing `n` is not
    sequenced with respect to the rest of the computation, including the
    assignment.
-   In C++, post-increment can be a performance trap, because it is expected to
    return the old value of the variable, which might otherwise not be
    preserved. The additional copying may be optimized away if the `operator++`
    can be inlined, at the cost of additional work for the compiler.
-   In C++, overloading an operator `$` does not automatically provide a
    matching `$=`, resulting in additional work or incomplete operator sets.

## Proposal

C-family assignment operators are provided as statements:

-   `variable = value;` is a simple assignment statement.
-   `variable $= value;` is a compound assignment for each binary operator `$`,
    other than comparisons. `<=` and `>=` mean "less than or equal to" and
    "greater than or equal to", not "compare and assign".
-   `++variable;` and `--variable;` are supported as increment and decrement
    syntax. Because these are statements, there is no distinction between pre-
    and post-increment, and post-increment is not provided.

These operations are translated into calls on interfaces.

## Details

See the changes to the design.

This proposal does not define the semantics of assignment that are provided for
classes by default. Leads issue
[#686](https://github.com/carbon-language/carbon-lang/issues/686) gives some
rules, but those rules are not part of this proposal.

## Rationale

-   [Language tools and ecosystem](/docs/project/goals.md#language-tools-and-ecosystem)
    -   The values of variables can only change if either the address is taken,
        including implicitly by `addr` self parameters, or at assignment
        statements, making it easier to reason about where the value of a
        variable can change in the control flow of a function.
-   [Software and language evolution](/docs/project/goals.md#software-and-language-evolution)
    -   This approach is conservative and can evolve to support assignment as a
        subexpression.
-   [Code that is easy to read, understand, and write](/docs/project/goals.md#code-that-is-easy-to-read-understand-and-write)
    -   Assignments in subexpressions tend to be hard for humans to read and
        understand. Disallowing them at the language level avoids the potential
        for confusion, making code easier to read at the cost of making certain
        constructs such as `n = arr[i++];` a little harder to write.
    -   Easier to write a complete operator set because both `$` and `$=` can be
        provided by implementing a single interface.
-   [Practical safety and testing mechanisms](/docs/project/goals.md#practical-safety-and-testing-mechanisms)
    -   Replaces correctness warning on `if (a = b)` with a language rule.
-   [Interoperability with and migration from existing C++ code](/docs/project/goals.md#interoperability-with-and-migration-from-existing-c-code)
    -   Providing largely the same set of symbols makes migration easier.
    -   Some cost may be imposed by forcing a translation from assignment in
        subexpressions to assignment as separate statements.

## Alternatives considered

### Allow assignment as a subexpression

We could allow some or all forms of assignment as subexpressions, either with
the same syntax or with some other syntax. This proposal does not support
assignment as a subexpression because the utility of this feature in C++ is very
limited and leads to problems where equality comparison and assignment are
easily confused. Modeling assignment as a statement also makes it easier to
treat it as the transition point between a variable being in an unformed state
and in a fully-formed state, as it prevents such transitions from happening at a
not-fully-sequenced point within the evaluation of an expression.

To avoid the syntactic confusion between assignment and comparison, we could
allow assignment as a subexpression with some other syntax. For example, we
could adopt Python's "walrus operator" `variable := value`. For now, we are
choosing to not pursue this option in order to determine how much motivation
there is for such a feature.

If we allow a three-term form of `for` statements, `for (init; cond; incr)`, we
could allow assignment in the `incr` term. However, we currently do not support
this syntax.

The absence of assignment as a subexpression can be worked around with an
ergonomic cost, either by using a direct function call `x.(Assign.Op)(y)` or
with a lambda wrapping an assignment. Post-increment `a++` could be transformed
into `${var b: auto = a; ++a; return b;}`, where `${...}` is a placeholder for
Carbon's eventual lambda syntax.

Workarounds that are not assignment statements would likely be treated as
capturing `x` for static analysis purposes, not as definitely initializing `x`,
so unformedness checks and other similar checks we might choose to include in
the Carbon language design may treat these workarounds conservatively.

```
var a: i32;
var b: i32;
var c: i32;
// ✅ OK, per pending proposal #2006.
a = 1;
// 🤷 Might result in an error or warning;
// `b` is in an unformed state.
b.(Assign.Op)(1);
// 🤷 Undecided whether we will guarantee that
// assignment to `c` precedes read from `c`.
a += c.(Assign.Op)(1) + c;
```

### Allow chained assignment

As a special case of assignment as a subexpression, we could allow chained
assignment:

```
a = b = c = 0;
```

We could restrict this to only apply to simple assignment, not cases like
`a += b -= c *= 2;`

In leads issue [#451](https://github.com/carbon-language/carbon-lang/issues/451)
it was decided that we would initially not support chained assignment, although
this was largely the result of a lack of arguments in favor of support that
would justify the complexity of adding a special case, and should be
reconsidered if compelling arguments in favor of chained assignment are
uncovered.

### Do not provide increment and decrement

We could remove `++var;` and `--var;` in favor of `var += 1;` and `var -= 1;`.
However, developers coming from C-family languages will expect these operators
to exist, and they may more directly convey the intended semantics of counting
and navigating in a one-dimensional granular space than a `+=` operation would.

### Treat increment as syntactic sugar for adding `1`

We could treat `++a;` as being syntactic sugar for `a += 1;`, in the same way
that we treat `a += 1;` as being syntactic sugar for `a = a + 1;`, and similarly
for `--a;`. This would mean that floating-point types gain increment and
decrement operators, as in C++.

The literal `1` has its own type, which means that this approach would not
require a type to support adding integers in general in order to support
increment and decrement, and types such as non-random-access iterators could
provide an `it + 1` operation without exposing a non-constant-time `it + n`
operation.

One potential advantage of this approach is that a generic constraint that is
sufficient to allow `a = a + 1;` to type-check would also allow `++a;` to
type-check. For example, `T:! Assign & Add where i32 is ImplicitAs(.Self)` would
suffice to allow such an increment, as would a facet type with a narrower
`ImplicitAs` constraint that permits only the literal `1`. However, this benefit
is minor, given that writing `a += 1;` instead is not a major burden.

The cost of making this change would be that the semantics of increment and
decrement are tied to a very numerical notion of "adding 1". That may not be
appropriate for all types that want to support a more general notion of "move to
the next value" or "move to the previous value", such as a more generalized
notion of cursor. It may also not be appropriate for all types that support
addition of exactly 1 to support increment; for example:

-   For floating-point types, adding exactly 1 does not necessarily produce a
    different number, and there is a different meaningful notion of "move to the
    next value" -- namely, moving to the next _representable_ value -- which may
    be intended instead. In C++ code, where increment of floating-point types is
    permitted, it is vanishingly rare.
-   For complex numbers, for example in a Gaussian integer type, having `++c;`
    move one unit in the real direction seems arbitrary.
-   For a rational number type, as for floating-point types, adding exactly 1
    seems unlikely to be a common "navigation" operation, even though there is
    no other reasonable notion of "move to the next value".

### Define `$` in terms of `$=`

Instead of defining `$=` in terms of `$` and `=`, we could define `$` in terms
of `$=` and copying. Our experience from C++ is that `$=` can frequently be
implemented more efficiently than `$`, by operating in-place and reusing
allocated storage from the left-hand operand, so this might be a better default.

There are a few reasons why we choose to not do this:

-   The direction in this proposal is expected to be less surprising. Defining
    `$=` in terms of `$` and `=` seems more in line with programmer expectations
    based both on the morphology of the token and on how it is generally taught.
-   Under the rules in this proposal, an `Add & Assign` constraint ends up being
    effectively equivalent to an `AddAssign` constraint, due to the blanket
    implementation of `AddAssign` in terms of `Add` and `Assign`. This means
    that constraining a type to provide both `Add` and `Assign` is sufficient to
    use `+=`, which seems desirable. If the defaults were reversed, this would
    not be achievable.
-   If the reverse rule were adopted, two separate `impl`s would still be
    required in order to permit implicit conversions on the left hand side of
    the `$` for a type, but not on the left hand side of a `$=`.
-   There are cases where `$` can be implemented more efficiently than making a
    copy and performing a `$=` operation. For example, if the type is large,
    implementing `$` in terms of `$=` can require two passes over the
    destination instead of one, which may increase the constant factor
    performance of the operation.

### Do not allow overloading the behavior of `=`

We could define that an `=` expression always carries out these steps:

-   Initialize a value of the left-hand operand's type from the right-hand
    operand.
-   Destroy the left-hand operand.
-   Move the value created earlier into the left-hand operand.

However, this removes some flexibility and could harm performance, for example
in the case where the left-hand operand has a buffer that it could reuse to
store its new value. This would also be a significant deviation from C++, where
some types take advantage of this additional flexibility, and would be expected
to harm interoperability and migration.

### Treat the left hand side of `=` as a pattern

We could allow pattern-like syntax on the left hand side of `=` instead of an
expression.

```
fn GCD(var a: i32, var b: i32) -> i32 {
  if (b < a) {
    // Swap `a` and `b`.
    (a, b) = (b, a);
  }
  while (a != 0) {
    // Calculate both `b % a` and `a`,
    // then assign to both `a` and `b`.
    (a, b) = (b % a, a);
  }
  return b;
}
```

However, this would be a novel interpretation of pattern syntax: there is no
mechanism in the current pattern syntax to assign to an existing variable.
Pattern-matching `(a, b)` against `(b % a, a)` would instead compare `a` to
`b % a` and compare `b` to `a` in the current mechanism. It is not clear that
this level of novelty is justified by the value added by this functionality.

### Different names for interfaces

We considered various different names for the interfaces in this proposal.
Differences from the proposed names are highlighted:

-   `Assign`, `AssignWith`, `OpAssign`, `OpAssignWith` (proposed)

    -   Consistently uses `With` suffix to describe the right-hand type.
    -   These names have a direct connection to the lexical operator syntax: the
        interface for `$=` is named as the interface for `$` followed by the
        interface for `=`.
    -   The word order in the name describes the order in which the operations
        are notionally performed: first `Op`, then `Assign`.
    -   Compound assignment interfaces will group alphabetically with the
        corresponding operator, rather than with assignments, which is likely to
        be better for people searching for items in a sorted list.
    -   Matches the choice made in Rust.

-   `Assign`, **_`AssignFrom`_**, `OpAssign`, `OpAssignWith`

    -   This reads more naturally in English. `With` isn't really the right word
        to use in this context, and may be confusing.
    -   Violates the consistency of using `...With` for all the parameterized
        operator interfaces.
    -   Given how common this interface is expected to be compared to the rest,
        the inconsistency of using `AssignFrom` might be acceptable, but for now
        we will use `AssignWith`. If there are sustained concerns with this name
        (if we don't "get used to it"), we should reconsider.

-   `Assign`, `AssignWith`, **_`AssignOp`_**, **_`AssignOpWith`_**

    -   Consistently uses `With` suffix to describe the right-hand type.
    -   The name `AssignOpWith(U)` decomposes as `Assign` + `OpWith(U)` in a way
        that describes the two operations being performed.
    -   When written as function calls, the behavior is `Assign(x, Op(x, y))`,
        which again uses the `AssignOp` word order.

-   `Assign`, **_`AssignGiven`_**, `OpAssign`, **_`OpAssignGiven`_**

    -   `Given` might be a less surprising word for simple assignment than
        `With`, but is still not ideal.
    -   This choice seems slightly worse in most existing cases that use `With`,
        and we didn't consider the benefit of `AssignGiven` over `AssignWith` to
        be sufficient to justify that cost.

-   `Assign`, `AssignWith` or **_`AssignFrom`_**, **_`InPlaceOp`_**,
    **_`InPlaceOpWith`_**

    -   Might make `AssignFrom` more viable by making `Assign` / `AssignFrom` no
        longer parallel `InPlaceOp` / `InPlaceOpWith` so closely.
    -   Might better match how these operations are described in everyday
        parlance.

Overall, the proposed set of names seem like the best choice, despite some of
the names not reading completely naturally. The English readability concern is
probably not much worse than for `LeftShiftWith`, where we already decided that
consistency was more important.
