# Conditional expressions

<!--
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

[Pull request](https://github.com/carbon-language/carbon-lang/pull/911)

<!-- toc -->

## Table of contents

-   [Problem](#problem)
-   [Background](#background)
-   [Proposal](#proposal)
-   [Details](#details)
    -   [Chaining](#chaining)
    -   [Precedence and ambiguity](#precedence-and-ambiguity)
-   [Rationale based on Carbon's goals](#rationale-based-on-carbons-goals)
-   [Alternatives considered](#alternatives-considered)
    -   [No conditional expression](#no-conditional-expression)
    -   [Use C syntax](#use-c-syntax)
    -   [No `then`](#no-then)
    -   [Require parentheses around the condition](#require-parentheses-around-the-condition)
    -   [Never require enclosing parentheses](#never-require-enclosing-parentheses)
    -   [Variable-precedence `if`](#variable-precedence-if)
    -   [Implicit conversions in both directions](#implicit-conversions-in-both-directions)
    -   [Support lvalue conditionals](#support-lvalue-conditionals)
-   [Future work](#future-work)
    -   [Too many user-facing interfaces](#too-many-user-facing-interfaces)
    -   [Incompatible `CommonType` implementations diagnosed late](#incompatible-commontype-implementations-diagnosed-late)
    -   [`impl` ordering depends on operand order](#impl-ordering-depends-on-operand-order)

<!-- tocstop -->

## Problem

Programs need to be able to select between multiple different paths of execution
and multiple different values. In a rich expression language, developers expect
to be able to do this within a subexpression of some overall expression.

## Background

C-family languages provide a `cond ? value1 : value2` operator.

-   This operator has confusing syntax, because both `cond` and `value2` are
    undelimited, and it's often unclear to developers how much of the adjacent
    expressions are part of the conditional expression. For example:
    ```
    int n = has_thing1 && cond ? has_thing2 : has_thing3 && has_thing4;
    ```
    is parsed as
    ```
    int n = (has_thing1 && cond) ? has_thing2 : (has_thing3 && has_thing4);
    ```
    Also, `value1` and `value2` are parsed with different rules:
    ```
    cond ? f(), g() : h(), i();
    ```
    is parsed as
    ```
    (cond ? f(), g() : h()), i();
    ```
-   In C++, this operator has confusing semantics, due to having a complicated
    set of rules governing how the target type is determined.
-   Despite the complications of the rules, the result type of `?:` is not
    customizable. Instead, C++ invented a `std::common_type` trait that models
    what the result of `?:` should have been.

Rust allows most statements to be used as expressions, with `if` statements
being an important case of this: `Use(if cond { v1 } else { v2 })`.

-   This has a number of behaviors that would be surprising to developers coming
    from C++ and C, such as a final `;` in a `{...}` making a semantic
    difference.
-   The expression semantics leak into the statement semantics. For example,
    Rust rejects:

    ```
    fn f() {}
    fn g() -> i32 {}

    fn main() {
      if true { f() } else { g() };
      return;
    }
    ```

    ... because the two arms of the `if` don't have the same type.

-   We have already
    [decided](https://github.com/carbon-language/carbon-lang/issues/430) that we
    do not want Carbon to treat statements such as `if` as being expressions
    without some kind of syntactic distinction.

## Proposal

Provide a conditional expression with the syntax:

> ```
> if cond then value1 else value2
> ```

`then` is a new keyword introduced for this purpose.

## Details

### Chaining

This syntax can be chained like `if` statements:

```
Print(if guess < value
      then "Too low!"
      else if guess > value
      then "Too high!"
      else "Correct!")
```

Unlike with `if` statements, this doesn't require a special rule.

### Precedence and ambiguity

An `if` expression can be used as a top-level expression, or within parentheses
or a comma-separated list such as a function call. They have low precedence, so
cannot be used as the operand of any operator, with the exception of assignment
(if assignment is treated as an operator), but they can appear in other contexts
where an arbitrary expression is permitted, for example as the operand of
`return`, the initializer of a variable, or even as the condition of another
`if` expression or `if` statement.

```
// Error, can't use `if` here.
var v: i32 = 1 * if cond then 2 else 3 + 4;
```

`value2` extends as far to the right as possible:

```
var v: i32 = if cond then 2 else 3 + 4;
```

is the same as

```
var v: i32 = if cond then 2 else (3 + 4);
```

not

```
var v: i32 = (if cond then 2 else 3) + 4;
```

The intent is that an `if` expression is used to produce a value, not only for
its side-effects. If only the side-effects are desired, an `if` statement should
be used instead. Because `value2` extends as far to the right as possible, if an
`if` expression appeared at the start of a statement, its value could never be
used:

```
if cond then value1 else value2;
```

For this reason and to avoid the need for lookahead or disambiguation, an `if`
keyword appearing at the start of a statement is always interpreted as beginning
an `if` statement and never as beginning an `if` expression.

## Rationale based on Carbon's goals

-   [Language tools and ecosystem](/docs/project/goals.md#language-tools-and-ecosystem)
    -   The `if ... then ... else` syntax should be easier to format
        automatically in an unsurprising way than a `?:` syntax because it is
        clear that the `then` and `else` keywords should be wrapped to the start
        of a new line when wrapping the overall conditional expression.
-   [Code that is easy to read, understand, and write](/docs/project/goals.md#code-that-is-easy-to-read-understand-and-write)
    -   Including such an expression is expected to improve ergonomics.
    -   An explicit delimiter for the start of the condition expression makes it
        easier to read, correctly write, and understand the precedence of
        conditional expressions.
    -   Making the `value2` portion as long as possible gives a simple rule that
        it seems feasible for every Carbon developer to remember. This rule is
        expected to be unsurprising both due to using the same rule for `value1`
        and `value2`, and because it means that `if` consistently behaves like a
        very low precedence prefix operator.
    -   The use of an explicit `if` keyword for flow control makes the
        distinction between flow control and linear computation clearer.
    -   The readability of a multi-line `if` expression is improved by having a
        `then` and `else` keyword of the same length
-   [Interoperability with and migration from existing C++ code](/docs/project/goals.md#interoperability-with-and-migration-from-existing-c-code)
    -   Migration is improved by providing an operator set that largely matches
        the C++ operator set.

## Alternatives considered

### No conditional expression

We could provide no conditional expression, and instead ask people to use a
different mechanism to achieve this functionality. Some options include:

-   Use of an `if` statement:
    ```
    var v: Result;
    if (cond) {
      v = value1;
    } else {
      v = value2;
    }
    Use(v);
    ```
-   A function call syntax:
    ```
    Use(cond.Select(value1, value2));
    ```
    or, with short-circuiting and lambdas:
    ```
    Use(cond.LazySelect($(value1), $(value2)));
    ```
-   An `if` statement in a lambda:
    ```
    Use(${ if (cond) { return value1; } else { return value2; } });
    ```

The above assumes a placeholder `$(...)` syntax for a single-expression lambda,
and a `${...}` syntax for a lambda with statements as its body.

Advantages:

-   No new dedicated syntax.

Disadvantages:

-   Conditional expressions are commonly used, commonly desired, and Carbon
    developers -- especially those coming from C++ and C -- will be disappointed
    by their absence.
-   Readability and ergonomics will be harmed by making this common operation
    more verbose, even if an idiom is established.

### Use C syntax

We could use the C `cond ? value1 : value2` syntax.

Advantages:

-   Familiar to developers coming from C++ and C.

Disadvantages:

-   These operators have serious precedence problems in C++ and C. We could
    address those by making more cases ambiguous, at the cost of harming
    familiarity and requiring parentheses in more cases.
-   The `:` token is already in use in name binding; using it as part of a
    conditional expression would be confusing.
-   The `?` token is likely to be desirable for use in optional unwrapping and
    error handling.

### No `then`

We could use

> ```
> if (cond) value1 else value2
> ```

instead of

> ```
> if cond then value1 else value2
> ```

Note that we cannot avoid parentheses in this formulation without risking
syntactic ambiguities.

Advantages:

-   Looks more like an `if` statement, albeit one with unbraced operands.
-   Slightly shorter.
-   Better line-wrapping for chained `if` expressions:
    ```
    Print(if (guess < value)
            "Too low!"
          else if (guess > value)
            "Too high!"
          else
            "Correct!")
    ```
    may be more readable than
    ```
    Print(if guess < value
          then "Too low!"
          else if guess > value
          then "Too high!"
          else "Correct!")
    ```
    or
    ```
    Print(if guess < value
            then "Too low!"
            else if guess > value
              then "Too high!"
              else "Correct!")
    ```

Disadvantages:

-   Potentially worse line wrapping. The `else` would presumably be wrapped onto
    a line by itself, wasting vertical space, whereas `then` and `else` when
    paired can both comfortably precede their values on the same line; consider
    ```
    F(if (cond)
        value1
      else
        value2)
    ```
    occupies more space than
    ```
    F(if cond
      then value1
      else value2)
    ```
-   May create confusion between `if` statements and `if` expressions by
    resembling an `if` statement but not matching the semantics.
-   May cause evolutionary problems due to syntactic conflict if we ever make
    the braces or parentheses in `if` statements optional.
-   Requires parentheses, and hence additional presses of "Shift" on US
    keyboards, making it slightly harder to type.

### Require parentheses around the condition

We could use:

> ```
> if (cond) then value1 else value2
> ```

However, it's not clear that there is value in requiring both parentheses and a
new keyword. It also seems jarring that this so closely resembles an `if`
statement but adds a `then` keyword that the `if` statement lacks.

### Never require enclosing parentheses

We could allow an `if` expression to appear anywhere a parenthesized expression
can appear, and retain the rule that `value2` extends as far to the right as
possible.

Advantages:

-   Removes extra ceremony from a construct that is already more verbose than
    the corresponding `?:` construct in C++.
-   The requirement to include parentheses may be irritating in cases where
    there is no other possible interpretation, such as
    `1 + (if cond then 2 else 3)`.

Disadvantages:

-   Visually ambiguous where `value2` ends in some cases, and violates
    precedence rules.
-   Hard for a simple yacc/bison parser to handle, due to ambiguity of `if` at
    the start of a statement and ambiguity when parsing `value2`.

### Variable-precedence `if`

We could allow an `if` expression to appear anywhere a parenthesized expression
can appear, and parse the `value1` and `value2` as if they appeared in place of
the `if` expression:

```carbon
var n: i32 = 1 + if cond then 2 * 3 else 4 * 5 + 6;
// ... is interpreted as ...
var n: i32 = (1 + (if cond then (2 * 3) else (4 * 5))) + 6;

// Error: expected `else` but found `+ 4`.
var m: i32 = 1 + if cond then 2 * 3 + 4 else 5 + 6;
```

Advantages:

-   Same as previous option.

Disadvantages:

-   Confusing to readers, because it's not clear locally where the expression
    after `else` ends, and discovering this requires looking backwards to before
    the `if`.
-   Hard for a simple yacc/bison parser to handle, due to needing at least one
    production for an `if` statement for each precedence level. Also, those
    productions will result in grammar ambiguities that will need to be
    resolved.

### Implicit conversions in both directions

Suppose we have two types where implicit conversions in both directions are
possible:

```carbon
class A {}
class B {}
impl A as ImplicitAs(B) { ... }
impl B as ImplicitAs(A) { ... }
```

By default, an expression `if cond then {} as A else {} as B` would be
ambiguous. If the author of `A` or `B` wishes to change this behavior:

-   If the common type should be `A`, then `impl A as CommonTypeWith(B)` must be
    provided specifying the common type is `A`.
-   If the common type should be `B`, then `impl B as CommonTypeWith(A)` must be
    provided specifying the common type is `B`.
-   If the common type should be something else, then both `impl`s need to be
    provided:
    ```
    impl A as CommonTypeWith(B) { let Result:! Type = C; }
    impl B as CommonTypeWith(A) { let Result:! Type = C; }
    ```

We could change the rules so instead, in any of the above cases, implementing
either `A as CommonTypeWith(B)` or `B as CommonTypeWith(A)` would suffice.

Advantages:

-   Simplifies the user experience in this case.

Disadvantages:

-   Introduces non-uniformity: the blanket `impl` of `CommonTypeWith` in terms
    of `ImplicitAs` would get this special treatment, but other blanket `impl`s
    would not.
-   Introduces complexity, which might not be fully hidden from users. At
    minimum, we would need to explain that `ImplicitAs` is treated specially
    here.
-   The case in which two `impl`s are required is a corner case. It's somewhat
    uncommon for implicit conversions to be possible in both directions between
    two types. In those cases, it's more uncommon for there to be a clear best
    "common type". And even then, most of the time the common type will be one
    of the two types being unified.

From a more abstract perspective: the process of finding a common type involves
asking each type to implicitly convert to the destination type that it thinks is
best, and then failing if both sides didn't convert to the same type. If `A`
implicitly converts to `B` and the other way around, then both sides of this
process should be overridden in order to get both types to implicitly convert to
`C` instead.

### Support lvalue conditionals

Carbon doesn't formally have a notion of lvalue or rvalue yet; this notion is
expected to be added by
[#821: Values, variables, pointers, and references](https://github.com/carbon-language/carbon-lang/pull/821).
In any case, we certainly intend to distinguish between expressions that
represent values and expressions that represent locations where values could
appear. We therefore need to decide whether a conditional expression can ever be
in the latter category. For example:

```carbon
var a: String;
var b: String;
var c: bool;
// Valid?
(if c then a else b) = "Hello";
```

We could permit this, as C++ does. For example, we could say:

> If both _value1_ and _value2_ are lvalues then
> `if cond then value1 else value2` is rewritten to
> `*(if cond then &value1 else &value2)` if those pointer types have a common
> type.

The other reason we might want to consider this alternative is performance. In
C++, this code avoids making a `std::string` copy:

```c++
std::string a;
std::string b;
std::string c;
bool cond;
// ...
bool equal = c == (cond ? a : b);
```

... by treating the conditional expression as an lvalue of type `std::string`
rather than as a prvalue. However, in Carbon, following #821, we would expect
that the equivalent of a prvalue of type `std::string` would not necessarily
imply that a copy is made. Rather, Carbon's equivalent of prvalues would
represent either a set of instructions to initialize a value (as in C++), or the
location of some existing value that we are temporarily "borrowing".

With that in mind:

Advantages:

-   More similar to C++.
-   Permits certain operations that have an obvious intended meaning, such as
    assignment to a conditional.

Disadvantages:

-   Modification through an lvalue conditional is seldom used in C++, indicating
    that this is not an important feature. The other benefits of a conditional
    producing an lvalue are expected to be obtained by #821.
-   Mutable inputs to operations ("out parameters") in Carbon are expected to be
    expressed as pointers under #821, so there will be a `&` somewhere anyway;
    given the choice between an lvalue conditional:
    ```
    F(&(if cond then a else b));
    ```
    and an rvalue-only conditional:
    ```
    F(if cond then &a else &b);
    ```
    the latter option would likely be preferred even if the former were
    available.
-   This would create an inconsistency in behavior, which would be particularly
    visible in a generic when determining what constraints are necessary to
    type-check an `if` expression -- the constraints would depend not only on
    operand types, but also on value category, and may result in a hard to
    express constraint such as "either `T*` and `U*` have a common type or `T`
    and `U` have a common type".
-   Certain kinds of lvalue conditional expression have turned out to be hard to
    implement in C++, such as a conditional involving bit-field lvalues. We can
    entirely avoid that class of implementation problems by treating conditional
    expressions as rvalues.

This should be revisited if the direction in #821 changes substantially from the
assumptions described above.

## Future work

There are some known issues with the way that the extensibility mechanism works
in this proposal. It is hoped that extensions to Carbon's generics mechanism
will provide simple ways to resolve these issues. This design should be
revisited once those mechanisms are available.

### Too many user-facing interfaces

We provide both `CommonTypeWith`, as an extension point, and `CommonType`, as a
constraint. It would be preferable to provide only a single name that functions
both as the extension point and as a constraint, but we don't have a good way to
automatically make `impl`s symmetric and avoid `impl` cycles if we use only one
interface.

### Incompatible `CommonType` implementations diagnosed late

Example:

```
class A {}
class B {}

impl A as CommonTypeWith(B) where .Result = A {}
impl B as CommonTypeWith(A) where .Result = B {}

fn F(a: A, b: B) -> auto { return if true then a else b; }
```

The definition of function `F` is rejected, because `A` and `B` have no
(consistent) common type. It would be preferable to reject the `impl`
definitions.

### `impl` ordering depends on operand order

Example:

```
class A(T:! Type) {}
class B(T:! Type) {}

interface Fungible {}
impl A(T:! Type) as Fungible {}
impl B(T:! Type) as Fungible {}

// #1
impl A(T:! Type) as CommonTypeWith(U:! Fungible) where .Result = A(T) {}
// #2
impl B(T:! Type) as CommonTypeWith(A(T)) where .Result = T {}

fn F(a: A(i32), b: B(i32)) -> auto { return if true then a else b; }
```

Here, reversed #2 is a better match than #1, because it matches both `A(?)` and
`B(?)`, so #2 should be consider the best-matching `impl`. However, we never
compare reversed #2 against non-reversed #1. Instead, we look for:

1.  `impl A(i32) as SymmetricCommonTypeWith(B(i32))`, which selects #1 as being
    better than the blanket `impl` that reverses operand order.
2.  `impl B(i32) as SymmetricCommonTypeWith(A(i32))`, which selects #2 as being
    better than the blanket `impl` that reverses operand order.

So we decide that the `if` in `F` is ambiguous, even though there is a unique
best `CommonTypeWith` match. If either #1 or #2 is written with the operand
order reversed, then `F` would be accepted.
