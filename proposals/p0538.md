# `return` with no argument

<!--
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

[Pull request](https://github.com/carbon-language/carbon-lang/pull/538)

<!-- toc -->

## Table of contents

-   [Problem](#problem)
    -   [Use cases](#use-cases)
    -   [What's wrong with the C++ rule?](#whats-wrong-with-the-c-rule)
-   [Background](#background)
-   [Proposal](#proposal)
-   [Details](#details)
    -   [Relation to upcoming proposals](#relation-to-upcoming-proposals)
-   [Rationale based on Carbon's goals](#rationale-based-on-carbons-goals)
-   [Alternatives considered](#alternatives-considered)
    -   [Retain the C++ rule](#retain-the-c-rule)
    -   [Fully divorce functions and procedures](#fully-divorce-functions-and-procedures)

<!-- tocstop -->

## Problem

### Use cases

We wish to support the following use cases:

-   Functions that return values. This should be supported for any type for
    which a value can be formed, and such support should be uniform and
    independent of whether the return type is `()`. A return value must always
    be produced.
-   Functions that do not return a value ("procedures"). In such functions, we
    do not need or want to be able to return a value. No return value is needed,
    so if control flow reaches the end of a procedure, it should return to its
    caller.
-   Functions with parameterized return types, that may be either of the above,
    depending on the parameterization. For example, a call wrapper might be
    parameterized by the type of its callee, and might return the same type that
    its callee returns, or a type computed based on that type.

### What's wrong with the C++ rule?

C++ treats `void` as a special case in a number of ways. We want to minimize the
impact of this special-case treatment for the corresponding Carbon types. One
way that this special treatment is visible in C++ is that functions with the
return type `void` obey different rules: the operand of `return` becomes
optional (but is still permitted), and reaching the end of the function becomes
equivalent to `return;`. In a function template, this can even happen invisibly,
with some instantiations having an implicit `return;` and others not.

This interferes with the ability to reason about programs. Consider:

```
template<typename T> auto f() {
  if (T::cond())
    return T::run();
}
```

Here, it is possible for control flow to reach the end of the function. However,
a compiler can't warn on this without false positives, because it's possible
that `T::run()` has type `void`, in which case this function has an implicit
`return;` added before its `}`, and indeed, it might be the case that `T::run()`
always has return type `void` for any `T` where `T::cond()` returns `false`.

## Background

See the following issues:

-   [Proposal: function declaration syntax](https://github.com/carbon-language/carbon-lang/pull/438)
-   [Proposal: `return` statements](https://github.com/carbon-language/carbon-lang/pull/415)
-   [Leads question: what is the relationship between `Void` and `()`?](https://github.com/carbon-language/carbon-lang/issues/443)
-   [Leads question: should we allow `return;` in functions with a `Void` return type?](https://github.com/carbon-language/carbon-lang/issues/518)

## Proposal

Instead of applying special-case rules based on whether the return type of a
function is `()`, we apply special-case rules based on whether a return type is
provided.

## Details

A function with no declared return type is a _procedure_. The return type of a
procedure is implicitly `()`, and a procedure always returns the value `()` if
and when it returns. Inside a procedure, `return` must have no argument, and if
control flow reaches the end of a procedure, the behavior is as if `return;` is
executed.

```
// F is a procedure.
fn F() {
  if (cond) {
    return;
  }
  if (cond2) {
    // Error: cannot return a value from a procedure.
    return F();
  }

  // Implicitly `return;` here.
}
```

A function with a declared return type is treated uniformly regardless of
whether that return type happens to be `()`. Every `return` statement must
return a value. There is no implicit `return` at the end of the function, and
instead a compile error is produced if control flow can reach the end of the
function, even if the return type is `()` -- or any other unit type.

```
fn G() -> () {
  if (cond) {
    // OK, F() returns ().
    return F();
  }
  if (cond2) {
    // Error: return without value in value-returning function.
    return;
  }

  // Error: control flow can reach end of value-returning function.
}
```

From the caller's perspective, there is no difference between a function
declared as

```
fn DoTheThing() -> ();
```

and a function declared as

```
fn DoTheThing();
```

As a result, the choice to include or omit the `-> ()` in the definition is an
implementation detail, and the syntax used in a forward declaration is not
required to match that used in a definition. The use of `-> ()` in idiomatic
Carbon code is expected to be rare, but it is permitted for uniformity, and in
case there is a reason to desire the value-returning-function rules or to
emphasize to the reader that `()` is the return type or similar.

### Relation to upcoming proposals

Issue [#510](https://github.com/carbon-language/carbon-lang/issues/510) asks
whether we should support named return variables:

```
fn F() -> var ReturnType: x {
  // initialize x
  return;
}
```

If we do, functions using that syntax should follow the rules for procedures in
this proposal, including the implicit `return;` if control reaches the end of
the function. In particular,

```
fn F() { ... }
```

would be exactly equivalent to

```
fn F() -> var (): _ = () { ... }
```

## Rationale based on Carbon's goals

-   **Software and language evolution**
    -   This proposal may decrease the number of places in which the return type
        of a procedure is used, by discouraging the use of
        `return Procedure();`. This in turn may make it easier to change a
        return type from `()` to something else, but this proposal by itself is
        insufficient to ensure that is always possible.
-   **Code that is easy to read, understand, and write**
    -   The conceptual integrity of the Carbon language is improved by making
        the same syntax result in the same semantics, regardless of whether a
        type happens to be `()` or not, and symmetrically by using different
        syntax for different semantics.
    -   The readability of Carbon code is improved and a source of surprise is
        eliminated by removing the possibility of `return F();` being mixed with
        `return;` in the same function.
-   **Practical safety guarantees and testing mechanisms**
    -   By making the presence or absence of an implicit `return` in a function
        be determined based on syntax alone, we permit checks for missing
        `return` statements to be provided in the definition of a template or
        generic, without needing to know the arguments. This is important for
        generics in particular, because we do not want monomorphization to be
        able to fail and because we do not in general guarantee that
        monomorphization will be performed.
-   **Fast and scalable development**
    -   The correct syntax for a `return` statement can be detected while
        parsing, using only syntactic information rather than contextual,
        semantic information. In practice, we will likely parse both kinds of
        `return` statement in all functions and check the return type from a
        context that has the semantic information, but the ability to do these
        checks syntactically may be useful for simple tools and editor
        integration.
-   **Interoperability with and migration from existing C++ code**
    -   This proposal rejects some constructs that would be valid in C++:
        ```
        return F();
        ```
        in a function with `void` return type would no longer be valid in a
        corresponding Carbon function with no specified return type, and would
        need to be translated into
        ```
        F();
        return;
        ```
        (possibly with braces added). However, the fact that this construct is
        valid in C++ is surprising to many, and the constructs that would be
        idiomatic in C++ are still valid under these rules.

## Alternatives considered

### Retain the C++ rule

The advantages of this approach compared to maintaining the C++ rule are
discussed above. The advantage of maintaining the C++ rule would be that Carbon
is more closely aligned with C++. However, the removed functionality --
specifically, the ability to return an expression of type `void` from a `void`
returning function -- is still available, albeit with a more verbose syntax, and
the existence of that functionality in C++ is a source of surprise to C++
programmers.

### Fully divorce functions and procedures

We could treat the choice of function with `()` return type versus procedure as
being part of the interface rather than being an implementation detail.

```
// F is a procedure.
fn F();
// F is a function returning ().
fn G() -> ();

// ...

// Error, procedure redeclared as a function.
fn F() -> () {
  return ();
}

// Error, function redeclared as a procedure.
fn G() {
}
```

Then, we could disallow any use of a procedure call in a context that depends on
its return value, treating a procedure call as a statement rather than as an
expression that can be used as a subexpression or an operand of an operator.

```
fn Func() -> ();
fn Proc();
// OK, x is of type ().
auto x = Func();
// Error, Proc is a procedure.
auto y = Proc();
```

Advantages:

-   Removes all special treatment of `()` in this context. Procedures no longer
    need to say that their return type is implicitly `()` nor that they
    implicitly return `()`.
-   Adding a return type to a procedure -- converting it to a function -- would
    be a non-breaking change.
    -   But we don't have evidence that this is a common problem.
-   Prevents a source of programming error where a returned `()` value is stored
    and used.
    -   But it's not clear that this would be a frequent error, and it would
        likely be caught in other ways due to the limited API of `()`.

Disadvantages:

-   Having distinct notions of function versus procedure would be a surprise for
    those coming from C++.
-   Supporting `auto x = F();` regardless of whether `F` is a function or
    procedure may be important for generic code.
-   When migrating C++ code to Carbon, this makes the choice of function versus
    procedure load-bearing, as there may be uses that depend on a return value
    existing, for example in templates.
