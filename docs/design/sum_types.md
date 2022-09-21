# Sum types

<!--
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

<!-- toc -->

## Table of contents

-   [Overview](#overview)
-   [`choice` declarations](#choice-declarations)
-   [User-defined sum types](#user-defined-sum-types)
-   [Alternatives considered](#alternatives-considered)
-   [References](#references)

<!-- tocstop -->

## Overview

In Carbon, a _sum type_ is a type whose values are grouped into several distinct
named cases, called _alternatives_. A value of a sum type logically consists of
a _discriminator_ tag that identifies which alternative is present, together
with that alternative's value if it has one. Sum types are typically handled
with pattern matching.

## `choice` declarations

The `choice` keyword is used to declare a sum type by specifying its interface,
leaving the implementation to the compiler. A `choice` declaration consists of
the `choice` keyword followed by the name of the type, and then a list of
alternatives inside curly braces. An alternative declaration consists of the
alternative name, optionally followed by a parameter list in parentheses. If
present, the parameter list has the same syntax as in a
[function declaration](README.md#functions). For example:

```carbon
choice OptionalI32 {
  Some(value: i32),
  None
}
```

This declares a sum type named `OptionalI32` with two alternatives: `Some`,
which holds a single `i32` value, and `None`, which is empty. Choice types can
also be parameterized,
[like class types](generics/details.md#parameterized-types):

```carbon
choice Optional(T:! Type) {
  Some(value: T),
  None
}
```

A value of a function-like alternative is specified by "calling" it like a
function, and a value of an empty alternative like `None` is specified by naming
it:

```carbon
var my_opt: Optional(i32) = Optional(i32).None;
my_opt = Optional(i32).Some(42);
```

The value of a choice type can be inspected using a `match` statement:

```carbon
match (my_opt) {
  case .Some(the_value: i32) => {
    Print(the_value);
  }
  case .None => {
    Print("None");
  }
}
```

## User-defined sum types

`choice` declarations are a convenience shorthand for common use cases, but they
have limited flexibility. There is no way to control the representation of a
`choice` type, or define methods or other members for it (although you can
extend it to implement interfaces, using an
[`external impl`](generics/overview.md#implementing-interfaces) or
[`adapter`](generics/overview.md#adapting-types)). However, a `class` type can
be extended to behave like a sum type. This is much more verbose than a `choice`
declaration, but gives the author full control over the representation and class
members.

The ability to create instances of the sum type can be straightforwardly
emulated with factory functions and static constants, and the internal storage
layout will presumably involve untagged unions or some other low-level storage
primitive which hasn't been designed yet, but the key to defining a sum type's
interface is enabling it to support pattern matching. To do that, the sum type
has to specify two things:

-   The set of all possible alternatives, including their names and parameter
    types, so that the compiler can typecheck the `match` body, identify any
    unreachable `case`s, and determine whether any `case`s are missing.
-   The algorithm that, given a value of the sum type, determines which
    alternative is present, and specifies the values of its parameters.

Here's how that would look if `Optional` were defined as a class:

```carbon
class Optional(T:! Type) {
  // Factory functions
  fn Some(value: T) -> Self;
  let None:! Self;

  private var has_value: bool;
  private var value: T;

  interface MatchContinuation {
    let template ReturnType:! Type;
    fn Some[addr me: Self*](value: T) -> ReturnType;
    fn None[addr me: Self*]() -> ReturnType;
  }

  external impl as Match(MatchContinuation) {
    fn Op[me: Self, Continuation:! MatchContinuation](
        continuation: Continuation*) -> Continuation.ReturnType {
      if (me.has_value) {
        return continuation->Some(me.value);
      } else {
        return continuation->None();
      }
    }
  }

  // Operations like destruction, copying, assignment, and comparison are
  // omitted for brevity.
}
```

In this code, `Optional` makes itself available for use in pattern matching by
declaring that it implements the `Match` interface. `Match` takes an interface
argument, called the _continuation interface_, which specifies the set of
possible alternatives by declaring a method for each one. In this case, we pass
`Optional.MatchContinuation` as the continuation interface.

When compiling a `match` statement, the compiler checks that the type being
matched implements `Match(C)` for some continuation interface `C`. Then, it
notionally transforms the `match` body into a class that implements `C`, with
one method for each `case`, and then passes that to `Match.Op` on the object
being matched. For example, the `match` statement shown earlier might be
transformed into:

```carbon
class __MatchStatementImpl {
  fn Make() -> Self { return {}; }

  impl as Match(Optional.MatchContinuation) where .ReturnType = () {
    fn Some(the_value: i32) {
      Print(the_value);
    }
    fn None() {
      Print("None");
    }
  }
}

my_opt.(Match.Op)(__MatchStatementImpl.Make());
```

Thus, a `match` statement works by invoking the sum type's `Match.Op` method,
which is responsible for determining which alternative the sum object
represents, and then invoking the compiler-supplied continuation that
corresponds to that alternative. In order for this scheme to work, `Match.Op` is
required to invoke exactly one method of `MatchContinuation`, and to do so
exactly once.

Notice that the names `Some` and `None` are defined twice, once as factory
functions/constant members of `Optional` and once as methods of
`MatchContinuation`, with the same parameter types in each case. The two
effectively act as inverses of each other: the `Optional` members are used to
create `Optional` values, and the `MatchContinuation` methods are used to report
the parameter values that would create a given `Optional`. This mirroring
between expression and pattern syntax is ultimately a design choice by the type
author; there is no language-level requirement that the alternatives correspond
to the factory functions, but it is **strongly** recommended.

## Alternatives considered

-   [Providing `choice` types only](/proposals/p0157.md#choice-types-only), with
    no support for user-defined sum types.
-   [Indexing alternatives by type](/proposals/p0157.md#indexing-by-type)
    instead of by name.
-   Implementing user-defined sum types in terms of
    [`choice` type proxies](/proposals/p0157.md#pattern-matching-proxies) rather
    than callbacks.
-   Implementing user-defined sum types in terms of invertible
    [pattern functions](/proposals/p0157.md#pattern-functions).

## References

-   Proposal
    [#157: Design direction for sum types](https://github.com/carbon-language/carbon-lang/pull/157)
