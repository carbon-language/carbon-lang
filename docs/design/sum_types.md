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
named cases, called _alternatives_. A value of a sum type notionally consists of
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
which holds a single `i32` value (the parameter name `value` has no effect other
than documentation), and `None`, which is empty. Choice types can also be
parameterized, [like class types](generics/details.md#parameterized-types):

```carbon
choice Optional(T:! type) {
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

It does so by implementing the `Match` interface, which is defined as follows:

```carbon
interface Match {
  interface BaseContinuation {
    let ReturnType:! type;
  }

  let template Continuation:! type;
  fn Op[me: Self, C:! Continuation](continuation: C*)
    -> C.(MatchContinuation.ReturnType);
}
```

`Continuation` must itself be an interface that extends
`Match.BaseContinuation`, and its definition specifies the set of possible
alternatives: each alternative is represented as a method of that interface.
When compiling a proper pattern (or set of patterns that includes a proper
pattern, as with the cases of a `match`) whose type is a sum type, the compiler
generates an implementation of `Continuation` and passes it to `Match.Op`. The
sum type's implementation of `Match.Op` is responsible for determining which
alternative is present and what its parameters are, and calling the
corresponding method of `continuation` with those parameters. The `Match.Op`
implementation is required to call exactly one such method exactly once before
returning. The compiler populates the `Continuation` method bodies with whatever
code should be executed when the corresponding alternatives match.

**TODO:** if Carbon has explicit support for tail calls, we should probably
require that `Match.Op` invoke the continuation as a tail call.

For example, here's how `Optional` can be defined as a class:

```carbon
class Optional(T:! type) {
  // Factory functions
  fn Some(value: T) -> Self;
  let None:! Self;

  private var has_value: bool;
  private var value: T;

  external impl as Match {
    interface Continuation {
      extends Match.BaseContinuation;
      fn Some[addr me: Self*](value: T) -> ReturnType;
      fn None[addr me: Self*]() -> ReturnType;
    }

    fn Op[me: Self, C:! Continuation](continuation: C*) -> C.ReturnType {
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

And here's how the compiler-generated implementation of
`Optional.(Match.Continuation)` for the `match` statement shown earlier might
look, if it were written in Carbon:

```carbon
class __MatchStatementImpl {
  impl as Match(Optional.MatchContinuation) where .ReturnType = () {
    fn Some(the_value: i32) {
      Print(the_value);
    }
    fn None() {
      Print("None");
    }
  }
}

my_opt.(Match.Op)({} as __MatchStatementImpl);
```

(The name `__MatchStatementImpl` is a placeholder for illustration purposes; the
actual generated class will be anonymous.)

The mechanism described above for proper patterns may also be used for
expression patterns if they have the form of an alternative pattern. An
expression pattern of type `T` has the form of an alternative pattern if `T`
implements `Match`, and the expression consists of an optional expression that
names `T`, followed by a designator that names a method of
`T.(Match.Continuation)`, optionally followed by a tuple expression. If an
expression pattern has that form, it may be matched using the mechanism above,
as if it were a proper pattern, rather than by evaluating the expression and
comparing it to the scrutinee using `==`. Both possible implementations must be
well-formed (and this is enforced by the compiler), but it is unspecified which
implementation is used to generate code.

As a result, it is **strongly** recommended that user-defined sum types ensure
that for every alternative there is a factory function or constant member with
the same name and parameter list, such that pattern-matching on the result will
correctly reproduce the arguments to the factory function. For example, the
definition of `Optional` above satisfies this requirement, because for any
regular type `T`, the expression `Optional(T).None` evaluates to a value that
matches the pattern `Optional(T).None` (under both possible matching
mechanisms), and for any `x` of type `T`, the expression `Optional(T).Some(x)`
evaluates to a value that matches the pattern `Optional(T).Some(y: T)` and binds
`y` to a value that's equal to `x`. Expression patterns involving a sum type
that doesn't meet this requirement will fail to compile, or have behavior that
observably changes depending on the compiler's implementation choices.

Another corollary of this rule is that if an alternative takes no arguments, its
pattern syntax is the same as its expression syntax. For example,
`case Optional(i32).None() => ...` is not well-formed, because
`Optional(i32).None()` has the form of an alternative pattern, but the
implementation in terms of `==` is not well-formed because
`Optional(i32).None()` is not a well-formed expression. If we had defined
`Optional.None` as a factory function instead of a constant,
`case Optional(i32).None() => ...` would be well-formed but
`case Optional(i32).None => ...` would not be.

Note that the compiler-generated continuation method bodies are not required to
contain the code in the `case` body (or whatever code is in the scope of the
pattern). For example, they might only store the parameter values and then
return an index that identifies the `case` body to be executed.

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
