# Lambdas

<!--
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

[Pull request](https://github.com/carbon-language/carbon-lang/pull/3848)

<!-- toc -->

## Table of contents

-   [Abstract](#abstract)
-   [Syntax Overview](#syntax-overview)
    -   [Syntax Defined](#syntax-defined)
-   [Introducer](#introducer)
-   [Positional Parameters](#positional-parameters)
    -   [Positional Parameter Restrictions](#positional-parameter-restrictions)
-   [Function Captures](#function-captures)
    -   [Capture Modes](#capture-modes)
    -   [Default Capture Mode](#default-capture-mode)
-   [Function Fields](#function-fields)
-   [Copy Semantics](#copy-semantics)
-   [Self and Recursion](#self-and-recursion)
-   [Rationale](#rationale)
-   [Alternatives Considered](#alternatives-considered)
    -   [Alternative Considered: Terse vs Elaborated](#alternative-considered-terse-vs-elaborated)
    -   [Alternative Considered: Sigil](#alternative-considered-sigil)
    -   [Alternative Considered: Additional Positional Parameter Restriction](#alternative-considered-additional-positional-parameter-restriction)
    -   [Alternative Considered: Recursive Self](#alternative-considered-recursive-self)
-   [Future Work](#future-work)
    -   [Future Work: Reference Captures](#future-work-reference-captures)

<!-- tocstop -->

## Abstract

This document proposes a path forward to add lambdas to Carbon. It further
proposes augmenting function declarations to create a more continuous syntax
between the two categories of functions. In short, both lambdas and function
declarations will be introduced with the `fn` keyword. The presence of a name
distinguishes a function declaration from a lambda expression, and the rest of
the syntax applies to both kinds. By providing a valid lambda syntax in Carbon,
migration from from C++ to Carbon will be made easier and more idiomatic. In
C++, lambdas are defined at their point of use and are often anonymous, meaning
replacing them solely with function declarations would create an ergonomic
burden compounded by the need for the migration tool to select a name.

Associated discussion docs:

-   [Lambdas Discussion 1](https://docs.google.com/document/d/1rZ9SXL4Voa3z20EQz4UgBMOZg8xc8xzKqA1ufPQdTao/)
-   [Lambdas Discussion 2](https://docs.google.com/document/d/14K_YLjChWyyNv3wv5Mn7uLFHa0JZTc21v_WP8RzC8M4/)
-   [Lambdas Discussion 3](https://docs.google.com/document/d/1VVOlRuPGt8GQpjsygMwH2B7Wd0mBsS3Qif8Ve2yhX_A/)
-   [Lambdas Discussion 4](https://docs.google.com/document/d/1Sevhvjo06Bc6wTigNL1pK-mlF3IXvzmU1lI2X1W9OYA/)

# Background

Refer to the following documentation about lambdas in other languages. What
separates these three and makes them more analegous to Carbon's direction is the
use of "captures" such that the lambda has state, a lifetime, etc.

-   [Lambdas in C++](https://en.cppreference.com/w/cpp/language/lambda)
-   [Closures in Swift](https://docs.swift.org/swift-book/documentation/the-swift-programming-language/closures/)
-   [Closures in Rust](https://doc.rust-lang.org/rust-by-example/fn/closures.html)

## Syntax Overview

**Proposal**: A largely continuous syntax between lambdas and function
declarations.

At a high level, lambdas and function declarations will look like the following.

```
// In a variable:
let lambda: auto = fn => T.Make();
// Equivalent in C++23:
// const auto lambda = [] { return T::Make(); };

// In a function call:
Foo(10, 20, fn => T.Make());
// Equivalent in C++23:
// Foo(10, 20, [] { return T::Make(); });
```

```
// In a variable:
let lambda: auto = fn -> T { return T.Make(); };
// Equivalent in C++23:
// const auto lambda = [] -> T { return T::Make(); };

// In a function call:
PushBack(my_list, fn => T.Make());
// Equivalent in C++23:
// PushBack(my_list, [] { return T::Make(); });
```

```
fn FunctionDeclaration => T.Make();
// Equivalent in C++23:
// auto FunctionDeclaration() { return T.Make(); }
```

```
fn FunctionDeclaration -> T { return T.Make(); }
// Equivalent in C++23:
// auto FunctionDeclaration() -> T { return T::Make(); }
```

There are functions which return an expression, such that the return type is
`auto`.

```
// In a variable:
let lambda: auto = fn => T.Make();
// Equivalent in C++23:
// const auto lambda = [] { return T::Make(); };

// In a function call:
Foo(fn => T.Make());
// Equivalent in C++23:
// Foo([] { return T::Make(); });
```

```
fn FunctionDeclaration => T.Make();
// Equivalent in C++23:
// auto FunctionDeclaration() { return T::Make(); }
```

And there are functions with an explicit return type that provide a body of
statements.

```
// In a variable:
let lambda: auto = fn -> T { return T.Make(); };
// Equivalent in C++23:
// const auto lambda = [] -> T { return T::Make(); };

// In a function call:
Foo(fn -> T { return T.Make(); })
// Equivalent in C++23:
// Foo([] -> T { return T::Make(); });
```

```
fn FunctionDeclaration -> T { return T.Make(); }
// Equivalent in C++23:
// auto FunctionDeclaration() -> T { return T::Make(); }
```

There are even functions that provide a body of statements but no return value.

```
// In a variable:
let lambda: auto = fn { Print(T.Make()); };
// Equivalent in C++23:
// const auto lambda = [] -> void { Print(T::Make()); };

// In a function call:
Foo(fn { Print(T.Make()); });
// Equivalent in C++23:
// Foo([] -> void { Print(T::Make()); });
```

```
fn FunctionDeclaration { Print(T.Make()); }
// Equivalent in C++23:
// auto FunctionDeclaration() -> void { Print(T::Make()); }
```

Functions support [captures](#function-captures), [fields](#function-fields) and
deduced parameters in the square brackets. In addition, `self: Self` or
`addr self: Self*` can be added to the square brackets of function declarations
that exist inside class or interface definitions.

```
fn Foo(x: i32) {
  // In a variable:
  let lambda: auto = fn [var x, var y: i32 = 0] { Print(++x, ++y); };
  // Equivalent in C++23:
  // const auto lambda = [x, y = int32_t{0}] mutable -> void { Print(++x, ++y); };

  // In a function call:
  Foo(fn [var x, var y: i32 = 0] { Print(++x, ++y); });
  // Equivalent in C++23:
  // Foo([x, y = int32_t{0}] mutable -> void { Print(++x, ++y); });

  fn FunctionDeclaration[var x, var y: i32 = 0] { Print(++x, ++y); }
  // Equivalent in C++23:
  // auto FunctionDeclaration = [x, y = int32_t{0}] mutable -> void { Print(++x, ++y); };
}
```

Functions also support so-called
["positional parameters"](#positional-parameters) that are defined at their
point of use using a dollar sign and a non-negative integer. They are implicitly
of type `auto`.

```
fn Foo() {
  let lambda: auto = fn { Print($0); };
  // Equivalent in C++23:
  // auto lambda = [](auto _0, auto...) -> void { Print(_0); };
  // Equivalent in Swift:
  // let lambda = { Print($0) };

  fn FunctionDeclaration { Print($0); }
  // Equivalent in C++23:
  // auto FunctionDeclaration = [](auto _0, auto...) -> void { Print(_0); };
  // Equivalent in Swift:
  // let FunctionDeclaration = { Print($0) };
}
```

Of course, functions can also have named parameters, but a single function can't
have both named and positional parameters.

```
fn Foo() {
  // In a variable:
  let lambda: auto = fn (v: auto) { Print(v); };
  // Equivalent in C++23:
  // const auto lambda = [](v: auto) -> void { Print(v); };

  // In a function call:
  Foo(fn (v: auto) { Print(v); });
  // Equivalent in C++23:
  // Foo([](v: auto) { Print(v); });

  fn FunctionDeclaration(v: auto) { Print(v); }
  // Equivalent in C++23:
  // auto FunctionDeclaration(v: auto) -> void { Print(v); }
}
```

And in additional the option between positional and named parameters, deduced
parameters are always permitted.

```
fn Foo() {
  let lambda: auto = fn [T:! Printable](t: T) { Print(t); };

  fn FunctionDeclaration[T:! Printable](t: T) { Print(t); }
}
```

### Syntax Defined

Function definitions and lambda expressions have one of the following syntactic
forms (where items in square brackets are optional and independent):

`fn` \[_name_\] \[_implicit-parameters_\] \[_tuple-pattern_\] `=>` _expression_
\[`;`\]

`fn` \[_name_\] \[_implicit-parameters_\] \[_tuple-pattern_\] \[`->`
_return-type_\] `{` _statements_ `}`

The first form is a shorthand for the second: "`=>` _expression_ `;`" is
equivalent to "`-> auto { return` _expression_ `; }`".

_implicit-parameters_ consists of square brackets enclosing a optional default
capture mode and any number of explicit captures, function fields, and deduced
parameters, all separated by commas. The default capture mode (if any) must come
first; the other items can appear in any order. If _implicit-parameters_ is
omitted, it is equivalent to `[]`.

The presence of _name_ determines whether this is a function definition or a
lambda expression. The trailing `;` in the first form is required for a function
definition, but is not part of the syntax of a lambda expression.

The presence of _tuple-pattern_ determines whether the function body uses named
or positional parameters.

The presence of "`->` _return-type_" determines whether the function body can
(and must) return a value.

To understand how the syntax between lambdas and function declarations is
reasonably "continuous", refer to this table of syntactic positions and the
following code examples.

| Syntactic Position |                        Syntax Allowed in Given Position (optional, unless otherwise stated)                        |
| :----------------: | :----------------------------------------------------------------------------------------------------------------: |
|         A1         |               Required Returned Expression ([positional parameters](#positional-parameters) allowed)               |
|         A2         |             Required Returned Expression ([positional parameters](#positional-parameters) disallowed)              |
|         B          |                                   [Default Capture Mode](#default-capture-mode)                                    |
|         C          | Explicit [Captures](#function-captures), [Function Fields](#function-fields) and Deduced Parameters (in any order) |
|         D          |                                                Explicit Parameters                                                 |
|         E1         |           Body of Statements (no return value) ([positional parameters](#positional-parameters) allowed)           |
|         E2         |          Body of Statements (with return value) ([positional parameters](#positional-parameters) allowed)          |
|         E3         |         Body of Statements (no return value) ([positional parameters](#positional-parameters) disallowed)          |
|         E4         |        Body of Statements (with return value) ([positional parameters](#positional-parameters) disallowed)         |
|         F          |                                                Required Return Type                                                |
|         G          |                                             Function Declaration Name                                              |

```
// Lambdas (all the following are in an expression context and are
// themselves expressions)

fn => A1

fn [B, C] => A1

fn (D) => A2

fn [B, C](D) => A2

fn { E1; }

fn -> F { E2; }

fn [B, C] { E1; }

fn [B, C] -> F { E2; }

fn (D) { E3; }

fn (D) -> F { E4; }

fn [B, C](D) { E3; }

fn [B, C](D) -> F { E4; }
```

```
// Function Declarations (all the following are allowed as statements in a
// function body or as declarations in other scopes)

fn G => A1;

fn G[B, C] => A1;

fn G(D) => A2;

fn G[B, C](D) => A2;

fn G { E1; }

fn G -> F { E2; }

fn G[B, C] { E1; }

fn G[B, C] -> F { E2; }

fn G(D) { E3; }

fn G(D) -> F { E4; }

fn G[B, C](D) { E3; }

fn G[B, C](D) -> F { E4; }
```

## Introducer

**Proposal**: Introduce with the `fn` keyword to mirror function declarations.
If a statement or declaration begins with `fn`, a name is required and it
becomes a function declaration. Otherwise, if in an expression context, `fn`
introduces a lambda.

```
let lambda1: auto = fn => T.Make();

let lambda2: auto = fn -> T { return T.Make(); };

fn FunctionDeclaration1 => T.Make();

fn FunctionDeclaration2 -> T { return T.Make(); }
```

## Positional Parameters

**Proposal**: Positional parameters, introduced in the body of a function by way
of the dollar sign and a corresponding non-negative parameter position integer
(ex: `$3`), are `auto` parameters to the function in which they are defined.
They can be used in any lambda or function declaration that lacks an explicit
parameter list (parentheses). They are variadic by design, meaning an unbounded
number of arguments can be passed to any function that lacks an explicit
parameter list. Only the parameters that are named in the body will be read
from, meaning the highest named parameter denotes the minimum number of
arguments required by the function. The function body is free to omit
lower-numbered parameters (ex: `fn { Print($10); }`).

This syntax was inpsired by Swift's
[Shorthand Argument Names](https://docs.swift.org/swift-book/documentation/the-swift-programming-language/closures/#Shorthand-Argument-Names).

```
// A lambda that takes two positional parameters being used as a comparator
Sort(my_list, fn => $0.val < $1.val);
// In Swift: { $0.val < $1.val }
```

### Positional Parameter Restrictions

**Proposal**: There are two restrictions applied to functions with positional
parameters. The first restriction is that the definitions of function
declarations must be attached to the declarations. The second restriction is
that positional parameters can only be used in a context where there is exactly
one enclosing function without an explicit parameter list. For example...

```
fn Foo1 {
  fn Bar1 {}  // ❌ Invalid: Foo1 is already using positional parameters
}

fn Foo2 {
  Print($0);
  fn Bar2 {}  // ❌ Invalid: Foo2 is already using positional parameters
}

fn Foo3 {
  fn Bar3 {
    Print($0);  // ❌ Invalid: Foo3 is already using positional parameters
  }
}

fn Foo4() {
  fn Bar4 {
    Print($0);  // ✅ Valid: Foo4 has explicit parameters
  }
}

fn Foo5 {
  fn Bar5() {}  // ✅ Valid: Bar5 has explicit parameters
}

fn Foo6() {
  my_list.Sort(
    fn => $0 < $1  // ✅ Valid: Foo6 has explicit parameters
  );
}
```

## Function Captures

**Proposal**: Function captures in Carbon mirror the non-init captures of C++. A
function capture declaration consists of a capture mode (for `var` captures)
followed by the name of a binding from the enclosing scope, and makes that
identifier available in the inner function body. The lifetime of a capture is
the lifetime of the function in which it exists. For example...

```
fn Foo() {
  let handle: Handle = Handle.Get();
  var thread: Thread = Thread.Make(fn [var handle] { handle.Process(); });
  thread.Join();
}
```

```
fn Foo() {
  let handle: Handle = Handle.Get();
  fn MyThread[handle]() { handle.Process(); }
  var thread: Thread = Thread.Make(MyThread);
  thread.Join();
}
```

### Capture Modes

**Proposal**: `let` and `var` can appear as function captures. They behave as
they would in regular bindings.

To prevent ambiguities, captures can only exist on functions where the
definition is attached to the declaration. This means they are supported on
lambdas (which always exist in an expression context) and they are supported on
function declarations that are immediately defined inside the body of another
function (which is in a statement context), but they are not supported on
forward-declared functions nor are they supported as class members where
`self: Self` is permitted.

Capture modes can be used as
[default capture mode specifiers](#default-capture-mode) or for explicit
captures as shown in the example code below.

```
fn Example {
  var a: i32 = 0;
  var b: i32 = 0;

  let lambda: auto = fn [a, var b] {
    a += 1;  // ❌ Invalid: by-value captures are immutable

    b += 1;  // ✅ Valid: Modifies the captured copy of the by-object capture
  };

  lambda();
}
```

```
fn Example {
  fn Invalid() -> auto {
    var s: String = "Hello world";
    return fn [s]() => s;
  }

  // ❌ Invalid: returned lambda references `s` which is no longer alive
  // when the lambda is invoked.
  Print(Invalid()());
}
```

Note: If a function object F has mutable state, either because it has a
by-object capture or because it has a by-object function field, then a call to F
should require the callee to be a reference expression rather than a value
expression. We need a mutable handle to the function in order to be able to
mutate its mutable state.

### Default Capture Mode

**Proposal**: By default, there is no capturing in functions. The lack of any
square brackets is the same as an empty pair of square brackets. Users can opt
into capturing behavior. This is done either by way of individual explicit
captures, or more succinctly by way of a default capture mode. The default
capture mode roughly mirrors the syntax `[=]` and `[&]` capture modes from C++
by being the first thing to appear in the square brackets.

```
fn Foo1() {
  let handle: Handle = Handle.Get();
  fn MyThread[var]() {
    handle.Process();  // `handle` is captured by-object due to the default capture
                       // mode specifier of `var`
  }
  var thread: Thread = Thread.Make(MyThread);
  thread.Join();
}

fn Foo2() {
  let handle: Handle = Handle.Get();
  fn MyThread[let]() {
    handle.Process();  // `handle` is captured by-value due to the default capture
                       // mode specifier of `let`
  }
  var thread: Thread = Thread.Make(MyThread);
  thread.Join();
}
```

## Function Fields

**Proposal**: Function fields mirror the behavior of init captures in C++. A
function field definition consists of an irrefutable pattern, `=`, and an
initializer. It matches the pattern with the initializer when the function
definition is evaluated. The bindings in the pattern have the same lifetime as
the function, and their scope extends to the end of the function body.

To prevent ambiguities, function fields can only exist on functions where the
definition is attached to the declaration. This means they are supported on
lambdas (which always exist in an expression context) and they are supported on
function declarations that are immediately defined inside the body of another
function (which is in a statement context), but they are not supported on
forward-declared functions nor are they supported as class members where
`self: Self` is permitted.

```
fn Foo() {
  var h1: Handle = Handle.Get();
  var h2: Handle = Handle.Get();
  var thread: Thread = Thread.Make(fn [a: auto = h1, var b: auto = h2] {
    a.Process();
    b.Process();
  });
  thread.Join();
}
```

## Copy Semantics

**Proposal**: To mirror the behavior of C++, function declarations and lambdas
will be as copyable as their contained function fields and function captures.
This means that, if a function holds a by-object function field, if the type of
the field is copyable, so too is the function that contains it. This also
applies to captures.

The other case is by-value function fields. Since C++ const references, when
made into fields of a class, prevent the class from being copied assigned, so
too should by-value function fields prevent the function in which it is
contained from being copied assigned.

## Self and Recursion

**Proposal**: To mirror C++'s use of capturing `this`, `self` should always come
from the outer scope as a capture. `self: Self` is never permitted on lambdas.
For function declarations, it is only permitted when the function is a member of
a class type or an interface, such that it refers to the class/interface and not
to the function itself.

Note: Given the direction in
[#3720](https://github.com/carbon-language/carbon-lang/pull/3720), an expression
of the form `x.(F)`, where `F` is a function with a `self` or `addr self`
parameter, produces a callable that holds the value of `x`, and does not hold
the value of `F`. As a consequence, we can't support combining captures and
function fields with a `self` parameter.

## Rationale

Lambdas in Carbon serve two purposes. The primary purpose is in support of the
["Code that is easy to read, understand, and write"](/docs/project/goals.md#code-that-is-easy-to-read-understand-and-write)
goal. It is because of this goal that we leverage syntactic features such as the
returned expression (indicated by `=>`) and positional parameters (indicated by
the lack of a tuple pattern of explicit parameters as well as the use of `$N` in
the body of such functions). In addition, Lambdas serve to support the
[Interoperability with and migration from existing C++ code](/docs/project/goals.md#interoperability-with-and-migration-from-existing-c-code)
goal. They are defined at their point of use and are often anonymous, meaning
replacing C++ lambdas solely with function declarations will create an ergonomic
burden compounded by the need for the migration tool to select a name.

## Alternatives Considered

### Alternative Considered: Terse vs Elaborated

Proposed above is a continuous syntax between lambdas and function declarations.
Alternatively, Carbon could adopt a few different categories of functions, as
was considered in a previous discussion doc
([Lambdas Discussion 2](https://docs.google.com/document/d/14K_YLjChWyyNv3wv5Mn7uLFHa0JZTc21v_WP8RzC8M4/)).
These categories would be terse lambdas, elaborated lambdas, and function
declarations. Unfortunately, separating these categories out presented a
syntactic challenge in the form of cliffs, explained below. As a result, they
were decided against.

Terse lambdas were slated to be the most compact form of a lambda. Combined with
a [sigil introducer](#alternative-considered-sigil), they would be syntactically
minimal. One way in which syntax was minimized was the granting of an
**implicit** default [capture](#function-captures) mode. If no square brackets
were present, by-value captures would be allowed. This, combined with the lack
of an arrow to signify a return value, created syntax of the following form
(being passed into the filter function below).

```
let zero: i32 = 0;
let list_all: List(i32) = GetAllValues();
let list_positive: List(i32) = list_all.Filter(
  @ $0 > zero
);
```

To give users more control over the feature set in a lambda, the next step up
was an elaborated lambda. This provided the ability to add both square brackets
and explicit parameters to lambdas at the cost of more syntax. Unfortunately,
this also meant there was a bit of a _syntactic cliff_ and a stumbling block. It
was considered desirable for empty square brackets to mean capturing is
disabled. But since the no-square-brackets form needed to support capturing for
terse lambdas, elaborated lambdas needed to both add the square brackets and
also add an explicit default capture mode at the same time just to maintain the
existing capturing behavior. The net result was code that looked like the
following (being passed into the filter function again).

```
let zero: i32 = 0;
let list_all: List(i32) = GetAllValues();
let list_positive: List(i32) = list_all.Filter(
  @[let](x: auto) x > zero
);
```

Finally, if a user wanted to upgrade a lambda to a function declaration, this
created another cliff where they needed to switch from the sigil to the `fn`
keyword, on top of adding a name. Ultimately these downsides suggested that a
continuous syntax was the better path forward, despite the face that the
shortest spellable lambda would be a bit less terse than the alternative
considered.

### Alternative Considered: Sigil

Proposed above is the use of `fn` as the [introducer](#introducer) for all
functions/lambdas. An alternative considered was to tntroduce with a sigil, such
as `$` or `@`. Since introducer punctuation is such a scarce resource, and since
there was no consensus on what sigil would best represent a lambda, and since
there was a desire to create a more continuous syntax between lambdas and
function declarations, this alternative was decided against. It would have
looked like the following:

```
let lambda1: auto = @ => T.Make();

let lambda2: auto = @[]() -> T { return T.Make(); };
```

### Alternative Considered: Additional Positional Parameter Restriction

In addition to
[the above proposed restrictions](#positional-parameter-restrictions) to
positional parameters, an additional restriction was considered. That being,
visibility of functions with positional parameters could be restricted to only
non-public interfaces. This alternative was considered by way of a leads
question ([#3860](https://github.com/carbon-language/carbon-lang/issues/3860))
and was decided against, with the speculation that such a restriction may be
enforced by way of an HOA rule as opposed to a compiler error.

### Alternative Considered: Recursive Self

Proposed above is a deliniation between function declarations that can provide a
`self` parameter and functions declarations (plus lambdas) which cannot. An
alternative was considered such that, for use in recursion, `self: Self` could
be permitted on all functions and lambdas and refer to the function itself.
Unfortunately, it created a bit of a discontinuity between class members and
non-class members and was thus decided against.

## Future Work

### Future Work: Reference Captures

Much discussion has been had so far about the implications of capturing by
reference. For now, such behavior is supported not through captures but instead
through function fields formed from the address of an object in the outer scope.
It is **imperative** that more work be done in this area to address the
ergonomic concerns of the current solution.
