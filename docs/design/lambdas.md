# Lambdas

<!--
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

<!-- toc -->

## Table of contents

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

<!-- tocstop -->

## Syntax Overview

**Proposal**: A largely continuous syntax between lambdas and function
declarations.

At a high level, lambdas and function declarations will look like the following.

```carbon
// In a variable:
let lambda: auto = fn => T.Make();
// Equivalent in C++23:
// const auto lambda = [] { return T::Make(); };

// In a function call:
Foo(10, 20, fn => T.Make());
// Equivalent in C++23:
// Foo(10, 20, [] { return T::Make(); });
```

```carbon
// In a variable:
let lambda: auto = fn -> T { return T.Make(); };
// Equivalent in C++23:
// const auto lambda = [] -> T { return T::Make(); };

// In a function call:
PushBack(my_list, fn => T.Make());
// Equivalent in C++23:
// PushBack(my_list, [] { return T::Make(); });
```

```carbon
fn FunctionDeclaration => T.Make();
// Equivalent in C++23:
// auto FunctionDeclaration() { return T.Make(); }
```

```carbon
fn FunctionDeclaration -> T { return T.Make(); }
// Equivalent in C++23:
// auto FunctionDeclaration() -> T { return T::Make(); }
```

There are functions which return an expression, such that the return type is
`auto`.

```carbon
// In a variable:
let lambda: auto = fn => T.Make();
// Equivalent in C++23:
// const auto lambda = [] { return T::Make(); };

// In a function call:
Foo(fn => T.Make());
// Equivalent in C++23:
// Foo([] { return T::Make(); });
```

```carbon
fn FunctionDeclaration => T.Make();
// Equivalent in C++23:
// auto FunctionDeclaration() { return T::Make(); }
```

And there are functions with an explicit return type that provide a body of
statements.

```carbon
// In a variable:
let lambda: auto = fn -> T { return T.Make(); };
// Equivalent in C++23:
// const auto lambda = [] -> T { return T::Make(); };

// In a function call:
Foo(fn -> T { return T.Make(); })
// Equivalent in C++23:
// Foo([] -> T { return T::Make(); });
```

```carbon
fn FunctionDeclaration -> T { return T.Make(); }
// Equivalent in C++23:
// auto FunctionDeclaration() -> T { return T::Make(); }
```

There are even functions that provide a body of statements but no return value.

```carbon
// In a variable:
let lambda: auto = fn { Print(T.Make()); };
// Equivalent in C++23:
// const auto lambda = [] -> void { Print(T::Make()); };

// In a function call:
Foo(fn { Print(T.Make()); });
// Equivalent in C++23:
// Foo([] -> void { Print(T::Make()); });
```

```carbon
fn FunctionDeclaration { Print(T.Make()); }
// Equivalent in C++23:
// auto FunctionDeclaration() -> void { Print(T::Make()); }
```

Functions support [captures](#function-captures), [fields](#function-fields) and
deduced parameters in the square brackets. In addition, `self: Self` or
`addr self: Self*` can be added to the square brackets of function declarations
that exist inside class or interface definitions.

```carbon
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

```carbon
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

```carbon
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

```carbon
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

```carbon
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

```carbon
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

```carbon
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

```swift
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

```carbon
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

```carbon
fn Foo() {
  let handle: Handle = Handle.Get();
  var thread: Thread = Thread.Make(fn [var handle] { handle.Process(); });
  thread.Join();
}
```

```carbon
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

```carbon
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

```carbon
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

```carbon
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

```carbon
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
