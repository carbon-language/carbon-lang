# Lambdas

<!--
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

<!-- toc -->

## Table of contents

-   [Syntax Overview](#syntax-overview)
    -   [Return type](#return-type)
        -   [Return expression](#return-expression)
        -   [Explicit return type](#explicit-return-type)
        -   [No return](#no-return)
    -   [Implicit parameters in square brackets](#implicit-parameters-in-square-brackets)
    -   [Parameters](#parameters)
    -   [Syntax defined](#syntax-defined)
-   [Positional parameters](#positional-parameters)
    -   [Positional parameter restrictions](#positional-parameter-restrictions)
-   [Captures](#captures)
    -   [Capture modes](#capture-modes)
    -   [Default capture mode](#default-capture-mode)
-   [Function fields in lambdas](#function-fields-in-lambdas)
-   [Copy semantics](#copy-semantics)
-   [Self and recursion](#self-and-recursion)
-   [Alternatives considered](#alternatives-considered)

<!-- tocstop -->

## Syntax Overview

One goal of Carbon's lambda syntax is to have continuity between lambdas and
function declarations. Below are some example declarations:

Implicit return types:

```carbon
// In a variable:
let lambda: auto = fn => T.Make();
// Equivalent in C++23:
// const auto lambda = [] { return T::Make(); };

// As an argument to a function call:
Foo(10, 20, fn => T.Make());
// Equivalent in C++23:
// Foo(10, 20, [] { return T::Make(); });
```

Explicit return types:

```carbon
// In a variable:
let lambda: auto = fn -> T { return T.Make(); };
// Equivalent in C++23:
// const auto lambda = [] -> T { return T::Make(); };

// As an argument to a function call:
PushBack(my_list, fn -> T { return T.Make() });
// Equivalent in C++23:
// PushBack(my_list, [] { return T::Make(); });
```

### Return type

There are three options for how a lambda expresses its return type, parallel to
[how function declarations express returns](functions.md#return-clause): using a
return expression, using an explicit return type, or having no return.

#### Return expression

A return expression is introduced with a double arrow (`=>`) followed by an
expression describing the function's return value. In this case, the return type
is determined by the type of the expression, as if the return type was `auto`.

```carbon
// In a variable:
let lambda: auto = fn => T.Make();
// Equivalent in C++23:
// const auto lambda = [] { return T::Make(); };

// As an argument to a function call:
Foo(fn => T.Make());
// Equivalent in C++23:
// Foo([] { return T::Make(); });
```

#### Explicit return type

An explicit return type is introduced with a single arrow (`->`), followed by
the return type, and finally the body of the lambda with a sequence of
statements enclosed in curly braces (`{`...`}`).

```carbon
// In a variable:
let lambda: auto = fn -> T { return T.Make(); };
// Equivalent in C++23:
// const auto lambda = [] -> T { return T::Make(); };

// As an argument to a function call:
Foo(fn -> T { return T.Make(); });
// Equivalent in C++23:
// Foo([] -> T { return T::Make(); });
```

#### No return

Lambdas that don't return anything end with a body of statements in curly braces
(`{`...`}`).

```carbon
// In a variable:
let lambda: auto = fn { Print(T.Make()); };
// Equivalent in C++23:
// const auto lambda = [] -> void { Print(T::Make()); };

// As an argument to a function call:
Foo(fn { Print(T.Make()); });
// Equivalent in C++23:
// Foo([] -> void { Print(T::Make()); });
```

### Implicit parameters in square brackets

Lambdas support [captures](#captures), [fields](#function-fields-in-lambdas) and
deduced parameters in the square brackets.

```carbon
fn Foo(x: i32) {
  // In a variable:
  let lambda: auto = fn [var x, var y: i32 = 0] { Print(++x, ++y); };
  // Equivalent in C++23:
  // const auto lambda = [x, y = int32_t{0}] mutable -> void { Print(++x, ++y); };

  // As an argument to a function call:
  Foo(fn [var x, var y: i32 = 0] { Print(++x, ++y); });
  // Equivalent in C++23:
  // Foo([x, y = int32_t{0}] mutable -> void { Print(++x, ++y); });
}
```

### Parameters

Lambdas also support so-called ["positional parameters"](#positional-parameters)
that are defined at their point of use using a dollar sign and a non-negative
integer. They are implicitly of type `auto`.

```carbon
fn Foo() {
  let lambda: auto = fn { Print($0); };
  // Equivalent in C++23:
  // auto lambda = [](auto _0, auto...) -> void { Print(_0); };
  // Equivalent in Swift:
  // let lambda = { Print($0) };
}
```

Of course, lambdas can also have named parameters, but a single lambda can't
have both named and positional parameters.

```carbon
fn Foo() {
  // In a variable:
  let lambda: auto = fn (v: auto) { Print(v); };
  // Equivalent in C++23:
  // const auto lambda = [](v: auto) -> void { Print(v); };

  // As an argument to a function call:
  Foo(fn (v: auto) { Print(v); });
  // Equivalent in C++23:
  // Foo([](v: auto) { Print(v); });
}
```

And in additional the option between positional and named parameters, deduced
parameters are always permitted.

```carbon
fn Foo() {
  let lambda: auto = fn [T:! Printable](t: T) { Print(t); };
}
```

### Syntax defined

Lambda expressions have one of the following syntactic forms (where items in
square brackets are optional and independent):

`fn`\[_implicit-parameters_\] \[_tuple-pattern_\] `=>` _expression_

`fn` \[_implicit-parameters_\] \[_tuple-pattern_\] \[`->` _return-type_\] `{`
_statements_ `}`

The first form is a shorthand for the second: "`=>` _expression_" is equivalent
to "`-> auto { return` _expression_ `; }`".

_implicit-parameters_ consists of square brackets enclosing a optional default
capture mode and any number of explicit captures, function fields, and deduced
parameters, all separated by commas. The default capture mode (if any) must come
first; the other items can appear in any order. If _implicit-parameters_ is
omitted, it is equivalent to `[]`.

Function definitions are distinguished from lambdas by the presence of a name
after the `fn` keyword.

The presence of _tuple-pattern_ determines whether the function body uses named
or positional parameters.

The presence of "`->` _return-type_" determines whether the function body can
(and must) return a value.

To understand how the syntax between lambdas and function declarations is
reasonably "continuous", refer to this table of syntactic positions and the
following code examples.

| Syntactic Position |                         Syntax Allowed in Given Position (optional, unless otherwise stated)                         |
| :----------------: | :------------------------------------------------------------------------------------------------------------------: |
|         A1         |                Required Returned Expression ([positional parameters](#positional-parameters) allowed)                |
|         A2         |              Required Returned Expression ([positional parameters](#positional-parameters) disallowed)               |
|         B          |                                    [Default capture mode](#default-capture-mode)                                     |
|         C          | Explicit [Captures](#captures), [Function fields](#function-fields-in-lambdas) and Deduced Parameters (in any order) |
|         D          |                                                 Explicit Parameters                                                  |
|         E1         |            Body of Statements (no return value) ([positional parameters](#positional-parameters) allowed)            |
|         E2         |           Body of Statements (with return value) ([positional parameters](#positional-parameters) allowed)           |
|         E3         |          Body of Statements (no return value) ([positional parameters](#positional-parameters) disallowed)           |
|         E4         |         Body of Statements (with return value) ([positional parameters](#positional-parameters) disallowed)          |
|         F          |                                                 Required Return Type                                                 |

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

## Positional parameters

Positional parameters, denoted by a dollar sign followed by a non-negative
integer (for example, $3), are auto-typed parameters defined within the lambda's
body.

```carbon
let lambda: auto = fn => $0
```

They can be used in any lambda declaration that lacks an explicit parameter list
(parentheses). They are variadic by design, meaning an unbounded number of
arguments can be passed to any function that lacks an explicit parameter list.
Only the parameters that are named in the body will be read from, meaning the
highest named parameter denotes the minimum number of arguments required by the
function. The lambda body is free to omit lower-numbered parameters (ex:
`fn { Print($10); }`).

This syntax was inpsired by Swift's
[Shorthand Argument Names](https://docs.swift.org/swift-book/documentation/the-swift-programming-language/closures/#Shorthand-Argument-Names).

```carbon
// A lambda that takes two positional parameters being used as a comparator
Sort(my_list, fn => $0.val < $1.val);
// In Swift: { $0.val < $1.val }
```

### Positional parameter restrictions

Lambdas with positional parameters have the restriction that they can only be
used in a context where there is exactly one enclosing function or lambda that
has no explicit parameter list. For example:

```carbon
fn Foo1 {
  // ❌ Invalid: Foo1 is already using positional parameters
  let lambda: auto = fn => $0 < $1
}

fn Foo2 {
  my_list.Sort(
    // ❌ Invalid: Foo2 is already using positional parameters
    fn => $0 < $1
  );
}

fn Foo3() {
  my_list.Sort(
    // ✅ Valid: Foo3 has explicit parameters
    fn => $0 < $1
  );
}

fn Foo4() {
  let lambda: auto = fn -> bool {
    // ❌ Invalid: Outer lambda is already using positional parameters
    return (fn => $0 < $1)($0, $1);
  };
}

fn Foo5() {
  let lambda: auto = fn (x: i32, y: i32) -> bool {
    // ✅ Valid: Outer lambda has explicit parameters
    return (fn => $0 < $1)(x, y);
  };
}
```

## Captures

Captures in Carbon mirror the non-init captures of C++. A capture declaration
consists of a capture mode (for `var` captures) followed by the name of a
binding from the enclosing scope, and makes that identifier available in the
inner function body. The lifetime of a capture is the lifetime of the function
in which it exists. For example...

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

### Capture modes

Lambdas can capture variables from their surrounding scope using `let` or `var`,
just like regular bindings.

Capture modes can be used as
[default capture mode specifiers](#default-capture-mode) or for explicit
captures as shown in the example code below.

```carbon
fn Example() {
  var a: i32 = 0;
  var b: i32 = 0;

let lambda: auto = fn [a, var b] {
  // ❌ Invalid: by-value captures are immutable (default `let`)
  a += 1;
  // ✅ Valid: `b` is a mutable copy (captured with `var`)
  b += 1;
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

### Default capture mode

By default, there is no capturing in lambdas. The lack of any square brackets is
the same as an empty pair of square brackets. Users can opt into capturing
behavior. This is done either by way of individual explicit captures, or more
succinctly by way of a default capture mode. The default capture mode roughly
mirrors the syntax `[=]` and `[&]` capture modes from C++ by being the first
thing to appear in the square brackets.

```carbon
fn Foo1() {
  let handle: Handle = Handle.Get();
  fn MyThread[var]() {
    // `handle` is captured by-object due to the default capture
    // mode specifier of `var`
    handle.Process();
  }
  var thread: Thread = Thread.Make(MyThread);
  thread.Join();
}

fn Foo2() {
  let handle: Handle = Handle.Get();
  fn MyThread[let]() {
    // `handle` is captured by-value due to the default capture
    // mode specifier of `let`
    handle.Process();
  }
  var thread: Thread = Thread.Make(MyThread);
  thread.Join();
}
```

## Function fields in lambdas

Function fields in lambdas mirror the behavior of init captures in C++. A
function field definition consists of an irrefutable pattern, `=`, and an
initializer. It matches the pattern with the initializer when the lambda
definition is evaluated. The bindings in the pattern have the same lifetime as
the function, and their scope extends to the end of the function body.

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

## Copy semantics

To mirror the behavior of C++, lambdas will be as copyable as their contained
function fields and function captures. This means that, if a function holds a
by-object function field, if the type of the field is copyable, so too is the
function that contains it. This also applies to captures.

The other case is by-value function fields. Since C++ const references, when
made into fields of a class, prevent the class from being copied assigned, so
too should by-value function fields prevent the function in which it is
contained from being copied assigned.

## Self and recursion

To mirror C++'s use of capturing `this`, `self` should always come from the
outer scope as a capture. `self: Self` is never permitted on lambdas.

```carbon
// ❌ Not allowed
let lambda: auto = fn [self: Self] { self.F(); };

// ✅ Captures `self` from outer scope
let lambda: auto = fn [self] { self.F(); };
```

Note: Following
[#3720](https://github.com/carbon-language/carbon-lang/pull/3720), an expression
of the form `x.(F)`, where `F` is a function with a `self` or `ref self`
parameter, produces a callable that holds the value of `x`, and does not hold
the value of `F`. As a consequence, we can't support combining captures and
function fields with a `self` parameter.

## Alternatives considered

-   [Terse vs Elaborated](/proposals/p3848.md#alternative-considered-terse-vs-elaborated)
-   [Sigil](/proposals/p3848.md#alternative-considered-sigil)
-   [Additional Positional Parameter Restriction](/proposals/p3848.md#alternative-considered-additional-positional-parameter-restriction)
-   [Recursive Self](/proposals/p3848.md#alternative-considered-recursive-self)
