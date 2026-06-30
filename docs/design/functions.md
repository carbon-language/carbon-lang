# Functions

<!--
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

<!-- toc -->

## Table of contents

-   [Syntax](#syntax)
    -   [Implicit parameters](#implicit-parameters)
    -   [Named and positional parameters](#named-and-positional-parameters)
    -   [Body](#body)
-   [Function and lambda definitions](#function-and-lambda-definitions)
    -   [Function signatures](#function-signatures)
        -   [Return specification](#return-specification)
        -   [Unused parameters](#unused-parameters)
    -   [`return` statements](#return-statements)
-   [Positional parameters](#positional-parameters)
-   [Captures](#captures)
    -   [Capture restrictions on named functions](#capture-restrictions-on-named-functions)
    -   [Capture modes](#capture-modes)
    -   [Default capture mode](#default-capture-mode)
    -   [Function fields](#function-fields)
-   [Copy semantics](#copy-semantics)
-   [Lambdas](#lambdas)
    -   [Lambdas may not take `self` as a parameter](#lambdas-may-not-take-self-as-a-parameter)
    -   [Lambda and function syntax comparison](#lambda-and-function-syntax-comparison)
-   [Forward declarations](#forward-declarations)
    -   [Redeclaration matching](#redeclaration-matching)
-   [Function types and values](#function-types-and-values)
    -   [Bound methods](#bound-methods)
-   [Function calls](#function-calls)
    -   [Direct calls](#direct-calls)
    -   [Indirect calls and the `Call` interface](#indirect-calls-and-the-call-interface)
    -   [Overloaded call operator](#overloaded-call-operator)
-   [Functions in other features](#functions-in-other-features)
-   [Alternatives considered](#alternatives-considered)
-   [References](#references)

<!-- tocstop -->

## Syntax

Functions are the core building block for applications. Carbon supports both
named function declarations and anonymous function expressions called _lambdas_.

A named function definition or declaration has one of the following syntactic
forms (where items in square brackets are optional and independent):

-   `fn` _name_ [_implicit-parameters_] [_tuple-pattern_] `=>` _expression_ `;`
-   `fn` _name_ [_implicit-parameters_] [_tuple-pattern_] [`->` _return-form_] `{`
    _statements_ `}`
-   `fn` _name_ [_implicit-parameters_] _tuple-pattern_ [`->` _return-form_] `;`

A lambda expression has one of the following syntactic forms:

-   `fn` [_implicit-parameters_] [_tuple-pattern_] `=>` _expression_
-   `fn` [_implicit-parameters_] [_tuple-pattern_] [`->` _return-form_] `{` _statements_
    `}`

Named function definitions are distinguished from lambdas by the presence of a
name after the `fn` keyword. If a statement or declaration begins with `fn`, a
name is required and it becomes a function declaration. Otherwise, if in an
expression context, `fn` introduces a lambda.

The first form in both cases is a shorthand: `=> expression` is equivalent to
`-> auto { return expression; }` (with a trailing semicolon for named
functions).

The syntax for parameters, captures, and returns is the same for functions and
lambdas.

### Implicit parameters

The optional _implicit-parameters_ part of a function declaration consists of
square brackets `[]` enclosing zero or more comma-separated items. Each item is
either a default capture mode, an explicit capture, a function field, or a
deduced parameter. The default capture mode, if present, must come first; the
other items can appear in any order. If _implicit-parameters_ is omitted, it is
equivalent to `[]`.

See [captures](#captures) and
[deduced parameters](generics/overview.md#deduced-parameters) for details.

### Named and positional parameters

The presence of _tuple-pattern_ determines whether the function body uses named
or positional parameters.

> _tuple-pattern_: parentheses `(`...`)` enclosing a list of explicit parameter
> patterns

See the
[pattern matching design](pattern_matching.md#pattern-syntax-and-semantics) for
details about named parameters, and
[positional parameters](#positional-parameters) for details about positional
parameters.

### Body

When a body is present (in `{`...`}` or after `=>`), it is a function or lambda
definition. The body introduces nested scopes which may contain local variable
declarations. A named function with only a signature and no body is a forward
declaration.

## Function and lambda definitions

A basic named function definition may look like:

```carbon
fn Add(a: i64, b: i64) -> i64 {
  return a + b;
}
```

Or using the shorthand `=>` return expression syntax:

```carbon
fn Add(a: i64, b: i64) => a + b;
```

These declare a function called `Add` which accepts two `i64` parameters, the
first called `a` and the second called `b`, and returns an `i64` result.

C++ might declare the same thing:

```cpp
std::int64_t Add(std::int64_t a, std::int64_t b) {
  return a + b;
}

// Or with trailing return type syntax:
auto Add(std::int64_t a, std::int64_t b) -> std::int64_t {
  return a + b;
}
```

### Function signatures

#### Return specification

The return type of a function or lambda can be specified using a return clause
(`->`), or it can be deduced using a signature return expression (`=>`).

-   `->` followed by a return form:
    -   Most commonly, this will be an _expression_ that directly states the
        return type, such as `i64`.
        -   The expression will be evaluated at compile-time, so must be valid
            in that context.
        -   For example, `fn ToString(val: i64) -> strbuf;` has a return type of
            `strbuf`.
    -   A return form can also use `val`, `ref`, and `var` to control the
        function call's expression category. For example, `-> ref i32` indicates
        that the function returns by reference. See
        ["Function calls and returns"](values.md#function-calls-and-returns) for
        details.
-   `->` followed by the `auto` keyword indicates that
    [type inference](type_inference.md) should be used to determine the return
    type.
    -   For example, `fn Echo(val: i64) -> auto { return val; }` will have a
        return type of `i64` through type inference.
    -   Forward declarations must have a known return type, so `auto` is not
        valid.
    -   The function must have precisely one `return` statement. That `return`
        statement's expression will then be used for type inference.
    -   The `auto` keyword may be preceded by `val`, `ref`, or `var` to specify
        the return expression category.
-   Omission of both `->` and `=>` indicates that the return type is the empty
    tuple, `()`.
    -   For example, `fn Sleep(seconds: i64);` is similar to
        `fn Sleep(seconds: i64) -> ();`.
    -   `()` is similar to a `void` return type in C++.
    -   If `->` is omitted, `return` statements in the function body must not be
        followed by an expression.
-   `=>` followed by an _expression_ defines a shorthand for a function body
    that returns the expression. The return type is deduced as if `-> auto` were
    used.
    -   For example, `fn Add(a: i64, b: i64) => a + b;` has a return type of
        `i64` based on the type of the expression `a + b`.
    -   Because the return type is deduced and not explicitly known, functions
        defined using `=>` cannot have a separate forward declaration.

> **TODO:** Update this section to cover extended return types, as discussed
> [here](values.md#function-calls-and-returns).

#### Unused parameters

When a parameter introduced in a function definition is not used in the function
body, a compiler warning is issued. To suppress this warning, a parameter can be
explicitly marked as unused in one of two ways:

-   **Anonymous parameters**: By using `_` in place of the parameter name (for
    example, `_: i32`).
-   **`unused` parameters**: By preceding the parameter name with the `unused`
    keyword (for example, `unused size: i32`), which allows preserving the
    parameter name for documentation purposes.

Both of these forms are patterns. For more details on the behavior of `unused`
name bindings and patterns, see the
[pattern matching design](pattern_matching.md#unused).

For example:

```carbon
// Function declaration (for example, in an API file)
fn Sum(x: List(i32), size: i32) -> i32;

// Implementation that does not use the `size` parameter, using an
// anonymous parameter:
fn Sum(x: List(i32), _: i32) -> i32 { ... }

// Or using the `unused` keyword to keep the name for documentation:
fn Sum(x: List(i32), unused size: i32) -> i32 { ... }
```

`unused` markers may only appear on definitions, not on non-defining
declarations. The names of parameters must match between redeclarations, but the
presence of the `unused` marker does not need to match, see
[redeclaration matching](#redeclaration-matching).

### `return` statements

The [`return` statement](control_flow/return.md) is essential to function
control flow. It ends the flow of the function and returns execution to the
caller.

When the [return clause](#return-specification) is omitted, the `return`
statement has no expression argument, and function control flow implicitly ends
after the last statement in the function's body as if `return;` were present.

When the return clause is provided, including when it is `-> ()`, the `return`
statement must have an expression that is convertible to the return type, and a
`return` statement must be used to end control flow of the function.

> **TODO:** Update this section to cover the requirements on the form of the
> expression.

## Positional parameters

Positional parameters, denoted by a dollar sign followed by a non-negative
integer (for example, `$3`), are auto-typed parameters defined within the
function or lambda body when the explicit parameter list (parentheses) is
omitted.

```carbon
let lambda: auto = fn => $0
```

They are variadic by design, meaning an unbounded number of arguments can be
passed to any function or lambda that lacks an explicit parameter list. Only the
parameters that are named in the body will be read from, meaning the highest
named parameter denotes the minimum number of arguments required by the
function. The body is free to omit lower-numbered parameters (for example,
`fn { Print($10); }`).

This syntax was inspired by Swift's
[Shorthand Argument Names](https://docs.swift.org/swift-book/documentation/the-swift-programming-language/closures/#Shorthand-Argument-Names).

```carbon
// A lambda that takes two positional parameters being used as a comparator
Sort(my_list, fn => $0.val < $1.val);
// In Swift: { $0.val < $1.val }
```

When positional parameters are used in a nested function definition, exactly one
of the enclosing functions must omit the explicit parameter list, and they are
interpreted as parameters of that function:

```carbon
fn F {
  fn G[let]() -> auto {
    // `$0` is a parameter of `F` that is captured by `G`,
    // not a parameter of `G`.
    return $0;
  }
}
```

This means that a single function or lambda cannot have both named and
positional parameters:

```carbon
fn Foo(x: i32) -> i32 {
  // ❌ Invalid since `Foo` is already using named parameters.
  return $0;
}
```

Positional parameters can only be used with function definitions, not forward
declarations.

## Captures

Captures in Carbon mirror the non-init captures of C++. A capture declaration
consists of a capture mode (for `var` captures) followed by the name of a
binding from the enclosing scope, and makes that identifier available in the
inner function body. These captures are specified in square brackets `[`...`]`
as part of the implicit parameter list. The lifetime of a capture is the
lifetime of the function in which it exists.

For example:

```carbon
fn InLambda() {
  let handle: Handle = Handle.Get();
  var thread: Thread = Thread.Make(fn [var handle] { handle.Process(); });
  thread.Join();
}
```

```carbon
fn InNamedFunction() {
  let handle: Handle = Handle.Get();
  fn MyThread[handle]() { handle.Process(); }
  var thread: Thread = Thread.Make(MyThread);
  thread.Join();
}
```

### Capture restrictions on named functions

While lambdas can use captures freely, named function definitions support
captures (and function fields) with these restrictions:

-   They can only be used on functions where the definition is attached to the
    declaration (so they cannot be forward declared).
-   Captures and function fields are only supported on local function
    definitions immediately defined inside the body of another function.

### Capture modes

Lambdas and local functions can capture variables from their surrounding scope
using `let` or `var`, just like regular bindings.

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

By default, there is no capturing in lambdas and functions. The lack of any
square brackets is the same as an empty pair of square brackets. Users can opt
into capturing behavior. This is done either by way of individual explicit
captures, or more succinctly by way of a default capture mode. The default
capture mode roughly mirrors the syntax `[=]` and `[&]` capture modes from C++
by being the first thing to appear in the square brackets.

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

### Function fields

Function fields mirror the behavior of init captures in C++. Function fields are
defined in the implicit parameter list, and are allowed only where captures are
allowed. A function field definition consists of an irrefutable pattern, `=`,
and an initializer. It matches the pattern with the initializer when the
function definition is evaluated. The bindings in the pattern have the same
lifetime as the function, and their scope extends to the end of the function
body.

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

To mirror the behavior of C++, lambdas and functions with captures or function
fields will be as copyable as their contained function fields and function
captures. This means that, if a function holds a by-object function field, if
the type of the field is copyable, so too is the function that contains it. This
also applies to captures.

The other case is by-value function fields. Since C++ const references, when
made into fields of a class, prevent the class from being copy assigned, so too
should by-value function fields prevent the function in which it is contained
from being copy assigned.

## Lambdas

One goal of Carbon's lambda syntax is to have continuity between lambdas and
named functions. Below are some example declarations:

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

### Lambdas may not take `self` as a parameter

To mirror C++'s use of capturing `this`, `self` should always come from the
outer scope as a capture. `self` is never permitted in the explicit parameter
list of a lambda.

```carbon
// ❌ Not allowed, lambdas can't be methods.
let lambda: auto = fn (self) { self.F(); };

// ✅ Captures `self` from outer scope
let lambda: auto = fn [self] { self.F(); };
```

Note: Following
[#3720](https://github.com/carbon-language/carbon-lang/pull/3720), an expression
of the form `x.(F)`, where `F` is a function with a `self` or `ref self`
parameter, produces a callable that holds the value of `x`, and does not hold
the value of `F`. As a consequence, we can't support combining captures and
function fields with a `self` parameter.

### Lambda and function syntax comparison

To understand how the syntax between lambdas and function declarations is
reasonably "continuous", refer to this table of syntactic positions and the
following code examples.

| Syntactic Position |                   Syntax Allowed in Given Position (optional, unless otherwise stated)                    |
| :----------------: | :-------------------------------------------------------------------------------------------------------: |
|         A1         |          Required Returned Expression ([positional parameters](#positional-parameters) allowed)           |
|         A2         |         Required Returned Expression ([positional parameters](#positional-parameters) disallowed)         |
|         B          |                               [Default capture mode](#default-capture-mode)                               |
|         C          | Explicit [Captures](#captures), [Function fields](#function-fields) and Deduced Parameters (in any order) |
|         D          |                                            Explicit Parameters                                            |
|         E1         |      Body of Statements (no return value) ([positional parameters](#positional-parameters) allowed)       |
|         E2         |     Body of Statements (with return value) ([positional parameters](#positional-parameters) allowed)      |
|         E3         |     Body of Statements (no return value) ([positional parameters](#positional-parameters) disallowed)     |
|         E4         |    Body of Statements (with return value) ([positional parameters](#positional-parameters) disallowed)    |
|         F          |                                           Required Return Type                                            |

Lambdas (all the following are in an expression context and are themselves
expressions):

```carbon
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

Function Declarations (all the following are allowed as statements in a function
body or as declarations in other scopes):

```carbon
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

## Forward declarations

Functions may be declared separate from the definition by providing only a
signature, with no body. This provides an API which may be called. For example:

```carbon
// Declaration:
fn Add(a: i64, b: i64) -> i64;

// Definition:
fn Add(a: i64, b: i64) -> i64 {
  return a + b;
}
```

The corresponding definition may be provided later in the same file or, when the
declaration is in an
[API file of a library](code_and_name_organization/#libraries), in an
implementation file of the same library.

A function may only be forward declared once in any given file, and any forward
declaration must appear before the definition.

To declare a function that is defined in a different library, the `extern`
modifier is used (for example, `extern fn F();`). A library that declares a
function as `extern` cannot define it. The `extern` modifier is only valid on
namespace-scoped functions, not on member functions of classes. For more details
on cross-library forward declarations and modifier merging, see the
[declaring entities design](declaring_entities.md#extern-and-extern-library).

### Redeclaration matching

Redeclarations of a function must match syntactically. The sequence of tokens
following the `fn` keyword (and optional scope name) up to the semicolon or open
brace must be identical.

Specifically, the following must match exactly between the forward declaration
and the definition:

-   **Parameter names**: You cannot change a parameter name or replace it with
    `_` in the definition.
-   **Parameter types**: The types and grouping parentheses must match exactly.
-   **Return clause**: The presence or omission of the return clause must match
    exactly (for example, an omitted return type behaves equivalent to `-> ()`,
    but they are syntactically different and cannot be mixed).

The only exception is the `unused` modifier on parameters, which is allowed on a
defining declaration (such as the definition) but disallowed on a non-defining
declaration.

Declaration modifiers (such as access control keywords or `virtual`) appear
before the `fn` keyword, so they are not involved in checking whether the two
signatures differ.

## Function types and values

A function declaration in Carbon introduces a new, unique, stateless type,
called a _function type_. The function name is bound to a value of that function
type.

Distinct functions have distinct function types, even if they have the same
signature. A function type is an empty, trivial type. There is no way to name a
function type other than asking for the type of the function value.

```carbon
fn F(x: i32) -> i32 { return x; }

// Compile-time function.
musteval fn TypeOf[T:! type](x: T) -> type { return T; }

// `F` is a first-class value with a first-class type.
let template FType:! type = TypeOf(F);
var my_f: FType = F;
```

Function values are regular values that can be stored in variables, passed to
functions, and so on.

```carbon
fn G() -> i32 {
  // `my_f` has function type `FType`. This is a direct call to `F`.
  return my_f(1);
}
```

For the purpose of the [orphan rule](generics/details.md#orphan-rule), a
function type is considered to be declared by the function declaration that
introduces the function value.

### Bound methods

A function with a `self` parameter is a method. The type of a method is a
stateless type, like other functions. Once the method is
[bound to an instance](expressions/member_access.md#instance-binding), for
example in the expression `object.MethodName`, the result is a _bound method
value_. The type of the result is a _bound method type_, with the same signature
as the method, but with the `self` parameter removed. A bound method type
describes the callee in a method call, and a bound method value specifies the
`self` parameter of the call.

```carbon
class HasMember {
  // `HasMember.F` has a stateless function type, with signature
  // `(self, n: i32) -> i32`.
  fn F(self, n: i32) -> i32;
}

fn F(h1: HasMember, h2: HasMember) -> i32 {
  // `h1.F` is a bound method value whose type is a bound method type,
  // with signature `(n: i32) -> i32`.
  var hf: auto = h1.F;
  // `h1.F` and `h2.F` are of the same bound method type.
  hf = h2.F;
  // Same as `h2.F(4)`.
  return hf(4);
}
```

## Function calls

Function calls use C-like syntax:

> _expression_ `(` _[ expression_ `,` _expression_ `,` _... ]_ `)`

It consists of an expression naming a callee followed by an argument list
enclosed in parentheses, which resembles a tuple of arguments. Calls take the
form `a(b, c, d)` or `a(b, c, d,)`, where:

-   `a` is the callee, which can be a name, a literal, a member access, or some
    more complex expression enclosed in parentheses.
-   `b`, `c`, `d` are any number of argument expressions, each optionally
    prefixed with `ref` if passing to a `ref` parameter. Arguments are separated
    by commas, and if the argument list is not empty, an optional trailing comma
    is permitted but not required after the final argument.

Call syntax is syntactically equivalent to a
[suffix expression](expressions/README.md#suffix-operators) followed by a tuple
literal, except that a tuple literal requires a trailing comma to form a
single-element tuple `(b,)`, whereas in call syntax both `a(b)` and `a(b,)` are
permitted.

A _callable value_ (or _callable_ for short) is a value that can be used as the
callee of a call expression. There are several kinds of callable:

-   Functions, and more generally values of function types.
-   Bound methods, such as `my_vector.Begin`.
-   Lambdas.
-   Parameterized entities, such as a generic class `Vector` or a generic
    interface `AddWith`.
-   Values of dependent types that are
    [constrained to be callable](#indirect-calls-and-the-call-interface).
-   User-defined class types that overload function call syntax.

Function calls are divided into _direct calls_ and _indirect calls_.

### Direct calls

A call expression is a _direct call_ when the callee:

-   is the name of a parameterized entity, like a generic class or interface, or
-   has a function type or bound method type.

Note that this includes virtual method calls, even though those can include some
indirection. In a direct call, a call signature is available which is used to
check the given arguments against the callee's declared implicit and explicit
parameters. This checking proceeds as follows:

-   Argument deduction is performed by comparing the declared parameter types
    against the actual argument types and deducing values for implicit arguments
    that make the types equal.
-   Then, for each binding in the explicit parameter list in turn, all argument
    values that have been deduced are substituted into the parameter.

    -   If the parameter is a `ref` parameter (other than `self`), the
        corresponding argument expression at the call-site must be prefixed with
        `ref`. It is a compile-time error if the call-site has a mismatched
        `ref` prefix:
        -   An argument to a non-`ref` parameter must not be prefixed with
            `ref`, and
        -   An argument to a `ref` parameter must be prefixed with `ref`, except
            in a generic context where the parameter's `ref` status may vary.
    -   If the parameter is a `template :!` binding, the argument expression is
        converted to have the same type as the binding and template constant
        expression phase.
    -   If the parameter is a symbolic `:!` binding, the argument expression is
        converted to have the same type as the binding and symbolic constant
        expression phase.
    -   Otherwise, the parameter is pattern-matched against the argument.

    If a parameter is a `:!` binding, its corresponding converted argument
    expression is evaluated, and its value is added to the list of deduced
    argument values before any later parameters are processed.

The result of the call expression depends on the callee:

-   If the callee is a parameterized entity such as a generic class or a generic
    interface, the result is the specific instance of that generic, such as a
    class or interface, and the call is a value expression of type `type`.
-   If the callee has a function type, the call is an expression whose form is
    the substituted return form of its signature. When evaluated, the call
    expression will invoke the function and produce whatever result it returns.
-   If the callee has a bound method type, it behaves the same as a function
    value, except that the `self` parameter of the called function is bound to
    the `self` value in the bound method value.

### Indirect calls and the `Call` interface

A generic parameter can be constrained to be a callable type using the `Call`
interface:

```carbon
interface Call(... each Arg: type) {
  let Result:! type;
  fn Op(self, ... each arg: each Arg) -> Result;
}
```

A call expression that is not a direct call is an _indirect call_. It is
translated into an invocation of `Call(Arg1, Arg2,` ... `ArgN).Op`, where
`Arg1`, `Arg2`, ... `ArgN` are the types of the call's arguments in order. So
`F(arg1, arg2)` is translated into `F.(Call(Arg1, Arg2).Op)(arg1, arg2)`.

For example, given:

```carbon
fn Sort[T:! type, F:! Call(T, T) where .Result = Ordering]
       (ref v: Vector(T), cmp: F) {
  // ...
  auto ord: auto = cmp(v[i], v[j]);
  // ...
}
```

The call `cmp(v[i], v[j])` is translated into:

```carbon
  auto ord: auto = cmp.(Call(T, T).Op)(v[i], v[j]);
```

A function type or bound method type implements the `Call` interface for every
set of runtime argument types that a direct call to the function or bound method
would accept. The behavior of `Call.Op` is to call the function or bound method
with the provided argument list.

Implicit conversions are permitted for parameters whose types do not involve
deduced parameters. The intent is for the `impl` to support indirect calls in
the same cases where the function supports direct calls, with the same meaning.

```carbon
fn TakeI32Fn[F:! Call(i32)](f: F);
fn I64Fn(n: i64);
fn Run() {
  // ✅ `I64Fn` can be called with an `i32`, because
  // `i32 impls ImplicitAs(i64)`.
  TakeI32Fn(I64Fn);
}
```

> **Future work:** The `Call` interface currently only supports value parameters
> and initializing returns. It is future work to remove this restriction.

### Overloaded call operator

The `Call` interface can be implemented to overload the meaning of the function
call operator for a type.

```carbon
class Func(Arg:! type) {
  impl as Call((Arg,)) where .Result = () {
    fn Op(self, arg: (Arg,)) { Print("hello, world"); }
  }
}

fn Run() {
  let f: Func(i32) = {};
  // ✅ Prints "hello, world".
  f(42);
}
```

There are no constraints on the callee type, beyond the normal constraints for
implementing an interface.

```carbon
class X { var n: i32; }

impl {.a: X} as Call(()) where .Result = i32 {
  fn Op(self, args: ()) -> i32 {
    return self.a.n;
  }
}
fn Run() -> i32 {
  // Returns 1.
  return {.a = {.n = 1} as X}();
}
```

## Functions in other features

Other designs build upon basic function syntax to add advanced features:

-   [Generic functions](generics/overview.md#generic-functions) adds support for
    deduced parameters and compile-time parameters.
-   [Member functions](classes.md#member-functions) adds support for methods and
    non-instance member functions.

## Alternatives considered

-   [Function keyword](/proposals/p000438-functions.md#function-keyword)
-   [Only allow `auto` return types if parameters are compile-time](/proposals/p000826-function-return-type-inference.md#only-allow-auto-return-types-if-parameters-are-generic)
-   [Provide alternate function syntax for concise return type inference](/proposals/p000826-function-return-type-inference.md#provide-alternate-function-syntax-for-concise-return-type-inference)
-   [Allow separate declaration and definition](/proposals/p000826-function-return-type-inference.md#allow-separate-declaration-and-definition)
-   [Signature-based function types](/proposals/p002875-functions-function-types-and-function-calls.md#signature-based-function-types)
-   [Make direct and indirect calls behave uniformly](/proposals/p002875-functions-function-types-and-function-calls.md#make-direct-and-indirect-calls-behave-uniformly)
-   [Terse vs Elaborated lambdas](/proposals/p003848-lambdas.md#alternative-considered-terse-vs-elaborated)
-   [Sigil for lambdas](/proposals/p003848-lambdas.md#alternative-considered-sigil)
-   [Additional Positional Parameter Restriction](/proposals/p003848-lambdas.md#alternative-considered-additional-positional-parameter-restriction)
-   [Recursive Self in lambdas](/proposals/p003848-lambdas.md#alternative-considered-recursive-self)
-   [Keep the `:!` syntax](/proposals/p007254-replace-and-with-keywords-and-contextual-defaults.md#keep-the--syntax)
-   [Alternative keyword names](/proposals/p007254-replace-and-with-keywords-and-contextual-defaults.md#alternative-keyword-names)
-   [Use `template generic` instead of just `template`](/proposals/p007254-replace-and-with-keywords-and-contextual-defaults.md#use-template-generic-instead-of-just-template)
-   [Context-independent syntax](/proposals/p007254-replace-and-with-keywords-and-contextual-defaults.md#context-independent-syntax)
-   [Erased model for generics](/proposals/p007254-replace-and-with-keywords-and-contextual-defaults.md#erased-model-for-generics)
-   [Context-sensitive defaults based on parameter type](/proposals/p007254-replace-and-with-keywords-and-contextual-defaults.md#context-sensitive-defaults-based-on-parameter-type)
-   [Allow redundant phase keywords](/proposals/p007254-replace-and-with-keywords-and-contextual-defaults.md#allow-redundant-phase-keywords)

## References

-   Proposal
    [#438: Add statement syntax for function declarations](https://github.com/carbon-language/carbon-lang/pull/438)
-   Proposal
    [#826: Function return type inference](https://github.com/carbon-language/carbon-lang/pull/826)
-   Proposal
    [#2022: Unused Pattern Bindings (Unused Function Parameters)](https://github.com/carbon-language/carbon-lang/pull/2022)
-   Proposal
    [#2875: Functions, function types, and function calls](https://github.com/carbon-language/carbon-lang/pull/2875)
-   Proposal
    [#3762: Merging forward declarations](https://github.com/carbon-language/carbon-lang/pull/3762)
-   Proposal
    [#3763: Matching redeclarations](https://github.com/carbon-language/carbon-lang/pull/3763)
-   Proposal
    [#3848: Lambdas](https://github.com/carbon-language/carbon-lang/pull/3848)
-   Proposal
    [#5434: `ref` parameters, arguments, returns and `val` returns](https://github.com/carbon-language/carbon-lang/pull/5434)
-   Proposal
    [#7254: Replace `:!` and `:?` with keywords and contextual defaults](https://github.com/carbon-language/carbon-lang/pull/7254)
