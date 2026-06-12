# Functions

<!--
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

<!-- toc -->

## Table of contents

-   [Overview](#overview)
-   [Function definitions](#function-definitions)
    -   [Return clause](#return-clause)
    -   [`return` statements](#return-statements)
-   [Function declarations](#function-declarations)
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

## Overview

> **TODO:** Update this document to reflect the changes to named functions in
> [#3848: Lambdas](/proposals/p003848-lambdas.md).

Functions are the core building block for applications. Carbon's basic function
syntax is:

-   _parameter_: _identifier_ `:` _expression_
-   _parameter-list_: _[ parameter_ `,` _parameter_ `,` _... ]_
-   _return-clause_: _[_ `->` _< expression |_ `auto` _> ]_
-   _signature_: `fn` _identifier_ `(` _parameter-list_ `)` _return-clause_
-   _function-definition_: _signature_ `{` _statements_ `}`
-   _function-declaration_: _signature_ `;`
-   _function-call_: _identifier_ `(` _[ expression_ `,` _expression_ `,` _...
    ]_ `)`

A function with only a signature and no body is a function declaration, or
forward declaration. When the body is a present, it's a function definition. The
body introduces nested scopes which may contain local variable declarations.

## Function definitions

A basic function definition may look like:

```carbon
fn Add(a: i64, b: i64) -> i64 {
  return a + b;
}
```

This declares a function called `Add` which accepts two `i64` parameters, the
first called `a` and the second called `b`, and returns an `i64` result. It
returns the result of adding the two arguments.

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

### Return clause

The return clause of a function specifies the return type using one of three
possible syntaxes:

-   `->` followed by an _expression_, such as `i64`, directly states the return
    type. This expression will be evaluated at compile-time, so must be valid in
    that context.
    -   For example, `fn ToString(val: i64) -> String;` has a return type of
        `String`.
-   `->` followed by the `auto` keyword indicates that
    [type inference](type_inference.md) should be used to determine the return
    type.
    -   For example, `fn Echo(val: i64) -> auto { return val; }` will have a
        return type of `i64` through type inference.
    -   Declarations must have a known return type, so `auto` is not valid.
    -   The function must have precisely one `return` statement. That `return`
        statement's expression will then be used for type inference.
-   Omission indicates that the return type is the empty tuple, `()`.
    -   For example, `fn Sleep(seconds: i64);` is similar to
        `fn Sleep(seconds: i64) -> ();`.
    -   `()` is similar to a `void` return type in C++.

> **TODO:** Update this section to cover return forms, as discussed
> [here](values.md#function-calls-and-returns).

### `return` statements

The [`return` statement](control_flow/return.md) is essential to function
control flow. It ends the flow of the function and returns execution to the
caller.

When the [return clause](#return-clause) is omitted, the `return` statement has
no expression argument, and function control flow implicitly ends after the last
statement in the function's body as if `return;` were present.

When the return clause is provided, including when it is `-> ()`, the `return`
statement must have an expression that is convertible to the return type, and a
`return` statement must be used to end control flow of the function.

> **TODO:** Update this section to cover the requirements on the form of the
> expression.

## Function declarations

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
implementation file of the same library. The signature of a function declaration
must match the corresponding definition. This includes the
[return clause](#return-clause); even though an omitted return type has
equivalent behavior to `-> ()`, the presence or omission must match.

## Function types and values

A function declaration in Carbon introduces a new, unique, stateless type,
called a _function type_. The function name is bound to a value of that function
type.

Distinct functions have distinct function types, even if they have the same
signature. A function type is an empty, trivial type. There is no way to name a
function type other than asking for the type of the function value (for example,
using `TypeOf`).

```carbon
fn F(x: i32) -> i32 { return x; }

// Compile-time function.
fn TypeOf[T:! type](x: T) -> type { return T; }

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

For each function type corresponding to a method, there is a corresponding
_bound method type_. When a member access is performed on an object of class
type to access a method, the result is a _bound method value_ of bound method
type. A bound method type describes the callee in a method call, and a bound
method value describes the `self` parameter of the call.

```carbon
class HasMember {
  // `HasMember.F` has a stateless function type, with signature
  // `[self: Self](n: i32) -> i32`.
  fn F[self: Self](n: i32) -> i32;
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

Function calls use C-like syntax: an expression naming a callable is followed by
an argument list enclosed in parentheses, which resembles a tuple of arguments.
Calls take the form `a(b, c, d)` or `a(b, c, d,)`, where:

-   `a` is the callee, which can be a name, a literal, a member access, or some
    more complex expression enclosed in parentheses.
-   `b`, `c`, `d` are any number of argument expressions. Arguments are
    separated by commas, and if the argument list is not empty, an optional
    trailing comma is permitted but not required after the final argument.

Call syntax is syntactically equivalent to a primary expression followed by a
tuple literal, except that a tuple literal requires a trailing comma to form a
single-element tuple `(b,)`, whereas in call syntax both `a(b)` and `a(b,)` are
permitted.

There are several kinds of callable:

-   Functions, and more generally values of function types.
-   Bound methods, such as `my_vector.Begin`.
-   Lambdas.
-   Parameterized entities, such as a generic class `Vector` or a generic
    interface `AddWith`.
-   Values of dependent types that are constrained to be callable.
-   User-defined class types that overload function call syntax.

Function calls are divided into _direct calls_ and _indirect calls_.

### Direct calls

A call expression is a _direct call_ when the callee:

-   is the name of a parameterized entity, like a generic class or interface, or
-   has a function type or bound method type.

In a direct call, a call signature is available which is used to check the given
arguments against the callee's declared implicit and explicit parameters. This
checking proceeds as follows:

-   Argument deduction is performed by comparing the declared parameter types
    against the actual argument types and deducing values for implicit arguments
    that make the types equal.
-   Then, for each binding in the explicit parameter list in turn, all argument
    values that have been deduced are substituted into the parameter.

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
-   If the callee is a function value, the call is an initializing expression
    whose type is the substituted return type of the function. When evaluated,
    the call expression will invoke the function and produce whatever value it
    returns.
-   If the callee is a bound method value, it behaves the same as a function
    value, except that the `self` parameter of the called function is bound to
    the `self` value in the bound method value.

### Indirect calls and the `Call` interface

A generic parameter can be constrained to be a callable type using the `Call`
interface:

```carbon
interface Call(... each Args: type) {
  let Result:! type;
  fn Op(self, ... each args: each Args) -> Result;
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

The `Call` interface only models function calls for which arbitrary runtime
values of the given parameter types can be passed to the function. If the
signature of the function has compile-time parameters in its explicit argument
list, the function type will not implement `Call`.

```carbon
fn Runtime[T:! type](x: T);
fn CompileTime(T:! type, x: T);

fn Run() {
  // ✅ Calls `Runtime(0 as i32)`.
  Runtime.(Call(i32).Op)(0);
  // ❌ Can't call `CompileTime` this way, it can't implement `Call(type, i32)`
  // because the type would be passed at runtime.
  CompileTime.(Call(type, i32).Op)(i32, 0);
}
```

### Overloaded call operator

The `Call` interface can be implemented to overload the meaning of the function
call operator for a type.

```carbon
class Func(Arg:! type) {
  impl as Call((Arg,)) where .Result = () {
    fn Op[self: Self](arg: (Arg,)) { Print("hello, world"); }
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
  fn Op[self: Self](args: ()) -> i32 {
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

## References

-   Proposal
    [#438: Add statement syntax for function declarations](https://github.com/carbon-language/carbon-lang/pull/438)
-   Proposal
    [#826: Function return type inference](https://github.com/carbon-language/carbon-lang/pull/826)
-   Proposal
    [#2875: Functions, function types, and function calls](https://github.com/carbon-language/carbon-lang/pull/2875)
