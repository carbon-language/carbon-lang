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
    -   [Captures, function fields, and positional parameters](#captures-function-fields-and-positional-parameters)
    -   [Specifying return type and return expressions](#specifying-return-type-and-return-expressions)
    -   [`return` statements](#return-statements)
    -   [Unused parameters](#unused-parameters)
-   [Function declarations](#function-declarations)
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

## Overview

Functions are the core building block for applications. A function definition or
declaration has one of the following syntactic forms (where items in square
brackets are optional and independent):

-   `fn` _name_ [_implicit-parameters_] [_tuple-pattern_] `=>` _expression_ `;`
-   `fn` _name_ [_implicit-parameters_] [_tuple-pattern_] [`->` _return-type_] `{`
    _statements_ `}`
-   `fn` _name_ [_implicit-parameters_] [_tuple-pattern_] [`->` _return-type_] `;`

The first form is a shorthand: `=> expression ;` is equivalent to
`-> auto { return expression; }`. When a body is present (the first and second
forms), it is a function definition. The body introduces nested scopes which may
contain local variable declarations. A function with only a signature and no
body (the third form) is a forward declaration.

The syntax for parameters and returns is the same for functions and
[lambdas](lambdas.md#syntax-overview):

-   _implicit-parameters_: square brackets `[`...`]` enclosing default capture
    modes, explicit captures, function fields, or deduced parameters
-   _tuple-pattern_: parentheses `(`...`)` enclosing a list of explicit
    parameter patterns.

## Function definitions

A basic function definition may look like:

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

### Captures, function fields, and positional parameters

Named function definitions support [captures](lambdas.md#captures),
[function fields](lambdas.md#function-fields), and
[positional parameters](lambdas.md#positional-parameters) in their signature and
body, with the following restrictions:

-   **Definition attached**: They can only be used on functions where the
    definition is attached to the declaration (so they cannot be forward
    declared).
-   **Scope limit**: Captures and function fields are only supported on local
    function definitions immediately defined inside the body of another
    function. They are not supported on namespace-scoped functions or member
    functions of classes/interfaces.
-   **Positional parameters**: Positional parameters can only be used in a
    context where there is exactly one enclosing function or lambda that has no
    explicit parameter list.

### Specifying return type and return expressions

The return type of a function can be specified using a return clause (`->`), or
it can be deduced using a return expression (`=>`):

-   `->` followed by an _expression_, such as `i64`, directly states the return
    type. This expression will be evaluated at compile-time, so must be valid in
    that context.
    -   For example, `fn ToString(val: i64) -> String;` has a return type of
        `String`.

*   `->` followed by the `auto` keyword indicates that
    [type inference](type_inference.md) should be used to determine the return
    type.
    -   For example, `fn Echo(val: i64) -> auto { return val; }` will have a
        return type of `i64` through type inference.
    -   Forward declarations must have a known return type, so `auto` is not
        valid.
    -   The function must have precisely one `return` statement. That `return`
        statement's expression will then be used for type inference.
*   Omission of `->` (when using `{` ... `}` body) indicates that the return
    type is the empty tuple, `()`.
    -   For example, `fn Sleep(seconds: i64);` is similar to
        `fn Sleep(seconds: i64) -> ();`.
    -   `()` is similar to a `void` return type in C++.
*   `=>` followed by an _expression_ defines a shorthand for a function body
    that returns the expression. The return type is deduced as if `-> auto` were
    used.
    -   For example, `fn Add(a: i64, b: i64) => a + b;` has a return type of
        `i64` based on the type of the expression `a + b`.
    -   Because the return type is deduced and not explicitly known, functions
        defined using `=>` cannot have a separate forward declaration.

> **TODO:** Update this section to cover return forms, as discussed
> [here](values.md#function-calls-and-returns).

### `return` statements

The [`return` statement](control_flow/return.md) is essential to function
control flow. It ends the flow of the function and returns execution to the
caller.

When the [return clause](#specifying-return-type-and-return-expressions) is
omitted, the `return` statement has no expression argument, and function control
flow implicitly ends after the last statement in the function's body as if
`return;` were present.

When the return clause is provided, including when it is `-> ()`, the `return`
statement must have an expression that is convertible to the return type, and a
`return` statement must be used to end control flow of the function.

> **TODO:** Update this section to cover the requirements on the form of the
> expression.

### Unused parameters

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

As specified in
[Matching redeclarations](/proposals/p003763-matching-redeclarations.md),
`unused` markers may only appear on definitions, not on non-defining
declarations. The names of parameters must match between redeclarations, but the
presence of the `unused` marker does not need to match.

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
    [#2022: Unused Pattern Bindings (Unused Function Parameters)](https://github.com/carbon-language/carbon-lang/pull/2022)
-   Proposal
    [#2875: Functions, function types, and function calls](https://github.com/carbon-language/carbon-lang/pull/2875)
-   Proposal
    [#3762: Merging forward declarations](https://github.com/carbon-language/carbon-lang/pull/3762)
-   Proposal
    [#3763: Matching redeclarations](https://github.com/carbon-language/carbon-lang/pull/3763)
-   Proposal
    [#3848: Lambdas](https://github.com/carbon-language/carbon-lang/pull/3848)
