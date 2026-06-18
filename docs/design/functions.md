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
    -   [Function signatures](#function-signatures)
        -   [Captures and function fields](#captures-and-function-fields)
        -   [Positional Parameters](#positional-parameters)
        -   [Return specification](#return-specification)
        -   [Unused parameters](#unused-parameters)
    -   [`return` statements](#return-statements)
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

## Overview

Functions are the core building block for applications. A function definition or
declaration has one of the following syntactic forms (where items in square
brackets are optional and independent):

-   `fn` _name_ [_implicit-parameters_] [_tuple-pattern_] `=>` _expression_ `;`
-   `fn` _name_ [_implicit-parameters_] [_tuple-pattern_] [`->` _return-form_] `{`
    _statements_ `}`
-   `fn` _name_ [_implicit-parameters_] [_tuple-pattern_] [`->` _return-form_] `;`

The first form is a shorthand: `=> expression ;` is equivalent to
`-> auto { return expression; }`. When a body is present (the first and second
forms), it is a function definition. The body introduces nested scopes which may
contain local variable declarations. A function with only a signature and no
body (the third form) is a forward declaration.

The syntax for parameters and returns is the same for functions and
[lambdas](lambdas.md#syntax-overview):

-   _implicit-parameters_: square brackets `[`...`]` enclosing default capture
    modes, explicit captures, function fields, or deduced parameters, see
    [lambdas](lambdas.md#implicit-parameters-in-square-brackets).
-   _tuple-pattern_: parentheses `(`...`)` enclosing a list of explicit
    parameter patterns, see
    [pattern matching](pattern_matching.md#pattern-syntax-and-semantics).

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

### Function signatures

#### Captures and function fields

Like lambdas, named function definitions support [captures](lambdas.md#captures)
and [function fields](lambdas.md#function-fields), with these restrictions:

-   They can only be used on functions where the definition is attached to the
    declaration (so they cannot be forward declared).
-   Captures and function fields are only supported on local function
    definitions immediately defined inside the body of another function. They
    are not supported on member functions of classes/interfaces.

#### Positional Parameters

Like lambdas, named function definitions support
[positional parameters](lambdas.md#positional-parameters), which are used when
the explicit parameter list is omitted. Like
[captures and function fields](#captures-and-function-fields), they may only be
used with function definitions and not forward declarations. In addition,
positional parameters can only be used in a context where there is exactly one
enclosing function or lambda that has no explicit parameter list.

#### Return specification

The return type of a function can be specified using a return clause (`->`), or
it can be deduced using a signature return expression (`=>`).

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
-   `=>` followed by an _expression_ defines a shorthand for a function body
    that returns the expression. The return type is deduced as if `-> auto` were
    used.
    -   For example, `fn Add(a: i64, b: i64) => a + b;` has a return type of
        `i64` based on the type of the expression `a + b`.
    -   Because the return type is deduced and not explicitly known, functions
        defined using `=>` cannot have a separate forward declaration.

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
