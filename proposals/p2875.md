# Functions, function types, and function calls

<!--
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

[Pull request](https://github.com/carbon-language/carbon-lang/pull/2875)

<!-- toc -->

## Table of contents

-   [Abstract](#abstract)
-   [Problem](#problem)
-   [Background](#background)
-   [Proposal](#proposal)
-   [Details](#details)
    -   [Functions and function types](#functions-and-function-types)
    -   [Bound methods](#bound-methods)
    -   [Call syntax](#call-syntax)
    -   [Direct calls](#direct-calls)
    -   [Generic callable parameters](#generic-callable-parameters)
    -   [Function types and `Call`](#function-types-and-call)
    -   [Overloaded call operator](#overloaded-call-operator)
-   [Rationale](#rationale)
-   [Alternatives considered](#alternatives-considered)
    -   [Signature-based function types](#signature-based-function-types)
    -   [Make direct and indirect calls behave uniformly](#make-direct-and-indirect-calls-behave-uniformly)
-   [Future work](#future-work)
    -   [Overloading](#overloading)
    -   [Overloading on expression category and phase](#overloading-on-expression-category-and-phase)
    -   [Variadics](#variadics)
    -   [Lambdas](#lambdas)
    -   [Function pointers](#function-pointers)

<!-- tocstop -->

## Abstract

Specify the behavior of function calls and the type and behavior of the entity
introduced by a function declaration.

## Problem

Functions are a primitive building block through which we define Carbon's
execution semantics, but we don't currently have a specification for what a
function is in Carbon nor how function calls work. We have open questions around
whether Carbon has first-class function types, how types can overload the
function call operation, and how callable entities such as the name of a
parameterized class fit into the picture -- are they functions or something
else?

## Background

This proposal assumes the reader understands how functions and function pointers
work in C or C++.

## Proposal

A function declaration in Carbon introduces a new, unique, stateless type,
called a _function type_. The function name is bound to a value of that function
type.

Function calls use C-like syntax: an expression naming a callable is followed by
a syntax resembling a tuple of arguments. There are several kinds of callable:

-   Functions, and more generally
    [values of function types](#functions-and-function-types).
-   [Bound methods](#bound-methods), such as `my_vector.Begin`.
-   Lambdas, once they are introduced to Carbon.
-   Parameterized entities, such as a generic class `Vector` or a generic
    interface `AddWith`.
-   Values of dependent types that are
    [constrained to be callable](#generic-callable-parameters).
-   User-defined class types that
    [overload function call syntax](#overloaded-call-operator).

In the first four cases, argument deduction is used to determine the values of
deduced arguments, and then pattern matching is used to bind parameter patterns
to argument values.

In the remaining cases, the built-in `Call` interface is used to determine the
semantics of the call, and the call `F(args)` is translated into
`F.(Call(ArgTypes).Op)(args)`. Note that this is itself a bound member function
call, and `Call(ArgTypes)` is a call to a parameterized entity, so this
transformation is only performed once.

## Details

### Functions and function types

A function declaration such as

```
fn F[T:! type](x: T) -> T { return x; }
```

introduces a value named `F`, whose type is a unique _function type_. Distinct
functions have distinct function types, even if they have the same signature. A
function type is an empty, trivial type. There is no way to name a function type
other than asking for the type of the function value.

```
// Compile-time function.
fn TypeOf[T:! type](x: T) -> type { return T; }

// ✅ `F` is a first-class value with a first-class type.
let template FType:! type = TypeOf(F);
var my_f: FType = F;
```

Function values are regular values that can be stored in variables, passed to
functions, and so on.

```
fn G() -> i32 {
  // ✅ `my_f` has function type `FType`. This is a direct call to `F`.
  return my_f(1);
}

fn Sort[template F:! type, T:! type](v: Vector(T)*, f: F);
fn Compare(a: i32, b: i32) -> Ordering { return a.(Ordered.Compare)(b); }
fn SortInts(v: Vector(i32)*) { Sort(v, Compare); }
```

For the purpose of the
[orphan rule](/docs/design/generics/details.md#orphan-rule), a function type is
considered to be declared by the function declaration that introduces the
function value.

### Bound methods

For each function type corresponding to a method, there is a corresponding
_bound method type_. When a member access is performed on an object of class
type to access a method, the result is a _bound method value_ of bound method
type. A bound method type describes the callee in a method call, and a bound
method value describes the `self` parameter of the call.

```
class HasMember {
  // `HasMember.F` has a stateless function type, with signature
  // `[self: Self](n: i32) -> i32`.
  fn F[self: Self](n: i32) -> i32;
}
fn F(h1: HasMember, h2: HasMember) -> i32 {
  // ✅ `h1.F` is a bound method value whose type is a bound method type,
  // with signature `(n: i32) -> i32`.
  var hf: auto = h1.F;
  // ✅ `h1.F` and `h2.F` are of the same bound method type.
  hf = h2.F;
  // ✅ Same as `h2.F(4)`.
  return hf(4);
}
```

### Call syntax

Calls take the form `a(b, c, d)` or `a(b, c, d,)`, where:

-   `a` is the callee, which can be a name, a literal, a member access, or some
    more complex expression enclosed parentheses.
-   `b`, `c`, `d` are any number of argument expressions. Arguments are
    separated by commas, and if the argument list is not empty, an optional
    trailing comma is permitted but not required after the final argument.

Call syntax is syntactically equivalent to a primary expression followed by a
tuple literal, except that a tuple literal requires a trailing comma to form a
single-element tuple `(b,)`, whereas in call syntax both `a(b)` and `a(b,)` are
permitted.

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

### Generic callable parameters

A generic parameter can be constrained to be a callable type using the `Call`
interface:

```
interface Call(Args: type) {
  let Result:! type;
  fn Op[self: Self](args: Args) -> Result;
}
```

> **TODO:** `Call` should be variadic. For now, we model it as taking a tuple
> type, and we model `Call.Op` as taking a tuple value.

For example:

```
fn Sort[T:! type, F:! Call((T, T)) where .Result = Ordering]
       (v: Vector(T)*, cmp: F) {
```

A call expression that is not a direct call is translated into an invocation of
`Call(Args).Op`, where `Args` is the type of the call's argument tuple.

```
  // In Sort...
  auto ord: auto = cmp((*v)[i], (*v)[j]);
  // ... is translated into ...
  auto ord: auto = cmp.(Call((T, T)).Op)((*v)[i], (*v)[j]);
```

Note that the types of the call arguments are modeled as an interface parameter.
This permits the call interface to model function overloading. However, the
return type is an associated type, not a parameter -- we do not permit
overloading on return types, and we don't want type information to propagate
from the context in which a call expression appears inwards to the call.

### Function types and `Call`

A function type or bound method type implements the `Call` interface for every
set of runtime argument types that a direct call to the function or bound method
would accept. The behavior of `Call.Op` is to call the function or bound method
with the provided argument list.

```
fn Select[T:! type](b: bool, x: T, y: T) -> T {
  return if b then x else y;
}

// Generated:
impl forall [T:! type, BType:! ImplicitAs(bool)]
    Select as Call((BType, T, T)) where .Result = T {
  fn Op[self: Self](args: (BType, T, T)) {
    let (b: bool, x: T, y: T) = args;
    return Select(b, x, y);
  }
}
```

Implicit conversions are permitted for parameters whose types do not involve
deduced parameters. The intent is for the `impl` to support indirect calls in
the same cases where the function supports direct calls, with the same meaning.

```
fn TakeI32Fn[F:! Call(i32)](f: F);
fn I64Fn(n: i64);
fn Run() {
  // ✅ `I64Fn` can be called with an `i32`, because
  // `i32 impls ImplicitAs(i64)`.
  TakeI32Fn(I64Fn);
}
```

> **Note:** A call made using the `Call` interface always takes its arguments as
> value expressions and the call is always an initializing expression. If the
> function takes its arguments using `var`, or if a language mechanism is added
> that permits a direct function call to not be an initializing expression,
> additional conversions will be required. The `Call` interface is treated as
> not being implemented in cases where those conversions are not possible, for
> example because an argument or return value is not copyable. If we add a
> mechanism to allow an `impl` to specify a different expression category for
> its parameters or result than the one in the interface, for example by taking
> a `var` parameter where the interface took a `let` parameter, that would
> automatically also be available for overloaded function calls.
>
> We anticipate revisiting these constraints in a future proposal, and expanding
> the function call interface to provide the same generality that is provided by
> `fn` declarations. This is described more in the
> [future work](#overloading-on-expression-category-and-phase) section.

The `Call` interface only models function calls for which arbitrary runtime
values of the given parameter types can be passed to the function. If the
signature of the function has compile-time parameters in its explicit argument
list or has any refutable pattern matching between the call arguments and
function parameters, the function type will not implement `Call`.

```
fn Runtime[T:! type](x: T);
fn CompileTime(T:! type, x: T);
// 🤷 Undecided whether this is valid.
fn RefutablePattern(1 as i32);

fn Run() {
  // ✅ Calls `Runtime(0)`.
  Runtime.(Call(i32).Op)(0);
  // ❌ Can't call `CompileTime` this way, it can't implement `Call(type, i32)`
  // because the type would be passed at runtime.
  CompileTime.(Call(type, i32).Op)(i32, 0);
  // ❌ Can't call `RefutablePattern` this way, it can't implement `Call(i32)`
  // for arbitrary i32 arguments.
  RefutablePattern.(Call(i32).Op)(0);
}
```

### Overloaded call operator

The `Call` interface can be implemented to overload the meaning of the function
call operator for a type.

```
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

```
class X { var n: i32; }
// ✅ OK, but inadvisable.
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

There is no way to directly define a function-like callable class type that
takes a compile-time argument. This is similar to the situation in C++ where
there is no way to define an `operator()` for a value `x` of class type so that
`x<T>()` passes `T` to `operator()` at compile time. However, this can be worked
around by putting the compile-time value in the type of an argument instead:

```
class Wrap(T:! type) {}
class Callable {
  impl forall [T:! Printable] as Call((Wrap(T), T)) where .Result = () {
    fn Op[self: Self](args: (Wrap(T), T)) {
      let (_: auto, v: T) = args;
      Print(v);
    }
  }
}
fn CallIt[F:! Call(Wrap(i32), i32)](f: F) {
  f({} as Wrap(i32), 0);
}
fn Run() {
  CallIt({} as Callable);
}
```

## Rationale

Principles:

-   [Principle: one static open extension mechanism](/docs/project/principles/static_open_extension.md).
    -   Overloaded calls are supported by implementing an interface.
-   [Principle: Prefer providing only one way to do a given thing](/docs/project/principles/one_way.md).
    -   There is only one obvious way to pass a callable to a generic function,
        and it works efficiently whether the argument is a function, a callable
        object, or (eventually) a lambda.

Goals:

-   [Performance-critical software](/docs/project/goals.md#performance-critical-software)
    -   Passing around functions is no less efficient than passing around
        callable objects.
-   [Code that is easy to read, understand, and write](/docs/project/goals.md#code-that-is-easy-to-read-understand-and-write)
    -   Function types in C and C++ are notoriously hard to read. We avoid this
        problem by not having signature-based function types.
-   [Interoperability with and migration from existing C++ code](/docs/project/goals.md#interoperability-with-and-migration-from-existing-c-code)
    -   As [future work](#function-pointers) we have the basis for a design for
        function pointers. C++ function pointers can be modeled in Carbon code
        as values that implement `Call`, as can member function pointers.

## Alternatives considered

### Signature-based function types

We could give each function signature a distinct type, as is done in C and C++.
This would provide a story for [function pointers](#function-pointers) that does
not rely on generics or type erasure or fancy representation optimizations.

The main down side of this approach is one we see frequently in C++: passing a
function to a template generic would be less efficient than passing a function
object to the same template. For example, in C++, given

```
std::vector<int> v;
bool cmp(int a, int b) { return simple_calculation(a, b); }
```

... calling `ranges::sort(v, cmp)` may be much less efficient than calling
`ranges::sort(v, [](int a, int b) { return cmp(a, b); })`, because the latter
performs a direct call and the former passes a function pointer, resulting in an
indirect call.

Making functions result in unique types means that calls to functions, lambdas,
and function-like objects have semantics that are more similar and have similar
efficiency properties.

### Make direct and indirect calls behave uniformly

It would be desirable to remove the difference between direct and indirect
calls. Unfortunately, that doesn't seem to be practical, for a number of
reasons:

-   Indirect calls need to be transformed into another call expression. We need
    a foundation for our semantics at some point, where we make an actual
    function call rather than transforming one function call into another.
-   We have chosen that
    [interfaces are the only static open extension mechanism](/docs/project/principles/static_open_extension.md).
    This means that any general mechanism we provide to overload function calls
    must fit within the boundaries of Carbon's `interface` and `impl`
    mechanisms.
-   We could try to support passing compile-time parameters to functions by
    specifying in the `impl` query whether each argument can be passed at
    compile time, but any such query will need to be able to fall back to
    passing each argument at runtime, which can result in an exponential-time
    search for a matching `impl`.

## Future work

### Overloading

We intend to support function overloading, and although it outside the scope of
this proposal, it is important to ensure that we have reasonable paths forward
to support overloading within this framework.

An overloaded function has multiple different signatures -- sets of implicit and
explicit parameters -- that will be checked against the provided arguments. In
this proposal, that means that a set of overloaded functions introduces a single
function type that supports being called in different ways.

Depending on how we approach overloading, overload selection might be done by
checking each candidate in turn until we find one that matches, or checking them
all and somehow selecting the best match, and our current intent is to use the
former approach. Translated into this proposal, one important consequence is
that the function type corresponding to the set of overloaded functions has
multiple parameterized implementations of the `Call(...)` interface, and the
implementation selected for a particular call will need to follow whatever rules
the overloading mechanism picks. For example, if function overloading picks the
first matching function, then given the following, using placeholder syntax:

```
overloaded fn Abs[T:! Unsigned](x: T) -> T { return x; }
overloaded fn Abs[T:! Floating](x: T) -> T { return x < 0 ? -x : x }
```

we would synthesize implementations that behave as follows:

```
match_first {
  impl forall [T:! Unsigned] Abs as Call((T,)) where .Result = T { ... }
  impl forall [T:! Floating] Abs as Call((T,)) where .Result = T { ... }
}
```

For a type that is `Unsigned & Floating`, the former `impl` will be selected,
matching the behavior of overload selection.

### Overloading on expression category and phase

We should consider whether calls can be overloaded on the expression category
and expression phase of the callee and arguments. For compile-time arguments, we
may also wish to support overloading on the constant value.

Other languages support overloading on the expression category of the callable
object. Rust provides separate `Fn`, `FnMut`, and `FnOnce` to distinguish
between different ways of passing the callable object into an overloaded
function call, and C++ permits `operator()` to take `*this` as `const`,
non-`const`, or even `&&`.

In Carbon, we could follow the Rust approach and provide multiple `Call`
interfaces to handle different kinds of `self` parameter, modeled in a similar
way to the `IndexWith` and `IndirectIndexWith` interfaces used for subscripting.
A more general, ambitious and ergonomic approach that we have started exploring
would be to allow the function call signature to be described as part of the
call interface:

```carbon
impl MyCallable as call(T:! type, x: T) -> T;
```

This might be a shorthand syntax for some way of expressing the signature as an
interface, or a new form of first-class language primitive.

The options in this space are currently being explored in issue
[#3154](https://github.com/carbon-language/carbon-lang/issues/3154).

### Variadics

Once Carbon supports variadics, the overloaded call mechanism should be revised
to make use of them, instead of using a tuple type to approximate a variadic
parameter list.

### Lambdas

A lambda is expected to behave much like a function under this proposal, with
the primary difference being that lambdas can be stateful, whereas functions are
stateless.

### Function pointers

For interoperability with C++, and for purposes of cheaply storing references to
stateless functions, function pointers should be supported. A function pointer
could be modeled as:

```
alias FunctionPtr(Args:! type, Result:! type) =
    DynPtr(Stateless & Call(Args) where .Result = Result);
```

where:

-   `Stateless` is an interface describing types that are empty, do not have a
    notion of identity, and for which instances can be created and destroyed
    trivially.
-   `DynPtr` is a
    [type that performs type erasure and runtime dispatch](/docs/design/generics/details.md#dynamic-types)
    for a given facet type.
-   `DynPtr` has an optimization that `DynPtr(Stateless & I)`, where `I` is an
    interface with exactly one associated function, is represented as a pointer
    to that function.
