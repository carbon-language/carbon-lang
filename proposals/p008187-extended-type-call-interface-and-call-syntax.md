# Extended-type call interface and `call` syntax

<!--
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

[Pull request](https://github.com/carbon-language/carbon-lang/pull/8187)

<!-- toc -->

## Table of contents

-   [Abstract](#abstract)
-   [Problem](#problem)
-   [Background](#background)
-   [Summary](#summary)
-   [Proposal](#proposal)
    -   [Primitive call interface](#primitive-call-interface)
    -   [`call`](#call)
    -   [`Core.Call` named constraint](#corecall-named-constraint)
-   [Details](#details)
    -   [Examples using `call`](#examples-using-call)
    -   [Conversions when calling a function argument](#conversions-when-calling-a-function-argument)
    -   [Representing deduced parameters](#representing-deduced-parameters)
    -   [Representing explicit compile-time parameters](#representing-explicit-compile-time-parameters)
    -   [Callables that may or may not mutate](#callables-that-may-or-may-not-mutate)
    -   [Forwarding](#forwarding)
        -   [Forwarding tuple parameters](#forwarding-tuple-parameters)
-   [Future work](#future-work)
    -   [Impl prioritization](#impl-prioritization)
    -   [Compile-time calls](#compile-time-calls)
-   [Rationale](#rationale)
-   [Alternatives considered](#alternatives-considered)
    -   [Drop `Core.Call` and rename `CallPrimitive` to `Call`](#drop-corecall-and-rename-callprimitive-to-call)
    -   [Put `self` inside `(...)` in `call` syntax](#put-self-inside--in-call-syntax)
    -   [Related alternatives in other proposals](#related-alternatives-in-other-proposals)

<!-- tocstop -->

## Abstract

We replace the `Core.Call` interface with a new `Core.CallPrimitive` extended
type interface that supports `val`, `ref`, `var`, and compile-time parameters
and returns, as well as mutating `ref self` callables. We introduce a
`call(`...`)` / `call[val/ref/var](`...`)` facet type sugar syntax for concisely
writing `Core.CallPrimitive` constraints and `impl`s, redefine `Core.Call` as a
type-oriented named constraint for progressive disclosure, and show how
parameterized `impl`s of `CallPrimitive` model deduced parameters, explicit
compile-time parameters, and full-fidelity argument forwarding (`Apply`).

## Problem

The `Core.Call` interface defined in
[proposal #2875](https://github.com/carbon-language/carbon-lang/pull/2875) does
not support all of the features of function calls:

-   `var` and `ref` parameters;
-   mutation of the callable object using `ref self`;
-   compile-time parameters, as designated by `generic`, `template`, or deduced
    `[`...`]` parameter lists; and
-   compile-time parameters that are mentioned elsewhere in the signature.

In addition, we would like there to be a way to forward arguments passed to one
function to another function it calls, preserving category and phase. For
example, there should be a way to write
[`Apply`](https://en.wikipedia.org/wiki/Function_application) that takes a
function and arguments to call it with and performs the call (so `Apply(F, x)`
returns `F(x)`).

Ideally, our call operator interface would support:

-   **Uniformity**: ("consistent and universal") Ordinary functions, lambdas,
    and bound methods are all values of types that implement the call operator
    interface with the relevant signature. This is to enable a single constraint
    on a parameter to accept any of these. This also implies that implementing
    the call operator interface is sufficient to express the full range of
    function signatures, including `var` and `ref` parameters.
-   **Functors**: Types may implement the interface to be callable using the
    normal function-calling syntax, like implementing `operator()` in C++. A
    closure is syntactic sugar for declaring and creating a value of such a
    type.
-   **Overloading**: There is a relatively convenient way to define a function
    value whose type has multiple implementations of the "call operator"
    interface.
-   **Dispatch**: There is a method of matching actual call invocations to
    implementations of the "call operator." It should be as similar as possible
    to other cases of `impl` selection, including supporting the same
    `match_first` `impl` prioritization system (or whatever we replace that
    with). However, it also needs to be able to support the expected implicit
    conversions from the argument values to the parameter types.
-   **Subsumption**: An overload set containing two functions, one accepting a
    `bool` and one accepting an `i32`, should be able to be passed as an
    argument to a function requiring a callable accepting a `bool`. This same
    mechanism should allow a generic function with signature that could accept a
    `bool`, such as `fn F[T: type](x: T)`, to be passed to the function
    requiring a callable accepting a `bool`, using the instantiation with
    `T == bool`.
-   **Closed**: There is a way to say "just these implementations and no
    others." This should be the default when defining ordinary functions and
    function overloads.

However, fully addressing the last four of these properties is out of scope of
this proposal, since it won't address declaring overloaded functions and `impl`
prioritization. We still would like the decisions in this proposal to be
consistent with a future solution that achieves those properties.

## Background

This proposal was split out of
[proposal #5389](https://github.com/carbon-language/carbon-lang/pull/5389);
earlier revision and review history may be found in that pull request.

This proposal depends on pending proposals also split out of
[proposal #5389](https://github.com/carbon-language/carbon-lang/pull/5389):

-   [proposal #8183: First-class extended types and `fwd`](https://github.com/carbon-language/carbon-lang/pull/8183)
    (`Core.ExtType`, `exttype(...)`, `exttypeof(...)`, `fwd` bindings,
    arguments, and returns, `Core.NoVarExtType`, and exact-value parameters),
-   [proposal #8185: Extended type interfaces and named constraint extensions](https://github.com/carbon-language/carbon-lang/pull/8185)
    (extended type interfaces, `Core.ValSelf`/`RefSelf`/`VarSelf`),
-   [proposal #8186: Interface variance (`implicit_into` and `implicit_from`)](https://github.com/carbon-language/carbon-lang/pull/8186)
    (`implicit_into`, `implicit_from`, `anchor`, and adapter facets), and
-   [proposal #8189: Associated constants in named constraints](https://github.com/carbon-language/carbon-lang/pull/8189)
    (used to define the `Core.Call` named constraint).

This proposal also builds upon:

-   [Proposal #2875: Functions, function types, and function calls](https://github.com/carbon-language/carbon-lang/pull/2875),
    which defines the previous `Core.Call` interface.
-   [Proposal #7016: Updating `self` syntax and adding `static` member variables](https://github.com/carbon-language/carbon-lang/pull/7016).
-   [Function overloading document](https://docs.google.com/document/d/1rTsDJ_ZNENH0mhi5E64MGjKJ29OYiI_dzBRz2GzYySU/edit?resourcekey=0-z6_xQ9xuJ5CZ6qJqiuSxuw&tab=t.0)
    with the initial ideas on how to support function overloading in Carbon.

## Summary

In short, this proposal introduces:

-   a change to the ...`(`...`)` function call operator. It now uses the
    `Core.CallPrimitive` extended type interface, instead of the `Core.Call`
    interface.
-   a `call` sugar syntax (`call(...)` and `call[val/ref/var](...)`) that makes
    it easier to specify a `Core.CallPrimitive` facet type with a syntax similar
    to a function declaration instead of using extended type literals.
-   redefining `Core.Call` as a named constraint that uses types to specify the
    call as the interface did before.
-   modeling deduced and explicit compile-time parameters using parameterized
    `impl`s of `Core.CallPrimitive`, enabling subsumption and full-fidelity
    argument forwarding (`Apply`).

## Proposal

### Primitive call interface

Combining extended types with variadics allows us to represent a more general
version of the `Call` interface that allows specifying extended types:

```carbon
package Core;

interface CallPrimitive(
    implicit_into anchor Self: ExtType,
    ... implicit_into each Param: ExtType) {
  let implicit_from ResultExtType: ExtType;
  fn Op(fwd self: Self, ... fwd each args: each Param)
      -> fwd ResultExtType;
}
```

Note that `Self` is variant, since we want to allow a `val T` `impl` to be used
for a `ref T` object. The variance uses the `anchor` modifier on `implicit_into`
(from
[proposal #8186](https://github.com/carbon-language/carbon-lang/pull/8186)) so
the _type_ of `Self` is not variant (for implicit conversions of `self` on
methods by way of member binding, see
[proposal #8188](https://github.com/carbon-language/carbon-lang/pull/8188)),
just the other aspects of the extended type.

### `call`

It can be cumbersome to write out extended type values for the arguments to this
interface, though, and we expect this to be the most common use case that would
otherwise require writing extended type values. So we introduce a `call(`...`)`
convenience syntax for writing a facet type of the `Core.CallPrimitive`
interface.

| `call` syntax           | `Core.ExtType` facet type value                           |
| :---------------------- | :-------------------------------------------------------- |
| `call()`                | `CallPrimitive()`                                         |
| `call(_: i32)`          | `CallPrimitive(exttype(val i32))`                         |
| `call(_: i32, _: bool)` | `CallPrimitive(exttype(val i32), exttype(val bool))`      |
| `call(var _: bool)`     | `CallPrimitive(exttype(var bool))`                        |
| `call(fwd _: X)`        | `CallPrimitive(X)`                                        |
| `call(... _: each T)`   | `CallPrimitive(... exttype(val each T))`                  |
| `call() -> i32`         | `CallPrimitive() where .ResultExtType = exttype(var i32)` |
| `call() -> val i32`     | `CallPrimitive() where .ResultExtType = exttype(val i32)` |
| `call() -> ()`          | `CallPrimitive() where .ResultExtType = exttype(var ())`  |
| `call() -> fwd R`       | `CallPrimitive() where .ResultExtType = R`                |

Note: If "`->` _\<return-type\>_" is not included at the end, the return type is
unconstrained. Use `-> ()` to mean "returns nothing."

Note: In `call(`...`)`, `var` functions as a modifier on a binding rather than
an independent pattern operator, so `call(var (_: bool, _: i32))` is not valid;
that signature would be written `call(var _: (bool, i32))` instead.

Observe that `CallPrimitive` is a `Core.ExtType` interface, so the above
expressions represent `Core.ExtType` facet types. In addition, you may specify
the non-type aspects of the extended type of `self` by writing `[`...`]` between
`call` and the `(` of the parameter list to produce an ordinary `type` facet
type, as in:

| `call` syntax | facet type value           |
| :------------ | :------------------------- |
| `call[val]()` | `ValSelf(CallPrimitive())` |
| `call[ref]()` | `RefSelf(CallPrimitive())` |
| `call[var]()` | `VarSelf(CallPrimitive())` |

> **FIXME: QUESTION:** Should this be changed to reflect the fact that
> [proposal #7016: "Updating `self` syntax and adding `static` member variables"](https://github.com/carbon-language/carbon-lang/pull/7016)
> changed the method syntax to put `self` inside the explicit parameter list in
> round parens `(`...`)`?

These can be combined, so `call[ref](var _: bool, _: i32*) -> ref i32` gives the
facet type:

```carbon
RefSelf(
  CallPrimitive(exttype(var bool),
                exttype(val i32*))
      where .ResultExtType = exttype(ref i32))
```

See [examples in the "Details" section](#examples-using-call).

This was discussed on
[2025-05-08](https://docs.google.com/document/d/1Yt-i5AmF76LSvD4TrWRIAE_92kii6j5yFiW-S7ahzlg/edit?tab=t.0#heading=h.vdognq1upsf5).

### `Core.Call` named constraint

We can use named constraints with associated constants (from
[proposal #8189](https://github.com/carbon-language/carbon-lang/pull/8189)) to
define `Core.Call` as a named constraint that acts like the interface before
this proposal, for use cases where the extra control over extended types is not
needed.

```carbon
package Core;

constraint Call(... each Param: type) {
  let Result: type;
  extend require impls
      call[val](... _: each Param) -> Result;
  // Equivalent to:
  extend require impls
      ValSelf(
        CallPrimitive(... each exttype(val Param)))
      where .ResultExtType = exttype(var Result);
}
```

> FIXME: **QUESTION:** Is this redundant with providing `call`? Should instead
> this be dropped and `CallPrimitive` be renamed `Call`? Or do we want a
> non-primitive version to parallel other operator interfaces?

## Details

### Examples using `call`

The `call` syntax may be used as the facet type in an `impl` to define the call
operator for a type, as in:

```carbon
class OverloadedCallable {
  var x: i32;
}

// First impl: as a type facet type.
impl OverloadedCallable as
    call[ref](_: i32) -> () {
  fn Op(ref self, d: i32) -> () {
    self.x += d;
  }
}

// Second impl: as a `Core.ExtType` facet type.
// Equivalent to:
//   impl exttype(val OverloadedCallable) as
//       CallPrimitive()
//       where .ResultExtType = exttype(var i32)
impl exttype(val OverloadedCallable) as
    call() -> i32 {
  // `Self` is a `Core.ExtType` value equal to
  // `exttype(val OverloadedCallable)` so
  // `fwd self: Self` is the same as
  // `self: OverloadedCallable`.
  fn Op(fwd self: Self) -> i32 {
    return self.x;
  }
}

var c: OverloadedCallable = {.x = 1};

// Uses second impl.
Assert(c() == 1);
// Uses first impl.
c(2);
// Uses second impl.
Assert(c() == 3);
```

These facet types may also be used as a constraint for a generic parameter:

```carbon
// `call` followed by `[ref]` creates an
// ordinary type facet type
fn InvokeRef[T: call[ref](_: i32) -> ()]
    (ref f: T, arg: i32) {
  f(arg);
}

// `call` without `[`...`]` creates a
// `Core.ExtType` facet type. Category of `f`
// will be deduced from the argument.
fn InvokeExtType[PET: call() -> i32]
    (fwd f: PET) -> i32 {
  return f();
}

// Uses first impl.
InvokeRef(ref c, 4);
// Uses second impl. Must be `ref` to
// match the category of `c`, which is
// used to determine the `InvokeExtType`
// parameter.
Assert(InvokeExtType(ref c) == 7);
```

Note that the `ref` marker on arguments follows whether the category of the
parameter was eventually determined to be `ref`. `InvokeExtType(c)` would be an
error instead of using value category (see the "`ref` on an argument controls
its category" alternative in
[proposal #8183](https://github.com/carbon-language/carbon-lang/pull/8183)).

A `call` expression generates a constraint that matches the corresponding lambda
expression with `fn` instead of `call` and names for parameters instead of `_`:

```carbon
var i: i32 = 0;
var lambda1: auto =
    fn (x: i32) -> () { i += x; };
InvokeRef(ref lambda1, 1);
let lambda2: auto =
    fn () -> i32 { return i; };
InvokeExtType(lambda2);
```

### Conversions when calling a function argument

Because `Core.CallPrimitive` marks its parameters `implicit_into` (using
interface variance from
[proposal #8186](https://github.com/carbon-language/carbon-lang/pull/8186)),
indirect calls through a function parameter support the same implicit
conversions as ordinary function calls:

```carbon
fn IntArg(x: i32);

fn ConcreteCall() {
  // ✅ Performs implicit conversion from
  // `Core.IntLiteral` to `i32`.
  IntArg(7);
}

fn GenericCall[FNT: call[val](_: i32)](f: FNT) {
  // Good to get the same conversion here, even
  // though we don't (and can't) have a blanket
  // `impl` that `f` accepts anything that
  // implicitly converts to `i32`.
  f(7);
}

GenericCall(IntArg);
```

### Representing deduced parameters

Observe that deduced parameters do not show up in the
[`Core.CallPrimitive` interface](#primitive-call-interface). They are not
modeled as part of the interface, but using parameterized `impl`s of that
interface. For example:

```carbon
fn Ident[T: type](x: T) -> T;
```

can be modeled as:

```carbon
class __Ident_Type {}
let generic Ident: __Ident_Type = {};
impl forall [T: type]
    exttype(val __Ident_Type)
    as CallPrimitive(exttype(val T))
    where .ResultExtType = exttype(var T);
```

or equivalently:

```carbon
class __Ident_Type {}
let generic Ident: __Ident_Type = {};
impl forall [T: type] __Ident_Type
    as call[val](_: T) -> T;
```

This partially addresses the ["subsumption" problem](#problem), since this
`Ident` could be passed to this function:

```carbon
fn GenericCall[FNT: call[val](_: i32)](f: FNT);
```

This will succeed since when it looks to see if `__Ident_Type` implements
`call[val](_: i32)`, it will find that the parameterized `impl` does provide
that when `T` is `i32`.

### Representing explicit compile-time parameters

Explicit compile-time parameters are modeled the same way as deduced parameters,
except they also are included in the list of parameter extended types passed to
`CallPrimitive`, using `exttypeof` (from
[proposal #8183](https://github.com/carbon-language/carbon-lang/pull/8183)).

For example:

```carbon
fn F(generic T: type, x: T) -> T;
```

can be modeled as:

```carbon
class __F_Type {}
let generic F: __F_Type = {};
impl forall [T: type] exttype(val __F_Type)
    as CallPrimitive(exttypeof(T),
                     exttype(val T))
    where .ResultExtType = exttype(var T);
```

or equivalently:

```carbon
class __F_Type {}
let generic F: __F_Type = {};
impl forall [T: type] __F_Type as
    call[val](fwd _: exttypeof(T), _: T) -> T;
```

An explicit `template` parameter, such as `fn H(template T: type, x: T) -> T;`,
is modeled the same way using `impl forall [template T: type]` with the same
`CallPrimitive(exttypeof(T), exttype(val T))` interface arguments. Per
[leads issue #2153](https://github.com/carbon-language/carbon-lang/issues/2153),
this accepts both template and symbolic compile-time arguments satisfying
`type`, while giving `T` template phase inside the `impl` so dependent parameter
extended types, `.ResultExtType`, and the `Op` body are checked at
monomorphization.

### Callables that may or may not mutate

In many cases, a function accepting a callable object will not care about
whether the call mutates the object.

Consider an example of two callables, one that has no mutation, and one that
does:

```carbon
fn CallbackNoMutation() -> i32 {
  // https://xkcd.com/221/
  return 4;
}

class Counter {
  var count: i32;
  impl as call[ref]() -> i32 {
    fn Op(ref self) -> i32 {
      self.count += 1;
      return self.count;
    }
  }
}
var counter: Counter = {.count = 0};
```

There are two approaches to accommodating this that are supported by the current
design. The function taking the callable object can be generic on its aspect in
the caller, as in:

```carbon
fn GenericCaller[PET: call() -> i32]
    (fwd f: PET) -> i32 {
  return f();
}

Assert(GenericCaller(CallbackNoMutation) == 4);
Assert(GenericCaller(ref counter) == 1);
```

Alternatively, we can have the function take its argument by value, and make a
wrapper that holds a pointer to make the call:

```carbon
// Only has to be defined once
class CalledThroughPointer(T: type) {
  var p: T*;
}

fn CallThroughPointer[T: type](p: T*)
    -> CalledThroughPointer(T) {
  return {.p = p};
}

impl forall
    [... each P: Core.ExtType,
     // Or: `T: call[ref](... fwd _: each P)`
     T: Core.RefSelf(Core.CallPrimitive(... each P))]
    exttype(val CalledThroughPointer(T)) as
    Core.CallPrimitive(... each P) {
  where ResultExtType = T.ResultExtType;
  fn Op(self, ... fwd each param: each P) -> fwd ResultExtType {
    return (*self.p)(... fwd ~each param);
  }
}

// Or: `FNT: Core.Call() where .Result = i32`
fn ValueCaller[FNT: call[val]() -> i32](f: FNT) -> i32 {
  return f();
}

Assert(ValueCaller(CallbackNoMutation) == 4);
Assert(ValueCaller(CallThroughPointer(&counter)) == 2);
```

NOTE: The `~` is there in anticipation of using moves to support forwarding, see
[the "Forwarding" section](#forwarding).

The first, generic approach is a bit more convenient but means using extended
types. The second approach only uses types once `CallThroughPointer` has been
defined, but is less convenient. Some may prefer the second approach since it
makes it explicit when the callable object may be mutated.

### Forwarding

Forwarding is where an argument is passed along generically, including aspects
of its extended type like category and phase. Input-to-input forwarding is where
the arguments passed to one function become the arguments passed to another. For
example, this `Apply` function takes a callable object and argument expressions
to call it with and performs the call:

```carbon
fn Apply[... each ArgET: Core.ExtType,
         FNT: call(... fwd _: each ArgET)]
        (fwd f: FNT,
         ... fwd each arg: each ArgET)
        -> fwd FNT.ResultExtType {
  return (~f)(... fwd ~each arg);
}
```

> **Note:** Forwarding of a `var` parameter will be marked with `~` (as in the
> example) once moves are supported. Unlike input-to-output forwarding (see `Id`
> in
> [proposal #8183](https://github.com/carbon-language/carbon-lang/pull/8183),
> which must initialize a separate caller-provided return address and therefore
> requires `Core.Move`), input-to-input forwarding of a `var` parameter is
> expected to reuse its caller-allocated `var` storage without requiring
> `Core.Move`; see `Core.ConvertsInto` in
> [proposal #8186](https://github.com/carbon-language/carbon-lang/pull/8186).

> **Note:** An expression with generic extended type is marked with `fwd`
> (rather than `ref`) when passed as an argument, since whether it instantiates
> to a reference is not known at definition time (see the "`fwd` on call
> arguments" alternative in
> [proposal #8183](https://github.com/carbon-language/carbon-lang/pull/8183)).

This works with `Apply` calling `f` once. To support multiple calls, we drop
support for using `var` parameters with initializing argument expressions:

```carbon
fn Apply2[... each ArgET: Core.NoVarExtType,
          FNT: call(... fwd _: each ArgET) &
                Core.NoVarExtType]
         (fwd f: FNT, ... fwd each arg: each ArgET) {
  f(... fwd each arg);
  f(... fwd each arg);
}
```

With this change, following the rules of extended type deduction (in
[proposal #8183](https://github.com/carbon-language/carbon-lang/pull/8183)),
initializing expressions will be converted to values.

This was discussed in open discussion:

-   [2025-05-07](https://docs.google.com/document/d/1Yt-i5AmF76LSvD4TrWRIAE_92kii6j5yFiW-S7ahzlg/edit?tab=t.0#heading=h.4zbo49wg5rmk):
    definition of `Apply` using the proposed extended types
-   [2025-05-12](https://docs.google.com/document/d/1Yt-i5AmF76LSvD4TrWRIAE_92kii6j5yFiW-S7ahzlg/edit?tab=t.0#heading=h.1mjh6unumnwu):
    calling conventions limit forwarding for `var` extended types

#### Forwarding tuple parameters

A tuple expression can have elements with different categories and other
different extended types, not just different types. This proposal relies on
compound extended type values (from
[proposal #8183](https://github.com/carbon-language/carbon-lang/pull/8183)) to
properly capture the extended type of each element, so a tuple expression can be
faithfully forwarded. Alternatives where the type of an expression was kept
separate from the other extended type aspects struggled to capture this, see the
[discussion on 2025-05-07](https://docs.google.com/document/d/1Yt-i5AmF76LSvD4TrWRIAE_92kii6j5yFiW-S7ahzlg/edit?tab=t.0#heading=h.4zbo49wg5rmk).

The approach from this proposal allows functions that pass variable-length
argument lists using tuple patterns, such as this definition of `ApplySum`:

```carbon
// Multiple variadic parameters have to be passed as tuples.
fn ApplySum[... each ArgET: Core.NoVarExtType,
            Fn: call(... fwd _: each ArgET) -> i32 & Core.NoVarExtType]
           (fwd f: Fn,
            (... fwd each a: each ArgET),
            (... fwd each b: each ArgET)) -> i32 {
  return f(... fwd each a) + f(... fwd each b);
}
```

to compose with the `Apply` function from
[the "Forwarding" section](#forwarding), as in:

```carbon
// Preserves the categories of the argument
// tuple expressions.
Apply(ApplySum, (...), (...))
```

This works since the extended type deduced for the arguments of `Apply` can be a
composite extended type value that captures the extended type of each element of
each tuple argument.

## Future work

### Impl prioritization

We still need to have a mechanism for picking an `impl` to use when there are
multiple with the most preferred type structure. Resolving this question will
allow us to support prioritization across the `CallPrimitive` implementations
for a type, which is how we expect to support selecting an overload from a
function overload set.

### Compile-time calls

Supporting functions that execute at compile time (`constexpr`/`consteval`) is
discussed as future work in
[proposal #8183](https://github.com/carbon-language/carbon-lang/pull/8183).

## Rationale

This proposal advances [Carbon's goals](/docs/project/goals.md) and
[principles](/docs/project/principles):

-   [Code that is easy to read, understand, and write](/docs/project/goals.md#code-that-is-easy-to-read-understand-and-write):
    `call(...)` provides concise, function-signature-like syntax for
    constraining and implementing callables, while supporting mutating functors,
    `ref`, `var`, and compile-time parameters, and perfect forwarding (`Apply`).
-   [Performance-critical software](/docs/project/goals.md#performance-critical-software):
    Modeling `val`, `ref`, and `var` parameters and returns directly in
    `Core.CallPrimitive` allows generic higher-order functions and forwarding
    wrappers without unnecessary copies or materializations.
-   [Principle: All APIs are library APIs](/docs/project/principles/library_apis_only.md):
    Ordinary functions, lambdas, bound methods, and user-defined functors all
    uniformly implement `Core.CallPrimitive`.
-   [Principle: Progressive disclosure](/docs/project/principles/progressive_disclosure.md):
    `call` sugar and the `Core.Call` named constraint let callers specify
    common callable signatures without writing `Core.ExtType` literals directly.

## Alternatives considered

### Drop `Core.Call` and rename `CallPrimitive` to `Call`

Because [`call` syntax](#call) already provides concise, type-friendly syntax
for common call signatures (such as `call[val](_: i32) -> bool`), keeping a
separate `Core.Call` named constraint may be redundant (and risks offering
[two ways to do the same thing](/docs/project/principles/one_way.md)). On the
other hand, keeping `Core.Call` parallels the other operator named constraints
(`Core.AddWith`, `Core.IndexWith`, `Core.As`, and so on in
[proposal #8188](https://github.com/carbon-language/carbon-lang/pull/8188)).
This remains an open question noted in
[the `Core.Call` section](#corecall-named-constraint).

### Put `self` inside `(...)` in `call` syntax

[Proposal #7016](https://github.com/carbon-language/carbon-lang/pull/7016) moved
method `self` declarations from square brackets `[self: Self]` into the explicit
parameter list `(`...`)`. Updating `call[val]()` / `call[ref]()` /
`call[var]()` to place `self` inside `(...)` would align with method syntax,
whereas keeping `[val]`/`[ref]`/`[var]` visually separates the callable receiver
category from the call arguments. This remains an open question noted in
[the `call` section](#call).

### Related alternatives in other proposals

Several alternatives closely tied to `Core.CallPrimitive` and `call` are
covered in the proposals this builds on:

-   Implicit conversion from `type` to `Core.ExtType` (rejected in favor of
    `call` sugar) and `fwd` on call arguments: see
    [proposal #8183](https://github.com/carbon-language/carbon-lang/pull/8183).
-   Supporting implicit conversions on `impl`s instead of marking variance on
    `CallPrimitive`: see
    [proposal #8186](https://github.com/carbon-language/carbon-lang/pull/8186).
