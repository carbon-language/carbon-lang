# Member binding operators

<!--
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

[Pull request](https://github.com/carbon-language/carbon-lang/pull/3720)

<!-- toc -->

## Table of contents

-   [Abstract](#abstract)
-   [Problem](#problem)
-   [Background](#background)
-   [Proposal](#proposal)
-   [Details](#details)
    -   [Inheritance and other implicit conversions](#inheritance-and-other-implicit-conversions)
    -   [Data fields](#data-fields)
    -   [Generic type of a class member](#generic-type-of-a-class-member)
        -   [Methods](#methods)
        -   [Fields](#fields)
    -   [C++ pointer to member](#c-pointer-to-member)
    -   [Instance interface members](#instance-interface-members)
    -   [Non-instance interface members](#non-instance-interface-members)
    -   [C++ operator overloading](#c-operator-overloading)
-   [Future work](#future-work)
    -   [Future: tuple indexing](#future-tuple-indexing)
    -   [Future: properties](#future-properties)
    -   [Future: building block for language features such as API extension](#future-building-block-for-language-features-such-as-api-extension)
-   [Rationale](#rationale)
-   [Alternatives considered](#alternatives-considered)
    -   [Swap the member binding interface parameters](#swap-the-member-binding-interface-parameters)
    -   [Member binding to references produces a value that wraps a pointer](#member-binding-to-references-produces-a-value-that-wraps-a-pointer)
    -   [Separate interface for compile-time member binding instead of type member binding](#separate-interface-for-compile-time-member-binding-instead-of-type-member-binding)
    -   [Non-instance members are idempotent under member binding](#non-instance-members-are-idempotent-under-member-binding)
    -   [Separate `Result` types for `BindToValue` and `BindToRef`](#separate-result-types-for-bindtovalue-and-bindtoref)
    -   [`BindToValue` is a subtype of `BindToRef`](#bindtovalue-is-a-subtype-of-bindtoref)
    -   [Directly rewrite all calls to interface member functions to method call intrinsics](#directly-rewrite-all-calls-to-interface-member-functions-to-method-call-intrinsics)

<!-- tocstop -->

## Abstract

Define the member binding operation used to compute the result of `x.y`, `p->y`,
`x.(C.y)`, and `p->(C.y)` as calling a method from user-implementable
interfaces.

## Problem

What happens when member binding is performed between an object instance and a
member of its type? We'd like to define the semantics in a way that is simple,
orthogonal, supports the use cases from C++, allows users to express their
intent in code in a natural and predictable way that is consistent with other
Carbon constructs, and is consistent with Carbon's goals.

Consider a class with a method and a field:

```carbon
class C {
  fn M[self: Self]();
  var f: i32;
}
var x: C = {.f = 2};
```

The expressions `C.M` and `C.f` correspond roughly to
[C++ pointers to members](https://en.cppreference.com/w/cpp/language/pointer#Pointers_to_members).
They may be used to access the members of `x` using Carbon's
[compound member syntax](/docs/design/expressions/member_access.md), as in
`x.(C.M)` or `x.(C.f)`. What is their type? Can they be passed to a function
separately from the instance of `C` to bind with it?

The expression `x.M` on the other hand doesn't have a trivial correspondence in
C++ despite being a useful to bind a specific instance and produce a stand-alone
callable object. We would like a model that allows `x.M` to be meaningful in a
way that is consistent with the existing meaning of `x.f` and generalizes well
across different kinds of methods and callables.

Another issue is how we clearly delineate the `self` associated with a method
signature as separate from the `self` of function values.

## Background

Member access has been specified in two proposals:

-   [Proposal #989: Member access expressions](https://github.com/carbon-language/carbon-lang/pull/989)
-   [Proposal #2360: Types are values of type `type`](https://github.com/carbon-language/carbon-lang/pull/2360)

The results of these proposals is recorded in the
["qualified names and member access" design document](/docs/design/expressions/member_access.md).
Notably, there is the process of
[instance binding](/docs/design/expressions/member_access.md#instance-binding),
that can convert a method into a bound method. This is described as an
uncustomizable process, with the members of classes being non-first-class names.

With
[proposal #3646: Tuples and tuple indexing](https://github.com/carbon-language/carbon-lang/pull/3646),
tuple indexing also uses the member-access syntax, except with numeric names for
the fields.

The currently accepted proposals for functions, most notably
[Proposal #2875: Functions, function types, and function calls](https://github.com/carbon-language/carbon-lang/pull/2875),
don't support all of the different function signatures for the `Call` interface.
For example, it does not support `addr self` or explicit compile-time
parameters. That is out of scope of this proposal, and will be addressed
separately, and means that `addr self` methods won't be considered here. The
difference between functions and methods, however, is in scope.

Other languages, such as
[C#](https://learn.microsoft.com/en-us/dotnet/csharp/programming-guide/delegates/using-delegates)
and [D](https://tour.dlang.org/tour/en/basics/delegates), have constructs that
represent bound and unbound methods, such as "delegates".

## Proposal

We propose that Carbon defines the compound member access operator, specifically
`x.(y)`, in terms of rewrites to invoking an interface method, like other
operators. There are three different interfaces used, depending on whether `x`
is a value expression, a reference expression, or a facet:

```carbon
// This determines the type of the result of member binding. It is
// a separate interface shared by `BindToValue` and `BindToRef` to
// ensure they produce the same result type. We don't want the
// type of an expression to depend on the expression category
// of the arguments.
interface Bind(T:! type) {
  let Result:! type;
}

// For a value expression `x` with type `T` and an expression
// `y` of type `U`, `x.(y)` is `y.((U as BindToValue(T)).Op)(x)`
interface BindToValue(T:! type) {
  extend Bind(T);
  fn Op[self: Self](x: T) -> Result;
}

// For a reference expression `x` using a member binding `var x: T`
// and an expression `y` of type `U`, `x.(y)` is
// `*y.((U as BindToRef(T)).Op)(&x)`
interface BindToRef(T:! type) {
  extend Bind(T);
  fn Op[self: Self](p: T*) -> Result*;
}

// For a facet value, which includes all type values, `T` and
// an expression `y` of type `U`, `T.(y)` is
// `y.((U as BindToType(T)).Op)()`.
interface BindToType(T:! type) {
  let Result:! type;
  fn Op[self: Self]() -> Result;
}
```

> **Note:** `BindToType` is its own interface since the members of a type are
> defined by their values, not by their type. Observe that this means that a
> generic function might not use `BindToType` on a symbolic value that was not
> known to be a facet, where it would use `BindToType` on the concrete value.

The other member access operators -- `x.y`, `x->y`, and `x->(y)` -- are defined
by how they rewrite into the `x.(y)` form using these two rules:

-   `x.y` is interpreted using the existing
    [member resolution rules](/docs/design/expressions/member_access.md#member-resolution).
    For example, `x.y` is treated as `x.(T.y)` for non-type values `x` with type
    `T`.
    -   Simple member access of a facet `T`, as in `T.y`, is not rewritten into
        the `T.(`\_\_\_`)` form.
-   `x->y` and `x->(y)` are interpreted as `(*x).y` and `(*x).(y)` respectively.

## Details

To use instance members of a class, we need to go through the additional step of
_member binding_. Consider a class `C`:

```carbon
class C {
  fn F[self: Self]() -> i32 { return self.x + 5; }
  fn Static() -> i32 { return 2; }
  var x: i32;
}
```

Each member of `C` with a distinct name will have a corresponding type (like
`__TypeOf_C_F`) and value of that type (like `__C_F`). There are two more types
for each member function (either static class function or method), though, that
[adapt](/docs/design/generics/terminology.md#adapting-a-type) `C` and represent
the type of binding that member with either a `C` value or variable.

```carbon
class __TypeOf_C_F {}
let __C_F:! __TypeOf_C_F = {};
class __Binding_C_F {
  adapt C;
}

// and similarly for Static
```

These are the types that result from
[instance binding](/docs/design/expressions/member_access.md#instance-binding)
an instance of `C` with these member names. They define the bound method value
and bound method type of [proposal #2875](/proposals/p2875.md#bound-methods).
For example,

```carbon
let v: C = {.x = 3};
Assert(v.F() == 8);
Assert(v.Static() == 2);
var r: C = {.x = 4};
Assert(r.F() == 9);
Assert(r.Static() == 2);
```

is interpreted as:

```carbon
let v: C = {.x = 3};
Assert((v as __Binding_C_F).(Call(()).Op)() == 8);
Assert((v as __Binding_C_Static).(Call(()).Op)() == 2);
var r: C = {.x = 4};
Assert((r as __Binding_C_F).(Call(()).Op)() == 9);
Assert((r as __Binding_C_Static).(Call(()).Op)() == 2);
```

How does this arise?

1. First the simple member access is resolved using the type of the receiver: \
   `v.F` -> `v.(C.F)`, `v.Static` -> `v.(C.Static)`, `r.F` -> `r.(C.F)`,
   `r.Static` -> `r.(C.Static)`. \
   Note that `C.F` is `__C_F` with type `__TypeOf_C_F`, and `C.Static` is
   `__C_Static` with type `__TypeOf_C_Static`.
2. It then looks at the expression to the left of the `.`:
    - If it is a facet value, the "member binding to type" (`BindToType`)
      operator is applied.
    - If it is a reference expression, the "member binding to reference"
      (`BindToRef`) operator is applied.
    - If it is a value expression, the "member binding to value" (`BindToValue`)
      operator is applied.
3. The result of the member binding has a type that implements the call
   interface.

> **Note:** The current wording in
> [member_access.md](/docs/design/expressions/member_access.md) says that
> `v.(C.Static)` and `r.(C.Static)` are both invalid, because they don't perform
> member name lookup, instance binding, nor impl lookup -- the `v.` and `r.`
> portions are redundant. That rule is removed by this proposal.
>
> Instead, tools such as linters can highlight such code as suspicious on a
> best-effort basis, particularly when the issue is contained in a single
> expression. Such tools may still allow code that performs the same operation
> across multiple statements, as in:
>
> ```carbon
> let M:! auto = C.Static;
> v.(M)();
> r.(M)();
> ```
>
> Note that if `M` is an overloaded name, it could be an instance member in some
> cases and a non-instance member in others, depending on the arguments passed.
> This is another reason to delegate this to linters analyzing a whole
> expression on a best-effort basis, rather than a strict rule just about member
> binding.

The member binding operators are defined using three dedicated interfaces --
`BindToValue`, `BindToRef`, and `BindToType` --
[as defined in the "proposal" section](#proposal). These member binding
operations are implemented for the types of the class members:

```carbon
impl __TypeOf_C_F as BindToValue(C)
    where .Result = __Binding_C_F {
  fn Op[unused self: Self](x: C) -> __Binding_C_F {
    return x as __Binding_C_F;
  }
}

// Note that the `Result` type has to match, since
// it is an associated type in the `Bind(C)` interface
// that both `BindToValue(C)` and `BindToRef(C)` extend.
impl __TypeOf_C_F as BindToRef(C)
    where .Result = __Binding_C_F {
  fn Op[unused self: Self](p: C*) -> __Binding_C_F* {
    return p as __Binding_C_F*;
  }
}
```

> **Note:** `BindToType` is used for
> [non-instance interface members](#non-instance-interface-members).

Those implementations are how we get from `__C_F` with type `__TypeOf_C_F` to
`v as __Binding_C_F` or `&r as __Binding_C_F*`, conceptually following these
steps:

```carbon
// `v` is a value and so uses `BindToValue`
v.F() == v.(C.F)()
      == v.(__C_F)()
      == __C_F.((__TypeOf_C_F as BindToValue(C)).Op)(v)()
      == (v as __Binding_C_F)()

// `r` is a reference expression and so uses `BindToRef`
r.F() == r.(C.F)()
      == r.(__C_F)()
      == (*__C_F.((__TypeOf_C_F as BindToRef(C)).Op)(&r))()
      == (*(&r as __Binding_C_F*))()
```

However, to avoid recursive application of these same rules, we need to avoid
expressing this in terms of evaluating `__C_F.(`...`)`. Instead the third step
uses an intrinsic compiler primitive, as in:

```carbon
// `v` is a value and so uses `BindToValue`
v.F() == v.(C.F)()
      == v.(__C_F)()
      == inlined_method_call_compiler_intrinsic(
              <function body (__TypeOf_C_F as BindToValue(C)).Op overload 0>,
              __C_F, (v))()
      == (v as __Binding_C_F)()

// `r` is a reference expression and so uses `BindToRef`
r.F() == r.(C.F)()
      == r.(__C_F)()
      == (*inlined_method_call_compiler_intrinsic(
              <function body (__TypeOf_C_F as BindToRef(C)).Op overload 0>,
              __C_F, (&r)))()
      == (*(&r as __Binding_C_F*))()
```

At this point we have resolved the member binding, and are left with an
expression of type `__Binding_C_F` followed by `()`. In the first case, that
expression is a value expression. In the second case, it is a reference
expression.

The last ingredient is the implementation of the call interfaces for these bound
types.

```carbon
// Member binding with `C.F` produces something with type
// `__Binding_C_F` whether it is a value or reference
// expression. Since `C.F` takes `self: Self` it can be
// used in both cases.
impl __Binding_C_F as Call(()) with .Result = i32 {
  fn Op[self: Self]() -> i32 {
    // Calls `(self as C).(C.F)()`, but without triggering
    // member binding again.
    return inlined_method_call_compiler_intrinsic(
        <function body C.F overload 0>, self as C, ());
  }
}

// `C.Static` works the same as `C.F`, except it also
// implements the call interfaces on `__TypeOf_C_Static`.
// This allows `C.Static()` to work, in addition to
// `v.Static()` and `r.Static()`.
impl __Binding_C_Static as Call(()) with .Result = i32 {
  // Other implementations of `Call(())` are the same.
  fn Op[unused self: Self]() -> i32 {
    // Calls `C.Static()`, without triggering member binding again.
    return inlined_call_compiler_intrinsic(
               <function body C.Static overload 0>, ());
  }
}
impl __TypeOf_C_Static as Call(()) where .Result = i32;
```

Going back to `v.F()` and `r.F()`, after member binding the next step is to
resolve the call. As described in
[proposal #2875](https://github.com/carbon-language/carbon-lang/pull/2875), this
call is rewritten to an invocation of the `Op` method of the `Call(())`
interface, using the implementations just defined. Note:

-   Passing `*(&r as __Binding_C_F*)` to the `self` parameter of `Call(()).Op`
    converts the reference expression to a value. Note that mutating
    (`addr self`) methods are [out of scope for this proposal](#background).
-   The `Call` interface is special. We don't
    [rewrite](#instance-interface-members) calls to `Call(__).Op` to avoid
    infinite recursion.

```carbon
v.F() == (v as __Binding_C_F)()
      == (v as __Binding_C_F).((__Binding_C_F as Call(())).Op)()
      == inlined_method_call_compiler_intrinsic(
            <function body (__Binding_C_F as Call(())).Op overload 0>,
            v as __Binding_C_F, ());
      == inlined_method_call_compiler_intrinsic(
             <function body C.F overload 0>,
             (v as __Binding_C_F) as C, ())
      == inlined_method_call_compiler_intrinsic(
             <function body C.F overload 0>, v, ())

r.F() == (*(&r as __Binding_C_F*))()
      == (*(&r as __Binding_C_F*)).((__Binding_C_F as Call(())).Op)()
      == inlined_method_call_compiler_intrinsic(
            <function body (__Binding_C_F as Call(())).Op overload 0>,
            *(&r as __Binding_C_F*) <as value expression>, ());
      == inlined_method_call_compiler_intrinsic(
             <function body C.F overload 0>,
             *(&r as __Binding_C_F*) as C, ())
      == inlined_method_call_compiler_intrinsic(
             <function body C.F overload 0>,
             r <as value expression>, ())
```

> **Note:** This rewrite results in compiler intrinsics for calling. This is to
> show that no more rewrites are applied.

### Inheritance and other implicit conversions

Now consider methods of a base class:

```carbon
base class B {
  fn F[self: Self]();
  virtual fn V[self: Self]();
}

class D {
  extend base: B;
  impl fn V[self: Self]();
}

var d: D = {}
d.(B.F)();
d.(B.V)();
```

To allow this to work, we need the implementation of the member binding
interfaces to allow implicit conversions:

```carbon
impl [T:! ImplicitAs(B)] __TypeOf_B_F as BindToValue(T)
    where .Result = __Binding_B_F {
  fn Op[self: Self](x: T) -> __Binding_B_F {
    return (x as B) as __Binding_B_F;
  }
}

impl [T:! type where .Self* impls ImplicitAs(B*)]
    __TypeOf_B_F as BindToRef(T)
    where .Result = __Binding_B_F {
  fn Op[self: Self](p: T*) -> __Binding_B_F* {
    return (p as B*) as __Binding_B_F*;
  }
}
```

This matches the expected semantics of method calls, even for methods of final
classes.

Note that the implementation of the member binding interfaces is where the
`Self` type of a method is used. If that type is different from the class it is
being defined in, as considered in
[#1345](https://github.com/carbon-language/carbon-lang/issues/1345), that will
be reflected in the member binding implementations.

```carbon
class C {
  // Note: not `self: Self` or `self: C`!
  fn G[self: Different]();
}

let c: C = {};
// `c.G()` is only allowed if there is an implicit
// conversion from `C` to `Different`.

let d: Different = {};
// Allowed:
d.(C.G)();
```

results in an implementation using `Different` instead of `C`:

```carbon
// `C.G` will only member bind to values that can implicitly convert
// to type `Different`.
impl [T:! ImplicitAs(Different)] __TypeOf_C_G as BindToValue(T)
    where .Result = __Binding_C_G;
```

### Data fields

The same `BindToValue` and `BindToRef` operations allow us to define access to
the data fields in an object, without any additional changes.

For example, given a class with a data member `m` with type `i32`:

```carbon
class C {
  var m: i32;
}
```

we want the usual operations to work, with `x.m` equivalent to `x.(C.m)`:

```carbon
let v: C = {.m = 4};
var x: C = {.m = 3};
x.m += 5;
Assert(x.(C.m) == v.m + v.(C.m));
```

To accomplish this we will, as before, associate an empty (stateless or
zero-sized) type with the `m` member of `C`, that just exists to support the
member binding operation. However, this time the result type of member binding
is simply `i32`, the type of the variable, instead of a new, dedicated type.

```carbon
class __TypeOf_C_m {}
let __C_m:! __TypeOf_C_m = {};

impl __TypeOf_C_m as BindToValue(C) where .Result = i32 {
  fn Op[self: Self](x: C) -> i32 {
    // Effectively performs `x.m`, but without triggering member binding again.
    return value_compiler_intrinsic(x, __OffsetOf_C_m, i32)
  }
}

impl __TypeOf_C_m as BindToRef(C) where .Result = i32 {
  fn Op[self: Self](p: C*) -> i32* {
    // Effectively performs `&p->m`, but without triggering member binding again,
    // by doing something like `((p as byte*) + __OffsetOf_C_m) as i32*`
    return offset_compiler_intrinsic(p, __OffsetOf_C_m, i32);
  }
}
```

These definitions give us the desired semantics:

```carbon
// For value `v` with type `T` and `y` of type `U`,
// `v.(y)` is `y.((U as BindToValue(T)).Op)(v)`
v.m == v.(C.m)
    == v.(__C_m)
    == v.(__C_m as (__TypeOf_C_m as BindToValue(C)))
    == __C_m.((__TypeOf_C_m as BindToValue(C)).Op)(v)
    == value_compiler_intrinsic(v, __OffsetOf_C_m, i32)

// For reference expression `var x: T` and `y` of type `U`,
// `x.(y)` is `*y.(U as BindToRef(T)).Op(&x)`
x.m == x.(C.m)
    == x.(__C_m)
    == *__C_m.((__TypeOf_C_m as BindToRef(C)).Op)(&x)
    == *offset_compiler_intrinsic(&x, __OffsetOf_C_m, i32)
// Note that this requires `x` to be a reference expression,
// so `&x` is valid, and produces a reference expression,
// since it is the result of dereferencing a pointer.
```

The fields of [tuple types](/docs/design/tuples.md) and
[struct types](/docs/design/classes.md#struct-types) operate the same way.

```carbon
let t_let: (i32, i32) = (3, 6);
Assert(t_let.(((i32, i32) as type).0) == 3);

var t_var: (i32, i32) = (4, 8);
Assert(t_var.(((i32, i32) as type).1) == 8);
t_var.(((i32, i32) as type).1) = 9;
Assert(t_var.1 == 9);

let s_let: {.x: i32, .y: i32} = {.x = 5, .y = 10};
Assert(s_let.({.x: i32, .y: i32}.x) == 5);

var s_var: {.x: i32, .y: i32} = {.x = 6, .y = 12};
Assert(s_var.({.x: i32, .y: i32}.y) == 12);
s_var.({.x: i32, .y: i32}.y) = 13;
Assert(s_var.y == 13);
```

For example, `{.x: i32, .y: i32}.x` is a value `__Struct_x_i32_y_i32_Field_x`,
analogous to `__C_m`, of a type `__TypeOf_Struct_x_i32_y_i32_Field_x` (that is
zero-sized / has no state), analogous to `__TypeOf_C_m`, that implements the
member binding interfaces for any type that implicitly converts to
`{.x: i32, .y: i32}`.

Note that for tuples, the `as type` is needed since `(i32, i32)` on its own is a
tuple, not a type. In particular `(i32, i32)` is not the type of `t_let` or
`t_var`. `(i32, i32).0` is just `i32`, and isn't the name of the first element
of an `(i32, i32)` tuple.

### Generic type of a class member

Given the above, we can now write a constraint on a symbolic parameter to match
the names of an unbound class member. There are a two cases: methods and fields.

#### Methods

Restricting to value methods, since mutating (`addr self`) methods are
[out of scope for this proposal](#background), the receiver object may be passed
by value. To be able to call the method, we must include a restriction that the
result of `BindToValue` implements `Call(())`:

```carbon
// `m` can be any method object that implements `Call(())` once bound.
fn CallMethod
    [T:! type, M:! BindToValue(T) where .Result impls Call(())]
    (x: T, m: M) -> auto {
  // `x.(m)` is rewritten to a call to `BindToValue(T).Op`. The
  // constraint on `M` ensures the result implements `Call(())`.
  return x.(m)();
}
```

This will work with any value method or static class function. This will also
work with inheritance and virtual methods, using
[the support for implicit conversions of self](#inheritance-and-other-implicit-conversions).

```carbon
base class X {
  virtual fn V[self: Self]() -> i32 { return 1; }
  fn B[self: Self]() -> i32 { return 0; }
}
class Y {
  extend base: X;
  impl fn V[self: Self]() -> i32 { return 2; }
}
class Z {
  extend base: X;
  impl fn V[self: Self]() -> i32 { return 3; }
}

var (x: X, y: Y, z: Z);

// Respects inheritance
Assert(CallMethod(x, X.B) == 0);
Assert(CallMethod(y, X.B) == 0);
Assert(CallMethod(z, X.B) == 0);

// Respects method overriding
Assert(CallMethod(x, X.V) == 1);
Assert(CallMethod(y, X.V) == 2);
Assert(CallMethod(z, X.V) == 3);
```

#### Fields

Fields can be accessed, given the type of the field

```carbon
fn GetField
    [T:! type, F:! BindToValue(T) where .Result = i32]
    (x: T, f: F) -> i32 {
  // `x.(f)` is rewritten to `f.((F as BindToValue(T)).Op)(x)`,
  // and `(F as BindToValue(T)).Op` is a method on `f` with
  // return type `i32` by the constraint on `F`.
  return x.(f);
}

fn SetField
    [T:! type, F:! BindToRef(T) where .Result = i32]
    (x: T*, f: F, y: i32) {
  // `x->(f)` is rewritten to `(*x).(f)`, which then
  // becomes: `*f.((F as BindToRef(T)).Op)(&*x)`
  // The constraint `F` says the return type of
  // `(F as BindToRef(T)).Op` is `i32*`, which is
  // dereferenced to get an `i32` reference expression
  // which may then be assigned.
  x->(f) = y;
}

class C {
  var m: i32;
  var n: i32;
}
var c: C = {.m = 5, .n = 6};
Assert(GetField(c, C.m) == 5);
Assert(GetField(c, C.n) == 6);
SetField(&c, C.m, 42);
SetField(&c, C.n, 12);
Assert(GetField(c, C.m) == 42);
Assert(GetField(c, C.n) == 12);
```

### C++ pointer to member

In [the generic type of member section](#generic-type-of-a-class-member), the
names of members, such as `D.K`, `X.B`, `X.V`, and `C.n`, refer to zero-sized /
stateless objects where all the offset information is encoded in the type.
However, the definitions of `CallMethod`, `SetField`, and `GetField` do not
depend on that fact and will be usable with objects, such as C++
pointers-to-members, that include the offset information in the runtime object
state. So we can define member binding implementations for them so that they may
be used with Carbon's `.(`**`)` and `->(`**`)` operators.

For example, this is how we expect C++ code to call the above Carbon functions:

```cpp
struct C {
  int F() const { return m + 1; }
  int m;
};

int main() {
  // pointer to data member `m` of class C
  int C::* p = &C::m;
  C c = {2};
  assert(c.*p == 2);
  assert(Carbon::GetField(c, p) == 2);
  Carbon::SetField(&c, p, 4);
  assert(c.m == 4);
  // pointer to method `F` of class C
  int (C::*q)() const = &C::F;
  assert(Carbon::CallMethod(&c, q) == 5);
}
```

### Instance interface members

Instance members of an interface, such as methods, can use this framework. For
example, given these declarations:

```carbon
interface I {
  fn F[self: Self]();
}
class C {
  impl as I;
}
let c: C = {};
```

Then `I.F` is its own value with its own type:

```carbon
class __TypeOf_I_F {}
let __I_F:! __TypeOf_I_F = {};
```

That type implements `BindToValue` for any type that implements the interface
`I`:

```carbon
class __Binding_I_F(T:! I) {
  adapt T;
}
impl forall [T:! I] __TypeOf_I_F as BindToValue(T)
    where .Result = __Binding_I_F(T) {
  fn Op[self: Self](x: T) -> __Binding_I_F(T) {
    // Valid since `__Binding_I_F(T)` adapts `T`:
    return x as __Binding_I_F(T);
  }
}
```

The actual dispatch to the `I.F` method of `C` happens in the implementation of
the `Call` interface of this adapter type that is the result of member binding
to a value. So, this implementation of `C as I`:

```carbon
impl C as I {
  fn F[self: Self]() {
    Fanfare(self);
  }
}
```

Results in this implementation:

```carbon
impl __Binding_I_F(C) as Call(()) where .Result = () {
  fn Op[self: Self]() {
    inlined_method_call_compiler_intrinsic(
        <function body (C as I).F overload 0>, self as C, ());
  }
}
```

A call such as `c.(I.F)()` goes through these rewrites:

```carbon
c.(I.F)() == c.(__I_F)()
          == __I_F.((__TypeOf_I_F as BindToValue(C)).Op)(c)()
          == (c as __Binding_I_F(C))()
          == (c as __Binding_I_F(C)).((__Binding_I_F(C) as Call(())).Op)()
```

Which results in invoking the above implementation that will ultimately call
`Fanfare(c)`.

> **Note:** The `Call` interface gets special treatment and does not get these
> rewrites to avoid recursing forever.

### Non-instance interface members

Non-instance members use the `BindToType` interface instead. For example, if `G`
is a non-instance function of an interface `J`:

```carbon
interface J {
  fn G();
}
impl C as J;
```

Again the member is given its own type and value:

```carbon
class __TypeOf_J_G {}
let __J_G:! __TypeOf_J_G = {};
```

Since this is a non-instance member, this type implements `BindToType` instead
of `BindToValue`:

```carbon
class __TypeBinding_J_G(T:! J) {}
impl forall [T:! J] __TypeOf_J_G as BindToType(T)
    where .Result = __TypeBinding_J_G(T) {
  fn Op[self: Self]() -> __TypeBinding_J_G(T) {
    return {};
  }
}
```

So, this implementation of `C as J`:

```carbon
impl C as J {
  fn G() {
    Fireworks();
  }
}
```

Results in this implementation:

```carbon
impl __TypeBinding_J_G(C) as Call(()) where .Result = () {
  fn Op[self: Self]() {
    Fireworks();
  }
}
```

A call such as `C.(J.G)()` goes through these rewrites:

```carbon
C.(J.G)() == C.(__J_G)()
          == __J_G.((__TypeOf_J_G as BindToType(C)).Op)()()
          == ({} as __TypeBinding_J_G(C))()
          == (({} as __TypeBinding_J_G(C)) as Call(())).Op()
```

Which calls the above implementation that calls `Fireworks()`.

> **Note:** Member binding for non-instance members doesn't work with
> `BindToValue`, we need `BindToType`. Otherwise there is no way to get the
> value `C` into the result type. Furthermore, we want `BindToType`
> implementation no matter which facet of the type is used in the code.

### C++ operator overloading

C++ does not support customizing the behavior of `x.y`. It does support
customizing the behavior of `operator*` and `operator->` which is frequently
used to support smart pointers and iterators. There is, however, nothing
restricting the implementations of those two operators to be consistent, so that
`(*x).y` and `x->y` are the same.

Carbon instead will only have a single interface for customizing dereference,
corresponding to `operator*` not `operator->`. All uses of `x->y` will be
rewritten to use `(*x).y` instead. This may cause some friction when porting C++
code where those operators are not consistent. If the C++ code is just missing
the definition of `operator*` corresponding to an `operator->`, a workaround
would be just to define `operator*`.

Other cases of divergence between those operators should be rare, since that is
both surprising to users and for the common case of iterators, violates the C++
requirements. If necessary, we can in the future introduce a specific construct
just for C++ interop that invokes the C++ arrow operator, such as
`CppArrowOperator(x)`, that returns a pointer.

**Context:** This was discuseed in
[2024-02-29 open discussion](https://docs.google.com/document/d/1s3mMCupmuSpWOFJGnvjoElcBIe2aoaysTIdyczvKX84/edit?resourcekey=0-G095Wc3sR6pW1hLJbGgE0g&tab=t.0#heading=h.5vj8ohrvqjqh)
and in
[a comment on this proposal](https://github.com/carbon-language/carbon-lang/pull/3720/files#r1507917882).

## Future work

### Future: tuple indexing

We can reframe the use of the compound member access syntax for tuple fields as
an implementation of member binding of tuples with compile-time integer
expressions. The specifics of how this works will be resolved later, once we
address how compile-time interacts with interfaces.

### Future: properties

If there was a way to implement the member binding operator to only produce
values, even when the expression to the left of the `.` was a reference
expression, then that could be used to implement read-only properties. This
would support something like:

```carbon
let Pi: f64 = 3.1415926535897932384626433832795;

class Circle {
  var radius: f64;
  read_property area -> f64 {
    return Pi * self.radius * self.radius;
  }
}

let c: Circle = {.radius = 2};
Assert(NearlyEqual(c.area, 4 * Pi));
```

In this example, the member binding of `c` of type `Circle` to `Circle.area`
would perform the computation and return the result as an `f64`.

If there was some way to customize the result of member binding, this could be
extended to support other kinds of properties, such as mutable properties that
use `get` and `set` methods to access and mutate the value. The main obstacle to
any support for properties with member binding is how the customization would be
done. The most natural way to support this customization would be to have
multiple interfaces. The compiler would try them in a specified order and use
the one it found first. This has the downside of the possibility of different
behavior in a checked generic context where only some of the implementations are
visible. Our choice to
[make the result type the same `Result` associated type of the `Bind` interface](#proposal)
independent of whether the `BindToValue` or `BindToRef` interface is used makes
this less concerning. Only the phase of the result, not the type, would depend
on which implementations were found, similar to
[how indexing works](/docs/design/expressions/indexing.md).

### Future: building block for language features such as API extension

We should be able to express other language features, such as API extension, in
terms of customized member binding, plus possibly some new language primitives.
This should be explored in a future proposal.

## Rationale

This proposal is about:

-   Orthogonality: separating the member binding process as a distinct and
    independent step of using the members of a type.
-   Being consistent with our overall strategy for defining operators in terms
    of interface implementations.
-   Allows member-binding-related functionality to be defined through
    [library APIs](/docs/project/principles/library_apis_only.md).
-   Increases uniformity by making member names into ordinary values with types.
-   Adds expressiveness, enabling member forwarding, passing a member as an
    argument, and other use cases.

These benefits advance Carbon's goals including:

-   [Language tools and ecosystem](/docs/project/goals.md#language-tools-and-ecosystem):
    by making it easier to reason about more Carbon entities within Carbon
    itself, and reducing the number of different concepts that have to be
    modeled.
-   [Code that is easy to read, understand, and write](/docs/project/goals.md#code-that-is-easy-to-read-understand-and-write):
    through increased consistency, uniformity, and expressiveness.
-   [Interoperability with and migration from existing C++ code](/docs/project/goals.md#interoperability-with-and-migration-from-existing-c-code):
    by adding support for pointer-to-member constructs.

## Alternatives considered

### Swap the member binding interface parameters

We considered instead making the receiver object the `Self` type of the
interface, and using the member type as the parameter to the interface. This
would have the advantage of matching the order that they appear in the source,
consistent with other operators.

> **Alternative:**
>
> ```carbon
> // For value `x` with type `T` and `y` of type `U`,
> // `x.(y)` is `x.((T as ValueBind(U)).Op)(y)`
> interface ValueBind(U:! type) {
>   extend Bind(U);
>   fn Op[self: Self](x: U) -> Result;
> }
>
> // For reference expression `var x: T` and `y` of type `U`,
> // `x.(y)` is `*x.((T as RefBind(U)).Op)(y)`
> interface RefBind(U:! type) {
>   extend Bind(U);
>   fn Op[addr self: Self*](x: U) -> Result*;
> }
> ```

This had some disadvantages however:

-   The binding property is more associated with the member than the receiver.
-   Some patterns are more awkward in the alternative syntax.

As an example of this last point, consider a function that takes multiple (or
even a variadic list) methods to call on a receiver object. With the proposed
approach, each method type is constrained:

```carbon
// `m1`, `m2`, and `m3` are methods on class `T`.
fn Call3Methods[T:! type,
                M1:! BindToValue(T) where .Result impls Call(()),
                M2:! BindToValue(T) where .Result impls Call(()),
                M3:! BindToValue(T) where .Result impls Call(())]
    (x: T, m1: M1, m2: M2, m3: M3) -> auto;
```

With the alternative, the type of the receiver would be constrained, and the
deduced types would be written in a different order:

> **Alternative:**
>
> ```carbon
> // `m1`, `m2`, and `m3` are methods on class `T`.
> fn Call3MethodsAlternative1
>     [M1:! type, M2:! type, M3:! type,
>      T:! ValueBid(M1) & ValueBind(M2) & ValueBind(M3)
>          where .(ValueBind(M1).Result) impls Call(())
>            and .(ValueBind(M2).Result) impls Call(())
>            and .(ValueBind(M3).Result) impls Call(())]
>     (x: T, m1: M1, m2: M2, m3: M3) -> auto;
> ```

Or, the constraints can be moved to the method types at the cost of additional
length:

> **Alternative:**
>
> ```carbon
> // `m1`, `m2`, and `m3` are methods on class `T`.
> fn Call3MethodsAlternative2
>     [T:! type,
>      M1:! type where T impls (ValueBind(.Self) where .Result impls Call(())),
>      M2:! type where T impls (ValueBind(.Self) where .Result impls Call(())),
>      M3:! type where T impls (ValueBind(.Self) where .Result impls Call(()))]
>     (x: T, m1: M1, m2: M2, m3: M3) -> auto;
> ```

### Member binding to references produces a value that wraps a pointer

Consider a mutating method on a class:

```carbon
class Counter {
  var count: i32 = 0;
  fn Increment[addr self: Self*]() {
    self->count += 1;
  }
}

var c: Counter = {};
```

This proposal says `c.Increment` is a reference expression with a type that
adapts `Counter`. For `c.Increment()` to affect the value of `c.count`, there
needs to be some way for the `Call` operator to mutate `c`. The current
definition of `Call` takes `self` by value, though, so this doesn't work.
Addressing this is [out of scope of the current proposal](#background).

We could instead make `c.Increment` be a value holding `&c`. That would allow
`Call` to work even when taking `self` by value. This is the solution likely
implied by the current [proposal #2875](/proposals/p2875.md#bound-methods),
though that proposal does not say what the bound method type is at all. It
leaves two other problems, however:

-   We will still need a way to support function objects that are mutated by
    calling them. This comes up, for example, with C++ types that define
    `operator()`.
-   We want the proposed behavior when member binding a [field](#data-fields).

As a result, it would be better to evaluate this alternative later as part of
considering mutation in calls.

### Separate interface for compile-time member binding instead of type member binding

The first proposed way to handle
[non-instance interface members](#non-instance-interface-members) was in
[the #typesystem channel on Discord on 2024-03-07](https://discord.com/channels/655572317891461132/708431657849585705/1215405895752618134).
The suggestion was to have a `CompileBind` interface used for any compile-time
value to the left of the `.`. It would have access to the value, which is needed
when accessing the members of a type.

We eventually concluded that the special treatment was specifically needed for
types, not all compile-time values. The
[insight](https://docs.google.com/document/d/1s3mMCupmuSpWOFJGnvjoElcBIe2aoaysTIdyczvKX84/edit?resourcekey=0-G095Wc3sR6pW1hLJbGgE0g&tab=t.0#heading=h.1k8r5fhfwyh6)
was that types are special because their members are defined by their values,
not by their type
([which is always `type`](https://github.com/carbon-language/carbon-lang/pull/2360)).

### Non-instance members are idempotent under member binding

In the current proposal, member binding of non-instance members results in an
adapter type, the same as an instance member. For example,

```carbon
class C {
  fn Static() -> i32;
}
```

is translated into something like:

> **Current proposal:**
>
> ```carbon
> class __TypeOf_C_Static {}
> let __C_Static:! __TypeOf_C_Static = {};
>
> class __Binding_C_Static {
>   adapt C;
> }
>
> impl __TypeOf_C_Static as BindToValue(C)
>     where .Result = __Binding_C_Static;
>
> impl __TypeOf_C_Static as BindToRef(C)
>     where .Result = __Binding_C_Static;
> ```

An alternative is that member binding of a non-instance member is idempotent, so
there is no `__Binding_C_Static` type and `BindToValue(C)` results in a value of
type `__TypeOf_C_Static` instead:

> **Alternative:**
>
> ```carbon
> class __TypeOf_C_Static {}
> // Might need to be a `var` instead?
> let __C_Static:! __TypeOf_C_Static = {};
>
> impl __TypeOf_C_Static as BindToValue(C)
>     where .Result = __TypeOf_C_Static;
> impl __TypeOf_C_Static as BindToRef(C)
>     where .Result = __TypeOf_C_Static;
> ```

There are a few concerns with this alternative:

-   This is less consistent with the instance member case.
-   There would be a discontinuity when adding an instance overload to a name
    that was previously only a non-instance member.
-   Member binding to a reference is trickier, since it would have to return the
    address of an object of type `__TypeOf_C_Static`. Perhaps a global variable?
-   The current proposal rejects `v.(v.(C.Static))`, which is desirable.

This was discussed in
[this comment on #3720](https://github.com/carbon-language/carbon-lang/pull/3720/files#r1513681915).

### Separate `Result` types for `BindToValue` and `BindToRef`

An earlier iteration of this proposal had separate `Result` associated types for
`BindToValue` and `BindToRef`, as in:

```carbon
interface BindToValue(T:! type) {
  let Result:! type;
  fn Op[self: Self](x: T) -> Result;
}

interface BindToRef(T:! type) {
  let Result:! type;
  fn Op[self: Self](p: T*) -> Result*;
}
```

However, this results in the type of a member binding depending on the what
[category](/docs/design/values.md#expression-categories) the expression to the
left of the dot has. This could change the interpretation of code using
[indexing](/docs/design/expressions/indexing.md), such as an expression like
`a[b].F()`, when the type of `a` is changed from or to a checked generic. This
is because the the expression is legal as long as the type of `a` implements
`IndexWith(typeof(b))`, but category of `a[b]` depends on whether the type of
`a` is known to implement `IndirectIndexWith(typeof(b))`.

To avoid this problem, we make the result type of the member binding the same
whether it is binding to a value or reference. See
[this comment on #3720](https://github.com/carbon-language/carbon-lang/pull/3720/files/99fb69aaaa24bd502d71cd4259f37cfa8346b244#r1597317217).

### `BindToValue` is a subtype of `BindToRef`

We could make `BindToValue` be a subtype of `BindToRef`, as suggested in
[this comment on #3720](https://github.com/carbon-language/carbon-lang/pull/3720/files/99fb69aaaa24bd502d71cd4259f37cfa8346b244#r1597317217).
This would be a step beyond just saying they have to have the same `Result`
type, which is achieved by that type being defined in the `Bind` interface they
both extend.

This approach would rule out the use case where value binding _computes_ a new
value rather than returning an existing one -- that is, a read-only property.
That use case isn't currently well supported by this proposal -- while you can
make `x.ComputeSize` work when `x` is a value expression, you can't make it work
when `x` is a reference expression. However, that use case can be supported with
the approach described in [future work](#future-properties).

### Directly rewrite all calls to interface member functions to method call intrinsics

In this proposal, the `Call` interface is given special treatment, in that
invoking its method is rewritten into a primitive operation rather than going
through the customizable member binding that other interfaces use. This is
described in the [Details](#details) and
[Instance interface members](#instance-interface-members) sections.

In a comment on #3720
([1](https://github.com/carbon-language/carbon-lang/pull/3720#discussion_r1597324225),
[2](https://github.com/carbon-language/carbon-lang/pull/3720/files#r1597324225)),
we considered the possibility that invoking any interface member would be
directly rewritten into a primitive operation. We realized the downside of this
approach in
[open discussion on 2024-05-16](https://docs.google.com/document/d/1s3mMCupmuSpWOFJGnvjoElcBIe2aoaysTIdyczvKX84/edit?resourcekey=0-G095Wc3sR6pW1hLJbGgE0g&tab=t.0#heading=h.3qlpp5e56u46),
that this would not allow interface members to support overloading.
