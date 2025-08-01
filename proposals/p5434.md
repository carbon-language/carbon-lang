# `ref` parameters, arguments, returns and `val` returns

<!--
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

[Pull request](https://github.com/carbon-language/carbon-lang/pull/5434)

<!-- toc -->

## Table of contents

-   [Abstract](#abstract)
-   [Problem](#problem)
-   [Background](#background)
-   [Proposal](#proposal)
    -   [`ref` bindings](#ref-bindings)
    -   [`ref` and `val` returns](#ref-and-val-returns)
-   [Details](#details)
    -   [Compound return forms and patterns](#compound-return-forms-and-patterns)
    -   [Nested binding patterns](#nested-binding-patterns)
    -   [Mututation restriction on objects bound to a value](#mututation-restriction-on-objects-bound-to-a-value)
    -   [No optimization on erroneous behavior](#no-optimization-on-erroneous-behavior)
    -   [`bound` parameters](#bound-parameters)
    -   [How addresses interact with `ref`](#how-addresses-interact-with-ref)
    -   [Improved interop and migration with C++ references](#improved-interop-and-migration-with-c-references)
    -   [Part of the expression type system, not object types](#part-of-the-expression-type-system-not-object-types)
    -   [Interaction with `returned var`](#interaction-with-returned-var)
    -   [Use case: `Deref` interface](#use-case-deref-interface)
    -   [Use case: indexing interfaces](#use-case-indexing-interfaces)
    -   [Use case: member binding interfaces](#use-case-member-binding-interfaces)
    -   [Use case: class accessors](#use-case-class-accessors)
    -   [Type completeness](#type-completeness)
    -   [Pointer value representation](#pointer-value-representation)
-   [Future work](#future-work)
    -   [Temporary lifetimes](#temporary-lifetimes)
    -   [`ref` bindings in lambdas](#ref-bindings-in-lambdas)
    -   [Interaction with effects](#interaction-with-effects)
    -   [More precise lifetimes](#more-precise-lifetimes)
    -   [Combining with compile-time](#combining-with-compile-time)
    -   [Interaction with `Call` or other interfaces](#interaction-with-call-or-other-interfaces)
    -   [Destructuring assignment](#destructuring-assignment)
    -   [Restore `addr`](#restore-addr)
-   [Rationale](#rationale)
-   [Alternatives considered](#alternatives-considered)
    -   [No `ref`, only pointers](#no-ref-only-pointers)
    -   [Remove pointers after adding references](#remove-pointers-after-adding-references)
    -   [Allow `ref` bindings in the fields of classes](#allow-ref-bindings-in-the-fields-of-classes)
    -   [No call-site annotation](#no-call-site-annotation)
    -   [Top-level `ref` introducer](#top-level-ref-introducer)
    -   [Allow immutable value semantic bindings nested within variable patterns](#allow-immutable-value-semantic-bindings-nested-within-variable-patterns)
    -   [Remove `var` as a top-level statement introducer](#remove-var-as-a-top-level-statement-introducer)
    -   [`ref` as a type qualifier](#ref-as-a-type-qualifier)
    -   [`bound` would change the default return to `val`](#bound-would-change-the-default-return-to-val)
    -   [Other return conventions](#other-return-conventions)
    -   [`return var` with compound return forms](#return-var-with-compound-return-forms)
    -   [Other syntax for compound return forms](#other-syntax-for-compound-return-forms)
    -   [`ref` parameters allow aliasing](#ref-parameters-allow-aliasing)
    -   [`let` to mark value returns instead of `val`](#let-to-mark-value-returns-instead-of-val)
    -   [`=>` infers form, not just type](#-infers-form-not-just-type)

<!-- tocstop -->

## Abstract

-   A parameter binding can be marked `ref` instead of `var` or the default. It
    will bind to reference argument expressions in the caller and produces a
    reference expression in the callee.
    -   Unlike pointers, a `ref` binding can't be rebound to a different object.
    -   This replaces `addr`, and is not restricted to the `self` parameter.
    -   A `ref` binding, like a value binding, can't be used in fields of
        classes or structs.
    -   When calling functions, arguments to non-`self` `ref` parameters are
        also marked with `ref`.
-   The return of a function can optionally be marked `ref`, `val`, or `var`.
    These control the category of the call expression invoking the function, and
    how the return expression is returned.
    -   These may be mixed for functions returning tuple or struct forms.
-   Any parameters whose lifetime needs to contain the lifetime of the return
    must be marked `bound`.
-   The address of a `ref` binding is `nocapture` and `noalias`.
-   We mark parameters of a function that may be referenced by the return value
    with `bound`.

## Problem

Reference bindings have come up multiple times:

-   as a better alternative to `addr self: Self*`;
-   for use in [lambda captures](/docs/design/lambdas.md);
-   to model different kinds of C++ references for interop and migration;
-   to support nested bindings within a destructured `var`, see
    [issue #5250](https://github.com/carbon-language/carbon-lang/issues/5250)
    and
    [proposal #5164](https://github.com/carbon-language/carbon-lang/pull/5164);
-   for forwarding arguments while preserving
    [expression category](/docs/design/README.md#expression-categories);
-   to add a feature to pattern matching to modify things after they have been
    matched;
-   to support refactoring code without changing all the uses of a name, a
    problem we are already seeing with `self` and `addr self`, and would be a
    point of friction in local pattern matching in the future; and
-   to support breaking up an expression into pieces without altering the
    expression category of individual pieces.

Reference returns have also come up before, particularly to support operators
such as indexing `[`...`]` and other functions that should produce a reference
expression. It is desirable, though, that this not introduce new memory unsafety
concerns, due to returning a reference to something with insufficient lifetime.

In addition, we have been interested in adding other return mechanisms that
support returning values in registers in cases that our current convention
won't.

## Background

-   Carbon has
    [reference expressions](/docs/design/values.md#reference-expressions).
-   Using
    [the `addr` keyword on mutating methods to get a `self` with a pointer type](/docs/design/classes.md#methods)
    was introduced in
    [proposal #722: "Nominal classes and methods"](/proposals/p0722.md#keyword-to-indicate-pass-by-address).
-   [Leads issue #5261: "We should add `ref` bindings to Carbon, paralleling reference expressions"](https://github.com/carbon-language/carbon-lang/issues/5261)
    supports adding `ref` bindings to Carbon.
-   [LLVM's `noalias` attribute](https://llvm.org/docs/LangRef.html#function-attributes)
    is used to mark a pointer as being aliased in only limited ways to enable
    optimization. Also see
    [LLVM's pointer aliasing rules](https://llvm.org/docs/LangRef.html#pointer-aliasing-rules).
-   Marking a pointer as not captured, to allow optimizations, was originally
    done with
    [LLVM's `nocapture` attribute](https://releases.llvm.org/11.0.0/docs/LangRef.html#parameter-attributes),
    which has become
    [`captures(none)` and `captures(ret: address, provenance)`](https://llvm.org/docs/LangRef.html#function-attributes),
    which is governed by
    [pointer capture rules](https://llvm.org/docs/LangRef.html#pointer-capture).
-   Clang allows C++ code to use the
    [`clang::lifetimebound` attribute](https://clang.llvm.org/docs/AttributeReference.html#lifetimebound)
    to mark parameters that may be referenced by the return value, in order to
    detect some classes of use-after-free memory-safety bugs.
-   [C++ has reference types](https://en.cppreference.com/w/cpp/language/reference).

## Proposal

### `ref` bindings

We introduce a new keyword `ref`. This may be added to a `:` binding to mark it
as binding to a reference expression, as in:

```carbon
fn F(ptr: i32*) {
  // A reference binding `x`.
  let ref x: i32 = *ptr;

  // Use of `x` is a reference expression that
  // refers to the same object as `*ptr`.
  Assert(&x == ptr);

  // Equivalent to `*ptr += 1;`.
  x += 1;
}

fn G() {
  var y: i32 = 2;
  F(&y);
  Assert(y == 3);
}
```

The use of the name (`x` in the example) of a `ref` binding forms a durable
reference expression. We ensure that reference expressions formed by way of
reference bindings _do not dangle_. A `ref` binding may only bind to a durable
reference expression or an expression that can be converted to one. The bound
durable reference expression must outlive the `ref` binding.

The address of a `ref` bound name gives the address of the bound object, so
`&x == ptr` above. The reference itself does not have an address, and unlike a
pointer can't be rebound to reference a different object.

We remove `addr`, and use instead use `ref` for the `self` parameter when an
object is required. Note that the type will change from `Self*` to `Self` in
this case. In the future, we might
[re-add `addr` back if needed](#restore-addr).

```carbon
class C {
  // ❌ No longer valid.
  fn OldMethod[addr self: Self*]() {
    // Previously would dereference `self` in
    // the body of the method.
    self->x += 3;
  }

  // ✅ Now valid.
  fn NewMethod[ref self: Self]() {
    // Now `self` is a reference expression,
    // and is not dereferenced.
    self.x += 3;
  }

  // ✅ Other uses are unchanged.
  fn Get[self: Self]() -> i32 {
    return self.x;
  }

  var x: i32;
}
```

Potentially abbreviating the syntax further (to allow `ref self` as a short form
of `ref self: Self`) is left as future work.

The `ref` modifier is allowed on `:` bindings that are not:

-   inside a `var` pattern,
-   a field of a `class` type, or
-   a field of a struct type.

```carbon
fn AddTwoToRef(ref x: i32) {
  x += 1;
  let ref y: i32 = x;
  y += 1;
}

// Equivalent to:
fn AddTwoToRef(ref x: i32) {
  x += 1;
  let y_ptr: i32* = &x;
  *y_ptr += 1;
}
```

We add support for `ref` and `var` in a
[struct pattern](/docs/design/pattern_matching.md#struct-patterns) when using
the shorthand `a: T` syntax for `.a = a: T`:

```carbon
let {var a: i32, ref b: i32} = ...;

// Now equivalent to:
let {.a = var a: i32, .b = ref b: i32} = ...;
```

> Note: This takes us one step closer to `{` ambiguity. Previously we could
> distinguish between a struct literal/pattern and a non-empty block with only
> up to two tokens of lookahead (the struct cases start with `.` or `_` or
> identifier followed by `:`, and the block cases don't). Now we have things
> like:
>
> ```carbon
> fn F() -> X { var a: i32 = 0; }
> ```
>
> ... where we're getting incrementally closer to ambiguity. We've got a few
> more steps before we get there, though, since we don't have an `X{...}`
> expression yet, and `var ...` is only allowed in struct patterns rather than
> struct expressions. So we're still fine, but this is cutting down our options
> for future syntactic expansion a little.

The `ref` modifier is forbidden on the bindings in `class` or struct type
fields.

```
var outer_size: i32 = 123;

class Invalid {
  // ❌ Invalid. We don't currently have runtime `let` bindings in classes,
  // or `ref` on `var`s, but the intent is to not have `ref` bindings as fields.
  let ref invalid_ref_field: i32 = outer_size;
}

// ❌ Invalid.
var invalid_struct_type_field:
    {ref .invalid: i32} = {.invalid = outer_size};
```

In a function argument list, arguments to non-`self` `ref` parameters are also
marked with `ref`. Continuing the example:

```carbon
var z: i32 = 3;
AddTwoToRef(ref z);
Assert(z == 5);

// No `ref` though on the `self` argument.
var c: C = {.x = 4};
c.NewMethod();
Assert(c.Get() == 7);
```

> **Note:** It is important that this restriction is syntactic, not just
> semantic, because it means that `ref` is never the first token of a full
> expression, and so we know without lookahead that a `ref` in a pattern context
> must be the start of a binding pattern, not the start of an expression
> pattern.

Normally an argument to a non-`ref` parameter should not be marked `ref`, but it
is allowed in a generic context where the parameter may sometimes be `ref`.

Expression operators will mostly not take `ref` parameters, with these
exceptions:

-   [the address-of operator](/docs/design/expressions/pointer_operators.md)
    `&`;
-   the first operand of
    [the indexing operator](/docs/design/expressions/indexing.md) `[`...`]`; and
-   [the member access operator](#use-case-member-binding-interfaces) introduced
    in
    [proposal #3720: "Member binding operators"](https://github.com/carbon-language/carbon-lang/pull/3720)
    `.`.

The statement operators now use `ref` instead of pointers:

-   the left-hand operand of [assignment operators](/docs/design/assignment.md)
    such as `=` and `+=`.
-   [the `++` and `--` operators](/docs/design/assignment.md).

Even in these cases, the arguments will not be marked with `ref` at the call
site. (Generally the `ref` parameter is the `self` parameter, and so wouldn't be
marked. The exception is `BindToRef`, but we don't want to mark its argument
with `ref`.)

As an _experiment_, we are saying a pointer formed by taking the address of a
`ref` bound name is LLVM-`captures(none)` and LLVM-`noalias`. This means that
while a `ref` parameter could be passed into a function by address, the
restrictions also allow a "move-in-move-out" approach (once we define the move
operation), assuming it is not [marked `bound`](#bound-parameters). The intent
here is to leave the door open to a calling convention using registers and less
indirection for small-enough objects.

This means that the following code is invalid:

```carbon
fn F(ref a: i32, ref b: i32) -> bool;

fn G() -> bool {
  var v: i32 = 1;
  return F(ref v, ref v);
}
```

Enforcing this restriction will be part of the memory safety story. Until then,
doing this is erroneous behavior. This
[means](#no-optimization-on-erroneous-behavior) that the compiler won't use
those LLVM attributes unless the compiler can itself prove that the restrictions
hold.

### `ref` and `val` returns

The return of a function can optionally be marked `ref` or `val`. These control
the category of the call expression invoking the function, and how the return
expression is returned.

```carbon
var global: i32 = 2;
fn ReturnRef() -> ref i32 {
  // ❌ Invalid: return 2;

  // ✅ Valid: return a reference expression with
  // sufficient lifetime.
  return global;
}
// Call `ReturnRef` and use the resulting reference.
ReturnRef() += 3;
Assert(global == 5);

// Result of `ReturnRef` can be bound using a `ref`
// binding.
fn AddFive() {
  let ref r: i32 = ReturnRef();
  r += 5;
}
AddFive();
Assert(global == 10);

fn ReturnVal() -> val i32 {
  return 2;
}
// ReturnVal() is a value expression.
let l: i32 = ReturnVal();

// Returning an initializing expression is the
// default.
fn ReturnDefault() -> i32 {
  return 2;
}
// ReturnDefault() is an initializing expression.
var j: i32 = ReturnDefault();

// Use `var` to explicitly specify returning an
// initializing expression.
fn ReturnVar() -> var i32 {
  return 2;
}
// `ReturnVar()` is the same as `ReturnDefault()`.
```

-   A call to a function declared `-> ref T` is a durable reference expression.
    The generated code for that function will return the address of a `T`
    object.
-   A call to a function declared `-> val T` is a value expression. The function
    will return the value representation of `T`. Since values have no address,
    the value representation may be returned in registers.
-   The behavior of a call to a function declared `-> T` is unchanged. It is an
    initializing expression, returning in place or by copy depending on the
    initializing representation of `T`. This is the same behavior as `-> var T`.
-   The behavior of `auto` as the return type is unchanged, but now supports an
    optional `ref`, `val`, or `var` between the `->` and `auto`. `-> auto`
    continues to return an initializing expression, as does `-> var auto`.
    `-> val auto` returns a value expression, and `-> ref auto` returns a
    durable reference expression.
-   Using `=>` to specify a return continues to return an initializing
    expression, as before. See
    [this relevant alternative considered](#-infers-form-not-just-type).

A function may have multiple returns, each with their own marker, by using a
tuple or struct compound return form.

```carbon
fn TupleReturn() -> (val bool, ref i32, C) {
  return (true, global, {.x = 3});
}

fn StructReturn()
    -> {.a: val bool,
        .b: ref i32,
        .c: C} {
  return {.a = true,
          .b = global,
          .c = {.x = 3}};
}
```

If the return of a function may reference the storage of one or more parameters
to the function, those parameters must be marked `bound`. This allows the
compiler to diagnose if the function's return is used after the lifetime of the
`bound` parameter ends. The semantics of `bound` are intended to match the
[`clang::lifetimebound` attribute](https://clang.llvm.org/docs/AttributeReference.html#id8).

```carbon
fn Member(bound ref c: C) -> ref i32 {
  return c.x;
}

// Lifetime of a pointer includes the lifetime
// of what it points to.
fn Deref(bound p: i32*) -> ref i32 {
  return *p;
}

fn Both(bound pc: C*) -> ref i32 {
  return p->x;
}

fn Invalid1() -> ref i32 {
  var x: i32 = 4;
  // ❌ Error: returning reference to `x`
  // whose lifetime ends when this function
  // returns.
  return x;
}

fn Invalid2() -> ref i32 {
  var c: C = {.x = 1};
  // ❌ Error: returning reference bound to `c`
  // whose lifetime ends when this function
  // returns.
  return Member(c)
}
```

The address of a `bound ref` parameter is the
[LLVM attribute `captures(ret: address, provenance)`](#background) instead of
[`captures(none)`](#background).

## Details

The intent is that we would encourage using references instead of pointers when
possible. Their benefits are related to their limitations, so to get those
benefits we should use them when a use is restricted enough to be within those
limitations.

### Compound return forms and patterns

Mirroring the [tuple](/docs/design/pattern_matching.md#tuple-patterns) and
[struct](/docs/design/pattern_matching.md#struct-patterns) pattern forms, we
also support tuple and struct return forms.

```carbon
// `->` begins a "return form"
fn F()     -> <return-form>;

// Within any return form, if the first token is
// `val`, `ref`, `var`, `(`, or `{`, it is not
// treated as type expression:

// Value return, with type as specified
fn Val()   -> val <type-expr>

// Reference return, with type as specified
fn Ref()   -> ref <type-expr>

// Initializing return, with type as specified
fn Var()   -> var <type-expr>

// Tuple compound return, with a list of
// return forms.
fn TupleCompound() -> ( <return-form>, ... )
// Tuple return, with a list of type
// expressions. Used if all members of the
// list are type expressions.
fn Tuple() -> ( <type-expr>, ... )

// Struct compound return, with a mapping from
// designators to return forms.
fn StructCompound()
    -> { .<id>: <return-form>, ... }
// Struct return, used if all of the members
// are type expressions.
fn Struct() -> { .<id>: <type-expr>, ... }

// Otherwise, implicit `var` means returns
// an initializing expression.
fn Other() -> <type-expr>
```

Note that in the absence of `val`, `ref`, and `var` keywords, the implicit `var`
is placed in the outermost position, minimizing the number of primitive forms
returned. So `fn F() -> (i32, i32)` means `fn F() -> var (i32, i32)` not
`fn F() -> (var i32, var i32)`. Generally the `var` is left off if not required,
and so will be rare in return forms, to minimize confusion with `val`.

```carbon
fn TupleReturn(...)
    -> (val bool, ref i32, C);
// Equivalent to:
//  -> (val bool, ref i32, var C);

let (a: bool, ref b: i32, var c: C)
    = TupleReturn(...);

fn StructReturn(...)
    -> {.a: val bool,
        .b: ref i32,
        .c: C};
// Equivalent to:
//  -> {.a: val bool,
//      .b: ref i32,
//      .c: var C};

// Binds to the names `x`, `y`, `z`:
let {.a = x: bool,
     .b = ref y: i32,
     .c = var z: C} = StructReturn(...);

// Binds to the names `a`, `b`, `c`:
let {a: bool,
     ref b: i32,
     var c: C} = StructReturn(...);

// Above two can be mixed, binding to
// names `a`, `y`, `z`.
let {a: bool,
     .b = ref y: i32
     .c = var z: C} = StructReturn(...);
```

Only types are allowed after a `-> val`, `-> ref`, or `-> var`, not a compound
return form. Examples:

```carbon
// Returns a tuple of type
// `(bool, f32, C, i32)`.
fn OneTupleReturn(...)
    -> (bool, f32, C, i32);

// Returns a compound tuple form
fn CompoundReturn(...)
    -> (bool, val f32);
// Equivalent to:
//  -> (var bool, val f32);

// ❌ Invalid, can't specify `ref` inside
// of `val`.
fn Invalid(...) -> val (bool, ref f32);
```

The compound return forms may be nested, as in:

```carbon
fn CompoundInParens(...)
    -> ({.a: bool, .b: val f32}, C, ref i32);
// Equivalent to:
//  -> ({.a: var bool, .b: val f32}, var C, ref i32);

let ({.a = var x: bool, .b = val y: f32},
     var c: C, ref d: i32) = CompoundInParens(...);
// or without renaming:
let ({var a: bool, b: f32},
     var c: C, ref d: i32) = CompoundInParens(...);

// Contrast with a compound tuple form containing
// a struct type (not compound):
fn StructInParens(...)
    -> ({.a: bool, .b: f32}, C, ref i32);
// Equivalent to:
//  -> (var {.a: bool, .b: f32}, var C, ref i32);

fn CompoundInBraces(...)
    -> {.a: bool, .b: (val f32, C), .c: ref i32};
// Equivalent to:
    -> {.a: var bool, .b: (val f32, var C), .c: ref i32};

let {a: bool,
     .b = (x: f32, var y: C),
     ref c: i32} = ParensInBraces(...);

// Contrast with a compound struct form containing
// a tuple type (not compound):
fn TupleInBraces(...)
    -> {.a: bool, .b: (f32, C), .c: ref i32};
// Equivalent to:
    -> {.a: var bool, .b: var (f32, C), .c: ref i32};
```

This feature is intended to support cases like `enumerate` that will want to
return a value for the index but a reference to the element of the sequence
being enumerated.

### Nested binding patterns

Since a `ref` binding may only bind to a durable reference expression, it can't
be used to bind the result of a function returning an initializing expression.
However, if the initializing expression is bound to a `var`, any nested patterns
are reference binding patterns bound to the subobject, following
[proposal #5164: "Updates to pattern matching for objects"](https://github.com/carbon-language/carbon-lang/pull/5164).

For example:

```carbon
fn F() -> (bool, (C, i32));
let (b: bool, var (c: C, i: i32)) = F();
```

is equivalent to:

```carbon
fn F() -> (bool, (C, i32));
let (b: bool, var v: (C, i32)) = F();
let ref c: C = v.0;
let ref i: i32 = v.1;
```

Note that `ref` is disallowed inside `var` since that would be redundant.

### Mututation restriction on objects bound to a value

Mutation of objects with a non-copy-value representation in an active value
binding ("borrowed objects") is erroneous behavior.

-   Our plan is to prevent mutation of borrowed objects in Carbon's strict safe
    dialect.
    -   We should only relax our stance here and consider making such mutation
        _allowed_ if we discover difficulty with this that we cannot overcome.
    -   But we _should_ revisit the underlying idea of mutation being erroneous
        if enforcing it in the strict mode proves fundamentally untenable due to
        ergonomic costs.
-   There will always be the potential for unchecked code, either unsafe Carbon
    code or C++ code by way of interop, to mutate a borrowed object, hence the
    need to define it as erroneous behavior.
-   There is no need to ever make it anything more than erroneous behavior, see
    below.
-   If we can prove the mutation doesn't occur, then we can use that to optimize
    under "as-if", and we don't need anything else.
-   We are deferring the decision of whether strict enforcement is enabled in
    Carbon's current C++-friendly mode, when not explicitly marking the code as
    "unsafe."

This was
[discussed 2025-05-22](https://docs.google.com/document/d/1Yt-i5AmF76LSvD4TrWRIAE_92kii6j5yFiW-S7ahzlg/edit?tab=t.0#heading=h.uot97ukynlsi)
and then made the subject of
[leads issue #5524](https://github.com/carbon-language/carbon-lang/issues/5524).

### No optimization on erroneous behavior

The Carbon compiler should not optimize on erroneous behavior, ever, unless the
compiler literally proves that it does not occur, ever. In which case, we don't
need any license to start optimizing on this, as it falls under "as-if".

The fact that undefined behavior ("UB",
[cppreference](https://en.cppreference.com/w/cpp/language/ub.html),
[wikipedia](https://en.wikipedia.org/wiki/Undefined_behavior)) provides "as-if"
_without_ a proof is precisely the risk of using UB for any semantics, and why
we don't use it here and elsewhere we use erroneous behavior.

### `bound` parameters

It is erroneous behavior to return something that references a local object that
won't live once the function returns, even if it is a parameter marked `bound`.
Local objects include local variables, temporary objects, and `var` parameters.

```carbon
fn Invalid1() -> i32* {
  var x: i32 = 4;
  // ❌ Invalid.
  return &x;
}

fn Invalid2(bound x: i32) -> ref i32 {
  var y: i32 = x;
  // ❌ Invalid.
  return y;
}

// ✅ Valid
fn Valid1(bound p: i32*) -> ref i32 {
  return *p;
}

fn Invalid3(bound var x: i32) -> ref i32 {
  // ❌ Invalid: lifetime of `var` parameter
  // ends when function returns.
  return x;
}

class ReturnMember {
  // ✅ Valid
  fn ValidRef[bound ref self: Self]() -> ref i32 {
    return self.m;
  }

  // ❌ Invalid: can't return reference to value.
  fn InvalidVal[bound self: Self]() -> ref i32 {
    return self.m;
  }

  // ❌ Invalid: `var self` lifetime ends.
  fn InvalidVar[bound var self: Self]()
      -> ref i32 { return self.m; }

  var m: i32;
}

class DerefPointerMember {
  // ✅ Valid
  fn ValidRef[bound ref self: Self]() -> ref i32 {
    return *self.pm;
  }

  // ✅ Valid
  fn ValidVal[bound self: Self]() -> ref i32 {
    return *self.pm;
  }

  // ✅ Valid
  fn ValidVar[bound var self: Self]()
      -> ref i32 { return *self.pm; }

  var pm: i32*;
}
```

Otherwise, `bound` parameters and global variables are the sources of storage
that can be referenced by a return, but need not be referenced, particularly not
on every code path.

```carbon
// Result references `r` if `b` is true, and `p`
// otherwise. Valid as long as both are marked `bound`.
fn Conditional(b: bool, bound ref r: C, bound p: C*)
    -> ref C {
  if (b) {
    return r;
  } else {
    return *p;
  }
}
```

The parameters of functions defined in an interface may also be marked as
`bound`. The `impl` of that interface for a type can omit occurrences of `bound`
from the interface, but cannot add new ones.

```carbon
interface I {
  fn F[bound ref self: Self]
      (a: Self, bound b: Self, bound c: Self*)
      -> ref Self;
}

impl C1 as I {
  // ✅ Valid: matches interface
  fn F[bound ref self: Self]
      (a: Self, bound b: Self, bound c: Self*)
      -> ref Self;
}

impl C2 as I {
  // ✅ Valid: proper subset of `bound` params
  fn F[ref self: Self]
      (a: Self, bound b: Self, c: Self*)
      -> ref Self;
}

impl C2 as I {
  // ❌ Invalid: `a` is not bound in `I.F`.
  fn F[ref self: Self]
      (bound a: Self, b: Self, c: Self*)
      -> ref Self;
}
```

Like `[[clang::lifetimebound]]` in C++, `bound` does not affect semantics or
calling conventions, just what code is legal. This helps avoid mismatches
between typechecking against the signatures in an interface when the `impl`
functions are different. Exception: the question of whether `bound` affects the
lifetime of temporaries is [future work](#temporary-lifetimes).

Note that all combinations of a `val`/`ref`/default return can be bound to a
value/`ref`/`var` parameter. Examples:

```carbon
fn RefToVal(bound ref x: C) -> val D { return x.d; }
fn ValToRef(bound y: C) -> ref D { return *y.ptr; }
fn VarToRef(bound var p: i32*) -> ref i32 { return *p; }
fn VarToDefault(bound var p: i32*) -> i32* { return p; }
```

For full safety, we need each bound parameter to be immutable for the duration
of the lifetime of the returned result. However, the objective for now is only
matching `[[clang::lifetimebound]]`, which has the goal of preventing some
classes of bugs, not full memory safety. We will reconsider this with the memory
safety design.

Clang's `lifetimebound` attribute also only applies to the immediately pointed
to objects (by pointers or reference parameters, or pointers or reference
subobjects of an aggregate parameter). We suggest a simpler, transitive model
here that is more restrictive but should be compatible. That said, pinning down
the exact and firm semantics of `bound`, especially in these complex cases, is
deferred to the full memory safety design as well.

### How addresses interact with `ref`

The address of a `ref` binding is `noalias` and either `captures(none)` or
`captures(ret: address, provenance)`, depending on whether the binding is marked
`bound`.

-   `noalias` means like C `restrict`; you can't observe mutations through
    aliases; mutation through a restricted pointer is not observable through
    another pointer
-   `captures(none)` means there is no transitive escape: you can pass a
    nocapture pointer to another nocapture function, but you can't store to
    memory or return
-   `captures(ret: address, provenance)`: is like `captures(none)` but may be
    referenced by a return.

The combination of `noalias` and `captures(none)` semantics are the minimum for
the "move-in-move-out" optimization. But this condition is hard to check, so
safe code will use a stricter criteria. Unsafe code will be required to adhere
to just the `noalias` restrictions, but will not be checked (except possibly by
a sanitizer at runtime). The details here will be tackled as part of the memory
safety design.

Optimizations will only be performed based on information that is enforced or
checked by the compiler, so these attributes won't be passed to LLVM unless
their requirements can be established. This avoids introducing undefined
behavior, which we particularly don't want to do in situations where C++
doesn't.

The goal of these rules it to nudge us towards function boundaries that don't
constructively create aliasing in their API boundary and don't capture pointers
unnecessarily.

These restrictions are experimental, and we should keep track of everything we
end up needing to do to work around these restrictions so any reconsideration
can be properly informed.

### Improved interop and migration with C++ references

We expect this to improve interop and migration by allowing significantly more
interface similarity between Carbon and C++. Previously, many things in C++ that
used references on interface boundaries would be forced to switch to pointers.
This adds ergonomic friction both at a basic level because of the forced change
but also a deeper level because it will make it significantly harder to see the
parallel usage across the boundary between C++ and Carbon. With reference
bindings, the vast majority of this dissonance will be removed.

This does create a migration concern, raised in
[open discussion on 2025-05-01](https://docs.google.com/document/d/1Yt-i5AmF76LSvD4TrWRIAE_92kii6j5yFiW-S7ahzlg/edit?tab=t.0#heading=h.dffumsu6wzlc),
that the `nocapture` and `noalias` modifiers don't match C++ restrictions,
particularly on the `this` parameter that we are going to require migrate to
`ref self`. We may have to add back in `addr` to allow a different pointer type
for those cases.

> **Future work:** Addressing how we model the various kinds of C++ references
> that Carbon code may need to interact with is
> [something we are actively considering](https://docs.google.com/document/d/1l5TbNuwZEcwm96ejGPLn9GdoQO1fByUW0tFRLU9BqXE/edit?tab=t.0)
> and will be tackled in a future proposal.

### Part of the expression type system, not object types

Much like value/`val` and `var` bindings, `ref` binding and the new return forms
are are part of the type system, but only through expression categories,
patterns (function parameters and so on), and returns. Specifically, we don't
expect them to be part of the _object types_ in Carbon. Like value bindings, we
retain a great deal of implementation flexibility around layout, and the
specifics of how they are lowered.

This specifically means we will need to incorporate `ref` bindings into the
`Call` interface and we will be adding complexity there that will need to be
handled by overloading. The changes to the `Call` interface is future work, and
overloading, once we add support, will need to carry additional complexity to
handle `ref`.

### Interaction with `returned var`

The rule is: `returned var` may only be used when there is a single atomic
return form, and it is the default `var` category.

```carbon
// ✅ Allowed
fn F(...) -> V {
  returned var v: V = ...;
  // ...
  return var;
}

fn F(...) -> {var .a : T} {
  // ❌ Invalid: composite form
  returned var ret: T = ...;
  // ...
}

fn F(...) -> val T {
  // ❌ Invalid: value return
  returned var ret: T = ...;
  // ...
}
```

We can revisit and expand this later if this does not handle use cases we would
like to support.

### Use case: `Deref` interface

To support customization of the prefix-`*` dereferencing operator, we introduce
the `Deref` interface.

```carbon
interface Deref {
  let Result:! type;
  fn Op[bound ref self: Self]() -> ref Result;
}

final impl forall [T:! type] T* as Deref {
  where Result = T;
  fn Op[bound self: Self]() -> ref T
      = "builtin.deref";
}
```

Then `*p` is rewritten to `p.(Deref.Op)()`, and `p->m` is rewritten to
`p.(Deref.Op)().m`. For example, this might be used by a smart pointer:

```
class SmartPtr(T:! type) {
  fn Make(p: T*) -> Self { return {.ptr = p}; }
  impl as Deref {
    where Result = T;
    fn Op[bound ref self: Self]() -> ref Result {
      return *self.ptr;
    }
  }
  private var ptr: T*;
}
```

### Use case: indexing interfaces

[Proposal #2274: "Subscript syntax and semantics"](https://github.com/carbon-language/carbon-lang/pull/2274)
added the interfaces used to support indexing with the subscripting operator
`[`...`]`. We change these in the following ways:

-   The `addr self` parameters are changed to `bound ref self`, to allow the
    result to reference the `self` object.
-   The `At` method returns by `val`.
-   The `Addr` methods are renamed `Ref` and return a reference instead of a
    pointer that is automatically dereferenced.

[This proposal's PR](https://github.com/carbon-language/carbon-lang/pull/5434)
makes those changes to the
[indexing design](/docs/design/expressions/indexing.md).

### Use case: member binding interfaces

The member binding interface used for reference expressions from
[proposal #3720](https://github.com/carbon-language/carbon-lang/pull/3720) can
now be changed to use references instead of pointers.

Before:

```carbon
// For a reference expression `x` with type `T`
// and an expression `y` of type `U`, `x.(y)` is
// `*y.((U as BindToRef(T)).Op)(&x)`
interface BindToRef(T:! type) {
  extend impl as Bind(T);
  fn Op[self: Self](p: T*) -> Result*;
}
```

After:

```carbon
// For a reference expression `x` with type `T`
// and an expression `y` of type `U`, `x.(y)` is
// `y.((U as BindToRef(T)).Op)(x)`
interface BindToRef(T:! type) {
  extend impl as Bind(T);
  fn Op[self: Self](bound ref p: T) -> ref Result;
}
```

Similarly, the `BindToValue` interface is changed to use a `val`/value return,
potentially avoiding a copy of large objects.

Before:

```carbon
interface BindToValue(T:! type) {
  extend Bind(T);
  fn Op[self: Self](x: T) -> Result;
}
```

After:

```carbon
interface BindToValue(T:! type) {
  extend Bind(T);
  fn Op[self: Self](bound x: T) -> val Result;
}
```

### Use case: class accessors

A `ref` return can be used to expose the state of an object in a way that can be
mutated:

```carbon
class Four {
  fn Get[self: Self](i: i32) -> i32 {
    Assert(i >= 0 and i < 4);
    return self.m[i];
  }
  fn GetMut[bound ref self: Self](i: i32) -> ref i32 {
    Assert(i >= 0 and i < 4);
    return self.m[i];
  }
  private var m: array(i32, 4);
}

var x: HasMember = {.m = (0, 2, 4, 6)};
x.GetMut(2) += 1;
fn Check(y: Four) {
  Assert(y.Get(2) == 5);
}
Check(x);
```

**Future work**: this will in the future often be done with an overloaded
method, as in:

```carbon
class Four {
  overload Access {
    fn [bound ref self: Self](i: i32) -> ref i32 {
      Assert(i >= 0 and i < 4);
      return self.m[i];
    }
    fn [self: Self](i: i32) -> i32 {
      Assert(i >= 0 and i < 4);
      return self.m[i];
    }
  }
  private var m: array(i32, 4);
}

var x: HasMember = {.m = (0, 2, 4, 6)};
x.Access(2) += 1;
fn Check(y: Four) {
  Assert(y.Access(2) == 5);
}
Check(x);
```

This may be a common enough use case that we will want to introduce a dedicated
syntax:

```carbon
class HasMember {
  fn Access[bound ref? self: Self](i: i32) -> ref? i32 {
    Assert(i >= 0 and i < 4);
    return self.m[i];
  }
  private var m: array(i32, 4);
}
```

### Type completeness

Not a change by this proposal, but note that our existing rules will require the
type in a `ref` binding to be complete in situations where it would not need to
be if you were using a value binding with a pointer type instead. We may need to
change this in the future to match C++ which treats reference types more like
pointer types for completeness purposes.

After this change, a `ref` binding to type `T` will require `T` to be complete
in the same situations that other bindings to type `T` require `T` to be
complete.

### Pointer value representation

Purely as a change in syntax, the way to specify that
[the value representation of a class](/docs/design/values.md#value-representation-and-customization)
uses a pointer is changed from writing `const Self *` to `const ref`.

## Future work

### Temporary lifetimes

For safety, we need bindings and returns that reference storage to only be used
while that storage remains valid. When the referenced storage is owned by a
temporary, we have a choice to either control the lifetime of the temporary or
diagnose when the lifetime of the temporary is insufficient. Deciding on our
policy is future work.

Note that in many cases we can explicitly provide storage in a variable instead
of referencing a temporary. For example, using
`var x: ... = ReturnsATemporary();` instead of
`let ref x: ... = ReturnsATemporary();`. This won't apply in all situations,
though, such as temporaries that are reachable transitively through pointers.

### `ref` bindings in lambdas

We have already identified
[future work to support reference captures in lambdas as part of proposal #3848](/proposals/p3848.md#future-work-reference-captures).
This might be a reason to support `ref` bindings as fields of objects, with all
the restrictions that comes with that.

### Interaction with effects

We still need to determine how references and the other return types interact
with effects, like `Optional`, errors, co-routines, and so on. For example, we
don't want to give up the benefits of being able to directly return a reference
when a function has an error path.

It is unclear if this will mean putting references into the object type system,
but we may be able to handle this with additional types or the ability to
customize return representations. For example, we might have an alternate
version of the `Optional` type that holds a reference:

```carbon
class OptionalRef(T:! type) {
  fn Make(bound ref r: T) -> Self {
    return {.p = &r};
  }
  fn MakeEmpty() -> Self {
    return {.p = Optional(T*).MakeEmpty()};
  }
  fn HasValue[self: Self]() -> bool {
    return p.HasValue();
  }
  fn Get[bound ref self: Self]() -> ref Result {
    Assert(self.HasValue());
    return *self.p.Get();
  }

  private var p: Optional(T*);
}
```

### More precise lifetimes

More precise lifetime tracking will be considered with the memory safety design.
For example, the `bound` approach does not distinguish different components of a
compound return, or different parts of a parameter object that might have
different lifetimes.

### Combining with compile-time

We plan to support references to compile-time state when executing a function at
compile time. That will be part of a future proposal.

### Interaction with `Call` or other interfaces

For now, `ref` is not represented in the `Call` interface introduced in
[proposal #2875: Functions, function types, and function calls](https://github.com/carbon-language/carbon-lang/pull/2875).
This will be tackled together in a future proposal with other aspects of
bindings not represented by the type, such as `var` and compile-time, along with
being generic across these aspects of bindings.

### Destructuring assignment

Having more support for multiple returns from a function opens the question of
how to do different things with the different returns. We may want a syntax for
saying some of the returns are bound to new names, and some are used in
assignments to existing variables. One possibility would be to have some pattern
syntax for re-initializing an existing object, as in:

```carbon
fn F() -> (-> bool, ->T);
fn G() {
  var x: T = ...;
  Consume(~x);
  let (b: bool, init x) = F();
  // Continue to use `x`...
}
```

This was discussed in
[the #syntax channel on Discord on 05-19-2025](https://discord.com/channels/655572317891461132/709488742942900284/1374126123595727000).

### Restore `addr`

There are two reasons we might restore `addr` as an alternative to `ref` for the
implicit `self` parameter:

-   As a way to express uses of `self` that we want to disallow for `ref`
    bindings generally but need an escape hatch for migrated C++ code that
    relies on these patterns.
-   To allow the `self` parameter to use one of the pointer semantics we create
    as part of the upcoming memory safety design, that can't be achieved with
    `ref`.

## Rationale

This proposal tries to advance these [Carbon goals](/docs/project/goals.md):

-   [Performance-critical software](/docs/project/goals.md#performance-critical-software)
    -   Having a "move-in-move-out" option as a calling convention is a
        potential performance improvement for using `ref` parameters instead of
        pointers.
    -   Giving additional options for the return convention gives opportunities
        for improved performance. Having this set by explicit return markings is
        about giving control and predictability to the code author.
-   [Code that is easy to read, understand, and write](/docs/project/goals.md#code-that-is-easy-to-read-understand-and-write)
    -   `ref` bindings and returns avoid the ceremony of round-tripping through
        pointers.
-   [Practical safety and testing mechanisms](/docs/project/goals.md#practical-safety-and-testing-mechanisms)
    -   Checking that reference bindings are not dangling is important for
        avoiding use-after-free bugs.
    -   `bound` markings on parameters to allow safety equivalent to Clang's
        `[[clang::lifetimebound]]` when returning a reference.
-   [Interoperability with and migration from existing C++ code](/docs/project/goals.md#interoperability-with-and-migration-from-existing-c-code)
    -   Including references in Carbon allows for less mismatch for C++ code
        using references.

## Alternatives considered

These ideas were discussed in open discussion on:

-   [2025-05-01](https://docs.google.com/document/d/1Yt-i5AmF76LSvD4TrWRIAE_92kii6j5yFiW-S7ahzlg/edit?tab=t.0#heading=h.dffumsu6wzlc)
-   [2025-05-06](https://docs.google.com/document/d/1Yt-i5AmF76LSvD4TrWRIAE_92kii6j5yFiW-S7ahzlg/edit?tab=t.0#heading=h.s42g5iv67d3c)
-   2025-05-07
    [a](https://docs.google.com/document/d/1Yt-i5AmF76LSvD4TrWRIAE_92kii6j5yFiW-S7ahzlg/edit?tab=t.0#heading=h.sfx9d7ltud5)
    [b](https://docs.google.com/document/d/1Yt-i5AmF76LSvD4TrWRIAE_92kii6j5yFiW-S7ahzlg/edit?tab=t.0#heading=h.4zbo49wg5rmk)
-   [2025-05-08](https://docs.google.com/document/d/1Yt-i5AmF76LSvD4TrWRIAE_92kii6j5yFiW-S7ahzlg/edit?tab=t.0#heading=h.vdognq1upsf5)
-   [2025-05-12](https://docs.google.com/document/d/1Yt-i5AmF76LSvD4TrWRIAE_92kii6j5yFiW-S7ahzlg/edit?tab=t.0#heading=h.1mjh6unumnwu)
-   [2025-05-13](https://docs.google.com/document/d/1Yt-i5AmF76LSvD4TrWRIAE_92kii6j5yFiW-S7ahzlg/edit?tab=t.0#heading=h.bdznj2d0by2g)
-   [2025-05-14](https://docs.google.com/document/d/1Yt-i5AmF76LSvD4TrWRIAE_92kii6j5yFiW-S7ahzlg/edit?tab=t.0#heading=h.52tb7l2he343)

They were also discussed in the
[#pointers-and-references channel in Discord starting 2025-05-05](https://discord.com/channels/655572317891461132/753021843459538996/1369085231038074901),
and
[#syntax on 2025-05-14](https://discord.com/channels/655572317891461132/709488742942900284/1372285365162872943).

### No `ref`, only pointers

The rationale to add `ref` instead of staying with pointers was discussed on
[2025-05-01](https://docs.google.com/document/d/1Yt-i5AmF76LSvD4TrWRIAE_92kii6j5yFiW-S7ahzlg/edit?tab=t.0#heading=h.dffumsu6wzlc).
In addition to the motivating problems given in
[the "Problem" section](#problem), that discussion included some additional
depth to the reasons to add reference bindings.

There is a tension between wanting to have mutating expressions and only having
pointers. You need some concept like a reference in order to mutate an object
with an expression. The question is how small a box the references are
restricted to, and where the line is drawn. C has lvalues, which contain
references but are restricted to a quite small box. Reference bindings
specifically are about keeping a small box around references while still adding
enough expressivity to support our use cases. We have started with a model
similar to C, but it fell down when it comes down to composition. Decomposing an
expression into pieces loses the tools the expression provided to you. The
missing tool for that was reference bindings.

We saw how much we were leaning on value bindings. The asymmetry between having
value binding but not reference bindings when have value expressions and
reference expressions was creating pressure. For example, when accessing members
of an object, we had to escape to pointers in that operator.

One downside of this change is that before indirections were more visible in the
code.

Also, this does fundamentally mean that we now have another kind of "pointer",
potentially adding complexity to any memory-safety story. However, this ship
already sailed to some extent with value bindings. Fundamentally, bindings are
allowed to have pointer-like semantics from a lifetime perspective, and so will
need to be considered as a pointer-like thing as we build out lifetime safety.

### Remove pointers after adding references

If we removed pointers after adding references, we would need something
rebindable for assignable objects. The viable path forward without separate
pointers and references is to have something rebindable like pointers but
automatically dereferenced like references, which is the approach Rust takes.
See
[this comment on issue #5261](https://github.com/carbon-language/carbon-lang/issues/5261#issuecomment-2786462775).

One of the features of a reference is what it cannot do, and we would have to
remove those restrictions to be able to satisfy the pointer use cases with
references.

### Allow `ref` bindings in the fields of classes

A type with reference binding fields would need a lot of restrictions since
reference bindings are not assignable. We did not see enough motivation to put
references into objects, given the complexity that it would introduce, so we are
keeping references out of types for now. This could change to support lambda
reference captures.

### No call-site annotation

This question was discussed on
[2025-05-07](https://docs.google.com/document/d/1Yt-i5AmF76LSvD4TrWRIAE_92kii6j5yFiW-S7ahzlg/edit?tab=t.0#heading=h.sfx9d7ltud5).

We decided that the marking is not about lifetime, but ability to mutate. A
`val` may reference an object in a similar way to a `ref`, restricting
operations on the original object, but we are not going to mark `val`s since
those restrictions are enforced by the compiler. We thought the ability to
mutate, though, was something important enough to highlight to readers of the
code, even at the expense of extra work for the writer.

Swift `inout` parameters are
[marked at caller with an `&` before the argument](https://docs.swift.org/swift-book/documentation/the-swift-programming-language/functions/#In-Out-Parameters).
Jordon Rose has published
[a regret](https://belkadan.com/blog/2021/12/Swift-Regret-inout-Syntax/) that
they didn't use `inout` to mark the argument instead of `&`.

On the other hand, not marking is not known to be a source of bugs.

This is a "try it and see how well it works" sort of decision.

### Top-level `ref` introducer

For now, we don't believe `let ref` to be so common as to need a shorter way to
write, unlike what we do for `var`. This was considered in
[leads issue #5523](https://github.com/carbon-language/carbon-lang/issues/5523),
which provided this rationale:

> I feel like this would often be used for non-local mutation due to it
> fundamentally deporting mutable value semantics and instead having reference
> semantics. Unlike the local mutation, that seems more worthwhile to have
> incentives around minimizing.
>
> However, this seems the easiest of all to revisit later if we discover that
> the added verbosity in practice is costing more than is worth any improvements
> from explicitly flagging mutable reference semantics, or if we find code is
> reaching for antipatterns due to the incentive.

In addition,
[this comment on leads issue #5522](https://github.com/carbon-language/carbon-lang/issues/5522#issuecomment-2972029100)
argued that it would be more consistent for `ref` and `val` to only apply to
bindings, and not introduce patterns, like `let` and `var`.

### Allow immutable value semantic bindings nested within variable patterns

This was considered in
[leads issue #5523](https://github.com/carbon-language/carbon-lang/issues/5523),
which provided this rationale:

> While it may be an obvious point of orthogonality, I think it adds choice
> without sufficient motivation, and even _having_ that choice does add some
> complexity to the language.
>
> It also seems like we could add this later if there is sufficient demand when
> we have larger usage experience body to pull from with the rest of the Carbon
> language. Currently, the affordance that feels more natural to me is what we
> have.

> I think we're happy to see motivating use cases and revisit this. At the
> moment, we've just not seen motivating use cases -- everything has seemed a
> bit too contrived.

### Remove `var` as a top-level statement introducer

This was considered in
[leads issue #5523](https://github.com/carbon-language/carbon-lang/issues/5523),
which provided this rationale:

> Locals are important, frequent, and frequently mutable. I don't think forcing
> varying locals to go through `let var` for orthogonality aids readability
> enough to offset the verbosity cost of added keywords on a reasonably common
> pattern.
>
> I'm still currently in the position that locally owned objects being mutable
> should not be "discouraged" or "disincentivized" by the language. And I think
> adding artificial incentives to try and avoid needing a mutable local variable
> would either have no effect beyond verbosity, or if it _did_ have effect, it
> wouldn't be a net positive effect due to code being written in a less
> straightforward manner in order to avoid mutation.
>
> To be clear, this is based on intuition and judgement based on my experience,
> not in any way based on data or specific motivating examples. I can imagine
> data or evidence or even a new perspective changing my position here, but so
> far the discussion we've had haven't done that.

### `ref` as a type qualifier

The big concern is that any effect that is represented by a type, like
`Optional` or `Result`, will want to compose with reference returns. This could
be done by allowing `ref` to create an object type that could be used as a
parameter to those, as in `Optional(ref T)`, but
[we are trying to avoid going down that path](https://github.com/carbon-language/carbon-lang/issues/5261#issuecomment-2790421894).
We have [future work](#interaction-with-effects) to tackle this problem
specifically.

There was also
[a concern that we might need `ref` types to represent argument lists with tuples](https://github.com/carbon-language/carbon-lang/issues/5261#issuecomment-2790506515),
but tuples already can't represent `var` or compile-time parameters. We have
other plans for this, instead of trying to stretch tuples to encompass these use
cases.

We also noted that including references in the type system led to a number of
inconsistencies in C++, such as no there not being references of references.

### `bound` would change the default return to `val`

We considered saying that `bound` would change the default return to use the
`->val` return convention. This was discussed on
[2025-05-01](https://docs.google.com/document/d/1Yt-i5AmF76LSvD4TrWRIAE_92kii6j5yFiW-S7ahzlg/edit?tab=t.0#heading=h.dffumsu6wzlc)
and
[2025-05-08](https://docs.google.com/document/d/1Yt-i5AmF76LSvD4TrWRIAE_92kii6j5yFiW-S7ahzlg/edit?tab=t.0#heading=h.vdognq1upsf5).
The idea is that `val` is expected to be efficient, so we should encourage using
it, but we can't always use `val`, since some types have a reference value
representation, but `bound` alleviates that concern.

Once we realized that `bound` is relevant for all return conventions, we
reconsidered that approach, since has a number of concerns:

-   Changing defaults is action at a distance, changing the behavior without
    changing the code in the relevant location.
-   We don't want to have to make changes to the return category of copy-paste
    of a function from an interface to an `impl` of it when removing `bound`.
-   Lifetimes in Rust and Clang's `[[clang::lifetimebound]]` don't change
    calling conventions, only what code is valid.

Going with an approach where less depends on `bound` makes sense for now, since
we are going to reconsider these issues as part of our upcoming memory safety
work.

### Other return conventions

We also considered other conventions for returning from functions, in
[a comment on #5434](https://github.com/carbon-language/carbon-lang/pull/5434/files#r2099145225),
on
[2025-05-08](https://docs.google.com/document/d/1Yt-i5AmF76LSvD4TrWRIAE_92kii6j5yFiW-S7ahzlg/edit?tab=t.0#heading=h.vdognq1upsf5)
and on
[2025-05-12](https://docs.google.com/document/d/1Yt-i5AmF76LSvD4TrWRIAE_92kii6j5yFiW-S7ahzlg/edit?tab=t.0#heading=h.1mjh6unumnwu),
most notably:

-   **in place**: This convention was like `->`, but always using the "in place"
    convention where the caller allocates storage and provides the callee with
    its address, the callee initializes the storage at that address, and the
    caller is responsible for destroying after the return.
-   **var without storage**: The callee returns a pointer to the storage of a
    subobject of a `bound var` parameter, that caller is then responsible for
    destroying. A call to this function is reference expression, but with
    additional responsibility to destroy.
-   **hybrid**: If the type has a copy value representation or trivial
    destructive move then return the object representation directly; otherwise
    caller passes a pointer and callee initializes it.

There were also some variations on what the conditions for returning in
registers using the default return convention.

We seriously considered "var without storage", but the fact that it couldn't
reliably be used to initialize a variable, particularly in the middle of an
object, meant it did not seem valuable enough to include.

It seemed more valuable to support the "in place" return convention. That return
form allows you to guarantee knowing the address of the object being
constructed, and was a good match for `returned var`. However, we realized that
`var` declarations shouldn't always be associated with in-memory storage, in
particular for types that may be trivially moved. For example, a `var` parameter
with the C++ type `std::unique_ptr<T>` should be passed in registers. A function
returning a `std::unique_ptr<T>` in place would not be as efficient as returning
it by moving it into registers.

### `return var` with compound return forms

We considered various syntax options on
[2025-05-12](https://docs.google.com/document/d/1Yt-i5AmF76LSvD4TrWRIAE_92kii6j5yFiW-S7ahzlg/edit?tab=t.0#heading=h.1mjh6unumnwu),
but none of them seemed good enough to justify inclusion at this time:

```carbon
fn F(...) -> (ref R, val L, V) {
  // No longer a `var` being returned. Ideally these
  // shouldn't have to be initialized together.
  returned ??? (ref r: R, val l: L, var v: V) = ...?
  return var;
}

// We could restrict to one `var` return component,
// but this is a lot of machinery for a small increase
// in expressiveness and applicability.
fn F(...) -> (ref R, val L, V) {
  returned var v: V = ...;
  let l: L = ...;
  return (*r, l, var);
}

fn F(...) -> (ref R, val L, V) {
  // These don't have the right category, and ideally
  // shouldn't have to be initialized together.
  returned var ret: (R, L, V) = ...?
  return var;
}

fn F(...) -> (ref R, val L, V) {
  returned var (_, _, var v: V) = <what goes here?>;
}
```

There was another approach we considered for `returned var` originally:

```carbon
fn F(...) -> (ref R, val L, v1: V1, var v2: V2) {
  // ...
  // Must use the same names for the `var` (implicit or explicit) returns
  // with bound names.
  return (r, l, v1, v2);
}
```

But this had downsides that still apply:

-   Requires `V1` and `V2` to have unformed states. Otherwise, `v1` and `v2`
    would need be initialized when they are declared.
-   This does not support only having some branches use `return var`.

Our current approach handles our main use case for `returned var`: factory
functions.

We could support an "only `var`s" approach in the future if we want:

```carbon
fn F(...) -> (var V1, var V2, var V3) {
  returned (var v1: V1, var v2: V2, var v3: V3) = ...;
  // ...
  return var;
}
```

### Other syntax for compound return forms

We considered other options for the syntax of compound return forms on
[2025-05-13](https://docs.google.com/document/d/1Yt-i5AmF76LSvD4TrWRIAE_92kii6j5yFiW-S7ahzlg/edit?tab=t.0#heading=h.bdznj2d0by2g),
[#syntax in Discord on 2025-05-14](https://discord.com/channels/655572317891461132/709488742942900284/1372285365162872943),
and
[2025-05-14](https://docs.google.com/document/d/1Yt-i5AmF76LSvD4TrWRIAE_92kii6j5yFiW-S7ahzlg/edit?tab=t.0#heading=h.52tb7l2he343).
The option of omitting the `->` in each component did not distinguish tuples
from tuple return forms sufficiently:

```carbon
-> (ref i32, var i32)
-> (bool, ref i32)

// Is this a single return of a tuple, or a triple
// return using the default return convention?
-> (i32, i32, i32)
```

We also considered an approach where compound return forms would start with
`->?`, but this raised concerns about what the meaning of that syntax would be
and whether we want to expose users to that in cases we might be able to avoid
it.

The original proposed syntax used an arrow `->` in each component of a compound
form.

```carbon
fn TupleReturn(...)
    -> (->val bool, ->ref i32, -> C);

fn StructReturn(...)
    -> {->val .a: bool,
        ->ref .b: i32,
        -> .c: C};
```

This avoided ambiguity, but was verbose and visually noisy. An alternative was
suggested in
[discussion on 2025-06-13](https://docs.google.com/document/d/1Yt-i5AmF76LSvD4TrWRIAE_92kii6j5yFiW-S7ahzlg/edit?tab=t.0#heading=h.52ru7ner80b4).
This alternative used a default of compound returns for paren `(`...`)` and
brace `{`...`}` expressions, which you could opt out of by using one of the
three category keywords such as `var` to introduce a type expression that would
not be considered a compound return.

This had the downside that `-> T` would be interpreted as `-> var T` even if `T`
was a tuple type like `(i32, i64)`. However, textually substituting in
`(i32, i64)` in for `T` to get `-> (i32, i64)` would instead be interpreted as
`-> (var i32, var i64)`.

To overcome this problem, in
[discussion on 2025-06-16](https://docs.google.com/document/d/1Yt-i5AmF76LSvD4TrWRIAE_92kii6j5yFiW-S7ahzlg/edit?tab=t.0#heading=h.rdbzk5jnin3x),
we switched to the approach from this proposal.

### `ref` parameters allow aliasing

If requiring `ref` parameters to be `noalias` ends up being too restrictive, we
could instead have the "move-in-move-out" optimization be done only when the
compiler can prove it safe. One strategy would be to generate an alternate
version of the function that is only used in the cases where the `noalias`
conditions can be shown to hold statically.

### `let` to mark value returns instead of `val`

This proposal initially used `let` instead of `val` to mark
[immutable value returns](#ref-and-val-returns). However, `let` is used, in
Carbon and other languages, primarily to bind names. In Carbon, the default
binding is a value binding, but that was not considered a close enough to
connect the `let` keyword to value semantics.

There was also a concern about reusing `let` in multiple contexts to mean
different things, and having separate keywords that were only used to mark the
category of the binding was deemed better separation of concerns.

We considered making a parallel change to use `init` instead of `var`, but this
had some problems:

-   By making initializing returns the default, there is little expected usage,
    so perhaps not worth spending another keyword on.
-   The `init` keyword would be particularly expensive, because C++ code
    commonly use that word in APIs.

This question was considered in
[leads issue #5522](https://github.com/carbon-language/carbon-lang/issues/5522).

### `=>` infers form, not just type

There was support for the idea that the `=>` return syntax introduced in
[proposal #3848](https://github.com/carbon-language/carbon-lang/pull/3848)
should deduce the form of the return, not just its type. This was discussed in
[#lambdas on 2025-05-20](https://discord.com/channels/655572317891461132/999638000126394370/1374462658723450981).

However, trying to infer the expression category from the category of the
expression after the `=>` runs into the problem of this often requiring the
parameters to be marked `bound`. A more complicated rule, for example using
whether any parameter is marked `bound`, could be adopted in the future, if the
simple rule proves inadequate.

Alternatively, the compiler could infer which parameters should be marked
`bound` in this case. That is something to consider with the memory safety
design.
