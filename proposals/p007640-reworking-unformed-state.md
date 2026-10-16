# Reworking unformed state

<!--
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

[Pull request](https://github.com/carbon-language/carbon-lang/pull/7640)

<!-- toc -->

## Table of contents

-   [Abstract](#abstract)
-   [Problem](#problem)
-   [Background](#background)
-   [Goals and use cases](#goals-and-use-cases)
-   [Proposal](#proposal)
    -   [Unsafe conversions](#unsafe-conversions)
    -   [Unsafe adapters](#unsafe-adapters)
    -   [`Core.MaybeUnformed(T)`](#coremaybeunformedt)
    -   [Initializing a variable with no initializer](#initializing-a-variable-with-no-initializer)
    -   [Declaring an unformed state for a type](#declaring-an-unformed-state-for-a-type)
    -   [Detecting the unformed state](#detecting-the-unformed-state)
    -   [Assignment and destruction](#assignment-and-destruction)
    -   [Hardening the unformed state](#hardening-the-unformed-state)
    -   [Flow-sensitive restrictions on objects that might be unformed](#flow-sensitive-restrictions-on-objects-that-might-be-unformed)
    -   [Putting all this together for our use cases](#putting-all-this-together-for-our-use-cases)
-   [C++ interop](#c-interop)
    -   [C++ types and unformed state](#c-types-and-unformed-state)
        -   [C++ standard library types](#c-standard-library-types)
    -   [Passing unformed objects into C++ code](#passing-unformed-objects-into-c-code)
-   [Further details](#further-details)
    -   [Expected standard type behavior](#expected-standard-type-behavior)
    -   [Class types with a vtable](#class-types-with-a-vtable)
    -   [Comparison to `MaybeUninit` from Rust](#comparison-to-maybeuninit-from-rust)
-   [Rationale](#rationale)
-   [Alternatives considered](#alternatives-considered)
    -   [Keeping `UnformedInit` as a marker interface](#keeping-unformedinit-as-a-marker-interface)
    -   [Requiring the first `ref` call to initialize](#requiring-the-first-ref-call-to-initialize)
    -   [Making the unformed state a property of the type](#making-the-unformed-state-a-property-of-the-type)
    -   [Bit-mask based unformed state](#bit-mask-based-unformed-state)
    -   [Conversion oriented API design](#conversion-oriented-api-design)
    -   [Switching to one of the simpler alternatives discussed in #257](#switching-to-one-of-the-simpler-alternatives-discussed-in-257)
    -   [Folding the hardened value into `UnformedInvalid` and `UnformedNoop`](#folding-the-hardened-value-into-unformedinvalid-and-unformednoop)
    -   [Deriving the hardened value from the build configuration](#deriving-the-hardened-value-from-the-build-configuration)
    -   [Using `private adapt` instead of `unsafe adapt`](#using-private-adapt-instead-of-unsafe-adapt)
    -   [Spelling `MaybeUnformed` as a keyword qualifier](#spelling-maybeunformed-as-a-keyword-qualifier)
    -   [Spelling unsafe conversions as `unsafe_as`](#spelling-unsafe-conversions-as-unsafe_as)

<!-- tocstop -->

## Abstract

Introduce a formal model for how unformed state is defined and checked for
types. This includes:

-   Defining `unsafe as` and `Core.UnsafeAs` for unsafe conversions between
    types.
-   Adding `unsafe adapt` for adapters where converting to or from the adapted
    type is unsafe, along with `extend impl` within an `impl` so an adapter can
    expose safe conversions in specific directions or under specific conditions.
-   Defining `Core.MaybeUnformed(T)`, the type of an object that might be
    unformed, and restricting its API to the fields initialized in the unformed
    state and member functions that take `Core.MaybeUnformed(Self)`.
-   Defining the semantics of a `var` declaration with no initializer.
-   Defining the members of `Core.UnformedInit` and adding
    `Core.UnformedInvalid`, `Core.UnformedNoop`, `Core.IsUnformed`,
    `Core.UnformedHardenInit`, and `Core.UnformedHarden`.
-   Rules for assignment, destruction, and hardening of objects that might be
    unformed.
-   Flow-sensitive rules for tracking when maybe-unformed objects become
    initialized, integrated with Carbon's memory safety model.
-   Rules for synthesizing an unformed state for C++ types, and for calling C++
    APIs that initialize an output parameter.

This proposal also updates the design documentation and incorporates the
decisions from leads issues
[#5930](https://github.com/carbon-language/carbon-lang/issues/5930),
[#6161](https://github.com/carbon-language/carbon-lang/issues/6161), and
[#6739](https://github.com/carbon-language/carbon-lang/issues/6739). An earlier
version of this proposal was reviewed in
[#5913](https://github.com/carbon-language/carbon-lang/pull/5913).

## Problem

Carbon is trying to bring a rigorous model to handle complex initialization
scenarios from C++ in a way that maximizes reliability (through bug-detection),
soundness, efficiency, and ergonomics.

Our initial design is detailed in
[proposal \#257: "Initialization of memory and variables"](/proposals/p000257-initialization-of-memory-and-variables.md).
While that direction remains promising, the specific mechanics suggested there
are incomplete and/or don't seem to achieve the desired result. For example,
that proposal suggests that objects in an unformed state can only be _assigned_
or _destroyed_, and any other operation is invalid. But that leads to a problem:
how does one _implement_ either the assignment or destructor in a way that
doesn't perform an invalid operation? Any access to a field would be just such
an invalid operation, but accessing a field seems like a necessity in this
model. Similarly, how does one establish an unformed state for a new object?
Whatever operation is used seems like it would inherently violate the
constraints trying to be enforced, even returning an object in unformed state is
declared an error in
[#257](/proposals/p000257-initialization-of-memory-and-variables.md).

In the current prelude, `Core.UnformedInit` is an empty marker interface:
implementing it indicates that a type supports an unformed state, and the
toolchain leaves variables of that type with no initializer uninitialized. This
has several limitations:

-   A type with an invalid representation (such as a null pointer for `T*`)
    cannot specify that representation as its unformed state or expose a query
    for it. For example, `Optional(T*)` in `core/prelude/types/optional.carbon`
    uses dedicated builtins to create and test for a null pointer and uses
    `unsafe as` for every value access.
-   A type with a non-trivial destructor cannot specify which of its fields are
    initialized in the unformed state, or have the language skip destruction
    when the object is unformed.
-   Member functions cannot declare that they accept a maybe-unformed receiver,
    leaving `Core.MaybeUnformed(T)` with no safe operations.
-   Composite types do not automatically derive an unformed state from their
    members, requiring boilerplate `impl as Core.UnformedInit {}` declarations
    across the prelude and `examples/` to support uninitialized locals and
    `returned var` declarations.

The design documentation also does not yet cover unformed state,
`Core.MaybeUnformed(T)`, `unsafe as` (leads issue
[#5930](https://github.com/carbon-language/carbon-lang/issues/5930)), or the
interaction between `Core.Default` and `Core.UnformedInit` for variables
declared without an initializer (leads issue
[#6739](https://github.com/carbon-language/carbon-lang/issues/6739)), a gap
tracked by [#1993](https://github.com/carbon-language/carbon-lang/issues/1993).

## Background

-   [Proposal \#257: "Initialization of memory and variables"](/proposals/p000257-initialization-of-memory-and-variables.md)
    and its
    [Background](/proposals/p000257-initialization-of-memory-and-variables.md#background)
    section
-   Rust's
    [`MaybeUninit`](https://doc.rust-lang.org/std/mem/union.MaybeUninit.html)
-   Carbon's [safety design](/docs/design/safety/README.md), including
    [initialization safety](/docs/design/safety/terminology.md#initialization-safety),
    [strict and permissive Carbon](/docs/design/safety/README.md#safety-modes),
    and [build modes](/docs/design/safety/README.md#build-modes)
-   Prior review and discussion on pull request
    [#5913](https://github.com/carbon-language/carbon-lang/pull/5913)
-   Leads issues
    [#5930](https://github.com/carbon-language/carbon-lang/issues/5930),
    [#6161](https://github.com/carbon-language/carbon-lang/issues/6161), and
    [#6739](https://github.com/carbon-language/carbon-lang/issues/6739)

## Goals and use cases

The goal of _unformed state_ is to provide for better balance between safety and
ergonomics prior to initialization and (once we add move semantics to Carbon)
after moves. For example, consider this motivating example control flow:

```carbon
var x: SomeType;

for (item in SomeLoopKnownNonEmpty()) {
  x = MakeSomeValue(...);
  ...
}

UseSomeValue(x);
```

Where we assign to a variable inside the loop, and it is beneficial to leave it
unformed until that point as we have no meaningful value prior and we always
enter the loop. Rather than needing a separate operation for the first iteration
compared to all subsequent ones, or needing to manufacture a meaningless value,
unformed state lets us assign in all iterations uniformly. After the loop, using
the now fully formed value `x` can be checked at compile time when the control
flow is statically known to initialize `x` on every path, or guarded by a cheap
local run-time check (when `SomeType` has a detectable invalid unformed state)
or an explicit `unsafe` assertion when the compiler cannot prove the loop runs
at least once.

The idea is to leverage either of two flexibilities that frequently are
available in the design of a type, and expose those to the language so that it
can automatically synthesize the necessary behavior for this pattern.

Specifically:

-   Types which have a detectable invalid state (such as null for a non-null
    pointer)
-   Types for which there are states that make the destructor a no-op

For types that _do_ support unformed state, they may do so in three broad
categories based on the specific combination of the above properties they have:

1.  Types where there is an invalid state that can be used as the unformed
    state, and it can _optionally_ be queried, but the destructor will be a
    no-op without checking for that state.
2.  Types where there is an invalid state that can be used as the unformed
    state, but destruction must _check_ for that invalid state and skip
    meaningful logic to remain correct for objects in that state.
3.  Types that have a valid state with a no-op destructor that they reuse when
    unformed. These cannot support querying whether they are unformed.

These different categories also result in tradeoffs in the capabilities and
implementation approach for unformed state. Types may be able to support more
than one of these strategies, and will have to select which one they use based
on which tradeoffs are right for that specific type.

Types may alternatively _not_ have unformed state, either by necessity or
choice. Types with neither of the above properties are the simple case as they
_cannot support unformed state_. Other types may choose to not support an
unformed state even when they are capable, as that may result in a better API
design despite the ergonomic tradeoffs in the absence of an unformed state.

We will illustrate the design of unformed state with examples in each of the
three categories that support unformed state. We will also use multiple examples
within a category to surface interesting choices about how to approach unformed
state in that category.

Our examples for category (1) are two important primitive types: a non-owning
pointer `T*` and `bool`. For the non-owning pointer `T*` Carbon expects the null
value to be invalid. And for bool objects, we expect to have an entire byte of
state, and so have a great deal of flexibility as only two states will be valid.

For category (2), the canonical example is an owning pointer type similar to
C++'s `std::unique_ptr`:

```carbon
// Original type, prior to adding support for unformed state.
class OwningPtr(T: type) {
  var ptr: T*;

  // Per #6161, user-defined destruction logic runs before member destruction
  // (spelled `fn destroy` in the design docs and `Core.Destructor` in #7362).
  fn destroy(ref self) {
    // Note: we don't want to do this if `self` is unformed.
    SomeDeallocationFunction(self.ptr);
  }
}
```

Here, because we are building on top of the more primitive `T*`, we will also
illustrate _re-using_ unformed state of a member to implement a containing
type's unformed state.

And lastly, for category (3), we consider both the primitive type `i32` with a
trivial destructor for all states, and something like an
`Optional(OwningPtr(T))` (ignoring niche optimizations) with an interesting
destructor in some states.

## Proposal

The largest constraint for how to model this comes from the checking we would
like to perform. We suggest at the highest level three tiers of enforcement:

1.  Compile-time checking when there is a path on which a variable is used while
    unformed, including paths that are dynamically unreachable at run time when
    the compiler cannot prove them unreachable (analogous to Clang's
    `-Wsometimes-uninitialized`).
2.  Implicit and cheap, local run-time checking for use of unformed variables
    whenever possible (such as checking `Core.IsUnformed`), if the compile-time
    checks are not sufficient.
3.  The potential to restrict code that can't satisfy either of these by
    explicitly marking them as unsafe, and the ability to apply additional
    runtime hardening when executing such code. When exactly to enable
    enforcement of the `unsafe` marker is left as future work.

Achieving (1) suggests modeling this using the type system. Fully modeling this
in the type system would require flow-sensitive typing, which introduces
significant complexity that we would like to avoid in the broader language just
for this feature. So we propose a system that keeps the flow sensitivity out of
the type system, and states its rules in the terms of Carbon's memory safety
model, which is already planned to involve flow-sensitive checking.

The sections below build on each other in order: unsafe conversions and the
adapters that use them, then the type for an object that might be unformed, then
the interfaces a type implements to have an unformed state, and finally the
flow-sensitive rules for when such an object may be used.

### Unsafe conversions

Working with an object that might be unformed requires a conversion for cases
where the developer relies on an invariant that the compiler cannot verify, such
as converting a maybe-unformed object back to its formed type or accessing its
underlying storage. This proposal introduces _unsafe conversions_ into the
design, using the `unsafe as` spelling decided in leads issue
[#5930](https://github.com/carbon-language/carbon-lang/issues/5930) (see
[alternatives](#spelling-unsafe-conversions-as-unsafe_as) below).

Unsafe conversions are written `<expr> unsafe as T`. They add a third and
least restrictive layer to the conversion model:

```carbon
interface UnsafeAs(Dest: type) {
  fn Convert(self) -> Dest;
}

interface As(Dest: type) {
  extend final impl as UnsafeAs(Dest);
}

interface ImplicitAs(Dest: type) {
  extend final impl as As(Dest);
}
```

Much as implicit conversions use `ImplicitAs` and explicit conversions with the
`as` keyword use `As`, unsafe conversions use `UnsafeAs`. Because each interface
extends the one below it, any conversion that `as` can perform is also available
through `unsafe as`. For built-in conversions between compatible types and
qualifier conversions, `as` and `unsafe as` preserve the operand's expression
category (including durable reference expressions) and refer to the same
underlying storage.

> **Note:** The prelude currently writes these two extensions as separate
> blanket `impl`s until
> [proposal #5337](/proposals/p005337-interface-extension-and-final-impl-update.md)
> is implemented in the toolchain.

Beyond what it inherits from `As`, `unsafe as` can perform pointer
reinterpretation (`T* unsafe as U*`) and can remove type qualifiers that `as`
cannot (`const`, `partial`, and [`MaybeUnformed`](#coremaybeunformedt)). Adding
`const` or `partial` is always a safe conversion, and adding `MaybeUnformed` is
safe for value expressions, `const` references and pointers, and mutable `ref`
bindings tracked by
[flow-sensitive checking](#flow-sensitive-restrictions-on-objects-that-might-be-unformed).
Removing a qualifier (or adding `MaybeUnformed` to an untracked mutable pointer
`T*` -> `Core.MaybeUnformed(T)*`, which would allow writing an unformed state
into a formed `T`) is only safe in specific cases:

-   Removing `const` requires `unsafe as` when the result is still a reference
    expression. When the conversion converts a reference expression to a value
    or initializing expression, removing `const` is safe because the result does
    not refer to the original object.
-   Removing `partial` requires `unsafe as` for a non-initializing expression
    when the class is not `final`. It is safe when the class is `final` (per
    [#6161](https://github.com/carbon-language/carbon-lang/issues/6161)) or for
    an initializing expression, because the vtable pointer is initialized as
    part of that conversion.
-   Removing `MaybeUnformed` from an expression of static type
    `Core.MaybeUnformed(T)` always requires `unsafe as`, and is only available
    for a value or reference expression. An initializing expression has not yet
    initialized an object, so `MaybeUnformed` cannot be removed from an
    initializing expression.

The same rule applies through a pointer: converting `T*` to `U*` requires
`unsafe as` whenever the qualifiers on `T` and `U` do not permit a safe
reference conversion from `T` to `U`.

Removing `const` corresponds to C++'s `const_cast`. Requiring `unsafe as` keeps
each unsafe conversion
[semantically narrow](/docs/design/safety/README.md#safe-and-unsafe-code).

Leads issue [#6161](https://github.com/carbon-language/carbon-lang/issues/6161)
noted that `UnsafeAs.Convert` (written `UnsafeAs.Op` in
[#6161](https://github.com/carbon-language/carbon-lang/issues/6161)) should
itself be marked as an unsafe function so that direct calls require the same
`unsafe` marking as `unsafe as`, while `As.Convert` and `ImplicitAs.Convert`
remain safe functions.

> **Future work:** How `unsafe` functions are declared and called is left to a
> future safety proposal.

### Unsafe adapters

We combine unsafe conversions with the concept of
[adapting a type](/docs/design/generics/details.md#adapting-types) by modifying
the `adapt` declaration with the `unsafe` keyword. Such an adapter has all the
same properties as a normal adapter, except that any conversion between
compatible types that crosses an `unsafe adapt` step is only available with
`unsafe as` (so an `unsafe adapt` cannot be bypassed by converting through a
second, safe adapter of the same underlying type). These adapters can then
explicitly implement `As` or `ImplicitAs`, potentially doing so _conditionally_
or imposing other restrictions such as only allowing conversion in one
direction.

To make that possible, when an `interface` uses `extend impl`, we propose that
an `impl` of that interface can also write `extend impl` within its `impl`.
Doing so requires that an `impl` of the extended interface be visible, and
incorporates that existing `impl` into the newly defined `impl` rather than
requiring it to be duplicated. This `extend impl` ends with a semicolon `;`,
instead of a definition block in curly braces `{`...`}`, and supplies the
definitions of the members inherited from the extended interface. Because it
reuses the existing `impl` unchanged rather than overriding it, the incorporated
`impl` is permitted to be `final`. This allows adapters to extend which layer of
these kinds of interface hierarchies they implement without breaking coherence:

```carbon
class A {}
class B {
  // Provides definitions of both `B as UnsafeAs(A)` and
  // `A as UnsafeAs(B)`.
  unsafe adapt A;
}

// Explicit conversion from B -> A
impl B as As(A) {
  // Uses the definition of `B as UnsafeAs(A)` provided
  // by `unsafe adapt A;` in the definition of `class B`.
  extend impl as UnsafeAs(A);
}
// No implicit conversion from B -> A.

// Implicit conversion from A -> B
impl A as ImplicitAs(B) {
  // Uses the definition of `A as UnsafeAs(B)` provided
  // by `unsafe adapt A;` in the definition of `class B`.
  extend impl as UnsafeAs(B);

  // Note that we don't need to mention `As(B)` here, because we'll use the
  // normal blanket impl for that one.
}
```

The end result is that safe adapters are syntactic sugar around unsafe adapters,
adding implementations of `As` that extend the implementations of `UnsafeAs`.

> **Open question:** How does `extend impl` within an `impl` interact with
> `final`? Should we require writing `extend final impl` in an `impl` when the
> interface uses `extend final impl`? Should we require the extending `impl` to
> be a `final impl`?

> **Future work:** Adding an extending impl of `UnsafeAs` seems like an unsafe
> operation, and might need an `unsafe` keyword for auditing. We should consider
> whether we have `unsafe interface` and `unsafe impl` or some other approach to
> tracking this in a future proposal around safety.

One important use case we imagine for these semantics is working with the raw,
underlying storage of an object by defining an unsafe adapter for its type. This
proposal doesn't try to define the specifics of this, that is expected to be
part of a subsequent proposal that covers both storage and initialization of
storage.

> **Future work:** Fully define how raw storage is represented for objects and
> the relevant operations on it.

### `Core.MaybeUnformed(T)`

`Core.MaybeUnformed(T)` is the type of an object of type `T` that might be in an
unformed state. It is a built-in type qualifier spelled as an unsafe adapter
class in the prelude:

```carbon
class MaybeUnformed(T: type) {
  unsafe adapt T;
}
```

As an adapter, `Core.MaybeUnformed(T)` has the same object representation as
`T`, and converting a reference between `T` and `Core.MaybeUnformed(T)` is a
no-op at run time. Its value representation must preserve all bits of the object
representation (including bit patterns that are invalid for `T`, such as a
non-0/1 byte in `bool`, or uninitialized bytes), so when `T`'s value
representation does not preserve all object bits, `Core.MaybeUnformed(T)` uses a
pointer value representation instead.

> **Note:** Until `unsafe adapt` is implemented, the prelude defines
> `Core.MaybeUnformed(T)` using `adapt` on a compiler builtin type.

`Core.MaybeUnformed(T)` exposes only:

-   The fields of `T` that participate in its unformed state (those named by
    `UnformedInit.StructT` below, with their types from `StructT`).
-   Member functions of `T` that opt in by declaring their `self` parameter with
    type `Core.MaybeUnformed(Self)`.

This mirrors the
[partial class type](/docs/design/classes.md#partial-class-type), where only
methods that take the partial class type may be called on an object under
construction. As with `const` and `partial`, `Core.MaybeUnformed` is idempotent:
`Core.MaybeUnformed(Core.MaybeUnformed(T))` is the same type as
`Core.MaybeUnformed(T)`.

> **Open question:** Whether a lookup on `Core.MaybeUnformed(T)`, `const T`, or
> `partial T` should fall back to an `impl` written for `T` is tracked in leads
> issue [#6068](https://github.com/carbon-language/carbon-lang/issues/6068).
> Requiring an explicit `impl` or receiver type here is forward-compatible with
> either resolution of
> [#6068](https://github.com/carbon-language/carbon-lang/issues/6068), since a
> fallback can be added later without breaking existing code.

A formed `T` is valid wherever `Core.MaybeUnformed(T)` is expected, so `T`
converts implicitly to `Core.MaybeUnformed(T)` (when `T` is not already
`Core.MaybeUnformed`):

```carbon
impl forall [T: type] T as ImplicitAs(Core.MaybeUnformed(T));
```

> **Note:** The toolchain currently provides this conversion as a built-in
> conversion on reference expressions.

Converting an expression of static type `Core.MaybeUnformed(T)` to `T` requires
[`unsafe as`](#unsafe-conversions). `unsafe as` is also used to convert an
unformed object to its raw storage type in order to initialize a new value into
it without running a destructor. Both conversions preserve reference expressions
and refer to the same underlying storage, and also apply to pointers to these
types.

> **Future work:** We should consider adding `impl`s to `Core.MaybeUnformed(T)`
> when `IsUnformed` is implemented, ideally matching those used for optional
> types, so that it participates in the language-level affordances we provide
> for optional types. A key goal should be using `Core.MaybeUnformed(T)` without
> any unsafe operations through tools like `if let`.

### Initializing a variable with no initializer

Per leads issue
[#6739](https://github.com/carbon-language/carbon-lang/issues/6739), when a
variable `var x: T;` is declared without an initializer:

-   If `T` implements `Core.Default`, the variable is initialized by calling
    `T.impl(Core.Default.Op)()` and starts in the _definitely initialized_ flow
    state.
-   Otherwise, if `T` implements `Core.UnformedInit`, the variable's storage is
    initialized in place by calling `T.impl(Core.UnformedInit.Op)()` (which
    returns `Core.MaybeUnformed(T)`) and starts in the _maybe unformed_ flow
    state.
-   Otherwise, the declaration is invalid.

```carbon
interface Default {
  fn Op() -> Self;
}
```

Leads issue [#6739](https://github.com/carbon-language/carbon-lang/issues/6739)
left open what happens when `T` is a generic parameter that is not known to
implement either interface. We propose rejecting the declaration unless `T` is
constrained to implement `Core.Default` or `Core.UnformedInit`, so that generic
code explicitly models the initialization behavior it relies on.

> **TODO:** Determine how to express the dispatch and priority between
> `Core.Default` and `Core.UnformedInit` in the design before this proposal is
> accepted. While the prelude currently uses a `Core.DefaultOrUnformed` helper
> interface whose `Op` returns `Core.MaybeUnformed(Self)` (with a `final impl`
> for `T: Default` and a non-final `impl` for `T: UnformedInit`), using that as
> the language design would model `var x: T;` as `Core.MaybeUnformed(T)` even
> when `T` implements `Core.Default`, triggering flow-sensitive checking of
> uses. Conversely, we cannot synthesize an unformed state from a `Core.Default`
> implementation or the other way around.

### Declaring an unformed state for a type

We give `UnformedInit` two members: the subset of the object that participates
in the unformed state, and a way of producing that state.

```carbon
interface UnformedInit {
  private default let StructT: type = {};
  default fn Op() -> Core.MaybeUnformed(Self) {
    return {} unsafe as Core.MaybeUnformed(Self);
  }
}

interface UnformedInvalid {
  require impls IsUnformed;
  private default let StructT: type = {};
  private default let Value: StructT = {};
}

interface UnformedNoop {
  private default let StructT: type = {};
  private default let Value: StructT = {};
}

// A type may implement both, so these are prioritized rather than each `final`.
final match_first {
  impl forall [T: UnformedInvalid] T as UnformedInit
      where .StructT = T.impl(UnformedInvalid.StructT) {
    fn Op() -> Core.MaybeUnformed(Self) {
      return T.impl(UnformedInvalid.Value) unsafe as Core.MaybeUnformed(Self);
    }
  }

  impl forall [T: UnformedNoop] T as UnformedInit
      where .StructT = T.impl(UnformedNoop.StructT) {
    fn Op() -> Core.MaybeUnformed(Self) {
      return T.impl(UnformedNoop.Value) unsafe as Core.MaybeUnformed(Self);
    }
  }
}
```

The `StructT` associated type of the `UnformedInit` interface is a struct type
with a subset of the field names of `Self`, and each field's type must be
compatible-with the corresponding field type of `Self`. It specifies which
fields of `Self` are initialized in the unformed state; any other fields are
left uninitialized. Producing an unformed object initializes those fields in
place from the corresponding fields of a `StructT` value, whose types are
compatible (including by way of `unsafe adapt`) but potentially different.
Because constructing a `Core.MaybeUnformed(T)` from an arbitrary `StructT` value
could bypass private field encapsulation or produce a representation that `T`'s
own `IsUnformed` or destructor does not recognize as unformed, converting
between `T.impl(UnformedInit.StructT)` and `Core.MaybeUnformed(T)` is a built-in
`unsafe as` conversion rather than a safe implicit conversion. Safe code outside
the prelude obtains an unformed `T` by calling `T.impl(Core.UnformedInit.Op)()`,
which only ever writes the type's own nominated `Value` (or runs its own `Op`).

Both `StructT` and `Value` default to `{}`, and `Op` defaults to producing that
(when `StructT` is overridden, `Value` or `Op` must also be provided). A type
that implements `UnformedNoop` (or `UnformedInit` directly without `IsUnformed`)
must have a no-op destructor and valid assignment for every representation whose
`StructT` fields equal `Value` and whose remaining bytes are arbitrary. With the
default `StructT = {}`, that requires a trivial destructor, so a type with a
trivial destructor can implement `UnformedNoop` or `UnformedInit` using the
defaults (`impl as Core.UnformedNoop {}`), leaving all fields uninitialized in
the unformed state. Existing `impl as Core.UnformedInit {}` declarations
continue to work unchanged.

For composite types that do not customize `UnformedInit` (tuples, structs,
arrays, and data classes), an unformed state is synthesized member-wise (or
element-wise) when all members implement `Core.UnformedInit`, initializing each
member in place with its own `UnformedInit.Op()`.

`StructT` and `Value` are `private` so that implementing an unformed state does
not expose the names, types, or sentinel values of private fields outside the
prelude and language implementation. Implementing types can specify `StructT`
and `Value`, while callers outside the prelude use the public `Op` function to
produce an unformed object (including when defining a composite type's unformed
state in terms of a member's `Op`) and
[`IsUnformed`](#detecting-the-unformed-state) to query whether an object is
unformed.

> **Open question:** Carbon has access control on an interface declaration but
> not on an interface member. Whether `private` is the right spelling for an
> associated member that an external `impl` can define but external callers
> cannot read is still an open question.

While the language dispatches through `UnformedInit`, most types implement
`UnformedInvalid` or `UnformedNoop` to specify a constant `Value` rather than a
function. Categories (1) and (2) above both use `UnformedInvalid` (differing
only in whether skipping destruction when `IsUnformed` returns `true` is an
optimization or required for correctness), while category (3) uses
`UnformedNoop`.

> **Future work:** We should probably provide default implementations of most of
> these interfaces when the members have implementations. Spelling that out and
> picking the specific default options isn't handled here and is future work.

> **Future work:** Eventually, we should design a more comprehensive system to
> expose invalid states, bit patterns, and so on, in order to facilitate
> stashing more bits into types for discriminants and other tools. At that
> point, we can look at more powerful ways of expressing both the basic invalid
> state and any invalid+hardened state.

### Detecting the unformed state

A type whose unformed state is an invalid representation implements `IsUnformed`
to test whether an object is currently in an unformed state:

```carbon
interface IsUnformed {
  fn Op(self: Core.MaybeUnformed(Self)) -> bool;
}

match_first {
  impl forall [
      T: UnformedInvalid & UnformedHarden
      where T.impl(UnformedInvalid.StructT) impls Eq
      and T.impl(UnformedHarden.StructT) impls Eq]
      T as IsUnformed {
    fn Op(self: Core.MaybeUnformed(Self)) -> bool {
      return (self unsafe as T.impl(UnformedInvalid.StructT))
                 == T.impl(UnformedInvalid.Value) or
             (self unsafe as T.impl(UnformedHarden.StructT))
                 == T.impl(UnformedHarden.Value);
    }
  }

  impl forall [
      T: UnformedInvalid
      where T.impl(UnformedInvalid.StructT) impls Eq]
      T as IsUnformed {
    fn Op(self: Core.MaybeUnformed(Self)) -> bool {
      return (self unsafe as T.impl(UnformedInvalid.StructT))
                 == T.impl(UnformedInvalid.Value);
    }
  }
}
```

Note that `Op` takes its object parameter as `Core.MaybeUnformed(Self)`, the
opt-in described above.

While code written against a concrete type `T` can read the individual fields
named by `StructT` directly on `Core.MaybeUnformed(T)` as part of its safe API,
generic prelude code does not know those field names and instead projects `self`
to `StructT` as a whole using `unsafe as`. That projection is sound here because
`StructT` names only the fields initialized in the unformed state, so those
fields hold a valid value whether or not the object is formed. Comparing two
`StructT` values compares them
[field-wise](/docs/design/classes.md#data-classes), so the blanket
implementations apply whenever `StructT`'s field types are comparable with `==`.

Types can implement `IsUnformed` directly (and must do so when `StructT` does
not implement `Eq`, since `UnformedInvalid` requires `IsUnformed`). Any custom
implementation must return `true` for every representation in the type's
unformed representation set, not only the representation written by
`UnformedInit.Op`.

`Core.MaybeUnformed(T)` also provides a forwarding `impl`:

```carbon
final impl forall [T: IsUnformed] Core.MaybeUnformed(T) as IsUnformed {
  fn Op(self) -> bool {
    // `T`'s implementation already takes its object parameter as
    // `Core.MaybeUnformed(T)`, which is our `Self`.
    return self.(T.impl(IsUnformed.Op))();
  }
}
```

This forwarding `impl` relies on `Core.MaybeUnformed(T)` being idempotent
(`Core.MaybeUnformed(Self)` is `Self`), and is needed because interface lookup
on `Core.MaybeUnformed(T)` does not automatically fall back to `impl`s for `T`.

### Assignment and destruction

Under the destructor design from leads issue
[#6161](https://github.com/carbon-language/carbon-lang/issues/6161) (and
[#7362](https://github.com/carbon-language/carbon-lang/pull/7362)), the
signature of `Destroy.Op` is fixed and complete-object destruction is
synthesized by the language rather than customized against
`Core.MaybeUnformed(Self)`. Instead, the language handles destruction of
maybe-unformed objects based on the type's unformed-state interfaces:

-   A type implementing `IsUnformed` (including every type using
    `UnformedInvalid`) has a detectable unformed state, so the language tests
    `IsUnformed` before destroying a maybe-unformed object and skips destruction
    when the test succeeds (when the type's destructor is trivial, both the test
    and the destructor are no-ops and can be elided).
-   A type using `UnformedNoop` (or implementing `UnformedInit` directly without
    `IsUnformed`) has a no-op destructor for every unformed representation, so
    destruction can run unconditionally or be skipped when known to be unformed.
-   For a composite type whose unformed state is synthesized member-wise,
    destroying a maybe-unformed object destroys each member according to its own
    rule.

Destroying an object of static type `Core.MaybeUnformed(T)` (which is not
stripped as a qualifier before this check) follows the same rule when `T`
implements `Core.UnformedInit`, so a field of type `Core.MaybeUnformed(T)` is
destroyed automatically by field-wise destruction as long as it holds either a
formed `T` or a representation in `T`'s unformed representation set. When `T`
does not implement `Core.UnformedInit` (for example, when
`Core.MaybeUnformed(T)` is used as storage alongside an external discriminant,
as in `Core.Optional(T)`), automatic destruction of `Core.MaybeUnformed(T)` does
nothing and the owner of the storage is responsible for destroying the `T` when
present. `Core.MaybeUnformed(T)` itself implements `Core.UnformedInit` for every
`T` (delegating to `T`'s `UnformedInit` when `T` implements it).

For assignment, an implementation of `AssignWith` can opt in to handling
maybe-unformed targets directly by declaring its object parameter as
`ref self: Core.MaybeUnformed(Self)` (narrowing the receiver requirement in the
same way an implementation can narrow `partial Self`), in which case it is
called whether or not the target is unformed and leaves the target definitely
initialized. When `AssignWith` does not take `Core.MaybeUnformed(Self)`:

-   If the type implements `IsUnformed` and the target is not statically known
    to be initialized, the language tests `IsUnformed` first and initializes the
    target directly from the right-hand side when it is unformed, matching
    [simple assignment semantics](/docs/design/assignment.md).
-   Otherwise (for `UnformedNoop` or direct `UnformedInit`), the unformed state
    is valid for assignment and has a no-op destructor, so the language may
    either call `AssignWith` or initialize the target directly.

> **Note:** In the toolchain today, assignment is a built-in operation rather
> than `Core.AssignWith`. When `Core.AssignWith` is added to the prelude, its
> declaration will permit implementations to opt in with
> `ref self: Core.MaybeUnformed(Self)`.

### Hardening the unformed state

In [build modes](/docs/design/safety/README.md#build-modes) with hardening
enabled (such as the release build mode), the compiler applies baseline
initialization hardening automatically. A type can also specify a dedicated
representation to use when hardening an unformed object, similar to how the
[partial class type](/docs/design/classes.md#partial-class-type) can initialize
an unformed vtable pointer to a null or poison vtable depending on the build
mode.

We distinguish three concepts for a type with an unformed state:

-   The _unformed representation set_ is the set of representations an object of
    the type may hold while unformed.
-   The _unformed value_ is the representation written by `UnformedInit`.
-   The _hardened unformed value_ is the representation written by
    `UnformedHardenInit` when hardening is enabled.

`IsUnformed` tests for membership in the unformed representation set rather than
equality with a single unformed value, so it recognizes both the normal and
hardened unformed values. Because whether hardening is applied can vary across
packages or compilation units within the same program (or across sites in a
function), `IsUnformed` must recognize both values regardless of the build mode
of the code performing the check.

```carbon
interface UnformedHardenInit {
  require impls UnformedInit;
  private default let StructT: type = {};
  default fn Op() -> Core.MaybeUnformed(Self) {
    return {} unsafe as Core.MaybeUnformed(Self);
  }
}

interface UnformedHarden {
  require impls UnformedInit;
  private default let StructT: type = {};
  private default let Value: StructT = {};
}

final impl forall [T: UnformedHarden] T as UnformedHardenInit
    where .StructT = T.impl(UnformedHarden.StructT) {
  fn Op() -> Core.MaybeUnformed(Self) {
    return T.impl(UnformedHarden.Value) unsafe as Core.MaybeUnformed(Self);
  }
}
```

`UnformedHardenInit.StructT` has the same restrictions as
`UnformedInit.StructT`, and must additionally be a superset of its fields, since
hardening may initialize more of the object but never less. _Hardening_ an
unformed object when the language leaves it unformed means setting those fields
to the result of `UnformedHardenInit.Op`, along with any additional
initialization (such as zero- or pattern-filling) the compiler performs for
security in the face of unsafe code. Any automatic compiler fill is applied
before (or only to bytes outside) the `StructT` fields written by
`UnformedInit.Op` or `UnformedHardenInit.Op`, so that the fields participating
in the unformed state are never overwritten and `IsUnformed` continues to
recognize the object. Hardening happens only when the language leaves an object
unformed or uninitialized (never to fully formed objects, and without replacing
explicit calls to `UnformedInit.Op` in user or library code), and there is no
restriction on whether both `UnformedInit.Op` and `UnformedHardenInit.Op` are
called or only one.

The hardened value must be consistent with the type's unformed-state category:

-   A type implementing `UnformedInvalid` must harden to an invalid
    representation that is in its unformed representation set and recognized by
    `IsUnformed`.
-   A type implementing `UnformedNoop` must harden to a representation whose
    destruction and assignment are also valid no-ops.

When a type implements both `UnformedInvalid` and `UnformedHarden` and relies on
the blanket `impl` of `IsUnformed`, `IsUnformed` automatically checks for both
`UnformedInvalid.Value` and `UnformedHarden.Value`. A custom `IsUnformed` or
`UnformedHardenInit` implementation must maintain this consistency; violating it
is ill-formed.

> **Open question:** The compiler cannot generally verify that a custom
> `IsUnformed` implementation returns `true` for the result of a custom
> `UnformedHardenInit.Op`. We could either restrict custom `UnformedHardenInit`
> implementations or check `IsUnformed.Op(UnformedHardenInit.Op())` at compile
> time when both are constant-evaluable.

We expect some code to fall back to unsafe initialization, especially around C++
interop or during a migration from existing C++ API designs. In these cases we
expect the compiler to do some amount of hardening automatically, to prevent any
bugs in that unsafe code from being as easily exploited, for example
[Clang's trivial auto variable initialization](https://clang.llvm.org/docs/ClangCommandLineReference.html#cmdoption-clang-ftrivial-auto-var-init).

When a pointer to a maybe-unformed object escapes the scope that tracks its
initialization state, the compiler cannot see whether a later `unsafe as`
conversion reads the object before it is initialized. We therefore expect
hardening to be applied before a pointer or reference to a maybe-unformed object
escapes the flow-tracked context.

> **Future work:** Specify the exact points where hardening is applied once the
> memory safety model's flow analysis and function effect annotations are
> finalized.

> **Future work:** We might want to support types opting into hardening
> _without_ an unformed state at all, which needs either a separate interface or
> splitting the hardening aspect out of `UnformedHarden`. We make
> `UnformedHarden` a superset for now, because a type that wants an explicit
> hardened state almost always wants an unformed one too.

### Flow-sensitive restrictions on objects that might be unformed

An object starts out unformed when it is declared without an explicit
initializer and its type has no `Core.Default` implementation:

```carbon
var object: SomeType;
```

> **Future work:** We also expect to add operations to Carbon that put objects
> into this state without a declaration, as we would like to use unformed state
> for non-destructively-moved-from objects as well, but those will come in
> subsequent and separate proposals.

Carbon's memory safety model is planned to include flow-sensitive checking, and
this proposal states its unformed-state rules in that model's terms, leaving the
model itself to a proposal focused on safety. We expect the details, and
particularly the syntax, to change.

Every object has a _place_. Alongside properties of a place that are fixed, such
as its static type, a place carries state that varies from one point in a
function to the next, including whether it is initialized. So a place is either
_definitely initialized_ or _maybe unformed_ at each point, and where control
flow merges the two, the result is maybe unformed.

Flow-sensitive state is computed after type checking, so it cannot participate
in overload resolution or `impl` selection:

-   A place declared with type `T` has static type `T` throughout its lifetime.
    When its flow state is _definitely initialized_, it may be used directly as
    a `T` with no conversion.
-   When a place of static type `T` is _maybe unformed_, using it directly as a
    formed `T` is an error unless explicitly bypassed with `unsafe as T` (or,
    if we adopt Tier 2 local run-time checking for `T: Core.IsUnformed`, guarded
    by an implicit fail-stop `IsUnformed` check). Without `unsafe as`, the place
    may only be assigned to (which makes it definitely initialized), destroyed,
    initialized member-by-member when its fields are visible (such as for a
    local or `returned var`, becoming definitely initialized once all fields are
    initialized), or implicitly converted from `T` to `Core.MaybeUnformed(T)` so
    that it is restricted to `Core.MaybeUnformed(T)`'s safe API.
-   An expression whose _static type_ is `Core.MaybeUnformed(T)` (such as a
    field, parameter, or dereferenced `Core.MaybeUnformed(T)*` pointer) always
    requires `unsafe as T` to be converted to `T`.

A function's effect on the initialization state of its arguments is declared in
its signature rather than inferred from how an argument is passed. A function
that initializes an argument or leaves a formed argument unformed declares that
effect in its signature. Passing a place of type `T` as a mutable
`ref Core.MaybeUnformed(T)` leaves the place definitely initialized if the
callee declares an initialization effect on it, and maybe unformed otherwise
(since a callee accepting `ref Core.MaybeUnformed(T)` is otherwise permitted to
leave it unformed). Taking the address `&x` of a maybe-unformed place of type
`T` produces `Core.MaybeUnformed(T)*`. Using a placeholder syntax for declared
effects:

```carbon
// Definitely initializes the place `i` refers to.
fn Init(ref i: Core.MaybeUnformed(i32)) [[init(^i)]];

// May leave the place `i` refers to unformed.
fn MoveFrom(ref i: i32) [[move_from(^i)]];

fn Run() {
  var i: i32;
  // ❌ Error: `^i` is maybe unformed.
  i += 1;
  Init(ref i);
  // ✅ `^i` is definitely initialized.
  i += 1;
  MoveFrom(ref i);
  // ❌ Error: `^i` is maybe unformed again.
  i += 1;
  i = 0;
  // ✅ Simple assignment also makes `^i` definitely initialized.
  i += 1;
}
```

Passing a `ref` argument to a `T` parameter therefore does not implicitly change
an object's initialization state; transitions come from declared effects in
either direction.

> **Future work:** Extend the effect annotations to support conditional
> initialization (for example, a function that initializes an output parameter
> only when it returns `true`). Until conditional initialization effects are
> supported, callers of such functions must use `unsafe as`.

When the compiler cannot prove that an object is initialized, `unsafe as` serves
as the explicit escape hatch, either bypassing the flow check on a
maybe-unformed place of type `T` (`x unsafe as T`) or converting an expression
of static type `Core.MaybeUnformed(T)` to `T` (including for references and
pointers). If the object is still unformed at run time,
[hardening](#hardening-the-unformed-state) ensures its storage holds a
deterministic hardened representation.

This results in a restrictive model that requires either initialization,
explicit code handling unformed values with `Core.MaybeUnformed`, or `unsafe`.
We suggest treating that as _experimental_ and revisiting it based on
experience, as it may be necessary to make the escape hatch occur automatically
in more cases, especially around C++ interop.

These rules also constrain fields and concurrent access:

-   A field whose type is `T` may be temporarily unformed only within a window
    that no other code can observe, which means it must be formed again before
    calling anything with transitive access to it and before the function
    returns. A field that can be observed while unformed must be declared as
    `Core.MaybeUnformed(T)` instead.
-   An atomic type must not have an unformed state at all. No local analysis can
    establish that another thread did not observe the object while it was
    unformed, so such a type must not implement these interfaces.

Note also that leaving a place unformed does not invalidate pointers to it. Such
a pointer may still be used, so long as the place is initialized again before
any operation that requires it to be formed.

> **Note:** Carbon's principle on
> [low context sensitivity](/docs/project/principles/low_context_sensitivity.md#flow-sensitive-typing)
> sets a high bar for flow-sensitive typing, on the grounds that the type of a
> name changing based on surrounding `if` statements is too subtle.
> Flow-sensitive initialization tracking does not change the static type of a
> variable: a variable declared with type `T` has static type `T` throughout its
> scope, and the flow state only determines whether uses of that variable are
> valid.

### Putting all this together for our use cases

For `bool` and `T*`, the following examples use class types with private integer
fields for illustration; in practice, these types are provided by the toolchain:

```carbon
// Imagined pseudo-implementation of `bool` for illustration.
class Bool {
  fn True() -> Bool { return {.value = 1}; }
  fn False() -> Bool { return {.value = 0}; }

  private var value: i8;

  impl as Core.UnformedInvalid
      where .StructT = {.value: i8}
      and   .Value   = {.value = -1} {}
}

// Imagined pseudo-implementation of `T*` for illustration.
class Ptr(T: type) {
  private var value: u64;

  impl as Core.UnformedInvalid
      where .StructT = {.value: u64}
      and   .Value   = {.value = 0} {}

  // For illustration, we also give pointers a hardened constant, using the
  // `0xAAAA_AAAA_AAAA_AAAA` pattern from LLVM's pattern initialization.
  impl as Core.UnformedHarden
      where .StructT = {.value: u64}
      and   .Value   = {.value = 0xAAAA_AAAA_AAAA_AAAA} {}
}
```

`OwningPtr(T)` builds on `Ptr(T)` above, adding a non-trivial destructor:

```carbon
class OwningPtr(T: type) {
  private var ptr: Ptr(T);

  // With reflection, we could potentially provide an automatic
  // way of delegating to a member, but spelling it out for now.
  impl as Core.UnformedInvalid
      where .StructT = {.ptr: Core.MaybeUnformed(Ptr(T))}
      and .Value = {.ptr = Ptr(T).impl(Core.UnformedInit.Op)()} {}

  impl as Core.UnformedHarden
      where .StructT = {.ptr: Core.MaybeUnformed(Ptr(T))}
      and .Value = {.ptr = Ptr(T).impl(Core.UnformedHardenInit.Op)()} {}

  // The blanket `impl` does not apply here, because it requires `StructT`
  // to implement `Eq` and `Core.MaybeUnformed(Ptr(T))` has no `==`.
  // Delegating to the member's own test is both possible and cheaper.
  impl as Core.IsUnformed {
    fn Op(self: Core.MaybeUnformed(Self)) -> bool {
      // `self.ptr` has type `Core.MaybeUnformed(Ptr(T))`, which still allows
      // calling `.(Core.IsUnformed.Op)()`.
      return self.ptr.(Core.IsUnformed.Op)();
    }
  }

  fn Reset(ref self: Core.MaybeUnformed(Self), var rhs: OwningPtr(T))
      [[init(^self)]] {
    if (not self.(Core.IsUnformed.Op)()) {
      // This is a fully formed object, so it has to be destroyed before its
      // storage is reused. Explicit destruction is itself an unsafe operation;
      // #7630 is deciding its name (currently spelled `SelfDestruct` in the
      // prelude).
      (self unsafe as Self).(Core.Destroy.SelfDestruct)();
    }

    // Imagined syntax to directly re-initialize storage,
    // not part of this proposal. `~rhs` is a destructive move.
    let ref storage: ??? = self unsafe as ???;
    raw_init storage = {.ptr = (~rhs).ptr};
  }

  // The destructor says nothing about unformed state. The language
  // tests `IsUnformed` before destroying and skips this when it holds.
  fn destroy(ref self) {
    SomeDeallocationFunction(self.ptr);
  }
}
```

An integer type `Int(N)` has no unused bit pattern to test for an unformed
state, so it implements `Core.UnformedNoop` instead (shown here with a private
field rather than an adapter to make the struct conversions explicit):

```carbon
class Int(N: IntLiteral) {
  private var value: MakeInt(N);

  // The unformed state writes no fields, which is the default, because
  // our destructor is trivial.
  impl as Core.UnformedNoop {}

  // But we might harden the integer to zero.
  impl as Core.UnformedHarden where .StructT = {.value: MakeInt(N)}
                              and   .Value   = {.value = (0 as Int(N)).value} {}

  // Taking `Core.MaybeUnformed(Self)` opts in to being called on an unformed
  // object, so this is called either way. Nothing needs destroying first, as
  // the destructor is always trivial.
  impl as Core.AssignWith(Int(N)) {
    fn Op(ref self: Core.MaybeUnformed(Self), other: Int(N)) {
      // Imagined syntax to directly re-initialize storage, not part of
      // this proposal.
      let ref storage: ??? = self unsafe as ???;
      raw_init storage = {.value = other.value};
    }
  }
}
```

`OptionalOwningPtr(T)` uses the unformed state of `OwningPtr(T)` to represent
its empty state. Because that bit pattern is a valid value of
`OptionalOwningPtr(T)`, `OptionalOwningPtr(T)` cannot implement
`Core.UnformedInvalid`, and instead implements `Core.UnformedNoop`:

```carbon
class OptionalOwningPtr(T: type) {
  fn Make(var ptr: OwningPtr(T)) -> Self {
    return {.ptr = ~ptr};
  }
  fn MakeEmpty() -> Self {
    // Works for any type with an unformed state, not just `OwningPtr(T)`.
    return {.ptr = OwningPtr(T).impl(Core.UnformedInit.Op)()};
  }

  let PtrT: type = Core.MaybeUnformed(OwningPtr(T));

  private var ptr: PtrT;

  // Note that this isn't invalid, just a noop.
  impl as Core.UnformedNoop
      where .StructT = {.ptr: PtrT}
      and   .Value   = {.ptr = OwningPtr(T).impl(Core.UnformedInit.Op)()} {}

  // No destructor is needed. The `ptr` member has type
  // `Core.MaybeUnformed(OwningPtr(T))`, so destroying it already tests
  // whether it holds a formed pointer.
}
```

For example, `Optional(T*)` in `core/prelude/types/optional.carbon` currently
uses compiler builtins to construct and test for a null pointer:

```carbon
// Current implementation in `core/prelude/types/optional.carbon`.
private fn PointerIsNull[T: type](value: MaybeUnformed(T*)) -> bool
    = "pointer.is_null";
private fn MakeUninitializedOptionalPointer(generic T: type)
    -> MaybeUnformed(T*) = "make_uninitialized";

final impl forall [T: type] T* as OptionalStorage
    where .Type = MaybeUnformed(T*) {
  fn None() -> MaybeUnformed(T*) = "pointer.make_null";
  fn Some(self) -> MaybeUnformed(T*) {
    returned var result: MaybeUnformed(T*) =
        MakeUninitializedOptionalPointer(T);
    result unsafe as T* = self;
    return var;
  }
  fn Has(value: MaybeUnformed(T*)) -> bool {
    return not PointerIsNull(value);
  }
  fn Get(value: MaybeUnformed(T*)) -> T* {
    return value unsafe as T*;
  }
  fn Copy(value: MaybeUnformed(T*)) -> MaybeUnformed(T*) = "primitive_copy";
}
```

Once `T*` implements `UnformedInvalid` with null as its unformed value,
`OptionalStorage` can use `UnformedInit` and `IsUnformed` instead of
pointer-specific builtins:

```carbon
final impl forall [T: type] T* as OptionalStorage
    where .Type = Core.MaybeUnformed(T*) {
  fn None() -> Core.MaybeUnformed(T*) {
    return (T*).impl(Core.UnformedInit.Op)();
  }
  // Relies on the implicit conversion from `T*`.
  fn Some(self) -> Core.MaybeUnformed(T*) { return self; }
  fn Has(value: Core.MaybeUnformed(T*)) -> bool {
    return not value.(Core.IsUnformed.Op)();
  }
  fn Get(value: Core.MaybeUnformed(T*)) -> T* {
    return value unsafe as T*;
  }
  fn Copy(value: Core.MaybeUnformed(T*)) -> Core.MaybeUnformed(T*)
      = "primitive_copy";
}
```

Only `Get` still requires `unsafe as`, because the precondition that `Has`
returned `true` is not tracked in the type system.

Several of the examples above also depend on operations that will be specified
in separate proposals:

-   Invoking the destructor on objects
-   Destructively moving objects
-   Turning an object into raw storage
-   Initializing raw storage with a new value

## C++ interop

### C++ types and unformed state

Providing an unformed state for C++ types where safe allows them to be used in
uninitialized variable declarations and composite unformed states. Because the
compiler cannot automatically identify an unused bit pattern for an arbitrary
C++ type, default synthesis for C++ types never implements `IsUnformed`.

A C++ type with a non-trivial default constructor implements `Core.Default`, so
`var x: Cpp.SomeType;` calls its default constructor and produces a fully formed
object, taking priority over `Core.UnformedInit`. By contrast, a trivially
default-constructible C++ type whose default constructor leaves fields
uninitialized cannot produce a formed object with indeterminate member values in
strict Carbon, so it uses an unformed state instead (unless `Core.Default`
performs value-initialization). Even when a C++ type implements `Core.Default`,
its unformed state is still used when a containing type synthesizes its own
unformed state from its members.

A C++ type has a synthesized unformed state under the following rules:

-   If the type is trivially default constructible and trivially destructible
    (and trivially assignable), then it implements `UnformedNoop` with an empty
    struct type and value. Requiring a trivial destructor ensures that
    destroying or assigning to a maybe-unformed object of the type never invokes
    a user-defined C++ destructor or assignment operator on uninitialized member
    bytes.
-   Otherwise, if the type is default constructible and trivially destructible,
    then the type implements `UnformedInit` directly, with `Op` returning a
    default-constructed instance of the type (it implements `UnformedInit`
    rather than `UnformedNoop` because a non-trivial C++ default constructor
    call is not a compile-time constant `Value`, while still having a no-op
    destructor).

Specific C++ types can also customize their unformed state by implementing the
unformed state interfaces in Carbon wrapper code.

#### C++ standard library types

Widely used C++ standard library vocabulary types should define unformed states
for ergonomic interop. Many vocabulary types map directly to Carbon types (such
as C++ pointers to Carbon pointers, or `std::unique_ptr` to Carbon's owning
pointer type) and inherit their unformed states; other standard library types
can implement the unformed state interfaces in their Carbon wrappers.

### Passing unformed objects into C++ code

C++ APIs that expect to initialize output parameters passed by non-`const`
reference or pointer (`T&` or `T*`) won't have the Carbon `Core.MaybeUnformed`
type to identify them. We propose that in
[permissive Carbon](/docs/design/safety/README.md#safety-modes) there is an
implicit `unsafe` escape for such arguments (applying defensive hardening before
the call), so that these APIs can be called with an unformed object provided the
API could legitimately initialize the type. After this call the object is
assumed fully formed, as it would be normally.

We expect [strict Carbon](/docs/design/safety/README.md#safety-modes) to reject
these without an _explicit_ `unsafe` operation, so that unsafe initialization is
gradually all marked in the source.

Writing the explicit conversion directly with `unsafe as` is verbose because it
casts a maybe-unformed variable of type `T` to its own type `T` to bypass the
flow check:

```carbon
var t: Cpp.T;
// Casting `t` to its own type to bypass the flow check.
Cpp.T.Init(ref (t unsafe as Cpp.T));
```

We propose providing a library function that performs this conversion and
declares the initialization effect on the argument's place:

```carbon
// Asserts that the callee initializes the place `x` refers to.
unsafe fn Escape[T: type](ref x: Core.MaybeUnformed(T)) -> ref T [[init(^x)]];

var t: Cpp.T;
Cpp.T.Init(ref Core.Escape(ref t));
```

An analogous overload for pointer output parameters converts
`Core.MaybeUnformed(T)*` to `T*`. If the double `ref` in
`ref Core.Escape(ref t)` is too verbose in practice, dedicated syntax can be
added later as sugar for this call. How `Escape` is marked `unsafe` at its
declaration and call site is deferred to the same future proposal as
`UnsafeAs.Convert`.

## Further details

### Expected standard type behavior

We expect Core types in Carbon to provide an unformed state whenever there is a
reasonable implementation strategy, and pointers and `bool` specifically to
implement `IsUnformed`, so that types built on them can reuse it.

### Class types with a vtable

> **Future work:** It would be very nice to allow types to reuse the
> vtable-pointer field to implement their unformed state, sharing the machinery
> `partial` already has for that field. This is left as future work to address
> how types control opting in and out of the behavior, and how it should work
> across inheritance.

### Comparison to `MaybeUninit` from Rust

Carbon's `Core.MaybeUnformed(T)` and Rust's `MaybeUninit<T>` serve overlapping
but distinct roles:

-   **Deferred initialization in safe code**: Unformed state lets a type
    designate an invalid or no-op representation so the language can support
    `var x: T;`, assignment, and destruction in safe code. When `T` implements
    `Core.UnformedInit`, `UnformedInit.Op()` initializes the participating
    fields (such as setting a pointer to null), `IsUnformed` can query that
    state at run time, and `Core.MaybeUnformed(T)`'s destructor automatically
    destroys formed values and skips unformed ones. By contrast, Rust's
    `MaybeUninit<T>::uninit()` writes nothing, cannot be inspected in safe code,
    and never runs `T`'s destructor automatically.
-   **Uninitialized storage in containers**: When `T` does not implement
    `Core.UnformedInit` (or when a container manages element lifetimes with an
    external discriminant or length), `Core.MaybeUnformed(T)` has a no-op
    destructor and requires `unsafe as` to access the `T`, matching the role of
    `MaybeUninit<T>` in Rust. Once Carbon's raw storage design is finalized,
    containers may also use raw storage directly.

## Rationale

-   [Performance-critical software](/docs/project/goals.md#performance-critical-software)
    -   Customizing the exact hardening approach gives added per-type control to
        library authors to get the best cost/benefit tradeoff between
        performance and security.
    -   Exposing an unformed state can reduce the branching required to
        represent control-dependent initialized objects.
    -   Strict handling of unformed state can reduce the need for defensive
        hardening of objects, providing the developer control over the costs of
        their code without loss of safety.
-   [Code that is easy to read, understand, and write](/docs/project/goals.md#code-that-is-easy-to-read-understand-and-write)
    -   The unformed state models common idioms used in C++ where types can
        model an otherwise-invalid state that still supports assignment and
        destruction to simplify initialization and moving code patterns.
    -   Making incorrect usage of objects in this state explicit in the language
        and type system allows better and earlier error messages during
        development.
-   [Practical safety and testing mechanisms](/docs/project/goals.md#practical-safety-and-testing-mechanisms)
    -   Supports both existing C++ idioms when mapped into Carbon without
        regressing safety and provides a clear path to increase safety in the
        space of initialization.
    -   `unsafe as` keeps the operations that cannot be checked narrow and
        auditable, not regions of unchecked code.
-   [Interoperability with and migration from existing C++ code](/docs/project/goals.md#interoperability-with-and-migration-from-existing-c-code)
    -   C++ types with non-trivial default constructors keep working unchanged,
        and trivially default-constructible types (or types with wrapper
        `UnformedInit` implementations) can still be declared without an
        initializer while participating in Carbon's initialization checking and
        hardening.
    -   The permissive and strict modes give migrated code a path from calling
        C++ output parameter APIs freely to marking each such call explicitly.

## Alternatives considered

### Keeping `UnformedInit` as a marker interface

We could keep `Core.UnformedInit` as an empty marker interface, where a type
implements it to indicate that it supports an unformed state, the compiler
leaves such objects uninitialized, and hardened builds zero-initialize them.

Advantages:

-   Simpler interface with no associated types or constants to specify.
-   Automatic zero-fill in hardened builds does not require type-specific
    annotations.

Disadvantages:

-   Types with an existing invalid representation (such as a null pointer)
    cannot use it as their unformed state or expose `IsUnformed` so containing
    types can query or compose it.
-   Types with non-trivial destructors cannot specify which fields are
    initialized in the unformed state or have destruction skipped when unformed.

We reject this alternative because supporting types with non-trivial destructors
and composable invalid states is a primary goal of Carbon's unformed state
model.

### Requiring the first `ref` call to initialize

An earlier draft of this proposal allowed an unformed variable of type `T` to be
passed as a `ref Core.MaybeUnformed(T)` argument and assumed that the first such
call initialized the variable, without requiring effect annotations or
flow-sensitive state tracking.

Advantages:

-   Does not require effect annotations on function declarations or general
    flow-sensitive state tracking.

Disadvantages:

-   Places an initialization obligation on the callee that is not expressed in
    the callee's signature, so the compiler cannot verify it when checking the
    callee.
-   Cannot distinguish functions that initialize a `ref` argument from functions
    that leave a `ref` argument unformed or only inspect it.

Whether a function initializes a `ref` parameter or leaves it unformed is part
of the function's contract and should be declared in its signature.

### Making the unformed state a property of the type

Instead of tracking initialization state per place, we could refine the static
type of a partially initialized object to record which of its fields are
unformed, such as `Thing | {x.f unformed}`.

Advantages:

-   Field granularity, rather than the whole-object granularity of
    `Core.MaybeUnformed(T)`.
-   No separate flow-sensitive state to specify.

Disadvantages:

-   Such a type names variable identifiers, so it cannot easily escape the scope
    of those names (for example, in fields, return types, or closures) without a
    significant extension to the type system.
-   It requires flow-sensitive typing, which conflicts with Carbon's
    [low context sensitivity](/docs/project/principles/low_context_sensitivity.md#flow-sensitive-typing)
    principle.

`Core.MaybeUnformed(T)` composes with the rest of the type system without
depending on local variable names. Per-field unformed state is instead handled
either locally within a scope or by declaring a field with type
`Core.MaybeUnformed(T)`.

### Bit-mask based unformed state

Rather than using a subset of the fields of an object, we could instead define a
bitmask of the object that is initialized in the unformed state.

Advantages:

-   Significantly finer granularity of initialization and querying of the
    object.
-   Potential to use invalid bit patterns that are not represented as fields.

Disadvantages:

-   We don't yet have a design for bit-fields in Carbon, which would likely
    intersect with this in many ways.
-   More complex model than using fields.

The suggestion is to not pursue this initially, but to revisit when fully
introducing bit-fields and thinking more holistically about bit-oriented type
layouts. That seems like the place where this would become most desirable and a
collection of design that any bit-oriented solution would need to integrate
cleanly with.

### Conversion oriented API design

Initially, this proposal pursued an API design for working with unformed values
by converting them to different types in order to access the fields available in
the unformed state.

The proposal shifted to the current direction because the resulting code with
conversions was complicated and difficult to understand. The model of an API
subset was much more easily understood, explained, and used in practice.

### Switching to one of the simpler alternatives discussed in #257

Fleshing out these details does raise the question of whether we should switch
Carbon to one of the alternatives to unformed state more generally. The set of
options here has not materially changed since
[#257](/proposals/p000257-initialization-of-memory-and-variables.md) and so we
don't duplicate that list and analysis here.

Fundamentally, this proposal suggests that there is still a good motivation to
try and match the idioms across C++ code where types have this "partially
formed" (what we're calling "unformed") state that is used for deferred
initialization and potentially moved-from states. This pattern continues to be
prevalent and well liked in C++. Relative to
[#257](/proposals/p000257-initialization-of-memory-and-variables.md), this
proposal revises a few specific choices to make the model work end-to-end: it
introduces `Core.MaybeUnformed(T)` so functions and fields can explicitly accept
or store possibly-unformed objects, exposes `Core.IsUnformed` so the language
and containing types can query invalid unformed states, and integrates
flow-sensitive place checking from Carbon's memory safety design rather than
allowing untracked address-of or use of unformed variables in strict Carbon.

### Folding the hardened value into `UnformedInvalid` and `UnformedNoop`

Rather than defining separate `UnformedHarden` and `UnformedHardenInit`
interfaces, `UnformedInvalid` and `UnformedNoop` could each include an optional
second constant for the hardened value.

Advantages:

-   Two fewer interfaces in the prelude.
-   Automatically ensures that a type's hardened value uses the same semantic
    category (`UnformedInvalid` versus `UnformedNoop`) as its unformed value.

Disadvantages:

-   Couples hardening to having an unformed state, making it harder to support
    types that want to customize hardening without supporting an unformed state.

We keep the hardening interfaces separate so they can be extended to types
without an unformed state in future work.

### Deriving the hardened value from the build configuration

Rather than a type specifying separate normal and hardened values,
`UnformedInvalid` could specify a single `Value` whose definition depends on the
build configuration.

Advantages:

-   One interface and one constant, with no possibility of the two disagreeing.
-   No need for `IsUnformed` to test for more than one representation.

Disadvantages:

-   Makes the build configuration an input to the type system, so the same
    source would produce different types in different build modes, complicating
    separate compilation and mixing libraries built in different modes.
-   Provides no way to express both a cheap unformed value and a more expensive
    hardened value simultaneously.

We reject this alternative because build configuration should not affect type
checking or separate compilation.

### Using `private adapt` instead of `unsafe adapt`

The class design already notes `private adapt` as future work for restricting
adapter conversions to the defining library. We could use `private adapt`
instead of introducing `unsafe adapt`.

Advantages:

-   Reuses an existing concept without adding `unsafe adapt`.

Disadvantages:

-   Conflates access control with memory safety: converting an unformed
    `Core.MaybeUnformed(T)` to `T` or accessing raw storage has safety
    preconditions even inside the defining library, and should be marked with
    `unsafe` for auditing.

Access control and safety are orthogonal, and a type may combine both as
`private unsafe adapt`.

### Spelling `MaybeUnformed` as a keyword qualifier

`Core.MaybeUnformed(T)` could be written as a keyword qualifier, such as
`unformed T`, matching `const T` and `partial T`.

Advantages:

-   Syntactically consistent with `const T` and `partial T`.
-   Shorter to write in parameter and field declarations.

Disadvantages:

-   `Core.MaybeUnformed(T)` is already used in the prelude and in accepted
    proposals (such as
    [#6357](/proposals/p006357-c-interop-mapping-pointer-types.md)).
-   A parameterized class syntax composes directly in generic code while still
    having qualifier semantics in the toolchain.

We retain the `Core.MaybeUnformed(T)` spelling for now; switching to a keyword
qualifier later would be a syntactic rename if needed.

### Spelling unsafe conversions as `unsafe_as`

The main alternative syntax considered was avoiding the two keywords in sequence
with `unsafe_as`. It also looked at `try as` or `try_as` for comparison.

Advantages of `unsafe_as`:

-   Lexically simpler as it is a single word.
-   Many other languages use underscores for compound keywords.
    -   But Python at least does provide precedent with `not in` for omitting
        the underscore.
-   The separate keywords don't work for all cases we could imagine, see below.

Disadvantages of `unsafe_as`:

-   Unclear how composition with `try_as` would work.
-   Makes the use of `unsafe` for auditing unsafe constructs more difficult.

The issue also examined whether we want this to be _specific_ to `unsafe`, but
that continued to pose compositional challenges.

The leads decided on `unsafe as`: separate modifier keywords, but specifically
where they are meaningfully modifiers, meaning there is a hierarchy of
operations with a core keyword whose variations the modifiers select, and the
keywords read well in isolation without changing meaning confusingly when
composed. A modifier structure highlights the relationship between the
operations and supports auditing on either component. We could imagine other
compound words that would struggle to meet both criteria, such as `raw_init`,
and this decision doesn't implicate those one way or the other. When we come to
such a keyword, we'll need to decide whether to have both forms at the same
time, or make some other adjustment.
