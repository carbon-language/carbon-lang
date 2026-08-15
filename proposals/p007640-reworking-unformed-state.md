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
    -   [Making `MaybeUnformed` a type qualifier](#making-maybeunformed-a-type-qualifier)
    -   [Spelling unsafe conversions as `unsafe_as`](#spelling-unsafe-conversions-as-unsafe_as)

<!-- tocstop -->

## Abstract

Introduce a more formal model for how unformed state is managed for types. This
includes:

-   Describing `unsafe as` and `Core.UnsafeAs`, which model unsafe conversions
    between compatible types.
-   Adding `unsafe adapt` for types that need such a conversion to reach their
    own representation, along with `extend impl` within an `impl` so that such a
    type can make some conversions safe again.
-   Describing `Core.MaybeUnformed(T)`, the type of an object that might be
    unformed, and narrowing its API to the fields that participate in the
    unformed state plus the members that opt in.
-   Describing what a variable declared with no initializer means.
-   Filling in `Core.UnformedInit`, which is a bare marker in the prelude today,
    and adding `Core.UnformedInvalid`, `Core.UnformedNoop`, `Core.IsUnformed`,
    `Core.UnformedHardenInit`, and `Core.UnformedHarden` alongside it.
-   Rules for assignment, destruction, and hardening of objects that might be
    unformed.
-   A flow-sensitive set of rules for how unformed objects become fully formed,
    stated in the terms of Carbon's memory safety model and left to a subsequent
    proposal to finalize.
-   Rules for synthesizing an unformed state for C++ types, and for calling C++
    APIs that initialize an output parameter.

It also updates the design documentation, which does not currently cover any of
this. Reflects the decisions in leads issues #5930, #6161, and #6739.

An earlier version of this proposal was reviewed at length in #5913, which was
closed before that discussion concluded. Much of the design here comes out of
it, and the reasoning behind several decisions is recorded there rather than
repeated below.

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
declared an error in [#257](/proposals/p000257-initialization-of-memory-and-variables.md).

`Core.UnformedInit` in the prelude shows the consequences. It is only a marker:
a type implements it to say that it has _some_ unformed state, and the compiler
responds by leaving such objects entirely uninitialized.

-   A type with a cheap invalid state it could use has no way to nominate it.
    `Optional(T*)` wants a null pointer to mean "empty", so
    `core/prelude/types/optional.carbon` reaches for builtins to make a null
    pointer, to test for one, and to produce an uninitialized value, and every
    access to the value goes through `unsafe as`.
-   A type with a non-trivial destructor has no way to say which of its fields
    are initialized, so the only correct unformed state is one that writes
    nothing.
-   There is no way to write a function that safely operates on an object that
    might be unformed, which means `Core.MaybeUnformed(T)` has no safe API at
    all.

The interface is also a tax on types that gain nothing from it. 19 types in the
prelude and 17 classes in `examples/` carry a bare
`impl as Core.UnformedInit {}` for no reason other than to keep
`returned var me: T;` working.

There is a second problem, of a different kind. No part of this area of the
design has a proposal behind it, and none of it appears in the design
documentation, which #1993 has tracked since 2022. `unsafe as`,
`Core.MaybeUnformed(T)`, and the dispatch between `Core.Default` and
`Core.UnformedInit` are all implemented in the toolchain, and a merged proposal
already depends on `Core.MaybeUnformed(T)`.

## Background

-   [Proposal \#257: "Initialization of memory and variables"](/proposals/p000257-initialization-of-memory-and-variables.md)
-   [Background](/proposals/p000257-initialization-of-memory-and-variables.md#background) of that proposal
-   Rust's
    [MaybeUninit](https://doc.rust-lang.org/std/mem/union.MaybeUninit.html)
-   Carbon's [safety design](/docs/design/safety/README.md), and specifically
    [initialization safety](/docs/design/safety/terminology.md#initialization-safety),
    the one category of memory safety where Carbon leans on run-time techniques
    for the sake of ergonomics
-   The vocabulary that design gives for the unsafe parts of this:
    [strict and permissive Carbon](/docs/design/safety/README.md#safety-modes)
    for when an `unsafe` marking is required, and the
    [build modes](/docs/design/safety/README.md#build-modes) for what hardening
    is applied

This proposal was first opened as #5913 and reviewed there by danakj, josh11b,
nmsmith, burakemir, and zygoloid before an inactivity bot closed it. That
discussion is where much of this design was worked out, and the review threads
on it carry the reasoning for several of the decisions here, including the
terminology for hardening, the shape of `Core.MaybeUnformed(T)`'s API, and the
escape hatch for C++ output parameters. It is worth reading alongside this
proposal rather than treated as superseded. This is a new pull request because
the branch history was rewritten and #5913 could no longer be reopened.

The toolchain already implements several pieces of this design. This proposal
describes them along with the rest, so that the design documents can cover them:

-   The `unsafe as` operator and the `Core.UnsafeAs` interface, added in #5993,
    with the spelling decided in leads issue #5930.
-   The `Core.MaybeUnformed(T)` type, added in #5989 and refined since. It is
    already used by
    [#6357](/proposals/p006357-c-interop-mapping-pointer-types.md) to model
    C++'s `nullptr_t`.
-   `Core.Default`, `Core.UnformedInit`, and `Core.DefaultOrUnformed`, added in
    #6934 to implement leads issue #6739, and extended to C++ types in #6962.

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

Where we assign to a variable at the start of the loop, but it is beneficial to
leave it uninitialized until that point as we have no meaningful value prior and
we always enter the loop. Rather than needing a separate operation for the first
iteration compared to all subsequent ones, or needing to manufacture a
meaningless value, unformed state lets us assign in all iterations uniformly and
then reliably use the now well formed value after the loop.

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

  impl as Core.Destroy {
    fn Op(ref self) {
      // Note: we don't want to do this if `self` is unformed.
      SomeDeallocationFunction(self.ptr);
    }
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

1.  Compile-time checking when there is either a use of an unformed variable or
    dead code that _would_ use the variable unformed if not dead (analogous to
    Clang's `-Wsometimes-uninitialized`).
2.  Implicit and cheap, local run-time checking for use of unformed variables
    whenever possible, if the compile time checks are not sufficient.
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

Working with an object that might be unformed needs a conversion that the type
system cannot check. Going from a possibly unformed object back to a fully
formed one is exactly the case where the developer knows an invariant that the
compiler does not, and the same will be true of reaching the raw storage
underneath an object.

The toolchain implements an operator for this, and leads issue #5930 decided its
spelling, with [alternatives](#spelling-unsafe-conversions-as-unsafe_as)
discussed below. This proposal is what brings it into the design.

_Unsafe conversions_ are written `<expr> unsafe as T`. They add a third and
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
through `unsafe as`.

> **Note:** The prelude currently spells these two extensions as separate
> blanket `impl`s labeled as workarounds, because this shape of interface
> extension awaits an implementation of
> [proposal #5337](/proposals/p005337-interface-extension-and-final-impl-update.md).
> The design is unchanged; only the spelling is a workaround.

Beyond what it inherits from `As`, `unsafe as` can remove type qualifiers that
`as` cannot. Carbon has three such qualifiers today, `const`, `partial`, and
[`MaybeUnformed`](#coremaybeunformedt). Adding any of them is a safe conversion.
Removing one is not, and the rule differs by qualifier:

-   Removing `const` needs `unsafe as` when the result is still a reference
    expression. When the conversion changes the expression category anyway, no
    reference to the original object survives and removing `const` is safe.
-   Removing `partial` needs `unsafe as` for a non-initializing expression. It
    is safe for an initializing expression, because the vtable pointer is
    initialized as part of that conversion.
-   Removing `MaybeUnformed` always needs `unsafe as`, and is only available for
    a value or reference expression. An initializing expression has not yet
    produced an object, so there is nothing whose formedness the developer could
    be asserting.

The same rule applies through a pointer: converting `T*` to `U*` needs
`unsafe as` whenever the qualifiers on `T` are not a subset of those on `U`.

Removing `const` is the operation C++ spells `const_cast`, which is already
understood to be dangerous by C++ developers, so requiring `unsafe` for it does
not impose a new burden when migrating. Note that this keeps each unsafe
capability [semantically narrow](/docs/design/safety/README.md#safe-and-unsafe-code):
removing `const` permits a write that the type system did not authorize, and
nothing more.

Leads issue #6161
decided that `UnsafeAs.Convert` should itself be marked as an unsafe function,
so that calling it directly is as visible to an audit as writing `unsafe as`.

> **Future work:** What an `unsafe` marking on a `fn` means is not yet decided,
> and is needed by other proposals as well. We use `unsafe as` here and leave
> the function marking to whichever proposal settles it.

### Unsafe adapters

We combine unsafe conversions with the concept of
[adapting a type](/docs/design/generics/details.md#adapting-types) by modifying
the `adapt` declaration with the `unsafe` keyword. Such an adapter has all the
same properties as a normal adapter, but the conversions between the two types
are only available with `unsafe as`. These adapters can then explicitly
implement other conversion interfaces, potentially doing so _conditionally_ or
imposing other restrictions such as only allowing conversion in one direction.

To make that possible, when an `interface` uses `extend impl`, we propose that
an `impl` of that interface can also write `extend impl` within its `impl`.
Doing so requires that an `impl` of the extended interface be visible, and
incorporates that into the newly defined `impl` rather than requiring it to be
duplicated. This `extend impl` will end with a semicolon `;`, instead of a
definition block in curly braces `{`...`}`, but acts as a definition of the
members of the extended interface. This also allows the found `impl` to be
`final` because this precludes any changes to the aspects of the `impl` that
were final. This allows adapters to extend which layer of these kinds of
interface hierarchies they implement without breaking coherence:

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

An object that might be unformed needs a type, both so that a function can
declare that it accepts one and so that the language can stop the object from
being used as though it were fully formed. That type is
`Core.MaybeUnformed(T)`, which we propose declaring as an unsafe adapter:

```carbon
class MaybeUnformed(T: type) {
  unsafe adapt T;
}
```

Being an adapter, it has the same object representation as `T`, so converting
between the two never touches the object. Its value representation may differ,
because the fields of `T` that are not part of its unformed state may hold
nothing at all.

The prelude spells this differently today, as a safe `adapt` of a type produced
by a builtin, because `unsafe adapt` does not exist yet.

The API of `Core.MaybeUnformed(T)` is deliberately small:

-   The fields of `T` that participate in its unformed state, which are the
    fields named by the `UnformedInit.StructT` type described below. Those are
    the only fields known to hold anything.
-   Member functions of `T` that opt in, by declaring their `self` parameter
    with the type `Core.MaybeUnformed(Self)`. This is how a function signals to
    its caller that it is safe to call on an object that might be unformed, and
    how the language enforces that its definition doesn't assume otherwise.

This mirrors the [partial class type](/docs/design/classes.md#partial-class-type),
where only methods that take the partial class type may be called on it, so
that methods have to opt in to being called on an object that isn't fully
constructed.

> **Open question:** Whether a lookup on `Core.MaybeUnformed(T)`, `const T`, or
> `partial T` should fall back to an `impl` written for `T` is leads issue
> #6068. We take
> the restrictive position here because it composes with either outcome: opting
> in is always available, and a fallback can be added later without invalidating
> code written against this rule. The reverse is not true.

Converting from `T` to `Core.MaybeUnformed(T)` is safe, and should be implicit:
a formed object is always an acceptable argument where a possibly unformed one
is expected. Converting back is
[an unsafe conversion](#unsafe-conversions), because nothing in the type system
establishes that the object is formed.

The safe direction should be available as an `impl` of `ImplicitAs`, so that
generic code can rely on it:

```carbon
impl forall [T: type] T as ImplicitAs(Core.MaybeUnformed(T));
```

> **Note:** The toolchain provides this conversion directly today, and only for
> reference expressions.

`unsafe as` is also how code reaches the raw storage under an unformed object,
in order to initialize a new object into it without running a destructor. Both
that and the conversion back to `T` operate on _reference expressions_ and
continue to refer to the same storage, so they can equally convert pointers
between these types. The conversion back to `T` also covers the case where the
unformed state is arranged to be a valid initialization of `T` and the operation
merely reifies it.

> **Future work:** We should consider adding `impl`s to `Core.MaybeUnformed(T)`
> when `IsUnformed` is implemented, ideally matching those used for optional
> types, so that it participates in the language-level affordances we provide
> for optional types. A key goal should be using `Core.MaybeUnformed(T)` without
> any unsafe operations through tools like `if let`.

### Initializing a variable with no initializer

A variable declared without an initializer is the case unformed state exists to
serve. This reflects the decision in leads issue #6739.

When a variable is declared without an initializer, and its type is `T`:

-   If `T` implements `Core.Default`, the variable is initialized by calling
    `Core.Default.Op` and is fully formed.
-   Otherwise, if `T` implements `Core.UnformedInit`, the variable is
    initialized to an unformed state.
-   Otherwise, the declaration is invalid.

Leads issue #6739 left open what happens when `T` is a generic parameter that is
not known to implement either interface. We propose rejecting the declaration,
rather than conservatively treating the object as unformed and discovering later
that `T` does not in fact use an unformed state. Generic code that needs a
variable with no initializer has to require one of the two interfaces, and so
model the behavior it depends on explicitly.

The toolchain expresses the priority between the two with a dispatch interface,
so that `impl` selection makes the choice rather than a rule in the compiler:

```carbon
interface Default {
  fn Op() -> Self;
}

interface DefaultOrUnformed {
  fn Op() -> Core.MaybeUnformed(Self);
}

// A fully formed default wins, so this `impl` is `final` and the one below
// cannot specialize it.
final impl forall [T: Default] T as DefaultOrUnformed {
  fn Op() -> Core.MaybeUnformed(Self) { return T.(Default.Op)(); }
}

impl forall [T: UnformedInit] T as DefaultOrUnformed {
  fn Op() -> Core.MaybeUnformed(Self) { return T.(UnformedInit.Op)(); }
}
```

`var x: T;` is then checked as if it were
`var x: T = (T as DefaultOrUnformed).Op();`, and the declaration being invalid
when neither interface is implemented falls out of there being no `impl` to
find.

> **TODO:** This dispatch is how the toolchain implements the rule today, but it
> should not be the design, and needs replacing before this proposal is
> accepted. `Op` has a single return type covering both cases, and neither
> choice works. `Core.MaybeUnformed(Self)`, as written above, describes the
> variable as maybe unformed even when `Core.Default` produced a fully formed
> object, which would subject it to flow-sensitive checking it should not need.
> `Self`, which the prelude declares today, claims a fully formed object in the
> case where there isn't one. Collapsing the two interfaces into one isn't an
> answer either, as an unformed state cannot be synthesized from a
> `Core.Default` implementation, nor a default from an unformed state.

### Declaring an unformed state for a type

We give `UnformedInit` two members: the subset of the object that participates
in the unformed state, and a way of producing that state.

```carbon
interface UnformedInit {
  private default let StructT: type = {};
  default fn Op() -> Core.MaybeUnformed(Self) { return {}; }
}

interface UnformedInvalid {
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
      where .StructT = T.(UnformedInvalid.StructT) {
    fn Op() -> Core.MaybeUnformed(Self) { return T.(UnformedInvalid.Value); }
  }

  impl forall [T: UnformedNoop] T as UnformedInit
      where .StructT = T.(UnformedNoop.StructT) {
    fn Op() -> Core.MaybeUnformed(Self) { return T.(UnformedNoop.Value); }
  }
}
```

The `StructT` associated type of the `UnformedInit` interface is a struct type
with a subset of the field names of `Self`, and each field's type must be
compatible-with the corresponding field type of `Self`. It says which fields of
the object the unformed state writes; the rest are left alone. Producing an
unformed object initializes those fields in place from the corresponding fields
of a `StructT` value, whose types are compatible but potentially different. The
conversion that does so is what the blanket `impl`s above return:

```carbon
impl forall [T: UnformedInit]
    T.(UnformedInit.StructT)
    as ImplicitAs(Core.MaybeUnformed(T)) {
  fn Convert(self) -> Core.MaybeUnformed(T) {
    // Initialize the fields of `T` that are part of `StructT`.
  }
}
```

It initializes exactly the fields named by `StructT` and leaves the rest alone,
which is what makes the result unformed. Whether the result is one that the
type's own `IsUnformed` recognizes is the type's contract to keep.

Both `StructT` and `Value` default to `{}`, and `Op` defaults to producing that,
so a type that implements one of these interfaces without saying anything more
keeps exactly the meaning it has today: the unformed state writes no fields, and
the destructor need not run. No existing bare `impl as Core.UnformedInit {}`
declaration has to change.

`StructT` and `Value` are `private` because they describe the representation. A
type that says its unformed state is `{.ptr = 0}` would otherwise be publishing
the name, type, and sentinel value of a private field, defeating the
encapsulation it maintains. An implementing type still sets them; what `private`
withholds is reading them outside the library declaring the interface, which
confines the representation to the prelude and the language. `Op` is public,
because producing an unformed object is not itself dangerous, and because it is
how one type builds its unformed state out of a member's. The public way to ask
whether an object is currently unformed is
[`IsUnformed`](#detecting-the-unformed-state) below.

> **Open question:** Carbon has access control on an interface declaration but
> not on an interface member. Whether `private` is the right spelling here, and
> whether an implementer outside the declaring library may set a member it
> cannot read, both need deciding.

While `UnformedInit` is the interface the language uses, we expect most types to
implement `UnformedInvalid` or `UnformedNoop`. Both let a type nominate a
constant value rather than write a function. The first two categories above both
use `UnformedInvalid`; they differ only in whether the language's `IsUnformed`
test before destruction is needed for correctness, or is merely an optimization
the type would be correct without. The third uses `UnformedNoop`.

> **Future work:** We should probably provide default implementations of most of
> these interfaces when the members have implementations. Spelling that out and
> picking the specific default options isn't handled here and is future work.

> **Future work:** Eventually, we should design a more comprehensive system to
> expose invalid states, bit patterns, and so on, in order to facilitate
> stashing more bits into types for discriminants and other tools. At that
> point, we can look at more powerful ways of expressing both the basic invalid
> state and any invalid+hardened state.

### Detecting the unformed state

A type whose unformed state is an _invalid_ state can be asked whether an object
is currently in it.

```carbon
interface IsUnformed {
  fn Op(self: Core.MaybeUnformed(Self)) -> bool;
}

match_first {
  impl forall [T: UnformedInvalid & UnformedHarden] T as IsUnformed {
    fn Op(self: Core.MaybeUnformed(Self)) -> bool {
      return (self unsafe as T.(UnformedInvalid.StructT))
                 == T.(UnformedInvalid.Value) or
             (self unsafe as T.(UnformedHarden.StructT))
                 == T.(UnformedHarden.Value);
    }
  }

  impl forall [T: UnformedInvalid] T as IsUnformed {
    fn Op(self: Core.MaybeUnformed(Self)) -> bool {
      return (self unsafe as T.(UnformedInvalid.StructT))
                 == T.(UnformedInvalid.Value);
    }
  }
}
```

Note that `Op` takes its object parameter as `Core.MaybeUnformed(Self)`, the
opt-in described above.

Reading the fields out of a possibly unformed object is an `unsafe as`, as it is
everywhere else. It is sound here because `StructT` names exactly the fields the
unformed state writes, so those fields hold a value whether or not the object is
formed, and because this is prelude code where the representation is visible.
Comparing two `StructT` values compares them
[field-wise](/docs/design/classes.md#data-classes), so a type's chosen field
types have to be comparable.

Types can implement `IsUnformed` themselves in the usual way for interfaces, but
such an implementation has to uphold the contract of definitively testing for
the object being in _any_ unformed representation, not just the one
`UnformedInit` writes.

`Core.MaybeUnformed(T)` needs a forwarding `impl` of its own:

```carbon
final impl forall [T: IsUnformed] Core.MaybeUnformed(T) as IsUnformed {
  fn Op(self) -> bool {
    // `T`'s implementation already takes its object parameter as
    // `Core.MaybeUnformed(T)`, which is our `Self`.
    return self.(T.(IsUnformed.Op))();
  }
}
```

This relies on `Core.MaybeUnformed(T)` being idempotent, so that
`Core.MaybeUnformed(Self)` here is `Self` and the declared signature matches. We
propose that as the rule: an object either might be unformed or is known not to
be, and a second application has no third thing to say.

The forwarding `impl` is required, not merely convenient. Because a
lookup on `Core.MaybeUnformed(T)` does not fall back to an `impl` written for
`T`, without it the interface would be unusable on exactly the objects it exists
to ask about.

### Assignment and destruction

The language handles destruction, not the type. Under the destructor design
being developed in #7362, the signature of `Destroy.Op` is fixed by the
interface and the operation that destroys a complete object cannot be
customized, so a type cannot write a destructor against
`Core.MaybeUnformed(Self)`. It does not need to, because the two categories of
unformed state each handle destruction on their own:

-   A type using `UnformedInvalid` implements `IsUnformed`, so the language
    tests for the unformed state before destroying and skips destruction when
    the test succeeds.
-   A type using `UnformedNoop` has by construction chosen a state that is valid
    and whose destruction does nothing, so destruction can run unconditionally.
    It can equally be skipped, which is what makes the state a no-op state.

Destroying a `Core.MaybeUnformed(T)` follows the same rule, which means a field
declared with that type is destroyed correctly by the ordinary field-wise
destruction of its containing object.

Everything else about destruction follows #7362.

Assignment, unlike destruction, may be handled by the type. It can opt in by
declaring its object parameter as `Core.MaybeUnformed(Self)`, in which case it
is called whether or not the object is unformed and is responsible for handling
both. When it does not opt in, the two categories again differ. For a type using
`UnformedInvalid` the language tests `IsUnformed` first, and initializes the
object instead of assigning to it when the test succeeds, which is the behavior
[the design already describes](/docs/design/assignment.md) for an unformed
object. For a type using `UnformedNoop` there is nothing to test, and the
implementation runs unconditionally, which is sound because the unformed state
is a valid value.

> **Future work:** Assignment is currently a builtin operation rather than an
> interface. When it becomes one, its declaration will need to permit an
> implementation to opt in this way.

### Hardening the unformed state

The [release build mode](/docs/design/safety/README.md#build-modes) includes a
baseline of hardening, and initialization is one of the places it applies. Most
of that should be automatic and require nothing from the type. But a type
sometimes knows a representation that is meaningfully better to leave behind
than the one it would otherwise use, and we give it a way to say so. The
[partial class type](/docs/design/classes.md#partial-class-type) already works
this way for the vtable pointer of an object under construction, which the build
mode fills with a null pointer or a poison vtable, so the precedent for a
build-mode-dependent representation of a not-yet-meaningful field exists.

An object that is unformed does not have _the_ unformed state, but one of
possibly several:

-   The _unformed representation set_ of a type is the set of representations an
    object may hold while unformed.
-   The _unformed value_ is the one `UnformedInit` writes.
-   The _hardened unformed value_ is the one written instead when the build mode
    asks for hardening.

`IsUnformed` tests for membership in the unformed representation set, not
equality with the unformed value, which is what makes it correct for a type to
harden to a different representation than it normally uses.

```carbon
interface UnformedHardenInit {
  require impls UnformedInit;
  private default let StructT: type = {};
  default fn Op() -> Core.MaybeUnformed(Self) { return {}; }
}

interface UnformedHarden {
  private default let StructT: type = {};
  private default let Value: StructT = {};
}

final impl forall [T: UnformedHarden] T as UnformedHardenInit
    where .StructT = T.(UnformedHarden.StructT) {
  fn Op() -> Core.MaybeUnformed(Self) { return T.(UnformedHarden.Value); }
}
```

`UnformedHardenInit.StructT` has the same restrictions as
`UnformedInit.StructT`, and must additionally be a superset of its fields, since
hardening may initialize more of the object but never less. _Hardening_ an
unformed value means setting those fields to the result of
`UnformedHardenInit.Op`, along with any additional initialization the compiler
performs for security in the face of unsafe code. It happens only to unformed or
uninitialized objects, never to well formed ones, and there is no restriction on
whether both `UnformedInit.Op` and `UnformedHardenInit.Op` are called or only
one.

The hardened value has to stay consistent with the rest of the type:

-   It must match whichever semantic model the type uses. A type using
    `UnformedInvalid` must harden to a representation that is also invalid, and
    a type using `UnformedNoop` to one that is also a no-op state. Mixing the
    two would give a build with hardening enabled different semantics from one
    without.
-   For a type using `UnformedInvalid`, it must also be in the unformed
    representation set, so that `IsUnformed` still recognizes it. A type using
    `UnformedNoop` has no `IsUnformed`, and so nothing to stay consistent with.

A type that implements `UnformedHardenInit` directly, rather than through
`UnformedHarden`, and whose own `IsUnformed` does not recognize its hardened
value, violates the second rule and is ill-formed. Types going through
`UnformedInvalid` and `UnformedHarden` cannot make that mistake, because the
blanket `impl` of `IsUnformed` for a type implementing both derives from both
constants.

> **Open question:** We can declare that combination ill-formed but cannot
> currently detect it. Doing better would mean either restricting
> `UnformedHardenInit` to types that go through `UnformedHarden`, or checking a
> hand-written `IsUnformed` against both constants.

These interfaces are not a substitute for broad automatic initialization and
should not be relied on as one. They let a type add hardening where it is most
useful, and surface an especially good value to use when it has one.

We expect some code to fall back to unsafe initialization, especially around C++
interop or during a migration from existing C++ API designs. In these cases we
expect the compiler to do some amount of hardening automatically, to prevent any
bugs in that unsafe code from being as easily exploited, for example
[Clang's trivial auto variable initialization](https://clang.llvm.org/docs/ClangCommandLineReference.html#cmdoption-clang-ftrivial-auto-var-init).

Hardening a pointed-to object is harder than hardening a local, because the
`unsafe as` that reaches it can be arbitrarily far from where the object was
created. We expect hardening to be applied defensively at the point where a
pointer to a possibly unformed object escapes the context that can see its
initialization, since any later use could be such a conversion.

> **Future work:** Where hardening is applied should be pinned down once the
> safety model, its flow checking, and the effects a function declares are in
> place. "The context that can see its initialization" is a flow-sensitive
> judgment, and until it can be stated in those terms the rule above is an
> expectation rather than a specification.

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
as its type, a place carries state that varies from one point in a function to
the next, including whether it is initialized. So a place is either _definitely
initialized_ or _maybe unformed_ at each point, and where control flow merges
the two, the result is maybe unformed.

Flow-sensitive state is computed after type checking, so it cannot participate
in overload resolution or `impl` selection.

The two conversions from earlier follow from this:

-   Converting `T` to `Core.MaybeUnformed(T)` is implicit and always allowed. It
    is also _required_ in order to operate on a place that is maybe unformed,
    which is what forces such an object through the restricted API.
-   Converting `Core.MaybeUnformed(T)` back to `T` is implicit when the place is
    definitely initialized, and otherwise requires `unsafe as`.

A function communicates its effect on the caller's flow state in its signature;
the caller does not infer it from how an argument was passed. A function that
initializes an argument says so, and one that may leave an argument unformed
says so as well. Using a placeholder syntax for those:

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
}
```

A `ref` argument therefore does not silently change an object's state; the
transition comes from a declared effect, and it can go in either direction.

> **Future work:** The placeholder syntax above has no way to say "may or may
> not initialize", which is the shape of a C++ output parameter that is only
> written on success. The effect model this builds on does contemplate
> conditional initialization, so this is a gap in the placeholder rather than in
> the model. Until it is spelled, such a call needs the `unsafe` escape hatch
> below.

The escape hatch is `unsafe as`, converting from `Core.MaybeUnformed(T)` to `T`
even where the place is maybe unformed, including for a reference expression or
in the pointee type of a pointer. When the object really is unformed, this is
where [hardening](#hardening-the-unformed-state) earns its cost, since the
subsequent code will use the full API of the type on an object that does not
satisfy its invariants.

This results in a restrictive model that requires either initialization,
explicit code handling unformed values with `Core.MaybeUnformed`, or `unsafe`.
We suggest treating that as _experimental_ and revisiting it based on
experience, as it may be necessary to make the escape hatch occur automatically
in more cases, especially around C++ interop.

This also constrains what a type can do, not only what a caller can do:

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
> name changing based on surrounding `if` statements is too subtle. Nothing here
> changes the type of a name. A variable of type `T` has type `T` throughout,
> and what varies is only whether the compiler can prove its place initialized,
> which affects validity, not meaning.

### Putting all this together for our use cases

Let's start with `bool` and `T*`. Here, we'll use class types that implement
both in terms of private integers for illustration purposes, but the expectation
is that in practice these would likely be directly provided by builtin
operations.

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
  // "infinite scream" value from LLVM's pattern initialization.
  impl as Core.UnformedHarden
      where .StructT = {.value: u64}
      and   .Value   = {.value = 0xAAAA_AAAA_AAAA_AAAA} {}
}
```

Now let's build the `OwningPtr` version. This will build on the `Ptr(T)` above,
but demonstrate the different category of this type: it requires a non-trivial
destructor!

```carbon
class OwningPtr(T: type) {
  private var ptr: Ptr(T);

  // With reflection, we could potentially provide an automatic
  // way of delegating to a member, but spelling it out for now.
  impl as Core.UnformedInvalid
      where .StructT = {.ptr: Core.MaybeUnformed(Ptr(T))}
      and .Value = {.ptr = Ptr(T).(Core.UnformedInit.Op)()} {}

  impl as Core.UnformedHarden
      where .StructT = {.ptr: Core.MaybeUnformed(Ptr(T))}
      and .Value = {.ptr = Ptr(T).(Core.UnformedHardenInit.Op)()} {}

  // The blanket `impl` does not apply here, because it compares `StructT`
  // values and `Core.MaybeUnformed(Ptr(T))` has no `==`. Delegating to the
  // member's own test is both possible and cheaper.
  impl as Core.IsUnformed {
    fn Op(self: Core.MaybeUnformed(Self)) -> bool {
      // `self.ptr` has type `Core.MaybeUnformed(Ptr(T))`, which still allows
      // calling `.(Core.IsUnformed.Op)()`.
      return self.ptr.(Core.IsUnformed.Op)();
    }
  }

  fn Reset(ref self: Core.MaybeUnformed(Self), var rhs: OwningPtr(T)) {
    if (not self.(Core.IsUnformed.Op)()) {
      // This is a fully formed object, so it has to be destroyed before its
      // storage is reused. Explicit destruction is itself an unsafe operation;
      // #7630 is deciding its name.
      (self unsafe as Self).(Core.Destroy.SelfDestruct)();
    }

    // Imagined syntax to directly re-initialize storage,
    // not part of this proposal. `~rhs` is a destructive move.
    let ref storage: ??? = self unsafe as ???;
    raw_init storage = {.ptr = (~rhs).ptr};
  }

  // The destructor says nothing about unformed state. The language
  // tests `IsUnformed` before destroying and skips this when it holds.
  impl as Core.Destroy {
    fn Op(ref self) {
      SomeDeallocationFunction(self.ptr);
    }
  }
}
```

Next, let's consider an integer where we _cannot_ test for unformed, but the
destructor is a no-op. We use a private member rather than adapting to make the
conversions a bit easier to read, but the effect should be the same.

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

Last but not least, let's imagine an `OptionalOwningPtr(T)` where the optional
itself borrows the unformed state of `OwningPtr(T)` as one of its _valid_
states, and so can't implement testing for being unformed directly. This still
requires a non-trivial destructor though.

```carbon
class OptionalOwningPtr(T: type) {
  fn Make(var ptr: OwningPtr(T)) -> Self {
    return {.ptr = ~ptr};
  }
  fn MakeEmpty() -> Self {
    // Works for any type with an unformed state, not just `OwningPtr(T)`.
    return {.ptr = OwningPtr(T).(Core.UnformedInit.Op)()};
  }

  let PtrT: type = Core.MaybeUnformed(OwningPtr(T));

  private var ptr: PtrT;

  // Note that this isn't invalid, just a noop.
  impl as Core.UnformedNoop
      where .StructT = {.ptr: PtrT}
      and   .Value   = {.ptr = OwningPtr(T).(Core.UnformedInit.Op)()} {}

  // No destructor is needed. The `ptr` member has type
  // `Core.MaybeUnformed(OwningPtr(T))`, so destroying it already tests
  // whether it holds a formed pointer.
}
```

The last example is not hypothetical. `Optional(T*)` in the prelude already uses
a null pointer as its empty representation, and it has to reach for builtins to
do it, because there is no way for `T*` to say that null is its unformed value
or for anyone to ask whether an object currently holds it:

```carbon
// What the prelude writes today.
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

With the interfaces above, and given that `T*` implements `UnformedInvalid` with
null as its value, every builtin here except the copy is replaced by an
operation the type already provides:

```carbon
final impl forall [T: type] T* as OptionalStorage
    where .Type = Core.MaybeUnformed(T*) {
  fn None() -> Core.MaybeUnformed(T*) {
    return (T*).(Core.UnformedInit.Op)();
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

The one remaining `unsafe as` is the one that should be there: `Get` is called
only when `Has` returned true, and nothing in the type system knows that.

Note that this is an illustration rather than a proposal about `Core.Optional`,
which does not yet have an accepted design of its own.

All of these rely in their implementation details on a few more facilities that
are not part of this proposal but expected to come in future proposals:

-   Invoking the destructor on objects
-   Destructively moving objects
-   Turning an object into raw storage
-   Initializing raw storage with a new value

However, the underlying model for the unformed state hopefully makes sense in
isolation. In an effort to decompose a large number of interconnected topics,
we're starting off by setting up the unformed state details and will have
follow-up proposals to fill in these gaps.

## C++ interop

### C++ types and unformed state

First, we want to provide unformed state for as many C++ types as we can without
creating painful surprises for users with the behavior. Provided there isn't
user surprise, types with unformed state have significantly better ergonomics.
However, we can't use default heuristics to synthesize an unformed and _invalid_
state, and so our defaults and heuristics don't provide any `IsUnformed`
implementation.

A default constructible C++ type already implements `Core.Default`, so
`var x: Cpp.SomeType;` calls its default constructor and produces a fully formed
object, and that takes priority over anything here. The unformed state still
matters for such a type, because a containing type composes its own unformed
state out of its members'.

The proposed initial rule for synthesizing an unformed state of a C++ type is if
one of the following applies:

-   If the type is trivially default constructible, then it implements
    `UnformedNoop` with an empty struct type and value. Note that we don't
    require a trivial destructor here, as even if technically non-trivial, it
    cannot correctly read any members. This does mean Carbon may skip a
    destructor call C++ would make, on an object whose members that destructor
    cannot legitimately read.
-   If the type is default constructible and trivially _destroyable_, then the
    type implements `UnformedInit` with a struct type of all its fields and the
    `Op` function returning a default constructed version of the type.

Beyond these defaults we can customize the behavior of specific types by
implementing their unformed state interfaces in wrapping Carbon code to select
specific values.

#### C++ standard library types

We suggest aggressively defining unformed states for widely used vocabulary
types in the C++ standard library to provide the best possible ergonomics during
interop. For many vocabulary types, this will likely be handled by mapping the
type into a Carbon-native type such as C++ pointers to Carbon pointers, and
`std::unique_ptr` into whatever owning pointer Carbon has. For other standard
library types, this should be done in their wrapping Carbon code.

### Passing unformed objects into C++ code

C++ APIs that expect to initialize output parameters passed by reference or
pointer won't have the Carbon `Core.MaybeUnformed` type to identify them. We
propose that in
[permissive Carbon](/docs/design/safety/README.md#safety-modes) there is an
implicit `unsafe` conversion, so that these APIs can be called with an unformed
object provided the API could legitimately initialize the type. After this
conversion the object is assumed well formed, as it would be normally.

We expect [strict Carbon](/docs/design/safety/README.md#safety-modes) to reject
these without an _explicit_ `unsafe` operation, so that unsafe initialization is
gradually all marked in the source.

The explicit form is awkward. A maybe-unformed object has to be converted to
`Core.MaybeUnformed(T)` to be operated on at all and then converted back, on a
variable that was just declared with that type:

```carbon
var t: Cpp.T;
// Casting `t` to its own type.
Cpp.T.Init(ref (t unsafe as Cpp.T));
```

We propose starting with a library function that packages this up:

```carbon
// Asserts that the callee initializes the place `x` refers to.
unsafe fn Escape[T: type](ref x: Core.MaybeUnformed(T)) -> ref T;

var t: Cpp.T;
Cpp.T.Init(ref Core.Escape(ref t));
```

This is more verbose than we would like, particularly with `ref` appearing
twice. If it proves to be a common enough pattern in practice, we can add
dedicated syntax for it later and define that syntax as being exactly this
library call, so nothing about the model changes. Marking `Escape` itself
`unsafe` depends on the same undecided question as `UnsafeAs.Convert`, above.

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

Unformed state and Rust's `MaybeUninit` can look superficially similar, and they
overlap more than the different names suggest.

The primary goal here is different from `MaybeUninit`. Unformed state is about
giving a type's own invalid or no-op states a name, so that the language can use
them to improve the ergonomics of initialization without the _user_ writing
unsafe code. It is aimed at type design. `MaybeUninit` is about modeling
uninitialized memory so that unsafe code can initialize it in ways the safe type
system cannot express, such as filling a buffer from a system call. It is a tool
applied to a type rather than one a type uses to build its own API.

They overlap on fields. A container holding storage for elements it has not
constructed yet declares that storage as `Core.MaybeUnformed(T)` and reaches it
with `unsafe as`, exactly as Rust would with `MaybeUninit<T>`.

The difference that remains is what it means when the compiler produces one.
Carbon's version carries the type's declared unformed state, so an unformed
`Core.MaybeUnformed(T)` may hold a meaningful representation such as a null
pointer, and `IsUnformed` can ask about it. Rust's carries nothing and cannot be
inspected. That is what allows the initialization ergonomics this proposal is
after.

Carbon may still end up needing something like `MaybeUninit` to manage
uninitialized memory that cannot be modeled safely. The plan is to use raw
storage for that directly, but if doing so surfaces a need for an abstraction
layer on top, we should add one.

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
    -   C++ types that are default constructible keep working unchanged, and the
        types that are not can still be declared without an initializer where
        that is what the code needs.
    -   The permissive and strict modes give migrated code a path from calling
        C++ output parameter APIs freely to marking each such call explicitly.

## Alternatives considered

### Keeping `UnformedInit` as a marker interface

We could leave `Core.UnformedInit` as it is: a type says it has an unformed
state, the compiler leaves such objects uninitialized, and hardening in the
release build zeroes them.

This costs nothing to specify and gives away roughly the same security, since
automatic zeroing does not depend on the type saying anything. Everything in
[Problem](#problem) above stays true, though: no type can nominate an invalid
state it already has, nothing can ask whether an object is currently unformed,
and the interface remains ceremony for the types that carry it. Without a way to
detect the state, unformed state cannot compose, and composition is what makes
it worth having in the language rather than in each type.

### Requiring the first `ref` call to initialize

An earlier form of this proposal restricted a known-unformed object to being
converted to `Core.MaybeUnformed(T)`, until it had been passed as a `ref`
parameter of that type to some call, and required that first call to initialize
it. This needs no flow-sensitive state and no effects, only a single forward
walk over the function, and nothing has to be declared on the function being
called.

It fails in both directions. It places an obligation on the callee that appears
nowhere in the callee's signature, so the check cannot be enforced where the
callee is defined. And a call taking a `ref` parameter is as likely to leave an
object unformed as to initialize it, which the rule has no way to express.
Whether a call initializes its argument or leaves it unformed is part of that
function's contract, so it has to be stated there.

### Making the unformed state a property of the type

Instead of a per-place initialized bit, we could refine the type of the object,
so that a partly initialized object has a type recording which of its fields are
unformed, along the lines of `Thing | {x.f unformed}`.

Advantages:

-   Field granularity, rather than the whole-object granularity of
    `Core.MaybeUnformed(T)`.
-   No separate flow-sensitive state to specify.

Disadvantages:

-   Such a type names identifiers, so it cannot escape the scope those names are
    in. Storing one in a field, returning it, or capturing it in a closure all
    require carrying the names along, which is a significant addition to the
    type system.
-   It is flow-sensitive typing of exactly the kind Carbon's
    [low context sensitivity](/docs/project/principles/low_context_sensitivity.md#flow-sensitive-typing)
    principle sets a high bar against.

`Core.MaybeUnformed(T)` names no identifiers, so it composes with the rest of
the type system without special handling. The cost is that per-field unformed
state is not expressible in a type, and has to be handled either within a window
that nothing else can observe, or by declaring the field itself as
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
options here has not materially changed since #257 and so we don't duplicate
that list and analysis here.

Fundamentally, this proposal suggests that there is still a good motivation to
try and match the idioms across C++ code where types have this "partially
formed" (what we're calling "unformed") state that is used for deferred
initialization and potentially moved-from states. This pattern continues to be
prevalent and well liked in C++. We hope that the model here allows Carbon to
provide a very natural access to this pattern while also providing a path to
ever more careful and strict checking of its correct usage.

### Folding the hardened value into `UnformedInvalid` and `UnformedNoop`

Rather than separate hardening interfaces, `UnformedInvalid` and `UnformedNoop`
could each gain a second constant, defaulting to something meaning "no separate
hardened value", which is used instead of `Value` when hardening.

This is two fewer interfaces, and it makes the consistency rules above
unnecessary: a type could not state a hardened value from a different semantic
category than its unformed value, because both would come from the same `impl`.
That is a real advantage, and it is the reason to revisit this if those rules
prove hard to follow in practice.

We keep them separate because a type that wants a hardened value but no unformed
state would otherwise have to claim an unformed state to get one. Such types
exist, and folding the two together makes the future work item on hardening
without unformed state harder rather than easier.

### Deriving the hardened value from the build configuration

Rather than a type naming two values, `UnformedInvalid` could name one value
whose definition depends on the build configuration, so that a hardened build
simply computes something different.

Advantages:

-   One interface and one constant, with no possibility of the two disagreeing.
-   No need for `IsUnformed` to test for more than one representation.

Disadvantages:

-   Build configuration becomes an input to the type system, so the same source
    would produce different types in different builds. That cuts against
    separate compilation and against libraries built in one mode being used in
    another.
-   A type could no longer state that its hardened representation is more
    expensive but safer, since there would be no place to say which is which.

Making build configuration an input to the type system is the disqualifying
part. Two values and a consistency rule is a smaller cost than that.

### Using `private adapt` instead of `unsafe adapt`

The design already contemplates restricting the casts between an adapter and its
adapted type by writing `private adapt`, where the conversions are available
only within the library defining the adapter. That reuses an existing concept,
needs no new keyword, and fits an adapter that records a validated invariant
only the library should be able to assert.

We reject it as a substitute, because it does not describe what is true of the
conversion. Reaching the raw storage of an object is unsafe wherever it is
written, including inside the library that owns the type, and `unsafe` is what
an audit looks for. Access control and safety are separate concerns, and using
one to express the other means neither can be applied on its own.

They are not exclusive, though. A type may well want both, and
`private unsafe adapt` should mean what it says.

### Making `MaybeUnformed` a type qualifier

`Core.MaybeUnformed(T)` could be written as a keyword qualifier, such as
`unformed T`, matching `const T` and `partial T`. The toolchain already models
all three the same way internally.

Advantages:

-   Consistent with the other two qualifiers, which it otherwise resembles
    closely.
-   Shorter to write, and reads better in a field declaration.

Disadvantages:

-   It is already spelled as a class in shipped code, including in the prelude
    and in an accepted proposal, so changing it now has a cost with no
    functional benefit.
-   A class composes with a type argument in generic code without any special
    handling, which a keyword would need.

We keep the class spelling, and note that switching later would be a
straightforward rename if the qualifier reads better in practice.

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
