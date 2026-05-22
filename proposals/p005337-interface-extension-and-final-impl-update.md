# Interface extension and `final impl` update

<!--
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

[Pull request](https://github.com/carbon-language/carbon-lang/pull/5337)

<!-- toc -->

## Table of contents

-   [Abstract](#abstract)
-   [Problem](#problem)
-   [Background](#background)
-   [Proposal](#proposal)
-   [Details](#details)
    -   [`extend require`](#extend-require)
    -   [`extend impl as` with parameterized interfaces](#extend-impl-as-with-parameterized-interfaces)
    -   [`extend impl as` restrictions](#extend-impl-as-restrictions)
    -   [Name conflicts](#name-conflicts)
    -   [Final `impl` priority](#final-impl-priority)
        -   [Overlapping final `impl`s](#overlapping-final-impls)
        -   [Using associated constants from `impl`s in a `final match_first`](#using-associated-constants-from-impls-in-a-final-match_first)
        -   [`impl` selection algorithm](#impl-selection-algorithm)
        -   [An `impl` that can never match is an error](#an-impl-that-can-never-match-is-an-error)
    -   [Impl names for prioritization](#impl-names-for-prioritization)
-   [Rationale](#rationale)
-   [Alternatives considered](#alternatives-considered)
    -   [Allow overlap between a non-final and final impl but only if no queries pick the non-final on the overlap](#allow-overlap-between-a-non-final-and-final-impl-but-only-if-no-queries-pick-the-non-final-on-the-overlap)
    -   [Forbid overlap between final impls](#forbid-overlap-between-final-impls)
    -   [Prioritize between final impls using type structure](#prioritize-between-final-impls-using-type-structure)
    -   [Allow mixing final and non-final impls in the same `match_first` block](#allow-mixing-final-and-non-final-impls-in-the-same-match_first-block)
    -   [No default `Self` in `require Self impls I`](#no-default-self-in-require-self-impls-i)
    -   [Different rules for prioritizing between final impls](#different-rules-for-prioritizing-between-final-impls)
    -   [Different `final match_first` associated constant rules](#different-final-match_first-associated-constant-rules)

<!-- tocstop -->

## Abstract

We make 5 changes:

-   Allow `require Self impls I` in an `interface` or `constraint` scope to omit
    the `Self`, so it can be written `require impls I`.
-   Rename `extend I` to `extend require impls I` in an `interface` or
    `constraint` scope.
-   Define `extend impl as I` and `extend final impl as I` in an `interface`
    scope to copy the members of `I` and define an `impl` of `I` in terms of the
    extending interface.
-   Allow a non-final `impl` to overlap a final `impl` as long as it isn't
    subsumed by the final `impl`. The final `impl` will be given priority on the
    overlap.
-   Allow `final` on a `match_first` block, used to declare overlapping final
    impls, which are required to be in a single file.

These features work together to allow a form of interface extension where:

-   Types only need to `impl` the extending interface to also get an `impl` of
    the extended interface.
-   Multiple interfaces can extend the same interface.
-   An interface can extend multiple interfaces.

## Problem

As part of investigating
[leads issue #4566](https://github.com/carbon-language/carbon-lang/issues/4566),
we discovered problems with using a single `impl` to implement multiple
interfaces.
[Proposal #5168 (Forward `impl` declaration of an incomplete interface)](https://github.com/carbon-language/carbon-lang/pull/5168)
removed that, but this changed the experience of `impl` of an interface that
uses `extend`. Consider two closely coupled interfaces, such as:

```carbon
interface As(T:! type) {
  fn Convert[self: Self]() -> T;
}

interface ImplicitAs(T:! type) {
  extend As(T);
}
```

With #5168, you can no longer just `impl` `ImplicitAs(T)` with something like

```carbon
impl ThisType as ImplicitAs(ThatType) {
  fn Convert[self: Self]() -> ThatType { ... }
}
```

Instead, two `impl` definitions are needed:

```carbon
impl ThisType as As(ThatType) {
  fn Convert[self: Self]() -> ThatType { ... }
}
impl ThisType as ImplicitAs(ThatType) { }
```

This is additional friction, which will increase with the depth of the interface
hierarchy, which we would like to eliminate. The suggestion in
[proposal #5168](https://github.com/carbon-language/carbon-lang/pull/5168), is
to support the `ImplicitAs` use case with a `final` blanket `impl` of the
`As(T)` interface for any type that `impls ImplicitAs(T)`, as in:

```carbon
interface As(T:! type) {
  fn Convert[self: Self]() -> T;
}

interface ImplicitAs(T:! type) {
  fn Convert[self: Self]() -> T;
}

final impl
    forall [T:! type, U:! ImplicitAs(T)]
    U as As(T) {
  fn Convert[self: Self]() -> T =
      U.(ImplicitAs(T).Convert);
}
```

This means that types that `impl as ImplicitAs(T)` automatically get an `impl`
of `As(T)`, and can't `impl as As(T)` themselves. However, there are some
concerns:

-   There is significant ceremony to express this relationship.
-   These semantics are closer to what we expect users will think `extend` in an
    interface will do.
-   We still want a way to get the current meaning of `extend` in an `interface`
    scope for a few reasons:
    -   This `final impl` requires the two interfaces to be defined in the same
        file, which won't always be possible.
    -   A `final impl` doesn't give much flexibility for situations where you
        need more control, for example to
        [reuse an impl of a specific interface](/docs/design/generics/details.md#use-case-defining-an-impl-for-use-by-other-types).
    -   Also the current `extend` matches the semantics of `extend` in a named
        constraint.
-   Having a `final impl` of `interface As(T)` currently makes it quite
    difficult to have other `impl`s of the same interface.

On this last point, under the current rules the following example would be
rejected:

```carbon
class C(T:! type) { ... }

interface I {}
impl forall [T:! I] C(T) as As(T) { ... }
```

The issue is that this `impl` overlaps with the `final impl` of
`interface As(T)` for any type `C(T)` that implements `ImplicitAs(T)`. This
`impl` is rejected preemptively because it conflicts on the overlap, whether or
not anything inhabits that overlap in the developer's program. This avoids
conflicts only being discovered by some consuming library.

## Background

-   [Proposal #553](https://github.com/carbon-language/carbon-lang/pull/553)
    introduced interface extension
-   [Proposal #983](https://github.com/carbon-language/carbon-lang/pull/983)
    introduced `final impl`
-   [Proposal #2760](https://github.com/carbon-language/carbon-lang/pull/2760)
    introduced `require Self impls I` in an `interface` or `constraint` scope,
    replacing the previous spelling of `impl as I` from
    [proposal #553](https://github.com/carbon-language/carbon-lang/pull/553).
-   [Proposal #2868](https://github.com/carbon-language/carbon-lang/pull/2868)
    allowed an `impl` to overlap a `final impl` as long as agreed on the
    overlap.

## Proposal

First, to make `require Self impls I` usage more consistent with
`impl Self as I` (and since we will be using this more after this proposal), we
allow the `Self` to be omitted. So:

-   `require Self impls I` ✅ allowed, same meaning as before
-   `require impls I` ✅ allowed, same as `require Self impls I`
-   `require T impls I` ✅ allowed, same meaning as before

Next, since we are creating another way that an `interface` can `extend` another
interface, we change the existing usage of `extend` from being an introducer of
its own declaration, to a modifier on a `require` declaration. In this case, the
only type that can be constrained is `Self`, since that is the type being
extended. To match the restriction that an `extend impl` must be followed by
`as` in `class` scope, `extend require` can only be followed by `impls` in
`interface` scope:

-   `extend require Self impls I` ❌ forbidden, `Self` must be omitted since no
    other value allowed
-   `extend require impls I` ✅ allowed
-   `extend require T impls I` ❌ forbidden

`extend require impls I` after this proposal has the same meaning as `extend I`
did before this proposal, so:

```carbon
interface A {
  fn F();
}

interface B {
  extend require impls A;
  fn G();
}
```

is equivalent to:

```carbon
interface A {
  fn F();
}

interface B {
  require impls A;
  alias F = A.F;
  fn G();
}
```

We then add another way to use `extend` as a modifier in an `interface` scope.
We define the meaning of `extend impl as I;` in an `interface J` scope to do two
things:

-   Copy the members of `I` to form new members of `J`.
-   Define a blanket `impl` that anything that `impls J` also `impls I`, by
    forwarding to the corresponding members of `J`.

These semantics are intended to be similar to `extend impl` in a `class`, except
that the definition of the blanket `impl` is generated automatically (and `impl`
without `extend` is not allowed in an interface).

In this example:

```carbon
interface I {
  let T:! type;
  fn F();
  fn G();
}

interface J {
  extend impl as I;
  fn H();
}
```

The definition of `interface J` is equivalent to:

```carbon
interface J {
  let T:! type;
  fn F();
  fn G();
  fn H();
}
impl forall [U:! J] U as I {
  let T:! type = U.(J.T);
  fn F() = U.(J.F);
  fn G() = U.(J.G);
}
```

The keyword `final` can be added between `extend` and `impl` to make the
generated `impl` definition `final`. For example, the
`final impl forall [T:! type, U:! ImplicitAs(T)] U as As(T)` from
[the "Problem" section](#problem) would be generated from this:

```carbon
interface ImplicitAs(T:! type) {
  extend final impl as As(T);
}
```

To allow a non-final `impl` to overlap a final `impl`, we say the final `impl`
is prioritized over a non-final `impl` anytime they overlap. This addresses a
[problem](#problem) caused by the fact that we don't support the negative
constraints that would allow a developer to avoid the overlap -- and the only
sound choice is to prefer the final `impl`. However, if a non-final `impl` can
never be used because it is completely subsumed by a final `impl`, that is an
error.

To allow multiple interfaces extending the same interface, we also want to
support a way to declare overlapping final `impl`s. We do this by listing the
names of the overlapping `impl`s in a `final match_first` block instead of
putting `final` on the `impl` declarations. Note this requires all of the final
`impl`s to be in the same file.

The `impl` generated by an `extend impl as Foo` in an interface `Bar` can be
named as `impl Bar.(as Foo)`. This name can be used to put such an `impl` into a
`match_first` block (or a `final match_first` block).

This example inspired by
[C++ iterator categories](https://en.cppreference.com/w/cpp/iterator) shows
using these pieces together:

```carbon
interface Iterator {
  fn Increment[addr self: Self*]();
}

interface InputIterator {
  let Element:! type;
  extend impl as Iterator;
  fn Get[addr self: Self*]() -> Element;
}

interface OutputIterator {
  let Element:! type;
  extend impl as Iterator;
  fn Set[addr self: Self*](x: Element);
}

// Makes both impls final and prioritizes them.
final match_first {
  impl InputIterator.(as Iterator);
  impl OutputIterator.(as Iterator);
}

interface ForwardIterator {
  extend final impl as InputIterator;
  fn Copy[self: Self]() -> Self;
  fn Equal[self: Self](compare: Self) -> bool;
}

interface OutputForwardIterator {
  // Need to be careful with `Element`, see the
  // "Name conflicts" section.
  let Element:! type;
  extend final impl as ForwardIterator
      where .Element = Element;
  extend final impl as OutputIterator
      where .Element = Element;
}

class MyIntIterator { ... }
// An impl of OutputForwardIterator also implements:
// Iterator, InputIterator, OutputIterator, and
// ForwardIterator.
impl MyIntIterator as OutputForwardIterator {
  where Element = i32;
  fn Increment[addr self: Self*]() { ... }
  fn Get[addr self: Self*]() -> i32 { ... }
  fn Set[addr self: Self*](x: i32) { ... }
  fn Copy[self: Self]() -> Self { ... }
  fn Equal[self: Self](compare: Self) -> bool { ... }
}
```

## Details

Note that `extend` being first, before `final` or `impl`, follows
[our other uses of `extend`](https://github.com/carbon-language/carbon-lang/issues/995#issuecomment-1417391051).

### `extend require`

Some notes and clarifications on `extend require`:

-   To be consistent with other uses of `extend`, we use the same name lookup
    rule: name lookup into a scope that extends other scopes first looks in that
    scope, and then if it finds nothing, it looks into all of the extended
    scopes, and the lookup is an error if there's more than one different
    result. Example:

    ```carbon
    interface A {
      fn F();
      fn G();
      fn H();
    }

    interface B {
      fn G();
      fn J();
      fn K();
    }

    interface C {
      extend require impls A;
      extend require impls B;
      fn F();
      fn J();
    }
    ```

    This definition of `interface C` means `A` and `B` are required and name
    lookup into `C` for:

    -   `F` finds `C.F`, hiding `A.F`;
    -   `G` is ambiguous, due to the conflict between `A.G` and `B.G`;
    -   `H` finds `A.H`;
    -   `J` finds `C.J`, hiding `B.J`;
    -   `K` finds `B.K`.

    Note that if we had two more interfaces:

    ```carbon
    interface D {
      fn G();
    }
    interface E {
      extend require impls C;
      extend require impls D;
    }
    ```

    then lookup into `E` for `G` is also ambiguous, since the ambiguous lookup
    into `C` for `G` still counts as a conflict with `D.G`.

-   If interface `B` has `extend require impls A`, then any `impl C as B` will
    require an `impl C as A`. We no longer support implementing the members of
    `A` in an `impl` of `B`, see
    [leads issue #4566](https://github.com/carbon-language/carbon-lang/issues/4566)
    and
    [proposal #5168](https://github.com/carbon-language/carbon-lang/pull/5168).

-   If we accept
    [the `extend api`/`extend alias` proposal #3802](https://github.com/carbon-language/carbon-lang/pull/3802),
    then `extend require impls A;` becomes equivalent to `require impls A;` plus
    `extend api A;` (or `extend alias A;`, depending on the syntax we choose).

### `extend impl as` with parameterized interfaces

When using `extend impl as` in a parameterized interface, those parameters end
up in the `forall` clause in the generated `impl`. For example:

```carbon
interface PointerContainer(T:! type) {
  extend impl as Container(T*);
}
```

generates this `impl`:

```carbon
impl forall
    [T:! type, U:! PointerContainer(T)]
    U as Container(T*) { ... }
```

### `extend impl as` restrictions

`extend impl as` can only be used if the generated `impl` would be legal. This
includes the [the orphan rule](/docs/design/generics/details.md#orphan-rule), so
the interface or type argument must be in the same file. For example:

```carbon
class BigInt { ... }

interface IntLike {
  // Only valid in the same library as `ImplicitAs`
  // or `BigInt`.
  extend impl as ImplicitAs(BigInt);
}
```

The
[additional restrictions on final impls](/docs/design/generics/details.md#libraries-that-can-contain-a-final-impl)
mean that `extend final impl as` can only be used in the same library as the
interface being extended, not the type arguments.

Similarly, the expression to the right of the `as` must correspond to a single
interface, due to
[leads issue #4566](https://github.com/carbon-language/carbon-lang/issues/4566)
and [proposal #5168](https://github.com/carbon-language/carbon-lang/pull/5168).

### Name conflicts

If two associated constants or functions have the same name in the two
interfaces, this is an error since there is no way to assign values to both in
an `impl` of the extending interface.

Example with an associated function:

```carbon
interface A {
  fn F();
}

interface B {
  fn F();
  // Name conflict with `F`.
  extend impl as A;
}

class C {
  impl as B {
    // Probably refers to the `F` explicitly declared in `B` --
    // no way to implement the `F` that comes from `A`.
    fn F() { ... }
  }
}
```

Example with an associated constant:

```carbon
interface A {
  let T:! type;
}

interface B {
  let T:! type;
  // Name conflict with `T`.
  extend impl as A;
}

class C {
  impl as B {
    // Two different associated constants named `T` and no way
    // to distinguish them.
    where T = i32;
  }
}
```

Notice that the rewrite to put the members of `A` into `B` would result in a
conflict. Further, if there is only one member with that name after the rewrite
in `B`, then there is no way to write the generated `impl` to set the member of
`A` from the values of the associated constants in `B`.

However, if the extending interface gives a value to the associated constant,
there is no need to specify that value in an `impl`, so that is not an error.

```carbon
interface A {
  let T:! type;
  fn F() -> T;
}

interface B {
  let T:! type;
  // Okay
  extend impl as A where .T = T*;
}

class C {
  impl as B {
    // `T` here is `B.T`
    where T = i32;
    // `F` here is `B.F` and `A.F`.
    fn F() -> i32*;
  }
}
```

This is equivalent to:

```carbon
interface A {
  let T:! type;
  fn F() -> T;
}

interface B {
  let T:! type;
  // Skip A.T due to conflict.
  fn F() -> T*;
}

impl forall [U:! B] U as A where .T = .(B.T)* {
  fn F() -> T = U.(B.F);
}

class C {
  impl as B {
    where T = i32;
    fn F() -> i32*;
  }
}
```

This was used in the `OutputForwardIterator` example in
[the "Proposal" section](#proposal).

Name conflicts are not expected to generally be a problem in practice since this
form of extension often requires the two interfaces to be defined in the same
file.

### Final `impl` priority

With the design prior to this proposal, a final `impl` can only overlap another
(final or non-final) `impl` if they have the same definition on their overlap.
This means that if there is a `final` blanket impl of an interface, as is
generated by `extend final impl as` in the interface's scope, then other `impl`s
of that interface may be overly restricted. This example:

```carbon
package Core;

interface As(T:! type) {
  fn Convert[self: Self]() -> T;
}

interface ImplicitAs(T:! type) {
  extend final impl as As(T);
}
```

is equivalent to:

```carbon
package Core;

interface As(T:! type) {
  fn Convert[self: Self]() -> T;
}

interface ImplicitAs(T:! type) {
  fn Convert[self: Self]() -> T;
}

final impl forall [T:! type, U:! ImplicitAs(T)] U as As(T) {
  fn Convert[self: Self]() -> T = U.(ImplicitAs(T).Convert);
}
```

Any other `impl` of `As(T)` is going to overlap unless we could establish that
the type does not implement `ImplicitAs(T)`. However, in a generic context, we
don't support (and don't intend to support) that sort of negative constraint. So
there would be no way to prove the following `impl` of `As(T)` doesn't overlap
the `final impl` of `As(T)`:

```carbon
import Core;

class X(T:! type) {
  impl as As(T);
}
```

In particular, an unrelated library defining `MyType` could define
`impl X(MyType) as ImplicitAs(MyType)`, which would create a conflict between
the `impl` in `X(T)` and the `final impl` associated with `ImplicitAs(T)`.

This motivates a change to the rules: we want to allow a non-final `impl` to
overlap with a final `impl`. The question is what to do for queries in the
overlap.

Now consider this second example:

```carbon
import Core;

class Array(Element:! type) { ... }

// Impl 1
impl forall [T:! type, U:! As(T)]
    Array(U) as As(Array(T)) { ... }

// Impl 2
impl forall [T:! type, U:! ImplicitAs(T)]
    Array(U) as ImplicitAs(Array(T)) { ... }
```

This creates a conflict:

1.  Impl 1 provides `impl Array(U) as As(Array(T))`.
2.  Impl 2 provides `impl Array(U) as ImplicitAs(Array(T))`.
3.  The
    [`Core` package defined at the beginning of this section](#final-impl-priority)
    has `final impl forall [T:! type, U:! ImplicitAs(T)] U as As(T)`.
4.  Combining 2 and 3 provides another definition of
    `impl Array(U) as As(Array(T))` which is final. Impl 1 has a more specific
    [type structure](/docs/design/generics/details.md#overlap-rule), though.

Furthermore, this example is realistic:

-   The final blanket `impl` defined in the `Core` package is the definition we
    expect to use.
-   We need impl 1 to support types `T` that only `impl` `As(U)` and not
    `ImplicitAs(U)`.
-   We need impl 2 to support types `T` that `impl` `ImplicitAs(U)`.
-   Carbon will define an implicit conversion from `i32` to `i64`, and it would
    be reasonable to ask if `Array(i32)` implicitly converts to `Array(i64)`,
    which would be a query in the overlap.

Any generic code that sees the final `impl` can assume it applies whenever it is
used. If we don't want to forbid this overlap situation (either when defining a
non-final `impl` with a more-specific type structure or when doing an `impl`
query in the overlap that selects the non-final one), the only sound choice is
to use the final `impl`. And we don't want to give an error here since:

-   It would reduce expressiveness, since there isn't a good workaround to
    prevent the overlap.
-   It would reduce the ability to compose libraries.
-   We want to support examples like this interaction between `Array`, `As`, and
    `ImplicitAs`.

Note that prioritizing the final `impl` over the non-final `impl` (even when the
non-final `impl` has a preferred type structure) produces the desired result in
the example above. By writing `final` we ensure that `Convert` in `ImplicitAs`
always matches `Convert` in `As`, when they are both defined.

Also note that this gives the desired behavior that this `final impl` of the
`CommonTypeWith` interface:

```
final impl forall [T:! type] T as CommonTypeWith(T) where .Result = T {}
```

is prioritized as if it is more specialized than user-written `impl`s of
`CommonTypeWith`,
[as desired](https://github.com/carbon-language/carbon-lang/blob/trunk/docs/design/expressions/if.md#same-type).

#### Overlapping final `impl`s

The next question is what to do if there are overlapping final `impl`s. We
considered options that allowed them to be in different files, but this led to
prioritization that was inconsistent with the
[type structure rules](/docs/design/generics/details.md#overlap-rule) for
non-final `impl`s.

In particular, the restrictions on
[which libraries can define a final `impl`](/docs/design/generics/details.md#libraries-that-can-contain-a-final-impl)
mean there are at most two files that can contain two final `impl` definitions
that both match a particular (type, interface) query: the one defining the root
type and the one defining the root interface. The only way to have the final
`impl`s in different files is if there is a
[blanket impl](/docs/design/generics/details.md#blanket-impl-declarations) of
the interface in the file defining the interface and the other in the file
defining the type. Generic code may only see the file with the interface, and
since it is marked "final" will assume it is safe to use. As a result, the only
sound solution is to prioritize the final blanket `impl` over the final `impl`
that specifies the type, the opposite of how non-final impls prioritize using
type structure. Furthermore, there would be queries that would match the final
`impl` that specifies the type but would not select that `impl`, a property we
would like to have. So we require that overlapping final `impl`s be declared in
the same file, a restriction that the compiler can enforce due to the file
limitations.

> **Future work:** If we determine that we do want to support overlapping final
> impls across the two files, we thought the second file could use
> `extend final match_first` to say "these entries are notionally added to the
> earlier `final match_first`.

Within a file, we have two tools here we could use to select between the
candidates: type structure and
[`impl_priority`/`match_first` blocks](/docs/design/generics/details.md#prioritization-rule).
(We will use the placeholder term of "`match_first` blocks" to refer to these
without asserting how they will be spelled.)

The decision here is to only use `match_first` blocks, not type structure to
choose between overlapping final `impl` definitions, for a few reasons:

-   Type structure is needed to pick betewen `impl` defined in different files,
    which we are not allowing for final impls.
-   A `match_first` block is very explicit way of prioritizing, and with final
    `impl`s in particular it is helpful to be clear about the conditions where
    they won't be selected even when they match.

With this decision not to use type structure to prioritize final `impl`s, it
doesn't make sense to mix `final` and non-final `impl`s in the same
`match_first` block. Furthermore, until the compiler sees the `match_first`
block, we can't treat an `impl` as final since we don't know if another final
`impl` will be appear earlier in the `match_first` block and used instead. Which
leads to the decision to put the `final` modifier on the `match_first` block
instead of the `impl` declaration, when it appears in a `match_first` block.
Example:

```carbon
// Compiler does not treat this as final, so it
// won't use `A = i32`.
impl forall [T:! Z] T as J {
  where A = i32;
  // ...
}

fn F(T:! Z) {
  // Can assume `T impls J`.
  // Can't assume `T.(J.A)` is `i32`.

  // If we were to mark the above `impl` as `final`,
  // this code would assume it is selected, and we
  // would have to poison the impl lookup to prevent
  // the following conflicting `impl` and
  // `match_first` declarations.
}

// Also not treated as final yet, so the compiler
// won't use `A = bool`.
impl forall [U:! Y] U as J {
  where A = bool;
  // ...
}

// Matching `bool` assignment to `J.A` as the
// previous impl.
impl forall [V:! W] V as J {
  where A = bool;
  // ...
}

final match_first {
  impl forall [U:! Y] U as J;
  impl forall [V:! W] V as J;
  impl forall [T:! Z] T as J;
}

// Now all are considered final, with the
// understanding that `impl forall [T:! Z] T as J`
// may not be used even when it matches. The
// compiler can now tell if it can use `A = bool`
// from `impl forall [U:! Y] U as J`.

fn G(U:! Y) {
  // Can assume `U impls J` and `U.(J.A)` is `bool`.
}

fn H(V:! W) {
  // Don't know if `V impls Y`, so we don't know
  // which `impl` is selected. We use a symbolic
  // witness for `V` and `V.(J.A)` is unknown
  // (despite that it would be `bool` in either
  // case).
}
```

> Note: `final match_first` was considered in
> [open discussion on 2023-09-13](https://docs.google.com/document/d/1s3mMCupmuSpWOFJGnvjoElcBIe2aoaysTIdyczvKX84/edit?resourcekey=0-G095Wc3sR6pW1hLJbGgE0g&tab=t.0#heading=h.4r37mjr7c42h)
> as a part of function overloading, and other times since.

This leads us to these rules:

-   An `impl` is considered "final" if is declared with the `final` keyword
    modifier or if it is named in a `final match_first` block.
    -   In the first case (a `final impl` declaration), that `impl` may not
        appear in any `match_first` block.
-   An `impl` may appear in at most one `match_first` block.
    -   This is true whether or not the block is marked `final`.
-   Two final `impl`s in the same file must appear in the same
    `final match_first` block if they overlap, as determined by their type
    structure.

#### Using associated constants from `impl`s in a `final match_first`

As established by
[proposal #5168](https://github.com/carbon-language/carbon-lang/pull/5168),
concrete `impl` queries resolve to a concrete `impl` witness, with the
associated constants from a single `impl` definition. A symbolic `impl` query
can in some cases use the associated constants from a final `impl`. If an `impl`
query matches an `impl` marked `final` (as opposed to one in a
`final match_first`), then that `impl` will definitely be used. In that case,
the associated constants from the `impl` can be used.

If a symbolic query selects an `impl` from a `final match_first` block, though,
that `impl`'s associated constants may only be used if it is the first that
_could_ match in its `final match` block, as determined by the type structures
of the `impl`s that appear earlier. Phrased another way, a query will use the
associated constants of a final `impl` if it both matches and has no overlap
with the type structure of any of the `impl`s that appear before it in its
`final match` first block. For example:

```carbon
interface L(T:! type) {
  let B:! type;
}
class C(T:! type) {}

final match_first {
  // Can use the associated constants from this impl,
  // since it is the first, so nothing could match ahead
  // of it.
  impl forall [U:! Y] C(U) as L(bool) where .B = i32;

  // Queries that match this impl and can't match the
  // previous impl's type structure can use the
  // associated constants from this impl. In this case,
  // that happens if the self type (left of the `as`) of
  // the query can't match `C(?)`.
  impl forall [V:! W] V as L(bool) where .B = bool;

  // Can use the associated constants from this impl,
  // since any query that matches `T as L(i32)` won't
  // match the type structure of the earlier impls in
  // the `final match_first`, since they all have
  // `L(bool)` to the right of `as`.
  impl forall [T:! Z] T as L(i32) where .B = f32;
}

class D(X:! type) {}
impl forall [X:! type] D(X) as W {}
// D(X) as L(bool) uses the middle `impl forall` from
// the `final match_first` so the return type is `bool`.
// Note that we can conclude `D(X)` does not match the
// type structure of the first `impl forall` of
// `C(?) as`... even though the query is not concrete.
fn F(X:! type) -> (D(X) as L(bool)).B {
  return true;
}

class E(T:! type) {}
impl forall [T:! type] C(E(T)) as W {}
// `C(E(T))` does impl `W`, so it impls `L(bool)`. But
// since we don't know whether `E(T)` impls `Y`, we
// don't know which impl it will ultimately use, and so
// `(C(E(T)) as L(bool)).B` is unknown.
```

#### `impl` selection algorithm

In conclusion, we prioritize final `impl`s over non-final `impl`s. In the case
that there are multiple matching final `impl`s defined in the same file, they
overlap so they must be in a single `final match_first` block. In that case,
pick the first matching one listed in the `final match_first` block.

If no final `impl` declarations match the query, we fall back to the original
rules:

-   The non-final `impl` declarations matching the query with the most preferred
    type structure must be in the same non-final `match_first` block, or a
    single non-final `impl` declaration not in a `match_first` can match.
    -   In the latter case, that `impl` is selected. Alternatively, `impl`
        declarations not in any `match_first` block could be considered to be
        the only member of their own `match_first` block.
-   If a non-final `match_first` block is chosen, the first matching `impl` in
    that block is selected.

Since it may be observable which `impl` declarations are considered during impl
lookup, due to the [acyclic rule](/docs/design/generics/details.md#acyclic-rule)
and [termination rule](/docs/design/generics/details.md#termination-rule), we
specify this more precisely:

-   Only (final or non-final) `impl`s with a type structure compatible with the
    query are considered.
-   Final `impl`s are considered in the order they appear in the
    `final match_first` block, if any.
    -   The first matching final `impl` considered is returned, skipping the
        remaining steps.
-   Non-final `impl`s are considered most specific type structure first.
-   Non-final `impl`s with the same type structure are considered in the order
    they appear in their common `match_first` block.
-   Once the first matching non-final `impl` is found:
    -   If it is not in a `match_first` block, it can be returned, skipping the
        remaining steps.
    -   The non-final `impl`s earlier in the same `match_first` block with
        compatible type structure are considered in the order they appear in the
        `match_first` block
        -   We skip those with an equal or preferred type structure compared to
            the matching `impl`, since those have already been considered and
            determined to not be matching.
    -   The first matching `impl` is selected. If no `impl` matches before the
        one used to select the `match_first` block, that `impl` is selected and
        no later `impl`s will be considered.

#### An `impl` that can never match is an error

Instead of the current rule that prevents a (final or non-final) `impl` from
overlapping an existing final `impl`, we have a replacement rule that says a
(final or non-final) `impl` that will never be selected due to being subsumed by
a final `impl` is an error.

The compiler can detect whether an `impl` declaration is subsumed by performing
an `impl` lookup (restricting to final impls) for the query represented by its
declaration (ignoring any trailing `where` clause setting associated constants).
For the declaration `impl forall [T:! I] X(T) as J where .A = bool`, as an
example, the query would be "`X(T) impls J` with `T impls I`".

Note that
[the rules for which libraries can define a final `impl`](/docs/design/generics/details.md#libraries-that-can-contain-a-final-impl)
mean that a subsuming final `impl` is either visible or is later in the same
file (and such will be diagnosed due to poisoning when we look for a subsuming
final `impl`). In order to detect the subsumption, the query for the final
`impl` should either be delayed to the end of the file, or poisoned.

> _**Concern**_: Consider a situation where a library defines a broad final
> `impl`, but has determined it is a bad idea and wants to remove it in the
> future. One strategy would be to ask clients of the library to define their
> own non-final `impl`s first. This non-final `impl` would be ignored before the
> transition since the final `impl` has higher priority, but would be used once
> the final `impl` is removed. In this case, we would like to suppress the error
> for the subsumed non-final `impl` defined in the client. This is future work,
> but our current idea is to associate the final `impl` with some build
> configuration constant to mark it as conditionally available. A final `impl`
> marked in that way would not be considered to subsume any `impl` not marked
> the same way. This could also be used to allow temporary changes made while
> debugging.

### Impl names for prioritization

The generated `impl` from an `extend impl` in an interface is given a name
similar to an `impl` defined in a `class` scope, as
[discussed on 2024-03-11](https://docs.google.com/document/d/1s3mMCupmuSpWOFJGnvjoElcBIe2aoaysTIdyczvKX84/edit?resourcekey=0-G095Wc3sR6pW1hLJbGgE0g&tab=t.0#heading=h.p69b78lovqb7),
mentioned in [proposal #3763](/proposals/p3763.md#redeclarations), and proposed
in pending
[proposal #5366](https://github.com/carbon-language/carbon-lang/pull/5366).

For example, these names can be used to allow multiple interfaces to extend a
single interface:

```carbon
interface A(T:! type) {
  fn F() -> T;
}

interface B {
  extend impl as A(i32);
  // Produces impl with name `B.(as A(i32))`.
}

interface C1 {
  let U:! type;
  extend impl as A(U);
  // Produces impl with name `C1.(as A(U))`.
}

interface C2 {
  let U:! type;
  // Only difference from C1 is using `Self.U` instead of `U`.
  extend impl as A(Self.U);
  // Produces impl with name `C2.(as A(Self.U))`.
}

interface D(V:! type) {
  extend impl as A(V);
  // Produces impl with name `D(V:! type).(as A(V))`.
}

interface E {}
impl forall [W:! E] W as A(W) { ... }

class F {}
impl F as A(F) { ... }

// Placeholder impl priority syntax
match_first {
  impl B.(as A(i32));

  impl C1.(as A(U));
  impl C2.(as A(Self.U));
  // Can't write `impl C1.(as A(Self.U))` or `impl C2.(as A(U))`.

  impl D(V:! type).(as A(V));

  impl forall [W:! E] W as A(W);

  impl F as A(F);
}
```

Note how the expression after the `as` has to match syntactically between the
two declarations.

Without the `impl` prioritization, these would be conflicting `impl`s of the
interface `A`. Note that by the
[overlapping final `impl` rules](#overlapping-final-impls), at most one
interface can `final extend impl as` a given interface. `final match_first` must
be used instead of multiple `final impl`s, so that there is an explicit
prioritization between them, as in this example:

```carbon
interface I {
  fn IFn();
}

interface J {
  // Can't mark as `final` here.
  extend impl as I;
  fn JFn();
}

interface K {
  // Can't mark as `final` here.
  extend impl as I;
  fn KFn();
}

// Makes both impls final and prioritizes them.
final match_first {
  impl J.(as I);
  impl K.(as I);
}
```

When the extending interface is parameterized, following the approach of
[proposal #3763: Matching redeclarations](https://github.com/carbon-language/carbon-lang/pull/3763),
the same sequence of tokens (between the `interface` and `{`) is used to name
the interface, allowing a syntactic match. For example:

```
interface Z(T:! type) { ... }

interface Y(T:! type) {
  extend impl as Z(T*);
}
// ...

final match_first {
  impl Y(T:! type).(as Z(T*));
  // ...
}
```

## Rationale

This proposal's end goal is to support a form of interface extension that
matches user's expectations, to aid the
["Code that is easy to read, understand, and write" goal](/docs/project/goals.md#code-that-is-easy-to-read-understand-and-write).
This goal also informed the choice to be explicit about prioritizing overlapping
final impls.

The choice to not treat an `impl` as final when it overlaps another final impl
until we see how to choose between them on the overlap is in accord with the
[information accumulation principle](/docs/project/principles/information_accumulation.md).

## Alternatives considered

### Allow overlap between a non-final and final impl but only if no queries pick the non-final on the overlap

This question was considered in the
["final `impl` priority" section](#final-impl-priority).

### Forbid overlap between final impls

The downside of allowing them to overlap is we no longer have the property that
a final `impl` is selected anytime it matches. This puts limits on when the
compiler can use the values of associated constants from a final `impl`.
However, the use cases, such as having multiple interfaces extend a single
interface, were compelling.

### Prioritize between final impls using type structure

This question was considered in the
["Overlapping final `impl`s" section](#overlapping-final-impls).

### Allow mixing final and non-final impls in the same `match_first` block

There were some ways we could imagine supporting a mix of final and non-final
impls in the same `match_first` block. With
[the decision to not use type structure to prioritize between final impls](#prioritize-between-final-impls-using-type-structure),
though, reconciling the differences seemed like it would lead to a lot of
complexity and surprising behavior, like priority inversions. We thought the
approach of putting `final` on the `match_first` block resulted in simpler
rules.

### No default `Self` in `require Self impls I`

Shortening of `require Self impls I` to just `require I` was considered in
[proposal #2760](p2760.md#allow-interfaces-to-require-another-interface-without-writing-self-impls).
It was not chosen to be consistent with `where` clauses, but the door was left
open if it was found to be too verbose. The `require impls I` approach adopted
by this proposal, though, could be extended to work with `where` as well. The
consideration of that change has been left to future proposals, and is out of
scope for this one.

### Different rules for prioritizing between final impls

We considered five options in discussions in
[a GitHub comment on this proposal](https://github.com/carbon-language/carbon-lang/pull/5337/files#r2072024118),
[#generics-and-templates on Discord](https://discord.com/channels/655572317891461132/941071822756143115/1367982721141440734)
on 2025-05-02, and
[open discussion on 2025-05-05](https://docs.google.com/document/d/1Yt-i5AmF76LSvD4TrWRIAE_92kii6j5yFiW-S7ahzlg/edit?tab=t.0#heading=h.g213t2menkod).

1.  Final `impl`s in the file with the interface are prioritized over the final
    `impl`s in the file with root self type of the `impl` (when they are
    different).
2.  Blanket final `impl`s with a `?` for the whole self type in the type
    structure are prioritized over final `impl`s with a concrete root self type.
3.  Final `impl`s are prioritized using the reverse of type structure
    prioritization.
4.  The syntax of final `impl`s associates them with either the interface
    (`final match_first`) or the root self type (`extend final match_first`).
    Those associated with the interface are prioritized first, and then those
    associated with the type are prioritized after. Ties are resolved by type
    structure and then `match_first`.
5.  Final `impl` overlap is only allowed within a `final match_first` block.

Note: Each final `impl`
[must be declared either in the file with the interface or the root type](/docs/design/generics/details.md#libraries-that-can-contain-a-final-impl).

The concern with option 1 was that concatenating files that contain overlapping
final `impl`s, such as when creating a reproduction of a bug, would necessitate
`match_first` changes.

The concern with option 2 was that it was a new rule to learn, unrelated to the
rule for non-final impls. It has the advantage of using explicit `match_first`
to prioritize almost as much as option 1, while allowing files to be
concatenated with fewer changes.

The concern with option 3 was that it might not do what developers expect, due
to being the opposite of non-final `impl` prioritization.

In general, options 1-4 were more complex than option 5. Option 5 maximized the
use of explicit `match_first` to resolve priority on final `impl` overlap,
rather than an implicit prioritization. We were also interested these two
properties:

-   If we have two impls A and B where A is preferred over B, and then we make
    them both final, we should not make them valid but with B preferred over A.
-   A final `impl` is selected for any query that it matches.

which options 1-3 did not satisfy. The main concern with option 5 was that it
might not be able to express desired use cases. We noted, though, that you could
create additional interfaces to support different prioritization policies. For
example:

```carbon
interface ImplicitAs(T:! type) { ... }
interface HighPriorityImplicitAs(T:! type) { ... }
interface LowerPriorityFrom(U:! type) { ... }

final match_first {
  impl forall [T:! type] T as ImplicitAs(T);
  impl forall [T:! type, U:! HighPriorityImplicitAs(T)]
      U as ImplicitAs(T);
  impl forall [U:! type, T:! LowerPriorityFrom(U)]
      U as ImplicitAs(T);
}
```

Notes:

-   This gives control over which type is given special treatment as the self
    type when implementing one of the auxiliary added interfaces.
-   This `final match_first` is with the interfaces, but that does not preclude
    a `final impl` of `HighPriorityImplicitAs` or `LowerPriorityFrom` in the
    files with the self root types.

### Different `final match_first` associated constant rules

We considered
[in discussion on 2025-05-06](https://docs.google.com/document/d/1Yt-i5AmF76LSvD4TrWRIAE_92kii6j5yFiW-S7ahzlg/edit?tab=t.0#heading=h.mygxcb6k4wpm)
a couple of alternative rules for determining when associated constants were
considered to be known, and what value to use, when matching an `impl` from a
`final match_first`.

One approach was to say that the `final match_first` would be allowed to have a
`where` clause specifying the value of a subset of the associated constants of
the interface. Every `impl` named in the body would have to have a consistent
assignment to those associated constants in their definitions. This had two
downsides:

-   It did not support the anticipated overloading use cases well, where the
    return type would vary across overloads.
-   Supporting associated constants whose value varied with the value of a type
    or interface parameter would have added a lot of complexity.

Another approach we considered was to say an associated constant would have a
value if every impl that could match gave the same value. For example, in
[the first example from the "Overlapping final `impl`s" section](#overlapping-final-impls),
the first two impls in the `final match_first` both set `A.J` to `bool`, so
anything matching either of them could use `bool` for the value of `A.J`. This
seemed hard to reason about. Also, there were concerns that this would be very
sensitive to changes, introduce additional fragility in the case of evolution.

We also considered only using associated constants that were included in the
`impl`'s name (using a `where` clause in the facet type instead of a `where`
declaration in the `impl` definition body). This had the advantage that the
logic for determining the values of those associated constants could be done
without the `impl` definitions. This would introduce implementation complexity,
and would cause `impl` names to be longer increasing verbosity. It would also
introduce a difference between a `final impl` and a single `impl` by itself
inside a `final match_first`.
