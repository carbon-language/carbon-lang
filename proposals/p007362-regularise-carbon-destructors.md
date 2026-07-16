# Regularise Carbon Destructors

<!--
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

[Pull request](https://github.com/carbon-language/carbon-lang/pull/7362)

<!-- toc -->

## Table of contents

-   [Abstract](#abstract)
-   [Problem](#problem)
-   [Background](#background)
-   [Proposal](#proposal)
-   [Details](#details)
    -   [`Destructor` interface](#destructor-interface)
        -   [Empty `Destructor.Op()` should be an error](#empty-destructorop-should-be-an-error)
    -   [`Core.Destroy` interface](#coredestroy-interface)
    -   [`Core.TrivialDestroy` interface](#coretrivialdestroy-interface)
    -   [`Core.DynamicDestroy` interface](#coredynamicdestroy-interface)
    -   [Constraining on `Destructor` should be an error](#constraining-on-destructor-should-be-an-error)
-   [Rationale](#rationale)
-   [Alternatives considered](#alternatives-considered)
    -   [Replace interface `Core.Destructor` with method `Core.Destroy.Destructor`](#replace-interface-coredestructor-with-method-coredestroydestructor)
-   [Future work](#future-work)

<!-- tocstop -->

## Abstract

Describe the relationship between `Destroy` and `class` types.

## Problem

Carbon has not specified how `Destroy` works for `class` types. This is a broad
problem because classes may customise how they are destroyed by implementing
`Destroy`.

Carbon exposes most of its language-driven behaviour through implementing and
constraining on interfaces. For example, the expression `a + b` is valid if the
respective types for `a` and `b` correctly implement `Core.AddWith`.

[Destructors] have an irregular design because they were required before Carbon
could support them in the conforming pattern. The core language has since
matured enough for us to redesign destructors so that they can be a regular part
of the design. A language with fewer irregularities has a better user experience
and is easier to learn.

The `Destroy` interface is not defined in the current design. We also provide a
formal definition in this proposal.

## Background

-   **[PR #1154]** proposed the [existing destructor design]. Class types can
    customise their destructor by defining the method `fn destroy(self)`.
    `destroy` has many associated rules.
-   **[Issue #6124]** discusses how `Destroy` should interact with `class`
    types. The Leads decided that destructors are separate from how objects are
    destroyed.
-   **[Issue #6161]** discusses whether `partial` types are destroyable. The
    Leads decided that `Destructor.Op` is valid for both `partial` and final
    types. Dynamic types are to be destroyed by the interface `DynamicDestroy`.
-   **[Issue #6464]** clarifies that trivial destruction is a property that code
    can query. The Leads confirmed this is the case.

## Proposal

This proposal intends to replace the `destroy` method with four interfaces:

-   **`Destructor`**: a user-implementable interface to describe how a single
    object handles its subobjects at the end of its lifetime. Typically used to
    release resources acquired during its lifetime.
-   **`Destroy`:** a toolchain-implemented interface to describe how objects are
    destroyed.
-   **`TrivialDestroy`:** a toolchain-implemented interface to describe that an
    object doesn't call any destructors during its destruction process.
-   **`DynamicDestroy`:** a toolchain-implemented interface so dynamic types can
    dispatch to the correct `Destroy` implementation (similar to
    [virtual destructors]).

## Details

### `Destructor` interface

```carbon
interface Destructor {
  private fn Op(ref self: partial Self);
}
```

`Destructor` describes how an object performs cleanup and resource deallocation.
`Destructor.Op()` tears down a single object of the class that implements the
interface. Subobjects and lifetime management are unaffected. Types only need to
implement `Destructor` when custom destruction logic is required. Types that do
not implement `Destructor` do not have a destructor.

A type has _trivial destruction_ if, and only if, it does not implement
`Destructor` and none of its subobjects implement `Destructor`.

#### Empty `Destructor.Op()` should be an error

An empty `Destructor.Op()` definition should be an error. We should allow users
to "reserve" non-trivial destruction by providing a spelling that communicates
"no behaviour is intentional" using a clearer syntax.

```carbon
class C1 {
  impl as Destructor {
    // error: `C1.(Destructor.Op)()` defined as an empty method
    fn Op(self) {}
  }
}

class C2 {
  // potential alternative spelling
  impl as Destructor = default;
}
```

### `Core.Destroy` interface

```carbon
interface Destroy {
  unsafe final fn Op(ref self) = "destroy.op";
}
```

`Destroy` destroys complete objects---including subobjects---and ends their
lifetimes. Classes cannot customise `Destroy`: it is exclusively implemented by
the compiler. Types requiring control over how subobjects are destroyed must use
a raw storage type for subobjects. `Destroy` behaves uniformly for all
destroyable types:

1.  Remove qualifiers from the `Self` type.
2.  Call `self.(Core.Destructor.Op)()`, if `Self.Destructor` is implemented.
3.  Call `Destroy.Op()` on each subobject. The `.base` subobject has a `partial`
    type during this step.

> [!WARNING] `Core.Destroy.Op` is unsafe.

Types automatically implement `Destroy` unless they:

-   are abstract and not partial;
-   have a subobject that cannot be destroyed;
-   have explicitly opted out of being destructible (see [Future work])

### `Core.TrivialDestroy` interface

```carbon
interface TrivialDestroy {
  final fn Op(ref self) = "destroy.trivial.op";
}
```

`TrivialDestroy` describes types that have trivial destruction.

The toolchain automatically implements the interface `TrivialDestroy` for all
types that have trivial destruction. `TrivialDestroy` cannot be explicitly
implemented.

### `Core.DynamicDestroy` interface

```carbon
interface DynamicDestroy {
  final fn Op(ref self) = "destroy.dynamic.op";
}
```

`DynamicDestroy` dispatches destruction for objects pointing to dynamic types.
`DynamicDestroy` can only be implemented by the toolchain, but there is interest
in relaxing this restriction. The toolchain automatically implements
`DynamicDestroy` for types that:

-   are final and implement `Destroy`, or
-   have a virtual pointer.

`DynamicDestroy.Op` directly calls `Destroy.Op` for final types. Types with a
virtual pointer are required to add a vtable entry containing `Destroy.Op`.
`DynamicDestroy.Op` makes a virtual call to this vtable entry for such types.

> [!NOTE] `Core.DynamicDestroy.Op` is a safe function.

### Constraining on `Destructor` should be an error

Limiting an API to only accept objects that can be destroyed is a helpful
constraint. Conversely, limiting an API to only accept objects that require some
amount of custom clean-up during the destroy operation is an over-constraint.

It is easy to mix up `Destructor` and `Destroy`. Users must implement
`Destructor` for their class types to have a destructor. Only the toolchain is
allowed to implement `Destroy`. Worse: Carbon and C++ differ on what these terms
mean when considered independently, but agree on their combined meaning. Carbon
class types that do not implement `Destructor` have no destructor, but they can
be destroyed. C++ class types that do not explicitly define a destructor will
have an implicit destructor synthesised by the compiler. C++ class types that do
not have a destructor cannot be destroyed.

Constraining on `Destructor` is considered to always be a mistake. Users should
constrain their APIs using `Destroy` to restrict the interface to objects that
can be destroyed. Users should constrain their APIs using `TrivialDestroy` to
restrict the interface to types that have trivial destruction.

## Rationale

The destruction model advances these goals:

-   [Code that is easy to read, understand, and write]: Types implement
    `Core.Destructor`, but are not able to customise how those types are
    destroyed.
-   [Software and language evolution]: This proposal is a successor to
    [P001154 Destructors]. The language has evolved to a point where we can
    implement destructors in a way that is consistent with the rest of the
    design.
-   [Practical safety and testing mechanisms]: `Core.Destructor` automates
    resource disposal during deinitialisation. Automated resource disposal helps
    avoid saftey-related bugs.
-   [Interoperability with and migration from existing C++ code]: C++ objects
    require destructors to be destroyed.

## Alternatives considered

### Replace interface `Core.Destructor` with method `Core.Destroy.Destructor`

The proposed model includes four interfaces. That makes the Carbon destructor
design clunkier than its contemporaries. It is tempting to propose a single
interface that provides multiple methods.

```carbon
interface Destroy {
  unsafe final fn Op(self) = "destroy.op";
  private fn Destructor(ref self: partial Self) = "destroy.trivial";
}
```

Abstract classes are not destroyable, but they can implement a destructor. A
single interface makes it difficult for generics to permit partial classes.
Attempts to address this open the possibility to tricking the generic into
calling `Destroy.Op` for an abstract class. This includes
`impl partial Self as Destroy` for abstract classes.

## Future work

-   Identify an alternative name for `Destructor` (see [Issue #7508]).
-   Add an opt-out from implementing `Destroy`.
-   Describe why `Destructor.Op` is unsafe.
    -   Issue #6124 notes that certain calls to `Destroy.Op` are considered
        safe. These should be codified in the detailed description.
-   Unify with copy and move semantics.
-   Explore how types with virtual pointers can:
    -   opt out from `DynamicDestroy`.
    -   manually implement `DynamicDestroy`.

<!-- Links -->

[PR #1154]: https://github.com/carbon-language/carbon-lang/pull/1154
[Issue #6124]: https://github.com/carbon-language/carbon-lang/issues/6124
[Issue #6161]: https://github.com/carbon-language/carbon-lang/issues/6161
[Issue #6464]: https://github.com/carbon-language/carbon-lang/issues/6464
[Issue #7508]: https://github.com/carbon-language/carbon-lang/issues/7508
[P001154 Destructors]: /proposals/p001154-destructors.md

[destructors]: /docs/design/classes.md#destructors
[existing destructor design]: https://github.com/carbon-language/carbon-lang/blob/a2890716ba7b73bb2bd337addceb3ac534558ee1/docs/design/classes.md#destructors
[virtual destructors]: https://en.wikipedia.org/wiki/Virtual_function#Virtual_destructors
[Future work]: #future-work

[Code that is easy to read, understand, and write]: /docs/project/goals.md#code-that-is-easy-to-read-understand-and-write
[Software and language evolution]: /docs/project/goals.md#software-and-language-evolution
[Practical safety and testing mechanisms]: /docs/project/goals.md#practical-safety-and-testing-mechanisms
[Interoperability with and migration from existing C++ code]: /docs/project/goals.md#interoperability-with-and-migration-from-existing-c-code
