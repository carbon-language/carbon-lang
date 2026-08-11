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
    -   [Explicitly destroying an object](#explicitly-destroying-an-object)
        -   [Naming](#naming)
    -   [Trivial destruction](#trivial-destruction)
-   [Rationale](#rationale)
-   [Alternatives considered](#alternatives-considered)
    -   [Two public interface model (previous design)](#two-public-interface-model-previous-design)
    -   [Replace interface `Core.Destructor` with method `Core.Destroy.Destructor`](#replace-interface-coredestructor-with-method-coredestroydestructor)
    -   [Alternatives considered from PR #1154](#alternatives-considered-from-pr-1154)
-   [Future work](#future-work)

<!-- tocstop -->

## Abstract

Redesign the Carbon destruction process to use interfaces.

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

1.  **Redesign destructors to use the [static open extension mechanism].** to
    resolve the above problem. This redesign is discussed in the details.
2.  **Add [Object destruction] to [docs/design/values.md]**. The static open
    extension mechanism unifies the destruction process for all Carbon objects.
    We describe object destruction in `values.md` to account for objects that
    aren't classes.
3.  **Remove [Destructors] in [docs/design/classes.md] with [Object destruction]
    in [docs/design/values.md].**

## Details

### Explicitly destroying an object

We can expose explicit object destruction either by way of a function in the prelude,
or as a method in `Core.Destroy`. How we call this operation will frame how we
read and discuss code. We should take extra care in choosing the mechanism for
explicit object destruction since it is both an unsafe and an exceptionally rare
operation.

-   Using a function frames the scope as the active participant (for example
    `Core.SelfDestruct(x)`). `x` is passive, and has destruction rendered unto
    it by its environment. This communicates that the object exists at the behest
    of something external.
-   Using a method frames the object as the active participant (for example
    `x.SelfDestruct()`). `x` appears to initiate its own destruction, and the
    environment just facilitates that. This communicates that the object is in
    control of its own destiny.

We do not know what idiomatic Carbon will look like, so we should pick a style
to experiment with, and revise as necessary. The toolchain can't facilitate a
friend functions right now, so we need to implement `SelfDestruct` as method for
the time being.

### Trivial destruction

From the [existing destructor design]:

> **Future work:** Allow or require destructors to be declared as taking
> `partial Self` in order to prove no use of virtual methods.
>
> Types satisfy the [`TrivialDestructor`] facet type if:
>
> -   the class declaration does not define a destructor or the class defines the
>     destructor with an empty body `{ }`,
> -   all data members implement `TrivialDestructor`, and
> -   all base classes implement `TrivialDestructor`.
>
> For example, a [struct type] implements `TrivialDestructor` if all its members
> do.
>
> `TrivialDestructor` implies that their destructor does nothing, which may be
> used to generate optimized specializations.
>
> There is no provision for handling failure in a destructor. All operations that
> could potentially fail must be performed before the destructor is called.
> Unhandled failure during a destructor call will abort the program.

This is resolved in the [Trivial destruction section of the new design].

[`TrivialDestructor`]: https://github.com/carbon-language/carbon-lang/blob/a2890716ba7b73bb2bd337addceb3ac534558ee1/docs/design/generics/details.md#destructor-constraints
[struct type]: https://github.com/carbon-language/carbon-lang/blob/a2890716ba7b73bb2bd337addceb3ac534558ee1/docs/design/classes.md#struct-types
[trivial destruction section of the new design]: /docs/design/values.md#trivial-destruction

### Calling `SelfDestruct`

`SelfDestruct` is a recursive through `SubobjectDestroy.Op`. This recursion could
cause trivially destroyable objects to end up doing "busy work". To avoid this
problem, the toolchain does not call `SelfDestruct` for trivially destroyable
objects at all.

The check pass will not generate SemIR instructions for calling `SelfDestruct`
when it knows that calling `Destroy.Op` and `SubobjectDestroy.Op` are trivial.
SemIR will be emitted when this isn't known, such as when checking generics. The
missing information becomes known by the time we reach lowering, we are always
able to prune the calls during lowering.

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

### Two public interface model (previous design)

The previous design intended to replace the `destroy` method with four interfaces:

-   **`Destructor`**: a user-implementable interface to describe how a single
    object handles its subobjects at the end of its lifetime. Typically used to
    release resources acquired during its lifetime.
-   **`Destroy`:** a toolchain-implemented interface to describe how objects are
    destroyed.
-   **`TrivialDestroy`:** as described in the proposed design.
-   **`DynamicDestroy`:** as described in the proposed design.

This design was the product of the conversations surrounding Issues #6124, #6161, and #6464.
We felt that the design still had too much special-casing, and iterated upon the
design to reach the current proposal.

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

### Alternatives considered from [PR #1154]

The following alternatives were removed from [docs/design/classes.md]. They have
been copied here to ensure readers can easily find the alternatives without needing
to do extensive archaeology.

-   [Types implement destructor interface](/proposals/p001154-destructors.md#types-implement-destructor-interface)
-   [Prevent virtual function calls in destructors](/proposals/p001154-destructors.md#prevent-virtual-function-calls-in-destructors)
-   [Allow functions to act as destructors](/proposals/p001154-destructors.md#allow-functions-to-act-as-destructors)
-   [Allow private destructors](/proposals/p001154-destructors.md#allow-private-destructors)
-   [Allow multiple conditional destructors](/proposals/p001154-destructors.md#allow-multiple-conditional-destructors)
-   [Don't distinguish safe and unsafe delete operations](/proposals/p001154-destructors.md#dont-distinguish-safe-and-unsafe-delete-operations)
-   [Don't allow unsafe delete](/proposals/p001154-destructors.md#dont-allow-unsafe-delete)
-   [Allow final destructors](/proposals/p001154-destructors.md#allow-final-destructors)

## Future work

-   Allow or require destructors to be declared as taking `(var self: Self)`.
-   Identify an alternative name for `Destructor` (see [Issue #7508]).
-   Add an opt-out from implementing `Destroy`.
-   Describe why `Destructor.Op` is unsafe.
    -   Issue #6124 notes that certain calls to `Destroy.Op` are considered
        safe. These should be codified in the detailed description.
-   Unify with copy and move semantics.
-   Explore how types with virtual pointers can:
    -   opt out from `DynamicDestroy`
    -   manually implement `DynamicDestroy`

<!-- # Links -->
[destructors]: https://github.com/carbon-language/carbon-lang/blob/a2890716ba7b73bb2bd337addceb3ac534558ee1/docs/design/classes.md#destructors
[PR #1154]: https://github.com/carbon-language/carbon-lang/pull/1154
[Issue #6124]: https://github.com/carbon-language/carbon-lang/issues/6124
[Issue #6161]: https://github.com/carbon-language/carbon-lang/issues/6161
[Issue #6464]: https://github.com/carbon-language/carbon-lang/issues/6464
[Issue #7508]: https://github.com/carbon-language/carbon-lang/issues/7508

[existing destructor design]: https://github.com/carbon-language/carbon-lang/blob/a2890716ba7b73bb2bd337addceb3ac534558ee1/docs/design/classes.md#destructors
[object destruction]: /docs/design/values.md#object-destruction
[`TrivialDestructor`]: https://github.com/carbon-language/carbon-lang/blob/a2890716ba7b73bb2bd337addceb3ac534558ee1/docs/design/generics/details.md#destructor-constraints
[struct type]: https://github.com/carbon-language/carbon-lang/blob/a2890716ba7b73bb2bd337addceb3ac534558ee1/docs/design/classes.md#struct-types
[trivial destruction section of the new design]: /docs/design/values.md#trivial-destruction

[Code that is easy to read, understand, and write]: /docs/project/goals.md#code-that-is-easy-to-read-understand-and-write
[Software and language evolution]: /docs/project/goals.md#software-and-language-evolution
[Practical safety and testing mechanisms]: /docs/project/goals.md#practical-safety-and-testing-mechanisms
[Interoperability with and migration from existing C++ code]: /docs/project/goals.md#interoperability-with-and-migration-from-existing-c-code
[P001154 Destructors]: /proposals/p001154-destructors.md
