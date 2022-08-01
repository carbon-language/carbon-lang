# Destructors

<!--
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

[Pull request](https://github.com/carbon-language/carbon-lang/pull/1154)

<!-- toc -->

## Table of contents

-   [Problem](#problem)
-   [Background](#background)
-   [Proposal](#proposal)
-   [Rationale based on Carbon's goals](#rationale-based-on-carbons-goals)
-   [Alternatives considered](#alternatives-considered)
    -   [Types implement destructor interface](#types-implement-destructor-interface)
    -   [Prevent virtual function calls in destructors](#prevent-virtual-function-calls-in-destructors)
    -   [Allow functions to act as destructors](#allow-functions-to-act-as-destructors)
    -   [Allow private destructors](#allow-private-destructors)
    -   [Allow multiple conditional destructors](#allow-multiple-conditional-destructors)
        -   [Allow direct implementation of `TrivialDestructor`](#allow-direct-implementation-of-trivialdestructor)
    -   [Type-of-type naming](#type-of-type-naming)
    -   [Other approaches to extensible classes without vtables](#other-approaches-to-extensible-classes-without-vtables)
        -   [Don't distinguish safe and unsafe delete operations](#dont-distinguish-safe-and-unsafe-delete-operations)
        -   [Don't allow unsafe delete](#dont-allow-unsafe-delete)
        -   [Allow final destructors](#allow-final-destructors)

<!-- tocstop -->

## Problem

C++ code supports defining custom
[destructors](<https://en.wikipedia.org/wiki/Destructor_(computer_programming)>)
for user-defined types, and C++ developers will expect to use that tool. Carbon
should support destructors, but should make some changes to protect against
deleting a value with a derived type through a pointer to the base class when
the base class destructor is not virtual. We also need some mechanism to allow
generic code to identify types that are safe to destroy, or have trivial
destructors and don't need to be explicitly destroyed.

## Background

Destructors may only be customized for nominal classes, which were introduced in
proposal [#722](https://github.com/carbon-language/carbon-lang/pull/722).
Destructors interact with inheritance, which was introduced in proposal
[#777](https://github.com/carbon-language/carbon-lang/pull/777).

Destructors were discussed in open discussion on these dates:

-   [2021-07-12](https://docs.google.com/document/d/14vAcURDKeH6LZ_TQCMRGpNJrXSZCACQqDy29YH19XGo/edit#heading=h.40jlsrcgp8mr)
-   [2021-08-30](https://docs.google.com/document/d/1YhwNKLxQsWf8NPVaRm9PvgPmSM3PIK_KlD1gpNuUfwY/edit#heading=h.4dobu6v1cdam)
-   [2021-10-04](https://docs.google.com/document/d/1YhwNKLxQsWf8NPVaRm9PvgPmSM3PIK_KlD1gpNuUfwY/edit#heading=h.a0rffns9r10n)
    discussed not all types being destructible, and that some types are
    `TriviallyDestructible`
-   [2021-10-18](https://docs.google.com/document/d/1YhwNKLxQsWf8NPVaRm9PvgPmSM3PIK_KlD1gpNuUfwY/edit#heading=h.uz59mgk5ezch)
-   [2022-03-24](https://docs.google.com/document/d/1UelNaT_j61G8rYp6qQZ-biRddTuGcxJtqXxrVbjB9rA/edit#heading=h.cy8m79pgct1v)

As part of
[proposal 777: Inheritance](https://github.com/carbon-language/carbon-lang/pull/777),
we decided to support extensible classes, that is non-abstract base classes,
[including with non-virtual destructors](p0777.md#no-extensible-objects-with-non-virtual-destructors).
["Extensible classes" Google doc](https://docs.google.com/document/d/1gbQJN_IMJBnquOUUd2orbHLlAIqZ4pL0Vt7h34DkQjg/edit?resourcekey=0-0lkEvh0umUU206ASFlWc7A#)
from that time considered options for making deleting extensible classes safer.

## Proposal

This proposal adds two sections to the design:

-   ["Destructors"](/docs/design/classes.md#destructors) to the
    [classes design](/docs/design/classes.md), and
-   ["Destructor constraint"](/docs/design/generics/details.md#destructor-constraints)
    to the [detailed generics design](/docs/design/generics/details.md).

## Rationale based on Carbon's goals

Destructors advance these goals:

-   [Practical safety and testing mechanisms](/docs/project/goals.md#practical-safety-and-testing-mechanisms),
    since destructors are about making sure clean-up actions are performed by
    doing them automatically, which is helpful for avoiding safety-related bugs.
-   [Interoperability with and migration from existing C++ code](/docs/project/goals.md#interoperability-with-and-migration-from-existing-c-code),
    since destructors are relied upon in C++.

## Alternatives considered

### Types implement destructor interface

We considered letting developers implement a destructor interface, as is done in
[Rust](https://doc.rust-lang.org/std/ops/trait.Drop.html). This would be more
consistent with the other customization points for classes, but had some
downsides that we ultimately decided were too large:

-   Destructors are relatively common, and implementing an interface involves
    more boilerplate than writing a destructor in C++, so we wanted a concise
    syntax.
-   Destructors need access to the private members of a class, so generally
    would have to be part of the class.
-   We didn't have any use cases where an author of a type used as a type
    argument to a class should be able to override the destructor for the class,
    so users would have to mark destructors as `final` or risk surprises.
-   Abstract classes without virtual destructors may have code that should run
    when the type is destroyed, but don't implement any destructor
    type-of-types.
-   More generally, we wanted the compiler to enforce that the correct
    type-of-types were implemented.

This was considered in the open discussions on
[2021-08-30](https://docs.google.com/document/d/1YhwNKLxQsWf8NPVaRm9PvgPmSM3PIK_KlD1gpNuUfwY/edit#heading=h.4dobu6v1cdam)
and
[2021-10-18](https://docs.google.com/document/d/1YhwNKLxQsWf8NPVaRm9PvgPmSM3PIK_KlD1gpNuUfwY/edit#heading=h.uz59mgk5ezch).
We decided to go with the proposed approach in
[the open discussion on 2022-03-24](https://docs.google.com/document/d/1UelNaT_j61G8rYp6qQZ-biRddTuGcxJtqXxrVbjB9rA/edit#heading=h.cy8m79pgct1v).

### Prevent virtual function calls in destructors

We considered disallowing virtual calls to be made while executing the
destructor. This would avoid having to assign to the vtable pointer between each
level of the class hierarchy's destructor calls. This seemed like an avoidable
expense since you can't call a derived method implementation from a destructor
anyway.

We observed that this assignment, which happens in C++, is unsynchronized and
could cause race conditions when there was synchronization inside a destructor.
In Carbon, we expect to mitigate that somewhat by doing the assignment at the
end of the destructor instead of the beginning, which is safe since we won't
have multiple inheritance. This means that at least the most derived class'
destructor can acquire locks before any unsynchronized writes occur.

We didn't decide to include this approach in this proposal, since we felt like
it would likely be too burdensome to prevent calling member functions from a
destructor unless the developer did some extra work to mark all functions it
transitively calls. We would consider this as a potential future extension. As
noted in the design, this could be achieved by using the `partial` type.

This was considered in the open discussions on
[2021-08-30](https://docs.google.com/document/d/1YhwNKLxQsWf8NPVaRm9PvgPmSM3PIK_KlD1gpNuUfwY/edit#heading=h.4dobu6v1cdam)
and
[2021-10-18](https://docs.google.com/document/d/1YhwNKLxQsWf8NPVaRm9PvgPmSM3PIK_KlD1gpNuUfwY/edit#heading=h.uz59mgk5ezch).

### Allow functions to act as destructors

[We considered, on 2021-08-30](https://docs.google.com/document/d/1YhwNKLxQsWf8NPVaRm9PvgPmSM3PIK_KlD1gpNuUfwY/edit#heading=h.4dobu6v1cdam),
how we might support returning a value, for example a failure, from a
destructor. This would require that the destructor be called explicitly, rather
than implicitly as part of a `Allocator.Delete()` call or a variable leaving
scope. This led to a design where you could have "named destructors" that were
ordinary methods except that they consumed their `me` parameter, as is possible
in Rust. As a result, the `me` parameter would be marked with a keyword like
`destroy` or `consume`, like:

```
class MyClass {
  fn MyDestroyer[destroy me: Self](x: i32) -> bool;
}
```

We would need some way to make the ending of the lifetime visible in the caller,
perhaps:

```
var c: MyClass = ...;
// `~c` indicates lifetime of `c` is ending.
var b: bool = (~c).MyDestroyer(3);
```

There were questions about how this would interact with unformed state.

We decided to keep this idea in mind, but there is not a pressing need for this
feature now since this doesn't mirror a capability of C++. We might reconsider
as part of error handling strategy, or a desire to model resources that can be
consumed.

### Allow private destructors

C++ allows destructors to be marked `private`, which has some use cases such as
controlling deletes of a reference-counted value. This introduces
[context-sensitivity](/docs/project/principles/low_context_sensitivity.md). It
is particularly concerning if whether a type implements a constraint depends on
which function is the enclosing scope.

This was considered in the
[open discussion on 2021-08-30](https://docs.google.com/document/d/1YhwNKLxQsWf8NPVaRm9PvgPmSM3PIK_KlD1gpNuUfwY/edit#heading=h.4dobu6v1cdam).

### Allow multiple conditional destructors

A parameterized type might want to specialize a destructor based on properties
of the parameter. For example, it might do something more efficient if its
parameter is a `TriviallyDestructible` type. An approach consistent with
[conditional methods](/docs/design/generics/details.md#conditional-methods) and
the [prioritization rule](/docs/design/generics/details.md#prioritization-rule)
would be to write something like:

```
class Vector(T:! Movable) {
  // Prioritize using `match_first`
  match_first {
    // Express conditions by using a
    // specific `Self` type.
    destructor [U:! TriviallyDestructible,
                me: Vector(U)];
    // If first doesn't apply:
    destructor [me: Self];
  }
}
```

#### Allow direct implementation of `TrivialDestructor`

For the specific case where the developer wants to specialize the destructor to
be trivial under certain circumstances, we could allow the developer to directly
implement `TrivialDestructor`. That implementation would include the criteria
when the trivial destructor should be used. For example, we might express that
`Optional` has a trivial destructor when its argument does by writing:

```
class Optional(T:! Concrete) {
  var value: Storage(T);
  var holds_value: bool;

  // It's perfectly safe, I assure you
  impl [U:! TrivialDestructor] Optional(U) as TrivialDestructor {}

  destructor [me: Self] { if (me.holds_value) value.Destroy(); }
}
```

### Type-of-type naming

We considered other names for the type-of-types introduced in this proposal.

The name `Deletable` is inconsistent with
[the decision on #1058](https://github.com/carbon-language/carbon-lang/issues/1058),
so we considered alternatives like `Delete`, `CanDelete`, `DeleteInterface`,
`DeleteConstraint`, or `SafeToDelete`. We agreed that it should match whatever
term was used in the allocation interface, and finalizing that was out of scope
for this proposal. So we left it at `Deletable` for now, with the understanding
that this was only a placeholder and not the final name.

Instead of `Destructible`, which is inconsistent with
[the decision on #1058](https://github.com/carbon-language/carbon-lang/issues/1058),
we considered:

-   `DestructorConstraint`: a bit too long
-   `HasDestructor`: means something slightly different
-   `CanDestroy`: also inconsistent with
    [the decision on #1058](https://github.com/carbon-language/carbon-lang/issues/1058)

We ultimately decided that the best name is likely `Unsafe` followed by the name
we chose in place of `Deletable`, and would wait until that is decided.

Originally `TrivialDestructor` was spelled `TriviallyDestructible`, but the word
"destructor" was more aligned with the `destructor` keyword being used in
declarations. We also considered replacing "trivial" with something like "no-op"
or "empty" to emphasize that the destructor does literally nothing and can be
completely omitted, but decided to stay consistent with C++ for now.

Instead of `Concrete`, we also considered `Instantiable`, but this choice is
both more concise and more clearly the opposite of "abstract."

### Other approaches to extensible classes without vtables

There were a few alternatives we considered that were specifically concerned
about how to handle the unsafe case of deleting a pointer to a base class that
does not have a virtual destructor, but may be pointing to a derived value. We
[decided to allow this case](p0777.md#no-extensible-objects-with-non-virtual-destructors)
as part of
[proposal 777: Inheritance](https://github.com/carbon-language/carbon-lang/pull/777).
We decided to go with the proposed approach in
[the open discussion on 2022-03-24](https://docs.google.com/document/d/1UelNaT_j61G8rYp6qQZ-biRddTuGcxJtqXxrVbjB9rA/edit#heading=h.cy8m79pgct1v).

#### Don't distinguish safe and unsafe delete operations

We could, like C++, have a single operation that includes both the safe and
unsafe cases. We decided we would like to try the safer option to see if it is
viable or if it causes problems with interoperation with C++. We expect that
migrated C++ code can fall-back to using `UnsafeDelete` if using `Delete` is too
burdensome.

This option was considered in the
["Option: C++ DWIM model" section of "Extensible classes" doc](https://docs.google.com/document/d/1gbQJN_IMJBnquOUUd2orbHLlAIqZ4pL0Vt7h34DkQjg/edit?resourcekey=0-0lkEvh0umUU206ASFlWc7A#heading=h.d2ybn2szry11).

#### Don't allow unsafe delete

For even more safety, we considered preventing deletion of extensible classes
without virtual destructors entirely. However, extensible classes without
virtual destructors were not that rare in C++ code we examined. It did not seem
likely this would provide enough C++ interop, and there were concerns that C++
developers would generally want to have an escape hatch for performing
operations that they could prove to themselves were safe even if the compiler
could not.

This option was considered in the
["Option: Like C++ but forbid unsafe" section of "Extensible classes" doc](https://docs.google.com/document/d/1gbQJN_IMJBnquOUUd2orbHLlAIqZ4pL0Vt7h34DkQjg/edit?resourcekey=0-0lkEvh0umUU206ASFlWc7A#heading=h.718ogac3yb9l).

#### Allow final destructors

One option we
[considered early on](https://docs.google.com/document/d/14vAcURDKeH6LZ_TQCMRGpNJrXSZCACQqDy29YH19XGo/edit#heading=h.40jlsrcgp8mr)
for extensible classes with non-virtual destructors, was to require the same
thing we do from non-virtual methods. That is, require that the implementation
in the base class is appropriate for derived classes. We ultimately decided on a
different approach, but we could still provide that as an option. You would
declare the destructor as `final`, and that would impose the necessary
restrictions on any derived class:

-   No custom destructor code.
-   Either no added data members, or, if we are willing to support unsized
    deletes, no added data members with non-trivial destructors.

Base classes with `final` destructors would be `Deletable`, like base classes
with `virtual` destructors. This would be a safe option without the overhead of
including a vtable in your object, but fairly restrictive in its applicability.

[We decided](https://discord.com/channels/655572317891461132/708431657849585705/958595054707023964)
is viable but not pressing since this isn't a feature present in C++, and we
would wait and see if this would fill a common need.
