# Carbon: alternatives to coherence

<!--
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

This document explains the rationale for choosing to make
[implementation coherence](terminology.md#coherence)
[a goal for Carbon](goals.md#coherence), and the alternatives considered.

<!-- toc -->

## Table of contents

-   [Approach taken: coherence](#approach-taken-coherence)
-   [The "Hashtable Problem"](#the-hashtable-problem)
-   [Rejected alternative: no orphan rule](#rejected-alternative-no-orphan-rule)
-   [Rejected alternative: incoherence](#rejected-alternative-incoherence)
    -   [Incoherence means context sensitivity](#incoherence-means-context-sensitivity)
    -   [Rejected variation: dynamic implementation binding](#rejected-variation-dynamic-implementation-binding)
    -   [Rejected variation: manual conflict resolution](#rejected-variation-manual-conflict-resolution)

<!-- tocstop -->

## Approach taken: coherence

The main thing to understand is that coherence is a desirable property, but to
get that property we need an orphan rule, and that rule has a cost. It in
particular limits how much control users of a type have over how that type
implements interfaces. There are a few main problematic use cases to consider:

-   Selecting between multiple implementations of an interface for a type. For
    example selecting the implementation of the `Comparable` interface for a
    `Song` type to support "by title", "by artist", and "by album" orderings.
-   Implementing an interface for a type when there is no relationship between
    the libraries defining the interface and the type.
-   When the implementation of an interface for a type uses an associated type
    that can't be referenced from the file or files where the implementation is
    allowed to be defined.

These last two cases are highlighted as concerns in Rust in
[Rust RFC #1856: orphan rules are stricter than we would like](https://github.com/rust-lang/rfcs/issues/1856).

Since Carbon is bundling interface implementations into types, for the
convenience and expressiveness that provides, we satisfy those use cases by
giving the user control over the type of a value. This means having facilities
for defining new [compatible types](terminology.md#compatible-types) with
different interface implementations, and casting between those types as needed.

## The "Hashtable Problem"

The "Hashtable problem" is that the specific hash function used to compute the
hash of keys in a hashtable must be the same when adding an entry, when looking
it up, and other operations like resizing. So a hashtable type is dependent on
both the key type, and the key type's implementation of the `Hashable`
interface. If the key type can have more than one implementation of `Hashable`,
there needs to be some mechanism for choosing a single one to be used
consistently by the hashtable type, or the invariants of the type will be
violated.

Without the orphan rule to enforce coherence, we might have a situation like
this:

-   Package `Container` defines a `HashSet` type.

    ```
    package Container;
    struct HashSet(Key:! Hashable) { ... }
    ```

-   A `Song` type is defined in package `SongLib`.
-   Package `SongHashArtistAndTitle` defines an implementation of `Hashable` for
    `SongLib.Song`.

    ```
    package SongHashArtistAndTitle;
    import SongLib;
    impl SongLib.Song as Hashable {
      fn Hash[self: Self]() -> u64 { ... }
    }
    ```

-   Package `SongUtil` uses the `Hashable` implementation from
    `SongHashArtistAndTitle` to define a function `IsInHashSet`.

    ```
    package SongUtil;
    import SongLib;
    import SongHashArtistAndTitle;
    import Containers;

    fn IsInHashSet(
        s: SongLib.Song,
        h: Containers.HashSet(SongLib.Song)*) -> bool {
      return h->Contains(s);
    }
    ```

-   Package `SongHashAppleMusicURL` defines a different implementation of
    `Hashable` for `SongLib.Song` than package `SongHashArtistAndTitle`.

    ```
    package SongHashAppleMusicURL;
    import SongLib;
    impl SongLib.Song as Hashable {
      fn Hash[self: Self]() -> u64 { ... }
    }
    ```

-   Finally, package `Trouble` imports `SongHashAppleMusicURL`, creates a hash
    set, and then calls the `IsInHashSet` function from package `SongUtil`.

    ```
    package Trouble;
    import SongLib;
    import SongHashAppleMusicURL;
    import Containers;
    import SongUtil;

    fn SomethingWeirdHappens() {
      var unchained_melody: SongLib.Song = ...;
      var song_set: auto = Containers.HashSet(SongLib.Song).Create();
      song_set.Add(unchained_melody);
      // Either this is a compile error or does something unexpected.
      if (SongUtil.IsInHashSet(unchained_melody, &song_set)) {
        Print("This is expected, but doesn't happen.");
      } else {
        Print("This is what happens even though it is unexpected.");
      }
    }
    ```

The issue is that in package `Trouble`, the `song_set` is created in a context
where `SongLib.Song` has a `Hashable` implementation from
`SongHashAppleMusicURL`, and stores `unchained_melody` under that hash value.
When we go to look up the same song in `SongUtil.IsInHashSet`, it uses the hash
function from `SongHashArtistAndTitle` which returns a different hash value for
`unchained_melody`, and so reports the song is missing.

**Background:** [This post](https://gist.github.com/nikomatsakis/1421744)
discusses the hashtable problem in the context of Haskell, and
[this 2011 Rust followup](https://mail.mozilla.org/pipermail/rust-dev/2011-December/001036.html)
discusses how to detect problems at compile time.

## Rejected alternative: no orphan rule

In Swift an implementation of an interface, or a "protocol" as it is called in
Swift, can be provided in any module. As long as any module provides an
implementation, that implementation is
[used globally throughout the program](https://stackoverflow.com/questions/48762971/swift-protocol-conformance-by-extension-between-frameworks).

In Swift, since some protocol implementations can come from the runtime
environment provided by the operating system, multiple implementations for a
protocol can arise as a runtime warning. When this happens, Swift picks one
implementation arbitrarily.

In Carbon, we could make this a build time error. However, there would be
nothing preventing two independent libraries from providing conflicting
implementations. Furthermore, the error would only be diagnosed at link time.

## Rejected alternative: incoherence

### Incoherence means context sensitivity

The undesirable result of incoherence is that the interpretation of source code
changes based on imports. In particular, imagine there is a function call that
depends on a type implementing an interface, and two different implementations
are defined in two different libraries. A call to that function will be treated
differently depending on which of those two libraries are imported:

-   If neither is imported, it is an error.
-   If both are imported, it is ambiguous.
-   If only one is imported, you get totally different code executed depending
    on which it is.

Furthermore, this means that the behavior of a file can depend on an import even
if nothing from that package is referenced explicitly. In general, Carbon is
[avoiding this sort of context sensitivity](/docs/project/principles/low_context_sensitivity.md).
This context sensitivity would make moving code between files when refactoring
more difficult and less safe.

### Rejected variation: dynamic implementation binding

One possible approach would be to bind interface implementations to a value at
the point it was created. In [the example above](#the-hashtable-problem), the
implementation of the `Hashable` interface for `Song` would be fixed for the
`song_set` `HashSet` object based on which implementation was in scope in the
body of the `SomethingWeirdHappens` function.

This idea is discussed briefly in section 5.4 on separate compilation of WG21
proposal n1848 for implementing "Indiana" C++0x concepts
([1](https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.86.9526&rep=rep1&type=pdf),
and [2](https://wg21.link/n1848)).

This has some downsides:

-   It is harder to reason about. The behavior of `SongUtil.IsInHashSet` depends
    on the dynamic behavior of the program. At the time of the call, we may have
    no idea where the `HashSet` argument was created.
-   An object may be created far from a call that has a particular interface
    requirement, with no guarantee that the object was created with any
    implementation of the interface at all. This error would only be detected at
    runtime, not at type checking time.
-   It requires more data space at runtime because we need to store a pointer to
    the witness table representing the implementation with the object, since it
    varies instead of being known statically.
-   It is slower to execute from dynamic dispatch and the inability to inline.
-   In some cases it may not be feasible to use dynamic dispatch. For example,
    if an interface method returns an associated type, we might not know the
    calling convention of the function without knowing some details about the
    type.

As a result, this doesn't make sense as the default behavior for Carbon based on
its [goals](/docs/project/goals.md). That being said, this could be a feature
added later as opt-in behavior to either allow users to reduce code size or
support use cases that require dynamic dispatch.

### Rejected variation: manual conflict resolution

Carbon could alternatively provide some kind of manual disambiguation syntax to
resolve problems where they arise. The problems with this approach have been
[considered in the context of Rust](https://github.com/Ixrec/rust-orphan-rules#whats-wrong-with-incoherence).

A specific example of this approach is called
[scoped conformance](https://forums.swift.org/t/scoped-conformances/37159),
where the conflict resolution is based on limiting the visibility of
implementations to particular scopes. This hasn't been implemented, but it has
the drawbacks described above. Depending on the details of the implementation,
either:

-   there are incompatible values with types that have the same name, or
-   it is difficult to reason about the program's behavior because it behaves
    like
    [dynamic implementation binding](#rejected-variation-dynamic-implementation-binding)
    (though perhaps with a monomorphization cost instead of a runtime cost).
