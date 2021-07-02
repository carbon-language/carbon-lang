# Carbon: alternatives to coherence

<!--
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

This document explains the rationale for choosing to make
[implementation coherence](terminology.md#coherence)
[a goal for Carbon](goals.md#coherence), and the alternatives considered.

## Approach taken: coherence

The main thing to understand is that coherence is a desirable property, but to
get that property we need an orphan rule, and that rule has a cost. It in
particular limits how much control users of a type have over how that type
implements interfaces. There are two main use cases to consider:

-   Selecting between multiple implementations of a `Comparable` interface for a
    `Song` type to support "by title", "by artist", and "by album" orderings.
-   Implementing an interface for a type when there is no relationship between
    the libraries defining the interface and the type.

Since Carbon is bundling interface implementations into types, for the
convenience and expressiveness that provides, we satisfy those use cases by
giving the user control over the type of a value. This means having facilities
for defining new [compatible types](terminology#compatible-types) with different
interface implementations, and casting between those types as needed.

## The "Hashtable Problem"

The "Hashtable" problem is that the specific hash function used to compute the
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
    struct HashSet(Hashable$ Key) { ... }
    ```

-   A `Song` type is defined in package `SongLib`.
-   Package `SongHashArtistAndTitle` defines an implementation of `Hashable` for
    `SongLib.Song`.

    ```
    package SongHashArtistAndTitle;
    import SongLib;
    impl SongLib.Song as Hashable {
      method (me: Self) Hash() -> UInt64 { ... }
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
        h: Containers.HashSet(SongLib.Song)*) -> Bool {
      return h->Contains(s);
    }
    ```

-   Package `SongHashAppleMusicURL` defines a different implementation of
    `Hashable` for `SongLib.Song` than package `SongHashArtistAndTitle`.

    ```
    package SongHashAppleMusicURL;
    import SongLib;
    impl SongLib.Song as Hashable {
      method (me: Self) Hash() -> UInt64 { ... }
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

FIXME: https://gist.github.com/nikomatsakis/1421744

## Problems with incohernce

FIXME

-   "Import what you use" is hard to measure: libraries `Y.T1` and `Z.T2` are
    important/used even though `Y` and `Z` are not mentioned outside the
    `import` statement.
-   The call `F(a)` has different interpretations depending on what libraries
    are imported:
    -   If neither is imported, it is an error.
    -   If both are imported, it is ambiguous.
    -   If only one is imported, you get totally different code executed
        depending on which it is.

## Rejected alternative: dynamic

FIXME

## Rejected alternative: manual conflict resolution

FIXME:
[Addressing "the hashtable problem" with type classes](https://mail.mozilla.org/pipermail/rust-dev/2011-December/001036.html)

## Rejected alternative: scoped conformance

FIXME:
[scoped conformances](https://forums.swift.org/t/scoped-conformances/37159).
