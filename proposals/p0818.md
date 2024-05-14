# Constraints for generics (generics details 3)

<!--
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

[Pull request](https://github.com/carbon-language/carbon-lang/pull/818)

<!-- toc -->

## Table of contents

-   [Problem](#problem)
-   [Background](#background)
-   [Proposal](#proposal)
-   [Rationale based on Carbon's goals](#rationale-based-on-carbons-goals)
-   [Alternatives considered](#alternatives-considered)
    -   [Alternatives to `where` clauses modifying type-of-types](#alternatives-to-where-clauses-modifying-type-of-types)
    -   [Different keyword than `where`](#different-keyword-than-where)
    -   [Inline constraints instead of `.Self`](#inline-constraints-instead-of-self)
    -   [Self reference instead of `.Self`](#self-reference-instead-of-self)
    -   [Implied constraints across declarations](#implied-constraints-across-declarations)
    -   [No implied constraints](#no-implied-constraints)
    -   [Type inequality constraints](#type-inequality-constraints)
    -   [`where .Self is ...` could act like an external impl](#where-self-is--could-act-like-an-external-impl)
        -   [Other syntax for external impl](#other-syntax-for-external-impl)
    -   [Other syntax for must be legal type constraints](#other-syntax-for-must-be-legal-type-constraints)
    -   [Using only `==` instead of also `=`](#using-only--instead-of-also-)
    -   [Automatic type equality](#automatic-type-equality)
        -   [Restricted equality constraints](#restricted-equality-constraints)
        -   [No explicit restrictions](#no-explicit-restrictions)

<!-- tocstop -->

## Problem

We want Carbon to have a high quality generics feature that achieves the goals
set out in [#24](https://github.com/carbon-language/carbon-lang/pull/24). This
is too big to land in a single proposal. This proposal continues
[#553](https://github.com/carbon-language/carbon-lang/pull/553) defining the
details of:

-   adapters
-   associated types and other constants
-   parameterized interfaces

## Background

This is a follow on to these previous generics proposals:

-   [#24: Generics goals](https://github.com/carbon-language/carbon-lang/pull/24)
-   [#447: Generics terminology](https://github.com/carbon-language/carbon-lang/pull/447)
-   [#524: Generics overview](https://github.com/carbon-language/carbon-lang/pull/524)
-   [#553: Generics details part 1](https://github.com/carbon-language/carbon-lang/pull/553)
-   [#731: Generics details 2: adapters, associated types, parameterized interfaces](https://github.com/carbon-language/carbon-lang/pull/731)

Some of the content for this proposal was extracted from a larger
[Generics combined draft proposal](https://github.com/carbon-language/carbon-lang/pull/36).

## Proposal

This is a proposal to add two sections to
[this design document on generics details](/docs/design/generics/details.md).

## Rationale based on Carbon's goals

Much of this rationale for generics was captured in the
[Generics goals proposal](https://github.com/carbon-language/carbon-lang/pull/24).
The specific decisions for this constraint design were motivated by:

-   [Software and language evolution](/docs/project/goals.md#software-and-language-evolution)
    -   The manual generic type equality approach being proposed could be
        extended to automatically observe more types as being equal without
        breaking existing code.
    -   Adding or growing an `observe` statement has been designed to be
        non-breaking since they can't introduce name conflicts, only make more
        code legal.
    -   This constraint system is expressive and does not have artificial limits
        that would be needed to support automatic type equality.
-   [Code that is easy to read, understand, and write](/docs/project/goals.md#code-that-is-easy-to-read-understand-and-write)
    -   Attaching `where` clauses to type-of-types addresses more use cases than
        attaching them to declarations, allowing the design to be smaller, see
        [#780](https://github.com/carbon-language/carbon-lang/issues/780).
    -   The manual generic type equality approach is very simple and easy to
        understand how it works. It unfortunately is a bit more verbose, leading
        to more to read and write.
-   [Fast and scalable development](/docs/project/goals.md#fast-and-scalable-development)
    -   The manual generic type equality approach does much less work at compile
        time than the automatic approach. The extra work would only be done to
        produce an error message that includes the changes to the source needed
        to avoid repeating that work. In effect, the source code holds a cache
        of the facts needed to compile it.

## Alternatives considered

### Alternatives to `where` clauses modifying type-of-types

Issue
[#780: How to write constraints](https://github.com/carbon-language/carbon-lang/issues/780)
considered other forms that constraints could be written.

One alternative is to place `where` clauses on declarations instead of types.

```
fn F[W:! Type, V:! D & E](v:V) -> W
    where V.X == W, V.Y == W, V.Z = i32;
```

The constraints are written after the names of types are complete, and so
constraints are naturally written in terms of those names. This is a common
approach used by several languages including Swift and Rust, and so is likely
familiar to a significant fraction of our users. The downside of this approach
is it doesn't let you put constraints on types that never get a names, such as
in
["protocols as types" or "existential types" in Swift](https://docs.swift.org/swift-book/LanguageGuide/Protocols.html#ID275)
and "trait objects" in Rust
([1](https://doc.rust-lang.org/book/ch17-02-trait-objects.html),
[2](https://doc.rust-lang.org/reference/types/trait-object.html)). This is a
problem in practice, since those are cases where constraints are needed, since
for type safety any associated types need to be constrained to be a specific
concrete type. This motivated Rust to add another syntax just for specifying
specific concrete types for associated types in the argument passing style. It
is also motivation for Swift to consider a feature they call
["generalized existentials"](https://github.com/apple/swift/blob/main/docs/GenericsManifesto.md#generalized-existentials),
that is very similar to this proposal.

Another alternative is to generalize the argument passing approach Rust allows
to handle more kinds of constraints.

```
fn F[W:! Type, V:! D{.X = W} & E{.Y = W, .Z = i32}](v: V) -> W;
```

This would treat constraints on interface parameters and associated types in a
more uniform way, but doesn't handle the full breadth of constraints we would
like to express. We at some point believed that this framing of the constraints
made it harder to express undecidable type equality problems and easier to
enforce restrictions that would make the constraints decidable, but we
eventually discovered that this formulation had the essentially the same
difficulties.

We considered a variation of argument passing that we called "whole expression
constraint intersections."

```
fn F[W:! Type, V:! D & E & {.X = W, .Y = W, .Z = i32}](v: V) -> W;
```

This variation made it easy to set associated types from two interfaces with the
same name to the same value, but introduced concerns that it would stop `&` from
being associative and commutative. It was not chosen because it had the same
downsides as the argument passing approach.

### Different keyword than `where`

We considered other keywords for introducing constraints, such as:

-   `requires`, like
    [C++](https://en.cppreference.com/w/cpp/language/constraints)
-   `with`
-   `if`, like [D](http://rosettacode.org/wiki/Constrained_genericity#D)
-   `when`, like [F#](http://rosettacode.org/wiki/Constrained_genericity#F.23).

The most common choice across popular languages is `where`, including:

-   [C#](http://rosettacode.org/wiki/Constrained_genericity#C.23)
-   [Haskell](http://rosettacode.org/wiki/Constrained_genericity#Haskell)
-   [Rust](https://doc.rust-lang.org/rust-by-example/generics/where.html)
-   Swift
    ([1](https://docs.swift.org/swift-book/LanguageGuide/Generics.html#ID553),
    [2](https://docs.swift.org/swift-book/LanguageGuide/Generics.html#ID557))

While C++ is particularly important to our target userbase, Carbon's constraints
work more similarly to languages with generics like Rust and Swift.

### Inline constraints instead of `.Self`

We considered the possibility of using
[named constraints](/docs/design/generics/details.md#named-constraints) instead
of `.Self` for
[recursive constraints](/docs/design/generics/details.md#recursive-constraints).
This comes under consideration since `.Self` outside the named constraint is the
same as `Self` inside. However, you can't always avoid using `.Self`, since
naming the constraint before using it doesn't allow you to define this
`Container` interface:

```
interface Container {
  ...
  let SliceType:! Container where .SliceType == .Self;
  ...
}
```

The problem that arises is to avoid using `.Self`, we would need to define the
named constraint using `Container` before `Container` is defined:

```
constraint ContainerIsSlice {
  // Error: forward reference
  extends Container where Container.SliceType == Self;
}

interface Container {
  ...
  let SliceType:! ContainerIsSlice;
  ...
}
```

To work around this problem, we could allow the named constraint to be defined
inline in the `Container` definition:

```
interface Container {
  ...
  constraint ContainerIsSlice {
    extends Container where Container.SliceType == Self;
  }
  let SliceType:! ContainerIsSlice;
  ...
}
```

This alternative seems too cumbersome.

### Self reference instead of `.Self`

Another alternative instead of using `.Self` for
[recursive constraints](/docs/design/generics/details.md#recursive-constraints),
is to use the name of the type being declared inside the type declaration, as in
`T:! HasAbs(.MagnitudeType = T)`. This had two downsides:

-   Using the name of the type before it is finished being defined created
    questions about what that name meant. Using the reserved token sequence
    `.Self` instead makes it clearer that it obeys different rules than other
    identifier references. For example, that we don't allow members to be
    accessed.
-   It doesn't address use cases where we are defining a type-of-type that isn't
    associated with a type that has a name.

### Implied constraints across declarations

If constraints on an associated type could be implied by any declaration in the
interface, readers and the type checker would be required to scan the entire
interface definition and perform many type declaration lookups to understand the
constraints on that associated type. This is particularly important when those
constraints can be obscured in recursive references to the same interface:

```
interface I {
  let A:! Type;
  let B:! Type;
  let C:! Type;
  let D:! Type;
  let E:! Type;
  let SwapType:! I where .A == B and .B == A and .C == C
                     and .D == D and .E == E;
  let CycleType:! I where .A == B and .B == C and .C == D
                      and .D == E and .E == A;
  fn LookUp(hm: HashMap(D, E)*) -> E;
  fn Foo(x: Bar(A, B));
}
```

This applies equally to parameters:

```
interface I(A:! Type, B:! Type, C:! Type, D:! Type, E:! Type) {
  let SwapType:! I(B, A, C, D, E);
  let CycleType:! I(B, C, D, E, A);
  fn LookUp(hm: HashMap(D, E)*) -> E;
  fn Foo(x: Bar(A, B));
}
```

All of the type arguments to `I` must actually implement `Hashable`, since
[an adjacent swap and a cycle generate the full symmetry group on 5 elements](https://www.mathcounterexamples.net/generating-the-symmetric-group-with-a-transposition-and-a-maximal-length-cycle/)).
And additional restrictions on those types would depend on the definition of
`Bar`. For example, this definition

```
class Bar(A:! Type, B:! ComparableWith(A)) { ... }
```

would imply that all the type arguments to `I` would have to be comparable with
each other. This propagation problem means that allowing constraints to be
implied in this context is substantial, and potentially unbounded, work for the
compiler and human readers.

From this we conclude that we need the initial declaration part of an
`interface`, type definition, or associated type declaration to include a
complete description of all needed constraints.

It is possible that this reasoning will apply more generally, but we will wait
until we implement type checking to see what restrictions we actually need. For
example, we might need to restate "parameterized type implements interface"
constraints when it is on an associated type in a referenced interface, as in:

```
interface HasConstraint {
  let T:! Type where Vector(.Self) is Printable;
}

interface RestatesConstraint {
  // This works, since it restates the constraint on
  // `HasConstraint.T` that `U` is equal to.
  let U:! Type where Vector(.Self) is Printable;
  // This doesn't work:
  // ❌ let U:! Type;

  let V:! HasConstraint where .T == U;
}
```

### No implied constraints

Issue [#809](https://github.com/carbon-language/carbon-lang/issues/809)
considered whether Carbon would support implied constraints. The conclusion was
that implied constraints was an important ergonomic improvement. The framing as
a rewrite to a `where` restriction that did not affect the generic type
parameter's unqualified API seemed like something that could be explained to
users. Potential problems where a specialization of a generic type might allow
that type to be instantiated for types that don't satisfy the normal constraints
will be considered in the future along with the specialization feature.

We also considered the alternative where the user would need to explicitly opt
in to this behavior by adding `& auto` or `& implied_requirements` to their type
constraint, as in:

```
fn LookUp[KeyType:! Type & auto](hm: HashMap(KeyType, i32)*,
                                 k: KeyType) -> i32;

fn PrintValueOrDefault[KeyType:! Printable & auto,
                       ValueT:! Printable & HasDefault]
    (map: HashMap(KeyType, ValueT), key: KeyT);
```

The feeling was that implied constraints were natural enough that they didn't
need to be signaled by additional marking, with the accompanying ergonomic and
brevity loss.

### Type inequality constraints

You might want an inequality type constraint, for example, to control overload
resolution:

```
fn F[T:! Type](x: T) -> T { return x; }
fn F(x: Bool) -> String {
  if (x) return "True"; else return "False";
}

fn G[T:! Type where .Self != Bool](x: T) -> T {
  // We need T != Bool for this to type check.
  return F(x);
}
```

There are some problems with supporting this feature, however.
[Negative reasoning in general has been a source of difficulties in the implementation of Rust's type system](http://aturon.github.io/tech/2017/04/24/negative-chalk/).
This is a type of constraint that is more difficult to use since it would have
to be repeated by any generic caller, since nothing else can generically
establish two types are different.

Our manual type equality approach will not be able to detect situations where
the two sides of the inequality in fact have to be equal due to a sequence of
equality constraints. In those situations, no type will be ever be able to
satisfy the inequality constraint.

We may be able to overcome these difficulties, but we don't expect this feature
is required since neither Rust nor Swift support it.

### `where .Self is ...` could act like an external impl

We considered making `T:! A where .Self is B` mean something different than
`T:! A & B`. In particular, in the former case we considered whether `B` might
be considered to be implemented externally for `T`. The advantage of this
alternative is it gives a convenient way of not polluting the members of `T`
with the members of `B`, however it wasn't clear how common that use case would
be. Furthermore, it introduced an inconsistency with other associated types. For
example:

-   `T:! Container where .ElementType = i32` means that `T.ElementType` has type
    `i32`.
-   Following that, given `T:! Container where .ElementType is Printable` and
    `x: T`, we expect that `x.Front().Print()` to be legal.
-   To be consistent with the last point, it seems like given
    `T:! Container where .Self is Printable` and `x: T`, then `x.Print()` should
    be legal as well.

This matches Rust, where `T: Container` is considered a synonym for
`T where T is Container`.

[See the Discord discussion](https://discord.com/channels/655572317891461132/708431657849585705/895794334723637258).

#### Other syntax for external impl

In a
[follow up Discord discussion](https://discord.com/channels/655572317891461132/708431657849585705/902728293080530954),
we considered allowing developers to write `where .Foo as Bar is Type` to mean
"`.Foo` implements interface `Bar` externally." We would consider this feature
in the future once we saw a demonstrated need.

### Other syntax for must be legal type constraints

Instead of `where HashSet(.Self) is Type` to say `.Self` must satisfy the
constraints on being an argument to `HashSet`, we considered other syntaxes like
`where HashSet(.Self) legal` and `where HashSet(.Self)`. The current choice is
intended to be consistent with the syntax for "Parameterized type implements
interface" and how implied constraints ensure that parameters to types
automatically are given their needed constraints.

### Using only `==` instead of also `=`

Originally this proposal only used `==` for both setting an associated type to a
specific concrete type and saying it was equal to another type variable. The two
cases affected the type differently. In the specific concrete type case, the
original type-of-type for the associated type doesn't affect the API, you just
get the API of the type. When equating two type variables, though, the
unqualified member names are unaffected. The only change was some interfaces may
be implemented externally.

In
[this Discord discussion](https://discord.com/channels/655572317891461132/708431657849585705/902713789735116821),
we decided it might be clearer to use two different operators to reflect the
different effect. The compiler or a tool can easily correct cases where the
developer used the wrong one. However, using `=` comes with the downside that it
doesn't follow the pattern of other constraints of looking like a boolean
expression returning `true` when the constraint is satisfied. We would consider
a different pair of operators here to address this point, perhaps `~` for type
variables and `==` for concrete type case.

### Automatic type equality

Other languages, such as Swift and Rust, don't require `observe` declarations or
casting for the compiler to recognize two types as transitively equal. The
problem is that there is no way to know how many equality constraints need to be
considered before being able to conclude whether two type expressions are equal.
In fact, the equality relations form a semi-group, where in general deciding
whether two sequences are equivalent is undecidable, see
[Swift type checking is undecidable - Discussion - Swift Forums](https://forums.swift.org/t/swift-type-checking-is-undecidable/39024).

#### Restricted equality constraints

One possible approach to this problem is to apply limitations to the equality
constraints developers are allowed to express. The goal is to identify some
restrictions such that:

-   We have a terminating algorithm to decide if a set of constraints meets the
    restrictions.
-   For constraints that meet the restrictions we have a terminating algorithm
    for deciding which expressions are equal.
-   Expected use cases satisfy the restrictions.

The expected cases are things of these forms:

-   `X == Y`
-   `X == Y.Z`
-   `X == X.Y`
-   `X == Y.X`
-   and _some_ combinations and variations

For example, here are some interfaces that have been translated to Carbon syntax
from Swift standard library protocols:

```
interface IteratorProtocol {
  let Element:! Type;
}

interface Sequence {
  let Element:! Type;
  let Iterator:! IteratorProtocol
    where .Element == Element;
}

interface Collection {
  extends Sequence;
  let Index:! Type;
  let SubSequence:! Collection
    where .Element == Element
      and .SubSequence == SubSequence
      and .Index == Index;
  let Indices:! Collection
    where .Element == Index
      and .Index == Index
      and .SubSequence == Indices;
}
```

One approach we considered is
[regular equivalence classes](p0818/regular_equivalence_classes.md), however we
have not yet been able to figure out how to ensure the algorithm terminates.
This is an approach we would like to reconsider if we find solutions to this
problem once can share this problem more widely.

Other approaches we considered worked in simple cases but had requirements that
could not be validated, since for example they were equivalent to solving the
same word problem that makes transitive equality undecidable in general.

#### No explicit restrictions

If you allow the full undecidable set of `where` clauses, there are some
unpleasant options:

-   Possibly the compiler won't realize that two expressions are equal even
    though they should be.
-   Possibly the compiler will reject some interface declarations as "too
    complicated", due to "running for too many steps", based on some arbitrary
    threshold.

Either way, the compiler will have to perform a lot more work at compile time,
slowing down builds. There is also a danger that composition of things that
separately work or incremental evolution of working code could end up over a
complexity or search depth threshold. In effect, there are still restrictions on
what combinations of equality constraints are allowed, it is just that those
restrictions are implicit in the specifics of the algorithm chosen and execution
limits used.

One approach that has been suggested by
[Slava Pestov](https://forums.swift.org/u/Slava_Pestov) in the context of Swift,
is to
[formalize type equality as a term rewriting system](https://forums.swift.org/t/formalizing-swift-generics-as-a-term-rewriting-system/45175).
Then the equality constraints in an interface or function declaration can be
completed by running the Knuth-Bendix completion algorithm
([1](https://en.wikipedia.org/wiki/Knuth%E2%80%93Bendix_completion_algorithm#Description_of_the_algorithm_for_finitely_presented_monoids),
[2](https://academic.oup.com/comjnl/article/34/1/2/427931)) for some limited
number of steps. If the algorithm completes successfully, type expressions may
then be efficiently canonicalized. However the algorithm can fail, or fail to
terminate before hitting the threshold number of steps, in which case the source
code would have to be rejected. See
[this example implementation](https://gist.github.com/slavapestov/75dbec34f9eba5fb4a4a00b1ee520d0b).
