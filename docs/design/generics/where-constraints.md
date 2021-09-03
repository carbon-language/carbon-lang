<!--
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

FIXME: Replace "structural interface" with "named constraint"

## Where constraints

So far, we have restricted a generic type parameter by saying it has to
implement an interface or a set of interfaces. There are a variety of other
constraints we would like to be able to express, such as applying restrictions
to their associated types and associated constants. This is done using the
`where` operator that adds constraints to a type-of-type.

The where operator can be applied to a type-of-type in a declaration context:

```
// Constraints on function parameters:
fn F[V:! D where ...](v: V) { ... }

// Constraints on a class parameter:
class S(T:! B where ...) { ... }

// Constraints on an interface parameter:
interface A(T:! B where ...) {
  // Constraints on an associated type or constant:
  let U:! C where ...;
  // Constraints on a method:
  fn G[me: Self, V:! D where ...](v: V);
}
```

We also allow you to name constraints using a `where` operator in a `let` or
`constraint` definition. The expressions that can follow the `where` keyword are
described in the ["constraint use cases"](#constraint-use-cases) section, but
generally look like boolean expressions that should evaluate to `true`.

The result of applying a `where` operator to a type-of-type is another
type-of-type. Note that this expands the kinds of requirements that
type-of-types can have from just interface requirements to also include the
various kinds of constraints discussed later in this section.

**Comparison to other languages:** Both Swift and Rust use `where` clauses on
declarations instead of in the expression syntax. These happen after the type
that is being constrained has been given a name and use that name to express the
constraint.

Rust also supports
[directly passing in the values for associated types](https://rust-lang.github.io/rfcs/0195-associated-items.html#constraining-associated-types)
when using a trait as a constraint. This is helpful when specifying concrete
types for all associated types in a trait in order to
[make it object safe so it can be used to define a trait object type](https://rust-lang.github.io/rfcs/0195-associated-items.html#trait-objects).

Rust is adding trait aliases
([RFC](https://github.com/rust-lang/rfcs/blob/master/text/1733-trait-alias.md),
[tracking issue](https://github.com/rust-lang/rust/issues/41517)) to support
naming some classes of constraints.

### Constraint use cases

#### Set an associated constant to a specific value

We might need to write a function that only works with a specific value of an
[associated constant](#associated-constants) `N`.

```
fn PrintPoint2D[PointT:! NSpacePoint where .N == 2](p: PointT) {
  Print(p.Get(0), ", ", p.Get(1));
}
```

Similarly in an interface definition:

```
interface {
  let PointT:! NSpacePoint where .N == 2;
}
```

To name such a constraint:

```
let Point2DInterface:! auto = NSpacePoint where .N == 2;
constraint Point2DInterface {
  extends NSpacePoint where .N == 2;
}
```

#### Set an associated type to a specific value

For example, we could make a the `ElementType` of an `Iterator` interface equal
to the `ElementType` of a `Container` interface as follows:

```
interface Iterator {
  let ElementType:! Type;
  ...
}
interface Container {
  let ElementType:! Type;
  let IteratorType:! Iterator where .ElementType == ElementType;
  ...
}
```

Functions accepting a generic type might also want to constrain an associated
type. For example, we might want to have a function only accept stacks
containing integers:

```
fn SumIntStack[T:! Stack where .ElementType == i32](s: T*) -> i32 {
  var sum: i32 = 0;
  while (!s->IsEmpty()) {
    sum += s->Pop();
  }
  return sum;
}
```

To name these sorts of constraints, we could use `let` statements or
`constraint` definitions.

```
let IntStack:! auto = Stack where .ElementType == i32;
constraint IntStack {
  extends Stack where .ElementType == i32;
}
```

#### Range constraints on associated constants

We can express range and inequality constraints on associated constants. This
exampls shows constraining the `N` member of `NSpacePoint` from
[the "associated constants" section](#associated-constants):

```
fn PrintPoint2Or3
    [PointT:! NSpacePoint where 2 <= .N, .N <= 3]
    (p: PointT);
```

Name this kind of constraint like so:

```
let HyperPoint:! auto = NSpacePoint where .N > 3;
constraint HyperPoint {
  extends NSpacePoint where .N > 3;
}
```

**Concern:** How should we express range constraints on generic integer
parameters?

```
fn TakesAtLeastAPair[N:! u32 where ___ >= 2](x: NTuple(N, i32));
fn TakesAtLeastAPair[N:! u32](x: NTuple(N, i32) where N >= 2);
```

Or on associated constants in an interface definition?

```
interface HyperPointInterface {
  let N:! u32 where ___ > 3;
  fn Get[addr me: Self*](i: i32) -> f64;
}
```

#### Type bound for associated type

Type restrictions in Carbon are represented by a type-of-type. So to express a
bound on an associated type, we need some way to

The type bounds that we might want to express on an associated type

-

FIXME: Already have a way to express this on types, but this section is about
doing the same for associated types.

##### Type bounds on associated types in declarations

You might constrain the element type to satisfy an interface (`Comparable` in
this example) without saying exactly what type it is:

```
interface Container {
  let ElementType:! Type;
  ...
}

fn SortContainer
    [ContainerType:! Container where .ElementType is Comparable]
    (container_to_sort: ContainerType*);
```

**Open question:** How do you spell that? This proposal provisionally uses `is`,
which matches Swift, but maybe we should have another operator that more clearly
returns a boolean like `has_type`?

**Note:** `Container` defines `ElementType` as having type `Type`, but
`ContainerType.ElementType` has type `Comparable`. This is because
`ContainerType` has type `Container where .ElementType is Comparable`, not
`Container`. This means we need to be a bit careful when talking about the type
of `ContainerType` when there is a `where` clause modifying it.

##### Type bounds on associated types in interfaces

Given these definitions (omitting `ElementType` for brevity):

```
interface IteratorInterface { ... }
interface ContainerInterface {
  let IteratorType:! IteratorInterface;
  ...
}
interface RandomAccessIterator {
  extends IteratorInterface;
  ...
}
```

We can then define a function that only accepts types that implement
`ContainerInterface` where its `IteratorType` associated type implements
`RandomAccessIterator`:

```
fn F[ContainerType:! ContainerInterface
     where .IteratorType is RandomAccessIterator]
    (c: ContainerType);
```

We would like to be able to define a `RandomAccessContainer` to be a
type-of-type whose types satisfy `ContainerInterface` with an `IteratorType`
satisfying `RandomAccessIterator`.

```
let RandomAccessContainer:! auto =
    ContainerInterface where .IteratorType is RandomAccessIterator;
// or
constraint RandomAccessContainer {
  extends ContainerInterface
      where .IteratorType is RandomAccessIterator;
}

// With the above definition:
fn F[ContainerType:! RandomAccessContainer](c: ContainerType);
// is equivalent to:
fn F[ContainerType:! ContainerInterface
     where .IteratorType is RandomAccessIterator]
    (c: ContainerType);
```

#### Same type constraints

Given an interface with two associated types

```
interface PairInterface {
  let Left:! Type;
  let Right:! Type;
}
```

we can constrain them to be equal in a function declaration:

```
fn F[MatchedPairType:! PairInterface where .Left == .Right]
    (x: MatchedPairType*);
```

or in an interface definition:

```
interface HasEqualPair {
  let P:! PairInterface where .Left == .Right;
}
```

This kind of constraint can be named:

```
let EqualPair:! auto =
    PairInterface where .Left == .Right;
constraint EqualPair {
  extends PairInterface where .Left == .Right;
}
```

Another example of same type constraints is when associated types of two
different interfaces are constrained to be equal:

```
fn Map[CT:! Container,
       FT:! Function where .InputType == CT.ElementType]
      (c: CT, f: FT) -> Vector(FT.OutputType);
```

In either situation, this can affect the type-of-type on the associated type
being constrained. For example, if `SortedContainer.ElementType` is
`Comparable`, then in this declaration

```
fn Contains
    [SC:! SortedContainer,
     CT:! Container where .ElementType == SC.ElementType]
    (haystack: SC, needles: CT) -> Bool;
```

the `where` constraint will cause `CT.ElementType` to be `Comparable` as well.

When the two generic type parameters are swapped:

```
fn Contains
    [CT:! Container,
     SC:! SortedContainer where .ElementType == CT.ElementType]
    (haystack: SC, needles: CT) -> Bool;
```

then `CT.ElementType` will still end up implementing `Comparable`, but it will
act like an [external implementation](#external-impl). That is, the type `CT`
won't have an unqualified `Compare` method, but can still be cast to
`Comparable`.

#### Combining constraints

Constraints can be combined by separating constraint clauses with a comma `,`.
This example expresses a constraint that two associated types are equal and
satisfy an interface:

```
fn EqualContainers
    [CT1:! Container
     CT2:! Container
         where .ElementType is HasEquality,
               .ElementType == CT1.ElementType]
    (c1: CT1*, c2: CT2*) -> Bool;
```

#### Recursive constraints

Just like we use `Self` to refer to the type implementing an interface, we
sometimes need to constrain a type to equal one of its associated types. In this
first example, we want to represent the function `Abs` which will return `Self`
for some but not all types, so we use an associated type `MagnitudeType` to
encode the return type:

```
interface HasAbs {
  extends Numeric;
  let MagnitudeType:! Numeric;
  fn Abs[me: Self]() -> MagnitudeType;
}
```

For types representing subsets of the real numbers, such as `i32` or `f32`, the
`MagnitudeType` will match `Self`. For types representing complex numbers, the
types will be different. For example, the `Abs()` applied to a `Complex64` value
would produce a `f32` result. The goal is to write a constraint to restrict to
the first case.

In a second example, when you take the slice of a type implementing `Container`
you get a type implementing `Container` which may or may not be the same type as
the original container type. However, taking the slice of a slice always gives
you the same type, and some functions want to only operate on containers whose
slice type is the same.

To solve this problem, we think of `Self` as an actual associated type member of
every interface. We can then address it using `.Self` like any other associated
type.

```
fn Relu[T:! HasAbs where .MagnitudeType == .Self](x: T) {
  // T.MagnitudeType == T so the following is allowed:
  return (x.Abs() + x) / 2;
}
fn UseContainer[T:! Container where .SliceType == .Self](c: T) -> Bool {
  // T.SliceType == T so `c` and `c.Slice(...)` can be compared:
  return c == c.Slice(...);
}
```

Notice that in an interface definition, `Self` refers to the type implementing
this interface while `.Self` refers to the associated type currently being
defined.

```
interface Container {
  let ElementType:! Type;

  let SliceType:! Container
      where .ElementType == ElementType,
            .SliceType == .Self;

  fn GetSlice[addr me: Self*]
      (start: IteratorType, end: IteratorType) -> SliceType;
}
```

These constraints can be named:

```
let RealAbs:! auto = HasAbs where .MagnitudeType == .Self;
constraint RealAbs {
  extends HasAbs where .MagnitudeType == Self;
}
let ContainerIsSlice:! auto =
    Container where .SliceType == .Self;
constraint ContainerIsSlice {
  extends Container where .SliceType == Self;
}
```

Note that using the `constraint` approach we can name these constraints without
using `.Self`.

#### Parameterized type implements interface

There are times when a function will pass a generic type parameter of the
function as an argument to a parameterized type, and the function needs the
result to implement a specific interface.

```
// Some parametized type.
class Vector(T:! Type) { ... }

// Parameterized type implements interface only for some arguments.
external impl Vector(String) as Printable { ... }

// Constraint: `T` such that `Vector(T)` implements `Printable`
fn PrintThree
    [T:! Type where Vector(.Self) is Printable]
    (a: T, b: T, c: T) {
  var v: Vector(T) = (a, b, c);
  Print(v);
}
```

#### Type inequality

**Open question:** It isn't clear if we should support this in Carbon, since it
isn't in either Swift or Rust.

You might need an inequality type constraint, for example, to control overload
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

Another use case for inequality type constraints would be to say something like
"define `ComparableTo(T1)` for `T2` if `ComparableTo(T2)` is defined for `T1`
and `T1 != T2`".

### Implicit constraints

Imagine we have a generic function that accepts an arbitrary `HashMap`:

```
class HashMap(KeyType:! Hashable, ValueType:! Type);

fn LookUp[KeyType:! Type](hm: HashMap(KeyType, Int)*,
                          k: KeyType) -> Int;

fn PrintValueOrDefault[KeyType:! Printable,
                       ValueT:! Printable & HasDefault]
    (map: HashMap(KeyType, ValueT), key: KeyT);
```

The `KeyType` in these declarations does not satisfy the requirements of
`HashMap`, which requires the type implement `Hashable` and other interfaces:

```
class HashMap(
    KeyType:! Hashable & Sized & EqualityComparable & Movable,
    ...) { ... }
```

In this case, `KeyType` gets `Hashable` and so on as _implicit constraints_.

FIXME: This is being decided in question-for-leads issue
[#809: Implicit/inferred generic type constraints](https://github.com/carbon-language/carbon-lang/issues/809).

**Open question:** Should we allow those function declarations, and implicitly
add needed constraints to `KeyType` implied by being used as an argument to a
parameter with those constraints? Or should we require `KeyType` to name all
needed constraints as part of its declarations?

In this specific case, Swift will accept the definition and infer the needed
constraints on the generic type parameter
([1](https://www.swiftbysundell.com/tips/inferred-generic-type-constraints/),
[2](https://github.com/apple/swift/blob/main/docs/Generics.rst#constraint-inference)).
This is both more concise for the author of the code and follows the
["don't repeat yourself" principle](https://en.wikipedia.org/wiki/Don%27t_repeat_yourself).
This redundancy is undesirable since it means if the needed constraints for
`HashMap` are changed, then the code has to be updated in more locations.
Further it can add noise that obscures relevant information. In practice, any
user of these functions will have to pass in a valid `HashMap` instance, and so
will have already satisfied these constraints.

**Note:** These implied constraints should affect the _requirements_ of a
generic type parameter, but not its _unqualified names_. This way you can always
look at the declaration to see how name resolution works, without having to look
up the definitions of everything it is used as an argument to.

FIXME: Not for interfaces. Maybe: The initial declaration part of an
`interface`, type definition, or associated type declaration should include
complete description of all needed constraints.

FIXME: Resolve with #809

**Alternative:** As an alternative, we could make it so the user would need to
explicitly opt in to this behavior by adding `& auto` or
`& implicit_requirements` to their type constraint, as in:

```
fn LookUp[KeyType:! Type & auto](hm: HashMap(KeyType, Int)*,
                                 k: KeyType) -> Int;

fn PrintValueOrDefault[KeyType:! Printable & auto,
                       ValueT:! Printable & HasDefault]
    (map: HashMap(KeyType, ValueT), key: KeyT);
```

### Restrictions

With the full expressive power of `where` clauses, determining whether two type
expressions are equal is in general undecidable, as
[has been shown in Swift](https://forums.swift.org/t/swift-type-checking-is-undecidable/39024).
In practice this means that a compiler would reject some legal programs based on
heuristics simply to avoid running for an unbounded length of time.

For Carbon, we instead introduce restrictions on `where` clauses so that type
questions can be decided by an efficient algorithm. This is an important part of
achieving
[Carbon's goal of fast and scalable development](/docs/project/goals.md#fast-and-scalable-development).
The intent is that these restrictions:

-   are understandable to users,
-   allow most use cases that arise in practice, and
-   when users hit the restrictions there is a clear path of action for
    resolving the issue.

The restrictions arise from the the algorithm used to answer type questions. It
works by first rewriting `where` operations to put a declaration, like a
function signature or interface definition, into a normalized form. This
normalized form can then be lazily evaluated to answer queries. Queries take a
dotted name and return an archetype that has a canonical type name and a
type-of-type.

The normalized form for a function declaration includes generic type parameters
and any associated types mentioned in a `where` constraint.

```
fn Sort[C:! Container where .Elt is Comparable](c: C*)
```

normalizes to:

```
{
  C.Elt:! Comparable;
  C:! Container{.Elt=C.Elt};
}
```

The normalized form for an interface includes the associated types as well as
dotted names mentioned in `where` constraints. It includes interface's name to
support recursive references. Given these interface definitions,

```
interface P {
  let T:! F;
}

interface Q {
  let Y:! H;
}

interface R {
  let X:! Q;
}

interface S {
  let A:! P;
  let B:! R where .X.Y == A.T;
}
```

the interface `S` normalizes to:

```
S {
  A.T, B.X.Y:! F & H;
  B.X:! Q{.Y = A.T};
  A:! P{.T = A.T as F};
  B:! R{.X = B.X};
}
```

There are a couple of ways this normalization can fail. The first is by
introducing a cycle:

```
interface Graph {
  let Vertex:! V;
  let Edge:! E where Vertex.Edge == .Self,
                     .Vertex == Vertex;
}
```

normalizes to:

```
Graph {
  Vertex, Edge.Vertex:! V{.Edge = Edge};
  Edge, Vertex.Edge:! E{.Vertex = Vertex};
}
```

This can happen even without a `.Self ==` constraint:

```
interface HasCycle {
  let A:! P;
  let B:! Q where .X.Y == A.T, .X == A.T.U;
}
```

which normalizes to:

```
HasCycle {
  A.T, B.X.Y:! ...{.U = A.T.U};
  A.T.U, B.X:! ...{.Y = A.T};
  A:! P{.T = A.T};
  B:! Q{.X = A.T.U};
}
```

The other failure is when setting two terms equal, we need to combine the
constraints of both terms to get a type that both satifsy. In many cases, this
combination is straightforward.

-   `combine(X, X) = X`.
-   If interface `BidirectionalIter` extends `ForwardIter`, then
    `combine(is ForwardIter, is BidirectionalIter) = is BidirectionalIter`.
-   More generally, if `P` implies `Q` then `combine(P, Q) = P`. For example,
    `combine(is A, is A & B) = is A & B`.
-   For two different interfaces `A` and `B`, `combine(is A, is B) = is A & B`.
-   For an interface `Printable` and a type `String` implementing that
    interface, `combine(is Printable, == String) = String`. If the type doesn't
    implement the interface, the compiler should generate a type error.
-   For two different associated types `X` and `Y` of the same interface `A`,
    `combine(is A{.X = T}, is A{.Y = U}) = is A(.X = T, .Y = U}`.

The interesting case is `intersect(is A{.X = T}, is A{.X = U})` when `T` and `U`
are different. We could in principle recursively add a rewrite setting them
equal, but to guarantee that the algorithm terminates, we instead give an error.
The insight is that in this case, the error is reasonably clear. In cases that
arise in practice, the error should be enough for the user to fix the issue.

The last restriction comes from the query algorithm. It imposes a condition on
recursive references to the same interface. The rewrite to normalized form tries
to avoid triggering this condition, so the only known examples hitting this
restriction require mutually recursive interfaces. That would require forward
declaration of interfaces, which is not permitted at this time, but we may add
in the future.

The query algorithm allows us to determine if two dotted names represent equal
types by querying both and comparing canonical type names. It establishes what
type should be used for a dotted name after constraints are taken into
consideration. For example, in the `Sort` function declaration above, this would
determine that `C.Elt`, `C.Iter.Elt`, `C.Slice.Elt`, and so on are all equal and
implement `Comparable`. This is despite `Elt` as being declared `let Elt:! Type`
in the definition of `Container`.

A more complete description of the normalization rewrite and querying algorithms
can be found in [this appendix](appendix-archetype-algorithm.md).

## Other constraints as type-of-types

There are some constraints that we will naturally represent as named
type-of-types. These can either be used directly to constrain a generic type
parameter, or in a `where` clause to constrain an associated type.

### Is a derived class

Given a type `T`, `Extends(T)` is a type-of-type whose values are types that are
derived from `T`. That is, `Extends(T)` is the set of all types `U` that are
subtypes of `T`.

```
fn F[T:! Extends(BaseType)](p: T*);
fn UpCast[U:! Type, T:! Extends(U)](p: T*, _:! singleton_type_of(U)) -> U*;
fn DownCast[T:! Type](p: T*, U:! Extends(T)) -> U*;
```

**Open question:** Alternatively, we could overload the `is` operator for this,

```
fn F[T:! Type where .Self is BaseType](p: T*);
fn UpCast[T:! Type](p: T*, U:! Type where T extends .Self) -> U*;
fn DownCast[T:! Type](p: T*, U:! Type where .Self extends T) -> U*;
```

or define a new `extends` operator:

```
fn F[T:! Type where .Self extends BaseType](p: T*);
fn UpCast[T:! Type](p: T*, U:! Type where T extends .Self) -> U*;
fn DownCast[T:! Type](p: T*, U:! Type where .Self extends T) -> U*;
```

**Comparison to other languages:** In Swift, you can
[add a required superclass to a type bound using `&`](https://docs.swift.org/swift-book/LanguageGuide/Protocols.html#ID282).

### Type compatible with another type

Given a type `U`, define the type-of-type `CompatibleWith(U)` as follows:

> `CompatibleWith(U)` is a type whose values are types `T` such that `T` and `U`
> are [compatible](terminology.md#compatible-types). That is values of types `T`
> and `U` can be cast back and forth without any change in representation (for
> example `T` is an [adapter](#adapting-types) for `U`).

To support this, we extend the requirements that type-of-types are allowed to
have to include a "data representation requirement" option.

`CompatibleWith` determines an equivalence relationship between types.
Specifically, given two types `T1` and `T2`, they are equivalent if
`T1 is CompatibleWith(T2)`. That is, if `T1` has the type `CompatibleWith(T2)`.

**Note:** Just like interface parameters, we require the user to supply `U`,
they may not be deduced. Specifically, this code would be illegal:

```
fn Illegal[U:! Type, T:! CompatibleWith(U)](x: T*) ...
```

In general there would be multiple choices for `U` given a specific `T` here,
and no good way of picking one. However, similar code is allowed if there is
another way of determining `U`:

```
fn Allowed[U:! Type, T:! CompatibleWith(U)](x: U*, y: T*) ...
```

#### Same implementation restriction

In some cases, we need to restrict to types that implement certain interfaces
the same way as the type `U`.

> The values of type `CompatibleWith(U, TT)` are types satisfying
> `CompatibleWith(U)` that have the same implementation of `TT` as `U`.

For example, if we have a type `HashSet(T)`:

```
class HashSet(T:! Hashable) { ... }
```

Then `HashSet(T)` may be cast to `HashSet(U)` if
`T is CompatibleWith(U, Hashable)`. The one-parameter interpretation of
`CompatibleWith(U)` is recovered by letting the default for the second `TT`
parameter be `Type`.

#### Example: Multiple implementations of the same interface

This allows us to represent functions that accept multiple implementations of
the same interface for a type.

```
enum CompareResult { Less, Equal, Greater }
interface Comparable {
  fn Compare[me: Self](that: Self) -> CompareResult;
}
fn CombinedLess[T:! Type](a: T, b: T,
                          U:! CompatibleWith(T) & Comparable,
                          V:! CompatibleWith(T) & Comparable) -> Bool {
  match ((a as U).Compare(b)) {
    case CompareResult.Less => { return True; }
    case CompareResult.Greater => { return False; }
    case CompareResult.Equal => {
      return (a as V).Compare(b) == CompareResult.Less;
    }
  }
}
```

Used as:

```
class Song { ... }
adapter SongByArtist for Song { impl as Comparable { ... } }
adapter SongByTitle for Song { impl as Comparable { ... } }
assert(CombinedLess(Song(...), Song(...), SongByArtist, SongByTitle) == True);
```

We might generalize this to a list of implementations:

```
fn CombinedCompare[T:! Type]
    (a: T, b: T, CompareList:! List(CompatibleWith(T) & Comparable))
    -> CompareResult {
  for (U: auto) in CompareList {
    var result: CompareResult = (a as U).Compare(b);
    if (result != CompareResult.Equal) {
      return result;
    }
  }
  return CompareResult.Equal;
}

assert(CombinedCompare(Song(...), Song(...), (SongByArtist, SongByTitle)) ==
       CompareResult.Less);
```

#### Example: Creating an impl out of other impls

And then to package this functionality as an implementation of `Comparable`, we
combine `CompatibleWith` with [type adaptation](#adapting-types):

```
adapter ThenCompare(T:! Type,
                    CompareList:! List(CompatibleWith(T) & Comparable)) for T {
  impl as Comparable {
    fn Compare[me: Self](that: Self) -> CompareResult {
      for (U: auto) in CompareList {
        var result: CompareResult = (this as U).Compare(that);
        if (result != CompareResult.Equal) {
          return result;
        }
      }
      return CompareResult.Equal;
    }
  }
}

let SongByArtistThenTitle = ThenCompare(Song, (SongByArtist, SongByTitle));
var song: Song = ...;
var song2: SongByArtistThenTitle = Song(...) as SongByArtistThenTitle;
assert((song as SongByArtistThenTitle).Compare(song2) == CaompareResult.Less);
```

### Type facet of another type

Similar to `CompatibleWith(T)`, `FacetOf(T)` introduces an equivalence
relationship between types. `T1 is FacetOf(T2)` if both `T1` and `T2` are facets
of the same type.

### Sized types and type-of-types

What is the size of a type?

-   It could be fully known and fixed at compile time -- this is true of
    primitive types (`i32`, `f64`, etc.) most other concrete types (for example
    most [classes](/docs/design/classes.md)).
-   It could be known generically. This means that it will be known at codegen
    time, but not at type-checking time.
-   It could be dynamic. For example, it could be a FIXME
    [dynamic type](#dynamic-pointer-type) such as `Dynamic(TT)`, a FIXME
    [variable-sized type](https://github.com/josh11b/carbon-lang/blob/structs/docs/design/structs.md#control-over-allocation),
    or you could dereference a pointer to a base class that could actually point
    to a FIXME
    [derived class](https://github.com/josh11b/carbon-lang/blob/structs/docs/design/structs.md#question-extension--inheritance).
-   It could be unknown which category the type is in. In practice this will be
    essentially equivalent to having dynamic size.

I'm going to call a type "sized" if it is in the first two categories, and
"unsized" otherwise. (Note: something with size 0 is still considered "sized".)
The type-of-type `Sized` is defined as follows:

> `Sized` is a type whose values are types `T` that are "sized" -- that is the
> size of `T` is known, though possibly only generically.

Knowing a type is sized is a precondition to declaring (member/local) variables
of that type, taking values of that type as parameters, returning values of that
type, and defining arrays of that type. There will generally be additional
requirements to initialize, move, or destroy values of that type as needed.

Example:

```
interface Foo {
  impl as DefaultConstructible;  // See "interface requiring other interfaces".
}
class Bar {  // Classes are "sized" by default.
  impl as Foo;
}
fn F[T: Foo](x: T*) {  // T is unsized.
  var y: T;  // Illegal: T is unsized.
}
// T is sized, but its size is only known generically.
fn G[T: Foo & Sized](x: T*) {
  var y: T = *x;  // Allowed: T is sized and default constructible.
}
var z: Bar;
G(&z);  // Allowed: Bar is sized and implements Foo.
```

**Note:** The compiler will determine which types are "sized", this is not
something types will implement explicitly like ordinary interfaces.

**Open question:** Even if the size is fixed, it won't be known at the time of
compiling the generic function if we are using the dynamic strategy. Should we
automatically [box](#boxed) local variables when using the dynamic strategy? Or
should we only allow `MaybeBox` values to be instantiated locally?

**Open question:** Should the `Sized` type-of-type expose an associated constant
with the size? So you could say `T.ByteSize` in the above example to get a
generic int value with the size of `T`. Similarly you might say `T.ByteStride`
to get the number of bytes used for each element of an array of `T`.

#### Model

This requires a special integer field be included in the witness table type to
hold the size of the type. This field will only be known generically, so if its
value is used for type checking, we need some way of evaluating those type tests
symbolically.

### `TypeId`

There are some capabilities every type can provide. For example, every type
should be able to return its name or identify whether it is equal to another
type. It is rare, however, for code to need to access these capabilities, so we
relegate these capabilities to an interface called `TypeId` that all types
automatically implement. This way generic code can indicate that it needs those
capabilities by including `TypeId` in the list of requirements. In the case
where no type capabilities are needed, for example the code is only manipulating
pointers to the type, you would write `T:! Type` and get the efficiency of
`void*` but without giving up type safety.

```
fn SortByAddress[T:! Type](v: Vector(T*)*) { ... }
```

In particular, we should in general avoid monomorphizing to generate multiple
instantiations of the function in this case.

**Open question:** Should `TypeId` be implemented externally for types to avoid
name pollution (`.TypeName`, `.TypeHash`, etc.) unless the function specifically
requests those capabilities?

## Alternatives considered

FIXME: Issue
[#780: How to write constraints](https://github.com/carbon-language/carbon-lang/issues/780)
considered other forms that constraints could be written:

-   `where` clauses on declarations instead of types (Swift and Rust)
-   parameter passing style used by Rust
-   whole expression constraint intersections

FIXME: other keywords: `requires`, `with`, `if`

FIXME: The
["parameterized type implements interface"](/docs/design/generics/details.md#parameterized-type-implements-interface)
case motivated Rust to
[add support for `where` clauses](https://rust-lang.github.io/rfcs/0135-where.html#motivation).

### Inline constraints instead of `.Self`

FIXME: Alternative to using `.Self` for
[recursive constraints](/docs/design/generics/details.md#recursive-constraints).

However, you can't always avoid using `.Self`, since naming the constraint
before using it doesn't allow you to define the `Container` interface above,
since the named constraint refers to `Container` in its definition.

**Rejected alternative:** To use this `constraint` trick to define `Container`,
you'd have to allow it to be defined inline in the `Container` definition:

```
interface Container {
  let ElementType:! Type;

  constraint ContainerIsSlice {
    extends Container where Container.SliceType == Self;
  }
  let SliceType:! ContainerIsSlice where .ElementType == ElementType;

  fn GetSlice[addr me: Self*](start: IteratorType,
                                    IteratorType: end) -> SliceType;
}
```

### Self reference instead of `.Self`

FIXME: Alternative to using `.Self` for
[recursive constraints](/docs/design/generics/details.md#recursive-constraints).

**Rejected alternative:** We could use the name of the type being declared
inside the type declaration, as in `T:! HasAbs(.MagnitudeType = T)`.

### No inferred/implied constraints for interfaces

FIXME

In interfaces, these constraints can be obscured:

```
interface I(A:! Type, B:! Type, C:! Type, D:! Type, E:! Type) {
  let SwapType:! I(B, A, C, D, E);
  let CycleType:! I(B, C, D, E, A);
  fn LookUp(hm: HashMap(D, E)*) -> E;
  fn Foo(x: Bar(A, B));
}
```

All type arguments to "I" must actually implement `Hashable` (since
[an adjacent swap and a cycle generate the full symmetry group on 5 elements](https://www.mathcounterexamples.net/generating-the-symmetric-group-with-a-transposition-and-a-maximal-length-cycle/)).
And additional restrictions on those types depend on the definition of `Bar`.
For example, this definition

```
class Bar(A:! Type, B:! ComparableWith(A)) { ... }
```

would imply that all the type arguments to `I` would have to be comparable with
every other. This propagation problem means that allowing implicit constraints
to be inferred in this context is substantial (potentially unbounded?) work for
the compiler, and these implied constraints are not at all clear to human
readers of the code either.

**Conclusion:** The initial declaration part of an `interface`, type definition,
or associated type declaration should include complete description of all needed
constraints.

Furthermore, inferring that two types are equal (in contrast to the type bound
constraints described so far) introduces additional problems for establishing
which types are equal in a generic context.
