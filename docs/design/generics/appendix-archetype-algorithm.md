# Carbon generics appendix: Archetype algorithm

<!--
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

<!-- toc -->

## Table of contents

-   [Overview](#overview)
-   [Rewrite to normalized form algorithm](#rewrite-to-normalized-form-algorithm)
    -   [Transform implied constraints](#transform-implied-constraints)
    -   [Parameters are assigned ids](#parameters-are-assigned-ids)
    -   [Where clauses are rewritten](#where-clauses-are-rewritten)
        -   [Combining type-of-types](#combining-type-of-types)
    -   [Some where clauses are preserved](#some-where-clauses-are-preserved)
    -   [Type checking](#type-checking)
-   [Query algorithm](#query-algorithm)
-   [Rejecting undecidable declarations](#rejecting-undecidable-declarations)

<!-- tocstop -->

## Overview

FIXME: used for answering what type and type-of-type an expression involving
generics has, and whether two types are necessarily equal even when the types
are not known. Tricky part is incorporating constraints that can change the type
from what that name is superficially declared as.

FIXME: input is an environment with interface definitions, a signature, and a
dotted-name query; output is an archetype consisting of a canonical type name
and a type. An _archetype_ is a representation of the most general type that
satifsies the constraints.

FIXME: Two main pieced of the algorithm: rewriting declarations into a
normalized form, which is done eagerly as the compiler encounters declarations,
and the composition of two normalized declarations which is computed lazily as
needed and then cached.

## Rewrite to normalized form algorithm

For every interface, function, or type declaration, the Carbon compiler will
convert it to normalized form when it first type checks it. The normalized form
is both to simplify type checking and support later queries.

The normalized form consists of a list of archetypes and constants. An archetype
has a few components:

-   a unique id like `$3`
-   a type, or a list of interfaces each of which may have name bindings,
-   a list of dotted names each with type-of-types

The first listed dotted name, if any, defines the canonical name for the
archetype. The list of interfaces defines the requirements on that specific id,
while the type-of-type associated with a dotted name defines the unqualified
names that are available as members of that specific name.

**Open question:** I don't know if I need the following restriction:

> There are no forward references allowed in the name bindings, which results in
> the ids generally being listed in decreasing order.

In addition to the normalized form, we maintain a list of `where` constraint
clauses that have yet to be incoroprated. These `where` clauses include the name
of the type they modify, if any, and are kept in the lexical order they appear
in the source code, to preserve the property that any names mentioned in a
`where` clause will be resolved before we resolve the `where` clause itself.

As an example, these interface and function declarations:

```
interface Iterable {
  var Elt:! Type;
}

interface Container {
  let Elt:! Type;
  let Iter:! Iterable where .Elt == Elt;
  let Slice:! Container where .Elt == Elt and .Slice == .Self;
}

fn Sort[C:! Container where .Elt is Comparable](c: C*)
```

normalize to:

```
Iterable
* $1 :! Type
  - Elt as Type
* $0 :! Iterable{.Elt = $1}
  - Self as Iterable

Container
* $3 :! Type
  - Elt as Type
  - Iter.Elt as Type
  - Slice.Elt as Type
* $2 :! Iterable{.Elt = $3}
  - Iter as Iterable
* $1 :! Container{.Elt = $3, .Slice = $1}
  - Slice as Container
* $0 :! Container{.Elt = $3, .Iter = $2, .Slice = $1}
  - Self as Container

Sort
* $2 :! Comparable
  - C.Elt as Comparable
* $1 :! Container{.Elt=$2}
  - C as Container
```

### Transform implied constraints

For function declarations, the first step of normalization is to transform
implied constraints into explicit constraints. This means adding a `where`
constraint on every generic type that is passed as a parameter to a type.

In this example, `KeyType` is passed as a parameter to `HashSet`

```
class HashSet(KeyType:! Hashable & HasEquals & Movable);

fn PrintSet[KeyType:! Printable](set: HashSet(KeyType));
```

so the transformation adds requirements to `KeyType` to satisfy the constraints
from `HashSet`

```
fn PrintSet[KeyType:! Printable]
    (set: HashSet(KeyType)
     where KeyType is KeyType:! Hashable & HasEquals & Movable);
```

Note that this introduced `where` constraint is not attached to a generic type
parameter, and so won't have a name in list of `where` constraint clauses.

### Parameters are assigned ids

Next, every generic type and constant parameter is extracted from the
declaration, and assigned an id. The `where` clauses are collected separately,
in the order they are declared.

```
fn Sort[C:! Container where .Elt is Comparable](c: C*);
```

is transformed to:

```
Sort
* $1 :! Container
  - C as Container

C where .Elt is Comparable
```

Interfaces also get a `Self` type assigned to reserved id `$0`, with name
bindings for all of its members.

```
interface Iterable {
  var Elt:! Type;
}
```

is transformed to:

```
Iterable
* $1 :! Type
  - Elt as Type
* $0 :! Iterable{.Elt = $1}
  - Self as Iterable
```

### Where clauses are rewritten

Then, `where` clauses are resolved by introducing name bindings, and ids as
needed in the normalized form. The `where` clauses should be rewritten in the
order they appear lexically in the original source code. This leverages the lack
of forward references to ensure names have their final type resolved before we
do anything that relies on that final type. There are a number of patterns that
can be replaced in this way.

-   Create ids for all prefixes of dotted names mentioned in a `where`
    constraint that don't already have names, including the dotted name itself.

    ```
    F
    * $1 :! A
      - Z as A

    Z where .X.Y is B
    ```

    would be rewritten to:

    ```
    F
    * $3 :! __typeof__(A.X.Y)
      - Z.X.Y as __typeof__(A.X.Y)
    * $2 :! __typeof__(A.X){.Y = $3}
      - Z.X as __typeof__(A.X)
    * $1 :! A{.X = $2}
      - Z as A

    Z where .X.Y is B
    ```

-   If the prefix has already been given an id, that id should be reused. In
    this example, `Z.Y` has already been given an id as part of resolving a
    constraint on `Z.Y.X`:

    ```
    F
    * $3 :! C
      - Z.Y.X as C
    * $2 :! B{.X = $3}
      - Z.Y as B
    * $1 :! A{.Y = $2}
      - Z as A

    Z where .Y.W is D
    ```

    Then the remaining `where` clause would be reuse the id for `Z.Y`:

    ```
    F
    * $4 :! __typeof__(A.Y.W)
      - Z.Y.W as __typeof__(A.Y.W)
    * $3 :! C
      - Z.Y.X as C
    * $2 :! B{.X = $3, .W = $4}
      - Z.Y as B
    * $1 :! A{.Y = $2}
      - Z as A

    Z where .Y.W is D
    ```

-   An `is` constraint uses the
    [`__combine__` operator](#combining-type-of-types) to add constraints to a
    type. It also modifies the type-of-type for any listed name that the `where`
    clause is on a prefix of.

    ```
    F
    * $2 :! B
      - Z.Y as B
      - X as B
    * $1 :! A{.Y = $2}
      - Z as A

    Z where .Y is C
    ```

    would be rewritten to:

    ```
    F
    * $2 :! __combine__(B, C)
      - Z.Y as __combine__(B, C)
      - X as B
    * $1 :! A{.Y = $2}
      - Z as A
    ```

    Note that since the `where` clause was defined on `Z`, the type-of-type of
    `Z.Y` is modified but `X` is not.

-   Given a list of parameters like `Z:! A, Y:! B where .X == Z`, the `where`
    clause on `Y` changes the type for members of `Y` but not other names. In
    this case `Y.X` changes but not `Z`. For `W:! C where .U == .V`, the types
    of both `W.U` and `W.V` are changed by the `where` clause.

    FIXME: This is why we keep the `where` clause associated to a name binding,
    because we know it only affects the visible type for names that have this
    name as prefix.

    FIXME: Example

-   Consider two different functions declarations, `F` and `G`:

    ```
    interface A {}
    interface B {
      let X:! D;
    }
    interface C {
      let V:! E;
    }

    fn F[Z:! A, Y:! B where .X == Z, W:! C where .V == Z](...);
    fn G[Z:! A, Y:! B where .X == Z, W:! C where .V == Y.X](...);
    ```

    The names for `W.V` are different between `F` and `G`, even though they end
    up naming the same id in both cases.

    ```
    F or G
    $3 :! A & D
      - Z as A
      - Y.X as A & D
    $2 :! B{.X = $3}
      - Y as B
    $1 :! C
      - W as C
      - W as C

    W where .V == Z  // <-- F
    W where .V == Y.X  // <-- G
    ```

    The only difference is the type of `C.V` in the rewrite:

    ```
    F or G
    $3 :! A & D & E
      - Z as A
      - Y.X as A & D
      - C.V as A & E  // <-- F
      - C.V as A & D & E  // <-- G
    $2 :! B{.X = $3}
      - Y as B
    $1 :! C{.V = $3}
      - W as C
    ```

    To get the type right, we need that the types of `Z` and `Y.X` to be
    finalized before rewriting the `where` clause for `C.V`. Since we don't
    allow forward references, it is suffient to process declarations in the
    order they are declared lexically. Then we set the type of `C.V` from
    `where .V == Something` to
    `__combine__(__typeof__(C.V), __typeof__(Something))` where
    `__typeof__(Something)` can be read out of the normalized form produced so
    far.

FIXME

type expression like FIXME: If reusing an id means changing its type, preserve
the type of anything that doesn't have as a prefix with the

    ```
    interface B {
      let X:! C;
      let W:! D;
    }

    fn K[Z:! A, Y:! B where .X == Z and .W == Z](...);
    ```

    is first transformed to:

    ```
    K
    * $2 :! A
      - Z as A
    * $1 :! B
      - Y as B where .X == Z and .W == Z
    ```

    To enforce the constraint `Y.X == Z`, id `$2` list of interfaces is changed
    from `A` to `__combine__(A, __typeof__(B.X)) == A & C`. Since we are rewriting a
    `where` clause on `Y`, we only want to change the types of names starting
    with `Y`. Since `Z` doesn't have `Y` as a prefix, it gains a cast to
    preserve its type.

    ```
    K
    * $2 :! A & C
      - Z as A
      - Y.X as A & C
    * $1 :! B{.X = $2}
      - Y as B where .W == Z
    ```

-   A `where` constraint saying a member must have a specific type or
    type-of-type creates an id with type combining the existing type and the
    type from the constraint.

    ```
    Sort
    * $1 :! Container
      - C as Container where .Elt is Comparable
    ```

    becomes:

    ```
    Sort
    * $2 :! __combine__(__typeof__(Container.Elt), Comparable)
      - C.Elt as ...
    * $1 :! Container{.Elt = $2}
      - C as Container
    ```

-   A `where` constraint equating two types needs to both ensure an id exists,
    combine their types, and bind both names to that id.

    ```
    G
    * $1 :! A
      - Z as A where .X == .Y
    ```

    becomes:

    ```
    G
    * $2 :! __combine__(__typeof__(A.X), __typeof__(A.Y))
      - Z.X as ...
      - Z.Y as ...
    * $1 :! A{.X = $2, .Y = $2)
      - Z as A
    ```

    Note that only members of the type with the `where` clause get the combined
    type. Any other expression will get a cast associated with its name binding.

    ```
    Container
    * $2 :! Type
      - Elt as Type
    * $1 :! Iterable
      - Iter as Iterable where .Elt == Elt;
    ```

    becomes:

    ```
    Container
    * $2 :! __combine__(Type, __typeof__(Iterable.Elt))
      - Elt as Type
      - Iter.Elt as __combine__(Type, __typeof__(Iterable.Elt))
    * $1 :! Iterable{.Elt = $2}
      - Iter as Iterable
    ```

-   Setting a member equal to `Self` or `.Self` is treated slightly differently.
    Instead of combining types to make sure it satisfies constraints, we just
    typecheck that the existing types do. However, since the rewrite to normal
    form is still in progress at this point, the type check is delayed until the
    [type check step](#type-checking) after the rewrites are complete.

    ```
    Container
    * $1 :! Container
      - Slice as Container where .Slice == .Self
    ```

    becomes:

    ```
    Container
    * $1 :! Container{.Slice = $1}
      - Slice as Container
    ```

-   `Self` is similar to `.Self` except it uses id `$0`. Unlike other ids, id
    `$0` may be referenced before its declaration.

    ```
    Tree
    * $1 :! Tree
      - Child as Tree where .Parent == Self
    * $0 :! Tree{.Child = $1}
      - Self as Tree
    ```

    becomes:

    ```
    Tree
    * $1 :! Tree{.Parent = $0}
      - Child as Tree
    * $0 :! Tree{.Child = $1}
      - Self as Tree
    ```

    Again, checking that `Self` satisfies the constraints on `.Parent` is
    delayed until the [type check step](#type-checking) after the rewrites are
    complete.

-   Setting a member to a specific type, like `i32`, first checks that the type
    satisfies all constraints on that member, and then binds the name of the
    member to the type, replacing the existing list of interfaces and bindings.
    To rewrite the `where` clause in this example, which results from the
    original source code of `Z:! A where .X is B and .X == i32`:

    ```
    H
    * $2 :! B
      - Z.X as B
    * $1 :! A{.X = $2}
      - Z as A where .X == i32
    ```

    the compiler must first check that `i32` implements `B` and report an error
    if it doesn't. Otherwise, it gets rewritten to:

    ```
    H
    * $2 = i32
      - Z.X
    * $1 :! A{.X = $2}
      - Z as A
    ```

-   If two constraints set the same member to types, that member should get the
    unification of those two types, or an error. For example, these parameters
    in the declaration of a function `J`:

    ```
    Y:! C, W:! B, Z:! A where .X == Pair(i32, W)
                          and .X == Pair(Y, f32)
    ```

    would first be rewritten to:

    ```
    J
    * $4 :! C
      - Y as C
    * $3 :! B
      - W as B
    * $2 = Pair(i32, $3)
      - Z.X
    * $1 :! A{.X = $2}
      - Z as A where .X == Pair($4, f32)
    ```

    This would trigger a type error unless `i32` implements `C` and `f32`
    implements `B`, and otherwise it is rewritten to:

    ```
    J
    * $4 = i32
      - Y as C
    * $3 = f32
      - W as B
    * $2 = Pair($4, $3)
      - Z.X
    * $1 :! A{.X = $2}
      - Z
    ```

#### Combining type-of-types

The rewrites in some cases employ the `__combine__` operator, that takes two
type-of-types and returns a type-of-type with all the restrictions of both. In
many cases, this combination is straightforward.

-   `__combine__(X, X) = X`.
-   If interface `BidirectionalIter` extends `ForwardIter`, then
    `__combine__(ForwardIter, BidirectionalIter) = BidirectionalIter`.
-   More generally, if `P` implies `Q` then `__combine__(P, Q) = P`. For
    example, `__combine__(A, A & B) = A & B`.
-   For two different interfaces `A` and `B`, `__combine__(A, B) = A & B`.
-   For two different associated types `X` and `Y` of the same interface `A`,
    `__combine__(A{.X = T}, A{.Y = U}) = A{.X = T, .Y = U}`.

The interesting case is `__combine__(A{.X = T}, A{.X = U})` when `T` and `U` are
different. We could in principle recursively add a rewrite setting them equal,
but to guarantee that the algorithm terminates, we instead give an error. The
insight is that in this case, the error is reasonably clear. In cases that arise
in practice, the error should be enough for the user to fix the issue.

**Open question:** For now, I am calling it "combine" since it is unioning the
constraints but intersecting the types satisfying the constraints, so neither
"union" nor "intersect" seems clear. We could instead call this "unification."

### Some where clauses are preserved

Some `where` constraints can't be eliminated by a rewrite:

-   `where Vector(T) is Printable`
-   `where T != U`

These constraints are left alone, except that they don't need to be associated
with a dotted name since they don't affect their unqualified API.

### Type checking

FIXME: Type check. It is sound to type check one level by induction, assuming
you type check every interface "satisfies its constraints given parameters and
Self meet their constraints." The caller adding additional constraints won't
invalidate that.

FIXME: Need to validate references from `Self` and `.Self` at this stage.

FIXME: invariants of normalized form

-   MAYBE: No cycles and no forward references.
-   Original/immediate associated types at the end and in the original source
    code order.

## Query algorithm

FIXME

However, with enough constraints, we can make an efficient decision procedure
for the argument passing formulation. The way we do this is by assigning every
type expression a canonical type, and then two types expressions are equal if
and only if they are assigned the same canonical type. To show how to assign
canonical types, lets work an example with interfaces `A` and `B` (letters from
the end of the alphabet will represent types), and this function declaration:

```
fn F2[Z:! A, V:! B(.Y = Z, .X = Z.W)](...) { ... }
```

We require the following rules to be enforced by the language definition and
compiler:

-   No forward references in a function declaration.
-   No forward references between items in an `interface` definition.
-   No implicit type equality constraints.

From these rules, we derive rules about which type expressions are canonical.
The first rule is:

> For purposes of type checking a function, the names of types declared in the
> function declaration to the left of the `:!` are all canonical.

This is because these can all be given distinct types freely, only their
associated types can be constrained to be equal to some other type. In this
example, this means that the types `Z` and `V` are both canonical. The second
rule comes from there being no forward references in declarations, and no
implicit type equality constraints:

> No declaration can affect type equality for any declaration to its left.

This means that the canonical types for type expressions starting with `Z.` are
completely determined by the declaration `Z:! A`. Furthermore, since the set of
type expressions starting with `Z.` might be infinite, we adopt the lazy
strategy of only evaluating expressions that are needed for something explicitly
mentioned.

We do need to evaluate `Z.W` though for the `V:! B(.Y = Z, .X = Z.W)`
expression. This is an easy case, though since `Z:! A` doesn't include any
assignments to any associated types. In this case, the associated types of `A`
are all canonical. An alias defined in `A` would of course not be, it would be
set to the canonical type for whatever it is an alias for. For example:

```
interface A {
  // `W` is canonical.
  let W:! A;
  // `U` is not canonical, is equal to `W.W`.
  alias U = W.W;
  // `T` is canonical, but `T.Y` is not.
  let T:! B(.Y = Self);
}
```

Next lets examine the definition of `B` so we can resolve expressions starting
with `V.`.

```
interface B {
  let S:! A;
  let Y:! A(.W = S);
  let X:! A;
  let R:! B(.X = S);
}
```

This time we also have assignments `V.Y = Z` and `V.X = Z.W`. As a consequence,
neither `V.Y` nor `V.X` are canonical, and their canonical type is determined
from their assignments. Furthermore, the assignment to `Y` determines `S`, since
`B.S = B.Y.W`, so `V.S` also isn't canonical, it is `V.Y.W` (not canonical)
which is `Z.W` (canonical). Observe that `V.R` is canonical since nothing
constrains it to equal any other type, even though `V.R.X` is not, since it is
`V.S == Z.W`.

The property that there are no forward references between items in interface
definitions ensures that we don't have any cycles that could lead to infinite
loops. That is, the members of an associated type in an interface definition can
only be constrained to equal values that don't depend on that member.

This is almost enough to ensure that the process terminates, except when an
associated type bound is the same interface recursively. The bad case is:

```
interface Broken {
  let Q:! Broken;
  let R:! Broken(.R = Q.R.R);
}

fn F[T:! Broken](x: T) {
  // T.R.R not canonical
  // == T.Q.R.R not canonical
  // == T.Q.Q.R.R not canonical
  // etc.
}
```

The problem here is that while we have a ordering for expressions that guaranees
there are no loops, we don't have a guarantee that there are only finitely many
smaller expressions when we have recursion. With recursion, we can create an
infinite sequence of smaller expressions by allowing their length to grow
without bound. This means we need to add one more rule to ensure that the
algorithm terminates:

> It is illegal to constrain a member of an associated type to (transitively)
> equal a longer expression with the same interface bound.

A few notes on this rule:

-   The word "transitively" is needed if mutual recursion is allowed between
    interfaces (as in `A` and `B` above).
-   There is an additional restriction if the expression has the same length
    that it only refer to earlier names. Without mutual recursion, this is
    already precluded by the "no forward references" rule.
-   We are relying on there being a finite number of interfaces, so we ignore
    [interface parameters](#parameterized-interfaces) when checking this
    condition.
-   This never applies to function declarations, since there is no recursion
    involved in that context.

The fix for this situation is to introduce new deduced associated types:

```
interface Fixed {
  [let RR:! Fixed];
  [let QR:! Fixed(.R = RR)];
  let Q:! Fixed(.R = QR);
  let R:! Fixed(.R = RR);
}

fn F[T:! Fixed](x: T) {
  // T.RR canonical
  // T.R.R == T.RR
  // T.Q.R.R == T.Q.RR == T.QR.R == T.RR
  // T.Q.Q.R.R == T.Q.Q.RR == T.Q.QR.R == T.Q.RR == T.QR.R == T.RR
  // etc.
}
```

The last concern is what happens when an expression is assigned twice. This is
only a problem if it is assigned to two values that resolve to two different
canonical types. That happens in this example:

```
fn F3[N:! A, P:! A, Q:! B(.S = N, .Y = P)](...) { ... }
```

The compiler is required to report an error rejecting this declaration. This is
because the constraints declared in `B` require that `Q.Y.W == Q.S == N` so
`P.Y == N`. This violates the "no implicit type equality constraint" rule since
`P` is not declared with any constraint forcing that to hold. We can't let `Q`'s
declaration affect earlier declarations, otherwise our algorithm would
potentially have to resolve cycles. The compiler should recommend the user
rewrite their code to:

```
fn F3[N:! A, P:! A(.Y = N), Q:! B(.S = N, .Y = P)](...) { ... }
```

This resolves the issue, and with this change the compiler can now correctly
determine canonical types.

**Note:** This algorithm still works with the `.Self` feature from the
["recursive constraints" section](#recursive-constraints). For example, the
expression `let Y:! A(.X = .Self)` means `Y.X == Y` and so the `.Self` on the
right-side represents a shorter and earlier type expression. This precludes
introducing a loop and so is safe.

**Open question:** Can we relax any of the restrictions? For example, perhaps we
would like to allow items in an interface to reference each other, as in:

```
interface D {
  let E:! A(.W = V);
  let V:! A(.W = E);
}
```

This example may come up for graphs where `E` is the edge type and `V` is the
vertex type. In this case `D.E.W == D.V` and `D.V.W == D.E` and we would need
some way of deciding which were canonical (probably `D.E` and `D.V`). This would
have to be restricted to cases where the expression on the right has no `.` to
avoid cycles or type expression that grow without bound. Another concern is if
there are type constructors involved:

```
interface Graph {
  let Edges:! A(.W = Vector(Verts));
  let Verts:! A(.W = Vector(Edges));
}
```

## Rejecting undecidable declarations

FIXME
