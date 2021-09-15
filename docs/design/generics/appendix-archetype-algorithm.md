# Carbon generics appendix: Archetype algorithm

<!--
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

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
-   a list of interfaces, each of which may have name bindings, or a type
-   an optional `where` clause, and
-   a list of dotted names with optional type-of-types.

The first listed dotted name defines the canonical name for the archetype. There
are no forward references allowed in the name bindings, which results in the ids
generally being listed in decreasing order.

As an example, these interface and function declarations

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
  - Elt
* $0 :! Iterable{.Elt = $1}
  - Self

Container
* $3 :! Type
  - Elt
  - Iter.Elt
  - Slice.Elt
* $2 :! Iterable{.Elt = $3}
  - Iter
* $1 :! Container{.Elt = $3, .Slice = $1}
  - Slice
* $0 :! Container{.Elt = $3, .Iter = $2, .Slice = $1}
  - Self

Sort
* $2 :! Comparable
  - C.Elt
* $1 :! Container{.Elt=$2}
  - C
```

### Transform implicit/implied/inferred constraints

For function declarations, the first step of normalization is to transform
implicit/implied/inferred constraints into explicit constraints. This means
adding a `where` constraint on every generic type that is passed as a parameter
to a type.

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

### Parameters are assigned ids

Next, every generic type and constant parameter is extracted from the
declaration, and assigned an id.

```
fn Sort[C:! Container where .Elt is Comparable](c: C*);
```

is transformed to:

```
Sort
* $1 :! Container where .Elt is Comparable
  - C
```

Interfaces also get a `Self` type assigned to reserved id `$0`, with name
bindings for all of its members. Id `$0` gets some special treatment, for
example it may be referenced before it is declared.

```
interface Iterable {
  var Elt:! Type;
}
```

is transformed to:

```
Iterable
* $1 :! Type
  - Elt
* $0 :! Iterable{.Elt = $1}
  - Self
```

### Where clauses are rewritten

Then, `where` clauses are replaced with name bindings, introducing ids as
needed. There are a number of patterns that can be replaced in this way.

-   Create ids for all prefixes of dotted names mentioned in a `where`
    constraint that don't already have names.

    ```
    F
    * $1 :! A where .X.Y is B
      - Z
    ```

    would be rewritten:

    ```
    F
    * $2 :! typeof(A.X) where .Y is B
      - Z.X
    * $1 :! A{.X = $2}
      - Z
    ```

-   If the prefix has already been given an id, that id should be reused.

    ```
    F
    * $1 :! A where .X.Y is B and .X.Z is C
      - Z
    ```

    might first be rewritten by the previous rule to:

    ```
    F
    * $2 :! typeof(A.X) where .Y is B
      - Z.X
    * $1 :! A{.X = $2} where .X.Z is C
      - Z
    ```

    Then the second `where` clause would be reuse the id for `Z.X`:

    ```
    F
    * $2 :! typeof(A.X) where .Y is B and .Z is C
      - Z.X
    * $1 :! A{.X = $2}
      - Z
    ```

-   A `where` constraint saying a member must have a specific type or
    type-of-type creates an id with type combining the existing type and the
    type from the constraint.

    ```
    Sort
    * $1 :! Container where .Elt is Comparable
      - C
    ```

    becomes:

    ```
    Sort
    * $2 :! __combine__(typeof(Container.Elt), Comparable)
      - C.Elt
    * $1 :! Container{.Elt = $2}
      - C
    ```

-   A `where` constraint equating two types needs to both ensure an id exists,
    combine their types, and bind both names to that id.

    ```
    G
    * $1 :! A where .X == .Y
      - Z
    ```

    becomes:

    ```
    G
    * $2 :! __combine__(typeof(A.X), typeof(A.Y))
      - Z.X
      - Z.Y
    * $1 :! A{.X = $2, .Y = $2)
      - Z
    ```

    Note that only members of the type with the `where` clause get the combined
    type. Any other expression will get a cast associated with its name binding.

    ```
    Container
    * $2 :! Type
      - Elt
    * $1 :! Iterable where .Elt == Elt;
      - Iter
    ```

    becomes:

    ```
    Container
    * $2 :! __combine__(Type, typeof(Iterable.Elt))
      - Elt as Type
      - Iter.Elt
    * $1 :! Iterable{.Elt = $2}
      - Iter
    ```

-   Setting a member to `Self` or `.Self` is treated slightly differently.
    Instead of combining types to make sure it satisfies constraints, we just
    typecheck that the existing types do.

    ```
    Container
    * $1 :! Container where .Slice == .Self
      - Slice
    ```

    checks that `Container.Slice` satisfies the requirements of `Self` and then
    becomes:

    ```
    Container
    * $1 :! Container{.Slice = $1}
      - Slice
    ```

    `Self` is similar except it uses id `$0`.

-   Setting a member to a specific type, like `i32`, first checks that the type
    satisfies all constraints on that member, and then binds the name of the
    member to the type, possibly replacing an existing id. To rewrite the
    `where` clause in this example, which results from the original source code
    of `Z:! A where .X is B and .X == i32`:

    ```
    H
    * $2 :! B
      - Z.X
    * $1 :! A{.X = $2} where .X == i32
      - Z
    ```

    the compiler must first check that `i32` implements `B` and report an error
    if it doesn't. Otherwise, it gets rewritten to:

    ```
    H
    * $2 = i32
      - Z.X
    * $1 :! A{.X = $2}
      - Z
    ```

    If two constraints set the same member to types, that member should get the
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
      - Y
    * $3 :! B
      - W
    * $2 = Pair(i32, $3)
      - Z.X
    * $1 :! A{.X = $2} where .X == Pair($4, f32)
      - Z
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

The interesting case is `combine(is A{.X = T}, is A{.X = U})` when `T` and `U`
are different. We could in principle recursively add a rewrite setting them
equal, but to guarantee that the algorithm terminates, we instead give an error.
The insight is that in this case, the error is reasonably clear. In cases that
arise in practice, the error should be enough for the user to fix the issue.

### Some where clauses are preserved

Some `where` constraints can't be rewritten and the only rewrite is to make sure
they are attached to the relevant id.

-   `where Vector(T) is Printable` constraints should be moved to `T`
-   `where T != U` constraints should be moved to the latter id
-   `where N > 3` constraints should be moved to `N`

### Type checking

FIXME: Type check. It is sound to type check one level by induction, assuming
you type check every interface "satisfies its constraints given parameters and
Self meet their constraints." The caller adding additional constraints won't
invalidate that.

FIXME: invariants of normalized form

-   No cycles and no forward references.
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
