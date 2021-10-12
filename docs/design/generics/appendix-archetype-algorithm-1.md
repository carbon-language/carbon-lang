<!--
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

## Rewrite to normalized form algorithm

FIXME

This leads to the question of whether we can describe a set of restrictions on
`where` clauses that would allow us to directly translate them into the argument
passing form. If so, we could allow the `where` clause syntax and still use the
above efficient decision procedure.

Consider an interface with one associate type that has `where` constraints:

```
interface Foo {
  // Some associated types
  let A:! ...;
  let B:! Z where B.X == ..., B.Y == ...;
  let C:! ...;
}
```

These forms of `where` clauses are allowed because we can rewrite them into the
argument passing form:

| `where` form                   | argument passing form   |
| ------------------------------ | ----------------------- |
| `let B:! Z where B.X == A`     | `let B:! Z(.X = A)`     |
| `let B:! Z where B.X == A.T.U` | `let B:! Z(.X = A.T.U)` |
| `let B:! Z where B.X == Self`  | `let B:! Z(.X = Self)`  |
| `let B:! Z where B.X == B`     | `let B:! Z(.X = .Self)` |

Note that the second example would not be allowed if `A.T.U` had the same type
as `B.X`, to avoid non-terminating recursion.

These forms of `where` clauses are forbidden:

| Example forbidden `where` form           | Rule                                     |
| ---------------------------------------- | ---------------------------------------- |
| `let B:! Z where B == ...`               | must have a dot on left of `==`          |
| `let B:! Z where B.X.Y == ...`           | must have a single dot on left of `==`   |
| `let B:! Z where A.X == ...`             | `A` ≠ `B` on left of `==`                |
| `let B:! Z where B.X == ..., B.X == ...` | no two constraints on same member        |
| `let B:! Z where B.X == B.Y`             | right side can't refer to members of `B` |
| `let B:! Z where B.X == C`               | no forward reference                     |

There is some room to rewrite other `where` expressions into allowed argument
passing forms. One simple example is allowing the two sides of the `==` in one
of the allowed forms to be swapped, but more complicated rewrites may be
possible. For example,

```
let B:! Z where B.X == B.Y;
```

might be rewritten to:

```
[let XY:! ...];
let B:! Z(.X = XY, .Y = XY);
```

except it may be tricky in general to find a type for `XY` that satisfies the
constraints on both `B.X` and `B.Y`. Similarly,

```
let A:! ...;
let B:! Z where B == A.T.U
```

might be rewritten as:

```
let A:! ...;
alias B = A.T.U;
```

unless the type bounds on `A.T.U` do not match the `Z` bound on `B`. In that
case, we need to find a type-of-type `Z2` that represents the intersection of
the two type constraints and a different rewrite:

```
let Z2:! B
[let AT:! ...(.U = B)];
let A:! ...(.T = AT);
```

**Note:** It would be great if the
['&' operator for type-of-types](#combining-interfaces-by-anding-type-of-types)
was all we needed to define the intersection of two type constraints, but it
isn't yet defined for two type-of-types that have the same interface but with
different constraints. And that requires being able to automatically combine
constraints of the form `B.X == Foo` and `B.X == Bar`.

**Open question:** How much rewriting can be done automatically?

**Open question:** Is there a simple set of rules explaining which `where`
clauses are allowed that we could explain to users?

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
