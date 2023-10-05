# Carbon: Rewrite constraint details

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

-   [Rewrite constraints](#rewrite-constraints)
-   [Combining constraints with `&`](#combining-constraints-with-)
-   [Combining constraints with `and`](#combining-constraints-with-and)
-   [Combining constraints with `extend`](#combining-constraints-with-extend)
-   [Combining constraints with `require` and `impls`](#combining-constraints-with-require-and-impls)
-   [Rewrite constraint resolution](#rewrite-constraint-resolution)
-   [Precise rules and termination](#precise-rules-and-termination)
    -   [Qualified name lookup](#qualified-name-lookup)
    -   [Type substitution](#type-substitution)
    -   [Examples](#examples)
    -   [Termination](#termination)

<!-- tocstop -->

## Rewrite constraints

Rewrite constraints are [`where` clauses](details.md#where-constraints) of the
form `.AssociatedConstant = Value`. Given `T:! A where .B = C`, references to
`T.(A.B)` are rewritten to `C`. This appendix describes the precise rules
governing them.

## Combining constraints with `&`

Suppose we have `X = C where .R = A` and `Y = C where .R = B`. What should
`C & X` produce? What should `X & Y` produce?

-   Combining two rewrite rules with different rewrite targets results in a
    facet type where the associated constant is ambiguous. Given `T:! X & Y`,
    the type expression `T.R` is ambiguous between a rewrite to `A` and a
    rewrite to `B`. But given `T:! X & X`, `T.R` is unambiguously rewritten to
    `A`.
-   Combining a constraint with a rewrite rule with a constraint with no rewrite
    rule preserves the rewrite rule, so `C & X` is the same as `X`. For example,
    supposing that `interface Container` extends `interface Iterable`, and
    `Iterable` has an associated constant `Element`, the constraint
    `Container & (Iterable where .Element = i32)` is the same as the constraint
    `(Container & Iterable) where .Element = i32` which is the same as the
    constraint `Container where .Element = i32`.

If the rewrite for an associated constant is ambiguous, the facet type is
rejected during [constraint resolution](#rewrite-constraint-resolution).

> **Alternative considered:** We could perhaps say that `X & Y` results in a
> facet type where the type of `R` has the union of the interface of `A` and the
> interface of `B`, and that `C & X` similarly results in a facet type where the
> type of `R` has the union of the interface of `A` and the interface originally
> specified by `C`.

## Combining constraints with `and`

It's possible for one `=` constraint in a `where` to refer to another. When this
happens, the facet type `C where A and B` is interpreted as
`(C where A) where B`, so rewrites in `A` are applied immediately to names in
`B`, but rewrites in `B` are not applied to names in `A` until the facet type is
[resolved](#rewrite-constraint-resolution):

```carbon
interface C {
  let T:! type;
  let U:! type;
  let V:! type;
}
class M {
  alias Me = Self;
}
// ✅ Same as `C where .T = M and .U = M.Me`, which is
// the same as `C where .T = M and .U = M`.
fn F[A:! C where .T = M and .U = .T.Me]() {}
// ❌ No member `Me` in `A.T:! type`.
fn F[A:! C where .U = .T.Me and .T = M]() {}
```

## Combining constraints with `extend`

Within an interface or named constraint, `extend` can be used to extend a
constraint that has rewrites.

```carbon
interface A {
  let T:! type;
  let U:! type;
}
interface B {
  extend A where .T = .U and .U = i32;
}

var n: i32;

// ✅ Resolved constraint on `T` is
// `B where .(A.T) = i32 and .(A.U) = i32`.
// `T.(A.T)` is rewritten to `i32`.
fn F(T:! B) -> T.(A.T) { return n; }
```

## Combining constraints with `require` and `impls`

Within an interface or named constraint, the `require T impls C` and
`require Self impls C` syntaxes do not change the type of `T` or `Self`,
respectively, so any `=` constraints that they specify do not result in rewrites
being performed when the type `T` or `Self` is later used. Such `=` constraints
are equivalent to `==` constraints:

```carbon
interface A {
  let T:! type;
  let U:! type;
}
constraint C {
  extend A where .T = .U and .U = i32;
}
constraint D {
  extend A where .T == .U and .U == i32;
}
interface B {
  // OK, equivalent to `require Self impls D;` or
  // `require Self impls A where .T == .U and .U == i32;`.
  require Self impls C;
}

var n: i32;

// ❌ No implicit conversion from `i32` to `T.(A.T)`.
// Resolved constraint on `T` is
// `B where T.(A.T) == T.(A.U) and T.(A.U) == i32`.
// `T.(A.T)` is single-step equal to `T.(A.U)`, and
// `T.(A.U)` is single-step equal to `i32`, but
// `T.(A.T)` is not single-step equal to `i32`.
fn F(T:! B) -> T.(A.T) { return n; }
```

Because `=` constraints are effectively treated as `==` constraints in an
`require Self impls C` or `require T impls C` declaration in an interface or
named constraint, it is an error to specify such a `=` constraint directly in
`C`. A purely syntactic check is used to determine if an `=` is specified
directly in an expression:

-   An `=` constraint is specified directly in its enclosing `where` expression.
-   If an `=` constraint is specified directly in an operand of an `&` or
    `(`...`)`, then it is specified directly in that enclosing expression.

For example:

```carbon
// Compile-time identity function.
fn Identity[T:! type](x:! T) -> T { return x; }

interface E {
  // ❌ Rewrite constraint specified directly.
  require Self impls A where .T = .U and .U = i32;
  // ❌ Rewrite constraint specified directly.
  require Self impls type & (A where .T = .U and .U = i32);
  // ✅ Not specified directly, but does not result
  // in any rewrites being performed.
  require Self impls Identity(A where .T = .U and .U = i32);
}
```

The same rules apply to `where`...`impls` constraints. Note that `.T == U`
constraints are also not allowed in this context, because the reference to `.T`
is rewritten to `.Self.T`, and `.Self` is ambiguous.

```carbon
// ❌ Rewrite constraint specified directly in `impls`.
fn F[T:! A where .U impls (A where .T = i32)]();

// ❌ Reference to `.T` in same-type constraint is ambiguous:
// does this mean the outer or inner `.Self.T`?
fn G[T:! A where .U impls (A where .T == i32)]();

// ✅ Not specified directly, but does not result
// in any rewrites being performed. Return type
// is not rewritten to `i32`.
fn H[T:! type where .Self impls C]() -> T.(A.U);

// ✅ Return type is rewritten to `i32`.
fn I[T:! C]() -> T.(A.U);
```

## Rewrite constraint resolution

When a facet type is used as the declared type of a facet `T`, the constraints
that were specified within that facet type are _resolved_ to determine the
constraints that apply to `T`. This happens:

-   When the constraint is used explicitly when declaring a symbolic binding,
    like a generic parameter or associated constant, of the form
    `T:! Constraint`.
-   When declaring that a type implements a constraint with an `impl`
    declaration, such as `impl T as Constraint`. Note that this does not include
    `require` ... `impls` constraints appearing in `interface` or `constraint`
    declarations.

In each case, the following steps are performed to resolve the facet type's
abstract constraints into a set of constraints on `T`:

-   If multiple rewrites are specified for the same associated constant, they
    are required to be identical, and duplicates are discarded.
-   Rewrites are performed on other rewrites in order to find a fixed point,
    where no rewrite applies within any other rewrite. If no fixed point exists,
    the generic parameter declaration or `impl` declaration is invalid.
-   Rewrites are performed throughout the other constraints in the facet type --
    that is, in any `==` constraints and `impls` constraints -- and the type
    `.Self` is replaced by `T` throughout the constraint.

```carbon
// ✅ `.T` in `.U = .T` is rewritten to `i32` when initially
// forming the facet type.
// Nothing to do during constraint resolution.
fn InOrder[A:! C where .T = i32 and .U = .T]() {}
// ✅ Facet type has `.T = .U` before constraint resolution.
// That rewrite is resolved to `.T = i32`.
fn Reordered[A:! C where .T = .U and .U = i32]() {}
// ✅ Facet type has `.U = .T` before constraint resolution.
// That rewrite is resolved to `.U = i32`.
fn ReorderedIndirect[A:! (C where .T = i32) & (C where .U = .T)]() {}
// ❌ Constraint resolution fails because
// no fixed point of rewrites exists.
fn Cycle[A:! C where .T = .U and .U = .T]() {}
```

To find a fixed point, we can perform rewrites on other rewrites, cycling
between them until they don't change or until a rewrite would apply to itself.
In the latter case, we have found a cycle and can reject the constraint. Note
that it's not sufficient to expand only a single rewrite until we hit this
condition:

```carbon
// ❌ Constraint resolution fails because
// no fixed point of rewrites exists.
// If we only expand the right-hand side of `.T`,
// we find `.U`, then `.U*`, then `.U**`, and so on,
// and never detect a cycle.
// If we alternate between them, we find
// `.T = .U*`, then `.U = .U**`, then `.V = .U***`,
// then `.T = .U**`, then detect that the `.U` rewrite
// would apply to itself.
fn IndirectCycle[A:! C where .T = .U and .U = .V* and .V = .U*]();
```

After constraint resolution, no references to rewritten associated constants
remain in the constraints on `T`.

If a facet type is never used to constrain a type, it is never subject to
constraint resolution, and it is possible for a facet type to be formed for
which constraint resolution would always fail. For example:

```carbon
package Broken api;

interface X {
  let T:! type;
  let U:! type;
}
let Bad:! auto = (X where .T = .U) & (X where .U = .T);
// Bad is not used here.
```

In such cases, the facet type `Broken.Bad` is not usable: any attempt to use
that facet type to constrain a type would perform constraint resolution, which
would always fail because it would discover a cycle between the rewrites for
`.T` and for `.U`. In order to ensure that such cases are diagnosed, a trial
constraint resolution is performed for all facet types. Note that this trial
resolution can be skipped for facet types that are actually used, which is the
common case.

## Precise rules and termination

This section explains the detailed rules used to implement rewrites. There are
two properties we aim to satisfy:

1.  After type-checking, no symbolic references to associated constants that
    have an associated rewrite rule remain.
2.  Type-checking always terminates in a reasonable amount of time, ideally
    linear in the size of the problem.

Rewrites are applied in two different phases of program analysis.

-   During qualified name lookup and type checking for qualified member access,
    if a rewritten member is looked up, the right-hand side's value and type are
    used for subsequent checks.
-   During substitution, if the symbolic name of an associated constant is
    substituted into, and substitution into the left-hand side results in a
    value with a rewrite for that constant, that rewrite is applied.

In each case, we always rewrite to a value that satisfies property 1 above, and
these two steps are the only places where we might form a symbolic reference to
an associated cosntant, so property 1 is recursively satisfied. Moreover, we
apply only one rewrite in each of the above cases, satisfying property 2.

### Qualified name lookup

Qualified name lookup into either a facet parameter or into an expression whose
type is a symbolic type `T` -- either a facet parameter or an associated facet
-- considers names from the facet type `C` of `T`, that is, from `T`’s declared
type.

```carbon
interface C {
  let M:! i32;
  let U:! C;
}
fn F[T:! C](x: T) {
  // Value is C.M in all four of these
  let a: i32 = x.M;
  let b: i32 = T.M;
  let c: i32 = x.U.M;
  let d: i32 = T.U.M;
}
```

When looking up the name `N`, if `C` is an interface `I` and `N` is the name of
an associated constant in that interface, the result is a symbolic value
representing "the member `N` of `I`". If `C` is formed by combining interfaces
with `&`, all such results are required to find the same associated constant,
otherwise we reject for ambiguity.

If a member of a class or interface is named in a qualified name lookup, the
type of the result is determined by performing a substitution. For an interface,
`Self` is substituted for the self type, and any parameters for that class or
interface (and enclosing classes or interfaces, if any) are substituted into the
declared type.

```carbon
interface SelfIface {
  fn Get[self: Self]() -> Self;
}
class UsesSelf(T:! type) {
  // Equivalent to `fn Make() -> UsesSelf(T)*;`
  fn Make() -> Self*;
  impl as SelfIface;
}

// ✅ `T = i32` is substituted into the type of `UsesSelf(T).Make`,
// so the type of `UsesSelf(i32).Make` is `fn () -> UsesSelf(i32)*`.
let x: UsesSelf(i32)* = UsesSelf(i32).Make();

// ✅ `Self = UsesSelf(i32)` is substituted into the type
// of `SelfIface.Get`, so the type of `UsesSelf(i32).(SelfIface.Get)`
// is `fn [self: UsesSelf(i32)]() -> UsesSelf(i32)`.
let y: UsesSelf(i32) = x->Get();
```

If a facet type `C` into which lookup is performed includes a `where` clause
saying `.N = U`, and the result of qualified name lookup is the associated
constant `N`, that result is replaced by `U`, and the type of the result is the
type of `U`. No substitution is performed in this step, not even a `Self`
substitution -- any necessary substitutions were already performed when forming
the facet type `C`, and we don’t consider either the declared type or value of
the associated constant at all for this kind of constraint. Going through an
example in detail:

```carbon
interface A {
  let T:! type;
}
interface B {
  let U:! type;
  // More explicitly, this is of type `A where .(A.T) = Self.(B.U)`
  let V:! A where .T = U;
}
// Type of W is B.
fn F[W:! B](x: W) {
  // The type of the expression `W` is `B`.
  // `W.V` finds `B.V` with type `A where .(A.T) = Self.(B.U)`.
  // We substitute `Self` = `W` giving the type of `u` as
  // `A where .(A.T) = W.(B.U)`.
  let u:! auto = W.V;
  // The type of `u` is `A where .(A.T) = W.(B.U)`.
  // Lookup for `u.T` resolves it to `u.(A.T)`.
  // So the result of the qualified member access is `W.(B.U)`,
  // and the type of `v` is the type of `W.(B.U)`, namely `type`.
  // No substitution is performed in this step.
  let v:! auto = u.T;
}
```

The more complex case of

```carbon
fn F2[Z:! B where .U = i32](x: Z);
```

is discussed later.

### Type substitution

At various points during the type-checking of a Carbon program, we need to
substitute a set of (binding, value) pairs into a symbolic value. We saw an
example above: substituting `Self = W` into the type `A where .(A.T) = Self.U`
to produce the value `A where .(A.T) = W.U`. Another important case is the
substitution of inferred parameter values into the type of a function when
type-checking a function call:

```carbon
fn F[T:! C](x: T) -> T;
fn G(n: i32) -> i32 {
  // Deduces T = i32, which is substituted
  // into the type `fn (x: T) -> T` to produce
  // `fn (x: i32) -> i32`, giving `i32` as the type
  // of the call expression.
  return F(n);
}
```

Qualified name lookup is not re-done as a result of type substitution. For a
template, we imagine there’s a completely separate step that happens before type
substitution, where qualified name lookup is redone based on the actual value of
template arguments; this proceeds as described in the previous section.
Otherwise, we performed the qualified name lookup when type-checking symbolic
expressions, and do not do it again:

```carbon
interface IfaceHasX {
  let X:! type;
}
class ClassHasX {
  class X {}
}
interface HasAssoc {
  let Assoc:! IfaceHasX;
}

// Qualified name lookup finds `T.(HasAssoc.Assoc).(IfaceHasX.X)`.
fn F(T:! HasAssoc) -> T.Assoc.X;

fn G(T:! HasAssoc where .Assoc = ClassHasX) {
  // `T.Assoc` rewritten to `ClassHasX` by qualified name lookup.
  // Names `ClassHasX.X`.
  var a: T.Assoc.X = {};
  // Substitution produces `ClassHasX.(IfaceHasX.X)`,
  // not `ClassHasX.X`.
  var b: auto = F(T);
}
```

During substitution, we might find a member access that named an opaque symbolic
associated constant in the original value can now be resolved to some specific
value. It’s important that we perform this resolution:

```carbon
interface A {
  let T:! type;
}
class K { fn Member(); }
fn H[U:! A](x: U) -> U.T;
fn J[V:! A where .T = K](y: V) {
  // We need the interface of `H(y)` to include
  // `K.Member` in order for this lookup to succeed.
  H(y).Member();
}
```

The values being substituted into the symbolic value are themselves already
fully substituted and resolved, and in particular, satisfy property 1 given
above.

Substitution proceeds by recursively rebuilding each symbolic value, bottom-up,
replacing each substituted binding with its value. Doing this naively will
propagate values like `i32` in the `F`/`G` case earlier in this section, but
will not propagate rewrite constants like in the `H`/`J` case. The reason is
that the `.T = K` constraint is in the _type_ of the substituted value, rather
than in the substituted value itself: deducing `T = i32` and converting `i32` to
the type `C` of `T` preserves the value `i32`, but deducing `U = V` and
converting `V` to the type `A` of `U` discards the rewrite constraint.

In order to apply rewrites during substitution, we associate a set of rewrites
with each value produced by the recursive rebuilding procedure. This is somewhat
like having substitution track a refined facet type for the type of each value,
but we don’t need -- or want -- for the type to change during this process, only
for the rewrites to be properly applied. For a substituted binding, this set of
rewrites is the rewrites found on the type of the corresponding value prior to
conversion to the type of the binding. When rebuilding a member access
expression, if we have a rewrite corresponding to the accessed member, then the
resulting value is the target of the rewrite, and its set of rewrites is that
found in the type of the target of the rewrite, if any. Because the target of
the rewrite is fully resolved already, we can ask for its type without
triggering additional work. In other cases, the rewrite set is empty; all
necessary rewrites were performed when type-checking the value we're
substituting into.

Continuing an example from [qualified name lookup](#qualified-name-lookup):

```carbon
interface A {
  let T:! type;
}
interface B {
  let U:! type;
  let V:! A where .T = U;
}

// Type of the expression `Z` is `B where .(B.U) = i32`
fn F2[Z:! B where .U = i32](x: Z) {
  // The type of the expression `Z` is `B where .U = i32`.
  // `Z.V` is looked up and finds the associated facet `(B.V)`.
  // The declared type is `A where .(A.T) = Self.U`.
  // We substitute `Self = Z` with rewrite `.U = i32`.
  // The resulting type is `A where .(A.T) = i32`.
  // So `u` is `Z.V` with type `A where .(A.T) = i32`.
  let u:! auto = Z.V;
  // The type of `u` is `A where .(A.T) = i32`.
  // Lookup for `u.T` resolves it to `u.(A.T)`.
  // So the result of the qualified member access is `i32`,
  // and the type of `v` is the type of `i32`, namely `type`.
  // No substitution is performed in this step.
  let v:! auto = u.T;
}
```

### Examples

```carbon
interface Container {
  let Element:! type;
}
interface SliceableContainer {
  extend Container;
  let Slice:! Container where .Element = Self.(Container.Element);
}
// ❌ Qualified name lookup rewrites this facet type to
// `SliceableContainer where .(Container.Element) = .Self.(Container.Element)`.
// Constraint resolution rejects this because this rewrite forms a cycle.
fn Bad[T:! SliceableContainer where .Element = .Slice.Element](x: T.Element) {}
```

```carbon
interface Helper {
  let D:! type;
}
interface Example {
  let B:! type;
  let C:! Helper where .D = B;
}
// ✅ `where .D = ...` by itself is fine.
// `T.C.D` is rewritten to `T.B`.
fn Allowed(T:! Example, x: T.C.D);
// ❌ But combined with another rewrite, creates an infinite loop.
// `.C.D` is rewritten to `.B`, resulting in `where .B = .B`,
// which causes an error during constraint resolution.
// Using `==` instead of `=` would make this constraint redundant,
// rather than it being an error.
fn Error(T:! Example where .B = .C.D, x: T.C.D);
```

```carbon
interface Allowed;
interface AllowedBase {
  let A:! Allowed;
}
interface Allowed {
  extend AllowedBase where .A = .Self;
}
// ✅ The final type of `x` is `T`. It is computed as follows:
// In `((T.A).A).A`, the inner `T.A` is rewritten to `T`,
// resulting in `((T).A).A`, which is then rewritten to
// `(T).A`, which is then rewritten to `T`.
fn F(T:! Allowed, x: ((T.A).A).A);
```

```carbon
interface MoveYsRight;
constraint ForwardDeclaredConstraint(X:! MoveYsRight);
interface MoveYsRight {
  let X:! MoveYsRight;
  // Means `Y:! MoveYsRight where .X = X.Y`
  let Y:! ForwardDeclaredConstraint(X);
}
constraint ForwardDeclaredConstraint(X:! MoveYsRight) {
  extend MoveYsRight where .X = X.Y;
}
// ✅ The final type of `x` is `T.X.Y.Y`. It is computed as follows:
// The type of `T` is `MoveYsRight`.
// The type of `T.Y` is determined as follows:
// -   Qualified name lookup finds `MoveYsRight.Y`.
// -   The declared type is `ForwardDeclaredConstraint(Self.X)`.
// -   That is a named constraint, for which we perform substitution.
//     Substituting `X = Self.X` gives the type
//     `MoveYsRight where .X = Self.X.Y`.
// -   Substituting `Self = T` gives the type
//     `MoveYsRight where .X = T.X.Y`.
// The type of `T.Y.Y` is determined as follows:
// -   Qualified name lookup finds `MoveYsRight.Y`.
// -   The declared type is `ForwardDeclaredConstraint(Self.X)`.
// -   That is a named constraint, for which we perform substitution.
//     Substituting `X = Self.X` gives the type
//     `MoveYsRight where .X = Self.X.Y`.
// -   Substituting `Self = T.Y` with
//     rewrite `.X = T.X.Y` gives the type
//     `MoveYsRight where .X = T.Y.X.Y`, but
//     `T.Y.X` is replaced by `T.X.Y`, giving
//     `MoveYsRight where .X = T.X.Y.Y`.
// The type of `T.Y.Y.X` is determined as follows:
// -   Qualified name lookup finds `MoveYsRight.X`.
// -   The type of `T.Y.Y` says to rewrite that to `T.X.Y.Y`.
// -   The result is `T.X.Y.Y`, of type `MoveYsRight`.
fn F4(T:! MoveYsRight, x: T.Y.Y.X);
```

### Termination

Each of the above steps performs at most one rewrite, and doesn't introduce any
new recursive type-checking steps, so should not introduce any new forms of
non-termination. Rewrite constraints thereby give us a deterministic,
terminating type canonicalization mechanism for associated constants: in `A.B`,
if the type of `A` specifies that `.B = C`, then `A.B` is replaced by `C`.
Equality of types constrained in this way is transitive.

However, some existing forms of non-termination may remain, such as template
instantiation triggering another template instantiation. Such cases will need to
be detected and handled in some way, such as by a depth limit, but doing so
doesn't compromise the soundness of the type system.
