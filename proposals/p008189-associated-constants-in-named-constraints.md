# Associated constants in named constraints

<!--
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

[Pull request](https://github.com/carbon-language/carbon-lang/pull/8189)

<!-- toc -->

## Table of contents

-   [Abstract](#abstract)
-   [Problem](#problem)
-   [Background](#background)
-   [Proposal](#proposal)
    -   [Deducible associated constants](#deducible-associated-constants)
    -   [Implementing named constraints with associated constants](#implementing-named-constraints-with-associated-constants)
    -   [`final` associated constants](#final-associated-constants)
-   [Future work](#future-work)
    -   [`default` associated constants in named constraints](#default-associated-constants-in-named-constraints)
-   [Rationale](#rationale)
-   [Alternatives considered](#alternatives-considered)
    -   [Require callers to constrain the underlying interface directly](#require-callers-to-constrain-the-underlying-interface-directly)
    -   [Introduce `forunique` or `forsome` existential syntax](#introduce-forunique-or-forsome-existential-syntax)

<!-- tocstop -->

## Abstract

We propose adding support for associated constant declarations in named
constraints, using the same `let`...`:`...`;` syntax as in interfaces, along
with `final let` associated constants with direct definitions. Non-`final`
associated constants in a named constraint must be deducible from the `impl`s of
interfaces required by the constraint.

## Problem

Named constraints allow Carbon to compose interface requirements and hide
lower-level complexity for
[progressive disclosure](/docs/project/principles/progressive_disclosure.md).
However, named constraints currently cannot declare their own associated
constants. This limits their usefulness in two ways:

-   A named constraint cannot project or transform an associated constant of a
    required interface into a simpler associated constant on the named
    constraint itself (for example, exposing an element type `Value` when an
    underlying `Map` interface has `.Value = Vector(Value)`, or exposing a type
    associated constant `Result` when a lower-level extended type interface uses
    an extended type associated constant `.ResultExtType`).
-   A named constraint cannot declare a `final` constant expression or alias to
    a member of a required interface as a named associated constant on the
    constraint.

## Background

This proposal was split out of
[proposal #5389](https://github.com/carbon-language/carbon-lang/pull/5389);
earlier revision and review history may be found in that pull request.

This proposal has no dependencies on pending proposals. It supports progressive
disclosure in several pending proposals also split out of
[proposal #5389](https://github.com/carbon-language/carbon-lang/pull/5389),
including:

-   [proposal #8185: Extended type interfaces and named constraint extensions](https://github.com/carbon-language/carbon-lang/pull/8185),
-   [proposal #8187: Extended-type call interface and `call` syntax](https://github.com/carbon-language/carbon-lang/pull/8187)
    (defining `Core.Call` as a named constraint with `let Result: type;`), and
-   [proposal #8188: Extended-type operator, conversion, and member binding interfaces](https://github.com/carbon-language/carbon-lang/pull/8188)
    (defining type-oriented operator, indexing, and member binding constraints
    such as `Core.AddWith`, `Core.IndexWith`, and `Core.BindToVal`).

We previously considered similar constructs (`forunique` or `forsome`) in the
context of template constraints in
[discussion on 2023-01-23](https://docs.google.com/document/d/1gnJBTfY81fZYvI_QXjwKk1uQHYBNHGqRLI2BS_cYYNQ/edit?resourcekey=0-ql1Q1WvTcDvhycf8LbA9DQ&tab=t.0#heading=h.klm1hw3hfm1n)
and
[as part of the design of supporting `for` loops with C++ interop](https://docs.google.com/document/d/157k0s1e6qgTKUtads4x7BRtLHq8Xesh6DUHtbQ01gjY/edit?tab=t.0#heading=h.bs060kxctkgo).

## Proposal

### Deducible associated constants

We add support for associated constant declarations in named constraints, using
the same `let`...`:`...`;` syntax as in interfaces. Such associated constants
must be used in such a way as to be deducible from the `impl`s of interfaces
required by the constraint. Deducible here means following the same rules as a
deduced parameter in a function signature. When seeing if a facet satisfies a
constraint, the associated constants are deduced.

```carbon
constraint Invalid {
  // ❌ Invalid: `Key` not deducible.
  let Key: type;
  extend require impls Map(Key);
}

constraint MultiMap(Key: type) {
  // Shadows `Value` from `Map`.
  let Value: type;
  extend require impls Map(Key)
      where .Value = Vector(Value);
}
// This is legal since a function like
//   fn F[Value: type]
//     (generic T: Map(Key) where .Value = Vector(Value));
// is valid, since `Value` is deducible from `T`.
```

### Implementing named constraints with associated constants

When implementing the named constraint directly, every associated constant of
the extended interface must be given a value, as before. This can now be done by
giving a value to an associated constant in the named constraint, which will
give values to any derived associated constants in the interface being
implemented.

Example using `MultiMap`:

```carbon
class S2I { ... }

impl S2I as Map(String)
  where .Value = Vector(i32) { ... }

fn MultiGet[T: MultiMap(String)](x: T)
    -> T.Value;
fn MapGet[T: Map(String)](x: T)
    -> T.Value;
fn Restricted(
    generic V: type,
    generic T: MultiMap(String) where .Value = V);

var s2i: S2I = {...};
// `(S2I as Map(String)).Value` is
// `Vector(i32)`, so
// `(S2I as MultiMap(String)).Value`
// is `i32`.
let r1: i32 = MultiGet(s2i);
let r2: Vector(i32) = MapGet(s2i);
// ✅ Allowed
Restricted(i32, S2I);
// ❌ Invalid:
// `(S2I as MultiMap(String)).Value`
// is `i32` not `bool`
Restricted(bool, S2I);

class S2B { ... }

// `(S2B as MultiMap(String)).Value`
// is `bool`, so
// `(S2B as Map(String)).Value` is
// `Vector(bool)`.
impl S2B as MultiMap(String)
  where .Value = bool { ... }

var s2b: S2B = {...};
let r3: bool = MultiGet(s2b);
let r4: Vector(bool) = MapGet(s2b);
// ❌ Invalid:
// `(S2B as MultiMap(String)).Value`
// is `bool` not `i32`
Restricted(i32, S2B);
// ✅ Allowed
Restricted(bool, S2B);
```

Whenever a named constraint is implemented, the resulting `impl` is checked to
verify that it satisfies the named constraint, including giving a single
consistent value to any associated constants it might have.

```carbon
interface X {
  let T: type;
  let U: type;
}

constraint N {
  let V: type;
  extend require impls X
      where .T = Vector(V)
        and .U = Map(V);
}

class Invalid1 {}

impl Invalid1 as N {
  where T = Vector(bool);
  // ❌ Inconsistent, `V` can't be both
  // `bool` and `i32`.
  where U = Map(i32);
}

class Invalid2 {}

impl Invalid2 as N {
  // ❌ Inconsistent, `T` has to be a
  // `Vector` so there is some legal
  // value for `V`.
  where T = i32;
  where U = Map(i32);
}
```

This feature can also be used to convert an extended type associated constant in
an extended interface into a type associated constant in a named constraint, as
in
[proposal #8187](https://github.com/carbon-language/carbon-lang/pull/8187)
(`Core.Call`) and
[proposal #8188](https://github.com/carbon-language/carbon-lang/pull/8188)
(operator, indexing, and member binding named constraints).

### `final` associated constants

The `final` modifier is allowed on associated constants in named constraints. In
this case, it is given a value directly instead of being deduced from the
`impl`s of required interfaces. Examples:

```carbon
constraint VectorI(T: type) {
  // Can be used to give a name to
  // an expression.
  final let Elt: type = Vector(T);

  require impls I where .Elt = Elt;

  // Can be used to create an alias to
  // a member of a required interface.
  final let template Iter: auto
      = I.Iter;
}
```

## Future work

### `default` associated constants in named constraints

Allowing the `default` modifier on associated constants in named constraints is
future work, which we expect to support once we have an algorithm for resolving
when there are multiple applicable defaults (likely using the defaults from the
outermost constraint first). Until then, note that the interface can still have
a default that can be applicable to a named constraint, as in:

```carbon
interface DefaultResult {
  default let ResultList: type = Vector(Self);
  fn F() -> ResultList;
}
constraint C {
  let Elem: type;
  extend require impls DefaultResult
      where .ResultList = Vector(Elem);
}
```

Here an `impl` of `C` that doesn't specify `Elem` will use the default
`DefaultResult.ResultList` of `Vector(Self)`, which will deduce a value of
`Self` for `C.Elem`.

## Rationale

This proposal advances [Carbon's goals](/docs/project/goals.md) and
[principles](/docs/project/principles):

-   [Code that is easy to read, understand, and write](/docs/project/goals.md#code-that-is-easy-to-read-understand-and-write):
    Named constraints can expose concise, domain-appropriate associated
    constants instead of forcing callers and implementers to repeat structural
    `where` constraints on underlying interfaces.
-   [Principle: Progressive disclosure](/docs/project/principles/progressive_disclosure.md):
    Named constraints can wrap lower-level interfaces (such as extended type
    interfaces in
    [proposal #8187](https://github.com/carbon-language/carbon-lang/pull/8187)
    and
    [proposal #8188](https://github.com/carbon-language/carbon-lang/pull/8188))
    while presenting familiar type-level associated constants to everyday users.

## Alternatives considered

### Require callers to constrain the underlying interface directly

Without associated constants in named constraints, callers could still express
the same requirements by adding deduced parameters and `where` clauses directly
on each generic function (for example, `fn MultiGet[Value: type, T: Map(String)
where .Value = Vector(Value)](x: T) -> Value`). However, this leaks
implementation details of the underlying interface to every caller and
implementer, preventing a named constraint from acting as a clean abstraction
boundary (working against
[progressive disclosure](/docs/project/principles/progressive_disclosure.md) and
[software and language evolution](/docs/project/goals.md#software-and-language-evolution)).

### Introduce `forunique` or `forsome` existential syntax

We previously considered existential quantifiers such as `forunique` or
`forsome` to introduce names bound within a constraint. Using standard
associated constant `let` syntax directly inside named constraints avoids
introducing new keywords or a separate mechanism, matches the existing interface
associated constant syntax, and naturally exposes those deduced constants as
named members of the constraint (so callers can write `T.Value` or
`MultiMap(String) where .Value = V`).
