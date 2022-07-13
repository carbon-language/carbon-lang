# Variadics

<!--
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

[Pull request](https://github.com/carbon-language/carbon-lang/pull/####)

<!-- toc -->

## Table of contents

-   [TODO: Initial proposal setup](#todo-initial-proposal-setup)
-   [Problem](#problem)
-   [Background](#background)
-   [Proposal](#proposal)
-   [Details](#details)
-   [Rationale](#rationale)
-   [Alternatives considered](#alternatives-considered)

<!-- tocstop -->

## TODO: Initial proposal setup

> TIP: Run `./new_proposal.py "TITLE"` to do new proposal setup.

2. Create a GitHub pull request, to get a pull request number.
    - Add the `proposal` and `WIP` labels to the pull request.
3. Rename `new.md` to `/proposals/p####.md`, where `####` should be the pull
   request number.
5. Update the link to the pull request (the `####` on line 11).
6. Delete this section.

TODOs indicate where content should be updated for a proposal. See
[Carbon Governance and Evolution](/docs/project/evolution.md) for more details.

FIXME cite to https://github.com/carbon-language/carbon-lang/issues/1162 somewhere

## Problem

Carbon needs a way to define functions and parameterized types that are 
_variadic_, meaning they can take a variable number of arguments.

## Background

C has long supported variadic functions (such as `printf`), but that mechanism
is heavily disfavored in C++ because it isn't type-safe. Instead, C++ provides
a separate feature for defining variadic _templates_, which can be functions,
classes, or even variables. However, variadic templates currently suffer
from several shortcomings. Most notably:

- They must be templates, which mean they typically must be
  defined in header files, are susceptible to code bloat due to template
  instantiation, and generally carry all the other costs associated with
  templating.
- It is inordinately difficult to define a variadic function whose parameters
  have a fixed type, and the signature of such a function does not clearly
  communicate that fixed type to readers.
- There is no procedural mechanism for iterating through a variadic parameter
  list. Although features like [fold expressions](https://en.cppreference.com/w/cpp/language/fold)
  cover some special cases, the only fully general way to express iteration
  is with recursion, which is often awkward, and results in more template
  instantiations.

There are a number of pending C++ standard proposals to address these issues,
and improve variadic templates in other ways, such as [P1306R1: Expansion Statements](http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2019/p1306r1.pdf),
[P1858R2: Generalized Pack Declaration and Usage](http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2020/p1858r2.html), and
[P2277R0: Packs Outside of Templates](http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2021/p2277r0.html).
P1858R2 has been especially influential for this proposal.

## Proposal

FIXME

## Details
FIXME throughout the following, clarify what parts apply to patterns.

### Statically sized arrays

`[T; N]` is the type of a statically-sized array of `N` objects of type `T`,
and `[T;]` is a pattern that matches the type `[T; N]` for any `N`. In
other words, it's a shorthand for `[T; _:! BigInt]`. FIXME Or maybe we should
treat it as an interface/constraint? See https://discord.com/channels/655572317891461132/969001583088123905/973681495891910716
Or maybe it's `[T; template _:! BigInt]` so that we can validly refactor it
to `[template N:! BigInt]([T; N])`? `auto` seems like it might be ambivalent
in a similar way, between `_:! Type` and `template _:! Type`.

The rest of the design
for statically-sized arrays is deferred to a separate proposal, because it's
not relevant to variadics.

### Pack expansions

`...,`, `...and`, and `...or` are prefix unary expression operators with the
same precedence as the corresponding non-pack operators. They are single tokens,
and may not contain internal whitespace. `...{` is an opening delimiter that
forms a balanced pair with `}`, and can contain any number of `;`-delimited
statements. An AST rooted at any of these operations is called a _pack
expansion_. A pack expansion cannot occur within another pack expansion,
although we may relax this restriction in the future.

### Packs

FIXME reassess if we still need pack values, once we have description of
pattern matching. Discussion of pack-type variables may be another reason we
need it. Even if we need them for that, we may not need them to be a separate
category of values; they could be tuple values, with different semantics imposed
by the type.

A _pack type_ is a kind of product type, like a tuple, and a value of a pack
type is called a _pack_. A pack type is itself a pack whose elements are types,
in the same way that a tuple type is itself a tuple whose elements are types. A
pack cannot be a sub-value of any other kind of value, although we intend to
support nested packs in the future. A name can be bound to a pack value, but
only if the name is local to a function.

Pack values are created using the `[:]` operator, which is a prefix unary
operator with the same precedence as unary `*`. If its operand is a tuple or
statically-sized array, its result is a pack containing the same elements, with
the same element types. For example, `[:]("a", "b", "c")` is a pack whose type
is `[:](String, String, String)`, and whose elements are `"a"`, `"b"`, and
`"c"`. `[:]` can also be applied to a statically-sized array type: `[:][T; N]`
is a pack type with `N` elements that are all `T`. Thus, the type of
`[:]("a", "b", "c")` could also be written `[:][String; 3]`. A pack type
which can be written this way is called a _homogeneous_ pack type.

There is no Carbon syntax for indexing into a pack. When we need to refer to
pack elements as part of specifying their semantics, we will use the notation
P<sub>k</sub> to refer to the kth element of the pack P.

In a pattern context, the `[:]` operator can also be applied to a name binding
pattern that has a tuple or statically-sized array type. FIXME find better home
for this?

The operand of `[:]` cannot contain a `[:]` operator, although we intend to
support this in the future as part of supporting nested packs. FIXME better
home for this too?

FIXME can nesting restrictions be evaded using deduced parameters, which are
semantically but not syntactically nested within the pattern?

FIXME I think we need to allow packs within other types in patterns, to
support cases like `vectors: Vector([:]ElementTypes)`, where `ElementTypes` is a deduced parameter
with type `[Type;]`.

### Expression and statement semantics

The expression and statement semantics of packs and pack expansions will
be specified in terms of a procedure for rewriting ASTs containing those tokens
to equivalent ASTs that do not. These rewrites take place at the same time as
monomorphization of generic functions, and in particular they take place after
name resolution and typechecking. The rules for typechecking these operations
will be explained in a subsequent section.

A usage of a `[:]` operator, or of a name that is bound to a variable with a
pack type, is called an _expansion argument_, and can only occur inside a pack
expansion. Every pack expansion must have at least one argument, and all
arguments of a given pack expansion must have the same number of elements,
which we will call the _arity_ of the pack expansion.

A pack expansion with arity N is rewritten to N instances of the
expansion, where in the Kth instance, every expansion argument is replaced by
the Kth element of that argument. The details of the rewrite vary slightly
depending on the root node of the expansion:

- In a `...{` expansion, each instance uses a `{` in place of the opening `...{`.
- In a `...,`, `...and`, or `...or` expansion, each instance has the expansion
  operator removed, and the instances are joined using the corresponding non-expansion
  operator.

An `...and` expansion with arity 0 is rewritten to `true` and an `...or`
expansion with arity 0 is rewritten to `false` (the identities of the respective
logical operations). A `...,` expansion with arity 0 is rewritten to `,`, and
any `,` immediately before or after the expansion is removed.

Even though an `[:]` operator can only appear within a pack expansion, it is
possible to apply these rewrite semantics to an isolated subexpression that
contains a `[:]` operator but not the enclosing pack operator. In those cases,
the result of the rewrite is an expression headed by a `[:]` operator, which
means that its value is a pack. Thus, we can say that
`([:]("a", "b", "c"), [:](1, 2, 3))` rewrites to
`[:](("a", 1), ("b", 2), ("c", 3))`.

### Pattern semantics

Pack expansions can also appear in patterns. The semantics are chosen to follow
the general principle that pattern matching is the inverse of expression
evaluation, so for example if the pattern `(..., [:]x: String)` matches some
scrutinee value `s`, the expression `(..., [:]x)` should be equal to `s`.

These are run-time semantics, so the scrutinee expression is fully evaluated
(doesn't contain pack expansions).

*Tuple pattern:*

There can be no more than one `...,` pattern in a tuple pattern.

The N elements of the pattern before the `...,` expansion are matched with the
first N elements of the scrutinee, and the M elements of the pattern after the
`...,` expansion are matched with the last M elements of the scrutinee. If the
scrutinee does not have at least N + M elements, the pattern does not match. The
operand of the `...,` pattern is iteratively matched with each of the remaining
K scrutinee elements, if any. The iterations are numbered sequentially from 1 to
K, and the number of the current iteration is called the _pack index_.

Observe that the pattern is irrefutable for a given scrutinee type if:
- At most one `...,` pattern
- Scrutinee type is a tuple type with at least N + M elements
- First N and last M pattern elements are irrefutable given the corresponding
  scrutinee element types
- For all I from 1 to K, the `...,` operand pattern with pack index I is
  irrefutable given the type of the Ith scrutinee element. (Note that the
  typechecker doesn't know K)

*Identifier that names a deduced parameter*

The deduced parameter is unified with the scrutinee value.

Pattern is irrefutable for a given scrutinee type if:
- Unification does not reach a contradiction.

The typechecker must also ensure that the deduced value is a valid value of
the deduced parameter type.

*Name binding pattern:*

If the type subpattern contains a "free" use of the `[:]` operator, the
name binding pattern behaves at run-time as if it were prefixed with `[:]`
(see below).

Otherwise, if the pattern is inside a pack expansion, but not inside a
pack argument pattern, the program is ill-formed (because the name would
be bound multiple times).

Name binding patterns are always irrefutable at run-time. However, typechecker
must also ensure the bound value is a valid value of the binding type.

*`[:]` pattern:*

FIXME maybe distribute these to the sections above?

Let I be the current pack index.

- If the operand is a tuple pattern, it cannot contain a `...,` pattern, so all
  elements are explicit. The Ith element of the tuple pattern is matched with
  the scrutinee.
- If the operand is an identifier that names a deduced parameter, the parameter
  must be bound to a K-tuple value (although it may have array or pack _type_),
  and its Ith element is unified with the scrutinee.
- If the operand is a name binding pattern, the name must be bound to a K-tuple
  value, and its Ith element is bound to the scrutinee.
- If the operand is an expression pattern, match the Ith element of its value
  against the scrutinee.

Pattern is irrefutable for a given scrutinee type if:
- (tuple pattern) the Ith element of the tuple pattern is irrefutable for the
  scrutinee type
- (identifier that names a deduced parameter) unification does not reach a
  contradiction
- (name binding pattern) true
- (expression pattern) false

Typechecker must also ensure the bound value (if any) is a valid value of the
binding type.

### Typechecking

Typechecking takes place prior to the rewrites specified above, so we need to
specify how the non-rewritten code is typechecked.

FIXME: need to match up each argument with its expansion, and check that all
arguments have the same arity. Could be tricky if arity is symbolic.

#### Variadic types

A _variadic type_ is a pattern that matches types, and that contains at least
one expansion argument whose value is unknown. For example, `[:](_:! [Type;])`
and `(..., [:](_:! [Type;]))` are variadic types. Once the expansion argument
values are known, a variadic type can be evaluated, producing a non-variadic
type. An _unbound_ variadic type is a type that contains at least one unknown
expansion argument that is not nested within an expansion operator such as
`...,`. Evaluating a variadic type yields a pack type if and only if the
variadic type is unbound.

Consequently, the most specific type of a value is never variadic. Variadic
types are necessary only prior to monomorphization, as the types of certain
patterns and expressions that depend on generic parameters. In this respect,
variadic types are much like types that depend on generic parameters, such
as `[T:! Type](Optional(T))`. In particular, variadic types are typically
not types-of-types, but are more like the union of an infinite family of
types-of-values.

A variadic type is _homogeneous_ if every unknown expansion argument has a
type that is either:
- an array type,
- a non-variadic tuple type whose elements are all equal,
- a variadic tuple type of the form `(..., T)` where `T` is homogeneous, or
- a non-tuple homogeneous variadic type.

FUTURE WORK: relax the typechecking rules below to cover non-homogeneous
variadic types.

For any homogeneous variadic type, we can construct a corresponding
_representative type_ which is not variadic, for use in typechecking. A
representative for a type `T` is constructed as follows:

- If `T` is a homogeneous variadic type, the representative behaves like a type
  formed by replacing each unknown expansion argument with a representative of
  its type.
- If `T` is an array type, the representative behaves like a representative of
  its element type.
- If `T` is a non-variadic type-of-types, the representative is a unique
  archetype for `T` (as discussed in the generics design).
- Otherwise, if `T` is not variadic, the representative is `T` itself.

A representative that "behaves like" another type is still distinct from that
type. This permits us to perform the reverse transformation, from a type
that contains representatives back to a variadic type.

#### Expressions

**`[:]` expressions:**

If the type of its operand is a tuple
or statically-sized array type `T`, the expression has type `[:]T`. If its
operand _is_ a statically-sized array type `[T; N]`, the expression has type
`[:][U; N]`, where `U` is the type of `T` (FIXME do we need this?).

**`...and` and `...or` expressions:**

An `...and` or `...or` expression has type `bool`. Its operand must have type
`[:][bool;]` or `[:][bool; N]` for some constant `N`.

**Tuple literals and the `...,` operator:**

A tuple literal consists of some nonnegative number of leading elements,
optionally followed by a single use of the `...,` operator, whose operand is
called the _expansion element_, and then some nonnegative number of trailing
elements. The type of a tuple literal `L` is another tuple literal, whose
leading and trailing elements are the types of the leading and trailing elements
of `L`, and whose expansion element is the type of the expansion element of `L`
(if any).

**Other expressions with variadic operands:**

Any other expression that has at least one operand with a variadic type is
typechecked as follows: first, we replace each variadic type with a
corresponding non-variadic representative type, and perform name resolution and
typechecking in terms of those types. Then, if the resulting type for the whole
expression contains any representative types, they are replaced with the
original types they represented. Finally, if any of the variadic types were
unbound, and the type of the whole expression would be a non-variadic type `T`,
it is instead given the type `[:][T;]`.

FIXME: That last step feels super
ad-hoc, but without it, cases like `(..., vectors.Size())` don't come out right.
But *with* it, cases like `...{ len += vectors.Size(); }` don't come out right.
Maybe we need to distinguish expression from statement expansions? Maybe
allow expansion with no arguments once we've determined the (symbolic) arity?
Maybe we give up on assigning a type to subexpressions inside an expansion? (Ick)

#### Patterns

Typechecking for a pattern matching operation proceeds in three phases:

1. The pattern is typechecked, and assigned a type.
2. The scrutinee expression is typechecked, and assigned a type.
3. The scrutinee type is checked against the pattern type.

If the pattern appears in a context that requires it to be irrefutable, such
as the parameter list of a function declaration, phase 3 ensures that
the pattern can match _any_ possible value of the scrutinee expression.
Otherwise, it ensures that the pattern can match _some_ possible value of
the scrutinee expression. For simplicity, this proposal will focus on the
rules for the first case, since it's by far the most important for variadics.

FUTURE WORK: specify rules for refutable matching of variadics, e.g. to support
C++-style recursive variadics.

This section will focus on phase 3. Phase 2 was described above, and phase 1
is generally quite mechanical, even when the pattern is variadic, so I
will not describe it in detail here. By way of illustration, the type of
`(..., [:]params: [i64;])` is `(..., [:][i64;])`, and the type of
`[ElementTypes:! [Type;]](..., vectors: Vector([:]ElementTypes))` is
`[ElementTypes:! [Type;]](..., Vector([:]ElementTypes))`.

If the pattern type has any deduced parameters, phase 3 is responsible for
deducing them. However, when the scrutinee occurs in a generic context, the
results of this deduction may not be concrete values, but rather symbolic
expressions in terms of the unknown parameters of the scrutinee. For
example:

```
fn F[T:! Type](T: arg);

fn G[U:! Type](Optional(U): arg) {
  F(arg);
}
```

When typechecking the expression `F(arg)`, phase 3 deduces that at that
callsite, `T` is equal to `Optional(U)`, even though it cannot yet deduce the
value of `U`.

A variadic pattern will usually have one or more deduced parameters whose type
is an array of unknown size. The size can be deduced, but if the scrutinee
occurs in a generic context, that deduced size may itself be a symbolic
expression rather than a concrete number. As a result, the deduced "value" of
such an array can be fairly complex. In the most general case, it consists of
some number of leading elements, some number of trailing elements, and an
"expansion element" representing all of the other elements. Each of these
elements can be a symbolic expression in terms of the unknown parameters of the
scrutinee, and also in terms of a special "expansion index" symbolic variable,
which represents the position of the current element in the symbolic array.

If the pattern type is not a variadic type, pattern matching follows the
usual non-variadic rules (which usually means that a variadic scrutinee type
would be rejected), so we will focus on the semantics of phase 3 when the
pattern type is a variadic type. Variadic types are themselves patterns, so
we will break this down into cases based on the possible forms of a pattern.


NOTE TO SELF Test cases:
A:
```
// OK
fn F[Ts:! [Type;]](..., [:]args: Ts) -> i32;

fn G[U:! Type; Us:! [Type;]](arg: Vector(U), ..., [:]args: Vector(Us)) -> i32 {
  return F(arg, ..., args);
}
```

B:
```
// OK (but maybe too complex to support?)
fn F[Ts:! [Type;], Ts:! [Type;]](arg1: (..., [:]Ts), arg2: (..., [:]Ts)) -> i32;

fn G[Vs:! [Type;]](..., [:]args: Vector(Vs)) -> i32 {
  return F((..., args), (..., args));
}
```

```
fn F[Ts:! [Type;]](arg1: (..., [:]Ts), arg2: (..., [:]Ts)) -> i32;

fn G[Vs:! [Type;], Ws:! [Type;]](arg1: (..., [:]Vector(Vs)), arg2: (..., [:]Vector(Ws))) -> i32 {
  // Error: Vs might be different from Ws.
  return F(arg1, arg2);
}
```

C:
```
fn F[Ts:! [Type;]](arg: i32, ..., [:]args: Ts) -> i32;

fn G[Us:! [Type;]](..., [:]args: Us, arg: i32) -> i32 {
  // Error: first element of args might not be i32
  return F(..., args, arg);
}

fn H() {
  G(1);
}
```

*Variadic tuple type*

A variadic tuple pattern consists of M leading elements, a variadic element
headed by the `...,` operator, and N trailing elements. The scrutinee type may
be an ordinary tuple type or a variadic tuple type. If the scrutinee type is an
ordinary tuple type, it must have at least M+N elements. If the scrutinee type
is a variadic tuple type, it must have at least M leading elements and at least
N trailing elements.

The first M elements of the scrutinee are checked against the leading
elements of the pattern type, the last N elements are checked against the
trailing elements of the pattern type, and the remaining scrutinee elements
(if any) are iteratively checked against the variadic element.

*Parameterized class type*

If the pattern and scrutinee types name the same parameterized class,
the scrutinee type argument list is checked against the pattern type argument
list. Otherwise, typechecking fails.

*Identifier that names a deduced parameter*

FIXME

*Name binding pattern:*

FIXME

*`[:]` pattern:*

FIXME






#### Statements

FIXME

## Rationale

TODO: How does this proposal effectively advance Carbon's goals? Rather than
re-stating the full motivation, this should connect that motivation back to
Carbon's stated goals and principles. This may evolve during review. Use links
to appropriate sections of [`/docs/project/goals.md`](/docs/project/goals.md),
and/or to documents in [`/docs/project/principles`](/docs/project/principles).
For example:

-   [Community and culture](/docs/project/goals.md#community-and-culture)
-   [Language tools and ecosystem](/docs/project/goals.md#language-tools-and-ecosystem)
-   [Performance-critical software](/docs/project/goals.md#performance-critical-software)
-   [Software and language evolution](/docs/project/goals.md#software-and-language-evolution)
-   [Code that is easy to read, understand, and write](/docs/project/goals.md#code-that-is-easy-to-read-understand-and-write)
-   [Practical safety and testing mechanisms](/docs/project/goals.md#practical-safety-and-testing-mechanisms)
-   [Fast and scalable development](/docs/project/goals.md#fast-and-scalable-development)
-   [Modern OS platforms, hardware architectures, and environments](/docs/project/goals.md#modern-os-platforms-hardware-architectures-and-environments)
-   [Interoperability with and migration from existing C++ code](/docs/project/goals.md#interoperability-with-and-migration-from-existing-c-code)

## Alternatives considered

TODO: What alternative solutions have you considered?

FIXME multiple `...,` patterns in a tuple