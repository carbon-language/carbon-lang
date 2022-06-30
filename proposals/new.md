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

### Rewrite semantics

The semantics of packs and pack expansions will
be specified in terms of a procedure for rewriting ASTs containing those tokens
to equivalent ASTs that do not. These rewrites take place at the same time as
monomorphization of generic functions, and in particular they take place after
name resolution and typechecking. The rules for typechecking these operations
will be explained in a subsequent section. In certain cases involving pattern
matching, the rewritten AST is permitted to have a form that would not be valid for "raw"
Carbon code directly written by a user, but *only* in the ways that are explicitly
identified below.

A usage of a `[:]` operator, or of a name that is bound to a variable with a
pack type, is called an _expansion argument_, and can only occur inside a pack
expansion. Every pack expansion must have at least one argument, and all
arguments of a given pack expansion must have the same number of elements,
which we will call the _arity_ of the pack expansion. Note that the number
of elements in an argument is determined during typechecking, and in a
pattern context it may be deduced from the scrutinee type, as discussed in the
section on typechecking below. FIXME previous sentence may be incorrect for
function parameter patterns.

A pack expansion with arity N is generally rewritten to N instances of the
expansion, where in the Kth instance, every expansion argument is replaced by
the Kth element of that argument. When the argument consists of the `[:]`
operator applied to a name binding pattern, the "Kth element" is a kind of
"sub-binding": a pattern that sets the value of the Kth element of the binding.
Note that there is no Carbon syntax for such a sub-binding; this is one of
the cases where the rewritten code would not be valid as "raw" Carbon code.

When the argument consists of the `[:]` operator applied to an identifier
expression I, the "Kth element" is of course "I `[` K `]`", but there's an
important nuance. When I names a deduced parameter of the current pattern, it is
unclear whether an expression like "I `[` K `]`" can contribute to deducing the
value of I, or even whether that expression is permitted at all. In the
rewritten code, the answer must be "yes", so this may be another case where the
rewritten code would not be valid as "raw" Carbon code.

The details of the rewrite also vary slightly depending on the root node of the expansion:

- In a `...{` expansion, each instance uses a `{` in place of the opening `...{`.
- In a `...,`, `...and`, or `...or` expansion, each instance has the expansion
  operator removed, and the instances are joined using the corresponding non-expansion
  operator.

An `...and` expansion with arity 0 is rewritten to `true` and an `...or`
expansion with arity 0 is rewritten to `false` (the identities of the respective
logical operations). A `...,` expansion with arity 0 is rewritten to `,`, and
any `,` immediately before or after the expansion is removed.

Prior to rewriting, a pattern cannot contain multiple bindings with the same
name. However, if a pack expansion pattern contains a binding that is not
part of a pack argument, the rewritten pattern may contain multiple bindings
with that name. This is the other case where the rewritten code would not
be valid as "raw" Carbon code. In this case, the binding will bind to a pack
consisting of the values bound to each instance of the binding.

FIXME split out subsection on pattern rewriting?

ALTERNATIVE: pattern semantics expressed without rewriting

These are run-time semantics, so scrutinee expression is fully evaluated
(doesn't contain packs).

*Tuple pattern:*

There can be no more than one `...,` pattern in a tuple pattern.

The N elements of the pattern before the `...,` expansion are matched with the
first N elements of the scrutinee, and the M elements of the pattern after the
`...,` expansion are matched with the last M elements of the scrutinee. The
remaining scrutinee elements, if any, are matched with the `...,` pattern.
If the scrutinee does not have at least N + M elements, the pattern does not
match.

Pattern is irrefutable for a given scrutinee type if:
- At most one `...,` pattern
- Scrutinee type is a tuple type with at least N + M elements
- First N and last M pattern elements are irrefutable given the corresponding
  scrutinee element types
- `...,` subpattern (if any) is irrefutable given the corresponding scrutinee
  element types

*`...,` pattern:*

Iteratively match operand pattern against each each of K scrutinee values, from
first to last, with the _pack index_ ranging from 1 to K.

Pattern is irrefutable for a given scrutinee type if:
- For all I from 1 to K, the operand pattern with pack index I is irrefutable
  given the type of the Ith scrutinee element. (Note that the typechecker
  doesn't know K)

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
must also ensure the bound value can be a valid value of the binding type.

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

#### Expressions

A `[:]` expression always has a pack type. If the type of its operand is a tuple
or statically-sized array type `T`, the expression has type `[:]T`. If its
operand _is_ a statically-sized array type `[T; N]`, the expression has type
`[:][U; N]`, where `U` is the type of `T`.

An `...and` or `...or` expression has type `bool`. Its operand must have type
`[:][bool; N]` for some constant `N`.

`...,` expressions will be addressed in the subsection below.

Any other expression AST node that has at least one pack-type operand is
typechecked as follows: without loss of generality, let Ty(A, B, C, X, Y) be the
typechecking function for this kind of node, which takes the types of the child
nodes and returns the type of the parent node, or reports that the program is
ill-formed. Also without loss of generality, let A, B, and C be pack types, and
let X and Y be non-pack types. If A, B and C do not all have the same size, the
program is ill-formed; otherwise, let N be that size. Ty(A, B, C, X, Y) will be
a pack type whose elements are Ty(A<sub>1</sub>, B<sub>1</sub>, C<sub>1</sub>,
X, Y), Ty(A<sub>2</sub>, B<sub>2</sub>, C<sub>2</sub>, X, Y), ...,
Ty(A<sub>N</sub>, B<sub>N</sub>, C<sub>N</sub>, X, Y).

FIXME maybe we can simplify the above by allowing `[:]` to be embedded within
the type, as with patterns? OTOH maybe not; that's more of a type _expression_
than a canonical type. But need to be able to handle ID-expressions with
non-canonical types.

Notice one important special case of that rule: if A, B, and C are homogeneous
pack types `[:][` T `;` N `]`, `[:][` U `;` N `]`, and `[:][` V `;` N `]`,
then Ty(A, B, C, X, Y) = `[:][` Ty(T, U, V, X, Y) `;` N `]`. 

##### `...,` expressions and value list types

To specify the typechecking of `...,` expressions in a rewrite-free way, we need
to specify the typechecking of `,` expressions.

A _value list type_ is a kind of product type that can be thought of as
representing a comma-separated list. A value list type can never be an element
type of another type (including another value list type). Value list types are
purely internal to the typechecking algorithm; there are no actual values of
any value list type, and there is no way of naming or referring to a value list
type in Carbon code.

`,` is an infix binary operator. If both operands have value list types,
the type of the `,` expression is a value list type formed by concatenating the
element types of its operands. If either operand has a non-value-list type T, it
is treated as having a value list type with a single element type T.

The operand of a `...,` expression must have a pack type, and the type of the
`...,` expression is a value list type with the same element types as the
operand pack type.

When the contents of a `()` expression have a value list type, the expression
has a tuple type with the same element types as the value list type.

Only `,` and `...,` expressions can have value list types. Conversely,
a value-list-typed expression can only be used as an operand of `,`, or as the
contents of a `()` expression.

#### Patterns

Typechecking for a pattern matching operation proceeds in three phases:

1. The pattern is typechecked, and assigned a type.
2. The scrutinee expression is typechecked, and assigned a type.
3. The scrutinee type is checked against the pattern type.

If the pattern appears in a context that requires it to be irrefutable, such
as the parameter list of a function declaration, phase 3 ensures that
the pattern can match _any_ possible value of the scrutinee expression.
Otherwise, it ensures that the pattern can match _some_ possible value of
the scrutinee expression. FIXME is that all it's responsible for?

To support this, the type of a pattern can itself be a pattern. When the scrutinee value is
known, the type pattern is matched against the scrutinee type, which binds values to
the names in the type pattern (i.e. deduces them).

TODO need to also allow for implicit conversions, but seems out of scope.

NOTE TO SELF maybe the type of `vectors: Vector([:]ElementType)` is `Vector([:]ElementType)`,
and that's a _variadic pack type_, which will become an ordinary pack type at
monomorphization time. By the same token, `..., Vector([:]ElementType)` is a
_variadic value list type_.

NOTE TO SELF Overall approach: check expression type against pattern type.
Cases will often be recursive. Checking can fail fast, but also builds up a
set of type equalities which are then solved by unification. Equalities will
sometimes refer to elements of a type array, possibly including a
"variadic element". Typecheck algorithm knows the current pack index in the
pattern, if applicable, but it can be an offset from the start, an offset from
the end, or a range whose bounds are from-start and from-end. The three are
always disjoint.

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

*Tuple type*

General case: one element of the tuple pattern type is a variadic value list
type, and one element of the scrutinee type is a variadic value list type.
Scrutinee type must have at least as many non-variadic leading and trailing
elements as the pattern type. Each leading and trailing element of the pattern
type is checked against the corresponding element of the scrutinee type in the
usual way. The remaining scrutinee elements are checked against the variadic
value list's element type (which is a variadic pack type). FUTURE WORK:
relax/generalize these rules if the match doesn't have to succeed, e.g. to
support C++-style recursively consuming parameters using `match` (and eventually
overloading?). TODO: This doesn't support example C above. Is that OK?

FIXME does anything happen regarding deduction at this point? How can it, when
the types may not be concrete?

FIXME: specify special cases where one of the two isn't variadic.

*Parameterized class type*

If the parameterized class is the same, recurse into argument list. Otherwise,
fail.

*Scrutinee is expansion argument*

Let I be the current pack index of the pattern. The Ith element of the scrutinee
type is unified with the Ith element of the pattern type







The type of a tuple pattern is a tuple of the types of the subpatterns.
If more than one subpattern is a `...,` pattern, the program is ill-formed.
The type of a `...,` pattern whose operand has type T is "`...,` T".
The type of a `[:]` pattern whose operand has type T is "`[:]` T".

FIXME need to address the type of _usages_ of bindings that involve packs, like
`vectors: Vector([:]ElementType)` (also, is that even a pack type? if so, does that mean it's a pack value? if not,
what is it?). Possibly belongs in statement section. Possibly typechecking
proceeds by substituting an archetype for the deduced parameter? Raises
question of what kinds of type patterns can have archetypes. How about when
we bind a name to the size, does that change typechecking? e.g. is this valid?
```
fn F[N:! BigInt, ElementType:! [Type; N]](..., vectors: Vector([:]ElementType))
    -> Vector(ElementType[N-1]) {
  var seq: [i32; N];
  // Loop to fill seq with 0, 1, 2, ...
  ...{
    if ([:]seq == N - 1) {
      return vectors;
    }
  }
}
```
I think we have to disallow this, but how? Seems like it might hinge on the
rules for :! bindings. @zygoloid suggests that :! bindings aren't usable
in a template :! context, and the size parameter of an array type might be
such a context. That would disallow the declaration of `seq`.

@josh11b reminds me that `:!` doesn't mean "caller knows value at compile time",
only "_somebody_ knows the value at compile time". So probably `N` has to be
a template parameter to be used as a subscript, even in the signature.
Note that because it's used as the size of the argument list, it _will_ be
known by the immediate caller, unless the immediate caller is forming the
argument list using pack expansion.

How about the non-variadic case? e.g. `fn F[T:! Type](vec: Vector(T))`.
How do we describe the type of this pattern? What's involved in typechecking it?
Seems like the type is `[T:! Type](Vector(T),)`. Interesting that
the part after the deduced params is an expression -- does that generalize?
It's not an arbitrary expression, though, because it needs to support deduction.
How about `fn F[T:! Type](vec: Vector(T), x: X(T))` where `X` is a function?
Could work if `T` were a template binding (deduce through `vec`, then pass result
of deduction to `X`), but as-is we have to disallow because we don't have a
type for `x` in the function body. @zygoloid suggests the high level rule is
something like "symbolic evaluation of non-template-dependent types is required
to succeed".


Consider also `fn SumInts(..., [:]params: [i64;]) -> i64` -- what's the type
of `params` in the function body? Something like `[N:! BigInt]([i64; N],)`
or `[template N:! BigInt]([i64; N],)`? Neither is quite right -- `N` needs to
be template in order to be an array size, but we don't want to force it to be
known at typechecking time. Maybe array size doesn't need to be template context?
No, @josh11b points out it has to be, so we can check that it's not negative.
But sounds like we may allow generic values to be used in template contexts,
in which case either one could work?

We could say the type of `params` is just `[i64;]`. We could think of that as an
existential type (i.e. a type-erased type like std::function). Very similar to
`RuntimeSizedArray(i64)` from my array sketch thread, although that discards
the fact that size is fixed at monomorphization time, and probably wouldn't
support `Size()`. Alternatively, we could
think of it as an interface, but it's an oddball because it's a closed interface
(only `[i64; N]` can implement it), and the implementation shouldn't involve
any witness tables, at least for indexing. But maybe for size?

(Assuming binding doesn't depend on a template parameter:)
Anyway, when typechecking a usage of a binding, if the binding type is a
symbolic value, we replace each symbolic variable (in declaration order)
with a representative value, and then fully evaluate the resulting expression to
obtain the type of the usage.
The representative value is generated based on the symbolic variable's type:
- If the type is `[T;]`, the representative is `(R,)`, where `R` is
  a unique representative value of `T`.
- If the type is an interface or named constraint, the representative is a
  unique archetype of the type.

This doesn't quite work -- it would let us pass a pack of Ts into a call that
accepts only a single T. So at minimum we need to track the "packness" as
part of typechecking.
We need to retain enough information
to typecheck the point of use (which may itself be a variadic call) in a way that
guarantees that the expanded callee pattern will match, and ideally drive deduction of the pack
size for the callee (in order to make the semantics coherent). The last part
doesn't work if we completely discard the size, but we don't necessarily need it.
But can we achieve the typechecking guarantees without the size?

SCRATCHPAD

Typechecking body of
```
fn Zip[ElementTypes:! [Type;]]
      (..., vectors: Vector([:]ElementTypes))
      -> Vector((..., [:]ElementTypes)) {
  var iters: auto = (..., vectors.Begin());
  var result: Vector((..., [:]ElementTypes));
  while (...and [:]iters != vectors.End()) {
    result.push_back((..., *[:]iters));
    ...{ ([:]iters)++; }
  }
  return result;
}
```

Maybe typecheck as if `ElementTypes` had single element, chosen as archetype?
Might not be enough, because e.g. consider difference between these:

```
fn Foo[ElementTypes:! [Type;]]((..., t1: [:]ElementTypes), (..., t2: [:]ElementTypes));
fn Bar[ElementTypes1:! [Type;], ElementTypes2:! [Type;]]
  ((..., t1: [:]ElementTypes1), (..., t2: [:]ElementTypes2));
```
The body of `Foo` can rely on matching types and arities, but the body of `Bar`
cannot. Maybe that's handled by the fact that separate bindings create separate
archetypes, but is that fundamental or did we just get lucky with this example?
I think it might be fundamental.

What if N is specified, and is a known constant? What if it's deduced? Maybe that
doesn't change anything? Feels weird, though. How about something like
```
fn Zip[N:! BigInt, ElementTypes1:! [Type; N], ElementTypes2:! [Type; N]]
  ((..., t1: [:]ElementTypes1), (..., t2: [:]ElementTypes2))
  -> (..., ([:]ElementTypes1, [:]ElementTypes2));
```
where body processes them pairwise? I think we make N an "archetype" of BigInt,
i.e. a unique value which is the same as itself but not the same as any other
BigInt (kind of like a transfinite value?). Then we can enforce the usual rule
that expansion argument arities must match. But that doesn't quite square with
the idea of typechecking as-if single element. Also, what about operations
besides equality/sameness? Can we apply `<` to N? Seems like we need to limit
to operations that can be evaluated symbolically.

Actually, might be better to think in terms of _symbolic values_ rather than
archetypes - avoids overfocus on types, and avoids mental model of materializing
a concrete type, which doesn't generalize very well.

FIXME Can we even allow `N` to be a non-template parameter here? Maybe, but
it's tricky: the typesystem has to guarantee at the callsite that the two
arrays/packs have equal sizes, without tracking the value they're equal to.
Not clear if there are enough use cases to justify the complexity; maybe
skip for now?


A _symbolic constant_ can be:
- A manifest constant (roughly, the result of a constant expression).
- An id-expression that names a `:!` binding.
- A tuple of symbolic constants.
- A type constructor whose argument tuple is a symbolic constant.
- An array type whose element type and size are symbolic constants.
- A pack whose elements are symbolic constants.
- (probably more cases?)

The type of a pattern can be a symbolic constant (but the type of an expression
is always a manifest constant, I think).

The _element count_ of a symbolic constant is a symbolic constant, defined as
follows:
- The element count of a tuple is the number of elements.
- The element count of an array type is its size.
- The element count of an id-expression that names a `:!` binding is the
  element count of its type.
- Otherwise, undefined.
(FIXME too much equivocating between values and types?)

The operand of `[:]` must be a symbolic constant expression with a well-defined
element count.

Let T be the type of x in "`[:]` x". The type of "`[:]` x" is defined as
follows:
- If T is a manifest constant, the type of "`[:]` x" is the pack type that results
  from evaluating "`[:]` T"
- If T is an id-expression that names a `:!` binding, the type of "`[:]` x" is
  FIXME
- If T is a tuple, the type of "`[:]` x" is the symbolic constant that results
  from evaluating "`[:]` T".
- If T is a type constructor, the program is ill-formed.
- If T is an array type, the type of "`[:]` x" is the element type. **In other
  words, we treat it as a single element.** FIXME ok for typechecking, maybe
  not ok for constant evaluation?
- If T is a pack type, the program is ill-formed.

If x is a symbolic constant, the value of "`[:]` x" is defined as follows:
- If x is a manifest constant, the value of "`[:]` x" is the result of
  evaluating it.
- If x is an id-expression that names a `:!` binding, the value of "`[:]` x"
  is FIXME

Maybe I'm overthinking this? Maybe we just say symbolic values are left
uninterpreted, but we sub in an archetype/representative value for typechecking
usages.

END SCRATCHPAD


So what exactly triggers this substitution? Would we also do it for e.g.
`fn Baz[ElementTypes:! [Type;]](arg: ElementTypes)`? I think maybe yes...
How about `fn Zip(..., vectors: Vector([:](_:! [Type;])))`.
Need that to still trigger, so it's not about deduced parameters, it's about
a binding whose rhs is a type-of-type, or array-of-type-of-type pattern.



FIXME: "matched with" terminology -- I think this whole section is defining what that means

A tuple type pattern consists of a pair of parentheses around a series of
comma-separated subpatterns. If none of the subpatterns is a `...,` pattern, the
scrutinee type must be a tuple type with the same number of elements as the pattern,
and each subpattern is matched with the corresponding element type of
the scrutinee.

Otherwise, let M be the number of subpatterns before the `...,` subpattern, let N be
the number of subpatterns after it (N is 0 if there is no `...,` subpattern),
and let S be the number of elements in the scrutinee tuple type (which must be at least
M + N). The first M subpatterns and the last N subpatterns are matched with
the corresponding elements of the scrutinee type. The operand of
the `...,` pattern is matched with a pack type whose elements are the remaining
S - (M + N) elements of the scrutinee type.

When the type pattern has the form "`[:]` T", the scrutinee type must be a
pack type. If T has the form `[` U `;` N `]`, the scrutinee type, U is matched
with each element of the scrutinee type,
and N is matched with the number of elements in the scrutinee type.
This rule also applies to type patterns of the form `[:][` T `;]`, which as
previously discussed is equivalent to `[:][` T `; _:! BigInt]`. If T is a
tuple type pattern, it is matched with a tuple type with the same elements
as the scrutinee pack type.
FIXME: should the above be moved to the rewrite section?

Otherwise, when the scrutinee type is a pack type, each element type is matched
with a rewritten form of the type pattern. When matching the Kth element type
of the scrutinee, each pack argument in the type pattern is replaced with
the Kth element of the pack argument, as described earlier.

FIXME Consider `Vector(Optional([:]ElementType))`. Can we say what the type of
`Optional([:]ElementType)` is, within that pattern?

#### Statements

FIXME

## Scratchpad

```
fn F(..., arg: [:][i64;]) {
  // Should work, feels unwieldy (but maybe that's OK?)
  let (..., arg_copy: [:][i64;]) = (..., arg);

  // Dubious: there's no enclosing expansion for `[:]` or `arg` to pair with,
  // plus seems to create ambiguity with next example
  let arg_copy: [:][i64;] = arg;

  // This needs to work
  ...{
    let arg_element: i64 = arg;
  }

  // This should still work, but there needs to be some other pack operand to
  // "drive" the `...{}`.
  ...{
    let (..., arg_copy: [:][i64;]) = (..., arg);
  }
}
```

Maybe full-expressions and full-patterns never have pack type, even though
bindings can? i.e. if an expression is used as a full expression, it is
coerced to the element type(s)



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