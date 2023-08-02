# Variadics

<!--
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

<!-- toc -->

## Table of contents

-   [Statically sized arrays](#statically-sized-arrays)
-   [Basics](#basics)
-   [Syntactic semantics](#syntactic-semantics)
-   [Reified semantics and pack values](#reified-semantics-and-pack-values)
-   [Pattern semantics](#pattern-semantics)
-   [Typechecking](#typechecking)
    -   [Pattern matching](#pattern-matching)
        -   [Identifying potential matchings](#identifying-potential-matchings)
        -   [The type-checking algorithm](#the-type-checking-algorithm)
-   [Alternatives considered](#alternatives-considered)
-   [References](#references)

<!-- tocstop -->

## Statically sized arrays

`[T; N]` is the type of a statically-sized array of `N` objects of type `T`, and
`[T;]` is a symbolic type expression (like `auto`) that deduces to the type
`[T; N]` for some `N`. Consequently, for purposes of symbolic evaluation and
typechecking, we will assume that all statically-sized array types have a size,
although that size may be a symbolic value rather than a constant.

A tuple of `N` elements of type `T` can be implicitly converted to `[T; N]`, and
an array of type `[T; N]` can be implicitly converted to a tuple of `N` elements
of type `T`. These conversions are combined transitively with the implicit
conversions among tuples, and implicit conversions from tuples of types to tuple
types. As a result, if `T` is a type of types, `[T; N]` is usable as a tuple
type.

We do not support deducing the element type of an array, as in
`fn F[T:! type](array: [T;]);`, because there is no way to deduce the type of an
array of size 0, and the variadic use cases we are focusing on do not give the
caller a way to bypass deduction with an explicit cast.

> **TODO:** Complete the design of statically-sized arrays, and move it to a
> separate document.

## Basics

This example illustrates many of the key concepts of variadics:

```carbon
// Takes an arbitrary number of vectors with arbitrary element types, and
// returns a vector of tuples where the i'th element of the vector is
// a tuple of the i'th elements of the input vectors.
fn Zip[ElementTypes:! [type;]]
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

`...,`, `...and`, and `...or` are prefix unary expression operators with the
same precedence as `,`, `and`, and `or`, respectively. They are single tokens,
and may not contain internal whitespace. `...{` is an opening delimiter that
forms a balanced pair with `}`, and can contain any number of statements. An AST
rooted at any of these operations is called a _pack expansion_.

A pack expansion must contain one or more _expansion arguments_. These are
usually marked by the `[:]` operator, but can also be unmarked usages of names
that have `[:]` in the type (called pack types). For example, in the body of
`Zip`, the loop condition `...and [:]iters != vectors.End()` is a pack expansion
with two arguments: `[:]iters` is explicitly marked as an argument, and
`vectors` is an argument because its type is `Vector([:]ElementTypes)`.

The _arity_ of an expansion argument is a compile-time value representing the
number of elements it evaluates to. Every pack expansion must contain at least
one expansion argument, and all arguments of a given expansion must have the
same arity (which we will also refer to as the arity of the expansion). If any
expansion argument is a non-expression pattern, the entire pack expansion is a
pattern. In particular, this means that the expansion arguments of a `...{`
expansion must be expressions.

> **Open question:** Is it possible to drop that requirement, and support code
> like `let [:]x: ElementTypes = *[:]iters;` within a `...{` expansion?

Pack expansions can be nested only if the inner expansion is within one of the
outer expansion's arguments. For example,
`(..., [:]vectors: (..., Vector([:]ElementTypes)))`, which is an alternative way
of writing the parameter list of `Zip` to avoid giving `vectors` a pack type. We
may further relax the restriction on nesting in the future.

There are two complementary models for the meaning of pack expansions, the
_syntactic_ model and the _reified_ model. In the syntactic model, we treat a
pack expansion as being rewritten during monomorphization as a series of copies
of the pack expansion, where in each copy, each expansion argument is replaced
with one of its elements. In the reified model, we define the semantics in terms
of _pack values_, which are created by the `[:]` operator and consumed by
`...,`, `...and`, `...or`, and `...{`, and we generalize most non-variadic
operations to apply element-wise to packs.

Both models work well for run-time expression evaluation, and should be
equivalent in that context. The syntactic model provides a unified description
of run-time expression and statement semantics, but is difficult to apply to
symbolic computation, such as type checking and pattern matching. On the other
hand, the reified model works well for symbolic computation and expression
evaluation, but does not naturally extend to statement semantics.

We will rely primarily on the reified model, with two exceptions:

-   We will use the syntactic model to describe the run-time semantics of `...{`
    expansions, because it's not clear how to describe the semantics of
    statements using the reified model, and it's difficult to bridge between the
    two models within a single pack expansion.
-   We will sometimes use the syntactic model when describing the run-time
    semantics of expressions, because expressions inside `...{` expansions must
    use that model, and because run-time expressions in other expansions will
    likely be implemented using that model.

Type-checking and pattern-matching will always use the reified model, because
they are inherently symbolic.

> **Open question:** Can we unify these two models? In particular, it seems
> possible that we could generalize the reified model to cover statements, such
> that the syntactic model can be derived from it as a special case. The "pack
> index" introduced below may provide a path: the syntactic model seems to
> consist of instantiating a series of copies of the code, where the symbolic
> pack index is replaced with successive integer constants.

## Syntactic semantics

In the syntactic model, the semantics of packs and pack expansions are specified
in terms of a procedure for rewriting an AST rooted at `...{`, `...,`, `...and`,
or `...or` to an equivalent AST that does not. This rewrite takes place at the
same time as monomorphization of generic functions, which means that all names
are resolved, all typable entities have known types, and all symbolic constants
have known values.

A pack expansion with arity N is rewritten to N instances of the expansion,
where in the Kth instance, every expansion argument is replaced by the Kth
element of that argument. The details of the rewrite vary slightly depending on
the root node of the expansion:

-   In a `...{` expansion, each instance uses a `{` in place of the opening
    `...{`.
-   In a `...,`, `...and`, or `...or` expansion, each instance has the expansion
    operator removed, and the instances are joined using the corresponding
    non-expansion operator.

An `...and` expansion with arity 0 is rewritten to `true` and an `...or`
expansion with arity 0 is rewritten to `false` (the identities of the respective
logical operations). A `...,` expansion with arity 0 is rewritten to `,`, and
any `,` immediately before or after the expansion is removed.

## Reified semantics and pack values

We can intuitively think of packs as being much like tuples. For every tuple
value there is a corresponding pack value, and the other way around, but the two
values are not identical, because they support different operations. In
particular, there is no way to access a specific element of a pack, whereas
that's almost the only operation that tuples support.

We typically think of tuples as fixed-size sequences of elements, but in order
to support variadics, we need to be able to reason symbolically about packs and
tuples that contain subsequences of an indeterminate number of elements that
share a common structure. For example, in a function with deduced parameters
`[N:! SizeType, M:! SizeType, Ts:! [type; N], Us:! [type; M]]`, the type
`(..., [:]Ts, i32, ..., Optional([:]Us))` is a tuple whose elements are an
indeterminate number of types, followed by `i32`, followed by a different
indeterminate number of types that all have the form `Optional(U)` for some type
`U`. Consequently, we need to represent tuples and packs in a more general way
during symbolic evaluation, by saying that a pack or tuple consists of a
sequence of _pack components_.

A pack component consists of a value called the _representative_, and an arity,
both of which may be symbolic. As their name suggests, pack components cannot
occur on their own, but only as part of an enclosing pack (or tuple). Pack
components have no literal syntax in Carbon, but for purposes of illustration we
will use the notation `<V, N>` to represent a pack component with representative
V and arity N. There is a special symbolic variable called the _pack index_,
which can only be used in the representative of a pack component, and only as
the subscript of an indexing expression on a tuple or statically sized array.
The pack index is used solely as a placeholder, and never has a value. We will
use `$I` to represent it for purposes of illustration. _Expanding_ a component C
with arity N (where N is a known constant) means replacing it with N components
that have arity 1, where the representative of the k'th such component is a copy
of the representative of C, where every usage of the pack index is replaced with
k-1.

For example, continuing the earlier example, the expression
`(..., [:]Ts, i32, ..., Optional([:]Us))` symbolically evaluates to
`(<Ts[$I], N>, <i32, 1>, <Us[$I], M>)` (the language rules leading to this
result will be discussed below). If `N` is known to be 2, we can rewrite that
value by expanding the first component, yielding
`(<Ts[0], 1>, <Ts[1], 1>, <i32, 1>, <Us[$I], M>)`.

A pack component is _variadic_ if its arity is unknown, and _singular_ if its
arity is known to be 1 and does not refer to the pack index. In contexts where
all pack components are known to be singular, we will sometimes refer to them as
"elements". Pack values are always assumed to be normalized, meaning that every
component is either singular or variadic. This is always possible because if the
component's arity is known but not 1, we can expand it, which produces a
sequence of singular components by construction. The _shape_ of a pack is
another pack, found by replacing all representatives with `()` and leaving the
arities unchanged. A _pack type_ is the type of a pack, and can be represented
as a pack value whose representatives are types. The type of a pack is found by
replacing each of its representatives with the representative's type.

Even during symbolic evaluation, we need to maintain the structural equivalence
between packs and tuples, so in this more generalized model, a tuple value
consists of a sequence of pack components. Name bindings can have pack types,
but only if they are local to a single function. Note that by construction,
bindings with pack types always have an explicit `[:]` operator in their type
expression:

```carbon
fn F(array: [i32;]) {
  // ✅ Allowed: `pack` is local to `F`.
  let (..., pack: [:][i32;]) = (..., [:]array);

  // ❌ Forbidden: `auto` can't deduce a pack type.
  let (..., pack: auto) = (..., [:]array);
}

// ❌ Forbidden: `pack` is not function-local.
let (..., pack: [:][i32;]) = (1, 2, 3);

// ✅ Allowed: `tuple` doesn't have a pack type.
let tuple: (..., [:][i32;]) = (1, 2, 3);
```

Most operations that are defined on singular values are defined on packs as
well, with the following semantics: all operands that are packs must have the
same shape, and the result of the operation will itself have the same shape as
the operands. The representatives are computed by simultaneously iterating
through the input packs and applying the non-pack version of the operation: the
representative of the k'th output pack component is the result of replacing each
input pack with the representative of its k'th component, and evaluating the
resulting operation using the ordinary non-pack rules. Note that for these
purposes a tuple value does not behave like a pack value.

For example, if `x` is a pack with the value `<p, N>, <42, 1>`, and `y` is a
pack with the value `<q, N>, <10, 1>`, then we can evaluate `x + y + 1` as
follows: the output will be a pack whose shape is the same as the input packs,
namely `<(), N>, <(), 1>`. To find the first representative, we evaluate the
expression `x + y + 1`, but with the first representative of `x` in place of
`x`, and likewise for `y`, yielding `p + q + 1`. To find the second
representative, we perform the same substitution with the second representatives
of `x` and `y`, yielding `42 + 10 + 1` = `53`. Thus, `x + y + 1` evaluates to
`<p + q + 1, N>, <53, 1>`.

If a tuple literal does not contain `...,`, its evaluation follows the rules for
arbitrary operations described in the previous paragraph. If it does contain
`...,`, the result is a tuple value formed by iterating through the literal
elements:

-   If the element is headed by `...,`, we evaluate its operand to produce a
    pack, and append its components to the result tuple.
-   Otherwise, we evaluate the element expression to produce a value `V`, and
    append a singular component whose representative is `V`.

The `[:]` operator transforms a tuple into a pack, or an unknown tuple value
into a pack of unknown values:

-   When the operand is a tuple value, the result is a pack value that's
    identical to the tuple value.
-   When the operand is a symbolic variable `X` whose type is a tuple, the
    result is a pack with the same shape as the tuple. Each pack component's
    representative is `X[$I]`, and its type is the representative of the
    corresponding component of the tuple.

For these purposes, an array type `[T; N]` is treated as a tuple type
`(..., [:][T; N])`, so:

-   When the operand is an array type `[T; N]`, the result is a pack consisting
    of a single component with representative `T` and arity `N`.
-   When the operand is a symbolic variable `X` with type `[T; N]`, the result
    is a pack consisting of a single pack component whose representative is
    `X[$I]`, and whose arity is `N`.

An identifier expression that names a variable whose type is a pack type behaves
like a `[:]` expression whose operand is a variable of the corresponding tuple
type. No other leaf AST node can evaluate to a pack value.

A tuple cannot be indexed unless all of its components are singular. There is no
syntax for indexing into a pack.

## Pattern semantics

Pack expansions can also appear in patterns. The semantics are chosen to follow
the general principle that pattern matching is the inverse of expression
evaluation, so for example if the pattern `(..., [:]x: auto)` matches some
scrutinee value `s`, the expression `(..., [:]x)` should be equal to `s`. These
are run-time semantics, so all types are known constants, and any pack
components in the scrutinee value are singular.

There can be no more than one `...,` pattern in a tuple pattern. The N elements
of the pattern before the `...,` expansion are matched with the first N elements
of the scrutinee, and the M elements of the pattern after the `...,` expansion
are matched with the last M elements of the scrutinee. If the scrutinee does not
have at least N + M elements, the pattern does not match. The operand of the
`...,` pattern is matched with the sub-pack consisting of any remaining
scrutinee elements.

If a name binding pattern has a pack type, it is bound to a pack consisting of
the K scrutinee values that it is matched against. Otherwise, if the pattern is
inside a pack expansion, the program is ill-formed, because the name would be
bound multiple times. For example, in `(..., (foo: i32, bar: [:]Ts))`,
`foo: i32` is ill-formed because it would match a value like
`((1, "foo"), (2, "bar"), (3, "baz"))`, and bind the name `foo` to `1`, `2`, and
`3` simultaneously, which is nonsensical.

A _pack argument pattern_ consists of a `[:]` token and a single operand
pattern. The scrutinee must be a pack, and the pack argument pattern matches if
the operand pattern matches a tuple consisting of the elements of the scrutinee
pack.

Other kinds of patterns cannot have a pack type or pack-type operands, and
cannot match a pack scrutinee.

> **Future work:** It is probably possible to define a rule that generalizes
> other kinds of patterns to support variadics. However, we don't yet have
> motivating use cases for that, and it appears that in many if not most cases
> such patterns would be refutable, which substantially limits their usefulness.

## Typechecking

Variadic typechecking for expressions follows the reified model, but applied to
the types of expressions instead of the values of expressions. For example, for
most operations, if any of the operands have a pack type, all pack-type operands
must have the same shape, the type of the whole operation will have the same
shape, and the component types are found by iterating through the input pack
types component-wise, performing ordinary singular type-checking for the
operation.

Typechecking a variadic pattern is much like typechecking a variadic expression:
we proceed bottom-up, generalizing the singular typechecking rules to apply
component-wise to variadics, and so forth. The most notable difference is that
patterns can contain name bindings, whose types are determined by symbolically
evaluating the type portion of the binding.

Variadic statements always use the syntactic model at run-time, but we cannot
use that model at typechecking time because the rewrites in that model haven't
happened yet. Instead, we generalize the rules for typechecking expressions to
support statements as well, by treating statements as having types, in a
restricted way: the type of a statement is either `()` or a pack where all
representatives are `()`. In other words, statements have types, but they only
carry information about pack shape.

With that in place, we can apply essentially the same rule: if any of the child
AST nodes has a pack type, all child nodes with pack type must have the same
shape. If the parent statement is a `...{}` block, it will have type `()` (and
there must be at least one child with a pack type), and otherwise its type will
be the shape of the pack-type children (or `()` if there are none).

### Pattern matching

Typechecking for a pattern matching operation proceeds in three phases:

1. The pattern is typechecked, and assigned a type.
2. The scrutinee expression is typechecked, and assigned a type.
3. The scrutinee type is checked against the pattern type.

If the pattern appears in a context that requires it to be irrefutable, such as
the parameter list of a function declaration, phase 3 ensures that the pattern
can match _any_ possible value of the scrutinee expression. Otherwise, it
ensures that the pattern can match _some_ possible value of the scrutinee
expression. For simplicity, we currently only support variadic pattern matching
in contexts that require irrefutability.

> **Future work:** specify rules for refutable matching of variadics, for
> example to support C++-style recursive variadics.

Phases 1 and 2 were described earlier, so we only need to describe phase 3. We
will focus on the forms of symbolic type that are most relevant to variadics:
array types, tuple types, and pack types.

An array type pattern `[T; N]` matches an array type scrutinee `[U; M]` if `T`
matches `U` and `N` matches `M`. It can also match a tuple scrutinee type if the
tuple's pack component patterns all match `T`, and the sum of the tuple's pack
component arities matches `N`.

A tuple type pattern matches an array type scrutinee `[T; N]` if it matches the
corresponding tuple type `(..., [:][T; N])`. A tuple type pattern matches a
tuple type scrutinee if the corresponding pack types match.

Pack types only match with other pack types. In particular, `auto` never matches
a pack type, in order to ensure that named packs are always syntactically marked
at the point of declaration. Correctly matching one pack type to another is
difficult, because the packs may not have the same shape. As a consequence, we
don't necessarily know which pattern pack components each scrutinee pack
component will match with, or the other way around. For example, consider the
following code:

```carbon
fn F(a: i32, ..., [:]b: [i32;], c: i32);

fn G(..., [:]x: [i32;]) {
  F(1, 2, ..., [:]x);
}
```

If `x` is empty, the `2` will match with `c`, and otherwise the `2` will match
with an element of `b`. Similarly, if `x` is not empty, its last element will
match `c`, and the remaining elements (if any) will match elements of `b`.
However, at type-checking time we don't know the size of `x` yet, so we don't
know which will occur. On the other hand, the `1` will always match `a`.

In general, we want type checking to fail if any possible monomorphization of
the generic code would fail to typecheck. In this case that means we want type
checking to fail if any of the potential argument-parameter mappings could fail
to typecheck after monomorphization. Furthermore, for reasons of readability as
well as efficiency, we want type checking to fail if any two potential mappings
would deduce inconsistent values for any deduced parameter. However, in general
this is intractable, because in the worst case the number of distinct ways to
map symbolic arguments to parameters is ${2n \choose n}$ for n variadic
arguments, which is only a factor of $\sqrt{n}$ away from exponential.

Introducing type deduction further complicates the situation. For example:

```carbon
fn H[Ts:! [type;]](a: i32, ..., [:]b: Ts, c: String) -> (..., [:]Ts);

external impl P as ImplicitAs(i32);
external impl Q as ImplicitAs(String);

fn I(x: [i32;], y: [f32;], z: [String;]) {
  var result: auto = H(..., [:]x, {} as P, ..., [:]y, {} as Q, ..., [:]z);
}
```

Here, the deduced type of `result` can have one of four different forms. The
most general case is
`(..., [:][i32;], P, ..., [:][f32;], Q, ..., [:][String;])`, and the other three
cases are formed by omitting the prefix ending with `P` and/or the suffix
starting with `Q` (corresponding to the cases where `x` and/or `z` are empty).
Extending the type system to support deduction that splits into multiple cases
would add a fearsome amount of complexity to the type system.

#### Identifying potential matchings

Our solution will rely on being able to identify which pattern pack components
can potentially match which scrutinee pack components. We can do so as follows:

We will refer to the components of the pack pattern as "parameters", and the
components of the scrutinee value as "symbolic arguments". We will use the term
"actual arguments" to refer to the elements of the scrutinee after
monomorphization, so a single symbolic argument may correspond to any number of
actual arguments, including zero (but only if the expression is variadic). We
will refer to arguments as "singular" if they have arity 1, and "variadic" if
they have indeterminate arity (these are the only possibilities, because packs
are normalized). Similarly, we will refer to the `...,` parameter as "variadic"
and the other parameters as "singular".

A pack pattern consists of $N$ leading singular parameters, optionally followed
by a variadic parameter headed by the `...,` operator, and then $M$ trailing
singular parameters. The scrutinee must have a pack type, and can have any
number of singular and variadic components, in any order.

There must be at least $N+M$ singular symbolic arguments, because otherwise if
all variadic symbolic arguments are empty, there will not be enough actual
arguments to match all the singular parameters. We will refer to the $N$'th
singular symbolic argument and the symbolic arguments before it as "leading
symbolic arguments". Similarly, we will refer to the $M$'th-from-last singular
symbolic argument and the symbolic arguments after it as "trailing symbolic
arguments", and any remaining symbolic arguments as "central symbolic
arguments". A "leading actual argument" is an argument that was produced by
rewriting a leading symbolic argument, and likewise for "central actual
argument" and "trailing actual argument". Note that if there is no variadic
parameter, $M$ is 0, and so all parameters, symbolic arguments, and actual
arguments are leading.

By construction, there will always be at least $N$ leading actual arguments,
because there are $N$ singular leading symbolic arguments. Likewise, there will
always be at least $M$ trailing actual arguments. As a result, a leading
parameter can only match a leading actual argument, and so it can only match a
leading symbolic argument, and likewise for trailing parameters. Consequently, a
leading symbolic argument cannot match a trailing parameter, a trailing symbolic
argument cannot match a leading parameter, and a central symbolic argument can
only match the variadic parameter.

Consider the $i$'th singular leading symbolic argument $E$. If all the variadic
symbolic arguments before it are empty, $E$ will match the $i$'th leading
parameter, so $E$ cannot match any earlier parameter. If there are any earlier
variadic symbolic arguments, $E$ can be made to match any later leading
parameter or the variadic parameter, by making one of those earlier variadic
arguments large enough, but as observed above, $E$ cannot match a trailing
parameter. If there are no earlier variadic symbolic arguments, $E$ cannot be
made to match any later parameter, so it can only match the i'th leading
parameter.

Next, consider a variadic leading symbolic argument $E$ that comes before the
$i$'th singular leading symbolic argument, but not before any earlier singular
symbolic argument. If $E$'s rewritten arity is sufficiently large, and all
earlier variadic symbolic arguments are empty, it will simultaneously match the
$i$'th leading parameter, all leading parameters after it, and the variadic
parameter, but as before, it cannot match a trailing parameter.

The same reasoning can be applied to trailing symbolic arguments, but with
"before" and "after" reversed. And as noted earlier, central symbolic arguments
can only match the variadic parameter. In summary, we can identify the possible
matches for a symbolic argument $E$ as follows:

-   If $E$ is leading, let $i$ be one more than the number of earlier singular
    symbolic arguments:
    -   If $E$ is singular, and there are no earlier variadic argument
        expressions, then $E$ can only match the $i$'th leading parameter.
    -   Otherwise, $$ can match the $i$'th leading parameter, any later leading
        parameters, and the variadic parameter.
-   If $E$ is trailing, let $i$ be one more than the number of later singular
    symbolic arguments:
    -   If $E$ is singular, and there are no later variadic symbolic arguments,
        then $E$ can only match the i'th trailing parameter from the end.
    -   Otherwise, $E$ can match the $i$'th trailing parameter from the end, any
        earlier trailing parameters, and the variadic parameter.
-   Otherwise, $E$ can only match the variadic parameter.

#### The type-checking algorithm

In order to avoid type deduction that splits into multiple cases, we require
that if the variadic parameter's type involves a deduced value that is used in
more than one place (as `Ts` is in the earlier example of this problem), there
cannot be any leading or trailing variadic symbolic arguments (in the sense
defined in the previous section). This ensures that each symbolic argument can
only match one parameter, and so type deduction deterministically produces a
single result.

> **Open question:** Is that restriction too strict? If so, is it possible to
> forbid only situations that would actually cause type deduction to split into
> multiple cases. As well as being much less restrictive, that would avoid the
> need to give special treatment to deduced arrays that are used only once. It
> would still disallow cases like the call to `H` above, but that call seems
> unnatural for reasons that seem closely related to the fact that its type
> splits into cases.

To avoid a combinatorial explosion, we will use a much more tractable
conservative approximation of the precise algorithm. We type-check each
parameter as follows:

-   If the parameter is variadic, we check it against the sub-pack consisting of
    all symbolic arguments that it could potentially match. The rule stated
    above ensures that this is safe: the concatenated type can only become
    visible outside this local type-check if all variadic arguments have a known
    arity, in which case that sub-pack is known to be exactly correct.
-   Otherwise:
    -   If there is only one argument it can match, which is also not variadic,
        we type-check the argument against the parameter in the ordinary way.
    -   Otherwise:
        -   If the parameter has a non-deduced type, we check each potential
            argument against that type.
        -   Otherwise, we check that all potential arguments have the same type,
            and then check that type against the parameter.

> **Open question:** When deducing a single type from a sequence of types, can
> and should we relax the requirement that all types in the sequence are the
> same? We can identify the common type of a pair of types using
> `CommonTypeWith`, but it is not clear whether or how we can generalize that to
> a sequence of types, since it might not be associative.

We believe that if the code type checks successfully under this algorithm, any
possible monomorphization can type check using the types deduced here, because
the restrictions imposed here are a superset of the restrictions that any
monomorphization needs to satisfy, and the information available to type
deduction here is a subset of the information that would be available after
monomorphization.

## Alternatives considered

-   [Decouple variadics from arrays](/proposals/p2240.md#decouple-variadics-from-arrays)
-   [Fold expressions](/proposals/p2240.md#fold-expressions)
-   [Allow multiple `...,` patterns in a tuple pattern](/proposals/p2240.md#allow-multiple--patterns-in-a-tuple-pattern)
-   [Allow nested pack expansions](/proposals/p2240.md#allow-nested-pack-expansions)
-   [Disallow named packs](/proposals/p2240.md#disallow-named-packs)
-   [Use postfix instead of prefix operators](/proposals/p2240.md#use-postfix-instead-of-prefix-operators)

## References

-   Proposal
    [#2240: Variadics](https://github.com/carbon-language/carbon-lang/pull/2240)
