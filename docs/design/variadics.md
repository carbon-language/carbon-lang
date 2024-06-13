# Variadics

<!--
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

<!-- toc -->

## Table of contents

-   [Basics](#basics)
    -   [Additional examples](#additional-examples)
-   [Execution Semantics](#execution-semantics)
    -   [Expressions and statements](#expressions-and-statements)
    -   [Pattern matching](#pattern-matching)
-   [Typechecking](#typechecking)
    -   [Tuples, packs, segments, and shapes](#tuples-packs-segments-and-shapes)
    -   [Iterative typechecking of pack expansions](#iterative-typechecking-of-pack-expansions)
    -   [Typechecking patterns](#typechecking-patterns)
    -   [Typechecking pattern matches](#typechecking-pattern-matches)
-   [Appendix: Type system formalism](#appendix-type-system-formalism)
    -   [Typing and shaping rules](#typing-and-shaping-rules)
    -   [Reduction rules](#reduction-rules)
    -   [Other equivalences](#other-equivalences)
    -   [Convertibility and parameter deduction](#convertibility-and-parameter-deduction)
    -   [Deduction algorithm](#deduction-algorithm)
        -   [Canonicalization algorithm](#canonicalization-algorithm)
-   [Alternatives considered](#alternatives-considered)
-   [References](#references)

<!-- tocstop -->

## Basics

A "pack expansion" is a syntactic unit beginning with `...`, which is a kind of
compile-time loop over sequences called "packs". Packs are initialized and
referred to using "each-names", which are marked with the `each` keyword at the
point of declaration and the point of use.

The syntax and behavior of a pack expansion depends on its context, and in some
cases on a keyword following the `...`:

-   In a tuple literal expression (such as a function call argument list), `...`
    iteratively evaluates its operand expression, and treats the values as
    successive elements of the tuple.
-   `...and` and `...or` iteratively evaluate a boolean expression, combining
    the values using `and` and `or`, and ending the loop early if the underlying
    operator short-circuits.
-   In a statement context, `...` iteratively executes a statement.
-   In a tuple literal pattern (such as a function parameter list), `...`
    iteratively matches the elements of the scrutinee tuple. In conjunction with
    pack bindings, this enables functions to take an arbitrary number of
    arguments.

This example illustrates many of the key concepts:

```carbon
// Takes an arbitrary number of vectors with arbitrary element types, and
// returns a vector of tuples where the i'th element of the vector is
// a tuple of the i'th elements of the input vectors.
fn Zip[... each ElementType:! type]
      (... each vector: Vector(each ElementType))
      -> Vector((... each ElementType)) {
  ... var each iter: auto = each vector.Begin();
  var result: Vector((... each ElementType));
  while (...and each iter != each vector.End()) {
    result.push_back((... each iter));
    ... each iter++;
  }
  return result;
}
```

A _pack_ is a sequence of a fixed number of values called "elements", which may
be of different types. Packs are very similar to tuple values in many ways, but
they are not first-class values -- in particular, no run-time expression
evaluates to a pack. The _arity_ of a pack is a compile-time value representing
the number of values in the sequence.

An _each-name_ consists of the keyword `each` followed by the name of a pack,
and can only occur inside a pack expansion. On the Nth iteration of the pack
expansion, an each-name refers to the Nth element of the named pack. As a
result, a binding pattern with an each-name, such as `each ElementType:! type`,
acts as a declaration of all the elements of the named pack, and thereby
implicitly acts as a declaration of the pack itself.

Note that `each` is part of the name syntax, not an expression operator, so it
binds more tightly than any expression syntax. For example, the loop condition
`...and each iter != each vector.End()` in the implementation of `Zip` is
equivalent to `...and (each iter) != (each vector).End()`.

A _pack expansion_ is an instance of one of the following syntactic forms:

-   A statement of the form "`...` _statement_".
-   A tuple expression element of the form "`...` _expression_", with the same
    precedence as `,`.
-   A tuple pattern element of the form "`...` _pattern_", with the same
    precedence as `,`.
-   An implicit parameter list element of the form "`...` _pattern_", with the
    same precedence as `,`.
-   An expression of the form "`...` `and` _expression_" or "`...` `or`
    _expression_", with the same precedence as `and` and `or`.

The statement, expression, or pattern following the `...` (and the `and`/`or`,
if present) is called the _body_ of the expansion.

The `...` token can also occur in a tuple expression element of the form "`...`
`expand` _expression_", with the same precedence as `,`. However, that syntax is
not considered a pack expansion, and has its own semantics: _expression_ must
have a tuple type, and "`...` `expand` _expression_" evaluates _expression_ and
treats its elements as elements of the enclosing tuple literal. This is
especially useful for using non-literal tuple values as function call arguments:

```carbon
fn F(x: i32, y: String);
fn MakeArgs() -> (i32, String);

F(...expand MakeArgs());
```

`...and`, `...or`, and `...expand` can be trivially distinguished with one token
of lookahead, and the other meanings of `...` can be distinguished from each
other by the context they appear in. As a corollary, if the nearest enclosing
delimiters around a `...` are parentheses, they will be interpreted as forming a
tuple rather than as grouping. Thus, expressions like `(... each ElementType)`
in the above example are tuple literals, even though they don't contain commas.

By convention, `...` is always followed by whitespace, except that `...and`,
`...or`, and `...expand` are written with no whitespace between the two tokens.
This serves to emphasize that the keyword is not part of the expansion body, but
rather a modifier on the syntax and semantics of `...`.

All each-names in a given expansion must refer to packs with the same arity,
which we will also refer to as the arity of the expansion. If an expansion
contains no each-names, it must be a pattern, or an expression in the type
position of a binding pattern, and its arity is deduced from the scrutinee.

A pack expansion or `...expand` expression cannot contain another pack expansion
or `...expand` expression.

An each-name cannot be used in the same pack expansion that declares it. In most
if not all cases, an each-name that violates this rule can be changed to an
ordinary name, because each-names are only necessary when you need to transfer a
pack from one pack expansion to another.

A pack expansion expression or statement can be thought of as a kind of loop
that executes at compile time (specifically, monomorphization time), where the
expansion body is implicitly parameterized by an integer value called the _pack
index_, which ranges from 0 to one less than the arity of the expansion. The
pack index is implicitly used as an index into the packs referred to by
each-names. This is easiest to see with statement pack expansions. For example,
if `a`, `x`, and `y` are packs with arity 3, then
`... each a += each x * each y;` is roughly equivalent to

```carbon
for (let i:! i32 in (0, 1, 2)) {
  a[:i:] += x[:i:] * y[:i:];
}
```

Here we are using `[::]` as a hypothetical pack indexing operator for purposes
of illustration; packs cannot actually be indexed in Carbon code. Note also that
this rewritten form would not typecheck under the usual rules, because the
expressions `a[:i:]`, `x[:i:]`, and `y[:i:]` may have different types depending
on the value of `i`.

`...and` and `...or` behave like chains of the corresponding boolean operator,
so `...and F(each x, each y)` behaves like
`true and F(x[:0:], y[:0:]) and F(x[:1:], y[:1:]) and F(x[:2:], y[:2:])`. They
can also be interpreted as looping constructs, although the rewrite is less
straightforward because Carbon doesn't have a way to write a loop in an
expression context. An expression like `...and F(each x, each y)` can be thought
of as evaluating to the value of `result` after executing the following code
fragment:

```
var result: bool = true;
for (let i:! i32 in (0, 1, 2)) {
  result = result && F(x[:i:], y[:i:]);
  if (result == false) { break; }
}
```

`...` in a tuple literal behaves like a series of comma-separated tuple
elements, so `(... F(each x, each y))` is equivalent to
`(F(x[:0:], y[:0:]), F(x[:1:], y[:1:]), F(x[:2:], y[:2:]))`. This can't be
expressed as a loop in Carbon code, but it is still fundamentally iterative.

A pack expansion pattern "`...` _subpattern_" appears as part of a tuple pattern
(or an implicit parameter list), and matches a sequence of tuple elements if
each element matches _subpattern_. Since _subpattern_ will be matched against
multiple scrutinees (or none) in a single pattern-matching operation, a binding
patterns within a pack expansion pattern must declare an each-name, and the Nth
iteration of the pack expansion will initialize the Nth element of the named
pack from the Nth scrutinee. The binding pattern's type expression may contain
an each-name, but if so, it must be a deduced parameter of the enclosing
pattern.

> **Future work:** That restriction can probably be relaxed, but we currently
> don't have motivating use cases to constrain the design.

### Additional examples

```carbon
// Computes the sum of its arguments, which are i64s
fn SumInts(... each param: i64) -> i64 {
  var sum: i64 = 0;
  ... sum += each param;
  return sum;
}
```

```carbon
// Concatenates its arguments, which are all convertible to String
fn StrCat[... each T:! ConvertibleToString](... each param: each T) -> String {
  var len: i64 = 0;
  ... len += each param.Length();
  var result: String = "";
  result.Reserve(len);
  ... result.Append(each param.ToString());
  return result;
}
```

```carbon
// Returns the minimum of its arguments, which must all have the same type T.
fn Min[T:! Comparable & Value](first: T, ... each next: T) -> T {
  var result: T = first;
  ... if (each next < result) {
    result = each next;
  }
  return result;
}
```

```carbon
// Invokes f, with the tuple `args` as its arguments.
fn Apply[... each T:! type, F:! CallableWith(... each T)]
    (f: F, args: (... each T)) -> auto {
  return f(...expand args);
}
```

```carbon
// Toy example of mixing variadic and non-variadic parameters.
// Takes an i64, any number of f64s, and then another i64.
fn MiddleVariadic(first: i64, ... each middle: f64, last: i64);
```

```carbon
// Toy example of using the result of variadic type deduction.
fn TupleConcat[... each T1: type, ... each T2: type](
    t1: (... each T1), t2: (... each T2)) -> (... each T1, ... each T2) {
  return (...expand t1, ...expand t2);
}
```

## Execution Semantics

### Expressions and statements

In all of the following, N is the arity of the pack expansion being discussed,
and `$I` is a notional variable representing the pack index. These semantics are
implemented at monomorphization time, so the value of N is a known integer
constant. Although the value of `$I` can vary during execution, it is
nevertheless treated as a constant.

A statement of the form "`...` _statement_" is evaluated by executing
_statement_ N times, with `$I` ranging from 0 to N - 1.

An expression of the form "`...and` _expression_" is evaluated as follows: a
notional `bool` variable `$R` is initialized to `true`, and then "`$R = $R and`
_expression_" is executed up to N times, with `$I` ranging from 0 to N - 1. If
at any point `$R` becomes false, this iteration is terminated early. The final
value of `$R` is the value of the expression.

An expression of the form "`...or` _expression_" is evaluated the same way, but
with `or` in place of `and`, and `true` and `false` transposed.

A tuple expression element of the form "`...` _expression_" evaluates to a
sequence of N values, where the k'th value is the value of _operand_ where `$I`
is equal to k - 1.

An each-name evaluates to the `$I`th value of the pack it refers to (indexed
from zero).

### Pattern matching

The semantics of pack expansion patterns are chosen to follow the general
principle that pattern matching is the inverse of expression evaluation, so for
example if the pattern `(... each x: auto)` matches some scrutinee value `s`,
the expression `(... each x)` should be equal to `s`. These semantics are
implemented at monomorphization time, so all types are known constants, and all
all arities are known.

A tuple pattern can contain no more than one subpattern of the form "`...`
_operand_". When such a subpattern is present, the N elements of the pattern
before the `...` expansion are matched with the first N elements of the
scrutinee, and the M elements of the pattern after the `...` expansion are
matched with the last M elements of the scrutinee. If the scrutinee does not
have at least N + M elements, the pattern does not match.

The remaining elements of the scrutinee are iteratively matched against
_operand_, in order. In each iteration, `$I` is equal to the index of the
scrutinee element being matched, minus N.

On the Nth iteration, a binding pattern binds the Nth element of the named pack
to the Nth scrutinee value.

## Typechecking

### Tuples, packs, segments, and shapes

In order to discuss the underlying type system for variadics, we will need to
introduce some pseudo-syntax to represent values and expressions that occur in
the type system, but cannot be expressed directly in user code. We will use
non-ASCII glyphs such as `⟪⟫‖⟬⟭` for that pseudo-syntax, to distinguish it from
valid Carbon syntax.

In the context of variadics, we will say that a tuple literal consists of a
comma-separated sequence of _segments_, and reserve the term "elements" for the
components of a tuple literal after pack expansion. For example, the expression
`(... each foo)` may evaluate to a tuple value with any number of elements, but
the expression itself has exactly one segment.

The type of a tuple literal is another tuple literal, which is computed
segment-wise. For example, suppose we are trying to find the type of `z` in this
code:

```carbon
fn F[... each T:! type]((... each x: Optional(each T)), (... each y: i32)) {
  let z: auto = (0 as f32, ... each x, ... each y);
}
```

We proceed by finding the type of each segment. The type of `0 as f32` is `f32`,
by the usual non-variadic typing rules. The type of `... each x` is
`... Optional(each T)`, because `Optional(each T)` is the declared type of
`each x`, and the type of a pack expansion is a pack expansion of the type of
its body.

The type of `... each y` is more complicated. Conceptually, it consists of some
number of repetitions of `i32`. We don't know exactly how many repetitions,
because it's implicitly specified by the caller: it's the arity of the second
argument tuple. Effectively, that arity acts as a hidden deduced parameter of
`F`.

So to represent this type, we need two new pseudo-syntaxes:

-   `‖each X‖` refers to the deduced arity of the pack expansion that contains
    the declaration of `each X`.
-   `⟪E; N⟫` evaluates to `N` repetitions of `E`. This is called a _arity
    coercion_, because it coerces the expression `E` to have arity `N`. `E` must
    not contain any pack expansions, each-names, or pack literals.

Combining the two, the type of `... each y` is `... ⟪i32; ‖each y‖⟫`. Thus, the
type of `z` is `(f32, ... Optional(each T), ... ⟪i32; ‖each y‖⟫)`.

Now, consider a modified version of that example:

```carbon
fn F[... each T:! type]((... each x: Optional(each T)), (... each y: i32)) {
  let (each z: auto) = (0 as f32, ... each x, ... each y);
}
```

`each z` is a pack, but it has the same elements as the tuple `z` in our earlier
example, so we represent its type in the same way, as a sequence of segments:
`⟬f32, Optional(each T), ⟪i32; ‖each y‖⟫⟭`. The `⟬⟭` delimiters make this a
_pack literal_ rather than a tuple literal. Notice one subtle difference: the
segments of a pack literal do not contain `...`. In effect, every segment of a
pack literal acts as a separate loop body.

The _shape_ of a pack literal is a tuple of the arities of its segments, so the
shape of `⟬f32, Optional(each T), ⟪i32; ‖each y‖⟫⟭` is
`(1, ‖each T‖, ‖each y‖)`. Other expressions also have shapes. In particular,
the shape of an arity coercion `⟪E; A⟫` is `(A)`, the shape of `each X` is
`‖each X‖`, and the shape of an expression that does not contain pack literals,
shape coercions, or each-names is 1. The arity of an expression is the sum of
the elements of its shape. See the [appendix](#typing-and-shaping-rules) for the
full rules for determining the shape of an expression.

A pack literal can be _expanded_, which moves its parent AST node inside the
pack literal, so long as the parent node is not `...`. For example,
`... Optional(⟬each X, Y⟭)` is equivalent to
`... ⟬Optional(each X), Optional(Y)⟭`. Similarly, an arity coercion can be
expanded so long as the parent node is not `...` or ` pack literal. See the
[appendix](#reduction-rules) for the full rules governing this operation. _Fully
expanding_ an expression that does not contain a pack expansion means repeatedly
expanding any pack literals and arity coercions within it, until they cannot be
expanded any further.

The _scalar components_ of a fully-expanded expression `E` are a set, defined as
follows:

-   If `E` is a pack literal, its scalar components are the union of the scalar
    components of the segments.
-   If `E` is an arity coercion `⟪F; S⟫`, the only scalar component of `E` is
    `F`.
-   Otherwise, the only scalar component of `E` is `E`.

The scalar components of any other expression are the scalar components of its
fully expanded form.

By construction, a segment of a pack literal never has more than one scalar
component. Also by construction, a scalar component cannot contain a pack
literal, pack expansion, or arity coercion, but it can contain each-names, so we
can operate on it using the ordinary rules of non-variadic expressions so long
as we treat the names as opaque.

### Iterative typechecking of pack expansions

Since the execution semantics of an expansion are defined in terms of a notional
rewritten form where we simultaneously iterate over each-names, in principle we
can typecheck the expansion by typechecking the rewritten form. However, the
rewritten form usually would not typecheck as ordinary Carbon code, because the
each-names can have different types on different iterations. Furthermore, the
difference in types can propagate through expressions: if `each x` and `each y`
can have different types on different iterations, then so can `each x * each y`.
In effect, we have to typecheck the loop body separately for each iteration.

However, at typechecking time we usually don't even know how many iterations
there will be, much less what type an each-name will have on any particular
iteration, because the types of the each-names are packs, which are sequences of
segments, not sequences of elements. To solve that problem, we require that the
types of all each-names in a pack expansion must have the same shape. This
enables us to typecheck the pack expansion by simultaneously iterating over
segments instead of input elements.

As a result, the type of an expression or pattern within a pack expansion is a
sequence of segments, or in other words a _pack_, representing the types it
takes on over the course of the iteration. Note, however, that even though such
an expression has a pack type, it does not evaluate to a pack value. Rather, it
evaluates to a sequence of non-pack values over the course of the pack expansion
loop, and its pack type summarizes the types of that sequence.

Within a given iteration, typechecking follows the usual rules of non-variadic
typechecking, except that when we need the type of an each-name, we use the
scalar component of the current segment of its type. As noted above, we can
operate on a scalar component using the ordinary rules of non-variadic
typechecking.

Once the body of a pack expansion has been typechecked, typechecking the
expansion itself is relatively straightforward:

-   A statement pack expansion requires no further typechecking, because
    statements don't have types.
-   An `...and` or `...or` expression has type `bool`, and every segment of the
    operand's type pack must have a type that's convertible to `bool`.
-   For a `...` tuple element expression or pattern, the segments of the
    operand's type pack become segments of the type of the enclosing tuple.

> **TODO:** Discuss typechecking `...expand`.

### Typechecking patterns

A _full pattern_ consists of an optional deduced parameter list, a pattern, and
an optional return type expression.

A pack expansion pattern has _fixed arity_ if it contains at least one usage of
an each-name that is not a parameter of the enclosing full pattern. Otherwise it
has _deduced arity_. A tuple pattern can have at most one segment with deduced
arity.

After typechecking a full pattern, we attempt to merge as many tuple segments as
possible, in order to simplify the subsequent pattern matching. For example,
consider the following function declaration:

```carbon
fn Min[T:! type](first: T, ... each next: T) -> T;
```

During typechecking, we rewrite that function signature so that it only has one
parameter:

```carbon
fn Min[T:! type](... each args: ⟪T; ‖each next‖+1⟫) -> T;
```

(We represent the arity as `‖each next‖+1` to capture the fact that `each args`
must match at least one element.)

When the pattern is heterogeneous, the merging process may be more complex. For
example:

```carbon
fn ZipAtLeastOne[First:! type, ... each Next:! type]
    (first: Vector(First), ... each next: Vector(each Next))
    -> Vector((First, ... each Next));
```

During typechecking, we transform that function signature to the following form:

```carbon
fn ZipAtLeastOne[... ⟬First, each Next⟭:! type]
    (... each args: Vector(⟬First, each Next⟭))
    -> Vector((...⟬First, each Next⟭));
```

In this rewritten form, we treat the pack of deduced parameter names
`⟬First, each Next⟭` as though it were the each-name of a single deduced
parameter, called a _synthetic deduced parameter_. This enables us to model this
function as only having one parameter. The fact that it's composed of an
ordinary name and an each-name has no further effect, except to record the fact
that this pack must have at least one element (and consequently `each args` must
too). Note in particular that `Vector(⟬First, each Next⟭)` is considered to be
fully expanded, and thus it has a single scalar component, itself.

A synthetic deduced parameter must satisfy the following requirements:

-   Its segments are all the names of deduced parameters of the full pattern.
-   No name occurs more than once.
-   One of the names must be an each-name.
-   All the names must have the same declared type.

The rewritten full pattern must satisfy the following requirements:

-   Names that are part of a synthetic deduced parameter are not used outside a
    synthetic deduced parameter, or in synthetic deduced parameters that are not
    equal.
-   A pack expansion containing a synthetic deduced parameter doesn't contain
    any pack literals that aren't synthetic deduced parameters.

See the [appendix](#deduction-algorithm) for a more formal discussion of the
rewriting process.

### Typechecking pattern matches

To typecheck a pattern match between a tuple pattern and a tuple scrutinee, we
try to split and merge the segments of the scrutinee type so that it has the
same number of segments as the pattern type, and corresponding segments have the
same arity. For example, consider this call to `ZipAtLeastOne` (as defined in
the previous section):

```carbon
fn F[... each T:! type](... each t: Vector(each T), u: Vector(i32)) {
  ZipAtLeastOne(... each t, u);
}
```

The pattern type is `(... Vector(⟬First, each Next⟭))`, so we need to rewrite
the scrutinee type `(... Vector(each T), Vector(i32))` to have a single tuple
segment with an arity that matches `‖each Next‖+1`. We can do that by merging
the scrutinee segments to obtain `(... ⟬Vector(each T), Vector(i32)⟭)`. This has
a single segment with arity `‖each T‖+1`, which can match `‖each Next‖+1`
because the deduced arity `‖each Next‖` behaves as a deduced parameter of the
pattern, so they match by deducing `‖each Next‖ == ‖each T‖`.

Note that when merging segments of the scrutinee, we can't form synthetic
deduced parameters (because the scrutinee is not in a deducing context), but we
also don't need to: we don't require a merged scrutinee segments to have a
single scalar component.

The search for this rewrite processes each pattern segment to the left of the
segment with deduced arity, in order from left to right. For each pattern
segment, it greedily merges unmatched scrutinee segments from left to right
until their cumulative shape is greater than or equal to the shape of the
pattern segment, and then splits off a scrutinee segment on the right if
necessary to make the shapes exactly match. Pattern segments to the right of the
segment with deduced arity are processed the same way, but with left and right
reversed, so that segments are always processed from the outside in.

See the [appendix](#appendix-type-system-formalism) for the rewrite rules that
govern merging and splitting.

Once we have the pattern and scrutinee segments in one-to-one correspondence, we
check each scalar component of the scrutinee type against the scalar component
of the corresponding pattern type segment (by construction, the pattern type
segment has only one scalar component). Since we are checking scalar components
against scalar components, this proceeds according to the usual rules of
non-variadic typechecking.

> **TODO:** Extend this approach to fall back to a complementary approach, where
> the pattern and scrutinee trade roles: we maximally merge the scrutinee tuple,
> while requiring each segment to have a single scalar component, and then
> merge/split the pattern tuple to match it, without requiring pattern tuple
> segments to have a single scalar component. This isn't quite symmetric with
> the current approach, because when processing the scrutinee we're not forming
> synthetic deduced parameters, we're forming synthetic `let` bindings.

## Appendix: Type system formalism

### Typing and shaping rules

The _body arity_ of a pack expansion pattern is determined as follows:

-   If it contains a usage of at least one each-name that is not a parameter of
    the pattern, the body arity is the shape of that each-name.
-   Otherwise, the body arity is a hidden deduced parameter of the pattern
    called the _deduced arity binding_.

The shapes of expressions and statements within a pack expansion are determined
as follows:

-   The shape of an arity coercion is the value of the expression after the `;`.
-   The shape of a pack literal is the concatenation of the arities of its
    segments.
-   The shape of a binding pattern that declares an each-name is the body arity
    of the enclosing pack expansion.
-   The shape of an each-name expression is the body arity of the pack expansion
    containing the declaration of the each-name.
-   Any other AST node is _well shaped_ if there is some shape `S` such that all
    of the node's children have shape either 1 or `S`. When that condition
    holds, the shape of the node is `S` (or 1 if all children have shape 1).

The type of an expression or pattern can be computed as follows:

-   If `E` is an expression that does not contain any each-names or pack
    literals, the type of `each x: E` is `⟪E; A⟫`, where `A` is the body arity
    of the enclosing pack expansion.
-   The type of `each x: auto` is `each X`, a newly-invented deduced parameter
    of the enclosing full pattern, which behaves as if it was declared as
    `... each X:! type`.
-   The type of an each-name expression is the type of the binding pattern that
    declared it.
-   The type of an arity coercion `⟪E; S⟫` is `⟪T; S⟫`, where `T` is the type of
    `E`.
-   The type of a pack literal is a pack literal consisting of the concatenated
    types of its segments. This concatenation flattens any nested pack literals
    (for example `⟬A, ⟬B, C⟭⟭` becomes `⟬A, B, C⟭`)
-   The type of a pack expansion expression or pattern is `...B`, where `B` is
    the type of its body.
-   The type of a tuple literal is a tuple literal consisting of the types of
    its segments.
-   If an expression `E` contains a pack literal or arity coercion that is not
    inside a pack expansion, the type of `E` is the type of the fully expanded
    form of `E`.

> **TODO:** address `...expand`, `...and` and `...or`.

### Reduction rules

Unless otherwise specified, all expressions in these rules must be free of side
effects. Note that every reduction rule is also an equivalence: the expression
before the reduction is equivalent to the expression after, so these rules can
sometimes be run in reverse (particularly during deduction).

Expressions that are reduced by these rules must be well-shaped (and the reduced
form will likewise be well-shaped), but need not be well-typed.

_Empty pack removal:_ `...⟬⟭` reduces to the empty string.

_Singular expansion removal:_ `...E` reduces to `E`, if `E` contains no pack
literals, arity coercions, or each-names.

_Pack expansion splitting:_ If `E` is a segment and `S` is a sequence of
segments, `...⟬E, S⟭` reduces to `...E, ...⟬S⟭`.

_Pack expanding:_ If `F` is a function, and `X` is an expression that does not
contain pack literals or each-names, then `F(⟬A1, A2⟭, X, ⟬B1, B2⟭, ⟪Y; S⟫)`
reduces to `⟬F(A1, X, B1, Y), F(A2, X, B2, Y)⟭`. This rule generalizes in
several dimensions:

-   `F` can have any number of non-pack-literal arguments, and any positive
    number of pack literal arguments, and they can be in any order.
-   The pack literal arguments can have any number of segments, so long as they
    all have the same number of segments.
-   `F()` can be any expression syntax other than `...`, not just a function
    call. For example, this rule implies that `⟬X1, X2⟭ * ⟬Y1, Y2⟭` reduces to
    `⟬X1 * Y1, X2 * Y2⟭`, where the `*` operator plays the role of `F`.

_Coercion expanding:_ If `F` is a function, `S` is a shape, and `Y` is an
expression that does not contain pack literals or arity coercions,
`F(⟪X; S⟫, Y, ⟪Z; S⟫)` reduces to `⟪F(X, Y, Z); S⟫`. As with pack expanding,
this rule generalizes:

-   `F` can have any number of non-arity-coercion arguments, and any positive
    number of arity coercion arguments, and they can be in any order.
-   `F()` can be any expression syntax other than `...` or pack literal
    formation, not just a function call.

_Coercion removal:_ `⟪E; 1⟫` reduces to `E`.

_Tuple indexing:_ Let `I` be an integer template constant, let `X` be a tuple
segment, and let `Ys` be a sequence of tuple segments.

-   If the arity `A` of `X` is less than `I+1`, then `(X, Ys).(I)` reduces to
    `(Ys).(I-A)`.
-   Otherwise:
    -   If `X` is not a pack expansion, then `(X, Ys).(I)` reduces to `X`.
    -   If `X` is of the form `...⟬⟪E; S⟫⟭`, then `(X, Ys).(I)` reduces to `E`.

### Other equivalences

Unless otherwise specified, all expressions in these rules must be free of side
effects.

_Coercion merging:_

-   `⟪E; M⟫, ⟪E; N⟫` is equivalent to `⟪E; M, N⟫`.
-   `⟪E; N⟫, E` is equivalent to `⟪E; N, 1⟫`.
-   `E, ⟪E; N⟫` is equivalent to `⟪E; 1, N⟫`.

_Coercion shape commutativty:_ `⟪E; S1⟫` is equivalent to `⟪E; S2⟫` if `S1` is a
permutation of `S2`.

### Convertibility and parameter deduction

Type convertibility is governed by a set of deduction rules for the "convertible
to" and "deducible from" relations. For example:

-   `T` is convertible to `U` if `T` implements `ImplicitAs(U)`.
-   `T` is convertible to `U` if `U` is deducible from `T`.
-   For any `X`, `X` is deducible from `X`.
-   Let `T` and `U` be tuple segments, and let `Ts` and `Us` be sequences of
    tuple segments. `(T, Ts)` is convertible to `(U, Us)` if `T` is convertible
    to `U` and `(Ts)` is convertible to `(Us)`.
-   Let `P` be a parameterized type. `P(A, B)` is deducible from `P(C, D)` if
    `A` is deducible from `C` and `B` is deducible from `D`.

Formally, these rules all implicitly propagate a _binding map_. For example, the
last rule above can be stated more precisely as:

-   Let `P` be a parameterized type. If `A` is deducible from `C` given a
    binding map `M1`, and `B` is deducible from `D` given a binding map `M2`,
    then `P(A, B)` is deducible from `P(C, D)` given a binding map `M1` ∪ `M2`.

A binding map is a set of pairs where the first element of the pair is a deduced
parameter of the pattern, and the second element is the deduced value of that
parameter. We model a parameter as a sequence of names, in order to accommodate
synthetic deduced parameters. A binding map must be _consistent_, which means
that if `N1` and `N2` are the names of two different name/value pairs in the
map, `N1` cannot be a (non-strict) subsequence of `N2`.

In almost all cases, the binding map in the conclusion is the union of the
binding maps in the premises, and in those cases we will usually leave the
binding maps implicit (as in the original formulation of the above rule). One
major exception is the following rule:

_Binding introduction:_ Let `X` be a deduced parameter with type `T`. If the
type of `E` is convertible to `T` given binding map `M`, then `X` is deducible
from `E` given a binding map `M` ∪ "`X` is bound to `E`".

The deduction rules that are specific to variadic types are as follows:

`...T` is convertible to `...U` if the shape of `U` is deducible from the shape
of `T`, and each scalar component of `T` is convertible to every scalar
component of `U`.

Let `X` be a deduced arity binding, and let `S` be a shape expression. `X` is
deducible from `S` given a binding map that consists of the single element "`X`
is bound to `S`".

Let `(S1)`, `(S2)`, `(S3)`, and `(S4)` be shapes. `(S1, S2)` is deducible from
`(S3, S4)` if `(S1)` is deducible from `(S3)` and `(S2)` is deducible from
`(S4)`.

### Deduction algorithm

A function type is in _normal form_ if every pack expansion is fully expanded,
and the only pack literals are synthetic deduced parameters. Note that by
construction, this means that the body of every pack expansion has a single
scalar component.

The _canonical form_ of a function type is the unique normal form that is
"maximally merged", meaning that if `C` is the canonical form, and `D` is any
other normal form of the function type, then either `D` has more pack expansions
than `C`, or

-   `D` has the same number of pack expansions as `C`,
-   the shape of the body of every pack expansion in `D` is a prefix or suffix
    of the shape of the body of the corresponding pack expansion in `C`, and
-   in at least one case, it is a _strict_ prefix or suffix.

> **TODO:** Specify algorithm for converting a function type to canonical form,
> or establishing that there is no such form. See next section for a start.

If a function with type `F` is called with argument type `A`, we typecheck the
call by converting `F` to canonical form, and then checking whether `A` is
convertible to the parameter type by applying the deduction rules in the
previous section. If that succeeds, we apply the resulting binding map to the
function return type to obtain the type of the call expression.

> **TODO:** Specify the algorithm more precisely. In particular, discuss how to
> rewrite `A` as needed to make the shapes line up, but don't rewrite `F` after
> canonicalization.

Typechecking for pattern match operations other than function calls is defined
in terms of typechecking a function call: We check a scrutinee type `S` against
a pattern `P` by checking `(S,)` against `(P,)->()`.

> **Future work:** Extend this approach to support merging the argument list as
> well as the parameter list.

#### Canonicalization algorithm

The canonical form can be found by starting with a normal form, and
incrementally merging an adjacent singular parameter type into the variadic
parameter type.

For example, consider the following function:

```carbon
fn F[First:! type, Second:! type, ... each Next:! type]
    (first: Vector(First), second: Vector(Second),
     ... each next: Vector(each Next)) -> (First, Second, ... each Next);
```

The function type derived from that signature is:

```carbon
(Vector(First), Vector(Second), ... Vector(each Next))
    -> (First, Second, ... each Next)
```

We wrap `each Next` in a pack literal to express it in normal form:

```carbon
(Vector(First), Vector(Second), ... Vector(⟬each Next⟭)) -> (First, Second, ... ⟬each Next⟭)
```

Then we attempt to merge `Vector(Second)` into the pack expansion:

```carbon
// Pack expanding
(Vector(First), Vector(Second), ... ⟬Vector(each Next)⟭) -> (First, Second, ...⟬each Next⟭)
// Pack expansion splitting (in reverse)
(Vector(First), ... ⟬Vector(Second), Vector(each Next)⟭) -> (First, ...⟬Second, each Next⟭)
// Pack expanding (in reverse)
(Vector(First), ... Vector(⟬Second, each Next⟭)) -> (First, ...⟬Second, each Next⟭)
```

This is a normal form, because `⟬Second, each Next⟭` is a valid synthetic
deduced parameter. We can now repeat that process to merge the remaining
parameter type:

```carbon
(Vector(First), ... Vector(⟬Second, each Next⟭)) -> (First, ...⟬Second, each Next⟭)
// Pack expanding
(Vector(First), ...⟬Vector(Second), Vector(each Next)⟭) -> (First, ...⟬Second, each Next⟭)
// Pack expansion splitting (in reverse)
(...⟬Vector(First), Vector(Second), Vector(each Next)⟭) -> (...⟬First, Second, each Next⟭)
// Pack expanding (in reverse)
(...Vector(⟬First, Second, each Next⟭)) -> (...⟬First, Second, each Next⟭)
```

Here again, this is a normal form, because `⟬First, Second, each Next⟭` is a
valid synthetic deduced parameter. There is demonstrably no way to perform any
further merging, so this must be the canonical form.

> **TODO:** define the algorithm in more general terms, and discuss ways that
> merging can fail.

## Alternatives considered

FIXME update this

-   [Member packs](/proposals/p2240.md#member-packs)
-   [First-class packs](/proposals/p2240.md#first-class-packs)
-   [Generalize `expand`](/proposals/p2240.md#generalize-expand)
-   [Omit `expand`](/proposals/p2240.md#omit-expand)
-   [Support expanding arrays](/proposals/p2240.md#support-expanding-arrays)
-   [Omit pack bindings](/proposals/p2240.md#omit-pack-bindings)
    -   [Disallow pack-type bindings](/proposals/p2240.md#disallow-pack-type-bindings)
-   [Fold expressions](/proposals/p2240.md#fold-expressions)
-   [Allow multiple pack expansions in a tuple pattern](/proposals/p2240.md#allow-multiple-pack-expansions-in-a-tuple-pattern)
-   [Allow nested pack expansions](/proposals/p2240.md#allow-nested-pack-expansions)
-   [Use postfix instead of prefix `...`](/proposals/p2240.md#use-postfix-instead-of-prefix-)
-   [Avoid context-sensitity in pack expansions](/proposals/p2240.md#avoid-context-sensitity-in-pack-expansions)
    -   [Fold-like syntax](/proposals/p2240.md#fold-like-syntax)
    -   [Variadic blocks](/proposals/p2240.md#variadic-blocks)
    -   [Keyword syntax](/proposals/p2240.md#keyword-syntax)
-   [Require parentheses around `each`](/proposals/p2240.md#require-parentheses-around-each)
-   [Fused expansion tokens](/proposals/p2240.md#fused-expansion-tokens)
-   [Support merging parameters](/proposals/p2240.md#support-merging-parameters)

## References

-   Proposal
    [#2240: Variadics](https://github.com/carbon-language/carbon-lang/pull/2240)
