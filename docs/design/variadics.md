# Variadics

<!--
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

<!-- toc -->

## Table of contents

-   [Basics](#basics)
    -   [Overview](#overview)
    -   [Packs and each-names](#packs-and-each-names)
    -   [Pack expansions](#pack-expansions)
        -   [Pack expansion expressions and statements](#pack-expansion-expressions-and-statements)
        -   [Pack expansion patterns](#pack-expansion-patterns)
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
    -   [Explicit deduced arities](#explicit-deduced-arities)
    -   [Typing and shaping rules](#typing-and-shaping-rules)
    -   [Reduction rules](#reduction-rules)
    -   [Equivalence, equality, and convertibility](#equivalence-equality-and-convertibility)
    -   [Pattern match typechecking algorithm](#pattern-match-typechecking-algorithm)
        -   [Canonicalization algorithm](#canonicalization-algorithm)
-   [Alternatives considered](#alternatives-considered)
-   [References](#references)

<!-- tocstop -->

## Basics

### Overview

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
    the values using `and` and `or`. Normal short-circuiting behavior for the
    resulting `and` and `or` operators applies at runtime.
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

### Packs and each-names

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

### Pack expansions

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

#### Pack expansion expressions and statements

A pack expansion expression or statement can be thought of as a kind of loop
that executes at compile time (specifically, monomorphization time), where the
expansion body is implicitly parameterized by an integer value called the _pack
index_, which ranges from 0 to one less than the arity of the expansion. The
pack index is implicitly used as an index into the packs referred to by
each-names. This is easiest to see with statement pack expansions. For example,
if `a`, `x`, and `y` are packs with arity 3, then
`... each a += each x * each y;` is roughly equivalent to

```carbon
a[:0:] += x[:0:] * y[:0:];
a[:1:] += x[:1:] * y[:1:];
a[:2:] += x[:2:] * y[:2:];
```

Here we are using `[:N:]` as a hypothetical pack indexing operator for purposes
of illustration; packs cannot actually be indexed in Carbon code.

> **Future work:** We're open to eventually adding indexing of variadics, but
> that remains future work and will need its own proposal.

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

#### Pack expansion patterns

A pack expansion pattern "`...` _subpattern_" appears as part of a tuple pattern
(or an implicit parameter list), and matches a sequence of tuple elements if
each element matches _subpattern_. For example, in the signature of `Zip` shown
earlier, the parameter list consists of a single pack expansion pattern
`... each vector: Vector(each ElementType)`, and so the entire argument list
will be matched against the binding pattern
`each vector: Vector(each ElementType)`.

Since _subpattern_ will be matched against multiple scrutinees (or none) in a
single pattern-matching operation, a binding pattern within a pack expansion
pattern must declare an each-name (such as `each vector` in the `Zip` example),
and the Nth iteration of the pack expansion will initialize the Nth element of
the named pack from the Nth scrutinee. The binding pattern's type expression may
contain an each-name (such as `each ElementType` in the `Zip` example), but if
so, it must be a deduced parameter of the enclosing pattern.

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
non-ASCII glyphs such as `«»‖⟬⟭` for that pseudo-syntax, to distinguish it from
valid Carbon syntax.

In the context of variadics, we will say that a tuple literal consists of a
comma-separated sequence of _segments_, and reserve the term "elements" for the
components of a tuple literal after pack expansion. For example, the expression
`(... each foo)` may evaluate to a tuple value with any number of elements, but
the expression itself has exactly one segment.

Each segment has a type, which expresses (potentially symbolically) both the
types of the elements of the segment and the arity of the segment. The type of a
tuple literal is a tuple literal of the types of its segments. For example,
suppose we are trying to find the type of `z` in this code:

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
-   `«E; N»` evaluates to `N` repetitions of `E`. This is called a _arity
    coercion_, because it coerces the expression `E` to have arity `N`. `E` must
    not contain any pack expansions, each-names, or pack literals (see below).

Combining the two, the type of `... each y` is `... «i32; ‖each y‖»`. Thus, the
type of `z` is `(f32, ... Optional(each T), ... «i32; ‖each y‖»)`.

Now, consider a modified version of that example:

```carbon
fn F[... each T:! type]((... each x: Optional(each T)), (... each y: i32)) {
  let (... each z: auto) = (0 as f32, ... each x, ... each y);
}
```

`each z` is a pack, but it has the same elements as the tuple `z` in our earlier
example, so we represent its type in the same way, as a sequence of segments:
`⟬f32, Optional(each T), «i32; ‖each y‖»⟭`. The `⟬⟭` delimiters make this a
_pack literal_ rather than a tuple literal. Notice one subtle difference: the
segments of a pack literal do not contain `...`. In effect, every segment of a
pack literal acts as a separate loop body. As with the tuple literal syntax, the
pack literal pseudo-syntax can also be used in patterns.

The _shape_ of a pack literal is a tuple of the arities of its segments, so the
shape of `⟬f32, Optional(each T), «i32; ‖each y‖»⟭` is
`(1, ‖each T‖, ‖each y‖)`. Other expressions and patterns also have shapes. In
particular, the shape of an arity coercion `«E; A»` is `(A,)`, the shape of
`each X` is `(‖each X‖,)`, and the shape of an expression that does not contain
pack literals, shape coercions, or each-names is `(1,)`. The arity of an
expression is the sum of the elements of its shape. See the
[appendix](#typing-and-shaping-rules) for the full rules for determining the
shape of an expression.

If a pack literal is part of some enclosing expression that doesn't contain
`...`, it can be _expanded_, which moves the outer expression inside the pack
literal. For example, `... Optional(⟬each X, Y⟭)` is equivalent to
`... ⟬Optional(each X), Optional(Y)⟭`. Similarly, an arity coercion can be
expanded so long as the parent node is not `...`, a pattern, or a pack literal.
See the [appendix](#reduction-rules) for the full rules governing this
operation. _Fully expanding_ an expression or pattern that does not contain a
pack expansion means repeatedly expanding any pack literals and arity coercions
within it, until they cannot be expanded any further.

The _scalar components_ of a fully-expanded expression `E` are a set, defined as
follows:

-   If `E` is a pack literal, its scalar components are the union of the scalar
    components of the segments.
-   If `E` is an arity coercion `«F; S»`, the only scalar component of `E` is
    `F`.
-   Otherwise, the only scalar component of `E` is `E`.

The scalar components of any other expression that does not contain `...` are
the scalar components of its fully expanded form.

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
arity. For example:

```carbon
class C(... each T:! type) {
  fn F[... each U:! type](... each t: each T, ... each u: each U);
}
```

In the signature of `F`, `... each t: each T` has fixed arity, since the arity
is determined by the arguments passed to `C`, before the call to `F`. On the
other hand, `... each u: each U` has deduced arity, because the arity of
`each U` is determined by the arguments passed to `F`.

After typechecking a full pattern, we attempt to merge as many tuple segments as
possible, in order to simplify the subsequent pattern matching. For example,
consider the following function declaration:

```carbon
fn Min[T:! type](first: T, ... each next: T) -> T;
```

During typechecking, we rewrite that function signature so that it only has one
parameter:

```carbon
fn Min[T:! type](... each args: «T; ‖each next‖+1») -> T;
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
fn ZipAtLeastOne[... ⟬First, each Next⟭:! «type; ‖each next‖+1»]
    (... each __args: Vector(⟬First, each Next⟭))
    -> Vector((... ⟬First, each Next⟭));
```

We can then rewrite that by replacing the pack of names `⟬First, each Next⟭`
with an invented name `each __Args`, so that the function has only one
parameter:

```carbon
fn ZipAtLeastOne[... each __Args:! «type; ‖each next‖+1»]
    (... each __args: Vector(each __Args))
    -> Vector((... each __Args));
```

We can replace a name pack with an invented each-name only if all of the
following conditions hold:

-   The name pack doesn't use any name more than once. For example, we can't
    apply this rewrite to `⟬X, each Y, X⟭`.
-   The name pack contains exactly one each-name. For example, we can't apply
    this rewrite to `⟬X, Y⟭`.
-   The replacement removes all usages of the constituent names, including their
    declarations. For example, we can't apply this rewrite to `⟬X, each Y⟭` in
    this code, because the resulting signature would have return type `X` but no
    declaration of `X`:
    ```carbon
    fn F[... ⟬X, each Y⟭:! «type; ‖each next‖+1»]
        (... each __args: each ⟬X, each Y⟭) -> X;
    ```
-   The pack expansions being rewritten do not contain any pack literals other
    than the name pack being replaced. For example, we can't apply this rewrite
    to `⟬X, each Y⟭` in this code, because the pack expansion in the deduced
    parameter list also contains the pack literal `⟬I, each type⟭`:
    ```carbon
    fn F[... ⟬X, each Y⟭:! ⟬I, each type⟭](... each __args: each ⟬X, each Y⟭);
    ```
    Notice that as a corollary of this rule, all the names in the name pack must
    have the same type.

See the [appendix](#pattern-match-typechecking-algorithm) for a more formal
discussion of the rewriting process.

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

When merging segments of the scrutinee, we don't attempt to form name packs and
replace them with invented names, but we also don't need to: we don't require a
merged scrutinee segments to have a single scalar component.

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
> the current approach, because when processing the scrutinee we can't merge
> deduced parameters (scrutinees don't have any), but we can invent new `let`
> bindings.

## Appendix: Type system formalism

A _pack literal_ is a comma-separated sequence of segments, enclosed in `⟬⟭`
delimiters. A pack literal can appear in an expression, pattern, or name
context, and every segment must be valid in the context where the pack literal
appears (for example, the segments of a pack literal in a name context must all
be names). Pack literals cannot be nested, and cannot appear outside a pack
expansion.

### Explicit deduced arities

In this formalism, deduced arities are explicit rather than implicit, so Carbon
code must be desugared into this formalism as follows:

For each pack expansion pattern, we introduce a binding pattern `__N:! Arity` as
a deduced parameter of the enclosing full pattern, where `__N` is a name chosen
to avoid collisions. Then, for each binding pattern of the form `each X: T`
within that expansion, if `T` does not contain an each-name, the binding pattern
is rewritten as `each X: «T; __N»`. If this does not introduce any usages of
`__N`, we remove its declaration.

`Arity` is a compiler-internal type which represents non-negative integers. The
only operation it supports is `+`, with non-negative integer literals and other
`Arity`s. `Arity` is used only during type checking, so `+` has no run-time
semantics, and its only symbolic semantics are that it is commutative and
associative.

### Typing and shaping rules

The shape of an AST node within a pack expansion is determined as follows:

-   The shape of an arity coercion is the value of the expression after the `;`.
-   The shape of a pack literal is the concatenation of the arities of its
    segments.
-   The shape of an each-name expression is the shape of the binding pattern
    that declared the name.
-   If a binding pattern's name and type components have the same number of
    segments, and each name segment is an each-name if and only if the
    corresponding type segment's shape is not 1, then the shape of the binding
    pattern is the shape of the type expression. Otherwise, the binding pattern
    is ill-shaped.
-   For any other AST node:
    -   If all the node's children have shape 1, its shape is 1.
    -   If there is some shape `S` such that all of the node's children have
        shape either 1 or `S`, its shape is `S`.
    -   Otherwise, the node is ill-shaped.

> **TODO:** The "well-shaped" rules as stated are slightly too restrictive. For
> example, `⟬each X, Y⟭: «Z; N+1»` is well-shaped, and `(⟬each X, Y⟭, «Z; N+1»)`
> is well-shaped if the shape of `each X` is `N`.

The type of an expression or pattern can be computed as follows:

-   The type of `each x: auto` is `each __X`, a newly-invented deduced parameter
    of the enclosing full pattern, which behaves as if it was declared as
    `... each __X:! type`.
-   The type of an each-name expression is the type expression of the binding
    pattern that declared it.
-   The type of an arity coercion `«E; S»` is `«T; S»`, where `T` is the type of
    `E`.
-   The type of a pack literal is a pack literal consisting of the concatenated
    types of its segments. This concatenation flattens any nested pack literals
    (for example `⟬A, ⟬B, C⟭⟭` becomes `⟬A, B, C⟭`)
-   The type of a pack expansion expression or pattern is `...B`, where `B` is
    the type of its body.
-   The type of a tuple literal is a tuple literal consisting of the types of
    its segments.
-   If an expression or pattern `E` contains a pack literal or arity coercion
    that is not inside a pack expansion, the type of `E` is the type of the
    fully expanded form of `E`.

> **TODO:** address `...expand`, `...and` and `...or`.

### Reduction rules

Unless otherwise specified, all expressions in these rules must be free of side
effects. Note that every reduction rule is also an equivalence: the utterance
before the reduction is equivalent to the utterance after, so these rules can
sometimes be run in reverse (particularly during deduction).

Utterances that are reduced by these rules must be well-shaped (and the reduced
form will likewise be well-shaped), but need not be well-typed. This enables us
to apply these reductions while determining whether an utterance is well-typed,
as in the case of typing an expression or pattern that contains a pack literal
or arity coercion, above.

_Singular pack removal:_ if `E` is a pack segment, `⟬E⟭` reduces to `E`.

_Singular expansion removal:_ `...E` reduces to `E`, if the shape of `E` is
`(1,)`.

_Pack expansion splitting:_ If `E` is a segment and `S` is a sequence of
segments, `...⟬E, S⟭` reduces to `...E, ...⟬S⟭`.

_Pack expanding:_ If `F` is a function, `X` is an utterance that does not
contain pack literals, each-names, or arity coercions, and `⟬P1, P2⟭` and
`⟬Q1, Q2⟭` both have the shape `(S1, S2)`, then
`F(⟬P1, P2⟭, X, ⟬Q1, Q2⟭, «Y; S1+S2»)` reduces to
`⟬F(P1, X, Q1, «Y; S1»), F(P2, X, Q2, «Y; S2»)⟭`. This rule generalizes in
several dimensions:

-   `F` can have any number of arity coercion and other non-pack-literal
    arguments, and any positive number of pack literal arguments, and they can
    be in any order.
-   The pack literal arguments can have any number of segments (but the
    well-shapedness requirement means they must have the same number of
    segments).
-   `F()` can be any expression syntax other than `...`, not just a function
    call. For example, this rule implies that `⟬X1, X2⟭ * ⟬Y1, Y2⟭` reduces to
    `⟬X1 * Y1, X2 * Y2⟭`, where the `*` operator plays the role of `F`.
-   `F()` can also a be a pattern syntax. For example, this rule implies that
    `(⟬x1: X1, x2: X2⟭, ⟬y1: Y1, y2: Y2⟭)` reduces to
    `⟬(x1: X1, y1: Y1), (x2: X2, y2: Y2)⟭`, where the tuple pattern syntax
    `( , )` plays the role of `F`.
-   When binding pattern syntax takes the role of `F`, the name part of the
    binding pattern must be a name pack. For example, `⟬x1, x2⟭: ⟬X1, X2⟭`
    reduces to `⟬x1: X1, x2: X2⟭`, but `each x: ⟬X1, X2⟭` cannot be reduced by
    this rule.

_Coercion expanding:_ If `F` is a function, `S` is a shape, and `Y` is an
expression that does not contain pack literals or arity coercions,
`F(«X; S», Y, «Z; S»)` reduces to `«F(X, Y, Z); S»`. As with pack expanding,
this rule generalizes:

-   `F` can have any number of non-arity-coercion arguments, and any positive
    number of arity coercion arguments, and they can be in any order.
-   `F()` can be any expression syntax other than `...` or pack literal
    formation, not just a function call. Unlike pack expanding, coercion
    expanding does not apply if `F` is a pattern syntax.

_Coercion removal:_ `«E; 1»` reduces to `E`.

_Tuple indexing:_ Let `I` be an integer template constant, let `X` be a tuple
segment, and let `Ys` be a sequence of tuple segments.

-   If the arity `A` of `X` is less than `I+1`, then `(X, Ys).(I)` reduces to
    `(Ys).(I-A)`.
-   Otherwise:
    -   If `X` is not a pack expansion, then `(X, Ys).(I)` reduces to `X`.
    -   If `X` is of the form `...⟬«E; S»⟭`, then `(X, Ys).(I)` reduces to `E`.

### Equivalence, equality, and convertibility

_Pack renaming:_ Let `Ns` be a sequence of names, let `⟬Ns⟭: «T; N»` be a name
binding pattern (which may be a symbolic or template binding as well as a
runtime binding), and let `__A` be an identifier that does not collide with any
name that's visible where `⟬Ns⟭` is visible. We can rewrite all occurrences of
`⟬Ns⟭` to `each __A` in the scope of the binding pattern (including the pattern
itself) if all of the following conditions hold:

-   `Ns` contains at least one each-name.
-   No name in `Ns` is used in the scope outside of `Ns`.
-   No name occurs more than once in `Ns`.
-   No other pack literals occur in the same pack expansion as an occurrence of
    `⟬Ns⟭`.

_Expansion convertibility:_ `...T` is convertible to `...U` if the arity of `U`
equals the arity of `T`, and the scalar components of `T` are each convertible
to all scalar components of `U`.

_Shape equality:_ Let `(S1s)`, `(S2s)`, `(S3s)`, and `(S4s)` be shapes.
`(S1s, S2s)` equals `(S3s, S4s)` if `(S1s)` equals `(S3s)` and `(S2s)` equals
`(S4s)`.

### Pattern match typechecking algorithm

A full pattern is in _normal form_ if it contains no pack literals, and every
arity coercion is fully expanded. For example,
`[__N:! Arity](... each x: Vector(«i32; __N»))` is not in normal form, but
`[__N:! Arity](... each x: «Vector(i32); __N»)` is. Note that all user-written
full patterns are in normal form. Note also that by construction, this means
that the type of the body of every pack expansion has a single scalar component.
The _canonical form_ of a full pattern is the unique normal form (if any) that
is "maximally merged", meaning that every tuple pattern and tuple literal has
the smallest number of segments. For example, the canonical form of
`[__N:! Arity](... each x: «i32; __N», y: i32)` is
`[__N:! Arity](... each __args: «i32; __N+1»)`.

> **TODO:** Specify algorithm for converting a full pattern to canonical form,
> or establishing that there is no such form. See next section for a start.

If a function with type `F` is called with argument type `A`, we typecheck the
call by converting `F` to canonical form, and then checking whether `A` is
convertible to the parameter type by applying the deduction rules in the
previous sections. If that succeeds, we apply the resulting binding map to the
function return type to obtain the type of the call expression.

> **TODO:** Specify the algorithm more precisely. In particular, discuss how to
> rewrite `A` as needed to make the shapes line up, but don't rewrite `F` after
> canonicalization.

Typechecking for pattern match operations other than function calls is defined
in terms of typechecking a function call: We check a scrutinee type `S` against
a pattern `P` by checking `__F(S,)` against a hypothetical function signature
`fn __F(P,)->();`.

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

First, we desugar the implicit arity:

```carbon
fn F[__N:! Arity, First:! type, Second:! type, ... each Next:! «type; __N»]
    (first: Vector(First), second: Vector(Second),
     ... each next: Vector(each Next)) -> (First, Second, ... each Next);
```

Then we attempt to merge `Second` with `each Next` as follows (note that for
brevity, some of the steps presented here actually contain multiple independent
reductions):

```carbon
// Singular pack removal (in reverse)
fn F[__N:! Arity, First:! type, Second:! type, ... ⟬each Next:! «type; __N»⟭]
    (first: Vector(First), second: Vector(Second),
     ... each next: Vector(⟬each Next⟭)) -> (First, Second, ... ⟬each Next⟭);
// Pack expanding
fn F[__N:! Arity, First:! type, Second:! type, ... ⟬each Next:! «type; __N»⟭]
    (first: Vector(First), second: Vector(Second),
     ... each next: ⟬Vector(each Next)⟭) -> (First, Second, ... ⟬each Next⟭);
// Pack expanding
fn F[__N:! Arity, First:! type, Second:! type, ... ⟬each Next:! «type; __N»⟭]
    (first: Vector(First), second: Vector(Second),
     ... ⟬each next: Vector(each Next)⟭) -> (First, Second, ... ⟬each Next⟭);
// Pack expansion splitting (in reverse)
fn F[__N:! Arity, First:! type, ... ⟬Second:! type, each Next:! «type; __N»⟭]
    (first: Vector(First), ... ⟬second: Vector(Second),
                                each next: Vector(each Next)⟭)
    -> (First, ... ⟬Second, each Next⟭);
// Pack expanding (in reverse)
fn F[__N:! Arity, First:! type, ... ⟬Second, each Next⟭:! «type; __N+1»]
    (first: Vector(First), ... ⟬second, each next⟭: ⟬Vector(Second), Vector(each Next)⟭)
    -> (First, ... ⟬Second, each Next⟭);
// Pack expanding (in reverse)
fn F[__N:! Arity, First:! type, ... ⟬Second, each Next⟭:! «type; __N+1»]
    (first: Vector(First), ... ⟬second, each next⟭: Vector(⟬Second, each Next⟭))
    -> (First, ... ⟬Second, each Next⟭);
// Pack renaming
fn F[__N:! Arity, First:! type, ... each __A:! «type; __N+1»]
    (first: Vector(First), ... each __a: Vector(each __A))
    -> (First, ... each __A);
```

This brings us back to a normal form, while reducing the number of tuple
segments. We can now repeat that process to merge the remaining parameter type:

```carbon
fn F[__N:! Arity, First:! type, ... ⟬each __A:! «type; __N+1»⟭]
    (first: Vector(First), ... each __a: Vector(⟬each __A⟭))
    -> (First, ... ⟬each __A⟭);
// Pack expanding
fn F[__N:! Arity, First:! type, ... ⟬each __A:! «type; __N+1»⟭]
    (first: Vector(First), ... each __a: ⟬Vector(each __A)⟭)
    -> (First, ... ⟬each __A⟭);
// Pack expanding
fn F[__N:! Arity, First:! type, ... ⟬each __A:! «type; __N+1»⟭]
    (first: Vector(First), ... ⟬each __a: Vector(each __A)⟭)
    -> (First, ... ⟬each __A⟭);
// Pack expansion splitting (in reverse)
fn F[__N:! Arity, ... ⟬First:! type, each __A:! «type; __N+1»⟭]
    (... ⟬first: Vector(First), each __a: Vector(each __A)⟭)
    -> (... ⟬First, each __A⟭);
// Pack expanding (in reverse)
fn F[__N:! Arity, ... ⟬First, each __A⟭:! «type; __N+2»⟭]
    (... ⟬first, each __a⟭: ⟬Vector(First), Vector(each __A)⟭)
    -> (... ⟬First, each __A⟭);
// Pack expanding (in reverse)
fn F[__N:! Arity, ... ⟬First, each __A⟭:! «type; __N+2»⟭]
    (... ⟬first, each __a⟭: Vector(⟬First, each __A⟭))
    -> (... ⟬First, each __A⟭);
// Pack renaming
fn F[__N:! Arity, ... __B:! «type; __N+2»⟭]
    (... __b: Vector(__B))
    -> (... __B);
```

Here again, this is a normal form, and there is demonstrably no way to perform
any further merging, so this must be the canonical form.

> **TODO:** define the algorithm in more general terms, and discuss ways that
> merging can fail.

## Alternatives considered

-   [Member packs](/proposals/p2240.md#member-packs)
-   [Single semantic model for pack expansions](/proposals/p2240.md#single-semantic-model-for-pack-expansions)
-   [Generalize `expand`](/proposals/p2240.md#generalize-expand)
-   [Omit `expand`](/proposals/p2240.md#omit-expand)
-   [Support expanding arrays](/proposals/p2240.md#support-expanding-arrays)
-   [Omit each-names](/proposals/p2240.md#omit-each-names)
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
-   [No parameter merging](/proposals/p2240.md#no-parameter-merging)
-   [Exhaustive function call typechecking](/proposals/p2240.md#exhaustive-function-call-typechecking)

## References

-   Proposal
    [#2240: Variadics](https://github.com/carbon-language/carbon-lang/pull/2240)
