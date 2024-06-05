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
    -   [Generalized tuple types](#generalized-tuple-types)
        -   [Tuple type equality and segment algebra](#tuple-type-equality-and-segment-algebra)
    -   [Iterative typechecking of pack expansions](#iterative-typechecking-of-pack-expansions)
    -   [Generalized tuple type deduction](#generalized-tuple-type-deduction)
        -   [Step 1: Arity deduction](#step-1-arity-deduction)
        -   [Step 2: Scrutinee adjustment](#step-2-scrutinee-adjustment)
        -   [Step 3: Pattern adjustment](#step-3-pattern-adjustment)
        -   [Step 4: Kernel deduction](#step-4-kernel-deduction)
-   [Alternatives considered](#alternatives-considered)
-   [References](#references)

<!-- tocstop -->

FIXME: Switch to new tuple indexing syntax

## Basics

A "pack expansion" is a syntactic unit beginning with `...`, which is a kind of
compile-time loop over sequences called "packs". Packs are initialized and
referred to using "each-names", which are marked with the `each` keyword at
the point of declaration and the point of use.

The syntax and behavior of a pack expansion depends on its context, and in some
cases by a keyword following the `...`:

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

TODO: Reconsider the statement that packs do not have types.

A _pack_ is a sequence of a fixed number of values called "elements", which may
be of different types. Packs are very similar to tuple values in many ways, but
they are not first-class values -- in particular, packs do not have types, and
no expression evaluates to a pack. The _arity_ of a pack is a compile-time value
representing the number of values in the sequence.

An _each-name_ consists of the keyword `each` followed by the name of a pack,
and can only occur inside a pack expansion. On the Nth iteration of the pack
expansion, an each-name refers to the Nth element of the named pack. As a result,
a pack binding pattern with an each-name, such as `each ElementType:! type`,
acts as a declaration of all the elements of the named pack, and thereby
implicitly acts as a declaration of the pack itself.

Note that `each` is part of the name syntax, not an expression operator, so it
binds more tightly than any expression syntax. For example, the loop condition
`...and each iter != each vector.End()` in the implementation of `Zip` is
equivalent to `...and (each iter) != (each vector).End()`.

FIXME: clean up all uses of "pack binding pattern", "each expression", "expansion site"

A pattern of the form "`...` _subpattern_" is called a _pack expansion pattern_.
It can only appear as part of a tuple pattern (or an implicit parameter list),
and it matches a sequence of tuple elements if each element matches
_subpattern_. Since _subpattern_ will be matched against multiple scrutinees (or
none) in a single pattern-matching operation, a binding patterns within a
pack expansion pattern must declare an each-name, and the Nth iteration of the
pack expansion will initialize the Nth element of the named pack from the
Nth scrutinee. The binding pattern's type expression may contain an each-name,
but if so, it must be a deduced parameter of the enclosing pattern.

> **Future work:** That restriction can probably be relaxed, but we currently
> don't have motivating use cases to constrain the design.

TODO: Move this passage up, so it doesn't come after discussion of pack expansion patterns?

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

An each-name cannot be used in the same pack expansion that declares it. In
most if not all cases, an each-name that violates this rule can be changed to an
ordinary name, because each-names are only necessary when you need to
transfer a pack from one pack expansion to another.

A pack expansion can be thought of as a kind of loop that executes at compile
time (specifically, monomorphization time), where the expansion body is
implicitly parameterized by an integer value called the _pack index_, which
ranges from 0 to one less than the arity of the expansion. The pack index is
implicitly used as an index into the packs referred to by each-names. This is easiest to
see with statement pack expansions. For example, if `a`, `x`, and `y` are packs
with arity 3, then `... each a += each x * each y;` is roughly
equivalent to

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

An each-name evaluates to the `$I`th value of the pack it refers to (indexed from zero).

### Pattern matching

The semantics of pack expansion patterns are chosen to follow
the general principle that pattern matching is the inverse of expression
evaluation, so for example if the pattern `(... each x: auto)` matches some
scrutinee value `s`, the expression `(... each x)` should be equal to `s`. These
semantics are implemented at monomorphization time, so all types are known
constants, and all all arities are known.

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

Packs, tuples, segments, shape
Typechecking within pack expansions
- Expressions/patterns have pack types
Pattern match type checking


## Background

Compile-time expression evaluation consists of applying reduction rules to
symbolic expressions. A _template constant value_ is an expression that
does not refer to any variables, and cannot be further reduced. A _symbolic value_
is an expression that may refer to symbolic bindings, but once those bindings'
template constant values are known, substituting them into the symbolic value
will produce a template constant value.

FIXME examples? Notation?

The type `Count` represents non-negative integers. It is used only at compile
time, to represent certain hidden quantities like arities, and it cannot be
referred to in user code.

The only arithmetic operation it supports is addition with other `Count` values,
and with integer literals, so a symbolic `Count` value is always a sum of
integer literals and the names of symbolic bindings of type `Count`. As a
result, a symbolic `Count` value can be thought of as a multiset of integer
literals and binding names, because the order is not significant, but the number
of repetitions is. If $X$ and $Y$ are `Count` values, $X < Y$ if the summands of
$X$ are a strict sub-multiset of the summands of $Y$, and the other ordering
relations can be defined similarly. This is not a total order, so for some pairs
of values none of these relations will hold. Note that these are not
operators within the Carbon language; they are mathematical relations
that we use for talking _about_ `Count` values.

FIXME examples?



### Generalized tuple types

The `...` operator lets us form tuples out of sequences whose size is not known
during typechecking. For example, in this code:

```carbon
fn F[... each T:! type]((... each x: i32), (... each y: Optional(each T))) {
  let z: auto = (... each x, 0 as f32, ... each y);
}
```

The type of `z` is a tuple whose elements are an indeterminate number of
repetitions of `i32`, followed by `f32`, followed by a different indeterminate
number of types that all have the form `Optional(T)` for some
type `T`. We can't represent this as an explicit list of element types until
those indeterminate numbers are known, so we need a more general representation
for tuple types.

===
FIXME: new approach: Heterogeneous segment must have the same shape as its kernel, and cover
whole kernel range; notation is `<kernel>`. Homogeneous segments need an
explicit shape; notation is `<kernel; shape>`. Note that singular segments are
homogeneous.

Splitting/merging are only allowed on homogeneous segments, in order to satisfy
that constraint. Otherwise, must perform substitution/extraction. Each segment
has a type, which is a segment with the same shape. As a result, substitution/extraction
must transform type segments the same way. In practice that usually that means
splitting/merging.

`{ }` is a pack, which is a sequence of segments. Can think of a pack as a
"vertical" sequence, as opposed to tuples, which are "horizontal". `...` converts
a pack to a tuple ("rotates" vertical to horizontal). A singuler tuple element `E`
can always be rewritten as `...{<E; 1>}`. `...{E1}, ...{E2}` can be rewritten
as `...{E1, E2}`.

===


In this model, at compile time a tuple value (or tuple type) consists of a sequence of _segments_.
A segment symbolically represents a
sequence of elements that share a common structure, represented by an expression
called the _kernel_ of the segment. The individual elements are computed
from the kernel at monomorphization time by applying pack expansion: the I'th element of the sequence is
found by replacing every each-name with the I'th element
of the named pack, and then evaluating the resulting expression.

Pack expansion can also be applied symbolically, expanding a segment into a
sequence of segments, using the segments rather than the elements of packs like `X`.
This will be described in more detail below.

A _shape_ is a kind of "structured arity": it represents the arity of a segment
in terms of the arities of the segments that it will be decomposed into by
symbolic pack expansion. This structure can be recursive: the arities of those
segments can themselves represented as shapes, if we have information about how
those segments will be expanded. Otherwise, they are represented as symbolic
bindings of type `Shape` (FIXME introduce that type or drop this), literal `1`s,
or sums thereof.
We will use a tuple notation to represent shapes; for example, `(1, (X, 1))` is
the shape of a segment that will be symbolically expanded into a singular
segment and a variadic segment, which will itself be expanded into a segment
with shape `X` followed by a singular segment. Note that there is no difference
between a shape consisting of a single element, and that single element in isolation --
in terms of tuple notation, shapes never have trailing commas.

The shape of a pack is a sequence consisting of the shapes of its segments. The
shape of a pack binding is the shape of its type. All pack bindings used in a
kernel must have the same shape, which is also the shape of the segment.

A _homogeneous_ segment is a segment whose kernel does not use any pack
bindings. As a result, the shape of a homogeneous segment is not implicit in its
kernel, but is an independent property of the segment. The elements of a
homogeneous segment all have the same value, which is the value of the kernel
expression. A segment that is not homogeneous is _heterogeneous_.

A _singular_ segment is a homogeneous segment where the arity is a template
constant value equal to 1. In contexts where all segments are known to be
singular, we will sometimes refer to them as elements. 

We will use the notation `<K>` to represent a heterogeneous segment with kernel
`K`, and `<K; S1, S2, S3>` to represent a homogeneous segment with kernel `K`
and shape `S1, S2, S3`, but note that this is just for purposes
of illustration; it isn't Carbon syntax.

Every segment has a type, which is a pack with the same shape as the segment.
A segment or pack binding is _uniform_ if its type consists of a single segment (FIXME ... that is uniform?). A homogeneous
segment's type is always a homogeneous segment, but a heterogeneous segment's
type may be a homogeneous segment, heterogeneous segment, or a pack of multiple
segments. 

FIXME explain how type is computed?
A homogeneous segment's type is a homogeneous segment with the same shape,
whose kernel is the type of the segment kernel.
In a heterogeneous segment, the type of the kernel is computed by ordinary
scalar typechecking, with placeholders for the types of `each` expressions, and
then pack-expanding that type expression using the types of the pack bindings.
Thus, a segment is uniform if the pack bindings it names are uniform.
I think that implies that a segment is uniform if its shape has no commas
(is a single element? shape is homogeneous? need better terminology).
FIXME: Maybe define uniform that way?
FIXME: maybe homogeneous segments have homogeneous shapes, which are sums rather
than tuples. Need to generalize "same shape" to something like "compatible shape"
though. But does that still meet the needs that motivated giving homogeneous
segments a shape to begin with? I think so.

FIXME: Can we extend this model to patterns? How?

We care about uniformity because we can type check a sequence of segments against
a uniform segment in O(N) time.


Pack-expanding a uniform segment with packs of uniform segments produces a pack of uniform segments
But the fact that we can do the expansion implies that the substituted packs have types that are
compatible with the uniform type of their bindings.
FIXME is this if and only if? Otherwise the below might be sufficient but not necessary.

What does that imply about merging?
Given a pack, we're trying to identify a uniform segment, and definitions for the
bindings it contains, that expand to the pack.
-> The segment must be uniform, so the bindings must be uniform (necessary and sufficient)
-> The starting pack must have uniform segments
-> The bound packs must have uniform segments

For typechecking purposes we usually want the resulting segment to be
equal to the original pack. 
That will be the case if for each binding, the binding's type segment equals the
bound pack's type. i.e. the binding's type segment expands to the bound pack's type.
Assuming the bound packs have more than one segment (i.e.
the merge is actually merging something), that can only happen if each bound
pack's segments have equal homogeneous types, or if we form the binding's
type via another layer of this process.

FIXME: No, this is too strict. Need to be able to accommodate stuff like
rewriting `F(a, b, ... each c)` to `F(... each args)` where they have types `A`, `B`, and `each C` that all satisfy some interface `I`.
ID-expressions for non-template bindings seem to play special role here, because of the fact that they're 
opaque. So a usage of `A` is just "an arbitrary instance of this type" except that
it's known to be equal to any other usage of `A`. We can drop the latter if the binding is otherwise unused, or if we can
perform the same transformation (with the same set of binding definitions!) on all other uses.
This does mean we can't perform the transformation greedily: we need to know exactly what span of
segments we must merge, because the validation of that property could be expensive.
On the other hand, for separate-compilation purposes it would be much cleaner
if caller and callee can agree on the underlying signature.

Possible way to think about this: bindings are opaque, so pack expansion normally
adds information, and inverse-expansion normally removes it. The special case
is if the pack binding value is formed from the concatenated names of bindings
that have the same phase as the binding, and equal types, and those names are not
used elsewhere after the transformation. Also, for an expression usage it's
sort of OK to lose information- can cause typechecking to fail spuriously, but
not succeed spuriously. But in a pattern context it's inverted, so we have to
be strict (maybe it's about deducing vs. non-deducing usage rather than pattern vs. expression?)

Probably this belongs in another section



In an expression context (i.e. arguments, not parameters), the pack type can
expand to a supertype of the segment types, but that means we're discarding
type information, so the result is usable in fewer contexts.
In pattern context it's the other way around: pack type can expand to a subtype
of the segment types, but result matches in fewer cases.







FIXME remember to discuss typechecking ...expand somewhere. Notably, do we
allow this kind of thing, where there's no way to replace `auto` with a
more specific type expression?
```
let (...each A: auto) = (...expand G());
```

FIXME have we explained that packs are made of segments yet?

As discussed above, segments are a generalized form of expression pack expansion: each segment
represents a kind of compile time loop, in which a notional "pack index" ranges
from 0 to the arity minus one, and on each iteration the kernel is evaluated,
using the pack index to select elements of the pack bindings that the kernel
refers to (if any).

As a result, variable substitution works differently on segments: when the values
of a segment's pack bindings are known, we don't textually substitute them into
the segment's kernel. Instead, we perform pack expansion on the segment, which
evaluates the loop in terms of those pack values. 



FIXME: notation for packs? e.g. curly braces?
FIXME: Rewrite this to only use `Vector` in one way, for clarity. Maybe pointer for the other?
Pack expansion on a segment can be expressed symbolically: for example,
if a pack `P` consists of the segments `<i32; 1>, <Vector(each T)>`,
and a pack `Q` consists of the segments `<f64; 1>, <Optional(each U)>`,
then the segment `<(each P, Vector(each Q))>` can be expanded to
`<(i32, Vector(f32)); 1>, <(Vector(each T), Vector(Optional(each U)))>`.

FIXME: explain that expansion applies simultaneously to the types. Maybe 
introduce a notation for a typed segment?

To perform a symbolic pack expansion, all uses of pack bindings in the kernel of the
source segment must be substituted, and all substituted pack values must have the same shape.

Recall that a symbolic value is an expression where, if we substitute
template constant values for its symbolic variables, the result will be a
template constant value. Since symbolic expansion is how we perform substitution
on a segment, that implies that a segment is a symbolic value if its
kernel is a symbolic value.
FIXME: we may not need any of that; the only nontrivial part is that the arity
is irrelevant (because we've established as an invariant that it's symbolic), which we don't
address here.

Segment expansion has an inverse operation, called _segment merging_, which
takes an expanded pack and a set of substituted packs and computes the
corresponding source segment. For example, given the definitions of `P` and
`Q` given earlier, the pack `<(i32, Vector(f32)); 1>, <(Vector(each T), Vector(Optional(each U)))>`
can be merged into a single segment `<(each P, Vector(each Q))>`.
FIXME: conditions for doing that: have to do the same thing to the types

FIXME: degenerate case where there are no substituted packs, just a desired shape
FIXME: build on this somewhere by explaining the use case (where the substituted packs are concatenations)

FIXME this probably needs to be much earlier:

The type of a pack binding is itself a pack, but the type can only consist of
multiple segments if the type is deduced from the initializer (currently this is
only possible with `auto`, but it would also apply to declarations like
`let [... each T:! type](... each t: T) = (...expand F());`, if Carbon supports
that). In all other cases, the type of a pack binding is a single segment, whose
kernel is the binding's declared type. If that kernel is homogeneous, the segment's arity
is the sum of the arities in the binding's initializer. If the initializer is not
visible at the point of declaration, we introduce a hidden deduced symbolic binding of type `Count`
to represent that unknown arity. As a consequence, if a pack binding has a
homogeneous type, its declaration must either have a visible initializer,
or occur in a context where deduced bindings are permitted (such as
a function signature).

Given a pack binding `B` whose initializer is not visible, we will use `|B|` as
a shorthand for the shape of `B`, but this is only for purposes of illustration;
it isn't Carbon syntax. (FIXME can/should we limit this to homogeneous-typed bindings?)

So, continuing the earlier example, the type of `z` is represented symbolically
as
`(<i32; |x|>, <f32; 0>, <Optional(each T)>)`.

FIXME should we explicitly talk about this example?

```carbon
let (... each x: i32) = (a, ... each b, c);
```

If a tuple contains a segment `<K; A; O>` such that, if `S` is the sum of
the arities of all earlier segments, `S` is known to be less than or equal to `I`, and
`S` + `A` is known to be greater
than `I`, then an indexing expression with subscript `I` symbolically reduces to `<K; 1; I-S>`. This most
commonly applies when `I` is a known concrete value, and the tuple begins
with at least `I`+1 singular segments. Note that it may be possible to
satisfy this criterion by merging segments.
FIXME: This might be more of a meta-rule that concrete reduction rules have to satisfy,
unless we can pin down symbolic arithmetic/comparison and reduction algorithm more precisely.
Maybe specify in terms of peeling off segments from the head/tail until we get
down to 1, then `(<K; A; O>).I` reduces to `<K; 1; O+I>`, then 2 segment reduction rules:
if homogeneous, reduce to `K`, otherwise reduce to `(... K).(O+I)`. Also,
represent symbolic indices as sums of symbolic indices. No subtraction or comparison per se,
just removing an element from the sum.

FIXME: Original motivation is to allow symbolic indexing
expressions that might not be valid, so long as they're not evaluated (e.g. for the
bound value of the type of a "first" parameter):

```carbon
fn ZipAtLeastOne[First:! type, ... each Next:! type]
      (first: Vector(First), ... each next: Vector(each Next))
      -> Vector((First, ... each Next));

fn F[... each T:! I](... each v: Vector(each T), w: Vector(i32)) -> auto {
  return ZipAtLeastOne(... each v, w);
}
```

====
Alternate approach

Matching `({<first: Vector(First); 1>, <each next: Vector(each Next)>})` against `({<each v>, <w; 1>})`.
Can't break down into segments, but
LHS is all binding patterns (guaranteed to match so long as they typecheck), so
we can model it as a single composite binding pattern.
So checking `({<Vector(First); 1>, <Vector(each Next)>})` against (i.e. is supertype of)
`({<Vector(each T)>, <Vector(i32); 1>})`.
By inverse pack expansion, checking `Vector({<First; 1>, <each Next>})`
against `Vector({<each T>, <i32; 1>})`, so checking `{<First; 1>, <each Next>}`
against `{<each T>, <i32; 1>}`. LHS is all deducing usages, which are like
binding patterns (guaranteed to match so long as they typecheck), so treat
as composite deducing usage, and bind `{<each T>, <i32; 1>}` to it.
Then, checking `{<type; 1>, <type; |T|>}` against `{<I; |T|>, <type; 1>}`.
Could check pairwise, but that leads to quadratic blow-up. Instead,
exploit homogeneity: `{<type; 1>, <type; |T|>}` -> `{<type; |T|+1>}`
-> `{<type; |T|>, <type; 1>}`. Then can match segment-wise.

Then the return type is `Vector((First, ...each Next))`, i.e.
`Vector(({<First; 1>, <each Next>}))`. We bound a value to `{<First; 1>, <each Next>}`,
so we can evaluate that to `Vector(({<each T>, <i32; 1>}))`

Variant: when we form the composite binding pattern, immediately try to rewrite
its type and type-of-type. Corresponds more closely to source code rewrite,
aligns better with codegen.

====

Checking `({<first: Vector(First); 1>, <each next: Vector(each Next)>})` against `({<each v>, <w; 1>})`.
So `first: Vector(First)` binds to `{<each v>, <w; 1>}.(0..1)`
and `each next: Vector(each Next)` binds to `{<each v>, <w; 1>}.(1..|v|+1)`.
We can't evaluate those until monomorphization. If we're OK with that, we still
need to verify they will typecheck. That means we're checking
`{<Vector(First)>}` against `{<Vector(each T)>, <Vector(i32)>}.(0..1)`
and `{<Vector(each Next)>}` against `{<Vector(each T)>, <Vector(i32)>}.(1..|T|+1)`.
By inverse pack expansion, we can reduce those to checking
`Vector({<First>})` against `Vector({<each T>, <i32>}.(0..1))`
and `Vector({<each Next>})` against `Vector({<each T>, <i32>}.(1..|T|+1))`.
So `First` binds to `{<each T>, <i32>}.(0..1)`
and `each Next` binds to `{<each T>, <i32>}.(1..|T|+1)`.
Again, we can't evaluate until monomorphization, by which point we don't care
because typechecking is over, but it might be OK if we don't need the value.
(Or we could declare that that _is_ a value, which is the branching-in-the-typesystem
option). We need to make sure this binding typechecks, so we're
checking `{<type; 1>}` against `{<I; |T|>, <type; 1>}.(0..1)`
and `{<type; |T|>}` against `{<I; |T|>, <type; 1>}.(1..|T|+1)`.
More precisely, we're checking that those right-hand expressions are subtypes
of the left-hand expressions. 
We can check all the cases, but that can go quadratic.


Then, the return type is `Vector((First, ...each Rest))`, i.e. `Vector((<First; 1>, <each Rest>))`.
Substitute (without evaluating!) to get
`Vector(({<each T>, <i32>}.(0..1), {<each T>, <i32>}.(1..|T|+1)))`.
which reduces to `Vector(({<each T>, <i32>}))` because the slices add up to the whole.

Uniform arguments case:

```carbon
fn G[... each T:! C1, V:! C3](... each x: each T, z: V);

fn H[A:! C1 & C2 & C3, ... each C:! C1 & C2 & C3]
    (a: A, ... each c: each C) {
  G(a, ... each c);
}
```

====
Alternate approach

Matching `({<each x: each T>, <z: V; 1>})` against `({<a; 1>, <each c>})`.
All binding patterns, so form composite binding pattern, then typecheck:
`{<each T>, <V; 1>}` against `{<A; 1>, <each C>}`. LHS is all binding
patterns, so form composite, then typecheck: `{<C1; |C|>, <C3; 1>}`
against `{<C1 & C2 & C3; 1>, <C1 & C2 & C3; |C|>}` -> `{<C1 & C2 & C3; |C| + 1>}`
-> `{<C1 & C2 & C3; |C|>, <C1 & C2 & C3; 1>}`, then check segmentwise.
Note this time we're exploiting homogeneity on the argument side.

variant: forming composite binding pattern fails because it's not uniform.
FIXME then what? need to form composite argument instead.

FIXME: does this capture the fact that this wouldn't work if the return type
were `A`?

====

Checking `({<each x: each T>, <z: V; 1>})` against `({<a; 1>, <each c>})`.
So `each x: each T` binds to `{<a; 1>, <each c>}.(0..|c|)`
and `z: V` binds to `{<a; 1>, <each c>}.(|c|..|c|+1)`.
As before, this is arguably a problem (can't evaluate until monomorphization).
Putting that aside, have to typecheck.
Checking `{<each T: C1>}` against `{<A; 1>, <each C>}.(0..|C|)`
and `{<V: C3; 1>}` against `{<A; 1>, <each C>}.(|C|..|C|+1)`.
Again, can't evaluate until monomorphization, but that will turn out to be OK.
Still need to typecheck.
Checking `{<C1; |C|>}` against `{<C1 & C2 & C3; 1>, <C1 & C2 & C3; |C|>}.(0..|C|)`
-> `{<C1 & C2 & C3; |C|+1>}.(0..|C|)` -> `{<C1 & C2 & C3; |C|>}`
and `{<C3; 1>}` against `{<C1 & C2 & C3; 1>, <C1 & C2 & C3; |C|>}.(|C|..|C|+1)`
-> `{<C1 & C2 & C3; |C|+1>}.(|C|..|C|+1)` -> `{<C1 & C2 & C3; 1>}`

Non-uniform case:

```carbon
fn F[X:! type, Y:! type, ... each Z:! type](x: X, y: Vector(Y), ... each z: Vector(each Z));
fn G[... each A:! type](... each a: Vector(each A), b: Vector(B)) {
  F(0 as i32, ... each a, b);
}
```

====
Alternate approach

Matching `{<x: X; 1>, <y: Vector(Y); 1>, <each z: Vector(each Z)>}` with
`{<0 as i32; 1>, <each a>, <b; 1>}`. We could eagerly match the leading segments
or form composite binding. Let's try the latter: checking
`{<X; 1>, <Vector(Y); 1>, <Vector(each Z)>}` against `{<i32; 1>, <Vector(each A)>, <Vector(B); 1>}`.
Can't form composite, or perform inverse pack expansion, so have to pair off
leading segments. Then remainder proceeds as above.

Since we had to split leading segments eventually, based purely on the parameter
list, there is no benefit to keeping them in the composite binding to begin with.
So we can model this as a purely signature-driven phase followed by a callsite-driven phase.
FIXME: No, I think it's more than that: the fact that we have to split means that
we _can't_ form composite binding in the first place.

====

Checking `{<i32; 1>, <Vector(each A)>, <Vector(B); 1>}` against
`{<X; 1>, <Vector(Y); 1>, <Vector(each Z)>}`.
-> checking `{<i32; 1>, <Vector(each A)>, <Vector(B); 1>}.(0..1)`
against `{<X; 1>}`,
checking `{<i32; 1>, <Vector(each A)>, <Vector(B); 1>}.(1..2)`
against `{<Vector(Y); 1>}`,
checking `{<i32; 1>, <Vector(each A)>, <Vector(B); 1>}.(2..|A|+2)`
against `{<Vector(each Z)>}`
-> etc

Key point is that we keep going until we reduce to an expression for each parameter (and then each deduced parameter),
driven purely by the structure of the parameter types. Content of argument type
is irrelevant except that reduction may succeed or fail

FIXME: example that merging approach rejects? e.g. `fn F(X, Y, each Z)` called
as `F(each A, B, C)` where all types are convertible to each other. Must reject
because it's quadratic and because we can't compute the witness tables at
typechecking time.

FIXME: pin down rules for inverse pack expansion in this framework

FIXME: how do we model the pre-merging approach in this framework?
Maybe we're checking against hypothetical most-general argument, and
infer constraints on it (i.e. infer a more general pattern) as we go?
Does this also solve the other problem, quadratic behavior?
What assumptions are we making in order to justify this?

Maybe start with uniform arguments case; seems a little simpler. Part of the
crux is that the signature doesn't actually use the deduced types; if it did,
we couldn't merge.




(<K1; A; O1>, S).(A + C) -> (S).(C)


In order to index into a tuple with subscript `I`, the tuple type's segment
sequence must start with at least `I` singular segments, so that we can
determine the type of the indexing expression.
FIXME: clarify how this works during symbolic evaluation, such that it can
support cases like matching `[T:! type](arg: T, ... each next: i32)` against
`(... each int_pack, 1 as i32)`

Similarly, in order to pattern-match a tuple pattern that does not contain a
pack expansion subpattern (and therefore contains a separate subpattern for each
element), the scrutinee tuple type's segments must all be singular.

#### Type expression evaluation

Consider this example:

```carbon
fn G[... each T:! C1, U:! C2, V:! C3](... each x: each T, y: U, z: V);

fn H[A:! C1 & C2 & C3, B:! C1 & C2 & C3, ... each C:! C1 & C2 & C3]
    (a: A, b: B, ... each c: each C) {
  let (... each Arg:! C1 & C2 & C3) = (A, B, ... each C);
  let (... each arg: each Arg) = (a, b, ... each c)
  G(... each arg);
}
```

Any possible monomorphization of this code would typecheck, so we would
like the generic version to typecheck as well. However, when we are typechecking
the call to `G`, we cannot compute a symbolic value for `V` to bind to: it
will be the result of `(A, B, ... each C).(|C|+1)`, which we can reduce to
`(B, ... each C).(|C|)` but can't reduce further until `|C|` is known
(which will not be until monomorphization).

To work around this problem, we will say that symbolic bindings bind to
symbolic _expressions_, not symbolic values, and are evaluated only when
the typechecker needs a symbolic value. As a result, the above code can typecheck, because
typechecking never needs the value of `V`.

FIXME move to proposal doc, since we're explaining a change, not a state?

The typechecker needs the symbolic value of an expression under circumstances
such as:

- When it needs to check that values are equal, pattern match on them, or
  query if a type satisfies a constraint. FIXME this is vague
  For example, if the last line of the example were
  `let result: B = G2(... each arg)`, the typechecker would need the symbolic
  value of `V` in order to check whether it is convertible to `B`.


> **Open question:** When else is a symbolic value needed?
>
> It is less clear what should happen if the last line were
> `let result: auto = G2(... each arg)',
> or if `H` discarded the return value of `G`. Does the typechecker
> evaluate the return type of every function call, or only when the return value
> is used, or only when the return value is used non-generically?

#### Function call rewriting

FIXME this section might belong somewhere else.

FIXME: new exposition approach: Need to coerce the shapes to be equal (why?). Segments are always "born"
with offset 0 and an unsplittable arity, so we merge as much as possible in
order to facilitate subsequent splitting.



Consider the following code:

```carbon
fn G[... each T:! C1, U:! C2, V:! C3](... each x: each T, y: U, z: V);

fn H[A:! C1 & C2 & C3, B:! C1 & C2 & C3, ... each C:! C1 & C2 & C3]
    (a: A, b: B, ... each c: each C) {
  G(a, b, ... each c);
}
```

Any possible monomorphization of this code would typecheck, so
we would like the generic version to typecheck, but we don't have a way to write
the expressions that the deduced parameters will bind to, because the arguments
and parameters don't line up structurally. (FIXME that doesn't really explain
the problem because they don't line up after the rewrite either. This also
seems to presuppose "comma significance")

However, we can fix that problem by
merging the arguments:

```carbon
fn H[A:! C1 & C2 & C3, B:! C1 & C2 & C3, ... each C:! C1 & C2 & C3]
    (a: A, b: B, ... each c: each C) {
  let (... each Arg:! C1 & C2 & C3) = (A, B, ... each C);
  let (... each arg: each Arg) = (a, b, ... each c)
  G(... each arg);
}
```

`<A; 1; 0>, <B; 1; 0>, <each C; |C|; 0>`



Consider the following code:

```carbon
fn ZipAtLeastOne[First:! type, ... each Next:! type]
      (first: Vector(First), ... each next: Vector(each Next))
      -> Vector((First, ... each Next));

fn F[... each T:! type](... each v: Vector(each T), w: Vector(i32)) -> auto {
  return ZipAtLeastOne(... each v, w);
}
```

We would like this code to typecheck, because any possible
monomorphization of it would typecheck, but there's no way to construct a
valid expression for the value that `First` binds to -- it could be either `i32`
or the first element of `each T`, depending on whether `each T` is empty
(which we don't know when we're typechecking `F`) (FIXME is that still accurate
with new model for symbolic bindings? If `(... each T, i32).0` is a valid expression
we need to explain this differently). However, consider this
alternative way of writing the declaration of `ZipAtLeastOne`:

```carbon
fn ZipAtLeastOne[... each Arg:! type](... each arg: Vector(each Arg))
      -> Vector((... each Arg));
```

This declaration avoids that problem, because we no longer need to worry about
extracting the type of the first parameter. Furthermore, this declaration is
equivalent to the original one from the caller's point of view, except that this
one fails to reject an empty argument list.

We can avoid even that problem if we
use the segment representation, leveraging the fact that it represents arity
explicitly. In that setting, the parameter type is
`(<Vector(First); 1>, <Vector(each Next)>)`,
and we rewrite it to `(<Vector(each Arg)>)` by introducing a new
deduced parameter `Arg`. 

Notice that the original parameter type `(<Vector(First); 1>, <Vector(each Next)>)`
is the result of symbolically pack-expanding the rewritten type `(<Vector(each Arg)>)`
when `Arg` is the concatenation of `First` and `each Next`, i.e.
`<First; 1>, <each Next>`. The original return type
is also the result of pack-expanding the rewritten return type with the same
value of `Arg`. 

FIXME: illustrate a case where return type merging fails

FIXME: example of rewriting with non-deduced-parameter. An argument rewrite
would work, but a parameter rewrite would help motivate integrated approach that
covers all cases.

FIXME: maybe we can unify by starting with the non-deduced case, then saying if
we replace a deducing usage, we have to replace all non-deducing usages, including
the definition of the replacement. Which makes the replacement tautological, so
it must itself be deduced. That also explains why all the segments must be
deducing if any of them are.





More generally, this kind of rewritten form is equivalent to
the original form if we can obtain the original form by applying symbolic pack expansion to the rewritten form, where the substituted packs are all deduced parameters
of the rewritten form, and each substituted pack's value is formed by
concatenating deduced parameters of the original form that have identical constraints.
Recall that symbolic pack expansion consists of splitting the expanded segment
to have the appropriate shape, followed by segment-by-segment substitution, so as a degenerate
case, this condition is satisfied if we can obtain the original form by
splitting segments of the rewritten form without applying any substitutions
(this enables us to do things like rewrite `fn F(x: i32, ... each y: i32)` to
`fn F(... each arg: i32)`).

This definition also points toward how we can compute such a rewritten form, by inverting
the process of pack expansion: given a sequence of segments representing the
parameter types, we simultaneously traverse the ASTs of
their kernels to compute the kernel of the rewritten parameter type. At each step:
- If the corresponding AST nodes are all identifier expressions that name
  deduced parameters, introduce a new deduced parameter to replace them,
  record that its value is equal to the concatenation of the original
  deduced parameters, and return an identifier expression naming the
  new parameter.
- If the corresponding AST nodes all have equal kinds, equal local states,
  and equal numbers of children, and recursively computing a rewritten form succeeds
  for each child, return a rewritten node with the same kind
  and local state, whose children are the rewritten children.
- Otherwise, fail.

We then rewrite any tuple types in the return type with a similar simultaneous
traversal of the ASTs of the segment kernels, except that rather than inferring
new deduced parameters from sequences of original deduced parameters, we
attempt to replace sequences of deduced parameters with the new deduced parameters
that were inferred in the previous step, and fail if we encounter any original
deduced parameters that cannot be rewritten in this way.

This algorithm can also be extended to rewrite a maximal subsequence of the
parameter types, rather than all of them, by keeping track of the beginning and
end of the candidate subsequence (initially all the segments), and narrowing
those bounds whenever doing so can prevent failure.

FIXME: Note that rewriting a declaration doesn't have to apply to the definition,
because they're equivalent.


FIXME: consider argument merging.

```
fn F[... each T:! C1, U:! C2, V:! C3](... each x: each T, y: U, z: V);

impl i32 as C1 ...;
impl i32 as C2 ...;
impl i32 as C3 ...;

fn G(a: i32, b: i32, ... each c: i32) {
  F(a, b, ... each c);
}

fn H[A:! C1 & C2 & C3, B:! C1 & C2 & C3, ... each C:! C1 & C2 & C3]
    (a: A, b: B, ... each c: each C) {
  F(a, b, ... each c);
}

fn J[D:! C1 & C2 & C3, E:! C1 & C2 & C3, ... each F:! C1 & C2 & C3]
    (d: D, e: E, ... each f: each F) {
  let (... each Q:! C1 & C2 & C3) = (D, E, ... each F);
  let (... each q:! ... each Q) = (d, e. ... each f);
  F(... each q);
}


fn F2[... each T:! C1, U:! C2, V:! C3](... each x: each T, y: U, z: V) -> V;

fn G2(a: i32, b: i32, ... each c: i32) -> i32 {
  return F2(a, b, ... each c);
}

fn H2[A:! C1 & C2 & C3, B:! C1 & C2 & C3, ... each C:! C1 & C2 & C3]
    (a: A, b: B, ... each c: each C) -> B {
  // Must reject because B only equals V when the pack is empty, non-generic.
  return F2(a, b, ... each c);
}

fn F3[T:! C1, U:! C2, ... each V:! C3](x: T, y: U, ...each z: each V);

fn G3(... each stuff: i32) {
  F3(... each stuff, 1 as i32, 2 as i32);
}
```

`H` is motivating example for non-homogeneous argument merging.
Type of `F` is `(<each T; |T|; 0>, <U; 1; 0>, <V; 1; 0>)`.
Initial argument type is `(<A; 1; 0>, <B; 1; 0>, <each C; |C|; 0>)`, where
`A`, `B`, and `C` have identical constraints.


FIXME Possibly this should go before ZipAtLeastOne?

Consider now this somewhat more abstract example:

```carbon
fn G[... each T:! C1, U:! C2, V:! C3](... each x: each T, y: U, z: V);

fn H[A:! C1 & C2 & C3, B:! C1 & C2 & C3, ... each C:! C1 & C2 & C3]
    (a: A, b: B, ... each c: each C) {
  G(a, b, ... each c);
}
```

Here again, any possible monomorphization of the code would typecheck, so
we would like the generic code to typecheck, but we don't have a way to write
the expressions that the deduced parameters will bind to, because the arguments
and parameters don't line up structurally. In this case we can't
fix that problem by merging the parameters, because they have different types
and constraints. However, we can fix the problem by merging the arguments:

```carbon
fn H[A:! C1 & C2 & C3, B:! C1 & C2 & C3, ... each C:! C1 & C2 & C3]
    (a: A, b: B, ... each c: each C) {
  let (... each Arg:! C1 & C2 & C3) = (A, B, ... each C);
  let (... each arg: each Arg) = (a, b, ... each c)
  G(... each arg);
}
```




=== OLD VERSION BELOW. NEW VERSION (02/07) ABOVE ===

FIXME: re-express this in terms of separate _merge_ (inverse of split) and
_change of variables_ (or _substitution_?) axioms?
Exposition approach: substitution makes the merge rule more powerful.
In the non-deduced case it's simple: you can let new variable be concatenation
of old ones, and rewrite in terms of that at will. In deduced case, there
are caveats: have to rewrite all uses of the old variables (since they are no
longer deduced), and ...


In principle, the splitting rule is invertible: if we have a sequence of two
segments `<R, X:Xt>, <[$I + X -> $I]R, Y:Yt>`, they can be rewritten as a single
segment `<R, X+Y:Xt+Yt>`. However, in practice that is very rarely directly applicable,
except in the degenerate case that if `R` does not contain any occurrences of `$I`,
we can rewrite `<R, X:Xt>, <R, Y:Yt>` as `<R, X+Y:Xt+Yt>`.

There are situations that aren't covered by that rule, where we would
nevertheless like to be able to merge adjacent segments. For example, consider
this version of `Zip` that requires at least one argument:

```carbon
fn ZipAtLeastOne[First:! type, ... each Next:! type]
      (first: Vector(First), ... each next: Vector(each Next))
      -> Vector((First, ... each Next));
```

From the caller's point of view, `ZipAtLeastOne` is just a function that takes one or more
arguments with arbitrary vector types -- there's nothing special about the
first parameter, other than the fact that it must exist. As a result, that
signature can almost be written so that there is only one parameter segment:

```carbon
fn ZipAtLeastOne[... each Param:! type](... each param: Vector(each Param))
      -> Vector((... each Param));
```

The only difference is that this simpler form can be called with no arguments,
which we could fix by making `param` have arity type `AtLeast(1)`.
We have no Carbon syntax for doing that, but that's not a barrier if this
rewrite is taking place in the compiler.

In effect, that rewrite works by merging `First` and `each Next` into a
composite variable `each Param`, so that `First` is
an alias for `Param[:0:]` and `each Next` is an alias for the remaining elements of `Param`
(i.e. `Next[:$I:]` is an alias for `Param[:$I + 1:]`), and `|Param|:AtLeast(1)` = `|Next| + 1`. As a result, the original
parameter tuple type `(<Vector(First), 1>, <Vector(Next[:$I:]), |Next|>)`
can be rewritten as `(<Vector(Param[:0:]), 1>, <Vector(Param[:$I + 1:]), (|Param|-1):AtLeast(0)>)`,
and then we can apply the inverted splitting rule to get
`(<Vector(Param[:$I:]), |Param|:AtLeast(1)>)`

Notice that we also applied a similar rewrite to the return type. That isn't just
because it makes the signature simpler: by rewriting the parameter tuple type,
we eliminated the deducing usages of `First` and `each Next`, so we have to
fully eliminate those variables from the signature, including their uses in
the return type. If it weren't possible to rewrite the return type in terms of
`Param`, we couldn't have introduced `Param` in the first place. For example,
there's no way to rewrite this function to have a single parameter segment,
even though its parameter list is identical:

```carbon
fn RotateZipAtLeastOne[First:! type, ... each Next:! type]
      (first: Vector(First), ... each next: Vector(each Next))
      -> Vector((... each Next, First));
```

FIXME explain why we're obligated to do the final inverted-split, even though we've
gotten rid of `First` and `each Next`.
- Connects back to discussion with Richard about symbolic values?
  `Vector((<Param[:$I:], |Param|>))` is a symbolic value, `Vector((<Param[:$I + 1:], |Param-1|>, <Param[:0:], 1>))`
  is not? What exactly is the problem?
  The `[:0:]`? The `+1` in `[:$I + 1:]`? The arity `|Param-1|`? The arity `1`? All of the above?
  Would a return type that mentioned only one of them still be bad?
  Yes, because the callsite might not be able to compute with it symbolically.
  `Vector((<Param[:$I:], |Param|>))` is a symbolic value because it's equivalent to
  `Vector((... each Param))`: the caller presumptively knows the symbolic value of `Param`,
  so they can substitute `Param` into that and the result will be a symbolic value.
  They don't know anything more granular than `Param`, because that's the whole point
  here: carving out a bounded exception to the rule that a `:!` binding (like `First` and `Next`)
  must bind to a symbolic value.
  I think that's a better way to approach this whole section: we're synthesizing
  "composite bindings" so that we can satisfy that rule




We will use the following notation to represent expression substitution:
- Let $V = V_1, V_2, ...$ be a sequence of metavariables (textual placeholders that
  will be rewritten, rather than Carbon variables).
- Let $S = S_1, S_2, ...$ be a sequence of Carbon expressions of the
  same size.
- Let $E$ be a Carbon expression that may contain metavariables
  from $V$.
Then $[V \mapsto S]E$ is a new expression in which, for all $1 \le i
\le |V|$, all occurrences of $V_i$ in $E$ have been replaced by $S_i$.






Using that notation, we can make the definition of "uniform" precise as follows:
a segment sequence $T = T_1, T_2, ...$ is uniform if there exists a Carbon
expression $E$, a sequence of metavariables $V = V_1, V_2, ...$
that are used in $E$, a $|T| \times |V|$ array of symbolic variables $P$,
and a $|T| \times |V|$ array of substituted
expressions $S$, and a non-negative initial offset `N` such that for all $1 \le i \le |T|$
and all $1 \le j \le |V|$:
- The kernel of $T_i$ is $[V \mapsto S_i]E$.
- If $T_i$ is variadic, $S_{ij}$ has the form " $P_{ij}$ `[:$I + N:]` " 
  if $i = 1$, or " $P_{ij}$ `[:$I:]` " if $i \gt 1$.
- If $T_i$ is singular, $S_{ij}$ is an identifier expression that names $P_{ij}$.
- For all $1 \le i' \le |T|$, $P_{ij}$ has the same type as $P_{i'j}$.
- If $P_{ij}$ is deducible in the context of $T$, then:
   - For all $1 \le i' \le |T|$, $P_{i'j}$ is deducible in the context of $T$,
     and has the same scope as $P_{ij}$.
   - $P_{ij}$ is unique in $P$, and is not named in $E$.
   - Usages of $P_{ij}$ outside the explicit parameter list only occur as part of a
     sequence of segments $T'$ that has the same shape as $T$, and there exists
     some Carbon expression $E'$ such that for all
     $1 \le i' \le |T'|$, the kernel of $T'_i$ is $[V \mapsto S_i]E'$.

We can then replace $T$ with a single segment by defining a new variadic Carbon
symbolic constant $R_j$ (for all $1 \le j \le |V|$) that can take the place of
the variables in $P_{ij}$. If $P_{0j}$ is not deducible in the context of $T$,
$R_j$ is bound to a sequence of segments with the same shape as $T$, where
the $i$ th segment's kernel is $P_{ij}$. If $P_{0j}$ is deducible in
the context of $T$, $R_j$ is declared in the same scope as $P_{0j}$, so that
it is deducible in the same way.

Then, $T$ is replaced with a single segment whose arity is the sum of the
arities of $T$, and whose kernel is $[V \mapsto R]E$, and any
segment sequence $T'$ (as defined earlier) is replaced with $[V \mapsto R]E'$
(for the $E'$ associated with that $T'$).

FIXME: This accommodates arguments as well as parameters, but should it?
Want to treat identical types as homogeneous. Almost anything else seems
like a problem because adding more type information could break it, which
seems undesirable (and maybe against established principles?)


### Iterative typechecking of pack expansions

Since the execution semantics of an expansion are defined in terms of a notional
rewritten form where we simultaneously iterate over the expansion sites, in
principle we can typecheck the expansion by typechecking the rewritten form.
However, the rewritten form usually would not typecheck as ordinary Carbon code,
because the expansion sites can have different types on different iterations.
Furthermore, the difference in types can propagate through expressions: if
`x[:$I:]` and `y[:$I:]` can have different types for different values of `$I`,
then so can `x[:$I:] * y[:$I:]`. In effect, we have to typecheck the loop body
separately for each iteration.

As a result, an expression or pattern in a pack expansion does not have a type,
it has a _type pack_ that represents the sequence of types it takes on over the
course of the iteration. Just as a pack is not quite a value, a type pack is not
quite a type, but type packs relate to packs in the same way that types relate
to values.

An expansion site's pack is composed of a sequence of elements from a tuple, so
its type pack consists of the types of those elements. However, as discussed
above, at typechecking time we don't necessarily know the type of each tuple
element, or even how many elements there are -- we only know the sequence of
segments. As a result, a type pack is represented as a sequence of segments.

In some cases, the typechecker may need to reason about the values contained in
a pack, as well as their types. For example:

```
fn f[...each A:! I1, ...each B:! I2]((... each a: each A), (...each b: each B)) {
  let (... each X:! type) = (...each A, ...each B);
  let (... each x: each X) = (...each a, ...each b);
}
```

In order to represent the declared type of `x`, the typechecker needs to model
the fact that `X` consists of the concatenation of `A` and `B`, even though it
does not yet know the arity or contents of either pack. It does this by
representing packs as sequences of segments. For example, `X` can be represented
as `(<A, 0..|A|>, <B, 0..|B|>)`.

Similarly, the typechecker sometimes needs to reason symbolically about values
with generalized tuple types (like the value of `(...each A, ...each B)`).
These, too, are represented as sequences of segments.

Since type packs are sequences of segments, typechecking must iterate over those
segments' kernels rather than over the (unknown) individual element
types. To ensure that this is valid, we require all sites of a given expansion
to have the same shape. We determine the shapes of the expansion sites as
follows:

-   If the expansion contains any non-deducing usages of pack bindings, the
    bindings they name must all have the same shape, and all other expansion
    sites are deduced to have the same shape.
-   Otherwise, the expansion must be a pattern pack expansion, and as discussed
    earlier, every pattern pack expansion is associated with a notional deduced
    symbolic variable representing its arity. We therefore treat all expansion
    sites as having a single segment with that arity. That arity will be deduced
    as part of type deduction, so the pattern must occur in a context where type
    deduction is permitted. For example, if it's part of a `let` statement, it
    must have an initializer.

The type packs of expressions and patterns in the expansion body are then
determined by iterative typechecking: within the k'th typechecking iteration,
typechecking and symbolic evaluation behave as they do during non-variadic
typechecking, except that:

-   For every expression/pattern in the expansion, the type of the k'th segment
    of its type pack takes the place of its type (on both reads and writes).
-   Any time we would use the type of an expression, and the expression is a
    usage of a pack binding, we instead use the type of the k'th segment of the
    binding it refers to (which has already been determined because of the rule
    that a pack binding pattern cannot be bound and used in the same pack
    expansion).
-   Any time we would evaluate a usage of a pack binding, we instead use the
    value of the k'th segment of the binding it refers to.
-   Any time we would deduce the value of a non-pack binding declared outside
    the expansion, it is an error if the value we deduce is different from the
    value deduced by any previous iteration. Note that this requirement does not
    apply when deducing a value for `auto`.
-   Any time we would deduce the value of a pack binding declared outside the
    expansion, we instead deduce the value of the k'th segment of that binding.

Once the body of a pack expansion has been typechecked, typechecking the
expansion itself is relatively straightforward:

-   A statement pack expansion requires no further typechecking, because
    statements don't have types.
-   An `...and` or `...or` expression has type `bool`, and every segment of the
    operand's type pack must have a type that's convertible to `bool`.
-   For a `...` tuple element expression or pattern, the segments of the
    operand's type pack become segments of the type of the enclosing tuple.

### Generalized tuple type deduction

The introduction of generalized tuple types complicates the design of type
deduction in a tuple pattern, because the pattern and the scrutinee can have
different shapes. We require that a tuple pattern contain no more than one
variadic segment, but the scrutinee may be an arbitrary combination of singular
and variadic segments.

Type deduction for a tuple pattern proceeds in four steps:

1. Deduce the arity of the variadic pattern segment.
2. Rewrite the pattern so that it has the same shape as the scrutinee.
3. Deduce the kernel of each pattern segment from the corresponding
   scrutinee segment.

FIXME: new approach:
1. Maximally merge parameters
2. Maximally merge arguments
3. Deal with singular argument/parameter pairs
4. Either the parameters or the arguments must consist of one variadic segment


Steps 1 and 3 are deduction steps: they apply information from the scrutinee
type to infer the unknown properties of the pattern type. Step 2 does not
perform deduction; instead, it structurally transform the pattern type while
preserving semantic equality (using the algebraic rules defined
[earlier](#tuple-type-equality-and-segment-algebra)). The purpose of step 2
is to enable the deduction in step 3, which is valid only if each segment of the
pattern is guaranteed to match every element of the corresponding scrutinee
segment.

#### Step 1: Arity deduction

Step 1 is straightforward: the sum of the arities of the pattern segments must
equal the sum of the arities of the scrutinee segments, which gives us an
equation that we can trivially solve for the unknown arity of the variadic
pattern segment. This is why we require that there is only one variadic pattern
segment: we cannot solve a single equation for more than one unknown.

The deduced arity must be expressed as a sum of non-negative values, such
as integer constants and the arities of scrutinee segments, in order to ensure
that the pattern segment does not have a negative arity. If this isn't possible,
that indicates that the scrutinee may have too few elements, so we raise an error.
FIXME: be more explicit about how we do this (basically, we're subtracting the
number of singular pattern segments, so we need to make sure the scrutinee has
that many singular segments)

#### Step 2: Pattern transformation

The goal of step 2 is to transform the pattern so that it has the same shape
as the scrutinee. We can do this by iterating simultaneously over the
pattern and scrutinee segments, maintaining the invariant that the first $i$
segments have the same arity (with $i$ initially 0). Let $A_i$ be the arity
of the $i$ th scrutinee segment.

For each $i$:
- While the arity of the $i$ th pattern segment is not greater than or equal to
  $A_i$:
  - Merge the segment with the segment after it, using the merging rule. If this
    is not possible, report an error.
- If the arity of the $i$ th pattern segment is greater than $A_i$, split
  it into two segments, where the first has arity $A_i$, using the splitting
  rule.

#### Step 3: Kernel deduction

Since the pattern and scrutinee now have identical shapes, each pattern segment
is guaranteed to match all of the elements of the corresponding scrutinee
segment. Consequently, we can perform type deduction segment-wise, deducing the
kernel type of each pattern segment from the kernel type of the
corresponding scrutinee segment.

Within a segment, type deduction takes place as normal. The only aspect that's
unique to variadics is that when unifying a pattern type expression
`B[:$I + A:]` or `B[:A:]` with a scrutinee expression `S`, we deduce that `S` is
the kernel of the corresponding segment of `B`.

FIXME Oh hell, I think this is where associating offsets with segments fails.
Suppose the pattern segment is `<P; |P|; 1>` and the scrutinee segment is
`<S; |S|; 0>`. We can't just unify `P` with `S`. Maybe we need to do the
splitting at the deduced-parameter level, like the merging? MIght not be enough
in the rare case of a non-deduced type pack. I think the right way to do this
is to treat the offset as context, e.g. if we're unifying
`<(P1, P2); X; 1>` with `<(S1, S2); X; 0>`, that recursively becomes
unifying `<P1; X; 1>` with `<S1; X; 0>` and `<P2; X; 1>` with
`<S2; X; 0>`. That naturally bottoms out in a piecewise definition of the
pattern parameters.

## Appendix

`<X; S>` is a _shape coercion_. `S` must be a shape, the shape of `X` must
be 1, and `X` cannot contain pack expansions.

A _name pack_ is a pack literal whose segments are names.

A pack expansion within a pattern has _fixed shape_ if it contains a usage of at least one
each-name that is not a parameter of the pattern; otherwise
it has _deduced shape_. Every pack expansion with deduced shape is
associated with a hidden _deduced shape binding_, which
symbolically represents the number of elements in the pack expansion, and
acts as a deduced parameter of the pattern. A shape
binding is usually written `|each P|`, where `each P` is an arbitrarily chosen
each-name that's bound by the pattern.

The values of shape bindings are _shapes_. A shape template constant is a
tuple of literal `1`s (i.e. a unary representation of the number of
elements), but we will normally be working with shape symbolic constants, which
are tuples of `1`s, the names of deduced shape bindings, and other shapes,
such as `((1, |each P|), |each Q|, 1)`.

Tuple and pack literals consist of _segments_ (the syntactic units separated by
commas).

Every expression, pattern, and statement has a shape, just like every expression
and pattern has a type. The shapes of expressions and statements are determined
as follows:
- The shape of a pack expansion is 1.
- The shape of a shape coercion is the value of the expression after the `;`.
- The shape of a pack literal is the concatenation of the shapes of its
  segments.
- The shape of a binding pattern that declares an each-name is deduced as
  described below.
- The shape of an each-name expression is the shape of the binding pattern that declared
  it.
- Any other AST node is _well shaped_ if there is some shape `S` such that all
  of the node's children have shape either 1 or `S`. When that condition holds,
  the shape of the node is `S` (or 1 if all children have shape 1).

Note that an AST's shape can only be variadic if it contains a pack literal,
shape coercion, or each-name.

The _shaped type_ of an expression `E` that has type `T` and shape `S` is
defined as follows:
- If the shape of `T` is `S`, the shaped type of `E` is `T`.
- Otherwise, the shaped type of `E` is `<T; S>`.

The type of an expression or pattern can be computed as follows:
- The type of a binding pattern with declared type `auto` is deduced as
  described below. The type of any other binding pattern is the expression
  following the `:`.
- The type of an each-name expression is the type of the binding pattern that
  declared it.
- The type of a shape coercion `<E; S>` is `<T; S>`, where `T` is the type of `E`.
- The type of a pack literal is a pack literal consisting of the shaped types of its
  segments.
- The type of a pack expansion expression or pattern is `...B`, where `B` is the
  shaped type of its body.
- The type of a tuple literal is a tuple literal consisting of the types
  of its segments.
- If an expression `E` contains a pack literal or shape coercion that is not nested
  inside a pack expansion, the type of `E` is found by repeatedly applying
  pack or coercion lifting (see below) until neither reduction is applicable,
  and then computing the type of the result.

> **TODO:** address `...expand`, `...and` and `...or`.

### Reduction rules

Unless otherwise specified, all expressions in these rules
must be free of side effects. Note that every reduction rule is also an
equivalence: the expression before the reduction is equivalent to the
expression after.

Expressions that are reduced by these rules must be well-shaped (and the reduced
form will likewise be well-shaped), but need not be well-typed.

*Empty pack removal:* `...{}` reduces to the empty string.

*Singular expansion removal:* `...E` reduces to `E`, if `E` contains no
pack literals, shape coercions, or each-names.

*Pack expansion splitting:* If `E` is a segment and `S` is a sequence of
segments, `...{E, S}` reduces to `...E, ...{S}`.

*Pack lifting:* If `F` is a function, and `X` is an expression that does not contain
pack literals, then
`F({A1, A2}, X, {B1, B2}, <Y; S>)` reduces to `{F(A1, X, B1, Y), F(A2, X, B2, Y)}`.
This rule generalizes in several dimensions:
- `F` can have any number of non-pack-literal arguments, and any positive number of
  pack literal arguments, and they can be in any order.
- The pack literal arguments can have any number of segments, so long as they
  all have the same number of segments.
- `F()` can be any expression syntax other than `...`, not just a function call. For example,
  this rule implies that `{X1, X2} * {Y1, Y2}`
  reduces to `{X1 * Y1, X2 * Y2}`, where the `*` operator
  plays the role of `F`.

*Coercion lifting:* If `F` is a function, `S` is a shape, and `Y` is an
expression that does not contain pack literals or shape coercions,
`F(<X; S>, Y, <Z; S>)`
reduces to `<F(X, Y, Z); S>`. As with pack lifting, this rule generalizes:
- `F` can have any number of non-shape-coercion arguments, and any positive
  number of shape coercion arguments, and they can be in any order.
- `F()` can be any expression syntax other than `...` or pack literal formation,
  not just a function call.

*Coercion removal:* `<E; 1>` reduces to `E`.

Claim: as a corollary of these rules, in a symbolic value (which by definition
is fully reduced), a pack literal can only occur as the direct child of a
pack expansion, and a shape coercion can only occur as the direct child of
a pack expansion or a segment of a pack literal.

### Other equivalences

Unless otherwise specified, all expressions in these rules
must be free of side effects.

*Coercion merging:*
- `<E; M>, <E; N>` is equivalent to `<E; M, N>`.
- `<E; N>, E` is equivalent to `<E; N, 1>`.
- `E, <E; N>` is equivalent to `<E; 1, N>`.

*Coercion shape commutativty:* `<E; S1>` is equivalent to
`<E; S2>` if `S1` is a permutation of `S2`.

### Convertibility and deduction rules

Formalism: we are checking if some scrutinee type is convertible to some pattern
type, by applying a set of deduction rules for the "convertible to" and "deducible from"
relations. For example:

- `T` is convertible to `U` if `T` implements `ImplicitAs(U)`.
- `T` is convertible to `U` if `U` is deducible from `T`.
- Let `P` be a parameterized type. `P(A, B)` is deducible from `P(C, D)` if `A`
  is deducible from `C` and `B` is deducible from `D`.
- For any `X`, `X` is deducible from `X`.

Formally, these rules all implicitly propagate a set of bindings for deduced
parameters of the pattern (such as "`X` is bound to the value `Y`"). For
example, the third rule above can be stated more precisely as

- Let `P` be a parameterized type. If `A` is deducible from `C` given a set of
  bindings `S1`, and `B` is deducible from `D` given a set of bindings
  `S2`, then `P(A, B)` is deducible from `P(C, D)` given a set of bindings
  `S1` ∪ `S2`.

In almost all cases, the set of bindings in the conclusion is the union of the
sets of bindings in the premises, and in those cases we will usually leave
the bindings implicit (as in the original formulation of the above rule).
One major exception is the following rule:

*Binding introduction:* Let `X` be a deduced parameter with type `T`. If the
type of `E` is convertible to `T` given a set of bindings `B`, then `X` is
deducible from `E` given a set of bindings `B` ∪ "`X` is bound to `E`".

A set of bindings must be _consistent_, which in particular means that all
bindings for a given variable must bind it to the same value.

Variadic extensions:

A _synthetic deduced parameter_ is a pack literal whose segments are all
deduced parameters of the enclosing pattern, and whose type can be written with
a single segment. A synthetic deduced parameter behaves like an ordinary
deduced parameter when applying the binding introduction rule, and
consequently it can be bound to a value. However, a binding set is inconsistent
if it contains bindings for both a synthetic deduced parameter and one of its
constituent names.

`...T` is convertible to `...U` if `T` is convertible to `U`.

`<T; S1>` is convertible to `<U; S2>` if `T` is convertible to `U` and
`S2` is deducible from `S1`.

Let `T` and `U` be tuple segments, and let `Ts` and `Us` be sequences of tuple
segments. `(T, Ts)` is convertible to `(U, Us)` if `T` is convertible to `U` and
`Ts` is convertible to `Us`. Note that `(T, Ts)` and `(U, Us)` are not required
to be symbolic values, and in particular `T` and `U` can contain pack literals,
so there may be many equivalent ways to write a given tuple, which will be
decomposed differently by this rule.

The _inner value_ of a pack literal segment is the result of replacing
every shape coercion `<E; S>` with `E`.

Let `T` be a pack literal segment and `Us` be a sequence of pack literal
segments. `{T}` is convertible to `{Us}` if the shape of `{Us}` is deducible from the
shape of `{T}`, and the inner value of `T` is convertible to the inner value
of each segment of `Us`.

Let `Ts` be a sequence of pack literal segments and `U` be a pack literal
segment. `{Ts}` is convertible to `{U}` if the shape of `{U}` is deducible from
the shape of `{Ts}`, and the inner value of each segment of `Ts` is convertible
to the inner value of `U`.

Let `X` be a deduced shape binding, and let `S` be a shape expression.
`X` is deducible from `S` given a set of bindings that consists of the
single element "`X` is bound to `S`".

Let `(S1)`, `(S2)`, `(S3)`, and `(S4)` be shapes. `(S1, S2)` is deducible from
`(S3, S4)` if `(S1)` is deducible from `(S3)` and `(S2)` is deducible from
`(S4)`.

### Deduction algorithm

A _full pattern_ consists of an optional deduced parameter list, followed by
a pattern, optionally followed by a return type expression.

A function type (or other pattern type) is in _deducing form_ if 
- The only pack literals are uniform name packs, where:
  - All names in a name pack are deduced parameters of the pattern, exactly one is an each-name,
    and no name repeats in the pack.
  - A name in a name pack never appears in the full-pattern outside a name pack
  - All name packs in the full-pattern containing a given name are equal.
- Every shape coercions is the direct child of a pack expansion.

Notice that the type of any user-written function can be trivially expressed
in deducing form, because it contains no pack literals.

The _canonical form_ of a function type (or other pattern type) is the unique
deducing form that is "maximally merged", meaning that if `C` is the canonical
form, and `D` is any other deducing form of the function type, then either
`D` has more pack expansions than `C`, or
- `D` has the same number of pack expansions as `C`,
- the shape of every pack expansion in `D` is a prefix or suffix of the shape
  of the corresponding pack expansion in `C`, and
- in at least one case, it is a _strict_ prefix or suffix.

> **TODO:** specify algorithm for converting a function type to canonical form,
> or establishing that there is no such form. See next section for a start.

> **TODO:** specify algorithm for finding a valid deduction from the argument
> type to the canonical form of the function type.

> **Future work:** Extend this approach to support merging the argument list
> as well as the parameter list.

#### Canonicalization algorithm

The canonical form can be found by starting with a deducing form,
and incrementally merging an adjacent singular parameter type into the variadic
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

We wrap `each Next` in a pack literal to express it in deducing form:

```carbon
(Vector(First), Vector(Second), ... Vector({each Next})) -> (First, Second, ... {each Next})
```

Then we attempt to merge `Vector(Second)` into the pack expansion:

```carbon
// Pack lifting
(Vector(First), Vector(Second), ... {Vector(each Next)}) -> (First, Second, ...{each Next})
// Pack expansion splitting (in reverse)
(Vector(First), ... {Vector(Second), Vector(each Next)}) -> (First, ...{Second, each Next})
// Pack lifting (in reverse)
(Vector(First), ... Vector({Second, each Next})) -> (First, ...{Second, each Next})
```

Then we must verify that the type of the name pack `{Second, each Next}` is
uniform, to show that this is a deducing form. By the typing rules given
earlier, the type of that pack is `{type, <type; |Next|>}`, and we can transform
that type to `{<type; 1, |Next|>}` by coercion merging. This shows that we have
reached a deducing form. We can then repeat that process to merge the remaining
parameter type:

```carbon
(Vector(First), ... Vector({Second, each Next})) -> (First, ...{Second, each Next})
// Pack lifting
(Vector(First), ...{Vector(Second), Vector(each Next)}) -> (First, ...{Second, each Next})
// Pack expansion splitting (in reverse)
(...{Vector(First), Vector(Second), Vector(each Next)}) -> (...{First, Second, each Next})
// Pack lifting (in reverse)
(...Vector({First, Second, each Next})) -> (...{First, Second, each Next})
```

The type of `{First, Second, each Next}` is `{type, type, <type; |Next|>}`,
which we can rewrite to `{<type; 1, 1, |Next|>}` using coercion merging, as
before. We have thus expressed the type of `F` in normal form with only one parameter
segment, so this must be the canonical form.

> **TODO:** define the algorithm in more general terms, and discuss ways that merging
> can fail.


## Alternatives considered

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
