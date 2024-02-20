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
referred to using "pack bindings", which are marked with the `each` keyword at
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

A _pack_ is a sequence of a fixed number of values called "elements", which may
be of different types. Packs are very similar to tuple values in many ways, but
they are not first-class values -- in particular, packs do not have types, and
no expression evaluates to a pack. The _arity_ of a pack is a compile-time value
representing the number of values in the sequence.

A pattern of the form "`...` _subpattern_" is called a _pack expansion pattern_.
It can only appear as part of a tuple pattern (or an implicit parameter list),
and it matches a sequence of tuple elements if each element matches
_subpattern_. Since _subpattern_ will be matched against multiple scrutinees (or
none) in a single pattern-matching operation, it cannot contain ordinary binding
patterns. However, it can contain _pack binding patterns_, which are binding
patterns that begin with `each`, such as `each ElementType:! type`. A pack
binding pattern can match an arbitrarily large number of times, and binds to a
pack consisting of all the matched values. A usage of a pack binding always has
the form "`each` _identifier_".

By default, a pack binding pattern can match any number of times, but `each` can
optionally be followed by "`(>=` _integer-literal_ `)`", which constrains the
binding to match at least that many times. This constraint also applies to all
other packs in the same pack expansion, and all other packs that are deduced to
have the same arity.

> **TODO:** The minimum-arity constraint syntax is a placeholder. Choose a final
> syntax.

The declared type of a pack binding can contain a usage of another pack binding,
but it must be a deduced parameter of the pattern.

> **Future work:** That restriction can probably be relaxed, but we currently
> don't have motivating use cases to constrain the design.

Note that the operand of `each` is always an identifier name or a binding
pattern, so it does not have a precedence. For example, the loop condition
`...and each iter != each vector.End()` in the implementation of `Zip` is
equivalent to `...and (each iter) != (each vector).End()`.

Usages of the `each` keyword (in other words, pack binding patterns and usages
of pack bindings) are called _expansion sites_. The _arity_ of an expansion site
is a symbolic value representing the arity of the underlying pack. Expansion
sites can only occur in the body of a _pack expansion_, which is an instance of
one of the following syntactic forms:

-   A statement of the form "`...` _statement_".
-   A tuple expression element of the form "`...` _expression_", with the same
    precedence as `,`.
-   A tuple pattern element of the form "`...` _pattern_", with the same
    precedence as `,`.
-   An implicit parameter list element of the form "`...` _pattern_", with the
    same precedence as `,`.
-   An expression of the form "`...` `and` _expression_" or "`...` `or`
    _expression_", with the same precedence as `and` and `or`.

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

In a pack expansion, the statement, expression, or pattern embedded in the
expansion is called the _body_ of the expansion. All sites of a given expansion
must have the same arity, which we will also refer to as the arity of the
expansion. If an expansion has no expansion sites, it must be a pack expansion
pattern, or an expression in the type position of a binding pattern, and its
arity is deduced from the scrutinee.

A pack expansion or `...expand` expression cannot contain another pack expansion
or `...expand` expression.

A pack binding cannot be used in the same pack expansion that declares it. In
most if not all cases, a binding that violates this rule can be changed to a
non-pack binding, because pack bindings are only necessary when you need to
transfer a pack from one pack expansion to another.

A pack expansion can be thought of as a kind of loop that executes at compile
time (specifically, monomorphization time), where the expansion body is
implicitly parameterized by an integer value called the _pack index_, which
ranges from 0 to one less than the arity of the expansion. The pack index is
implicitly used as an index into the expansion site packs. This is easiest to
see with statement pack expansions. For example, if `a`, `x`, and `y` are pack
bindings with arity 3, then `... each a += each x * each y;` is roughly
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

An expression of the form "`each` _identifier_", where _identifier_ names a pack
binding, evaluates to the `$I`th value that it was bound to (indexed from zero).

### Pattern matching

`...` expansions can also appear in patterns. The semantics are chosen to follow
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

A pack binding pattern binds the name to each of the scrutinee values, in order.

## Typechecking

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

In this model, at compile time a tuple value (or tuple type) consists of a sequence of _segments_, and a segment
consists of an expression called the _kernel_, an arity, and an _offset_. 
The kernel can have an arbitrary type, but the arity and offset have type `Count`.
We will use the notation `<K; A; O>` to
represent a segment with kernel `K`, arity `A`, and offset `O`, but note
that this is just for purposes of illustration; it isn't Carbon syntax.
Each segment represents a subsequence of elements that share a common structure.
The arity specifies the number of elements in that sequence, and the kernel
and offset specify the values of those elements: to determine the i'th element
of the sequence, we take the kernel and replace each mention of a pack
binding with the k'th element of the pack, where k is the sum of i and the
segment's offset. For example, if `P1` and `P2` are packs, the segment `<(P1, P2, X), 3, 2>` represents
the element sequence
`(P1[:2:], P2[:2:], X), (P1[:3:], P2[:3:], X), (P1[:4:], P2[:4:], X)`.

FIXME: explain how the transformation to element form can occur during
monomorphization (if not before). Maybe explain how this leads to the
conclusion that a segment is a symbolic value iff its properties are symbolic
values. I.e. a symbolic value is an expression which can be monomorphized.

If the kernel refers to any packs, they must all have the same arity, which we will
call the _kernel arity_.
A segment is _heterogeneous_ if its kernel refers to at least one pack, and
_homogeneous_ otherwise. The offset has no meaning on a homogeneous
segment; homogeneous segments are equal if they have equal kernels and arities,
whereas heterogeneous segments are equal only if their kernels, arities, and
offsets are all equal.
A segment `<K; A1+A2; O>` can be split into two consecutive segments
`<K; A1; O>, <K; A2; O+A1>` (if `A1` and `A2` are not negative) and conversely
two consecutive segments `<K; A1; O>, <K; A2; O+A1>` can be merged
into a single segment `<K; A1+A2; O>`.

A segment is _singular_ if its arity is known to be 1, and _variadic_
if its arity is unknown. In contexts where all segments are known to be
singular, we will sometimes refer to them as "elements". 

The kernel arity must be
greater than or equal to the sum of the segment's arity and offset. A segment
is _normalized_ if all of the following hold:
- The segment's offset is 0.
- Either the segment is homogeneous or its arity is equal to the kernel arity.
- If the segment's arity is a template constant, it is equal to 1.

A pack or tuple value can always be written as a sequence of normalized segments.
More precisely, any complete sequence of segments has the following invariants:
- If a segment's arity is a template constant, it is equal to 1.
- If a segment is heterogeneous, the sum of its arity and offset is less than or equal to the kernel
  arity.
- If a segment has the form `<K; A; O>` where `O` is not 0, it is immediately
  preceded by a segment `<K; B; P>` where `O` = `B+P`.
- If a segment has the form `<K; A; O>` where `A+O` is less than the kernel
  arity, it is immediately followed by a segment `<K; B; A+O>`.
FIXME: the above may not be complete, and some of the conditions apply only to
heterogeneous segments (is there a way to make them apply to homogeneous segments
as well?)

FIXME: An alternate representation might make the invariants more intuitive.
Maybe left-offset, arity, right-offset, which together sum to the kernel arity?
Or a more radical change: each segment covers the whole kernel arity, and it's
split into sub-segments (fragments?)? Or maybe we don't need sub-segments at
all? That doesn't quite work because of homogeneous segments (where the arity
is extrinsic).


The
_shape_ of a tuple type is the sequence of arities of its segments, so the shape
of the type of `z` is `(|x|, 1, |T|)`.

A segment
is a symbolic value only if the kernel doesn't name any packs, or the
segment's arity is equal to the arities of the packs named in the kernel.
(FIXME explain why?)

FIXME have we explained that packs are made of segments yet?

Segments are a generalized form of expression pack expansion: each segment
represents a kind of compile time loop, in which a notional "pack index" ranges
from the offset to the sum of the offset and the arity minus one, and on each
iteration the kernel is evaluated, using the pack index to select elements of
the pack bindings that the kernel refers to.

As a result, variable substitution works differently on segments: when the values
of a segment's pack bindings are known, we don't textually substitute them into
the segment's kernel. Instead, we perform pack expansion on the segment, which
evaluates the loop in terms of those pack values.

Pack expansion on a segment can be expressed symbolically: for example,
if a pack `P` consists of the segments `<i32; 1; 0>, <Vector(each T); X; 4>`,
and a pack `Q` consists of the segments `<f64; 1; 0>, <Optional(each U); X; 4>`,
then the segment `<(each P, Vector(each Q)); X+1; 0>` can be expanded to
`<(i32, Vector(f32)); 1; 0>, <(Vector(each T), Vector(Optional(each U))); X; 4>`.
This expansion actually consists of two steps: first, we split the source segment
into `<(each P, each Q); 1; 0>, <(each P, each Q); X; 1>` following the rule
described above, so that its shape
matches the shape of `P` and `Q` (which we will call the _substituted packs_).
In the next step, we iterate over the segments in parallel, substituting the kernels
from the `P` and `Q` segments into the kernel of the source segment,
and replacing the source segment's offset with the corresponding substituted
packs' offset.

To perform a symbolic pack expansion, all of the following conditions
must hold:
- The substituted packs must have the same shape, and corresponding segments
  of the substituted packs must have the same offsets.
- The source segment must have offset 0.
- The arity of the source segment must equal the sum of the arities of the
  segments in a substituted pack.
- All packs named in the kernel of the source segment must be substituted.

Recall that a symbolic value is an expression where, if we substitute
template constant values for its symbolic variables, the result will be a
template constant value. Since symbolic expansion is how we perform substitution
on a segment, that implies that a segment is a symbolic value if its
kernel, arity, and offset are symbolic values, and the above conditions are
guaranteed to hold.

FIXME: Pin down how that works with the constraint on the substituted packs--
does it go via their types? Maybe it follows from the sum-of-arities constraint
and some invariant of how we create and transform packs?
Candidate invariant: we never discard parts of a pack. If a heterogeneous segment
has a nonzero offset, it's always preceded by a segment it's contiguous with,
and if its offset plus arity is less than the arity of the kernel, it's
always followed by a segment it's contiguous with. Then maybe we can merge and then split
to make it true. Maybe "normalized" means those are all merged, so arity matches kernel arity?
FIXME: This isn't fully true, though -- user code could choose to violate that, like the
return type of `ZipRotate`. But maybe that's the point: within the signature,
that property isn't violated. It could be violated by the callsite, but we
don't let it. In other words, this invariant applies within a given scope.
Also, I think it only applies at compile time.
We could do something similar within a scope using local variables, but then
they're opaque, so you still never see packs that don't have this property


Segment expansion has an inverse operation, called _segment merging_, which
takes an expanded pack and a set of substituted packs and computes the
corresponding source segment. For example, given the definitions of `P` and
`Q` given earlier, the pack `<(i32, Vector(f32)); 1; 0>, <(Vector(each T), Vector(Optional(each U))); X; 4>`
can be merged into a single segment `<(each P, Vector(each Q)); X+1; 0>`.
FIXME: conditions for doing that
FIXME: degenerate case where there are no substituted packs, just a desired shape
FIXME: build on this somewhere by explaining the use case (where the substituted packs are concatenations)

Every pack expansion pattern, and every pack expansion expression in the type
position of a binding pattern, has a hidden deduced symbolic parameter that represents
its arity. Given a pack binding `B`, we will use `|B|` to represent the
deduced arity parameter of the expansion pattern that contains the definition
of `B`, but this is only for purposes of illustration; it isn't Carbon syntax.

So, continuing the earlier example, the type of `z` is represented symbolically
as
`(<i32; |x|; 0>, <f32; 1; 0>, <Optional(each T); |T|; 0>)`.

A hidden arity parameter can be thought of as an integer, but its type also records
the minimum possible value, which is typically 0, but can be
greater in some circumstances. For example, given this code:

```carbon
let (... each x: auto) = (x, ... each y, z);
```

The arity of `x` must be at least the minimum arity of `y`, plus 2, and that
fact will be recorded in the type of `|x|`. However, this applies only when
it is not possible to forward-declare `x` (such as if it is local to a function).
If `x` can be forward-declared, any forward-declaration of `x` also implicitly
declares `|x|`, and without an initializer for `x`, it cannot deduce any
minimum other than 0, so the defining declaration must have the same minimum to
avoid inconsistency.

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
`(<Vector(First); 1; 0>, <Vector(each Next); |Next|; 0>)`,
and we rewrite it to `(<Vector(each Arg); |Next|+1; 0>)` by introducing a new
deduced parameter `Next`. Recall that `|Next|` is shorthand for a
hidden deduced parameter, which is independent of `Next` even though it happens
to correspond to the arity of `Next`, so we can keep using `|Next|` even though
`Next` itself was eliminated as part of the rewrite. `|Next|` cannot be negative,
so the arity `|Next| + 1` correctly captures the fact that the parameter tuple
must have at least one element.

Notice that the original parameter type `(<Vector(First); 1; 0>, <Vector(each Next); |Next|; 0>)`
is the result of symbolically pack-expanding the rewritten type `(<Vector(each Arg); |Next|+1; 0>)`
when `Arg` is the concatenation of `First` and `each Next`, i.e.
`<First; 1; 0>, <each Next; |Next|; 0>`. The original return type
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
