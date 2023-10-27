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
        -   [Step 4: Representative deduction](#step-4-representative-deduction)
-   [Alternatives considered](#alternatives-considered)
-   [References](#references)

<!-- tocstop -->

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
fn Min[T:! Comparable & Value](... each(>=1) param: T) -> T {
  let (var result: T, ... each next: T) = (... each param);
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

### Generalized tuple types

The `...` operator lets us form tuples out of sequences whose size is not known
during typechecking. For example, in this code:

```carbon
fn F[... each T:! type]((... each x: i32), (... each(>=1) y: Optional(each T))) {
  let z: auto = (... each x, 0 as f32, ... each y);
}
```

The type of `z` is a tuple whose elements are an indeterminate number of
repetitions of `i32`, followed by `f32`, followed by a different indeterminate
number of types (but at least one) that all have the form `Optional(T)` for some
type `T`. We can't represent this as an explicit list of element types until
those indeterminate numbers are known, so we need a more general representation
for tuple types.

In this model, a tuple type consists of a sequence of _segments_, and a segment
consists of a type called the _representative_, and an arity, both of which may
be symbolic expressions. The arity has a type, which is an instance of one of
two generic types, `AtLeast(template N:! Core.BigInt)` and
`Exactly(template N:! Core.BigInt)`, which constrain the arity to be at least
`N` or exactly `N`, respectively (`N` cannot be negative). There is a subtype
relation between arity types: `Exactly(N)` and `AtLeast(N)` are subtypes of
`AtLeast(M)` if `M <= N`.

Each segment is associated with a special symbolic variable called the _pack
index_, which has type `AtLeast(0)` and is scoped to that segment's
representative. The pack index can only be used in the subscript of an indexing
expression on a pack, and only by adding it to an arity arithmetic expression
(see [below](#tuple-type-equality-and-segment-algebra)) called the _offset_,
which doesn't involve `$I`.

Every pack expansion pattern, and every pack expansion expression in the type
position of a binding pattern, has a hidden deduced parameter that represents
its arity.

These types, values, variables, and operations are notional, and are not
available to user code. For purposes of illustration, the notation `<R, A:T>`
represents a segment with representative `R` and arity `A` of type `T`, `$I`
represents the pack index, and given a pack binding `B`, `|B|` represents the
deduced arity parameter of the expansion pattern that contains the definition of
`B`, and `B[:$I:]` represents the `$I`th value bound by `B`.

So, continuing the earlier example, the type of `z` is represented symbolically
as
`(<i32, |x|:AtLeast(0)>, <f32, 1:Exactly(1)>, <Optional(T[:$I:]), |T|:AtLeast(1)>)`.

A segment is _variadic_ if its arity type is an instance of `AtLeast`, and
_singular_ if its arity type is `Exactly(1)` and its representative does not
refer to the pack index. In contexts where all segments are known to be
singular, we will sometimes refer to them as "elements". Segments are always
assumed to be normalized, meaning that every segment is either singular or
variadic. This is always possible because if a segment's arity is known to be
some fixed value N other than 1, we can replace it with N singular segments. The
_shape_ of a tuple type is the sequence of arities of its segments, so the shape
of the type of `z` is `(|x|:AtLeast(0), 1:Exactly(1), |T|:AtLeast(1))`.

As a notational convenience, if the arity type is omitted, it is understood to
be `Exactly(N)` if the arity value is an integer literal `N`, and `AtLeast(0)`
in all other cases. So the type of `z` could also be written as
`(<i32, |x|>, <f32, 1>, <Optional(T[:$I:]), |T|:AtLeast(1)>)`.

In order to index into a tuple with subscript `I`, the tuple type's segment
sequence must start with at least `I` singular segments, so that we can
determine the type of the indexing expression. Note that this rule applies only
to user-written subscript operations, not to the notional `[: :]` operations
introduced by the compiler when rewriting a pack expansion.

Similarly, in order to pattern-match a tuple pattern that does not contain a
pack expansion subpattern (and therefore contains a separate subpattern for each
element), the scrutinee tuple type's segments must all be singular.

#### Tuple type equality and segment algebra

Two generalized tuple types are _structurally equal_ if they have the same
number of segments and the segments have equal representatives, arities, and
arity types. They are _semantically equal_ if any instance of one is guaranteed
to be an instance of the other, and the other way around. Generalized tuple
types can be semantically equal without being structurally equal; for example,
these three types are semantically equal, but all structurally different:

-   `(<i32, |x|>, <f32, 1>, <Optional(T[:$I:]), |T|:AtLeast(1)>)`
-   `(<i32, |x|>, <f32, 1>, <Optional(T[:0:]), 1>, <Optional(T[:$I+1:]), |T|-1>)`
-   `(<i32, |x|>, <f32, 1>, <Optional(T[:$I:]), |T|-1:AtLeast(0)>, <Optional(T[:|T|-1:]), 1>)`

We can use a kind of type algebra to transform generalized tuple types to
different structural forms while preserving semantic equality.

Arity types can be added and subtracted according to the following rules:

-   `Exactly(M) + Exactly(N) == Exactly(M+N)`
-   `Exactly(M) + AtLeast(N) == AtLeast(M+N)`
-   `AtLeast(M) + Exactly(N) == AtLeast(M+N)`
-   `AtLeast(M) + AtLeast(N) == AtLeast(M+N)`
-   `Exactly(M) - Exactly(N) == Exactly(M-N)`
-   `Exactly(M) - AtLeast(N)` is ill-formed.
-   `AtLeast(M) - Exactly(N) == Exactly(M-N)`
-   `AtLeast(M) - AtLeast(N)` is ill-formed. (Recall that `M` and `N` are always
    template constants.)

Arity values can also be added and subtracted, with a result type determined by
applying the same operation to the operand types, so if we have `X:AtLeast(2)`
and `Y:Exactly(3)`, then the type of `X+Y` is `AtLeast(2) + Exactly(3)`, or
`AtLeast(5)`. These operations are valid if and only if the result type is
well-formed.

The **splitting rule**: A segment `<R, X+Y:Xt+Yt>` can be rewritten as a
sequence of two segments `<R, X:Xt>, <S, Y:Yt>`, where `S` is obtained by
substituting `$I + X` for every occurrence of `$I` in `R`. Notice that rewriting
the representative in this way preserves the invariant that a pack indexing
expression always indexes into a user-declared pack binding, with a subscript
that is always the sum of `$I` and an offset (or an offset alone).

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
as `(<A[:$I:], |A|>, <B[:$I:], |B|>)`.

Similarly, the typechecker sometimes needs to reason symbolically about values
with generalized tuple types (like the value of `(...each A, ...each B)`).
These, too, are represented as sequences of segments.

Since type packs are sequences of segments, typechecking must iterate over those
segments' representatives rather than over the (unknown) individual element
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
2. Adjust the shape of the scrutinee so that each singular pattern segment
   corresponds to a singular scrutinee segment.
3. Adjust the shape of the pattern so that each remaining scrutinee segment
   corresponds to a pattern segment with the same arity.
4. Deduce the representative of each pattern segment from the corresponding
   scrutinee segment.

Steps 1 and 4 are deduction steps: they apply information from the scrutinee
type to infer the unknown properties of the pattern type. Steps 2 and 3 do not
perform deduction; instead, they structurally transform the two types while
preserving semantic equality (using the algebraic rules defined
[earlier](#tuple-type-equality-and-segment-algebra)). The purpose of those steps
is to enable the deduction in step 4, which is valid only if each segment of the
pattern is guaranteed to match every element of the corresponding scrutinee
segment. Step 2 ensures that property for the singular pattern segments, and
step 3 ensures it for the variadic pattern segment.

#### Step 1: Arity deduction

Step 1 is straightforward: the sum of the arities of the pattern segments must
equal the sum of the arities of the scrutinee segments, which gives us an
equation that we can trivially solve for the unknown arity of the variadic
pattern segment. This is why we require that there is only one variadic pattern
segment: we cannot solve a single equation for more than one unknown.

This equation is expressed in terms of _typed_ arity values, and when we solve
for the arity of the variadic pattern segment, we are also deducing the arity
type of that segment. And as always, the deduced type must be a valid type, and
a subtype of the declared type. This requirement ensures that the scrutinee is
guaranteed to have at least as many elements as the pattern expects.

#### Step 2: Scrutinee adjustment

The purpose of step 2 is to support cases like the first line of the body of
`Min`:

```carbon
fn Min[T:! Comparable & Value](... each(>=1) param: T) -> T {
  let (var result: T, ... each next: T) = (... each param);
```

Here the pattern has type `(<T, 1>, <T, |next|:AtLeast(0))` and the scrutinee
has type `(<T, |T|:AtLeast(1)>)`, so we must transform the scrutinee to a form
that has a leading singular segment:

-   `(<T, |T|:AtLeast(1)>)`
-   = `(<T, 1+(|T| - 1):(Exactly(1) + AtLeast(0))>)` (arity arithmetic)
-   = `(<T, 1:Exactly(1)>, <T, |T|-1:AtLeast(0)>)` (splitting rule) (Note that
    in this example we don't need to rewrite the representative expressions in
    the last step, because the representative doesn't depend on `$I`).

In general, we can split out up to `N` leading or trailing singular segments
from a variadic segment that has arity type `AtLeast(N)`. So if the pattern has
`N` leading singular segments, and the scrutinee has `M` leading singular
segments where `M < N`, step 2 verifies that the first variadic scrutinee
segment's arity type is a subtype of `AtLeast(N-M)`, and splits out `N-M`
leading singular segments from it. It then applies the same procedure to last
variadic scrutinee segment to ensure there are enough trailing singular
segments.

The rewrites in step 2 do not take place when pattern matching as part of a
function call: we assume that physically separate function parameters are
logically separate as well, so if they are not separate at the callsite, that
likely indicates a bug.

#### Step 3: Pattern adjustment

Step 2 ensures that at the start of step 3 we have a subsequence of scrutinee
segments which contain all the elements that the variadic pattern segment will
match, and which cannot match any other pattern segment. Furthermore, the arity
of the pattern segment that we deduced in step 1 must be equal to the sum of the
arities of that scrutinee subsequence. As a result, step 3 can apply the
splitting rule to decompose the pattern segment into a sequence with the same
shape as the scrutinee subsequence.

Note that any pack indexing expression in the original variadic pattern segment
is guaranteed to have the form `B[:$I:]` (with no offset), where `B` is a
deduced pack binding. The decomposition in this step will rewrite that
expression in each segment to `B[:$I + A:]` (or `B[:A:]` if the segment is
singular), where `A` is the sum of the arities of the earlier segments in the
decomposition. This has the effect of decomposing the unknown contents of `B`
into a sequence of segments with unknown representatives that has the same shape
as the scrutinee subsequence.

#### Step 4: Representative deduction

Since the pattern and scrutinee now have identical shapes, each pattern segment
is guaranteed to match all of the elements of the corresponding scrutinee
segment. Consequently, we can perform type deduction segment-wise, deducing the
representative type of each pattern segment from the representative type of the
corresponding scrutinee segment.

Within a segment, type deduction takes place as normal. The only aspect that's
unique to variadics is that when unifying a pattern type expression
`B[:$I + A:]` or `B[:A:]` with a scrutinee expression `S`, we deduce that `S` is
the representative of the corresponding segment of `B`.

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
