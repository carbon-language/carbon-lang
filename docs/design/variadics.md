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
-   [Semantics](#semantics)
    -   [Expressions and statements](#expressions-and-statements)
    -   [Pattern matching](#pattern-matching)
-   [Typechecking](#typechecking)
    -   [Generalized tuple types](#generalized-tuple-types)
    -   [Iterative typechecking](#iterative-typechecking)
    -   [Pattern matching with generalized tuple types](#pattern-matching-with-generalized-tuple-types)
        -   [Identifying potential matchings](#identifying-potential-matchings)
        -   [The type-checking algorithm](#the-type-checking-algorithm)
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
none) in a single pattern-matching operation, it cannot contain ordinary
binding patterns. However, it can contain _pack binding patterns_, which are
binding patterns that begin with `each`, such as `each ElementType:! type`. A
pack binding pattern can match any number of times (including zero), and binds
to a pack consisting of all the matched values. A usage of a pack binding always
has the form "`each` _identifier_".

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
of lookahead, and the other meanings of `...` can be distinguished from each other by the context
they appear in. As a corollary, if the nearest enclosing delimiters around a
`...` are parentheses, they will be interpreted as forming a tuple rather than
as grouping. Thus, expressions like `(... each ElementType)` in the above
example are tuple literals, even though they don't contain commas.

By convention, `...` is always followed by whitespace, except that `...and`,
`...or`, and `...expand` are written with no whitespace between the two tokens.
This serves to emphasize that the keyword is not part of the expansion body, but
rather a modifier on the syntax and semantics of `...`.

In a pack expansion, the statement, expression, or pattern embedded in the
expansion is called the _body_ of the expansion, and a pack expansion body must
contain at least one expansion site. All sites of a given expansion must have
the same arity (which we will also refer to as the arity of the expansion).

A pack expansion cannot occur within another pack expansion.

A pack binding cannot be used in the same pack expansion that declares it. In
almost all cases, a binding that violates this rule can be changed to a non-pack
binding, because pack bindings are only necessary when you need to transfer a
pack from one pack expansion to another.

A pack expansion can be thought of as a kind of loop that executes at compile
time (specifically, monomorphization time), where the expansion body is
implicitly parameterized by an integer value called the _pack index_, which
ranges from 0 to one less than the arity of the expansion. The pack index is
implicitly used as an index into the expansion site packs. This is easiest to
see with statement pack expansions. For example, if `a`, `x`, and `y` are pack
bindings with arity 3, then `... each a += each x * each y;` is roughly
equivalent to

```carbon
for (let template i:! i32 in (0, 1, 2)) {
  a[:i:] += x[:i:] * y[:i:];
}
```

Here we are using `[::]` as a hypothetical pack indexing operator for purposes
of illustration; packs cannot actually be indexed in Carbon code. Note also that
this rewritten form would not typecheck under the usual rules, because the
expressions `a[:i:]`, `x[:i:]`, and `y[:i:]` may have different types depending
on the value of `i`.

`...and` and `...or` can likewise be interpreted as looping constructs, although
the rewrite is less straightforward because Carbon doesn't have a way to write a
loop in an expression context. An expression like `...and F(each x, each y)` can
be thought of as evaluating to the value of `result` after executing the
following code fragment:

```
var result: bool = true;
for (let template i:! i32 in (0, 1, 2)) {
  result = result && F(x[:i:], y[:i:]);
  if (result == false) { break; }
}
```

`...` in a tuple literal can't be modeled in terms of a code rewrite, because it
evaluates to a sequence of values rather than a singular value, but it is still
fundamentally iterative: an expression like `(... F(each x, each y))` evaluates
to a tuple whose `i`th element is `F(x[:i:], y[:i:])`.

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
//
// Note that this implementation is not recursive. We split the parameters into
// first and rest in order to forbid calling `Min` with no arguments.
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

## Semantics

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
constants, and all tuple elements are singular.

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
fn F[... each T:! type]((... each x: i32), (... each y: Optional(each T))) {
  let z: auto = (... each x, 0 as f32, ... each y);
}
```

The type of `z` is a tuple whose elements are an indeterminate number of
repetitions of `i32`, followed by `f32`, followed by a different indeterminate
number of types that all have the form `Optional(T)` for some type `T`. We can't
represent this as an explicit list of element types until those indeterminate
numbers are known, so we need a more general representation for tuple types.

In this model, a tuple type consists of a sequence of _segments_, and a segment
consists of a type called the _representative_, and an arity, both of which may
be symbolic. There is a special symbolic variable called the _pack index_, which
can only be used in the representative of a segment, and only as the subscript
of an indexing expression on a tuple. The pack index is used solely as a
placeholder, and never has a value. There are also symbolic variables
representing the arity of every pack binding. For purposes of illustration, the
notation `<V, N>` represents a segment with representative `V` and arity `N`,
`$I` represents the pack index, and given a pack binding `B`, `|B|` represents
the arity of `B`, and `B[:$I:]` represents the `$I`th value bound by `B`.

So, continuing the earlier example, the type of `z` is represented symbolically
as `(<i32, |x|>, <f32, 1>, <Optional(T[:$I:]), |T|>)`.

A segment is _variadic_ if its arity is unknown, and _singular_ if its arity is
known to be 1 and its representative does not refer to the pack index. In
contexts where all segments are known to be singular, we will sometimes refer to
them as "elements". Segments are always assumed to be normalized, meaning that
every segment is either singular or variadic. This is always possible because if
a segment's arity is known to be some fixed value N other than 1, we can replace
it with N singular segments. The _shape_ of a tuple type is the sequence of
arities of its segments, so the shape of the type of `z` is `(|x|, 1, |T|)`.

In order to index into a tuple with subscript `I`, the tuple type's segment
sequence must start with at least `I` singular segments, so that we can
determine the type of the indexing expression. Note that this rule applies only
to user-written subscript operations, not to the notional `[:$I:]` operations
introduced by the compiler when rewriting a pack expansion.

Similarly, in order to pattern-match a tuple pattern that does not contain a
pack expansion subpattern (and therefore contains a separate subpattern for each
element), the scrutinee tuple type's segments must all be singular.

### Iterative typechecking

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
representing packs as sequences of segments. For example, `Y` can be represented
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
-   Otherwise, we invent a local symbolic variable `N`, and all expansion sites
    are treated as having the shape `(N,)` (representing a single segment with
    unknown arity).

The type packs of expressions and patterns in the expansion body are then
determined by iterative typechecking: within the k'th typechecking iteration,
typechecking and symbolic evaluation behave as they do during non-variadic
typechecking, except that:

-   For every expression/pattern in the expansion, the type of the k'th segment
    of its type pack takes the place of its type (on both reads and writes).
    // - We reuse (rather than attempt to recompute) any type information that
    was // computed in the course of the shape-checking described above.
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

### Pattern matching with generalized tuple types

Tuple pattern matching involves unifying the type of the tuple pattern with the
type of the scrutinee. This becomes substantially more challenging when the
tuple types can contain variadic segments, because the two tuple types may not
have the same shape. As a consequence, we don't necessarily know which pattern
segments each scrutinee segment will match with, or the other way around. For
example, consider the following code:

```carbon
fn F(a: i32, ... each b: i32, c: i32);

fn G(... each x: i32) {
  F(1, 2, ... each x);
}
```

If `G` is called with no arguments, the `2` will match with `c`, and otherwise
the `2` will match with an element of `b`. Similarly, if `G` is called with one
or more arguments, the last argument will match `c`, and the remaining elements
(if any) will match values from `b`. However, at type-checking time we don't
know how many arguments the caller will pass, so we don't know which will occur.
On the other hand, the `1` will always match `a`.

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
fn H[each T:! type](a: i32, ... each b: each T, c: String) -> (... each T);

external impl P as ImplicitAs(i32);
external impl Q as ImplicitAs(String);

fn I((... each x: i32), (... each y: f32), (... each z: String)) {
  var result: auto = H(... each x, {} as P, ... each y, {} as Q, ... each z);
}
```

Here, the deduced type of `result` can have one of four different forms. The
most general case is `(<i32, |x|-1>, <P, 1>, <f32, |y|>, <Q, 1>, <String, |z|-1>)`,
and the other three cases are formed by omitting the first two and/or last two
segments (corresponding to the cases where `x` and/or `z` do not match any
arguments). Extending the type system to support deduction that splits into
multiple cases would add a fearsome amount of complexity to the type system.

#### Identifying potential matchings

Our solution will rely on being able to identify which segments of the pattern
can potentially match which segments of the scrutinee. We can do so as follows:

We will refer to the segments of the pattern as "parameters", and the segments
of the scrutinee value as "symbolic arguments". We will use the term "actual
arguments" to refer to the elements of the scrutinee after monomorphization, so
a single symbolic argument may correspond to any number of actual arguments,
including zero (but only if the expression is variadic). We will refer to
arguments as "singular" if they have arity 1, and "variadic" if they have
indeterminate arity (these are the only possibilities, because segments are
normalized). Similarly, we will refer to the `...` parameter as "variadic" and
the other parameters as "singular".

A tuple pattern consists of $N$ leading singular parameters, optionally followed
by a variadic parameter headed by the `...` operator, and then $M$ trailing
singular parameters. The scrutinee must have a tuple type, and can have any
number of singular and variadic segments, in any order.

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
    -   Otherwise, $E$ can match the $i$'th leading parameter, any later leading
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
more than one place (as `T` is in the earlier example of this problem), there
cannot be any leading or trailing variadic symbolic arguments (in the sense
defined in the previous section). This ensures that each symbolic argument can
only match one parameter, and so type deduction deterministically produces a
single result.

> **Open question:** Is that restriction too strict? If so, is it possible to
> forbid only situations that would actually cause type deduction to split into
> multiple cases. It would still disallow cases like the call to `H` above, but
> that call seems unnatural for reasons that seem closely related to the fact
> that its type splits into cases.

To avoid a combinatorial explosion, we will use a much more tractable
conservative approximation of the precise algorithm. We type-check each
parameter as follows:

-   If the parameter is variadic, we check it against the sub-sequence
    consisting of all symbolic arguments that it could potentially match. The
    rule stated above ensures that this is safe: the concatenated type can only
    become visible outside this local type-check if all variadic arguments have
    a known arity, in which case that sub-sequence is known to be exactly
    correct.
-   Otherwise:
    -   If there is only one argument it can match, which is also not variadic,
        we type-check the argument against the parameter in the ordinary way.
    -   Otherwise:
        -   If the parameter has a non-deduced type, we check each potential
            argument against that type.
        -   Otherwise, if the parameter has type `auto`, no further checking is
            performed (`auto` can deduce to different types on different
            iterations within a single segment).
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

## References

-   Proposal
    [#2240: Variadics](https://github.com/carbon-language/carbon-lang/pull/2240)
