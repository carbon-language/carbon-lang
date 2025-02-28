# Variadics

<!--
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

[Pull request](https://github.com/carbon-language/carbon-lang/pull/2240)

<!-- toc -->

## Table of contents

-   [Abstract](#abstract)
-   [Problem](#problem)
-   [Background](#background)
-   [Proposal](#proposal)
    -   [Examples](#examples)
    -   [Comparisons](#comparisons)
-   [Rationale](#rationale)
-   [Alternatives considered](#alternatives-considered)
    -   [Member packs](#member-packs)
    -   [Single semantic model for pack expansions](#single-semantic-model-for-pack-expansions)
    -   [Generalize `expand`](#generalize-expand)
    -   [Omit `expand`](#omit-expand)
    -   [Support expanding arrays](#support-expanding-arrays)
    -   [Omit each-names](#omit-each-names)
        -   [Disallow pack-type bindings](#disallow-pack-type-bindings)
    -   [Fold expressions](#fold-expressions)
    -   [Allow multiple pack expansions in a tuple pattern](#allow-multiple-pack-expansions-in-a-tuple-pattern)
    -   [Allow nested pack expansions](#allow-nested-pack-expansions)
    -   [Use postfix instead of prefix `...`](#use-postfix-instead-of-prefix-)
    -   [Avoid context-sensitity in pack expansions](#avoid-context-sensitity-in-pack-expansions)
        -   [Fold-like syntax](#fold-like-syntax)
        -   [Variadic blocks](#variadic-blocks)
        -   [Keyword syntax](#keyword-syntax)
    -   [Require parentheses around `each`](#require-parentheses-around-each)
    -   [Fused expansion tokens](#fused-expansion-tokens)
    -   [No parameter merging](#no-parameter-merging)
    -   [Exhaustive function call typechecking](#exhaustive-function-call-typechecking)

<!-- tocstop -->

## Abstract

Proposes a set of core features for declaring and implementing generic variadic
functions.

A "pack expansion" is a syntactic unit beginning with `...`, which is a kind of
compile-time loop over sequences called "packs". Packs are initialized and
referred to using "each-names", which are marked with the `each` keyword at the
point of declaration and the point of use.

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

## Problem

Carbon needs a way to define functions and parameterized types that are
_variadic_, meaning they can take a variable number of arguments.

## Background

C has long supported variadic functions through the "varargs" mechanism, but
that's heavily disfavored in C++ because it isn't type-safe. Instead, C++
provides a separate feature for defining variadic _templates_, which can be
functions, classes, or even variables. However, variadic templates currently
suffer from several shortcomings. Most notably:

-   They must be templates, which means they cannot be definition-checked, and
    suffer from a variety of other costs such as needing to be defined in header
    files, and code bloat due to template instantiation.
-   It is inordinately difficult to define a variadic function whose parameters
    have a fixed type, and the signature of such a function does not clearly
    communicate that fixed type to readers.
-   The design encourages using recursion rather than iteration to process the
    elements of a variadic parameter list. This results in more template
    instantiations, and typically has at least quadratic overhead in the size of
    the pack (at compile time, and sometimes at run time). In recent versions of
    C++ it is also possible to iterate over packs procedurally, using a
    [fold expressions](https://en.cppreference.com/w/cpp/language/fold) over the
    comma operator, but that technique is awkward to use and not widely known.

There have been a number of C++ standard proposals to address some of these
issues, and improve variadic templates in other ways, such as
[P1219R2: Homogeneous variadic function parameters](https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2019/p1219r2.html),
[P1306R1: Expansion Statements](http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2019/p1306r1.pdf),
[P1858R2: Generalized Pack Declaration and Usage](http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2020/p1858r2.html),
and
[P2277R0: Packs Outside of Templates](http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2021/p2277r0.html).
However, C++ has chosen not to pursue definition-checking even for non-variadic
functions, so definition-checked variadics seem out of reach. The most recent
proposal to support fixed-type parameter packs was
[rejected](https://github.com/cplusplus/papers/issues/297). A proposal to
support iterating over parameter packs was inactive for several years, but has
very recently been [revived](https://github.com/cplusplus/papers/issues/156).

Swift supports
[variadic parameters](https://docs.swift.org/swift-book/documentation/the-swift-programming-language/functions/#Variadic-Parameters)
so long as all elements have the same type, and has recently approved
[SE-0393: Value and Type Parameter Packs](https://github.com/apple/swift-evolution/blob/main/proposals/0393-parameter-packs.md),
which adds support for definition-checked, heterogeneous variadic parameters
(with a disjoint syntax).
[SE-0404: Pack Iteration](https://github.com/simanerush/swift-evolution/blob/se-0404-pack-iteration/proposals/0404-pack-iteration.md),
which extends that to support iterating through a variadic parameter list, has
been positively received, but hasn't yet been approved.

There have been several attempts to add such a feature to Rust, but that work is
[currently inactive](https://github.com/rust-lang/rfcs/issues/376#issuecomment-830034029).

## Proposal

See `/docs/design/variadics.md` in this pull request.

### Examples

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
fn Min[T:! Comparable & Value](var result: T, ... each next: T) -> T {
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

### Comparisons

The following table compares selected examples of Carbon variadics against
equivalent code written in C++20 (with and without the extensions discussed
[earlier](#background)) and Swift.

<table>
<tr>
<th>Carbon</th>
<th>C++20</th>
<th>C++20 with extensions</th>
<th>Swift with extensions</th>
</tr>

<tr>
<td>

```carbon
// Computes the sum of its arguments, which are i64s
fn SumInts(... each param: i64) -> i64 {
  var sum: i64 = 0;
  ... sum += each param;
  return sum;
}
```

</td>
<td>

```cpp
template <typename... Params>
    requires (std::convertible_to<Params, int64_t> && ...)
int64_t SumInts(const Params&... params) {
  return (static_cast<int64_t>(params) + ... + 0);
}
```

</td>
<td>

With P1219R2:

```C++
int64_t SumInts(int64... params) {
  return (static_cast<int64_t>(params) + ... + 0);
}
```

</td>
<td>

(No extensions)

```swift
func SumInts(_ params: Int64...) {
  var sum: Int64 = 0
  for param in params {
    sum += param
  }
  return sum
}
```

</td>
</tr>
<tr>
<td>

```carbon
fn Min[T:! Comparable & Value](first: T, ... each next: T) -> T {
  var result: T = first;
  ... if (each next < result) {
    result = each next;
  }
  return result;
}
```

</td>
<td>

```cpp
template <typename T, typename... Params>
    requires std::totally_ordered<T> && std::copyable<T> &&
             (std::same_as<T, Params> && ...)
T Min(T first, Params... rest) {
  if constexpr (sizeof...(rest) == 0) {
    // Base case.
    return first;
  } else {
    T min_rest = Min(rest...);
    if (min_rest < first) {
      return min_rest;
    } else {
      return first;
    }
  }
}
```

</td>
<td>

With P1219R2 and P1306R2

```cpp
template <typename T>
    requires std::totally_ordered<T> && std::copyable<T>
T Min(const T& first, const T&... rest) {
  T result = first;
  template for (const T& t: rest) {
    if (t < result) {
      result = t;
    }
  }
  return result;
}
```

</td>
<td>

(No extensions)

```swift
func Min<T: Comparable>(_ first: T, _ rest: T...) -> T {
  var result: T = first;
  for t in rest {
    if (t < result) {
      result = t
    }
  }
  return result
}
```

</td>
</tr>
<tr>
<td>

```carbon
fn StrCat[... each T:! ConvertibleToString](... each param: each T) -> String {
  var len: i64 = 0;
  ... len += each param.Length();
  var result: String = "";
  result.Reserve(len);
  ... result.Append(each param.ToString());
  return result;
}
```

</td>
<td>

```cpp
template <ConvertibleToString... Ts>
std::string StrCat(const Ts&... params) {
  std::string result;
  result.reserve((params.Length() + ... + 0));
  (result.append(params.ToString()), ...);
  return result;
}
```

</td>
<td>

With P1306R2

```cpp
template <ConvertibleToString... Ts>
std::string StrCat(const Ts&... params) {
  std::string result;
  result.reserve((params.Length() + ... + 0));
  template for (auto param: params) {
    result.append(param.ToString());
  }
  return result;
}
```

</td>
<td>

With SE-0393 and SE-404

```swift
func StrCat<each T: ConvertibleToString>(_ param: repeat each T) -> String {
  var len: Int64 = 0;
  for param in repeat each param {
    len += param.Length()
  }
  var result: String = ""
  result.reserveCapacity(len)
  for param in repeat each param {
    result.append(param.ToString())
  }
  return result
}
```

</td>
</tr>
</table>

## Rationale

Carbon needs variadics to effectively support
[interoperation with and migration from C++](/docs/project/goals.md#interoperability-with-and-migration-from-existing-c-code),
where variadic templates are fairly common. Variadics also make code
[easier to read, understand, and write](/docs/project/goals.md#code-that-is-easy-to-read-understand-and-write),
because some APIs (such as `printf`) can't be naturally expressed in terms of a
fixed number of parameters.

Furthermore, Carbon needs to support _generic_ variadics for the same reasons it
needs to support generic non-variadic functions: for example,
definition-checking makes APIs
[easier to read, understand, and write](/docs/project/goals.md#code-that-is-easy-to-read-understand-and-write),
and [easier to evolve](/docs/project/goals.md#software-and-language-evolution).
Furthermore, the language as a whole is easier to understand and write code in
if separate features like variadics and generics compose in natural ways, rather
than being mutually exclusive.

Variadics are also important for supporting
[performance-critical software](/docs/project/goals.md#performance-critical-software),
because variadic APIs can be more efficient than their non-variadic
counterparts. For example, `StrCat` is fundamentally more efficient than
something like a chain of `operator+` calls on `std::string`, because it does
not need to materialize a series of partial results, and it can pre-allocate a
buffer large enough for the final result.

Variadics are also needed to support the principle that
[all APIs are library APIs](/docs/project/principles/library_apis_only.md),
because the library representations of types such as tuples and callables will
need to be variadic. This proposal may appear to deviate from that principle in
some ways, but that appearance is misleading:

-   The design of pack expansion expressions treats the tuple literal syntax as
    built-in, but this isn't a problem because literal syntaxes are explicitly
    excluded from the principle.
-   The design of pack expansion patterns treats tuple types as built-in. This
    is arguably consistent with the principle, if we regard a tuple pattern as a
    kind of tuple literal (note that they have identical syntax). This proposal
    also revises the text of the principle to make that more explicit.
-   Pack types themselves are built-in types, with no library API. However, the
    principle only applies to first-class types, and pack types are decidedly
    not first-class: they cannot be function return types, they cannot even be
    named, and an expression cannot evaluate to a value with a pack type unless
    it's within a pack expansion _and_ it has compile-time expression phase (and
    even that narrow exception only exists to make the formalism more
    convenient).

## Alternatives considered

### Member packs

We could potentially support declaring each-names as class members. However,
this raises some novel design issues. In particular, pack bindings currently
rely exclusively on type deduction for information like the arity of the pack,
but for class members, there usually isn't an initializer available to drive
type deduction.

In addition, it's usually if not always possible to work around the lack of
member packs by using members with tuple or array types instead. Consequently,
this feature is deferred to future work.

### Single semantic model for pack expansions

There's a subtle discrepancy in how this proposal models expression pack
expansions: at run time, all pack expansions are modeled as procedural loops
that successively evaluate the expansion body for each element of the input
pack, and within each iteration, expressions have scalar values. However, in the
type system, expressions within a pack expansion are notionally evaluated once,
producing a pack value. In effect, this treats pack expansions like SIMD code,
with expressions operating on "vectors" of data in parallel, rather than
iteratively executing the code on a series of scalar values.

This discrepancy leads to an impedance mismatch where the two models meet. In
particular, it leads to the result that expressions within a pack expansion have
pack types, but do not evaluate to pack values. This contravenes one of the
basic expectations of a type system, that the type of an expression equals (or
is at least a supertype of) the type of its value.

It's tempting to resolve the inconsistency by applying the parallel model at run
time as well as in the type system. However, that isn't feasible, because the
parallel model has the same limitation for variadics as it does for SIMD: it
can't model branching control flow. For example, consider
`(... if (expand cond) then F(expand param) else G(expand param))`: if
`expand param` truly evaluated to a pack value, then evaluating this expression
would require N calls to _both_ `F` and `G`, rather than N calls to _either_ `F`
or `G`. Even for expressions that don't contain control flow, the same problem
applies when they occur within a statement pack expansion that does. We can't
even statically detect these problems, because a branch could be hidden inside a
function call. And this isn't just a performance problem -- if `F` or `G` have
side effects, it can also be a correctness problem.

An earlier version of this proposal tried to address this problem in a more
limited way by saying that expressions within a pack expansion don't have types
at all, but instead have "type packs". This shift in terminology nominally
avoids the problem of having expressions that don't evaluate to a value of the
expression's type, but it doesn't seem to be very clarifying in practice, and it
doesn't address the substance of the problem.

### Generalize `expand`

The syntax "`...expand` _expression_" behaves like syntactic sugar for
`... each x`, where `x` is an invented pack binding in the same scope, defined
as if by "`let (... each x: auto) =` _expression_". We could generalize that by
saying that `expand` is a prefix operator with the same precedence as `*` that
can be used anywhere in a pack expansion, where "`expand` _expression_" is
syntactic sugar for `each x` (with `x` defined as before, in the scope
containing the pack expansion). This would make `expand` more useful, and also
resolve the anomaly where `...expand` is the only syntax that begins with `...`
but is not a pack expansion. It is also a precondition for several of the
alternatives discussed below.

However, those semantics could be very surprising in practice. For example:

```carbon
...if (Condition()) {
  var x: auto = expand F(y);
}
```

In this code, `F(y)` is evaluated before the pack expansion is entered, which
means that it is evaluated unconditionally, and it cannot refer to names
declared inside the `if` block.

We can avoid the name-resolution issue by disallowing `expand` in statement pack
expansions, but the sequencing of evaluation could still be surprising,
particularly with `if` expressions.

### Omit `expand`

As noted above, `...expand` is fundamentally syntactic sugar, so we could omit
it altogether. This would somewhat simplify the design, and avoid the anomaly of
having one syntax that starts with `...` but isn't a pack expansion. However,
that would make it substantially less ergonomic to do things like expand a tuple
into an argument list, which we expect to be relatively common.

### Support expanding arrays

Statically-sized arrays are very close to being a special case of tuple types:
the only difference between an array type `[i32; 2]` (using Rust syntax) and a
tuple type `(i32, i32)` is that the array type can be indexed with a run-time
subscript. Consequently, it would be fairly natural to allow `expand` to operate
on arrays as well as tuples, and even to allow arrays of types to be treated as
tuple types (in the same way that tuples of types can be treated as tuple
types).

This functionality is omitted from the current proposal because we have no
motivating use cases, but it could be added as an extension. Note that there are
important motivating use cases under some of the alternatives considered below.

### Omit each-names

Rather than having packs be distinguished by their names, we could instead
distinguish them by their types. For example, under the current proposal, the
signature of `Zip` is:

```carbon
fn Zip[... each ElementType:! type]
      (... each vector: Vector(each ElementType))
      -> Vector((... each ElementType));
```

With this alternative, it could instead be written:

```carbon
fn Zip[ElementTypes:! [type;]]
      (... vectors: Vector(expand ElementTypes))
      -> Vector((... expand ElementTypes));
```

This employs several features not in the primary proposal:

-   In cases where the declared type of the each-name does not vary across
    iterations (like `ElementType`), we can re-express it as an array binding if
    [`expand` supports arrays](#support-expanding-arrays), and if
    [`expand` is a stand-alone operator](#generalize-expand). Note that we only
    need this in type position of a binding pattern, where we could more easily
    restrict `expand` to avoid the problems discussed earlier.
-   In cases where the declared type of the binding does vary, that fact alone
    implies that the binding refers to a pack, so we can effectively infer the
    presence of `each` from the type, rather than make the user spell it out
    explicitly.

This slight change in syntax belies a much larger shift in the underlying
semantics: since these are ordinary bindings, a given call to `Zip` must bind
each of them to a single value that represents the whole sequence of arguments
(which is why their names are now plural). In the case of `ElementTypes`, that
follows straightforwardly from its type: it represents the argument types as an
array of `type`s. The situation with `vectors` is more subtle: we have to
interpret `Vector(expand ElementTypes)` as the type of the whole sequence of
argument values, rather than as a generic description of the type of a single
argument. In other words, we have to interpret it as a pack type, and that means
`vectors` notionally binds to a run-time pack value.

Consequently, when `vectors` is used in the function body, it doesn't need an
`each` prefix: we've chosen to express variadicity in terms of types, and it
already has a pack type, so it can be directly used as an expansion site.

This approach has a few advantages:

-   We don't have to introduce the potentially-confusing concept of a binding
    that binds to multiple values simultaneously.
-   It avoids the anomaly where we have pack types in the type system, but no
    actual values of those types.
-   Removing the `each` keyword makes it more natural to spell `expand` as a
    symbolic token (earlier versions of this proposal used `[:]`), which is more
    concise and doesn't need surrounding whitespace.
-   For fully homogeneous variadics (such as `SumInts` and `Min`) it's actually
    possible to write the function body as an ordinary loop with no variadics,
    by expressing the signature in terms of a non-pack binding with an array
    type.

However, it also has some major disadvantages:

-   The implicit expansion of pack-type bindings hurts readability. For example,
    it's easy to overlook the fact that the loop condition
    `while (...and expand iters != vectors.End())` in `Zip` has two expansion
    sites, not just one. This problem is especially acute in cases where a
    non-local name has a pack type.
-   We have to forbid template-dependent names from having pack types (see
    [leads issue #1162](https://github.com/carbon-language/carbon-lang/issues/1162)),
    because the possibility that an expression might be an expansion site in
    some instantiations but not others would cause serious readability and
    implementability issues.
-   A given _use_ of such a binding really represents a single value at a time,
    in the same way that the iteration variable of a for-each loop does, so
    giving the binding a plural name and a pack type creates confusion in that
    context rather than alleviating it.

It's also worth noting that we may eventually want to introduce operations that
treat the sequence of bound values as a unit, such as to determine the length of
the sequence (like `sizeof...` in C++), or even to index into it. This approach
might seem more amenable to that, because it conceptually treats the sequence of
values as a value in itself, which could have its own operations. However, this
approach leaves no "room" in the syntax to spell those operations, because any
mention of a pack-type binding implicitly refers to one of its elements.

Conversely, the status quo proposal seems to leave a clear syntactic opening for
those operations: you can refer to the sequence as a whole by omitting `each`,
so `each vector.Size()` refers to the size of the current iteration's `vector`,
whereas `vector.Size()` could refer to the size of the sequence of bound values.
However, this could easily turn out to be a "wrong default": omitting `each`
seems easy to do by accident, and easy to misread during code review.

There are other solutions to this problem that work equally well with the status
quo or this alternative. In particular, it's already possible to express these
operations outside of a pack expansion by converting to a tuple, as in
`(... each vector).Size()` (status quo) or `(... vectors).Size()` (this
alternative). That may be sufficient to address those use cases, especially if
we relax the restrictions on nesting pack expansions. Failing that,
variadic-only spellings for these operations (like `sizeof...` in C++) would
also work with both approaches. So this issue does not seem like an important
differentiator between the two approaches.

#### Disallow pack-type bindings

As a variant of the above approach, it's possible to omit both each-names and
pack-type bindings, and instead rely on variadic tuple-type bindings. For
example, the signature of `Zip` could instead be:

```carbon
fn Zip[ElementTypes:! [type;]]
      (... expand vectors: (... Vector(expand ElementTypes)))
      -> Vector((... expand ElementTypes));
```

This signature doesn't change the callsite semantics, but within the function
body `vectors` will be a tuple rather than a pack. This avoids or mitigates all
of the major disadvantages of pack-type bindings, but it comes at a substantial
cost: the function signature is substantially more complex and opaque. That
seems likely to be a bad tradeoff -- the disadvantages of pack-type bindings
mostly concern the function body, but readability of variadic function
signatures seems much more important than readability of variadic function
bodies, because the signatures will be read far more often, and by programmers
who have less familiarity with variadics.

This approach requires us to relax the ban on nested pack expansions. This does
create some risk of confusion about which pack expansion a given `expand`
belongs to, but probably much less than if we allowed unrestricted nesting.

The leads chose not to pursue this approach in
[leads issue #1162](https://github.com/carbon-language/carbon-lang/issues/1162).

### Fold expressions

We could generalize the `...and` and `...or` syntax to support a wider variety
of binary operators, and to permit specifying an initial value for the chain of
binary operators, as with C++'s
[fold expressions](https://en.cppreference.com/w/cpp/language/fold). This would
be more consistent with C++, and would give users more control over
associativity and over the behavior of the arity-zero case.

However, fold expressions are arguably too general in some respects: folding
over a non-commutative operator like `-` is more likely to be confusing than to
be useful. Similarly, there are few if any plausible use cases for customizing
the arity-zero behavior of `and` or `or`. Conversely, fold expressions are
arguably not general enough in other respects, because they only support folding
over a fixed set of operators, not over functions or compound expressions.

Furthermore, in order to support folds over operator tokens that can be either
binary or prefix-unary (such as `*`), we would need to choose a different syntax
for tuple element lists. Otherwise, `...*each foo` would be ambiguous between
`*foo[:0:], *foo[:1:],` etc. and `foo[:0:] * foo[:1:] *` etc.

Note that even if Carbon supported more general C++-like fold expressions, we
would still probably have to give `and` and `or` special-case treatment, because
they are short-circuiting.

As a point of comparison, C++ fold expressions give special-case treatment to
the same two operators, along with `,`: they are the only ones where the initial
value can be omitted (such as `... && args` rather than `true && ... && args`)
even if the pack may be empty. Furthermore, folding over `&&` appears to have
been the original motivation for adding fold expressions to C++; it's not clear
if there are important motivating use cases for the other operators.

Given that we are only supporting a minimal set of operators, allowing `...` to
occur in ordinary binary syntax has few advantages and several drawbacks:

-   It might conflict with a future general fold facility.
-   It would invite users to try other operators, and would probably give less
    clear errors if they do.
-   It would substantially complicate parsing and the AST.
-   It would force users to make a meaningless choice between `x or ...` and
    `... or x`, and likewise for `and`.

See also the discussion [below](#fold-like-syntax) of using `...,` and `...;` in
place of the tuple and statement forms of `...`. This is inspired by fold
expressions, but distinct from them, because `,` and `;` are not truly binary
operators, and it's targeting a different problem.

### Allow multiple pack expansions in a tuple pattern

As currently proposed, we allow multiple `...` expressions within a tuple
literal expression, but only allow one `...` pattern within a tuple pattern. It
is superficially tempting to relax this restriction, but fundamentally
infeasible.

Allowing multiple `...` patterns would create a potential for ambiguity about
where their scrutinees begin and end. For example, given a signature like
`fn F(... each xs: i32, ... each ys: i32)`, there is no way to tell where `xs`
ends and `ys` begins in the argument list; every choice is equally valid. That
ambiguity can be avoided if the types are different, but that would make type
_non_-equality a load-bearing part of the pattern. That's a very unusual thing
to need to reason about in the type system, so it's liable to be a source of
surprise and confusion for programmers, and in particular it looks difficult if
not impossible to usefully express with generic types, which would greatly limit
the usefulness of such a feature.

Function authors can straightforwardly work around this restriction by adding
delimiters. For example, the current design disallows
`fn F(... each xs: i32, ... each ys: i32)`, but it allows
`fn F((... each xs: i32), (... each ys: i32))`, which is not only easier to
support, but makes the callsite safer and more readable, since the boundary
between the `xs` and `ys` arguments is explicitly marked. By contrast, if we
disallowed multiple `...` expressions in a function argument list, function
callers who ran into that restriction would often find it difficult or
impossible to work around. Note, however, that this workaround presupposes that
function signatures can have bindings below top-level, which is
[currently undecided](https://github.com/carbon-language/carbon-lang/issues/1229).

To take a more abstract view of this situation: when we reuse expression syntax
as pattern syntax, we are effectively inverting expression evaluation, by asking
the language to find the operands that would cause an expression to evaluate to
a given value. That's only possible if the operations involved are invertible,
meaning that they do not lose information. When a tuple literal contains
multiple `...` expressions, evaluating it effectively discards structural
information about for example where `xs` ends and `ys` begins. The operation of
forming a tuple from multiple packs is not invertible, and consequently we
cannot use it as a pattern operation. Our rule effectively says that if the
function needs that structural information, it must ask the caller to provide
it, rather than asking the compiler to infer it.

### Allow nested pack expansions

Earlier versions of this design allowed pack expansions to contain other pack
expansions. This is in some ways a natural generalization, but it added
nontrivial complexity to the design. In particular, when an each-name is
lexically within two or more pack expansions, we need a rule for determining
which pack expansion iterates over it, in a way that is unsurprising and
supports the intended use cases. However, we have few if any motivating use
cases for it, which made it difficult to evaluate that aspect of the design.
Consequently, this proposal does not support nested pack expansions, although it
tries to avoid ruling them out as a future extension.

### Use postfix instead of prefix `...`

`...` is a postfix operator in C++, which aligns with the natural-language use
of "…", so it would be more consistent with both if `...`, `...and`, and `...or`
were postfix operators spelled `...`, `and...`, and `or...`, and likewise if
statement pack expansions were marked by a `...` at the end rather than the
beginning.

However, prefix syntaxes are usually easier to parse (particularly for humans),
because they ensure that by the time you start parsing an utterance, you already
know the context in which it is used. This is clearest in the case of
statements: the reader might have to read an arbitrary amount of code in the
block before realizing that the code they've been reading will be executed
variadically, so that seems out of the question. The cases of `and`, `or`, and
`,` are less clear-cut, but we have chosen to make them all prefix operators for
consistency with statements.

### Avoid context-sensitity in pack expansions

This proposal "overloads" the `...` token with multiple different meanings
(including different precedences), and the meaning depends in part on the
surrounding context, despite Carbon's principle of
[avoiding context-sensitivity](/docs/project/principles/low_context_sensitivity.md).
We could instead represent the different meanings using separate syntaxes.

There are several variants of this approach, but they all have substantial
drawbacks (see the following subsections). Furthermore, the problems associated
with context-sensitivity appear to be fairly mild in this case: the difference
between a tuple literal context and a statement context is usually quite local,
and is usually so fundamental that confusion seems unlikely.

#### Fold-like syntax

We could use a modifier after `...` to select the expansion's meaning (as we
already do with `and` and `or`). In particular, we could write `...,` to
iteratively form elements of a tuple, and write `...;` to iteratively execute a
statement. This avoids context-sensitivity (apart from `...,` having a dual role
in expressions and patterns, like many other syntaxes), and has an underlying
unity: `...,`, `...;` `...and`, and `...or` represent "folds" over the `,`, `;`,
`and`, and `or` tokens, respectively. As a side benefit, this would preserve the
property that a tuple literal always contains a `,` character (unlike the
current proposal).

However, this approach has major readability problems. Using `...;` as a prefix
operator is completely at odds with the fact that `;` marks the end of a
statement, not the beginning. Furthermore, it would probably be surprising to
use `...;` in contexts where `;` is not needed, because the end of the statement
is marked with `}`.

The problems with `...,` are less severe, but still substantial. In this syntax
`,` does not behave like a separator, but our eyes are trained to read it as
one, and that habit is difficult to unlearn. For example, most readers have
found that they can't help automatically reading `(..., each x)` as having two
sub-expressions, `...` and `each x`. This effect is particularly disruptive when
skimming a larger body of code, such as:

```carbon
fn TupleConcat[..., each T1: type, ..., each T2: type](
    t1: (..., each T1), t2: (..., each T2)) -> (..., each T1, ..., each T2) {
  return (..., expand t1, ..., expand t2);
}
```

#### Variadic blocks

We could replace the statement form of `...` with a variadic block syntax such
as `...{ }`. However, this doesn't give us an alternative for the tuple form of
`...`, and yet heightens the problems with it: `...{` could read as as applying
the `...` operator to a struct literal.

Furthermore, it gives us no way to variadically declare a variable that's
visible outside the expansion (such as `each iter` in the `Zip` example). This
can be worked around by declaring those variables as tuples, but this adds
unnecessary complexity to the code.

#### Keyword syntax

We could drop `...` altogether, and use a separate keyword for each kind of pack
expansion. For example, we could use `repeat` for variadic lists of tuple
elements, `do_repeat` for variadic statements, and `all_of` and `any_of` in
place of `...and` and `...or`. This leads to code like:

```carbon
// Takes an arbitrary number of vectors with arbitrary element types, and
// returns a vector of tuples where the i'th element of the vector is
// a tuple of the i'th elements of the input vectors.
fn Zip[repeat each ElementType:! type]
      (repeat each vector: Vector(each ElementType))
      -> Vector((repeat each ElementType)) {
  do_repeat var each iter: auto = each vector.Begin();
  var result: Vector((repeat each ElementType));
  while (all_of each iter != each vector.End()) {
    result.push_back((repeat each iter));
    repeat each iter++;
  }
  return result;
}
```

This approach is heavily influenced by
[Swift variadics](https://github.com/swiftlang/swift-evolution/blob/main/proposals/0393-parameter-packs.md),
but not quite the same. It has some major advantages: the keywords are more
consistent with `each` (and `expand` to some extent), substantially less
visually noisy than `...`, and they may also be more self-explanatory. However,
it does have some substantial drawbacks.

Most notably, there is no longer any syntactic commonality between the different
tokens that mark the root of an expansion. That makes it harder to visually
identify expansions, and could also make variadics harder to learn, because the
spelling does not act as a mnemonic cue. And while it's already not ideal that
under the primary proposal a tuple literal is identified by the presence of
either `,` or `...`, it seems even worse if one of those two tokens is instead a
keyword.

Relatedly, the keywords have less clear precedence relationships, because
`all_of` and `any_of` can't as easily "borrow" their precedence from their
non-variadic counterparts. For example, consider this line from `Zip`:

```carbon
while (...and each iter != each vector.End()) {
```

Under this alternative, that becomes:

```carbon
while (all_of each iter != each vector.End()) {
```

I find the precedence relationships in the initial `all_of expand iters !=` more
opaque than in `...and expand iters !=`, to the extent that we might need to
require additional parentheses:

```carbon
  while (all_of (expand iters != each vectors.End())) {
```

That avoids outright ambiguity, but obliging readers to maintain a mental stack
of parentheses in order to parse the expression creates its own readability
problems.

It's appealing that the `repeat` keyword combines with `each` to produce code
that's almost readable as English, but it creates a temptation to read `expand`
the same way, which will usually be misleading. For example, `repeat expand foo`
sounds like it is repeatedly expanding `foo`, but in fact it expands it only
once. It's possible that a different spelling of `expand` could avoid that
problem, but I haven't been able to find one that does so while also avoiding
the potential for confusion with `each`. This is somewhat mitigated by the fact
that `expand` expressions are likely to be rare.

It's somewhat awkward, and potentially even confusing, to use an imperative word
like `repeat` in a pattern context. By design, the pattern language is
descriptive rather than imperative: it describes the values that match rather
than giving instructions for how to match them. As a result, in a pattern like
`(repeat each param: i64)`, it's not clear what action is being repeated.

Finally, it bears mentioning that the keywords occupy lexical space that could
otherwise be used for identifiers. Notably, `all_of`, `any_of`, and `repeat` are
all names of functions in the C++ standard library. This is not a fundamental
problem, because we expect Carbon to have some way of "quoting" a keyword for
use as an identifier (such as Rust's
[raw identifiers](https://doc.rust-lang.org/rust-by-example/compatibility/raw_identifiers.html)),
but it is likely to be a source of friction.

### Require parentheses around `each`

We could give `each` a lower precedence, so that expressions such as
`each vector.End()` would need to be written as `(each vector).End()`. This
could make the code clearer for readers, especially if they are new to Carbon
variadics. However, this would make the code visually busier, and might give the
misleading impression that `each` can be applied to anything other than an
identifier. I propose that we wait and see whether the unparenthesized syntax
has readability problems in practice, before attempting to solve those problems.

We have discussed a more general solution to this kind of problem, where a
prefix operator could be embedded in a `->` token, in order to apply the prefix
operator to the left-hand operand without needing parentheses. However, this
approach is much more appealing when the prefix operator is a symbolic token:
`x-?>y` may be a plausible alternative to `(?x).y`, but `x-each>y` seems much
harder to visually parse. Furthermore, this approach is hard to reconcile with
treating `each` as fundamentally part of the name, rather than an operator
applied to the name.

### Fused expansion tokens

Instead of treating `...and` and `...or` as two tokens with whitespace
discouraged between them, we could treat them as single tokens. This might more
accurately reflect the fact that they are semantically different operations than
`...`, and reduce the potential for readability problems in code that doesn't
follow our recommended whitespace conventions. However, that could lead to a
worse user experience if users accidentally insert a space after the `...`.

### No parameter merging

Under the current proposal, the compiler attempts to merge function parameters
in order to support use cases like this one, where merging the parameters of
`Min` enables us to pair each argument with a single logical parameter that will
match it:

```carbon
fn Min[T:! type](first: T, ... each next: T) -> T;

fn F(... each arg: i32) {
  Min(... each arg, 0 as i32);
}
```

However, this approach makes typechecking hard to understand (and predict),
because the complex conditions governing merging mean that subtle differences in
the code can cause dramatic differences in the semantics. For example:

```carbon
fn F[A:! I, ... each B:! I](a: A, ... each b: each B);
fn G[A:! I, ... each B:! I](a: A, ... each b: each B) -> A;
```

These two function signatures are identical other than their return types, but
they actually have different requirements on their arguments: `G` requires the
first argument to be singular, whereas `F` only requires _some_ argument to be
singular. It seems likely to be hard to teach programmers that the function's
return type sometimes affects whether a given argument list is valid. Relatedly,
it's hard to see how a diagnostic could concisely explain why a given call to
`G` is invalid, in a way that doesn't seem to also apply to `F`.

We could solve that problem by omitting parameter merging, and interpreting all
of the above signatures as requiring that the first argument must be singular,
because the first parameter is singular. Thus, there would be a clear and
predictable connection between the parameter list and the requirements on the
argument list.

In order to support use cases like `Min` where the author doesn't intend to
impose such a requirement, we would need to provide some syntax for declaring
`Min` so that it has a single parameter, but can't be called with no arguments.
More generally, this syntax would probably need to support setting an arbitrary
minimum number of arguments, not just 1. For example, an earlier version of this
proposal used `each(>=N)` to require that a parameter match at least N
arguments, so `Min` could be written like this:

```carbon
fn Min[T:! type](... each(>=1) param: T) -> T;
```

However, this alternative has several drawbacks:

-   We haven't been able to find a satisfactory arity-constraint syntax. In
    addition to its aesthetic problems, `each(>=1) param` disrupts the mental
    model where `each` is part of the name, and it's conceptually awkward
    because the constraint actually applies to the pack expansion as a whole,
    not to the each-name in particular. However, it's even harder to find an
    arity-constraint syntax that could attach to `...` without creating
    ambiguity. Furthermore, any arity-constraint syntax would be an additional
    syntax that users need to learn, and an additional choice they need to make
    when writing a function signature.
-   Ideally, generic code should typecheck if every possible monomorphization of
    it would typecheck. This alternative does not live up to that principle --
    see, for example, the above example of `Min`. The current design does not
    fully achieve that aspiration either, but it's far more difficult to find
    plausible examples where it fails.
-   The first/rest style will probably be more natural to programmers coming
    from C++, and if they define APIs in that style, there isn't any plausible
    way for them to find out that they're imposing an unwanted constraint on
    callers, until someone actually tries to make a call with the wrong shape.

### Exhaustive function call typechecking

The current proposal uses merging and splitting to try to align the argument and
parameter lists so that each argument has exactly one parameter than can match
it. We also plan to extend this design to also try the opposite approach,
aligning them so that each parameter has exactly one argument that it can match.
However, it isn't always possible to align arguments and parameters in that way.
For example:

```carbon
fn F[... each T:! type](x: i32, ... each y: each T);

fn G(... each z: i32) {
  F(... each z, 0 as i16);
}
```

Every possible monomorphization of this code would typecheck, but we can't merge
the parameters because they have different types, and we can't merge the
arguments for the same reason. We also can't split the variadic parameter or the
variadic argument, because either of them could be empty.

The fundamental problem is that, although every possible monomorphization
typechecks, some monomorphizations are structurally different from others. For
example, if `each z` is empty, the monomorphized code converts `0 as i16` to
`i32`, but otherwise `0 as i16` is passed into `F` unmodified.

We could support such use cases by determining which parameters can potentially
match which arguments, and then typechecking each pair. For example, we could
typecheck the above code by cases:

-   If `each z` is empty, `x: i32` matches `0 as i16` (which typechecks because
    `i16` is convertible to `i32`), and `each y: each T` matches nothing.
-   If `each z` is not empty, `x: i32` matches its first element (which
    typechecks because `i32` is convertible to `i32`), and `each y: each T`
    matches the remaining elements of `each z`, followed by `0 as i16` (which
    typechecks by binding `each T` to `⟬«i32; ‖each z‖-1», i16⟭`).

More generally, this approach works by identifying all of the structurally
different ways that arguments could match parameters, typechecking them all in
parallel, and then combining the results with logical "and".

However, the number of such cases (and hence the cost of typechecking) grows
quadratically, because the number of cases grows with the number of parameters,
and the case analysis has to be repeated for each variadic argument.
[Fast development cycles](/docs/project/goals.md#fast-and-scalable-development)
are a priority for Carbon, so if at all possible we want to avoid situations
where compilation costs grow faster than linearly with the amount of code.

Furthermore, typechecking a function call doesn't merely need to output a
boolean decision about whether the code typechecks. In order to typecheck the
code that uses the call, and support subsequent phases of compilation, it needs
to also output the type of the call expression, and that can depend on the
values of deduced parameters of the function.

These more complex outputs make it much harder to combine the results of
typechecking the separate cases. To do this in a general way, we would need to
incorporate some form of case branching directly into the type system. For
example:

```carbon
fn P[T:! I, ... each U:! J](t: T, ... each u: each U) -> T;

fn Q[X:! I&J, ... each Y:! I&J](x: X, ... each y: each Y) -> auto {
  return P(... each y, x);
}

fn R[A:! I&J ... each B:! I&J](a: A, ... each b: each B) {
  Q(... each b, a);
}
```

The typechecker would need to represent the type of `P(... each x, y)` as
something like `(... each Y, X).0`. That subscript `.0` acts as a disguised form
of case branching, because now any subsequent code that depends on
`P(... each y, x)` needs to be typechecked separately for the cases where
`... each Y` is and is not empty. In this case, that even leaks back into the
caller `R` through `Q`'s return type, which compounds the complexity: the type
of `Q(... each b, a)` would need to be something like
`((... each B, A).(1..‖each B‖), (... each B, A).0).0` (where `.(M..N)` is a
hypothetical tuple slice notation).

All of this may be feasible, but the cost in type system complexity and
performance would be daunting, and the benefits are at best unclear, because we
have not yet found plausible motivating use cases that benefit from this kind of
typechecking.
