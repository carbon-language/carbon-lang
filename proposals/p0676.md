# `:!` generic syntax

<!--
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

[Pull request](https://github.com/carbon-language/carbon-lang/pull/676)

<!-- toc -->

## Table of contents

-   [Problem](#problem)
-   [Background](#background)
-   [Proposal](#proposal)
-   [Rationale based on Carbon's goals](#rationale-based-on-carbons-goals)
-   [Alternatives considered](#alternatives-considered)
    -   [Default based on context](#default-based-on-context)
        -   [Square brackets](#square-brackets)
    -   [Other spellings that were considered](#other-spellings-that-were-considered)
    -   [`Template` as a type-of-type](#template-as-a-type-of-type)
-   [Alternatives not considered](#alternatives-not-considered)

<!-- tocstop -->

## Problem

Carbon design docs provisionally used `:$` to mark generic parameters. Since
then, [issue #565](https://github.com/carbon-language/carbon-lang/issues/565)
decided to use `:!` more permanently. This proposal is to implement that
decision.

## Background

Most popular languages put generic parameters inside angle brackets (`<`...`>`),
as can be seen on rosettacode.org:
[1](http://rosettacode.org/wiki/Generic_swap),
[2](http://rosettacode.org/wiki/Constrained_genericity).

## Proposal

Generic parameters will be marked using `:!` instead of `:` in the parameter
list. They are listed with the regular parameters if they are to be specified
explicitly by callers.

```
fn Zero(T:! ConvertFrom(Int)) -> T;

var zf: Float32 = Zero(Float32);
```

If they are instead deduced from the (types of) the regular parameters, they are
listed in square brackets (`[`...`]`) before the parameter list in round parens
(`(`...`)`).

```
fn Swap[T:! Movable](a: T*, b: T*);

var i: Int = 1;
var j: Int = 2;
Swap(&i, &j);
```

Template parameters use both a `template` keyword before the parameter and `:!`
in place of `:`.

```
fn FieldNames(template T:! Type) -> String;

Assert(FieldNames(struct {.x: Int, .y: Int}) == "x, y");
```

For both generic and template parameters, the `!` means "compile time." There is
some precedent for this meaning; Rust uses `!` to mark macro calls, that is
calls that happen at compile time.

## Rationale based on Carbon's goals

We are attempting to choose a syntax that advances Carbon's goal of having
[code that is easy to read, understand, and write](/docs/project/goals.md#code-that-is-easy-to-read-understand-and-write).
This option was chosen since it has the advantage of being very simple and not
relying on any context, in accordance with the
[#646: low-context-sensitivity principle](https://github.com/carbon-language/carbon-lang/pull/646).

For ease of parsing, we've been trying to avoid using angle brackets (`<`...`>`)
outside of declarations. This is primarily a concern for parameterized types
that can appear in expressions alongside comparison operators (`<` and `>`).

The choice to mark template parameters with a keyword was to make it very
visible when that powerful and dangerous feature was being used. We also liked
the similarities to how a `template` keyword also introduces a C++ template
declaration.

The choice to use the specific symbols `:!` over `:$` or other possibilities was
just a matter of taste.

## Alternatives considered

There were a few other options considered to designate generics in
[issue #565](https://github.com/carbon-language/carbon-lang/issues/565).

Note that we at first considered the possibility that type parameters might
accidentally be declared as dynamic if that was the default. We eventually
decided that we could forbid dynamic type parameters for now, and revisit this
problem if and when we decided to add a dynamic type parameter feature.

### Default based on context

In a given syntactic context, one option is going to be more common than others:

-   Regular explicit parameters would most commonly be dynamic.
-   Deduced parameters would most commonly be generic.
-   Parameters to interfaces and types would most commonly be generic.

We considered making `x: T` be generic or dynamic based on this context. In
cases where this default was not what was intended, there would be a keyword
(`dynamic`, `generic`, or `template`) to explicitly pick.

There were a few variations about whether to treat parameters used in types
differently. This had the downside of being harder to discern at a glance.

The main benefits of this approach were:

-   It handled dynamic type parameters being allowed but uncommon more
    gracefully.
-   Users could use `:` and it would generally do the right thing.
-   Keywords are generally easier to find in search engines, and more
    self-explanatory.
-   Template parameters in particular were highlighted, a property shared with
    the approach recommended by this proposal.

The main objections to this approach was that it was context-sensitive and there
was a lack of syntactic consistency in the context. That is, there were two
kinds of context, generic and dynamic, and two kinds of brackets, square and
parens, but sometimes the parens would be generic and sometimes not.

#### Square brackets

Using `[`...`]` for generics creates the opposite problem of the brackets being
inconsistent with the deduced or explicit distinction.

```
// `T` is an explicit generic parameter to `Vector`
class Vector[T: Type] { ... }
// `T` and `DestT` are generic parameters, with `T` deduced and
// `DestT` explicit.
fn CastAVector[T: Type](v: Vector[T], generic DestT: Type) -> Vector[DestT];
```

### Other spellings that were considered

There were a number of other options considered. None of them were compelling,
though this mostly came down to taste.

```
class Vector(<T: Type>) { ... }
fn CastAVector<T: Type>(v: Vector(T), <DestT: CastFrom(T)>) -> Vector(DestT);
var from: Vector(i32) = ...;
var to: Vector(i64) = CastAVector(from, i64);
```

-   not trivial to parse, but doable
-   too much punctuation
-   nice that `<`...`>` is associated with generics, but with enough differences
    to be concerning

```
fn CastAVector[T: Type](v: Vector(T), [DestT: Type]) -> Vector(DestT);
class Vector([T: Type]) { ... }
var from: Vector(i32) = ...;
var to: Vector(i64) = CastAVector(from, i64);
```

-   There was no reason to prefer this over the previous option, since `<`...`>`
    is more associated with generics than `[`...`]`.

Other different spellings of the `:!` position that came up during brainstorming
but were not found to be compelling included:

-   `<id>: Type`
-   `id:# Type`
-   `id:<> Type`
-   `id: <Type>`
-   `generic id: Type`

### `Template` as a type-of-type

We talked about the alternative of using a keyword like `Auto` or `Template` as
a type-of-type to indicate that a parameter was a template.

```
fn FieldNames(T: Template) -> String;
```

This would be able to be combined with other type-of-types using the `&`
operator to constrain the allowed types, as in `Container & Template`. The idea
is that `Auto` or `Template` would act like an interface that contained any
calls used by the function that were not in any other interface constraint for
that parameter.

It had two downsides:

-   This approach only worked for type parameters. We would need something else
    for non-type template parameters.
-   It didn't seem like you would want to be able to hide an `& Auto` or
    `& Template` clause by declaring it in a constant:

    ```
    let CT = Container & Template;
    fn SurpriseIHaveATemplateParam[T: CT](c: T);
    ```

That suggests that the `Auto` or `Template` keyword would not act like other
type-of-type expressions and would need special treatment.

## Alternatives not considered

We never really broke out of the idea that `[`...`]` were for deduced
parameters. As a result we didn't really consider options where type expressions
used square brackets for generic parameters, as in `Vector[Int]`, even though
that addresses the parsing problems of angle brackets. For example, if we were
to revisit this decision, we might use three kinds of brackets for the three
different cases:

-   `<`...`>` for deduced and generic parameters
-   `[`...`]` for explicit and generic parameters
-   `(`...`)` for explicit and dynamic parameters

Code would most commonly use square brackets both when declaring and using a
parameterized type or an interface. Parens would be required for functions, with
generic parameters typically specified using angle brackets.

```
class Vector[T: Type] { ... }
fn CastAVector<T: Type>[DestT: CastFrom[T]](v: Vector[T]) -> Vector[DestT];
var from: Vector[i32] = ...;
var to: Vector[i64] = CastAVector[i64](from);
```

One concern is that use of square brackets will be in the same contexts as
`[`...`]` will be used for indexing. Another concern is that there is
[some motivation](http://open-std.org/JTC1/SC22/WG21/docs/papers/2019/p1045r1.html)
for putting generic and template parameters together with regular parameters in
the `(`...`)` parameter list

[Go ultimately did decided to use square brackets for generics](https://go.googlesource.com/proposal/+/refs/heads/master/design/43651-type-parameters.md).
This is even though Go also uses square brackets for slices and indexing.
