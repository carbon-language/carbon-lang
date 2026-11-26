# First-class extended types and `fwd`

<!--
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

[Pull request](https://github.com/carbon-language/carbon-lang/pull/8183)

<!-- toc -->

## Table of contents

-   [Abstract](#abstract)
-   [Problem](#problem)
-   [Background](#background)
-   [Summary](#summary)
-   [Proposal](#proposal)
    -   [Extended types and values](#extended-types-and-values)
        -   [Primitive extended type literals](#primitive-extended-type-literals)
        -   [`exttypeof`](#exttypeof)
        -   [Compound extended type values](#compound-extended-type-values)
        -   [`type` property of extended type values](#type-property-of-extended-type-values)
    -   [Extended type facet types](#extended-type-facet-types)
    -   [Specifying the extended type of a binding](#specifying-the-extended-type-of-a-binding)
    -   [Specifying the extended type of a return](#specifying-the-extended-type-of-a-return)
    -   [Any extended type with this type](#any-extended-type-with-this-type)
    -   [`auto` extended type](#auto-extended-type)
        -   [`auto` extended type bindings](#auto-extended-type-bindings)
        -   [Positional parameters](#positional-parameters)
        -   [`auto` extended type returns](#auto-extended-type-returns)
-   [Details](#details)
    -   [Deducing an extended type value](#deducing-an-extended-type-value)
    -   [Compile-time parameter equal to a specified value](#compile-time-parameter-equal-to-a-specified-value)
    -   [Parameter requires exact type match](#parameter-requires-exact-type-match)
    -   [Input-to-output forwarding](#input-to-output-forwarding)
-   [Future work](#future-work)
    -   [Compile-time calls](#compile-time-calls)
    -   [`ref?`](#ref)
-   [Rationale](#rationale)
-   [Alternatives considered](#alternatives-considered)
    -   [Decoupling type from the rest of the extended type](#decoupling-type-from-the-rest-of-the-extended-type)
    -   [Other ways to write "any extended type with a specified type"](#other-ways-to-write-any-extended-type-with-a-specified-type)
    -   [Don't have types that represent all binding information](#dont-have-types-that-represent-all-binding-information)
    -   [Represent all extended type information in ordinary types](#represent-all-extended-type-information-in-ordinary-types)
    -   [Different kinds of extended types for parameters and returns](#different-kinds-of-extended-types-for-parameters-and-returns)
    -   [Implicit conversion from `type` to `Core.ExtType`](#implicit-conversion-from-type-to-coreexttype)
    -   [Different extended type literal syntax](#different-extended-type-literal-syntax)
    -   [`ref` on an argument controls its category](#ref-on-an-argument-controls-its-category)
    -   [Modifiers and pattern operators on `fwd` bindings](#modifiers-and-pattern-operators-on-fwd-bindings)
    -   [Separate template and symbolic phases in `Core.ExtType`](#separate-template-and-symbolic-phases-in-coreexttype)
    -   [`fwd` on call arguments](#fwd-on-call-arguments)

<!-- tocstop -->

## Abstract

We propose:

-   A special type `Core.ExtType` whose values describe the extended type
    (type, category, phase, and compile-time value) of parameters, bindings, or
    returns.
-   An `exttype(`...`)` literal syntax, `exttypeof(`...`)` operator, compound
    tuple/struct `as Core.ExtType` conversions, and `.type` property for
    working with `Core.ExtType` values.
-   Extended type facet types (`Core.NoValExtType`, `Core.NoRefExtType`,
    `Core.NoVarExtType`, `Core.NoCompileTimeExtType`, `Core.NoRunTimeExtType`),
    composable using `&` and `where .type`.
-   A `fwd` binding modifier, `fwd` call-argument marker for symbolic extended
    types, and `fwd` return tag parameterized by an extended type value.
-   `auto` extended type deduction for `let fwd` bindings, `-> fwd auto`
    returns, and positional parameters.
-   Deduction rules for `Core.ExtType` parameters, enabling exact-value
    compile-time parameters, exact-type parameters, and input-to-output
    forwarding.

## Problem

Carbon's type system models types as first-class compile-time values of type
`type`, allowing generics to parameterize bindings and returns by type. However,
expressions, bindings, and returns also carry non-type extended type information
(expression category, phase, and compile-time value). Without a way to represent
and parameterize over extended types:

-   Functions cannot forward arguments or returns while preserving expression
    category and phase (such as writing a category-preserving identity function
    or forwarding parameters to another function).
-   We cannot express referentially transparent `let` bindings that preserve
    phase and category in addition to type, as discussed in leads issues
    [#996: Generic let with auto?](https://github.com/carbon-language/carbon-lang/issues/996)
    and
    [#1371: Is let referentially transparent?](https://github.com/carbon-language/carbon-lang/issues/1371)
    (which also blocks defining a more powerful `alias` construct in
    [leads issue #5028](https://github.com/carbon-language/carbon-lang/issues/5028)).
-   Positional parameters (`$0`, `$1`, ...) only deduce types and cannot accept
    `ref` arguments for mutation.
-   There is no way to declare a compile-time parameter equal to a specified
    value (working around parameter ordering constraints as discussed in
    [leads issue #578](https://github.com/carbon-language/carbon-lang/issues/578))
    or to require an exact type match without implicit conversions.
-   Generics and operator interfaces (such as `Core.Call`, indexing,
    conversions, and arithmetic/comparison operators) lack the foundation needed
    to parameterize over `val`, `ref`, `var`, and compile-time parameters and
    returns.

## Background

This proposal was split out of
[proposal #5389](https://github.com/carbon-language/carbon-lang/pull/5389);
earlier revision and review history may be found in that pull request.

This proposal has no dependencies on pending proposals. It provides the
foundation for several pending proposals also split out of
[proposal #5389](https://github.com/carbon-language/carbon-lang/pull/5389):

-   [proposal #8184: Redefining `alias`, `using`, and namespace aliases](https://github.com/carbon-language/carbon-lang/pull/8184),
-   [proposal #8185: Extended type interfaces and named constraint extensions](https://github.com/carbon-language/carbon-lang/pull/8185),
-   [proposal #8186: Interface variance (`implicit_into` and `implicit_from`)](https://github.com/carbon-language/carbon-lang/pull/8186),
-   [proposal #8187: Extended-type call interface and `call` syntax](https://github.com/carbon-language/carbon-lang/pull/8187),
    and
-   [proposal #8188: Extended-type operator, conversion, and member binding interfaces](https://github.com/carbon-language/carbon-lang/pull/8188).

This proposal builds upon these prior issues and proposals:

-   [Leads issue #578: Value patterns as function parameters](https://github.com/carbon-language/carbon-lang/issues/578)
    decided that we would not have refutable patterns in function signatures,
    and discusses workarounds where this is helpful to work around ordering
    constraints on compile-time parameters.
-   [Proposal #3848: Lambdas](p003848-lambdas.md#positional-parameters)
    introduces positional parameters and `=>` shorthand returns.
-   [Proposal #5434: `ref` parameters, arguments, returns and `val` returns](https://github.com/carbon-language/carbon-lang/pull/5434)
    expands the set of categories for parameters and returns, increasing the
    need for a way to represent them.
-   [Proposal #5545: Expression form basics](https://github.com/carbon-language/carbon-lang/pull/5545)
    introduces extended types (originally called "forms") as capturing aspects
    of bindings and returns beyond their type.
-   [Proposal #7254: Replace `:!` and `:?` with keywords and contextual defaults](p007254-replace-and-with-keywords-and-contextual-defaults.md)
    adopts the "extended type" terminology, `Core.ExtType`, `exttype(...)`, and
    the `fwd` keyword direction from
    [leads issue #6932](https://github.com/carbon-language/carbon-lang/issues/6932).

Some of the thinking that led to this proposal:

-   [Suggestion on leads issue #5261 to support parameterizing bindings with categories](https://github.com/carbon-language/carbon-lang/issues/5261#issuecomment-2790947831).
-   [Idea in open discussion on 2025-05-07](https://docs.google.com/document/d/1Yt-i5AmF76LSvD4TrWRIAE_92kii6j5yFiW-S7ahzlg/edit?tab=t.0#heading=h.4zbo49wg5rmk).

## Summary

In short, this proposal introduces:

-   a type `Core.ExtType` of extended type values, which describe the type and
    non-type aspects of bindings and returns. Values of this type can be created
    as generic parameters of type `Core.ExtType`, or using:
    -   Primitive extended type literal syntax `exttype(val T)`,
        `exttype(ref T)`, and `exttype(var T)` for value, reference, and
        initializing extended types respectively.
    -   `exttypeof(`_expression_`)` gives the extended type of the _expression_
        without any runtime evaluation.
    -   `(`_et1_`,` _et2_`,` _et3_`) as Core.ExtType` creates a compound tuple
        extended type value.
    -   `{.a = `_et1_`, .b =` _et2_`} as Core.ExtType` creates a compound struct
        extended type value.
-   that extended type values have a `.type` property that gives the unqualified
    (without `partial` or `const` modifiers) version of the type associated with
    the extended type.
-   extended type facet types that are restricted versions of `Core.ExtType`,
    such as `Core.NoValExtType`, as well as `where .type = T` and
    `where .type impls C` constraints.
-   a `fwd` binding modifier `fwd `_name_`: `_extended type value_, where the
    extended type of the binding is determined by the _extended type value_,
    along with a `fwd` marker on call arguments with generic extended type.
-   a `fwd` return tag `fwd `_extended type value_ (usable anywhere `val`,
    `ref`, or `var` is allowed in a `->` return clause), where the extended type
    of that part of the return is determined by the _extended type value_.
-   `auto` in a position where a `Core.ExtType` is expected (`let fwd` bindings,
    `-> fwd auto` returns, and implicitly for positional parameters).
-   deduction rules for `Core.ExtType` parameters, including exact-value
    compile-time parameters (`fwd _: exttypeof(V)`) and exact type matching.

## Proposal

### Extended types and values

First, we introduce the new special type `Core.ExtType` that describes the
extended type of bindings (such as function parameters or in a `let`
declaration), returns, or expressions.

-   The new `fwd` binding modifier takes an extended type value to
    [specify the extended type of a binding](#specifying-the-extended-type-of-a-binding).
-   The new `fwd` return tag takes an extended type value to
    [specify the extended type of a return](#specifying-the-extended-type-of-a-return).

Values of `Core.ExtType` are compile-time only, like types. Extended type values
can be compound extended types or primitive extended types. A compound extended
type is either a tuple extended type or struct extended type.

A primitive `Core.ExtType` has (at least):

-   a type
-   a category, which can be `val`/value, `ref`/reference, or `var`/initializing
    (only `val`/value is currently supported when the phase is "compile time")
-   the phase, which can be "compile time" or "run time" (meaning "when the
    function is run")
-   for compile time phase, a specific compile-time value

> **Future work:** Once we have a "compile-time evaluation" proposal, "run time"
> should likely be replaced by "eval time."

Unlike [extended types of expressions](/docs/design/values.md#extended-types),
which distinguish `"template"` and `"symbolic"` expression phases,
`Core.ExtType` uses a single `"compile time"` phase:

-   Whether a compile-time constant is concrete, template-dependent, or
    checked-symbolic is a property of the constant itself (and becomes concrete
    under substitution). A `fwd` binding with a compile-time extended type binds
    to that constant, giving the bound name `"template"` expression phase when
    the constant is concrete or template-dependent and `"symbolic"` expression
    phase when it is a checked-symbolic constant. Note that this is different
    from `let generic`, even when the result is symbolic, since the constant's
    value is retained rather than replaced with a new opaque symbol.
-   Both checked-generic and `template` compile-time parameters accept either
    kind of compile-time constant per
    [leads issue #2153](https://github.com/carbon-language/carbon-lang/issues/2153)
    and
    [proposal #2964](p002964-expression-phase-terminology.md#proposal),
    so parameter matching on `Core.ExtType` does not distinguish them.

See
[the alternative considered](#separate-template-and-symbolic-phases-in-coreexttype).

#### Primitive extended type literals

A primitive `Core.ExtType` value with "run time" phase can be written
`exttype(`_cat_ _type_`)`, where _cat_ is a required category modifier, and
_type_ is a type expression:

| `exttype` syntax | category           |
| :--------------- | :----------------- |
| `exttype(val T)` | `val`/value        |
| `exttype(ref T)` | `ref`/reference    |
| `exttype(var T)` | `var`/initializing |

This was
[discussed on 2025-06-04](https://docs.google.com/document/d/1Yt-i5AmF76LSvD4TrWRIAE_92kii6j5yFiW-S7ahzlg/edit?tab=t.0#heading=h.stp5dpvithkz).
We use `val` for the value category in alignment with using `val` for value
category in returns in the resolution of
[leads issue #5522](https://github.com/carbon-language/carbon-lang/issues/5522).

#### `exttypeof`

You may also write `exttypeof(`_expression_`)` to get the extended type of the
expression argument. If the expression has "compile time" phase, then this will
result in an extended type with that phase and also the value (it will have
value category, since that is the only category for compile-time expressions).
If the argument to `exttypeof` is a call expression, the result will be the
extended type of that function's return. However, only compile-time evaluation
will be performed on the argument to `exttypeof`, the argument is never runtime
evaluated.

Examples:

```carbon
let template c: i32 = 1;
// exttypeof(c) == exttypeof(1 as i32)

let v: i32 = 2;
// exttypeof(v) == exttype(val i32)

var r: i32 = 3;
// exttypeof(r) == exttype(ref i32)

fn Prints() -> i32 {
  Core.Print("This isn't printed");
  return 4;
}
// exttypeof(Prints()) == exttype(var i32)
```

This was
[discussed on 2025-06-04](https://docs.google.com/document/d/1Yt-i5AmF76LSvD4TrWRIAE_92kii6j5yFiW-S7ahzlg/edit?tab=t.0#heading=h.stp5dpvithkz).

#### Compound extended type values

To get a compound tuple extended type value, take a tuple of extended type
values and cast it to `Core.ExtType`, as in:

```carbon
(exttype(val T), exttype(var U), exttypeof(7))
    as Core.ExtType
```

Similarly, a compound struct extended type value can be constructed from a
struct of extended type values cast to `Core.ExtType`:

```carbon
{.a = exttype(val T),
 .b = exttype(var U),
 .c = exttypeof(7)
} as Core.ExtType
```

These may be nested, as in:

```carbon
// Struct extended type containing a tuple extended type
{.a = (exttype(val T), exttype(var U)),
 .b = exttype(ref V)
} as Core.ExtType

// Tuple extended type containing a struct extended type
({.a = exttype(val T), .b = exttype(var U)},
 exttype(ref V)
) as Core.ExtType
```

Note that `fwd` bindings and `fwd` return tags cast their `Core.ExtType` operand
to `Core.ExtType`, much like ordinary `:` bindings and `->` return types cast
their operand to `type`, so the `as Core.ExtType` can be left off in those
positions.

This was
[discussed on 2025-06-04](https://docs.google.com/document/d/1Yt-i5AmF76LSvD4TrWRIAE_92kii6j5yFiW-S7ahzlg/edit?tab=t.0#heading=h.stp5dpvithkz).

#### `type` property of extended type values

Every extended type value has a `type` property, giving its _unqualified type_,
accessed by writing _\<exttype_value\>_`.type`, following these rules:

-   `exttype(val T).type`, `exttype(var T).type`, and `exttype(ref T).type` are
    all `T`
-   `((X, Y) as Core.ExtType).type` is `(X.type, Y.type) as type`
-   `({.a = X, .b = Y} as Core.ExtType).type` is `{.a: X.type, .b: Y.type}`

This definition of the type part of an extended type is consistent with
[proposal #5545: "Expression form basics"](https://github.com/carbon-language/carbon-lang/pull/5545).
We additionally remove any `const` or `partial` qualifiers to get the
_unqualified_ type. For example:

-   `exttype(val const T).type` is `T`
-   `({.a = exttype(ref const T), .b = exttype(var partial U)} as Core.ExtType).type`
    is `{.a: T, .b: U}`

We want the unqualified type here, since `.type` will be used for the orphan
rule in
[proposal #8185](https://github.com/carbon-language/carbon-lang/pull/8185),
where we don't want to distinguish types with the `const` or `partial`
modifiers. See
[discussion on 2025-06-24](https://docs.google.com/document/d/1Yt-i5AmF76LSvD4TrWRIAE_92kii6j5yFiW-S7ahzlg/edit?tab=t.0#heading=h.d985djkv6qti).

### Extended type facet types

_Extended type facet types_ are our way of expressing an extended type with some
additional restrictions. The values of an extended type facet type are _extended
type facets_, which are values that are implicitly convertible to
`Core.ExtType`. A value of `Core.NoRefExtType`, like
`exttype(val i32) as Core.NoRefExtType`, is implicitly convertible to
`Core.ExtType`, resulting in `exttype(val i32)`. Here are some extended type
facet types with different restrictions:

-   `Core.NoValExtType`
-   `Core.NoRefExtType`
-   `Core.NoVarExtType`
-   `Core.NoCompileTimeExtType`
-   `Core.NoRunTimeExtType`

When converting an extended type value to a different extended type facet type,
the conversion fails if the extended type is not allowed by the destination
type. For example, converting `exttype(var i32)` to `Core.NoVarExtType` or
`exttypeof(7)` to `Core.NoCompileTimeExtType` are both invalid.

Use `&` to combine extended type facet types to get one that requires its values
to satisfy the constraints from both, as in
`Core.NoRefExtType & Core.NoCompileTimeExtType` to say an extended type that
doesn't use `ref` or compile-time extended types.

**QUESTION:** An earlier iteration of
[proposal #5389](https://github.com/carbon-language/carbon-lang/pull/5389) used
`Core.NoRefExtType` for primitive operator interfaces (now in
[proposal #8188](https://github.com/carbon-language/carbon-lang/pull/8188)). Now
that those uses were changed to `Core.ExtType`, it isn't clear that this feature
is pulling its weight.

### Specifying the extended type of a binding

The `fwd n: B` binding syntax creates a binding to the name `n` with extended
type given by `B` (as suggested in
[leads issue #6932](https://github.com/carbon-language/carbon-lang/issues/6932)
and
[proposal #7254](p007254-replace-and-with-keywords-and-contextual-defaults.md#extended-types)).
With the `fwd` modifier, `B` is implicitly converted to `Core.ExtType`, rather
than implicitly converting the right-hand side of `:` to `type`. Example:

```carbon
fn VarFn(fwd x: exttype(var i32));
// is equivalent to:
fn VarFn(var x: i32);

fn RefFn(fwd x: exttype(ref i32));
// is equivalent to:
fn RefFn(ref x: i32);
```

While this is valid, this is not how this feature is expected to be used.
Instead, the extended type value will typically be a generic parameter.

```carbon
fn Explicit(generic PET: Core.ExtType, fwd arg: PET);

// ✅ Valid since `5` can be passed to a
// `var _: i32` parameter.
Explicit(exttype(var i32), 5);

// ❌ Invalid, `5` is not a reference
// expression.
Explicit(exttype(ref i32), 5);

// ❌ Invalid, `true` can't be cast to `f32`.
Explicit(exttype(val f32), true);
```

A `Core.ExtType` parameter is deducible from an argument expression, see
[the "Deducing an extended type value" section](#deducing-an-extended-type-value).
In fact, the extended type of a binding is _about_ the expressions it binds to.
In contrast, the result of a `var` and `ref` binding is a reference expression
in both cases.

When passing an expression whose extended type is symbolic (for example, a `fwd`
parameter or a call returning a symbolic extended type) as an argument to a
function or operator, the argument is marked with `fwd` at the call site
(analogously to marking a concrete reference argument with `ref` when passing it
to a `ref` parameter). When moving from a symbolic-extended-type binding with
`~`, `fwd` precedes `~`, as in `F(fwd ~y)`. As with `ref`, `fwd` is not written
on method receivers (such as `c.Go()`) or on the callable expression in a
function call (such as `f()`). See
[the alternative considered](#fwd-on-call-arguments).

A `fwd` binding cannot appear directly or indirectly in a pattern operator other
than `unused` or be combined with another keyword binding modifier (such as
`ref`, `generic`, `template`, or `runtime`). This disallows patterns like
`var fwd x: PET`, `ref fwd x: PET`, `generic fwd x: PET`, and
`template fwd x: PET`, as well as `var` declarations like
`var fwd x: auto = y;`. In contexts where a contextual default phase applies
(such as deduced `[...]` parameter lists, where parameters default to
`generic`), `fwd` overrides the contextual default rather than combining with
it. Note that `each` is part of an
[_each-name_](/docs/design/variadics.md#packs-and-each-names) rather than a
binding modifier, so `fwd each x: each PET` is allowed. Nesting a `fwd` binding
inside tuple patterns, struct patterns, pack expansions, and
[`unused` patterns](/docs/design/pattern_matching.md#unused) (as well as
anonymous `fwd _: PET` bindings) is allowed.

### Specifying the extended type of a return

A `fwd R` return tag specifies a return with extended type `R`, where `R` is
implicitly converted to `Core.ExtType` rather than `type`. `fwd` is allowed
anywhere a [category tag](/docs/design/values.md#function-calls-and-returns)
(`val`, `ref`, or `var`) is allowed in a `->` return clause — both at top level
(`-> fwd R` and `-> fwd auto`) and nested inside compound tuple or struct
extended return types (as discussed in
[leads issue #6932](https://github.com/carbon-language/carbon-lang/issues/6932)
and
[proposal #7254](p007254-replace-and-with-keywords-and-contextual-defaults.md#extended-types)).

```carbon
// Equivalent to: `fn Init() -> i32;`
fn Init() -> fwd exttype(var i32);

// Equivalent to: `fn Val() -> val i32;`
fn Val() -> fwd exttype(val i32);

// Equivalent to: `fn Ref() -> ref i32;`
fn Ref() -> fwd exttype(ref i32);

// Determines the extended return type from
// the argument. `Param(exttype(ref i32))`
// returns an `i32` ref.
fn Param(generic RET: Core.ExtType) -> fwd RET;

// `fwd` may also be nested inside tuple and struct extended return types:
fn Pair(generic R1: Core.ExtType, generic R2: Core.ExtType)
    -> (fwd R1, fwd R2);
fn Struct(generic R1: Core.ExtType)
    -> {.a: fwd R1, .b: ref i32};
```

### Any extended type with this type

To create an [extended type facet type](#extended-type-facet-types) expressing
"any extended type with unqualified type `T`", we use the
[`type` property](#type-property-of-extended-type-values) in a `where` clause,
as in `Core.ExtType where .type = T`.

This approach can also be used to express "any extended type with an unqualified
type satisfying facet type `C`" as in `Core.ExtType where .type impls C`. (For
converting in the opposite direction — from an extended type facet type to a
type facet type — see
[proposal #8185](https://github.com/carbon-language/carbon-lang/pull/8185).)

In both cases, the `Core.ExtType` part can be replaced by another
[extended type facet type](#extended-type-facet-types), in order to compose
extended type constraints.

This approach also allows deduction of a type through an extended type, as in:

```carbon
fn F[T: I](generic PET: Core.ExtType where
           .type = Vector(T))
    -> T.Iterator;
```

since the `T` value is uniquely determined by `PET`. (See
[proposal #8185](https://github.com/carbon-language/carbon-lang/pull/8185) for
how `where .type = ...` also interacts with the orphan rule in `impl`
declarations.)

This was
[discussed on 2025-06-11](https://docs.google.com/document/d/1Yt-i5AmF76LSvD4TrWRIAE_92kii6j5yFiW-S7ahzlg/edit?tab=t.0#heading=h.rebuyhkgncpq),
along with alternative ways we could specify this constraint.

### `auto` extended type

If `auto` is used in a position where a `Core.ExtType` is expected, such as in a
`fwd` binding or `-> fwd` return, the whole extended type is deduced, not just
the type.

For `fwd` `auto` parameters, specifically, the deduced extended type will not be
`ref` (using `val` instead) unless the argument expression is marked `ref` (or
`fwd` for a generic-extended-type expression). See
[the "Deducing an extended type value" section](#deducing-an-extended-type-value).

#### `auto` extended type bindings

This allows us to create a binding to another value without changing its phase
or category.

```carbon
let fwd x: auto = y;
// Equivalent to (if it was allowed):
let [P: Core.ExtType] fwd x: P = y;

// In contrast,
let a: auto = b;
// only allows the type to vary
// and forces `a` to have value
// category and runtime phase.
```

This allows us to address the use case of creating a binding that is
[referentially transparent](https://en.wikipedia.org/wiki/Referential_transparency),
as considered in leads issues
[#996](https://github.com/carbon-language/carbon-lang/issues/996) and
[#1371](https://github.com/carbon-language/carbon-lang/issues/1371). For
example, if the `y` in the example is a reference expression, `x` will be a
reference binding to it. If `y` is a compile-time value, `x` will be the same
value, with the same phase. However, if `y` is an initializing expression, `x`
is a `var` binding and so naming `x` produces a reference expression rather than
an initializing expression, and you will need to move from it with `~x` (once
supported) to get an initializing expression.

(This construct is also used to redefine `alias` in
[proposal #8184](https://github.com/carbon-language/carbon-lang/pull/8184).)

This was discussed on
[2025-05-07](https://docs.google.com/document/d/1Yt-i5AmF76LSvD4TrWRIAE_92kii6j5yFiW-S7ahzlg/edit?tab=t.0#heading=h.4zbo49wg5rmk).

#### Positional parameters

[Positional parameters](/docs/design/functions.md#positional-parameters),
introduced in
[proposal #3848](p003848-lambdas.md#positional-parameters),
previously had type `auto`. This proposal changes them to have `auto` extended
type. So `fn F {`...`}` acts as if it had signature `fn F(... fwd _: auto)`.
This means that an argument that is a reference expression that is modified by
the function body must be marked `ref`, as in:

```carbon
var x: i32 = 1;
let IncBy: auto = fn { $0 += $1; };
IncBy(ref x, 2);
Assert(x == 3);

let Add: auto = fn => $0 + $1;
// Will deduce a value extended type since
// not marked `ref`:
Assert(Add(x, 3) == 6);
```

#### `auto` extended type returns

Using `-> fwd auto` as the return of a function causes the extended return type,
not just type, to be determined by the extended type of the return expression.
For example, returning a reference expression will result in a `ref` return, as
in:

```carbon
// Equivalent to ...`-> ref i32`:
fn Id(ref x: i32) -> fwd auto {
  return x;
}
```

Note that the `=>` return syntax introduced in
[proposal #3848](https://github.com/carbon-language/carbon-lang/pull/3848) is
not changed to deduce extended type, as decided in
[proposal #5434](p005434-ref-parameters-arguments-returns-and-val-returns.md#-infers-form-not-just-type).
This still may be revisited once the memory safety story is more developed.

## Details

### Deducing an extended type value

A `Core.ExtType` parameter is deducible from an argument expression, allowing it
to capture the argument's expression category, phase, and compile-time value.

-   A compile-time argument will deduce a constant `exttypeof(N)`, including for
    symbolic constants.
-   An initializing expression argument will deduce a `var` extended type.
-   A reference expression argument will deduce a `ref` extended type.

```carbon
fn Deduced[PET: Core.ExtType](fwd arg: PET);

let l: bool = true;
// Deduces PET = exttype(val bool).
Deduced(l);

var v: i32 = 4;
// Deduces PET = exttype(ref i32).
Deduced(ref v);

// A call to `Make()` is an initializing expression.
fn Make() -> i32;

// Deduces PET = exttype(var i32).
Deduced(Make());

// Deduces PET = exttypeof(7).
Deduced(7);

fn SymbolicN(generic N: i32) {
  // Deduces PET = exttypeof(N).
  Deduced(N);
}
```

In contrast to
[extended type facet type conversion](#extended-type-facet-types), when an
extended type facet parameter is deduced, the compiler is willing to change
categories to get a legal value of the extended type facet type.

-   When deducing a `Core.NoValExtType`, use category `var`/initializing instead
    of `val`/value.
-   When deducing a `Core.NoRefExtType`, use category `val`/value instead of
    `ref`/reference.
-   When deducing a `Core.NoVarExtType`, use category `val`/value instead of
    `var`/initializing.

```carbon
fn NoVar[P: Core.NoVarExtType](fwd x: P);

// Deduces P = exttype(val i32)
NoVar(Make());
```

If there is only one allowed destination category, that will be the category of
the result (as in `var`/initializing for
`Core.NoValExtType & Core.NoRefExtType`). However, this case would generally be
better supported by a type facet type (see
[proposal #8185](https://github.com/carbon-language/carbon-lang/pull/8185)).

An argument expression will only deduce a `ref` extended type if it is marked
`ref` (or `fwd`, for a generic-extended-type expression that may instantiate to
a reference). Otherwise, it is as if the parameter has a `& Core.NoRefExtType`
restriction. This was discussed
[in #pointers-and-references on Discord on 2025-06-18](https://discord.com/channels/655572317891461132/753021843459538996/1385028689904926763).

### Compile-time parameter equal to a specified value

The [`fwd` binding syntax](#specifying-the-extended-type-of-a-binding) and
[`exttypeof` operator for computing an extended type value](#exttypeof) may be
combined to write a parameter binding that will only match a single specified
value, as in:

```carbon
// Argument must be a compile-time value equal to `i32`
fn F(fwd _: exttypeof(i32));

// ✅ Allowed
F(i32);

// ❌ not something with type `i32`
F(7 as i32);

// ❌ not a different type
F(i16);

fn G(generic T: type) {
  // ❌ not known to be `i32`
  F(T);
}

fn H(template T: type) {
  F(T);
}

// ✅ Allowed
H(i32);

// ❌ error substituting in `i16` for `T`
H(i16);
```

This was discussed on
[2025-05-13](https://docs.google.com/document/d/1Yt-i5AmF76LSvD4TrWRIAE_92kii6j5yFiW-S7ahzlg/edit?tab=t.0#heading=h.bdznj2d0by2g).
It provides a resolution of
[Leads issue #578](https://github.com/carbon-language/carbon-lang/issues/578),
giving a method to work around ordering constraints on parameters:

```carbon
fn DoACast[U: type, T: ImplicitAs(U)]
    (x: T, fwd _: exttypeof(U)) -> U {
  return x as U;
}

DoACast(7, i32);
```

This feature is also used in
[proposal #8187](https://github.com/carbon-language/carbon-lang/pull/8187) to
model explicit compile-time parameters in `Core.CallPrimitive`, so they can be
used in other parts of the signature.

### Parameter requires exact type match

We can also prevent implicit type conversions for a parameter. Recall that using
a deduced type means that the compiler won't consider types that implicitly
convert to that type, relying on deduction to match instead. If the deduced type
parameter is constrained to have a specific compile-time value using the
technique from
[the previous section](#compile-time-parameter-equal-to-a-specified-value), then
only that type will be permitted.

For example:

```carbon
// This function can only be passed values of type
// `bool`. `T` will be deduced from the argument
// type, but can only have the value `bool`.
fn ExactlyBool[fwd T: exttypeof(bool)](x: T);

// ✅ Allowed
ExactlyBool(true);
ExactlyBool(false);

class Truthy {
  impl as ImplicitAs(bool) {
    fn Op(self) -> bool { return true; }
  }
}
let truthy: Truthy = {};

// ❌ Invalid
ExactlyBool(truthy);
```

Note that this mechanism can be defeated. Of course wrapping the function will
do it:

```carbon
fn BoolConversionFirst(x: bool) {
  ExactlyBool(x);
}

// ✅ Allowed
BoolConversionFirst(truthy);
```

Similarly, passing `ExactlyBool` as a callable parameter constrained to accept a
`bool` (as in
[proposal #8186](https://github.com/carbon-language/carbon-lang/pull/8186) and
[proposal #8187](https://github.com/carbon-language/carbon-lang/pull/8187)) will
also lose the exact type match property because the facet creates an adapter
with the constraint's signature.

### Input-to-output forwarding

Forwarding is where an argument is passed along generically, including aspects
of its extended type like category and phase. Input-to-output forwarding is
where argument extended types are returned from a function, as in:

```carbon
fn Id[ET: Core.NoVarExtType](fwd x: ET) -> fwd ET {
  return x;
}
```

This requires `Core.NoVarExtType`, since the calling convention for an
initializing return requires the ability to construct the return value at an
address provided by the caller for some types, different from the address the
argument is passed in.

> **Future work:** Once supported, a requirement that the type be movable would
> allow preserving category, including for initializing expressions. Perhaps
> something like:
>
> ```carbon
> fn Moves
>     [ET: Core.ExtType where .type impls Core.Move]
>     (fwd x: ET) -> fwd ET {
>   return ~x;
> }
> ```

Input-to-input forwarding of generic callables (such as `Apply`, `Apply2`, and
forwarding tuple parameters) builds on this foundation in
[proposal #8187](https://github.com/carbon-language/carbon-lang/pull/8187).

This was discussed in open discussion:

-   [2025-05-07](https://docs.google.com/document/d/1Yt-i5AmF76LSvD4TrWRIAE_92kii6j5yFiW-S7ahzlg/edit?tab=t.0#heading=h.4zbo49wg5rmk):
    forwarding using the proposed extended types
-   [2025-05-12](https://docs.google.com/document/d/1Yt-i5AmF76LSvD4TrWRIAE_92kii6j5yFiW-S7ahzlg/edit?tab=t.0#heading=h.1mjh6unumnwu):
    calling conventions limit forwarding for `var` extended types

## Future work

### Compile-time calls

A function executed at compile time has a return value that will be determined
at compile time. This is a statement about the extended type of the function's
return. Ideally Carbon will support functions that can be executed either at
compile or run time (like C++'s
[`constexpr`](https://en.cppreference.com/w/cpp/language/constexpr.html)), and
compile-time functions (like C++'s
[`consteval`](https://en.cppreference.com/w/cpp/language/consteval.html)) that
aren't instantiated for every distinct compile-time input argument.

This was
[discussed on 2025-06-03](https://docs.google.com/document/d/1Yt-i5AmF76LSvD4TrWRIAE_92kii6j5yFiW-S7ahzlg/edit?tab=t.0#heading=h.hdj6hk4bwd2z).
In short, we expect:

-   To introduce another phase that represents values that are constant, in the
    sense of having a value determined at compile time, but where the value is
    unavailable for typechecking.
-   We will rename "runtime" to "execute time," to include the values of
    functions whether they are executed at compile time or runtime.

### `ref?`

We would like to support being generic across `val` and `ref` for accessor
functions -- where the `self` and the return are both `val` or both `ref`. One
issue is we can do input-to-input, output-to-output, and output-to-input
forwarding of all three categories, but accessors want to do input-to-output
forwarding, which we can't do for initializing expressions. One possibility
would be to define a special syntax for just this case like `ref?`, as in:

```carbon
class C {
  private var x: D;
  fn AccessX(ref? self) -> ref? D {
    return self.x;
  }
}
```

This would be equivalent to an overload of the two versions of the function.
`ref?` would have to be used on at least one explicit or `self` parameter, and
the `ref` version would be used if all of those arguments were reference
expressions (we might require explicit arguments to be marked with `ref`). An
initializing expression counts as a `val` expression for those purposes, since
`ref` would mean enabling mutation of a temporary.

On
[2025-09-04](https://docs.google.com/document/d/1Yt-i5AmF76LSvD4TrWRIAE_92kii6j5yFiW-S7ahzlg/edit?tab=t.0#heading=h.ruyknn9cg56v),
we discussed that this feature would also be useful in an `impl` to implement
the same extended type interface twice.

Is it a flaw in the design that `ref?` looks to be a useful extended type
genericity, but is not well supported without a separate feature? We see this
use case coming up with indexing, where indexing into an array gives you a
reference if you started with a reference to the array, and a value if you
started with a value.

One possible approach would be to have something like:

```carbon
fn AccessX[PET: ValOrRef(Self)](fwd self: PET) -> fwd ExtTypeWithCategory(PET, D);
```

`ValOrRef(Self)` (passed a type and returns an extended type facet type) is
equivalent to `Core.NoVarExtType where .type = Self`. However, we are missing
`ExtTypeWithCategory` (passed an `ExtType` and a type, returns an extended type
value), and the machinery to definition check this generically. This option was
considered in
[discussion on 2025-05-13](https://docs.google.com/document/d/1Yt-i5AmF76LSvD4TrWRIAE_92kii6j5yFiW-S7ahzlg/edit?tab=t.0#heading=h.bdznj2d0by2g).

A C++-like approach would be to have conditional versions of our existing `var`,
`ref`, `template` keywords, perhaps with a trailing `_`:

```carbon
fn AccessX[IsRef: bool](ref_(IsRef) self)
  -> ref_(IsRef) D;
```

This last option was discussed on
[2025-06-09](https://docs.google.com/document/d/1Yt-i5AmF76LSvD4TrWRIAE_92kii6j5yFiW-S7ahzlg/edit?tab=t.0#heading=h.xktpa55flu0z).

## Rationale

This advances [Carbon's goals](/docs/project/goals.md):

-   [Code that is easy to read, understand, and write](/docs/project/goals.md#code-that-is-easy-to-read-understand-and-write):
    This proposal allows some programs and APIs to be written that couldn't
    otherwise be expressed in Carbon (such as referentially transparent
    bindings, exact-value compile-time parameters, and category-preserving
    forwarding).
-   [Performance-critical software](/docs/project/goals.md#performance-critical-software):
    Preserving expression categories across generic bindings and returns avoids
    unnecessary copies and materializations.

It does this while adhering to [Carbon's principles](/docs/project/principles):

-   [Principle: All APIs are library APIs](/docs/project/principles/library_apis_only.md):
    Making extended types first-class compile-time values lays the groundwork
    for expressing callables, operators, and conversions as library interfaces
    in follow-up proposals.
-   [Principle: Progressive disclosure](/docs/project/principles/progressive_disclosure.md):
    Developers don't need to be aware of `Core.ExtType` or `fwd` unless they
    are writing generic code that needs to abstract over category or phase.

## Alternatives considered

### Decoupling type from the rest of the extended type

The goal of this approach was to avoid introducing extended type interfaces (in
[proposal #8185](https://github.com/carbon-language/carbon-lang/pull/8185)).
Instead, the non-type parts of the extended type could be passed in as a
separate parameter to supplement the `Self` type. This ran into a few problems
though:

-   It didn't support compound extended types well, and as a result had trouble
    supporting some forwarding use cases.
-   For the compile-time value extended type `exttypeof(N)`, the additional
    information beyond the type includes a value of that type. As a result, the
    additional information was parameterized by the type, which was complex and
    introduced dependency cycles in some cases.

There are cases where we still would like to separately specify type from the
rest of the extended type, but we do that without a representation for the rest
of the extended type. See
[the "Any extended type with this type" section](#any-extended-type-with-this-type).

See also the
["Mixed expression categories" alternative in proposal #5545](p005545-expression-form-basics.md#mixed-expression-categories)
for a related discussion of keeping types and expression categories separate in
general.

This was discussed on
[2025-05-07](https://docs.google.com/document/d/1Yt-i5AmF76LSvD4TrWRIAE_92kii6j5yFiW-S7ahzlg/edit?tab=t.0#heading=h.4zbo49wg5rmk)
and
[2025-06-11](https://docs.google.com/document/d/1Yt-i5AmF76LSvD4TrWRIAE_92kii6j5yFiW-S7ahzlg/edit?tab=t.0#heading=h.rebuyhkgncpq).

### Other ways to write "any extended type with a specified type"

We
[considered on 2025-06-11](https://docs.google.com/document/d/1Yt-i5AmF76LSvD4TrWRIAE_92kii6j5yFiW-S7ahzlg/edit?tab=t.0#heading=h.rebuyhkgncpq)
other ways of specifying any extended type with a specified type than
[the `Core.ExtType where .type` approach we settled on](#any-extended-type-with-this-type).

One approach was to have a constraint `ExtTypeWithType(`<type>`)`. However, the
approach we settled on more naturally supported getting the type from the
extended type using `.type` (for example in `impl`s in
[proposal #8185](https://github.com/carbon-language/carbon-lang/pull/8185)),
which
[we wanted to work with other rewrite constraints](https://discord.com/channels/655572317891461132/941071822756143115/1381487716776022107).

The other approach we considered was decomposing extended types into types and
non-type information, which ran into the problem that
[the non-type information depended on the type](#decoupling-type-from-the-rest-of-the-extended-type).

### Don't have types that represent all binding information

We could have aspects of bindings without reflecting them into the type system.
For example, Swift has reference bindings, but does not represent references in
its type system. This means no way to do perfect forwarding. More significantly
to Carbon, it would mean we wouldn't have a way to represent calls using an
interface (as in
[proposal #8187](https://github.com/carbon-language/carbon-lang/pull/8187)).
This was
[discussed on 2025-05-28](https://docs.google.com/document/d/1Yt-i5AmF76LSvD4TrWRIAE_92kii6j5yFiW-S7ahzlg/edit?tab=t.0#heading=h.9pvtvm7atpo5).

### Represent all extended type information in ordinary types

We could represent the additional information present in extended types in
ordinary types. This would avoid splitting constructs, such as different kinds
of interfaces for types and extended types (in
[proposal #8185](https://github.com/carbon-language/carbon-lang/pull/8185)). The
concern is that there are cases where we want to consider type separate from the
other aspects of extended type. Further, we want the rules for extended types to
be different than types. For example, category and phase are independent, not
ordered like type composition. Also, different categories are mutually
exclusive.

Additionally, having extended types separate from ordinary types means we can
hide some complexity from people learning the language under
[progressive disclosure](/docs/project/principles/progressive_disclosure.md).

This was
[discussed on 2025-05-28](https://docs.google.com/document/d/1Yt-i5AmF76LSvD4TrWRIAE_92kii6j5yFiW-S7ahzlg/edit?tab=t.0#heading=h.9pvtvm7atpo5).

### Different kinds of extended types for parameters and returns

We originally considered using different types for the extended types used in
parameters and returns. This allowed some modelling of the differences, for
example between `var` parameters and initializing returns, see the discussions
on
[2025-05-14](https://docs.google.com/document/d/1Yt-i5AmF76LSvD4TrWRIAE_92kii6j5yFiW-S7ahzlg/edit?tab=t.0#heading=h.52tb7l2he343)
and
[2025-05-13](https://docs.google.com/document/d/1Yt-i5AmF76LSvD4TrWRIAE_92kii6j5yFiW-S7ahzlg/edit?tab=t.0#heading=h.bdznj2d0by2g).
Ultimately this led to a proliferation of different extended type kinds,
increasing complexity in an area that is already quite complex. Merging has some
downsides, for example that compile-time value information is only currently
used for parameters, but in
[a discussion on 2025-06-03](https://docs.google.com/document/d/1Yt-i5AmF76LSvD4TrWRIAE_92kii6j5yFiW-S7ahzlg/edit?tab=t.0#heading=h.hdj6hk4bwd2z)
we determined that is expected to change once we support
[functions executed at compile-time](#compile-time-calls).

The additional type information couldn't be used to determine the variance of an
extended type parameter (see
[proposal #8186](https://github.com/carbon-language/carbon-lang/pull/8186)),
since extended types in `fwd` bindings can appear in positions with different
variance, as
[discussed on 2025-06-24](https://docs.google.com/document/d/1Yt-i5AmF76LSvD4TrWRIAE_92kii6j5yFiW-S7ahzlg/edit?tab=t.0#heading=h.dqgfbeb33txh).

Example (using `CallPrimitive` and `call` from
[proposal #8187](https://github.com/carbon-language/carbon-lang/pull/8187)):

```carbon
interface CallbackRegistry(PET: Core.ExtType) {
  fn Register[T: CallPrimitive(PET)](fwd f: T);
}

class CR { }
impl CR as CallbackRegistry(exttype(val i32)) {
  fn Register[T: call(_: i32)](fwd f: T) {
    let i: i32 = 999999;
    f(i);
  }
}

fn F(x: i16);

fn G(generic U: CallbackRegistry(_: i16)) {
  U.Register(F);
}

// Type error: calling `F` with an `i32` value.
G(CR);
```

In this next example, `PET` needs to be invariant, since it is used in both
covariant and contravariant positions:

```carbon
interface J(PET: Core.ExtType) {
  fn F(c: C(PET));
  fn G() -> C(PET);
}
```

### Implicit conversion from `type` to `Core.ExtType`

We originally defined an implicit conversion from type to extended type, for
example in
[discussion on 2025-05-08](https://docs.google.com/document/d/1Yt-i5AmF76LSvD4TrWRIAE_92kii6j5yFiW-S7ahzlg/edit?tab=t.0#heading=h.vdognq1upsf5).
Originally this would result in a value extended type, which would allow briefer
specification of the parameters or return for `CallPrimitive` (in
[proposal #8187](https://github.com/carbon-language/carbon-lang/pull/8187)).
However, in
[discussion on 2025-05-14](https://docs.google.com/document/d/1Yt-i5AmF76LSvD4TrWRIAE_92kii6j5yFiW-S7ahzlg/edit?tab=t.0#heading=h.52tb7l2he343)
we decided against having any implicit conversion, for a few reasons:

-   The natural default for parameter extended types (value) is different from
    extended return types (initializing). Picking one would lead to confusion
    when using the other. Making it different somehow using contextual
    information would lead to inconsistencies that were also expected to be
    confusing.
-   It did not allow you to deal only in types instead of using extended types
    in any use case we foresaw.
-   Extended types are an advanced feature that is expected to be used rarely.
    As a result, brevity is not as valuable, and readers are less likely to be
    familiar with which default would be used. The `CallPrimitive` use case is
    already well supported by using `call` sugar (in
    [proposal #8187](https://github.com/carbon-language/carbon-lang/pull/8187))
    instead.
-   Using the implicit conversion for specifying arguments to `CallPrimitive`
    could lead to ambiguity. For example `CallPrimitive((T, U))` could be
    equivalent to `call((_: T, _: U))` or `call(_: (T, U))`.

### Different extended type literal syntax

In a
[discussion on 2025-06-04](https://docs.google.com/document/d/1Yt-i5AmF76LSvD4TrWRIAE_92kii6j5yFiW-S7ahzlg/edit?tab=t.0#heading=h.stp5dpvithkz),
we considered other ways to write the extended type of a specific compile-time
value, like `7 as i32`. We considered using `exttype(`...`)` with an introducer
before an expression that evaluates to the value:

-   `exttype(== 7 as i32)`
-   `exttype(=> 7 as i32)`
-   `exttype(is 7 as i32)`

Of these, `exttype(=> 7 as i32)` was the most promising, since it matched the
syntax for returning an expression from a function or `case`. The `is` keyword
was considered too expensive to introduce just for this use.

We also considered using no introducer inside `exttype(`...`)`, as in
`exttype(7 as i32)`, but it had some downsides:

-   Required lookahead when parsing.
-   We didn't want `exttype(i32)` to be legal and mean something other than
    `exttype(val i32)`, which is more likely what the code author intended.

Ultimately we didn't think that this was a common case that needed to be short.

We settled on [the `exttypeof` approach](#exttypeof). We expect that
`exttypeof(`...`)` will be a valuable construct that we want outside of
compile-time value extended types.

For compound extended types, we first tried to represent them using parens
`(`...`)` and curly braces `{`...`}` inside `exttype(`...`)`. There was some
concern about ambiguity between the `()` value and the empty tuple extended
type, similarly for `{}`, but that was addressed by switching to `exttypeof` for
compile-time constant values. However, we also needed a way to compose
`exttype(`...`)`, `exttypeof(`...`)`, and arbitrary `Core.ExtType` expressions
(such as generic extended type parameters) into compound extended type values.
Allowing mixed elements inside `exttype(`...`)` (such as
`exttype((var i32, ref f64, exttypeof(7)))`) would require a dedicated grammar
distinct from `->` extended return types (where a bare expression is a type with
implicit `var` category, and a `Core.ExtType` expression must be prefixed with
`fwd`). The
[compound extended type approach we settled on](#compound-extended-type-values)
had the advantage of supporting all the composition we wanted without needing
any dedicated syntax. It is a bit verbose, but that is perhaps acceptable given
that this isn't a commonly used construct. It mirrored our approach for tuple
types, but not for struct types.

As future work, we could potentially expand what is allowed inside
`exttype(`...`)`, if we determine that the current approach is too verbose. This
would build on the work done in
[proposal #5434](https://github.com/carbon-language/carbon-lang/pull/5434) to
define a compound `->` extended return type syntax that was done since
[proposal #5389](https://github.com/carbon-language/carbon-lang/pull/5389) was
originally authored. For now, since the mixed case will continue to use the
current approach, we are avoiding introducing a second way of expressing these
extended types.

In a
[discussion on 2025-06-05](https://docs.google.com/document/d/1Yt-i5AmF76LSvD4TrWRIAE_92kii6j5yFiW-S7ahzlg/edit?tab=t.0#heading=h.itqt68gnfeob)
we considered `exttype(-> T)` instead of `exttype(var T)`, but we wanted to be
consistent about using keyword introducers for the different categories. We also
considered `exttype(init T)`, which was better for returns but worse for
parameters, but was not worth making `init` into a keyword.

In that discussion, we also considered `exttype(T)` instead of
`exttype(val T)`, but it seemed like it would lead to mistakes for returns where
`val` was not the default.

### `ref` on an argument controls its category

We considered an approach where `ref` on an argument would control whether a
reference expression was converted to a value expression before overload
resolution. Consider this example (using the overloading syntax proposed in
[discussion on 2025-03-28](https://docs.google.com/document/d/1Iut5f2TQBrtBNIduF4vJYOKfw7MbS8xH_J01_Q4e6Rk/edit?resourcekey=0-mc_vh5UzrzXfU4kO-3tOjA&tab=t.0#heading=h.t6l733mu79i7)):

```carbon
overload F {
  // Overload R
  fn (ref r: i32) -> i32;
  // Overload V
  fn (v: i32) -> bool;
}

var a: i32 = 1;
F(a);
```

With this approach, the call to `F` would convert the reference expression `a`
into a value before overload resolution and select "Overload V."

We instead decided that `ref` on an argument only affects deduction. In this
model, the call to `F` would instead select "Overload R" and then give an error
because the call site does not mark the argument with the `ref` keyword. This
means an overload set with different parameter categories behaves differently
than a function that allows the category to vary using extended type generics.
The benefits of this decision are that this approach:

-   makes `ref` more like a readability annotation than a category cast;
-   makes `ref` argument marking easier to remove as part of language evolution;
-   makes pattern matching of argument to function parameters more consistent
    with other pattern matching contexts; and
-   didn't introduce concerns about converting to a value interfering with
    converting the argument to the parameter's type.

This was discussed on
[2025-05-28](https://docs.google.com/document/d/1Yt-i5AmF76LSvD4TrWRIAE_92kii6j5yFiW-S7ahzlg/edit?tab=t.0#heading=h.9pvtvm7atpo5)
and in
[the #pointers-and-references Discord channel](https://discord.com/channels/655572317891461132/753021843459538996/1377506479355203635).

### Modifiers and pattern operators on `fwd` bindings

We considered allowing other keyword modifiers (such as `template` or `ref`) on
`fwd` bindings, or allowing `fwd` bindings inside `var` or `ref` patterns, to
override aspects of the specified extended type. We decided to disallow these
for two reasons:

-   Patterns such as `ref fwd x: exttype(val i32)`,
    `var fwd x: exttype(val i32)`, or `var (fwd x: exttype(val i32), y: i32)`
    have multiple plausible interpretations (such as overwriting the extended
    type's category, failing to match, or performing a conversion), and lack
    clear motivating use cases.
-   Allowing `template`, `generic`, `runtime`, `ref`, or `var` would provide
    syntax for coercing an extended type binding to some phases and categories,
    without any corresponding syntax for coercing it to value category (and with
    confusing interactions with contextual default phases).

This was discussed on
[PR #5389](https://github.com/carbon-language/carbon-lang/pull/5389#discussion_r2824307199)
and in
[Discord](https://discord.com/channels/655572317891461132/748959784815951963/1473464233697218611).

### Separate template and symbolic phases in `Core.ExtType`

We considered distinguishing `"template"` and `"symbolic"` phases in
`Core.ExtType`, matching the three phases (`"runtime"`, `"symbolic"`, and
`"template"`) of
[extended types of expressions](/docs/design/values.md#extended-types). We chose
a single `"compile time"` phase instead for three reasons:

-   Under
    [leads issue #2153](https://github.com/carbon-language/carbon-lang/issues/2153),
    a checked-generic function `fn Sym(generic T: C)` accepts template constants
    such as `Sym(i32)`, and a template function `fn Tmpl(template T: C)` accepts
    symbolic constants such as `Tmpl(U)` for `generic U: C`. If `exttypeof(T)`
    had `"symbolic"` phase in `Sym`'s `CallPrimitive` implementation (in
    [proposal #8187](https://github.com/carbon-language/carbon-lang/pull/8187))
    and `"template"` phase in `Tmpl`'s, `ConvertsInto` (in
    [proposal #8186](https://github.com/carbon-language/carbon-lang/pull/8186))
    would have to convert `"template"` extended types to `"symbolic"` extended
    types for `Sym(i32)` and `"symbolic"` extended types to `"template"`
    extended types for `Tmpl(U)` (including when forwarding through `Apply`).
    Converting in both directions would prevent `Core.ExtType` constraints from
    distinguishing the two phases anyway, whereas allowing implicit `"symbolic"`
    to `"template"` expression conversions in general would silently make
    checked-generic expressions template-dependent without an explicit
    `template` binding.
-   Substituting a concrete constant such as `i32` for a symbolic parameter `T`
    turns `exttypeof(T)` into `exttypeof(i32)`. Keeping `"symbolic"` and
    `"template"` as distinct `Core.ExtType` phases for equal constants would
    complicate substitution and `impl` coherence.
-   Whether a compile-time constant is concrete, template-dependent, or
    checked-symbolic is already carried by the constant itself, so a single
    `"compile time"` phase still preserves `"template"` and `"symbolic"`
    expression phases across `fwd` bindings, `alias`, and forwarding.

This was discussed in
[a suggestion thread on PR #5389](https://github.com/carbon-language/carbon-lang/pull/5389#discussion_r2337766010).

### `fwd` on call arguments

[Proposal #7254](p007254-replace-and-with-keywords-and-contextual-defaults.md#extended-types)
raised the open question of whether `fwd` should be required on call arguments,
analogously to how `ref` is required on arguments for reference parameters. We
considered three options:

-   **Never write `fwd` on call arguments**: An expression with generic extended
    type would be passed without any marker (for example, `F(~y)` and
    `(~f)(... ~each arg)`), while concrete reference expressions passed to a
    `ref` or deduced `fwd` parameter would continue to require `ref`.
-   **Require `fwd` on every argument to a `fwd` parameter**: Any call passing
    an argument to a parameter declared with `fwd` would write `fwd` on the
    argument. This would unnecessarily couple the caller to how the callee is
    declared, without any change to how the argument is passed. This works
    against Carbon's
    [evolution goal](/docs/project/goals.md#software-and-language-evolution). In
    addition, concrete reference arguments should continue to be visibly marked
    `ref` rather than `fwd` (or redundantly `fwd ref`), matching
    [the rule for `ref` on deduced extended type parameters](#deducing-an-extended-type-value).
-   **Require `fwd` when passing an expression with generic extended type
    (chosen)**: Call sites passing concrete expressions follow the normal rules
    (no marker for values and initializing expressions, `ref` for references),
    while call sites passing an expression whose extended type is generic mark
    the argument with `fwd` (for example, `F(fwd ~y)`,
    `(~f)(... fwd ~each arg)`, and `self.Compare(fwd u)`).

We went with the last option to highlight the unusual argument forwarding at the
call site, including that the argument might be passed by `ref`, allowing
mutation, without imposing `fwd` on callers passing concrete arguments.
