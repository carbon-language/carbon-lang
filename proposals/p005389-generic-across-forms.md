# Generic across forms

<!--
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

[Pull request](https://github.com/carbon-language/carbon-lang/pull/5389)

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
    -   [Extended type interfaces](#extended-type-interfaces)
    -   [Extended types in `impl`s](#extended-types-in-impls)
    -   [`where`...`impls`](#whereimpls)
    -   [`extend`](#extend)
    -   [Converting from an extended type facet type to a type facet type](#converting-from-an-extended-type-facet-type-to-a-type-facet-type)
    -   [Any extended type with this type](#any-extended-type-with-this-type)
    -   [`Core.ConvertsInto`](#coreconvertsinto)
    -   [`implicit_into` interface parameter variance](#implicit_into-interface-parameter-variance)
        -   [Implied constraints](#implied-constraints)
        -   [Modifiers for coherence](#modifiers-for-coherence)
        -   [Facets create adapters](#facets-create-adapters)
    -   [`implicit_from` variance](#implicit_from-variance)
        -   [`Core.ConvertsFrom`](#coreconvertsfrom)
        -   [Covariant](#covariant)
    -   [Primitive call interface](#primitive-call-interface)
        -   [`call`](#call)
    -   [Named constraints](#named-constraints)
        -   [Named extended type constraints](#named-extended-type-constraints)
        -   [Associated constants in named constraints](#associated-constants-in-named-constraints)
        -   [`match_first` in named constraints](#match_first-in-named-constraints)
        -   [`Core.Call` named constraint](#corecall-named-constraint)
    -   [Interaction between `anchor`, `exact`, and named constraints](#interaction-between-anchor-exact-and-named-constraints)
    -   [Primitive indexing interface](#primitive-indexing-interface)
    -   [Primitive member binding interface](#primitive-member-binding-interface)
    -   [Primitive conversion interfaces](#primitive-conversion-interfaces)
        -   [with `partial`](#with-partial)
        -   [with `const`](#with-const)
    -   [Primitive versions of other operator interfaces](#primitive-versions-of-other-operator-interfaces)
    -   [`auto` extended type](#auto-extended-type)
        -   [`auto` extended type bindings](#auto-extended-type-bindings)
        -   [Positional parameters](#positional-parameters)
        -   [`alias`](#alias)
        -   [namespace aliases](#namespace-aliases)
        -   [`using`](#using)
        -   [`auto` extended type returns](#auto-extended-type-returns)
    -   [Implicit conversions of `self`](#implicit-conversions-of-self)
-   [Details](#details)
    -   [Coherence modifier examples](#coherence-modifier-examples)
    -   [Examples using `call`](#examples-using-call)
    -   [`impl` lookup for an interface with variant parameters](#impl-lookup-for-an-interface-with-variant-parameters)
    -   [Conversions when calling a function argument](#conversions-when-calling-a-function-argument)
    -   [Deducing an extended type value](#deducing-an-extended-type-value)
    -   [Compile-time parameter equal to a specified value](#compile-time-parameter-equal-to-a-specified-value)
    -   [Parameter requires exact type match](#parameter-requires-exact-type-match)
    -   [Representing deduced parameters](#representing-deduced-parameters)
    -   [Representing explicit compile-time parameters](#representing-explicit-compile-time-parameters)
    -   [Callables that may or may not mutate](#callables-that-may-or-may-not-mutate)
    -   [Forwarding](#forwarding)
        -   [Forwarding tuple parameters](#forwarding-tuple-parameters)
-   [Future work](#future-work)
    -   [Compile-time calls](#compile-time-calls)
    -   [Impl prioritization](#impl-prioritization)
    -   [Parameterized aliases](#parameterized-aliases)
    -   [`ref?`](#ref)
    -   [Allow more variance](#allow-more-variance)
    -   [`extend` in `where`...`impls` expressions](#extend-in-whereimpls-expressions)
-   [Rationale](#rationale)
-   [Alternatives considered](#alternatives-considered)
    -   [Decoupling type from the rest of the extended type](#decoupling-type-from-the-rest-of-the-extended-type)
    -   [Other ways to write "any extended type with a specified type"](#other-ways-to-write-any-extended-type-with-a-specified-type)
    -   [Don't have types that represent all binding information](#dont-have-types-that-represent-all-binding-information)
    -   [Represent all extended type information in ordinary types](#represent-all-extended-type-information-in-ordinary-types)
    -   [Different kinds of extended types for parameters and returns](#different-kinds-of-extended-types-for-parameters-and-returns)
    -   [Implicit conversion from `type` to `Core.ExtType`](#implicit-conversion-from-type-to-coreexttype)
    -   [Other ways to determine variance in interfaces](#other-ways-to-determine-variance-in-interfaces)
    -   [Support implicit conversions in impls](#support-implicit-conversions-in-impls)
    -   [Infer `anchor` in `impl` declarations](#infer-anchor-in-impl-declarations)
    -   [Different extended type literal syntax](#different-extended-type-literal-syntax)
    -   [`ref` on an argument controls its category](#ref-on-an-argument-controls-its-category)
    -   [Replace `require` for interface-interface requirements](#replace-require-for-interface-interface-requirements)
    -   [Modifiers and pattern operators on `fwd` bindings](#modifiers-and-pattern-operators-on-fwd-bindings)
    -   [Separate template and symbolic phases in `Core.ExtType`](#separate-template-and-symbolic-phases-in-coreexttype)
    -   [`alias` of a reference expression](#alias-of-a-reference-expression)
    -   [`fwd` on call arguments](#fwd-on-call-arguments)

<!-- tocstop -->

## Abstract

We propose:

-   Special type `Core.ExtType` whose values describe the extended type
    (category, phase, and so on) of parameters or returns in the signature of
    functions.
-   An `exttype(`...`)` literal syntax for writing values of `Core.ExtType`.
-   Extended type facet types for expressing constraints on extended types,
    which may be composed using `&`.
-   `fwd` binding modifier and return syntax parameterized by an extended type
    value.
-   Extended type versions of interfaces, facets, and facet types.
-   Use these new extended type facet types to create new "primitive" versions
    of the operator interfaces, like `Core.CallPrimitive`, that allow control
    over the extended type of the parameters and return.
-   A `call(`...`)` syntax for concisely and naturally writing facet types of
    the `Core.CallPrimitive` interface.
-   Impl lookup will allow some conversions between `Core.ExtType` values in
    interface arguments and associated constants, when specified by the
    interface, replacing the existing mechanism where `impl`s use `like` to
    allow implicit conversions in arguments.
-   A number of increased capabilities for named constraints.

Together these changes will allow:

-   the operator interfaces to support Carbon's full range of function
    semantics, particularly in the `Core.CallPrimitive` extended type interface;
-   preserving the existing type-oriented constraints like `Core.Call` and
    `Core.AddWith`, now implemented as named constraints in terms of the
    extended-type-oriented primitive interfaces, for common-case uses that don't
    need to deal with extended types;
-   a new, more capable definition of `alias`, along with a shorthand `using`;
-   full-fidelity forwarding of function parameters; and
-   a syntax for a compile-time parameter that is equal to a specified value.

## Problem

The `Core.Call` interface defined in
[proposal #2875](https://github.com/carbon-language/carbon-lang/pull/2875) does
not support all of the features of function calls:

-   `var` and `ref` parameters;
-   mutation of the callable object using `ref self`;
-   compile-time parameters, as designated by `generic`, `template`, or deduced
    `[`...`]` parameter lists; and
-   compile-time parameters that are mentioned elsewhere in the signature.

In addition, we want to be able to express these aspects of parameters in other
contexts:

-   Some interfaces, such as operators, should support `impl`s specifying these
    aspects for the parameters of the interface's member functions. For example,
    a type should be able to implement `Core.AddWith` in a way that uses `var`
    or compile-time parameters.
-   We would like there to be a way to forward arguments passed to one function
    to another function it calls, preserving those aspects. For example, there
    should be a way to write
    [`Apply`](https://en.wikipedia.org/wiki/Function_application) that takes a
    function and arguments to call it with and performs the call (so
    `Apply(F, x)` returns `F(x)`).
-   Some conversions, such as adapter conversions, can return a reference to the
    original object with a new type. Right now, that is only possible for
    built-in conversions, since the conversion interface always returns an
    initializing expression.

Ideally, our call operator interface would support:

-   **Uniformity**: ("consistent and universal") Ordinary functions, lambdas,
    and bound methods are all values of types that implement the call operator
    interface with the relevant signature. This is to enable a single constraint
    on a parameter to accept any of these. This also implies that implementing
    the call operator interface is sufficient to express the full range of
    function signatures, including `var` and `ref` parameters.
-   **Functors**: Types may implement the interface to be callable using the
    normal function-calling syntax, like implementing `operator()` in C++. A
    closure is syntactic sugar for declaring and creating a value of such a
    type.
-   **Overloading**: There is a relatively convenient way to define a function
    value whose type has multiple implementations of the "call operator"
    interface.
-   **Dispatch**: There is a method of matching actual call invocations to
    implementations of the "call operator." It should be as similar as possible
    to other cases of `impl` selection, including supporting the same
    `match_first` `impl` prioritization system (or whatever we replace that
    with). However, it also needs to be able to support the expected implicit
    conversions from the arguments values to the parameter types.
-   **Subsumption**: An overload set containing two functions, one accepting a
    `bool` and one accepting an `i32`, should be able to be passed as an
    argument to a function requiring a callable accepting a `bool`. This same
    mechanism should allow a generic function with signature that could accept a
    `bool`, such as `fn F[T: type](x: T)`, to be passed to the function
    requiring a callable accepting a `bool`, using the instantiation with
    `T == bool`.
-   **Closed**: There is a way to say "just these implementations and no
    others." This should be the default when defining ordinary functions and
    function overloads.

However, fully addressing the last four of these properties is out of scope of
this proposal, since it won't address declaring overloaded functions and `impl`
prioritization. We still would like the decisions in this proposal to be
consistent with a future solution that achieves those properties.

[Leads issue #5028: More powerful aliasing feature](https://github.com/carbon-language/carbon-lang/issues/5028)
documents our desires to improve the shortcomings of our current definition of
the `alias` construct. Leads issues
[#996: Generic let with auto?](https://github.com/carbon-language/carbon-lang/issues/996)
and
[#1371: Is let referentially transparent?](https://github.com/carbon-language/carbon-lang/issues/1371)
considered whether `let` could be combined with `auto` to make something
[referentially transparent](https://en.wikipedia.org/wiki/Referential_transparency),
as you would want from an `alias` construct, and concluded the answer was no. A
solution to the above problems about capturing information beyond type might
also allow us to define semantics for a referentially transparent `alias`
feature without its current shortcomings.

## Background

This proposal builds upon these prior issues and proposals:

-   [Leads issue #578: Value patterns as function parameters](https://github.com/carbon-language/carbon-lang/issues/578)
    decided that we would not have refutable patterns in function signatures,
    and discusses workarounds where this is helpful to work around ordering
    constraints on compile-time parameters.
-   [Proposal #2875: Functions, function types, and function calls](https://github.com/carbon-language/carbon-lang/pull/2875)
    defines the `Call` interface, with its current restriction to value
    parameters and initializing returns.
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

-   [Function overloading document](https://docs.google.com/document/d/1rTsDJ_ZNENH0mhi5E64MGjKJ29OYiI_dzBRz2GzYySU/edit?resourcekey=0-z6_xQ9xuJ5CZ6qJqiuSxuw&tab=t.0)
    with the initial ideas on how to support function overloading in Carbon, and
    links to early discussions on this topic.
-   [Suggestion on leads issue #5261 to support parameterizing bindings with categories](https://github.com/carbon-language/carbon-lang/issues/5261#issuecomment-2790947831).
-   [Idea in open discussion on 2025-05-07 to allow parameterizing interfaces by extended types](https://docs.google.com/document/d/1Yt-i5AmF76LSvD4TrWRIAE_92kii6j5yFiW-S7ahzlg/edit?tab=t.0#heading=h.4zbo49wg5rmk).

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
    such as `Core.NoValExtType`.
-   a `fwd` binding modifier `fwd `_name_`: `_extended type value_, where the
    extended type of the binding is determined by the _extended type value_,
    along with a `fwd` marker on call arguments with generic extended type.
-   a `fwd` return tag `fwd `_extended type value_ (usable anywhere `val`,
    `ref`, or `var` is allowed in a `->` return clause), where the extended type
    of that part of the return is determined by the _extended type value_.
-   extended type interfaces, where the `Self` is of type `Core.ExtType` or an
    extended type facet type instead of `type`.
-   that an `impl` of an extended type interface, or an interface with extended
    type parameters, uses the `.type` property of extended type arguments to
    determine the type to use for the orphan rule.
-   some uses of `extend` that previously assumed `Self` and didn't allow it to
    be specified, now allow `Self` or an extended type using `Self` to be
    specified.
-   `Core.ValSelf`, `Core.RefSelf`, and `Core.VarSelf` functions that take a
    `Core.ExtType` facet type and return an ordinary `type` facet type with the
    specified category used for `Self`.
-   `Core.ConvertsInto(PET: Core.ExtType)` and
    `Core.ConvertsFrom(RET: Core.ExtType)` which are `Core.ExtType` constraints:
    `ConvertsInto(PET)` matches all extended types that can be converted to a
    parameter with extended type `PET`, and `ConvertsFrom(RET)` matches all
    extended types that can be converted from a return with extended type `RET`.
-   `implicit_into` which may be used to mark `Core.ExtType` interface
    parameters, including the `Self` parameter of an extended type interface.
    Those parameters are called "variant", specifically "contravariant." The
    arguments to a query in that position may be converted, and are described by
    the `Core.ConvertsInto` constraint.
    -   An `impl` of such an interface can have multiple type structures,
        depending on whether the types in the query match or need to be
        converted.
    -   An `impl` may use `anchor` or `exact` to limit the conversions allowed,
        which can be needed to satisfy the orphan rule.
    -   Implied constraints are not added for variant parameters.
    -   When normally we would create a facet of the argument type for the
        interface, if that interface has variant parameters, we instead create
        an adapter for the argument type that implements the interface. This
        adapter performs the conversion between the argument and parameter
        extended types, and allows the implementations of the interface to
        differ from the original type without introducing coherence concerns.
-   `implicit_from` which similarly may be used to mark an interface parameter
    or associated constant. These parameters are also "variant," specifically
    "covariant." These allow conversions in the opposite direction as
    `implicit_into`, and are described by the `Core.ConvertsFrom` constraint.
-   updates to named constraints to support extended types.
    -   We add support for associated constants in named constraints. Their
        values must be deducible from an `impl` of an interface required by the
        named constraint.
    -   A named constraint can contain a `match_first` block to prioritize
        between requirements that overlap.
-   a change to the ...`(`...`)` function call operator. It now uses the
    `Core.CallPrimitive` extended type interface, instead of the `Core.Call`
    interface.
    -   A `call` sugar syntax makes it easier to specify a specific
        `Core.CallPrimitive` with a syntax similar to a function declaration
        instead of using extended type literals.
    -   The `Core.Call` interface is changed to a named constraint that uses
        types to specify the call as the interface did before.
-   a change to the ...`[`...`]` indexing operator. It now uses the
    `Core.IndexWithPrimitive` extended type interface instead of a combination
    of `Core.IndexWith` and `Core.IndirectIndexWith` type interface.
    -   `Core.IndirectIndexWith` becomes a named constraint that requires
        `Core.IndexWithPrimitive`.
    -   `Core.IndexWith` becomes a named constraint that requires both
        `Core.IndexValWith` and `Core.IndexRefWith`. `Core.IndexWith` can't be
        implemented directly, implement `Core.IndexValWith` and
        `Core.IndexRefWith` separately instead.
-   a change to the
    [member binding interfaces](/docs/design/expressions/member_access.md#member-binding-interfaces)
    (introduced in
    [proposal #3720](https://github.com/carbon-language/carbon-lang/pull/3720)
    and updated in
    [proposal #7697](https://github.com/carbon-language/carbon-lang/pull/7697)).
    Those type interfaces are replaced with named constraints that
    `extend require` `MemberBind`, a single interface using extended types that
    is used for all cases.
-   extended type interfaces `Core.AsPrimitive` and `Core.ImplicitAsPrimitive`
    that replace type interfaces `Core.As` and `Core.ImplicitAs`. This gives
    `impl`s the ability to control the resulting category. `Core.As` and
    `Core.ImplicitAs` become named constraints requiring the corresponding
    primitive extended type interfaces.
    -   With this greater control, we define how the conversions with `partial`-
        and `const`-qualified types interact with expression category.
-   new `Primitive` extended type interfaces for other operators that allow
    control over extended type, not just type. The previous interface names are
    replaced with named constraints that allow you to continue to use types.
-   `auto` may now be used to deduce an extended type in a `fwd` extended type
    binding or `-> fwd` extended type return.
    -   This is used implicitly for positional parameters.
    -   This is used to give a new definition of the `alias` construct.
    -   This new `alias` construct doesn't apply to namespaces, so a new
        namespace aliasing facility is introduced.

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
We use `val` for the value category in alignment with the using `val` for value
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

```
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

These may be nested, as in

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

We want the unqualified type here, since `.type` will be
[used for the orphan rule](#extended-types-in-impls), where we don't want to
distinguish types with the `const` or `partial` modifiers. See
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

**QUESTION:** An earlier iteration of this proposal used `Core.NoRefExtType` for
[primitive operator interfaces](#primitive-versions-of-other-operator-interfaces).
Now that those uses were changed to `Core.ExtType`, it isn't clear that this
feature is pulling its weight.

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

//  ✅ Valid since `5` can be passed to a
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
function call (such as `f()`). See [the "Forwarding" section](#forwarding) and
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

### Extended type interfaces

Previously, interfaces had an implicit `Self` type parameter, set to the type
that is implementing the interface. We add the capability to explicitly declare
the `Self` parameter with an extended type facet type as the first parameter in
parentheses `(`...`)`, similar to the `self` parameter of methods in
[proposal #7016](p007016-updating-self-syntax-and-adding-static-member-variables.md):

```carbon
interface Callback(Self: Core.ExtType) {
  // Uses `Self` as the extended type of a
  // parameter using `fwd`.
  fn Go(fwd self: Self);
}

interface DefaultConstructible
    (Self: Core.ExtType) {
  // Uses `Self` as the extended type of the
  // return using `fwd`.
  fn MakeDefault() -> fwd Self;
}
```

`Callback` and `DefaultConstructible` are _extended type interfaces_
(specifically `Core.ExtType` interfaces, since `Self` is declared with
`Core.ExtType` rather than a more restricted
[extended type facet type](#extended-type-facet-types)). A value of an extended
type interface is an _extended type facet_. Extended type interfaces and
extended type facets work the same way as ordinary "type" interfaces and type
facets, except `Self` is an extended type value satisfying `Self`'s declared
extended type facet type rather than a `type` value, and an extended type facet
implicitly converts to `Core.ExtType` (and to `Self`'s declared extended type
facet type) rather than to `type`.

An ordinary "type" interface acts as if `Self` is declared `Self: type` as the
first parameter in `(`...`)`, and `Self: type` may also be written explicitly.
Other `type` facet types (such as another interface, named constraint, or
`where` expression) are not allowed as the type of `Self`; express such
requirements with `require` declarations in the body instead (see
[the alternative considered](#replace-require-for-interface-interface-requirements)).

`Self` is bound to the subject of the `impl` or facet type, and is omitted from
the argument list when referencing the interface.

Let's break down what happens with an ordinary "type" interface:

```carbon
// Equivalent to: interface I(Self: type)
interface I {
  fn F(self);
}

// Since `I` is effectively declared `I(Self: type)`, that means
// `T` is a facet of a type, which means a type that is known to
// implement `I`. Since `T` is a facet of a type, it has an implicit
// conversion to `type`, so it can be used to the right of a `:`
// to give a type to `x`.
fn A[T: I](x: T) {
  x.F();
  // - Evaluates `typeof(x).F`, which is `T.F`:
  //   - `T` is a facet of type `I`, so look in `I` for `F`, finding `I.F`
  //     (which has signature `I(Self: type).fn F(self)`)
  //   - Since `I.F` is an associated entity, perform `impl` lookup with
  //     `Self = T as type`
  // - Requires that the type of `x` implements `I`
  //   - Satisfied by the bindings `x: T` and `T: I`, giving `T.impl(I.F)`
  // - Performs instance binding of `x` to that member, then calls it
}
```

Now look at how this works analogously with the `Core.ExtType` interface
`Callback`:

```carbon
// Since `Callback` is declared `Callback(Self: Core.ExtType)`, that means
// `CET` is a facet of a `Core.ExtType`, which means a `Core.ExtType` that is
// known to implement `Callback`. Since `CET` is a facet of a `Core.ExtType`,
// it has an implicit conversion to `Core.ExtType`, so it can be used in a
// `fwd` binding to give an extended type to `c`.
fn InvokeCallback[CET: Callback](fwd c: CET) {
  c.Go();
  // - Looks up `Go` using the extended type of `c`:
  //   - If the extended type is a paren extended type, allow `.0`, `.1` access
  //   - If the extended type is a curly brace extended type, allow `.x`, `.c` access
  //   - If the extended type is primitive, look in its type (as above)
  //   - Since the extended type `CET` is symbolic, look in `CET`'s type
  //     (`Callback`), finding `Callback.Go` (which has signature
  //     `Callback(Self: Core.ExtType).fn Go(fwd self: Self)`)
  //   - Since `Callback.Go` is an associated entity, perform `impl` lookup
  //     using the *extended type* of `c` (`CET as Core.ExtType`) as `Self`
  // - Requires the *extended type* of `c` to implement `Callback`
  //   - Satisfied by the bindings `fwd c: CET` and `CET: Callback`
  // - Performs instance binding of `c` to that member, then calls it
}
```

Other operations such as implementing an extended type interface also work
analogously, using extended type values instead of type values.

```carbon
class V {}
// `Core.ExtType` values implement `Callback`, instead of type
// values, since `Callback` is a `Core.ExtType` interface (and
// a `Core.ExtType` facet type).
impl exttype(val V) as Callback {
  // Notice the extended type of the `self` parameter is
  // `exttype(val V)`, the value of `Self`. This is the
  // same as `fn Go(fwd self: Self)`...
  fn Go(self: V) {
    Core.Print("Printed!");
  }
}

let v: V = {};
// The extended type of the expression `v` is `exttype(val V)`.
// This `Core.ExtType` value implements `Callback` so this
// call typechecks with `CET` deduced to be
// `exttype(val V) as Callback`.
//
// The `c.Go()` call in `InvokeCallback` prints "Printed!".
InvokeCallback(v);
```

With an extended type interface, we can change other aspects of the extended
type beyond the type. Here we show a similar example except with a `ref`
`Core.ExtType`:

```carbon
class R {
  var m: i32;
}
impl exttype(ref R) as Callback {
  // Notice the extended type of the `self` parameter is
  // `exttype(ref R)`, the value of `Self`.
  fn Go(ref self: R) {
    self.m += 2;
  }
}
var r: R = {.m = 3};
// This time `r` is a reference expression so `CET` is
// deduced to be `exttype(ref R) as Callback`.
InvokeCallback(ref r);
Assert(r.m == 5);
```

Finally, here is an example using `DefaultConstructible` instead of `Callback`:

```carbon
// Initializing extended type
impl exttype(var R) as DefaultConstructible {
  fn MakeDefault() -> R {
    return {.m = 7};
  }
}
r = (exttype(var R) as DefaultConstructible)
      .MakeDefault();
Assert(r.m == 7);

var global: R = {.m = 6};
// `ref` extended type
impl exttype(ref R) as DefaultConstructible {
  fn MakeDefault() -> ref R {
    return global;
  }
}
(exttype(ref R) as DefaultConstructible)
      .MakeDefault().m += 3;
Assert(global.m == 9);
```

### Extended types in `impl`s

Interfaces can also use `Core.ExtType` (or an extended type facet type) as the
types of their parameters or associated constants.

```carbon
interface OneArg(P: Core.ExtType) {
  let Result: Core.ExtType;
  fn Fun(self, fwd a: P) -> fwd Result;
}

impl MyType as OneArg(exttype(val i32)) {
  where Result = exttype(var bool);
  fn Fun(self, a: i32) -> bool;
}
```

To satisfy coherence by
[the orphan rule](/docs/design/generics/details.md#orphan-rule), extended types
are converted to types in the type structure using the
[`.type` property](#type-property-of-extended-type-values) (as
[discussed on 2025-06-11](https://docs.google.com/document/d/1Yt-i5AmF76LSvD4TrWRIAE_92kii6j5yFiW-S7ahzlg/edit?tab=t.0#heading=h.rebuyhkgncpq)).
The extended type value itself is used, though, when prioritizing during `impl`
lookup.

### `where`...`impls`

A `where` constraint may now contain a requirement `A impls B` where `A` is an
extended type or facet of an extended type, and `B` is an extended type facet
type. For example, a generic binding
`T: type where exttype(val .Self) impls Callback` will bind to types `T` such
that there is an implementation of `Callback` for `exttype(val T)`.

### `extend`

Prior to this proposal, the `extend` modifier on some declarations means that
some type argument must be omitted, and is presumed to be `Self`, although
there's a syntactic position available for it:

-   In an interface or named constraint, `extend require`, as defined in
    [proposal #5337](https://github.com/carbon-language/carbon-lang/pull/5337),
    must be followed by `impls` instead of allowing a type to be specified.
-   In a class, `extend impl` and `extend final impl` must be followed by `as`
    instead of allowing a type to be specified.
-   Similarly, in an interface, `extend impl` and `extend final impl`, as
    defined in
    [proposal #5337](https://github.com/carbon-language/carbon-lang/pull/5337),
    must be followed by `as` instead of allowing a type to be specified.

In all of these cases, we drop that restriction, and allow a type or an extended
type to be specified in the expected position. These will generally only be
useful if the `Self` type or extended type can implicitly convert to what is
placed in that position, though we allow anything in the spirit of supporting
extension method approach of
[leads issue #1345](https://github.com/carbon-language/carbon-lang/issues/1345)
and
[deferred proposal #1122](https://github.com/carbon-language/carbon-lang/pull/1122).
This doesn't override the need for `Self` to appear in a deducible position for
an `impl` in a generic `class` scope, as described in
[leads issue #5251](https://github.com/carbon-language/carbon-lang/issues/5251#issuecomment-2779776355).

### Converting from an extended type facet type to a type facet type

Given an extended type facet type, we can construct an ordinary `type` facet
type in a few different ways. One is to construct a named constraint. The `Self`
values of the named constraint are types, but the requirements of the constraint
can be on extended type values using the `Self` type.

```carbon
// Equivalent to `constraint ValCallback(Self: type)`
constraint ValCallback {
  extend require exttype(val Self) impls Callback;
}

constraint RefCallback {
  extend require exttype(ref Self) impls Callback;
}

constraint VarCallback {
  extend require exttype(var Self) impls Callback;
}
```

Another way is with [a `where`...`impls` expression](#whereimpls):

-   `type where exttype(val .Self) impls Callback`
-   `type where exttype(ref .Self) impls Callback`
-   `type where exttype(var .Self) impls Callback`

This however doesn't give a way to have name lookup into the facet type find
names in `Callback`, like the named constraints accomplish with
[`extend`](#extend). For that, we introduce functions that take a `Core.ExtType`
facet type and return an ordinary `type` facet type: `Core.ValSelf`,
`Core.RefSelf`, and `Core.VarSelf`. They are used like:

```carbon
fn InvokeValCallback
    [T: Core.ValSelf(Callback)](c: T) {
  // Name lookup into `T` finds names from
  // `Callback`. `T.Go` has signature
  // `fn Go(self: T)`.
  c.Go();
}

fn InvokeRefCallback
    [T: Core.RefSelf(Callback)](ref c: T) {
  // Name lookup into `T` finds names from
  // `Callback`. `T.Go` has signature
  // `fn Go(ref self: T)`.
  c.Go();
}

fn InvokeVarCallback
    [T: Core.VarSelf(Callback) & Core.Move]
    (var c: T) {
  // Name lookup into `T` finds names from
  // `Callback`. `T.Go` has signature
  // `fn Go(var self: T)`.
  // The upcoming move proposal will define
  // how this works, but is expected to make
  // `~c` an initializing expression in this
  // case.
  (~c).Go();
}

fn InvokeWithResultOfFunctionCall
    [T: Core.Call() where
         .Result impls VarSelf(Callback)]
    (f: T) {
  // Not `f().Go()` since `.Result impls`
  // doesn't give `.Result` names.
  f().(VarSelf(Callback).Go)();
}
```

This was discussed on
[2025-05-08](https://docs.google.com/document/d/1Yt-i5AmF76LSvD4TrWRIAE_92kii6j5yFiW-S7ahzlg/edit?tab=t.0#heading=h.vdognq1upsf5);
generalizing this with
[`extend` in `where`...`impls` expressions](#extend-in-whereimpls-expressions)
is deferred to future work.

### Any extended type with this type

To create an [extended type facet type](#extended-type-facet-types) expressing
"any extended type with unqualified type `T`", we use the
[`type` property](#type-property-of-extended-type-values) in a `where` clause,
as in `Core.ExtType where .type = T`.

This approach can also be used to express "any extended type with an unqualified
type satisfying facet type `C`" as in `Core.ExtType where .type impls C`. This
gives the conversion in the other direction from
[the previous section](#converting-from-an-extended-type-facet-type-to-a-type-facet-type).

In both cases, the `Core.ExtType` part can be replaced by another
[extended type facet type](#extended-type-facet-types), in order to compose
extended type constraints.

Since
[extended types in an `impl` use `.type` to get the types for the orphan rule](#extended-types-in-impls),
we end up with a situation where the type to satisfy the orphan rule can come
from the `forall` clause. In this example:

```carbon
impl [PET: Core.ExtType where .type = MyClass] PET as I;
```

The type structure for the orphan rule is `PET.type as I` which, after applying
the rewrite from `PET`'s constraint, becomes `MyClass as I`.

This approach also allows deduction of a type through an extended type, as in:

```carbon
fn F[T: I](generic PET: Core.ExtType where
           .type = Vector(T))
    -> T.Iterator;
```

since the `T` value is uniquely determined by `PET`.

This was
[discussed on 2025-06-11](https://docs.google.com/document/d/1Yt-i5AmF76LSvD4TrWRIAE_92kii6j5yFiW-S7ahzlg/edit?tab=t.0#heading=h.rebuyhkgncpq),
along with alternative ways we could specify this constraint.

### `Core.ConvertsInto`

The `Core` package defines the `ExtType` facet type
`ConvertsInto(PET: ExtType)`, which says which `ExtType`s describe a binding
that would be allowed to convert to the binding described by `PET`. So
`exttype(val i32)` is a value of `ConvertsInto(exttype(val i64))` since there is
an implicit conversion from `i32` to `i64`. More precisely, if we create a
function `F` with `PET` as the extended type of its parameter, and `y` a binding
with extended type `SRC` (also an `ExtType` value), as in

```carbon
fn F(fwd x: PET);
fn G(fwd y: SRC) {
  F(fwd ~y);
}
```

then `SRC` is a value of `ConvertsInto(PET)` if and only if the call
`F(fwd ~y)` is legal.

Note that this depends on the meaning of the planned move operator `~`. In this
case, we need:

-   `~v` when `v` is a value parameter gives you the value of `v` before the
    expression;
-   `~r` when `r` is a `ref` parameter gives you a reference expression that
    references whatever `r` did before the expression;
-   `~i` when `i` is a `var` parameter gives you an initializing expression that
    initializes an object to the state that `i` was in before the expression.

Defining `ConvertsInto` in terms of forwarding a parameter `fwd y: SRC` by way
of `fwd ~y` ensures that the notional `impl`s and
[adapters](#facets-create-adapters) for
[`implicit_into`](#implicit_into-interface-parameter-variance) work by
definition. Alternatively, `ConvertsInto` could be framed in terms of passing an
expression of extended type `SRC` directly to `F` (such as `F(fwd Y())` for
`fn Y() -> fwd SRC;` when `SRC` is a valid extended return type). We expect the
upcoming definition of moves to make these two formulations equivalent, because
input-to-input forwarding `fwd ~y` of a `var` parameter `y` will be able to
reuse `y`'s caller-allocated `var` storage without requiring `Core.Move`.

As an example, `ConvertsInto(exttype(var i64))` includes the facets of
`exttype(val i32)`, `exttype(ref i16)`, and `exttypeof(3)`, since this code is
legal:

```carbon
fn F(var x: i64);

fn G1(y1: i32) {
  F(~y1);
}

fn G2(ref y2: i16) {
  F(~y2);
}

fn G3(fwd y3: exttypeof(3)) {
  F(~y3);
}
```

Also, the facet of `exttype(var i64)` is also a member of
`ConvertsInto(exttype(var i64))`. In fact `PET` is a member of
`ConvertsInto(PET)` for all `PET`.

Note that this is different from the
[extended type facet type conversions](#extended-type-facet-types), which are
about converting between one extended type facet type to another. This is about
what conversions would be allowed to a binding described by a particular value
of `Core.ExtType`.

This is also different from [`ImplicitAs`](#primitive-conversion-interfaces):

-   With `ImplicitAs`, the parameter specifies the destination type, not the
    destination extended type, and the `impl` determines the resulting extended
    type. `ConvertsInto` uses `ImplicitAs` to determine if types convert, but in
    combination with other interfaces, such as `Copy` and `Move`, which
    determine if other aspects of the extended type such as category convert.
    For example, a reference to an object of a derived class implicitly converts
    to a reference to its base class. It additionally `ConvertsInto` a _value_
    of the base class, since we can follow that type conversion with a
    [value acquisition](/docs/design/values.md#value-acquisition).
-   You can't implement `ConvertsInto`, it may only be used as a constraint.

### `implicit_into` interface parameter variance

An extended type interface parameter (including `Self` for an extended type
interface) may be marked with the `implicit_into` keyword binding modifier, as
in this example:

```carbon
interface I(implicit_into Self: Core.ExtType,
            implicit_into P: Core.ExtType) {
  fn F(fwd self: Self, fwd a: P);
}

impl exttype(ref u64) as I(exttype(var i64)) {
  fn F(ref self: u64, var a: i64) {
    self += a;
  }
}
```

This means `impl` lookup for queries involving `I` will consider `impl`
declarations that don't match exactly, unless the corresponding extended type
values in the `impl` declarations have
[a mark that requires them to match, such as `exact`](#modifiers-for-coherence).
In particular, in those positions a query `X` will match `Y` if `X` is a value
of type `ConvertsInto(Y)`. For example, the `impl` defined above acts as if it
was replaced by a collection of `impl`s for purposes of `impl` lookup:

```carbon
// `ForwardTo` is the `F` member of the `impl`
// above. It is equivalent to:
//   fn (ref self: u64, var a: i64) {
//     self += a;
//   }
alias ForwardTo =
  (exttype(ref u64) as I(exttype(var i64))).F;

impl forall [PET1: ConvertsInto(exttype(ref u64)),
             PET2: ConvertsInto(exttype(var i64))]
    PET1 as I(PET2) {
  fn F(fwd self: PET1, fwd a: PET2) {
    // This is allowed, by the definition of
    // `ConvertsInto` and that the extended type of `self`
    // of `ForwardTo` is `ref _: u64` and
    // the extended type of its argument is `var _: i64`.
    (~self).(ForwardTo)(fwd ~a);
  }
}

impl forall [PET1: ConvertsInto(exttype(ref u64))]
   PET1 as I(exact exttype(var i64)) { ... }

impl forall [PET2: ConvertsInto(exttype(var i64))]
   exact exttype(ref u64) as I(PET2) { ... }

impl exact exttype(ref u64)
    as I(exact exttype(var i64)) {
  fn F(ref self: u64, var a: i64) = ForwardTo;
}
```

Note that these `impl` definitions are not created, but `impl` lookup acts as if
they exist. This replaces the `like` mechanism introduced in
[proposal #1144](https://github.com/carbon-language/carbon-lang/pull/1144), that
would add this parameterization to the `impl` itself, rather than making it part
of the `interface`.

Consequences:

-   The type structure of the match for purposes of prioritization counts
    `implicit_into` parameters as a pair of a type and the complete extended
    type, and changes depending on how well the query matches the `impl`.
    -   If the type of the `impl` doesn't match the type in the query, then both
        parts of the pair are considered to be `?`.
    -   If the type matches but rest of the extended type doesn't, the type
        structure is considered to be `(T, ?)`.
    -   If the whole extended type matches, the type structure is considered to
        have a concrete match for both parts of the pair.
    -   This allows a type to implement the same extended type interface for
        multiple categories. For example, if a type implements an interface once
        with a value extended type and once with a reference extended type, a
        value or reference query will choose the `impl` with the matching
        category, while an initializing or constant query will choose the
        value-extended-type `impl` (the only one that applies, since we don't
        want to bind a reference parameter to a temporary). A separate
        `var`-extended-type `impl` can also be provided.
-   As a result, a single `impl` written in the source can correspond to
    multiple type structures. If any of the type structures of two different
    `impl`s are the same, they must be put together in a `match_first` block.
-   These additional `impl`s can only cover cases where we can reliably forward
    a source extended type parameter as an argument to the original function.
    The job of the `ConvertsInto` constraints is to restrict to those cases.
-   The number of notional `impl`s generated triples for every variant parameter
    (minus one for every variant parameter that is marked
    [`anchor`](#modifiers-for-coherence)).
-   An `implicit_into` parameter may only be used in a
    [contravariant position](<https://en.wikipedia.org/wiki/Covariance_and_contravariance_(computer_science)>)
    within the interface definition, as either:

    -   the
        [extended type of an associated function `fwd` parameter](#specifying-the-extended-type-of-a-binding),
        or
    -   the argument to
        [an `implicit_from` interface parameter](#implicit_from-variance) that
        is used as the type of a parameter.

    Note that parameters of a class are always _invariant_, even if that type is
    used as the type of a parameter.

Now consider a generic call to invoke `I.F`:

```carbon
fn CallF[T: Core.RefSelf(I(exttype(val i16)))]
    (ref x: T) {
  x.F(3);
}

var y: u64 = 2;
CallF(y);
```

A few things to note here:

-   The constraint on `T` requires `exttype(ref T)` to implement
    `I(exttype(val i16))`. The `impl` defined above is for
    `exttype(ref u64) as I(exttype(var i64))`, which doesn't match in the
    parameter extended type. However since that parameter is variant, it acts as
    if there is an impl that does satisfy that constraint:

    ```carbon
    impl forall [PET2: ConvertsInto(exttype(var i64))]
        exact exttype(ref u64) as I(PET2) {

      fn F(ref self: u64, fwd a: PET2) {
        // Performs an implicit conversion of `a`
        // to call the original implementation that
        // has one `var _: i64` parameter.
      }
    }
    ```

-   We could not pass in a reference to a `u32` object here, since although a
    `u32` value will implicitly convert to `u64`, a `u32 ref` won't convert to a
    `u64 ref`.
-   Inside of `CallF`, `x.F` takes a `val i16` parameter. The literal `3` gets
    implicitly converted to `i16` to make that call, and then get converted
    again to `i64` to call the original implementation of `I`. See
    [the "Conversions when calling a function argument" section](#conversions-when-calling-a-function-argument).

#### Implied constraints

Consider a class that uses the interface with a variant parameter from before as
a parameter constraint:

```carbon
interface I(implicit_into Self: Core.ExtType,
            implicit_into P: Core.ExtType) {
  fn F(fwd self: Self, fwd a: P);
}

class C(P: I(exttype(val i16))) { ... }

fn G[P: I(exttype(val i32))](c: C(P));
```

Unlike with interfaces with invariant parameters, multiple different `impl`s
could satisfy that constraint. In this case, the `C(P)` parameter type in `G`
would be legal with the existing constraint of `I(exttype(val i32))` on `P`, but
implied constraints would normally add another constraint that
`P impls I(exttype(val i16))`. When
[we considered this on 2025-05-29](https://docs.google.com/document/d/1Yt-i5AmF76LSvD4TrWRIAE_92kii6j5yFiW-S7ahzlg/edit?tab=t.0#heading=h.dj0nqhfqo67w),
we decided we didn't want implied constraints adding constraints when there was
a choice of how to do it, so implied constraint don't apply to interfaces with
variant parameters.

#### Modifiers for coherence

Note that, unless modified by the `impl`, parameters marked `implicit_into`
can't be used to satisfy the orphan rule for coherence. As such we allow these
modifiers on those parameters, which can be used to ensure that there is a local
type in the `impl`'s type structure:

-   `exact`: the type and category must match between the query and `impl`,
    restricting the effect of `implicit_into` to other aspects of extended type
    for this parameter in this `impl`
-   `anchor`: at least one of the parameters marked `anchor` must have a
    matching type between the query and `impl`.

See [examples in the "Details" section](#coherence-modifier-examples).

We also allow using `anchor` to modify `implicit_into` on the _interface_
parameters to only allow conversions of extended type that don't change the
type. This can be useful since otherwise an interface that uses `implicit_into`
on all its parameters always has to use `anchor` when defining `impl`s.

> FIXME: **QUESTION:** Is it `anchor implicit_into` or `implicit_into anchor`?

Either use of `anchor` in effect adds a
[`where .type = T` constraint](#any-extended-type-with-this-type) on the
`ConvertsInto(`...`)` extended type facet type used to allow conversions to use
that `impl`.

Note that the presence or absence of `anchor`, `exact`, `implicit_into`, or
`implicit_from` is not included in `Core.ExtType` values.

> **Alternative considered:** We considered having `exact` mean exact extended
> type match, but this had some concerns such as:
>
> -   We want to allow evolution of what is included in an extended type. For
>     example we might want to include the source location of the argument
>     expression in the extended type.
> -   Additional knowledge of the compile-time value in the query should not
>     prevent matching.
>
> We also considered allowing `exact` to only match on type, excluding category,
> but that didn't match the expected interpretation of the word "exact," where
> it should definitely include anything explicitly written in an extended type
> literal value.

#### Facets create adapters

The facets of an interface without variant parameters preserve the original
type. When forming a facet of an interface with variant parameters, however, we
instead get an adapter type that also implements the interface. In this case,
the implementation of the interface for the adapter type performs the `i16` to
`i64` conversion and then delegates to the original implementation. This change
in type occurs anytime the compiler can't prove it can use the original type
with its original `impl` unchanged, and gives the freedom to have differences
from the original `impl` without violating coherence. A few differences can
occur:

-   Using the original type may mean different implicit conversions occur. In
    our example, the original type's `impl` would have directly converted `3` to
    `i64`, without going through `i16`.
-   As a result of using different implicit conversions, they may accept
    different input types. The original version could be called with `i32`, but
    the version inside `CallF` could be called with a type that converts to
    `i16` but not to `i64`.
-   The original type might have another `impl` of the same interface that could
    apply. The adapted type only has the ones specified by the constraint. Only
    the `impl` used to satisfy the constraint will be used by the adapted type.
-   Use of `anchor` or `exact` on the original `impl` will be ignored, only
    aspects of the `impl` that are explicitly in the constraint are preserved by
    the adapter.

For `template` parameters, if the call can be resolved at definition time using
the adapter type, then when the template is instantiated the call must be
resolved the same way. This follows the general consistency required of template
parameters with their constraints, see for example in
[member access](/docs/design/expressions/member_access.md#lookup-ambiguity).

This was discussed on
[2025-05-29](https://docs.google.com/document/d/1Yt-i5AmF76LSvD4TrWRIAE_92kii6j5yFiW-S7ahzlg/edit?tab=t.0#heading=h.dj0nqhfqo67w).

### `implicit_from` variance

Extended type associated constants and interface parameters of an interface may
be marked `implicit_from`. This allows variance as part of `impl` lookup, but in
the opposite direction from
[`implicit_into` parameters](#implicit_into-interface-parameter-variance). This
example shows an `implicit_from` associated constant:

```carbon
interface Factory {
  let implicit_from Result: Core.ExtType;
  fn Create(self) -> fwd Result;
}

fn UseFactory
    [T: Factory where .Result = exttype(var i64)]
    (x: T) -> i64 {
  return x.Create();
}

class HasFactory {
  impl as Factory {
    where Result = exttype(val i32);
    fn Create(self) -> val i32 {
      return 1;
    }
  }
}

let f: HasFactory = {};
UseFactory(f);
```

Like for `implicit_into` parameters, `implicit_from` associated constants mean
that creating a facet for a type may
[create an adapter type](#facets-create-adapters). In this case though, the
adapter converts between the extended return types. In our example, it is as if
the `UseFactory(f);` call was replaced by:

```carbon
class __HasFactory_adapter {
  adapt HasFactory;
  impl as Factory {
    where Result = exttype(var i64);
    fn Create(self) -> i64 {
      // Does implicit conversion i32 -> i64.
      return (self as HasFactory).Create();
    }
  }
}

UseFactory(f as __HasFactory_adapter);
```

This example shows an `implicit_from` interface parameter, using
[the `Core.CallPrimitive` interface](#primitive-call-interface):

```carbon
interface Registry(implicit_from P: Core.ExtType) {
  fn Register[FN: Core.CallPrimitive(P)](ref self, f: FN);
  fn DefaultArg() -> fwd P;
}
```

#### `Core.ConvertsFrom`

The mismatch between the extended type value in the constraint and `impl` is
allowed when this transformation is valid. In particular, there is a
`Core.ConvertsFrom` constraint that is the mirror of
[`Core.ConvertsInto`](#coreconvertsinto). If we create a function `F` with `RET`
as the extended type of its return, and `G()` a function returning extended type
`DEST` (also an `ExtType` value), as in

```carbon
fn F() -> fwd RET;
fn G() -> fwd DEST {
  return F();
}
```

then `DEST` is a value of `ConvertsFrom(RET)` if and only if `return F();` is
legal.

For example, an `impl` of `Registry`:

```carbon
class R {}
impl R as Registry(exttype(val i32)) {
  fn Register
      [FN: Core.CallPrimitive(exttype(val i32))]
      (ref self, f: FN) {
    f(R.DefaultArg());
  }
  fn DefaultArg() -> val i32 {
    return 7;
  }
}
```

is as if it was written:

```carbon
class R {}
impl forall [RET: Core.ConvertsFrom(exttype(val i32))]
    R as Registry(RET) { ... }
```

This means that since `i32` implicitly converts to `i64`,
`R impls Registry(exttype(val i64))` as well, and the following use is allowed:

```carbon
fn TakesI64(x: i64);

// In the `DoRegister(r)` call below, `T`
// is an adapter for `R` that adds the
// `i32` -> `i64` conversion.
fn DoRegister[T: Registry(exttype(val i64))]
    (ref r: T) {
  // Calls `TakesI64` with `7 as i32`
  // which gets converted to an `i64`.
  r.Register(TakesI64);
}

var r: R;
DoRegister(r);
```

#### Covariant

An `implicit_from` interface parameter or associated constant may only be used
in a
[covariant position](<https://en.wikipedia.org/wiki/Covariance_and_contravariance_(computer_science)>)
within an interface definition, either:

-   [the extended type of an associated function `fwd` return](#specifying-the-extended-type-of-a-return),
    as in `Factory.Create` and `Registry.DefaultArg`
    [earlier](#implicit_from-variance), or
-   the argument to
    [an `implicit_into` interface parameter](#implicit_into-interface-parameter-variance)
    that is used as the type of a parameter, as in `Registry.Register`
    [earlier](#implicit_from-variance).

### Primitive call interface

Combining extended types with variadics allows us to represent a more general
version of the `Call` interface that allows specifying extended types:

```carbon
package Core;

interface CallPrimitive(
    implicit_into anchor Self: ExtType,
    ... implicit_into each Param: ExtType) {
  let implicit_from ResultExtType: ExtType;
  fn Op(fwd self: Self, ... fwd each args: each Param)
      -> fwd ResultExtType;
}
```

Note that the `Self` is variant, since we want to allow a `val T` `impl` to be
used for a `ref T` object. The variance is using
[`anchor` modifier on `implicit_into`](#modifiers-for-coherence) so the _type_
of `Self` is not variant (though see
[the "Implicit conversions of `self`" section](#implicit-conversions-of-self)),
just the other aspects of the extended type.

#### `call`

It can be cumbersome to write out extended type values for the arguments to this
interface, though, and we expect this to be the most common use case that would
otherwise require writing extended type values. So we introduce a `call(`...`)`
convenience syntax for writing a facet type of the `Core.CallPrimitive`
interface.

| `call` syntax           | `Core.ExtType` facet type value                              |
| :---------------------- | :----------------------------------------------------------- |
| `call()`                | `CallPrimitive()`                                            |
| `call(_: i32)`          | `CallPrimitive(exttype(val i32))`                            |
| `call(_: i32, _: bool)` | `CallPrimitive(exttype(val i32), exttype(val bool))`         |
| `call(var _: bool)`     | `CallPrimitive(exttype(var bool))`                           |
| `call(fwd _: X)`        | `CallPrimitive(X)`                                           |
| `call(... _: each T)`   | `CallPrimitive(... exttype(val each T))`                     |
| `call() -> i32`         | `CallPrimitive() where .ResultExtType = exttype(var i32)`    |
| `call() -> val i32`     | `CallPrimitive() where .ResultExtType = exttype(val i32)`    |
| `call() -> ()`          | `CallPrimitive() where .ResultExtType = exttype(var ())`     |
| `call() -> fwd R`       | `CallPrimitive() where .ResultExtType = R`                   |

Note: If "`->` _\<return-type\>_" is not included at the end, the return type is
unconstrained. Use `-> ()` to mean "returns nothing."

Note: In `call(`...`)`, `var` functions as a modifier on a binding rather than
an independent pattern operator, so `call(var (_: bool, _: i32))` is not valid,
that signature would be written `call(var _: (bool, i32))` instead.

Observe that `CallPrimitive` is a `Core.ExtType` interface, so the above
expressions represent `Core.ExtType` facet types. In addition, you may specify
the non-type aspects of the extended type of `self` by writing `[`...`]` between
`call` and the `(` of the parameter list to produce an ordinary `type` facet
type, as in:

| `call` syntax | facet type value           |
| :------------ | :------------------------- |
| `call[val]()` | `ValSelf(CallPrimitive())` |
| `call[ref]()` | `RefSelf(CallPrimitive())` |
| `call[var]()` | `VarSelf(CallPrimitive())` |

These can be combined, so `call[ref](var _: bool, _: i32*) -> ref i32` gives the
facet type:

```carbon
RefSelf(
  CallPrimitive(exttype(var bool),
                exttype(val i32*))
      where .ResultExtType = exttype(ref i32))
```

See [examples in the "details" section](#examples-using-call).

This was discussed on
[2025-05-08](https://docs.google.com/document/d/1Yt-i5AmF76LSvD4TrWRIAE_92kii6j5yFiW-S7ahzlg/edit?tab=t.0#heading=h.vdognq1upsf5).

### Named constraints

We want named constraints to provide
[progressive disclosure](/docs/project/principles/progressive_disclosure.md) for
extended types, providing alternative ways of accessing functionality that uses
extended types at the lowest level, but can use a named constraint with types
instead in common scenarios. This is similar to how we use named constraints in
other circumstances to hide complexity, such as providing `Add` as an easier
alternative to `AddWith` when you don't need the additional flexibility.

#### Named extended type constraints

We allow an optional `Self:`... declaration as the first parameter in `(`...`)`
on named constraints, with the same allowed types for `Self` as on
[extended type interfaces](#extended-type-interfaces) (including explicit
`Self: type`, the default when `Self` is omitted). A `require impls` or
`require Self impls` in a constraint is allowed when the type of `Self` matches
the type of the `Self` of the interface or constraint being required.

```carbon
constraint DefaultConstructibleCallback
    (Self: Core.ExtType) {
  // `Self` is `Core.ExtType`, matches the `Self`
  // of `Callback` and `DefaultConstructible`.
  extend require impls Callback;
  extend require impls DefaultConstructible;
}
```

#### Associated constants in named constraints

We add support for associated constant declarations in named constraints, using
the same `let`...`:`...`;` syntax as in interfaces. Such associated constants
must be used in such a way as to be deducible from the `impl`s of interfaces
required by the constraint. Deducible here means following the same rules as a
deduced parameter in a function signature. When seeing if a facet satisfies a
constraint, the associated constants are deduced.

```carbon
constraint Invalid {
  // ❌ Invalid: `Key` not deducible.
  let Key: type;
  extend require impls Map(Key);
}

constraint MultiMap(Key: type) {
  // Shadows `Value` from `Map`.
  let Value: type;
  extend require impls Map(Key)
      where .Value = Vector(Value);
}
// This is legal since a function like
//   fn F[Value: type]
//     (generic T: Map(Key) where .Value = Vector(Value));
// is valid, since `Value` is deducible from `T`.
```

When implementing the named constraint directly, every associated constant of
the extended interface must be given a value, as before. This can now be done by
given a value to an associated constant in the named constraint, which will give
values to any derived associated constants in the interface being implemented.

Example using `MultiMap`:

```carbon
class S2I { ... }

impl S2I as Map(String)
  where .Value = Vector(i32) { ... }

fn MultiGet[T: MultiMap(String)](x: T)
    -> T.Value;
fn MapGet[T: Map(String)](x: T)
    -> T.Value;
fn Restricted(
    generic V: type,
    generic T: MultiMap(String) where .Value = V);

var s2i: S2I = {...}
// `(S2I as Map(String)).Value` is
// `Vector(i32)`, so
// `(S2I as MultiMap(String)).Value`
// is `i32`.
let r1: i32 = MultiGet(s2i);
let r2: Vector(i32) = MapGet(s2i);
// ✅ Allowed
Restricted(i32, S2I);
// ❌ Invalid:
// `(S2I as MultiMap(String)).Value`
// is `i32` not `bool`
Restricted(bool, S2I);

class S2B { ... }

// `(S2B as MultiMap(String)).Value`
// is `bool`, so
// `(S2B as Map(String)).Value` is
// `Vector(bool)`.
impl S2B as MultiMap(String)
  where .Value = bool { ... }

var s2b: S2B = {...}
let r3: bool = MultiGet(s2b);
let r4: Vector(bool) = MapGet(s2b);
// ❌ Invalid:
// `(S2B as MultiMap(String)).Value`
// is `bool` not `i32`
Restricted(i32, S2B);
// ✅ Allowed
Restricted(bool, S2B);
```

Whenever a named constraint is implemented, the resulting `impl` is checked to
verify that its satisfies the named constraint, including giving a single
consistent value to any associated constants it might have.

```carbon
interface X {
  let T: type;
  let U: type;
}

constraint N {
  let V: type;
  extend require impls X
      where .T = Vector(V)
        and .U = Map(V);
}

class Invalid1 {}

impl Invalid1 as N {
  where T = Vector(bool);
  // ❌ Inconsistent, `V` can't be both
  // `bool` and `i32`.
  where U = Map(i32);
}

class Invalid2 {}

impl Invalid2 as N {
  // ❌ Inconsistent, `T` has to be a
  // `Vector` so there is some legal
  // value for `V`.
  where T = i32;
  where U = Map(i32);
}
```

This feature can be used to convert an extended type associated constant in the
extended interface into a type associated constant in the named constraint, as
in [`Core.Call`](#corecall-named-constraint) and
[other](#primitive-versions-of-other-operator-interfaces) named constraints.

The `final` modifier is allowed on associated constants in named constraints. In
this case, it is given a value directly instead of being deduced from the
`impl`s of required interfaces. Examples:

```carbon
constraint VectorI(T: type) {
  // Can be used to give a name to an
  // an expression.
  final let Elt: type = Vector(T);

  require impls I where .Elt = Elt;

  // Can be used to create an alias to
  // a member of a required interface.
  final let template Iter: auto
      = I.Iter;
}
```

Allowing the `default` modifier here is future work, which we expect to support
once we have an algorithm for resolving when there are multiple applicable
defaults (likely using the defaults from the outermost constraint first). Until
then, note that the interface can still have a default that can be applicable to
a named constraint default, as in:

```carbon
interface DefaultResult {
  default let ResultExtType: Core.ExtType
      = exttype(var Self);
  fn F() -> fwd ResultExtType;
}
constraint C {
  let Result: type;
  extend require impls DefaultResult
      where .ResultExtType = exttype(var Result);
}
```

Here an `impl` of `C` that doesn't specify `Result` will use the default
`DefaultResult.ResultExtType` of `exttype(var Self)`, which will deduce a value
of `Self` for `C.Result`.

We considered similar constructs (`forunique` or `forsome`) in the context of
template constraints in
[discussion on 2023-01-23](https://docs.google.com/document/d/1gnJBTfY81fZYvI_QXjwKk1uQHYBNHGqRLI2BS_cYYNQ/edit?resourcekey=0-ql1Q1WvTcDvhycf8LbA9DQ&tab=t.0#heading=h.klm1hw3hfm1n)
and
[as part of the design of supporting `for` loops with C++ interop](https://docs.google.com/document/d/157k0s1e6qgTKUtads4x7BRtLHq8Xesh6DUHtbQ01gjY/edit?tab=t.0#heading=h.bs060kxctkgo).

#### `match_first` in named constraints

Consider a situation where an interface with variant parameters is used twice in
the constraint of a generic parameter, as in this example:

```carbon
fn H[T: call(_: i32) & call(_: i64)]
    (fwd g: T, x: i16) {
  // ❌ Ambiguous whether it uses
  // `call(_: i32)` or `call(_:i64)`.
  g(x);
}
```

Due to the allowed implicit conversions, it can be ambiguous which
implementation to use. In this example, the different implementations of `call`
could have different return types. Without a way to choose between them, the
compiler is forced to issue a diagnostic that the call is ambiguous.

To resolve this ambiguity, we need additional information from the developer in
the constraint. Due to the difficulty of expressing this inline in a constraint
and a desire to avoid a second syntax for expressing the prioritization between
`impl`s, we've decided to express this out-of-line in a named constraint using
the same `match_first` syntax used to prioritize `impl`s outside this context.

```carbon
constraint X(Self: Core.ExtType) {
  match_first {
    require Self impls call(_: i32);
    require Self impls call(_: i64);
  }
}

fn J[T: X](fwd g: T, x: i16) {
  // ✅ Uses `call(_: i32)`.
  g(x);
}
```

This was discussed on
[2025-05-29](https://docs.google.com/document/d/1Yt-i5AmF76LSvD4TrWRIAE_92kii6j5yFiW-S7ahzlg/edit?pli=1&tab=t.0#heading=h.dj0nqhfqo67w).

#### `Core.Call` named constraint

We can use these new features to define `Core.Call` as a named constraint that
acts like the interface before this proposal, for use cases where the extra
control over extended types is not needed.

```carbon
package Core;

constraint Call(... each Param: type) {
  let Result: type;
  extend require impls
      call[val](... _: each Param) -> Result;
  // Equivalent to:
  extend require impls
      ValSelf(
        CallPrimitive(... each exttype(val Param)))
      where .ResultExtType = exttype(var Result);
}
```

> FIXME: **QUESTION:** Is this redundant with providing `call`? Should instead
> this be dropped and `CallPrimitive` be renamed `Call`? Or do we want a
> non-primitive version to parallel other operator interfaces?

### Interaction between `anchor`, `exact`, and named constraints

Without named constraints, we would only use `anchor` and `exact` in impls and
only in the position of a variant parameter to the interface. With named
constraints, we could want to use them in two other cases:

-   A type parameter in a named constraint may end up being a variant extended
    type argument to the underlying interface. For example, this would happen
    with the named constraint `AddWith`, which delegates to `AddWithPrimitive`.
    To support this, we might need to allow types to be marked with `anchor`.
    Further, Carbon would have to propagate whether an expression uses `anchor`
    as extra-type information that isn't used by type checking except if the
    expression ends up being the variant argument to an interface. We might also
    want to diagnose using `anchor` on a type in a way that doesn't end up being
    used as a variant argument, but that seems hard to determine.
-   If you have a constraint like `Add` which takes only the `Self` parameter
    and uses it to set all of the variant interface parameters, it is better to
    put `anchor` in `Add` than in all impls of `Add`. We could support this by
    allowing `anchor` or `exact` before interface arguments in
    `extend require impls` declarations in a named constraint. This would be
    ignored when using it as a constraint, but are used in an `impl` of the
    named constraint.

> FIXME: **QUESTION:** Is there a way to avoid some of this complexity?

This was discussed on
[2025-06-13](https://docs.google.com/document/d/1Yt-i5AmF76LSvD4TrWRIAE_92kii6j5yFiW-S7ahzlg/edit?tab=t.0#heading=h.52ru7ner80b4).

### Primitive indexing interface

Proposal
[#2274: Subscript syntax and semantics](https://github.com/carbon-language/carbon-lang/pull/2274)
introduced the `IndexWith` and `IndirectIndexWith` interfaces. The rewrites from
a `lhs[index]` expression into invoking methods of interfaces depended on both
the category of the `lhs` part of the expression, and which interfaces were
implemented. This was complex due to lacking the tools for dispatching on
category provided by this proposal, and resulted in undesirable dependence on
the information that a particular interface was not implemented. Which meant
that the rewrite could change if another constraint was added. We compensated
with a `final impl` to make the two rewrites have type consistency. With the
tools of this proposal, however, we can take a simpler approach of having a
single interface, `IndexWithPrimitive`, that uses extended types, and two
different use cases expressed using named constraints:

-   The `IndirectIndexWith` constraint has a single requirement: given a value
    `Self` the indexing operator produces a reference result.
-   The `IndexWith` constraint requires that both the `IndexValWith` and
    `IndexRefWith` constraints be implemented. The `IndexValWith` constraint
    provides a value result from indexing given a value `Self`, and the
    `IndexRefWith` constraint provides a reference result given a reference
    `Self`. In effect, using two implementations to provide
    [`ref?` semantics](#ref). `IndexValWith` and `IndexRefWith` use
    [`alias`es](#alias) to give different names (`At` and `Ref`) to the `Op`
    functions of `IndexWithPrimitive` with different parameters. During `impl`
    lookup, the `IndexRefWith` version will be preferred for reference
    expression due to the rules for
    [`implicit_into` interface parameter variance](#implicit_into-interface-parameter-variance).
    Note that `IndexWith` may be used as parameter's constraint, but can't be
    directly implemented, since we don't support implementing multiple
    interfaces together (see
    [leads issue #4566: Implementing multiple interfaces with a single `impl` definition](https://github.com/carbon-language/carbon-lang/issues/4566)
    and
    [proposal #5168: Forward `impl` declaration of an incomplete interface](https://github.com/carbon-language/carbon-lang/pull/5168)).
    This gap in the design affects this operator, but will need to be addressed
    separately as future work, possibly by either changing the rules for
    implementing multiple interfaces or by using mixins to address this use
    case. This was
    [discussed on 2025-06-13](https://docs.google.com/document/d/1Yt-i5AmF76LSvD4TrWRIAE_92kii6j5yFiW-S7ahzlg/edit?tab=t.0#heading=h.52ru7ner80b4).

Observe that a type that defines an `impl` of `IndirectIndexWith` will also
satisfy the `IndexWith` constraint:

-   The "value `Self` returning a value" requirement can be satisfied due to the
    [`implicit_from` associated constant variance](#implicit_from-variance)
    allowing the reference return to be converted to a value return.
-   The "reference `Self` returning a reference" requirement takes advantage of
    [variance](#implicit_into-interface-parameter-variance) to support
    converting the reference `Self` to the value expected by the `impl`.

To support coherence, `IndexWithPrimitive` uses `implicit_into anchor` on the
`Self` parameter, since that is the parameter expected to be used to find the
`impl`.

See [Indexing](/docs/design/expressions/indexing.md) for details.

Switching the indexing interface to extended types was
[discussed on 2025-06-10](https://docs.google.com/document/d/1Yt-i5AmF76LSvD4TrWRIAE_92kii6j5yFiW-S7ahzlg/edit?tab=t.0#heading=h.stp5dpvithkz).

### Primitive member binding interface

[Member binding](/docs/design/expressions/member_access.md#member-binding-interfaces)
(introduced in
[proposal #3720](https://github.com/carbon-language/carbon-lang/pull/3720) and
updated in
[proposal #7697](https://github.com/carbon-language/carbon-lang/pull/7697)) also
uses multiple interfaces (`Bind`, `BindToValue`, and `BindToRef`) in order to
handle different expression categories. We can now redefine them as named
constraints, in terms of a general `MemberBind` interface:

```carbon
interface MemberBind(implicit_into LHS: Core.ExtType) {
  let implicit_from ResultExtType: Core.ExtType;
  fn Op(self, fwd lhs: LHS) -> fwd ResultExtType;
}

// Formerly BindToValue
constraint BindToVal(T: type) {
  let Result: type;
  extend require impls
      MemberBind(exttype(val T))
      where .ResultExtType = exttype(val Result);
}

constraint BindToRef(T: type) {
  let Result: type;
  extend require impls
      MemberBind(exttype(ref T))
      where .ResultExtType = exttype(ref Result);
}
```

Observe that this replicates the semantics of `BindToValue` and `BindToRef`: in
particular, the
[`implicit_into` variance](#implicit_into-interface-parameter-variance) of the
`LHS` parameter supports
[implicit conversions](/docs/design/expressions/member_access.md#inheritance-and-other-implicit-conversions).

However, there are a few differences:

-   There is no longer a `Bind` interface used to guarantee that the result type
    of `BindToVal` and `BindToRef` would be the same.
-   `BindToVal` will be used for reference expressions if `BindToRef` is not
    implemented, avoiding the need to routinely implement both.

In the future, `MemberBind` could be made into an extended type interface, but
currently there is no motivation to use anything other than the type of `Self`
for `impl` selection.

### Primitive conversion interfaces

We also replace the conversion interfaces, `As` and `ImplicitAs` with extended
type interfaces `AsPrimitive` and `ImplicitAsPrimitive`. These continue to each
have a parameter that specifies destination _type_, but an associated constant
(set by the `impl`) determines the destination _extended type_. The destination
extended type is constrained to
[have type matching](#any-extended-type-with-this-type) the parameter. We also
define named constraints `As` and `ImplicitAs` that operate on types, as before,
for
[progressive disclosure](/docs/project/principles/progressive_disclosure.md).
See [`as` expressions](/docs/design/expressions/as_expressions.md#extensibility)
and
[implicit conversions](/docs/design/expressions/implicit_conversions.md#extensibility).

These new capabilities allow us to support conversions that preserve category,
without workarounds unavailable to libraries, supporting our principle that
["All APIs are library APIs"](/docs/project/principles/library_apis_only.md).

-   Adapter types automatically implement `AsPrimitive`, which allows those
    conversions to preserve category.
-   Derived-to-base implicit conversions use `ImplicitAsPrimitive` to preserve
    category.

This was
[discussed on 2025-06-11](https://docs.google.com/document/d/1Yt-i5AmF76LSvD4TrWRIAE_92kii6j5yFiW-S7ahzlg/edit?tab=t.0#heading=h.rebuyhkgncpq).

#### with `partial`

Using category allows both preserving category and limiting to those cases that
are safe when casting between a `partial` class type, and the whole class type:

-   Implicit conversion `C` \-\> `partial C` is allowed for any category, and
    preserves category, so `exttype(val C)`, `exttype(ref C)`, and
    `exttype(var C)` all implement `ImplicitAs(partial C)`, with a corresponding
    `.ResultExtType`:
    -   For `exttype(val C)`, `.ResultExtType = exttype(val partial C)`.
    -   For `exttype(ref C)`, `.ResultExtType = exttype(ref partial C)`.
    -   For `exttype(var C)`, `.ResultExtType = exttype(var partial C)`.
-   Implicit conversion `partial C` \-\> `C` is only allowed only when the
    source is an initializing expression. A `partial C` initializing expression
    always has dynamic type `C`. With other expression categories, it could be a
    derived type.

    -   `exttype(var partial C)` `impls`
        `ImplicitAs(C) where .ResultExtType = exttype(var C)`

    ```
    fn F() -> partial C;
    fn G(c: C);
    // ✅ Allowed
    G(F());
    ```

Other result categories are possible by doing a category conversion after the
implicit type conversion.

This was discussed on
[2025-06-24](https://docs.google.com/document/d/1Yt-i5AmF76LSvD4TrWRIAE_92kii6j5yFiW-S7ahzlg/edit?tab=t.0#heading=h.d985djkv6qti).

#### with `const`

Similarly with `const`, we can capture the fact that value expressions are
immutable, and therefore imply `const`.

-   Conversion `C` \-\> `const C` is allowed for any category and preserves
    category. This is since we are using a meaning of `const` like C++ `const`
    which means "not changed through this reference" rather than "reference to
    something immutable."
-   Conversion `const C` \-\> `C` is allowed for values.
    -   `exttype(val const C)` `impls`
        `ImplicitAs(C) where .ResultExtType = exttype(val C)`
    -   A value with `exttype(ref const C)` will be able to use that conversion
        due to
        [`implicit_into` variance](#implicit_into-interface-parameter-variance),
        requiring a
        [value acquisition](/docs/design/values.md#value-acquisition).

This was discussed on
[2025-06-24](https://docs.google.com/document/d/1Yt-i5AmF76LSvD4TrWRIAE_92kii6j5yFiW-S7ahzlg/edit?tab=t.0#heading=h.d985djkv6qti).

### Primitive versions of other operator interfaces

In general, operator interfaces from before this proposal are replaced with a
named constraint to a new interface with a `Primitive` suffix. The new interface
uses `Core.ExtType` or [extended type facet types](#extended-type-facet-types),
where appropriate for parameters and associated constants describing returns.
Operators are rewritten to these new "Primitive" interfaces instead.

For example, the expression `a + b` used to be rewritten to
`a.(AddWith(typeof(b)).Op)(b)`, calling a member of the `AddWith` interface. It
is now rewritten to `a.(AddWithPrimitive(exttypeof(b)).Op)(b)`, where
`AddWithPrimitive` is defined like so:

```carbon
package Core;

interface AddWithPrimitive(
    implicit_into Self: ExtType,
    implicit_into P: ExtType) {
  let implicit_from ResultExtType: ExtType;
  fn Op(fwd self: Self, fwd rhs: P) -> fwd ResultExtType;
}
```

This "Primitive" interface has some changes:

-   Uses `Core.ExtType` instead of `type`.
-   The parameters are marked
    [`implicit_into`](#implicit_into-interface-parameter-variance). In some
    cases the left parameter will also be marked
    [`anchor`](#modifiers-for-coherence).
-   The associated constant controlling the return is marked
    [`implicit_from`](#implicit_from-variance).
-   Uses [`fwd` bindings](#specifying-the-extended-type-of-a-binding) and
    [`fwd` returns](#specifying-the-extended-type-of-a-return).

For
[progressive disclosure](/docs/project/principles/progressive_disclosure.md), we
continue to provide `Core.AddWith` that uses types instead of extended types,
but it is now a named constraint:

```carbon
constraint AddWith(T: type) {
  let Result: type;

  extend require impls ValSelf(
      AddWithPrimitive(exttype(val T))
          where .ResultExtType = exttype(var Result));
}
```

The following interfaces are updated, with the changes applied to
[`/docs/design/`](/docs/design/):

-   In [Arithmetic](/docs/design/expressions/arithmetic.md#extensibility):
    `Negate`, `AddWith`, `SubWith`, `MulWith`, `DivWith`, `ModWith`.
-   In [Assignment](/docs/design/assignment.md#extensibility): `AssignWith`,
    `AddAssignWith`, `SubAssignWith`, `MulAssignWith`, `DivAssignWith`,
    `ModAssignWith`, `BitAndAssignWith`, `BitOrAssignWith`, `BitXorAssignWith`,
    `LeftShiftAssignWith`, `RightShiftAssignWith`. These use `ref` for the left
    argument, the extended type does not vary. `Inc` and `Dec` are unchanged.
-   In
    [Bitwise and shift operators](/docs/design/expressions/bitwise.md#extensibility):
    `BitComplement`, `BitAndWith`, `BitOrWith`, `BitXorWith`, `LeftShiftWith`,
    `RightShiftWith`.
-   In
    [Comparison operators](/docs/design/expressions/comparison_operators.md#extensibility):
    `EqWith`, `OrderedWith`.

Note that the common type interfaces are unchanged.

> **Alternative considered:** We originally had restrictions that the extended
> type parameters to primitive interfaces not be references. This avoided
> questions about whether the caller would have to write `ref` before the
> argument in that case, as in `(ref x) + (ref y)` for the addition operator. It
> was also intended to avoid having the possibility of surprising mutating side
> effects. However, this would cause an interop problem since C++ operators can
> take reference parameters. This comes up in practice with the `<<` operator
> when applied to streams, as in:
>
> ```cpp
> std::ostream& operator<<(std::ostream& lhs, Foo rhs);
> ```
>
> This was discussed in #interop on
> [2026-01-20](https://discord.com/channels/655572317891461132/768530752592805919/1463317216739983505)
> and
> [-23](https://discord.com/channels/655572317891461132/768530752592805919/1464391757038096489).

> **Decision:** Arguments to operators that are not always references but select
> an `impl` using a `ref` extended type must be marked with `ref` at the
> callsite, as in `(ref out_stream) << x;`. This was discussed in
> [#interop on 2026-01-23](https://discord.com/channels/655572317891461132/768530752592805919/1464427960026005648).
> Similarly, operator operands with generic extended type are marked with `fwd`,
> as in `(fwd self) == (fwd u)`.

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

```
var x: i32 = 1;
let IncBy: auto = fn { $0 += $1; };
IncBy(ref x, 2);
Assert(x == 3);

let Add: auto = fn => $0 + $1;
// Will deduce a value extended type since
// not marked `ref`:
Assert(Add(x, 3) == 6);
```

#### `alias`

Prior to this proposal, `alias` is a construct we have been assuming would
operate purely in the domain of names, but has only been specified in a few
contexts. This approach has limitations that we need to address, as described in
[leads issue #5028: "More powerful aliasing feature"](https://github.com/carbon-language/carbon-lang/issues/5028).
The highest priority here is being able to give a name to a member of a
specific. For example, we've frequently assumed that declarations like
`alias String = BasicString(u8);`, `alias T = i32;` (equivalent to
`alias T = Core.Int(32);`), and `alias X = C(i32).X;` would somehow be allowed.
So we give `alias` a definition in terms of `let` bindings instead of operating
on names:

> `alias A = B;` converts `B` to a value `V` (which must be a compile-time
> expression), then means `let fwd A: auto = V;`

```carbon
alias a = 7;
alias String = BasicString(u8);
```

Note that this is more referentially transparent than `let a: auto =`... since
it preserves phase in addition to type. Unlike `let generic a: auto =`... it
doesn't erase the value of the right-hand side. Unlike
`let template a: auto =`... it doesn't change the phase to "template."

```carbon
fn F(generic T: type, template X: i32) {
  // `a` is symbolic, but not the same as `let generic`
  // since the fact that it is `buf(T)` is preserved.
  alias a = buf(T);
  // `b` has template phase.
  alias b = X;
}
```

In order to support renaming a variable, we allow an additional form:

> `alias ref A = R;` means `let ref A: auto = R;` (equivalently,
> `let fwd A: auto = R;`)

where `R` is a reference expression with address known at compile time (such as
the name of a variable). This syntax was decided in
[open discussion on 2026-11-18](https://docs.google.com/document/d/1mjllGO3ZCL4qGt9uJHUtcxKoHAGEY7Y999ie4EtBWB8/edit?tab=t.uvyw2c4zrz9e#heading=h.edvediz60r3d)
(see [the alternatives considered](#alias-of-a-reference-expression)). All other
expressions (including non-reference runtime expressions and expressions with
mixed compile-time and runtime phase) are forbidden. For example:

```carbon
var v: i32 = 7;
// ❌ Not a compile-time constant, need
// `ref` to alias a reference expression.
alias v_error = v;

// ✅ Allowed, reference expression with
// known address.
alias ref va = v;

var arr: array(i32, 64);
// ❌ Invalid with or without `ref`, since
// address unknown at compile time.
alias ref aa = arr[v];

// ❌ Invalid with or without `ref`, since
// non-reference runtime expression.
alias p_err1 = v + 2;
alias ref p_err2 = v + 2;
```

We could support more runtime expressions if there is a demonstrated need as
future work, perhaps by capturing the right-hand side as a lambda. For now we
don't want to support that since it introduces ambiguity concerns around when
the lambda is evaluated.

An `alias` declaration is allowed in these contexts:

-   at file scope or in the body of a function;
-   in the body of a `class`, but only to define a compile-time constant;
-   in the body of an `impl` in order to specify that an associated function is
    to be implemented by an existing function; and
-   in the body of an interface or named constraint, where the rewrite to `let`
    is additionally given the `final` modifier.

An `alias` may have an access modifier such as `private`, when allowed in that
scope.

This definition of `alias` was discussed on
[2025-06-03](https://docs.google.com/document/d/1Yt-i5AmF76LSvD4TrWRIAE_92kii6j5yFiW-S7ahzlg/edit?tab=t.0#heading=h.hdj6hk4bwd2z).
This approach has the benefit of defining `alias` in terms of other Carbon
features, reducing the need to specify its semantics separately and removing a
situation where there were two ways to do something that were slightly
different.

Support for parameterized aliases is [future work](#parameterized-aliases).

> **Note:** This definition of `alias` is partially implemented in
> [PR #7190: Relax alias restrictions](https://github.com/carbon-language/carbon-lang/pull/7190).

#### namespace aliases

However, this definition of `alias` only works with first-class values. The good
news here is that almost everything we want to alias is first class,
particularly with
[proposal #3720](https://github.com/carbon-language/carbon-lang/pull/3720) which
made more entities like class and interface members first class. The main
exception is that we have no intention of making namespaces first class, because
the set of members is incoherent (different between libraries depending on what
they import) and in fact changes as new entities are declared in the namespace.
The solution is to, like C++, use another mechanism to alias namespaces. To this
end, we say that a `namespace` declaration may now optionally declare a
namespace alias by using a `=`:

```carbon
// Previous syntax: declare a namespace
// `A.B`:
namespace A.B;

// New syntax: declare `C` as an alias
// for namespace `A.B`:
namespace C = A.B;

// New syntax: declare `D` as a private
// namespace alias:
private namespace D = A.B;
```

Aliases of namespaces that belong to other packages are restricted being
declared `private` or in an impl file.

Support for parameterized aliases is [future work](#parameterized-aliases).

#### `using`

We define `using A.B.C;` to be a shorthand for `alias C = A.B.C;`. Note that the
argument is not required to have an periods, so `using D;` is the same as
`alias D = D;`, which creates an alias to a `D` found in a containing scope.
However, the expression must either be _something_`.`_word_ or a word by itself,
so that the resulting name is clear.

Example:

```carbon
class C {
  class D {}
}

class E(T: type) {
  // Equivalent to `alias D = C.D;`:
  using C.D;

  // Equivalent to `alias C = package.C;`:
  using C;

  // Afterward `E(X).T == X`.
  using T;
}
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

### Implicit conversions of `self`

Consider a class with an alias to a method in another class, with an implicit
conversion between the two classes:

```carbon
class Dog {
  // Equivalent to:
  // fn Bark(self: Dog);
  fn Bark(self) {
    DoBark();
  }
}

class Poodle {
  impl as ImplicitAs(Dog) {
    fn Convert(self) -> Dog {
      return {};
    }
  }
  // Equivalent to
  // `alias Bark = Dog.Bark;`
  using Dog.Bark;
}

var poodle: Poodle = {};
// Equivalent to:
// ((poodle as Dog).(Dog.Bark))()
poodle.Bark();
```

The implicit conversion of `poodle` to type `Dog` does not happen because of the
[`Core.CallPrimitive` interface](#primitive-call-interface). That interface
declares its `Self` parameter as `implicit_into anchor`, which means it won't
look for different types as part of `impl` lookup, which is important for
coherence. Instead, that implicit conversion happens as a result of the
[instance binding](/docs/design/expressions/member_access.md#instance-binding)
of `poodle` to `Dog.Bark`, when calling the `Op` method of the
[member binding interface](#primitive-member-binding-interface). In fact, there
is another conversion after that, to an adapter of `Dog` that implements
`Core.CallPrimitive` with the body of the `Dog.Bark` method.

```carbon
class __TypeOf_Dog_Bark {}
let generic __Dog_Bark: __TypeOf_Dog_Bark = {};

class __Binding_Dog_Bark {
  adapt Dog;
  impl as call[val]() -> () {
    fn Op(self) {
       DoBark();
    }
  }
}

impl __TypeOf_Dog_Bark as BindToRef(Dog) {
  where Result = __Binding_Dog_Bark;
  fn Op(unused self, ref r: Dog)
      -> ref __Binding_Dog_Bark {
    return r as __Binding_Dog_Bark;
  }
}

impl __TypeOf_Dog_Bark as BindToVal(Dog) {
  where Result = __Binding_Dog_Bark;
  fn Op(unused self, x: Dog)
      -> val __Binding_Dog_Bark {
    return x as __Binding_Dog_Bark;
  }
}
```

Only after binding is complete and the `poodle` has been converted to a `Dog` is
the `CallPrimitive` implementation looked up as part of resolving the `()`
operation. That lookup will only find calls where the type of `Self` is `Dog`.

This works with
[our approach of allowing implicit conversions with the `self` parameter](/docs/design/expressions/member_access.md#inheritance-and-other-implicit-conversions).
See extension methods
[leads issue #1345: "How is the type of me restricted, and where can methods be declared?"](https://github.com/carbon-language/carbon-lang/issues/1345)
and
[deferred proposal #1122](https://github.com/carbon-language/carbon-lang/pull/1122).

## Details

### Coherence modifier examples

Given this package:

```carbon
package P;

interface J
    (implicit_into P1: Core.ExtType,
     implicit_into P2: Core.ExtType,
     P3: Core.ExtType) { }

class C {}

// ✅ Allowed since `J` is local, even
// though `i32` is not.
impl i32 as
    J(exttype(val i32), exttype(val i32), exttype(val i32)) {}
// As if 4 impls with type structure:
// - i32 as J(i32, i32, i32)
// - i32 as J(?, i32, i32)
// - i32 as J(i32, ?, i32)
// - i32 as J(?, ?, i32)
```

A library using the interface from that package needs to ensure all of the
notional `impl`s implied by these `impl` declarations reference a local type.
This means that there needs to be a reference to a local type:

-   in a parameter that is not in a `implicit_into` parameter of the interface,
-   in a parameter that is an `exact` parameter of the `impl`, or
-   in every parameter that is an `anchor` parameter of the `impl`.

Note that the `Self` parameter before the `as` counts as a parameter for these
purposes, but local types in a `forall` clause do not.

```carbon
import P;

// ❌ Invalid since nothing is local.
impl bool as P.J(exttype(val P.C),
                 exttype(val P.C),
                 exttype(val P.C)) {}

class D {}

// ❌ Invalid since only local type is
// `D`, which is in `implicit_into` parameters.
impl i16 as P.J(exttype(val D),
                exttype(val D),
                exttype(val P.C)) {}
// As if 4 impls with type structure:
// - i16 as J(D, D, P.C)
// - i16 as J(?, D, P.C)
// - i16 as J(D, ?, P.C)
// - i16 as J(?, ?, P.C) <- problem


// ✅ Allowed: `D` is local and in a
// non-`implicit_into` parameter.
impl i8 as P.J(exttype(val P.C),
               exttype(val P.C),
               exttype(val D)) {}
// As if 4 impls with type structure:
// - i8 as J(P.C, P.C, D)
// - i8 as J(?, P.C, D)
// - i8 as J(P.C, ?, D)
// - i8 as J(?, ?, D) <- okay, has D

// ✅ Allowed: `D` is local and in an
// `exact` parameter.
impl i64 as P.J(exact exttype(val D),
                exttype(val P.C),
                exttype(val P.C)) {}
// As if 2 impls with type structure:
// - i64 as J(D, P.C, P.C)
// - i64 as J(D, ?, P.C) <- okay, has D

class E {}

// ✅ Allowed: every `anchor` parameter
// has a local type.
impl u16 as P.J(anchor exttype(val D),
                anchor exttype(val E),
                exttype(val P.C)) {}
// As if 3 impls with type structure:
// - u16 as J(D, E, P.C)
// - u16 as J(?, E, P.C) <- okay, has E
// - u16 as J(D, ?, P.C) <- okay, has D
// No J(?, ?, P.C) due to `anchor`

// ❌ Invalid since only one `anchor` is
// required to have type matching the
// query, and if it is the second
// `anchor`, then this declaration has
// no local type.
impl u32 as P.J(anchor exttype(val D),
                anchor exttype(val P.C),
                exttype(val P.C)) {}
// - u32 as J(D, P.C, P.C)
// - u32 as J(?, P.C, P.C) <- problem
// - u32 as J(D, ?, P.C)
// No J(?, ?, P.C) due to `anchor`
```

### Examples using `call`

The `call` syntax may be used as the facet type in an `impl` to define the call
operator for a type, as in:

```carbon
class OverloadedCallable {
  var x: i32;
}

// First impl: as a type facet type.
impl OverloadedCallable as
    call[ref](_: i32) -> () {
  fn Op(ref self, d: i32) -> () {
    self.x += d;
  }
}

// Second impl: as a `Core.ExtType` facet type.
// Equivalent to:
//   impl exttype(val OverloadedCallable) as
//       CallPrimitive()
//       where .ResultExtType = exttype(var i32)
impl exttype(val OverloadedCallable) as
    call() -> i32 {
  // `Self` is a `Core.ExtType` value equal to
  // `exttype(val OverloadedCallable)` so
  // `fwd self: Self` is the same as
  // `self: OverloadedCallable`.
  fn Op(fwd self: Self) -> i32 {
    return self.x;
  }
}

var c: OverloadedCallable = {.x = 1};

// Uses first impl.
Assert(c() == 1);
// Uses second impl.
c(2);
// Uses first impl.
Assert(c() == 3);
```

These facet types may also be used as a constraint for a generic parameter:

```carbon
// `call` followed by `[ref]` creates an
// ordinary type facet type
fn InvokeRef[T: call[ref](_: i32) -> ()]
    (ref f: T, arg: i32) {
  f(arg);
}

// `call` without `[`...`]` creates a
// `Core.ExtType` facet type. Category of `f`
// will be deduced from the argument.
fn InvokeExtType[PET: call() -> i32]
    (fwd f: PET) -> i32 {
  return f();
}

// Uses second impl.
InvokeRef(ref c, 4);
// Uses first impl. Must be `ref` to
// match the category of `c`, which is
// used to determine the `InvokeExtType`
// parameter.
Assert(InvokeExtType(ref c) == 7);
```

Note that the `ref` marker on arguments follows whether the category of the
parameter was eventually determined to be `ref`. `InvokeExtType(c)` would be an
error instead of using value category. See
[alternative considered: "`ref` on an argument controls its category"](#ref-on-an-argument-controls-its-category).

A `call` expression generates a constraint that matches the corresponding lambda
expression with `fn` instead of `call` and names for parameters instead of `_`:

```carbon
var i: i32 = 0;
var lambda1: auto =
    fn (x: i32) -> () { i += x };
InvokeRef(ref lambda, 1);
let lambda2: auto =
    fn () -> i32 { return i; };
InvokeExtType(lambda2)
```

### `impl` lookup for an interface with variant parameters

There are a few changes needed to
[the current `impl` selection algorithm from proposal #5337](p005337-interface-extension-and-final-impl-update.md#impl-selection-algorithm):

-   The initial filter for `impl`s that have a compatible type structure for the
    query is changed to allow differences in the type structure in the
    [variant parameters](#implicit_into-interface-parameter-variance) to the
    interface.
-   If the interface or `impl` has [`anchor`](#modifiers-for-coherence)
    parameters, at least one must have a type match with the query.
-   If an `impl` passes those tests, its type structure must be determined so
    the `impl` may be prioritized against other candidates. This is done
    following the rules
    [described in the proposal](#implicit_into-interface-parameter-variance).

Note that the only observable semantics of `impl` lookup are the sequence of
`impl`s that are considered (including their order), and the `impl` that is
eventually matched. This allows the compiler to avoid work on other `impl`s if
it can prove they won't affect what's observable.

### Conversions when calling a function argument

One advantage of the approach of
[marking the variance on the interface parameter](#implicit_into-interface-parameter-variance)
instead of the approach prior to this proposal where the `impl` would be
parameterized to allow implicit conversions, is we can get implicit conversions
even when we can't have parameterized `impl`s. This comes up with function
parameters, and this proposal allows them to work similarly to ordinary function
calls:

```carbon
fn IntArg(x: i32);

fn ConcreteCall() {
  // ✅ Performs implicit conversion from
  // `Core.IntLiteral` to `i32`.
  IntArg(7);
}

fn GenericCall[FNT: call[val](_: i32)](f: FNT) {
  // Good to get the same conversion here, even
  // though we don't (and can't) have a blanket
  // `impl` that `f` accepts anything that
  // implicitly converts to `i32`.
  f(7);
}

GenericCall(IntArg);
```

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
better supported by a
[type facet type](#converting-from-an-extended-type-facet-type-to-a-type-facet-type).

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

This feature is also needed by this proposal to model
[explicit compile-time parameters](#representing-explicit-compile-time-parameters),
so they can be used in other parts of the signature.

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

More surprising is that passing `ExactlyBool` as a function parameter will also
lose the exact type match property. This is because it will use an
[adapter](#facets-create-adapters) with the signature declared in the type
parameter instead of what was declared for `ExactlyBool`:

```
fn ErasedByGeneric
    [T: call(_: bool) -> (),
     U: ImplicitAs(bool)](fwd f: T, x: U) {
  // The deduction of parameters to `f`
  // is done by this point, so nothing to
  // prevent conversion of `x` to `bool`.
  f(x);
}

// `ExactlyBool`'s type does implement
// `call(_: bool) -> ()`.
ErasedByGeneric(ExactlyBool, t);
```

### Representing deduced parameters

Observe that deduced parameters do not show up in the
[`Core.CallPrimitive` interface](#primitive-call-interface). They are not
modeled as part of the interface, but using parameterized `impl`s of that
interface. For example:

```carbon
fn Ident[T: type](x: T) -> T;
```

can be modeled as

```carbon
class __Ident_Type {}
let generic Ident: __Ident_Type = {};
impl forall [T: type]
    exttype(val __Ident_Type)
    as CallPrimitive(exttype(val T))
    where .ResultExtType = exttype(var T);
```

or equivalently:

```carbon
class __Ident_Type {}
let generic F: __Ident_Type = {};
impl forall [T: type] __Ident_Type
    as call[val](_: T) -> T;
```

This partially addresses the ["subsumption" problem](#problem), since this
`Ident` could be passed to this function:

```carbon
fn GenericCall[FNT: call[val](_: i32)](f: FNT);
```

This will succeed since when it looks to see if `__Ident_Type` implements
`call[val](_: i32)`, it will find that the parameterized `impl` does provide
that when `T` is `i32`.

### Representing explicit compile-time parameters

Explicit compile-time parameters are modeled the same way as deduced parameters,
except they also are included in the list of parameter extended types passed to
`CallPrimitive`, using [`exttypeof`](#exttypeof).

For example:

```carbon
fn F(generic T: type, x: T) -> T;
```

can be modeled as:

```carbon
class __F_Type {}
let generic F: __F_Type = {};
impl forall [T: type] exttype(val __F_Type)
    as CallPrimitive(exttypeof(T),
                     exttype(val T))
    where .ResultExtType = exttype(var T);
```

or equivalently:

```carbon
class __F_Type {}
let generic F: __F_Type = {};
impl forall [T: type] __F_Type as
    call[val](fwd _: exttypeof(T), _: T) -> T;
```

An explicit `template` parameter, such as `fn H(template T: type, x: T) -> T;`,
is modeled the same way using `impl forall [template T: type]` with the same
`CallPrimitive(exttypeof(T), exttype(val T))` interface arguments. Per
[leads issue #2153](https://github.com/carbon-language/carbon-lang/issues/2153),
this accepts both template and symbolic compile-time arguments satisfying
`type`, while giving `T` template phase inside the `impl` so dependent parameter
extended types, `.ResultExtType`, and the `Op` body are checked at
monomorphization.

### Callables that may or may not mutate

In many cases, a function accepting a callable object will not care about
whether the call mutates the object.

Consider an example of two callables, one that has no mutation, and one that
does:

```carbon
fn CallbackNoMutation() -> i32 {
  // https://xkcd.com/221/
  return 4;
}

class Counter {
  var count: i32;
  impl as call[ref]() -> i32 {
    fn Op(ref self) -> i32 {
      self.count += 1;
      return self.count;
    }
  }
}
var counter: Counter = {.count = 0};
```

There are two approaches to accommodating this that are supported by the current
design. The function taking the callable object can be generic on its aspect in
the caller, as in:

```carbon
fn GenericCaller[PET: call() -> i32]
    (fwd f: PET) -> i32 {
  return f();
}

Assert(GenericCaller(CallbackNoMutation) == 4);
Assert(GenericCaller(ref counter) == 1);
```

Alternatively, we can have the function take its argument by value, and make a
wrapper that holds a pointer to make the call:

```carbon
// Only has to be defined once
class CalledThroughPointer(T: type) {
  var p: T*;
}

fn CallThroughPointer[T: type](p: T*)
    -> CalledThroughPointer(T) {
  return {.p = p};
}

impl forall
    [... each P: Core.ExtType,
     // Or: `T: call[ref](... fwd _: each P)`
     T: Core.RefSelf(Core.CallPrimitive(... each P))]
    exttype(val CalledThroughPointer(T)) as
    Core.CallPrimitive(... each P) {
  where ResultExtType = T.ResultExtType;
  fn (self, ... fwd each param: each P) -> fwd ResultExtType {
    return (*self.p)(... fwd ~each param);
  }
}

// Or: `FNT: Core.Call() where .Result = i32`
fn ValueCaller[FNT: call[val]() -> i32](f: FNT) -> i32 {
  return f();
}

Assert(ValueCaller(CallbackNoMutation) == 4);
Assert(ValueCaller(CallThroughPointer(&counter)) == 2);
```

NOTE: The `~` is there in anticipation of using moves to support forwarding, see
[the "Forwarding" section](#forwarding).

The first, generic approach is a bit more convenient but means using extended
types. The second approach only uses types once `CallThroughPointer` has been
defined, but is less convenient. Some may prefer the second approach since it
makes it explicit when the callable object may be mutated.

### Forwarding

Forwarding is where an argument is passed along generically, including aspects
of its extended type like category and phase. Input-to-input forwarding is where
the arguments passed to one function become the arguments passed to another. For
example, this `Apply` function takes a callable object and argument expressions
to call it with and performs the call:

```carbon
fn Apply[... each ArgET: Core.ExtType,
         FNT: call(... fwd _: each ArgET)]
        (fwd f: FNT,
         ... fwd each arg: each ArgET)
        -> fwd FNT.ResultExtType {
  return (~f)(... fwd ~each arg);
}
```

> **Note:** Forwarding of a `var` parameter will be marked with `~` (as in the
> example) once moves are supported. Unlike input-to-output forwarding below
> (which must initialize a separate caller-provided return address and therefore
> requires `Core.Move`), input-to-input forwarding of a `var` parameter is
> expected to reuse its caller-allocated `var` storage without requiring
> `Core.Move`; see [`Core.ConvertsInto`](#coreconvertsinto).

> **Note:** An expression with generic extended type is marked with `fwd`
> (rather than `ref`) when passed as an argument, since whether it instantiates
> to a reference is not known at definition time. See
> [the alternative considered](#fwd-on-call-arguments).

This works with `Apply` calling `f` once. To support multiple calls, we drop
support for using `var` parameters with initializing argument expressions:

```carbon
fn Apply2[... each ArgET: Core.NoVarExtType,
          FNT: call(... fwd _: each ArgET) &
                Core.NoVarExtType]
         (fwd f: FNT, ... fwd each arg: each ArgET) {
  f(... fwd each arg);
  f(... fwd each arg);
}
```

With this change, following the
[rules of deduction](#deducing-an-extended-type-value), initializing expressions
will be converted to values.

Input-to-output forwarding is where argument extended types are returned from a
function, as in:

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

This was discussed in open discussion:

-   [2025-05-07](https://docs.google.com/document/d/1Yt-i5AmF76LSvD4TrWRIAE_92kii6j5yFiW-S7ahzlg/edit?tab=t.0#heading=h.4zbo49wg5rmk):
    definition of `Apply` using the proposed extended types
-   [2025-05-12](https://docs.google.com/document/d/1Yt-i5AmF76LSvD4TrWRIAE_92kii6j5yFiW-S7ahzlg/edit?tab=t.0#heading=h.1mjh6unumnwu):
    calling conventions limit forwarding for `var` extended types

#### Forwarding tuple parameters

A tuple expression can have elements with different categories and other
different extended types, not just different types. This proposal uses
[compound extended type values](#compound-extended-type-values) to properly
capture the extended type of each element, so a tuple expression can be
faithfully forwarded. Alternatives where the type of an expression was kept
separate from the other extended type aspects struggled to capture this, see the
[discussion on 2025-05-07](https://docs.google.com/document/d/1Yt-i5AmF76LSvD4TrWRIAE_92kii6j5yFiW-S7ahzlg/edit?tab=t.0#heading=h.4zbo49wg5rmk).

The approach from this proposal allows functions that pass variable-length
argument lists using tuple patterns, such as this definition of `ApplySum`

```carbon
// Multiple variadic parameters have to be passed as tuples.
fn ApplySum[... each ArgET: Core.NoVarExtType,
            Fn: call(... fwd _: each ArgET) -> i32 & Core.NoVarExtType]
           (fwd f: Fn,
            (... fwd each a: each ArgET),
            (... fwd each b: each ArgET)) -> i32 {
  return f(... fwd each a) + f(... fwd each b);
}
```

to compose with the `Apply` function from
[the "forwarding" section](#forwarding), as in:

```carbon
// Preserves the categories of the argument
// tuple expressions.
Apply(ApplySum, (...), (...))
```

This works since the extended type deduced for the arguments of `Apply` can be a
composite extended type value that captures the extended type of each element of
each tuple argument.

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

-   To introduce another phase that represents a values that are constant, in
    the sense of have a value determined at compile time, but where the value is
    unavailable for typechecking.
-   We will rename "runtime" to "execute time," to include the values of
    functions whether they are executed at compile time or runtime.

### Impl prioritization

We still need to have a mechanism for picking an `impl` to use when there are
multiple with the most preferred type structure. This issue is now more acute
with extended types, since there is more non-type information that can now
appear in an `impl` signature. Resolving this question will allow us to support
prioritization across the `CallPrimitive` implementations for a type, which is
how we expect to support selecting an overload from a function overload set.

### Parameterized aliases

Just like
[C++ supports parameterized type aliases](https://en.cppreference.com/w/cpp/language/type_alias.html),
it would be useful for Carbon to support parameterized aliases. These would act
a bit like functions, except that the arguments to the alias may be deduced. For
example, this generic `Double` alias:

```carbon
alias Double(T: type) = (T, T) as type;
```

would allow you to define a function that accepted any pair of two values of the
same type:

```carbon
// `T` is deduced from the argument type.
fn SumPair[T: Add](p: Double(T)) -> T {
  return p.0 + p.1;
}

let three: i32 = 3;
Assert(SumPair((three, three)) == 6);
```

A parameterized alias was mentioned in
[proposal #2875](p002875-functions-function-types-and-function-calls.md#function-pointers)
as a way to define a parameterized `FunctionPtr` type.

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
expressions (we might require explicit argument to be marked with `ref`). An
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

### Allow more variance

If motivated by use cases, we could add support for `implicit_into` variance on
associated constants, or variance on types, not just extended types. This would
be above and beyond types that are eventually used to construct an extended
type, as considered in
[the "Interaction between `anchor`, `exact`, and named constraints" section](#interaction-between-anchor-exact-and-named-constraints).

### `extend` in `where`...`impls` expressions

A core-language syntax allowing `extend` inside a
[`where`...`impls`](#whereimpls) expression, such as
`type where exttype(val .Self) extend impls Callback`, could expose names from
the required interface directly on the resulting facet type (for example, in
`InvokeWithResultOfFunctionCall`
[above](#converting-from-an-extended-type-facet-type-to-a-type-facet-type), as
well as replacing `Core.ValSelf`, `Core.RefSelf`, and `Core.VarSelf`). We defer
designing general `where`...`extend impls` semantics as future work in order to
limit the scope of this proposal.

## Rationale

This advances [Carbon's goals](/docs/project/goals.md):

-   [Code that is easy to read, understand, and write](/docs/project/goals.md#code-that-is-easy-to-read-understand-and-write):
    This proposal allows some programs and APIs to be written that couldn't
    otherwise be expressed in Carbon.
-   [Practical safety and testing mechanisms](/docs/project/goals.md#practical-safety-and-testing-mechanisms):
    The increased expressivity of extended types allows us to prevent some bugs.
    For example, limiting conversion of `partial` types to initializing
    expressions avoids misuses of those values.

It does this while adhering to [Carbon's principles](/docs/project/principles):

-   [Principle: All APIs are library APIs](/docs/project/principles/library_apis_only.md):
    This machinery allows many more constructs to be expressed within the
    language as opposed to being special capabilities only available to
    built-ins. This includes abilities like how conversion to an adapter type
    can preserve category.
-   [Principle: Progressive disclosure](/docs/project/principles/progressive_disclosure.md):
    Developers don't need to be aware of this machinery unless they are engaged
    in code that needs these advanced features.

## Alternatives considered

### Decoupling type from the rest of the extended type

The goal of this approach was to avoid introducing extended type interfaces.
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
[the "any extended type with this type" section](#any-extended-type-with-this-type).

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
approach we settled on more naturally supported
[getting the type from the extended type using `.type`](#extended-types-in-impls),
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
interface. This was
[discussed on 2025-05-28](https://docs.google.com/document/d/1Yt-i5AmF76LSvD4TrWRIAE_92kii6j5yFiW-S7ahzlg/edit?tab=t.0#heading=h.9pvtvm7atpo5).

### Represent all extended type information in ordinary types

We could represent the additional information present in extended types in
ordinary types. This would avoid splitting constructs, such as different kinds
of interfaces for types and extended types. The concern is that there are cases
where we want to consider type separate from the other aspects of extended type.
Further, we want the rules for extended types to be different than types. For
example, category and phase are independent, not ordered like type composition.
Also, different categories are mutually exclusive.

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
increasing complexity in area that is already quite complex. Merging has some
downsides, for example that compile-time value information is only currently
used for parameter, but in
[a discussion on 2025-06-03](https://docs.google.com/document/d/1Yt-i5AmF76LSvD4TrWRIAE_92kii6j5yFiW-S7ahzlg/edit?tab=t.0#heading=h.hdj6hk4bwd2z)
we determined that is expected to change once we support
[functions executed at compile-time](#compile-time-calls).

The additional type information couldn't be used to determine the variance of an
extended type parameter, since extended types in `fwd` bindings can appear in
positions with different variance, as
[discussed on 2025-06-24](https://docs.google.com/document/d/1Yt-i5AmF76LSvD4TrWRIAE_92kii6j5yFiW-S7ahzlg/edit?tab=t.0#heading=h.dqgfbeb33txh).

Example:

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
specification of the parameters or return for `CallPrimitive`. However, in
[discussion on 2025-05-14](https://docs.google.com/document/d/1Yt-i5AmF76LSvD4TrWRIAE_92kii6j5yFiW-S7ahzlg/edit?tab=t.0#heading=h.52tb7l2he343)
we decided against having any implicit conversion, for a few reasons:

-   The natural default for parameter extended types (value) is a different from
    extended return types (initializing). Picking one would lead to confusion
    when using the other. Making it different somehow using contextual
    information would lead to inconsistencies that were also expected to be
    confusing.
-   It did not allow you to deal only in types instead of using extended types
    in any use case we foresaw.
-   Extended types are an advanced feature that is expected to be used rarely.
    As a result, brevity is not as valuable, and readers are less likely to be
    familiar with which default would be used. The `CallPrimitive` use case is
    already well supported by using [`call` sugar](#call) instead.
-   Using the implicit conversion for specifying arguments to `CallPrimitive`
    could lead to ambiguity. For example `CallPrimitive((T, U))` could be
    equivalent to `call((_: T, _: U))` or `call(_: (T, U))`.

### Other ways to determine variance in interfaces

We first considered `in` and `out` (such as in discussions on
[2025-05-28](https://docs.google.com/document/d/1Yt-i5AmF76LSvD4TrWRIAE_92kii6j5yFiW-S7ahzlg/edit?tab=t.0#heading=h.9pvtvm7atpo5)
and
[2025-05-29](https://docs.google.com/document/d/1Yt-i5AmF76LSvD4TrWRIAE_92kii6j5yFiW-S7ahzlg/edit?pli=1&tab=t.0#heading=h.dj0nqhfqo67w))
to mark the different kinds of variance instead of
[`implicit_into`](#implicit_into-interface-parameter-variance) and
[`implicit_from`](#implicit_from-variance), following other languages such as
Scala, Kotlin, Typescript, and C#. We
[decided to switch 2025-06-24](https://docs.google.com/document/d/1Yt-i5AmF76LSvD4TrWRIAE_92kii6j5yFiW-S7ahzlg/edit?tab=t.0#heading=h.dqgfbeb33txh)
that `in` and `out` were particularly expensive words to reserve, and which
corresponded to each kind of variance was hard to remember. The chosen keywords
are intended to emphasize what implicit conversion will be used.

We also considered making the variance keyword part of the parameter name, much
like `each` on variadic parameters. This seemed unnecessary verbosity given
that:

-   the compiler would enforce that the variance was consistent with usage,
    there was no choice about what variance to use in a given position; and
-   the `impl` you write is actual `impl` you get, there is just there's another
    layer around it.

We considered inferring variance, or just marking that there was variance and
inferring only the direction of variance. This had some downsides:

-   Variance comes with significantly different semantics, causing
    [facets to create adapters](#facets-create-adapters), so better if it is opt
    in.
-   Don't always want variance, for example since it requires extra annotations
    on impls to avoid coherence problems.
-   It goes against our
    ["information accumulation" principle](/docs/project/principles/information_accumulation.md)
    and general policy of not determining properties of parameters based on how
    they are used in the body of the definition:
    -   This allows forward declarations to have the information needed to be
        used.
    -   Avoids action at a distance. For example, having to look at the body of
        the interface and referenced interfaces.
    -   Makes the contract visible and explicit.
    -   Avoids potentially expensive reasoning when there are recursive
        references to the containing type.

### Support implicit conversions in impls

The support for implicit conversions prior to this proposal was provided by
parameterizing an `impl` so that it also applied to types that could be
implicitly converted from. This had the downside that the following code is an
error:

```carbon
fn F[T: call[val](_: i32)->bool](f: T) -> bool {
  // Error: `T` supports `i32` arguments, not
  // `Core.IntLiteral`.
  return f(7);
}
```

Given Carbon's reliance on implicit conversions in various contexts, this was
considered important to address. We did not see a path to addressing this by
adding `forall` requirements that the impls providing the implicit conversions
exist. The issue is that those would be very hard to satisfy in the presence of
specialization. In addition, `forall` constraint would preclude being called
with an overload set with overlap in arities. In general, the concern is that
there is nothing preventing the creation of a new library with a new type that
implicitly converts to the parameter types of both members of the overload set.
This would be an evolution hazard.

This was
[discussed on 2025-05-28](https://docs.google.com/document/d/1Yt-i5AmF76LSvD4TrWRIAE_92kii6j5yFiW-S7ahzlg/edit?tab=t.0#heading=h.9pvtvm7atpo5).

### Infer `anchor` in `impl` declarations

The `anchor` annotation could be inferred instead of marked explicitly by which
types mentioned in the `impl` declaration are local to that file. That has some
downsides:

-   It causes the `impl` declaration to be sensitive to declarations that can be
    anywhere earlier in the same file. This causes readers of the code to need
    to scan the rest of the file to understand the which arguments need a type
    match, and makes the impact of changes less local. This is against the
    ["low context-sensitivity" principle](/docs/project/principles/low_context_sensitivity.md).
-   This would cause semantic changes when concatenating files, for example to
    make a bug report. In particular, more types could then be used as anchors
    for satisfying the orphan rule. This is against the
    ["file concatenation should preserve meaning" principle](/docs/project/principles/file_concatenation.md).

This was
[discussed on 2025-05-28](https://docs.google.com/document/d/1Yt-i5AmF76LSvD4TrWRIAE_92kii6j5yFiW-S7ahzlg/edit?tab=t.0#heading=h.9pvtvm7atpo5).

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
define a compound `->` extended return type syntax that was done since this
proposal was originally authored. For now, since the mixed case will continue to
use the current approach, we are avoiding introducing a second way of expressing
these extended types.

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

### Replace `require` for interface-interface requirements

In some cases, a `require` declaration in an interface to add a constraint that
another interface must be implemented, as in:

```carbon
interface J {}

interface I {
  require impls J;
}
```

is equivalent to an interface with a constraint on `Self`:

```carbon
interface I(Self: J) {}
```

This however had concerns which led us to keep using `require` and disallow
interface, named constraint, or `where` facet types on `Self` (avoiding
[two ways to express the same requirement](/docs/project/principles/one_way.md)):

-   It raised questions about how qualified name lookup into `Self` would work.
-   We preferred these requirements to be expressed in the body of the interface
    definition, not declaration, from a reading perspective.
-   Having the requirements in the declaration added restrictions on `I`'s
    declaration.

This was discussed on
[2025-05-08](https://docs.google.com/document/d/1Yt-i5AmF76LSvD4TrWRIAE_92kii6j5yFiW-S7ahzlg/edit?tab=t.0#heading=h.vdognq1upsf5).

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
    had `"symbolic"` phase in `Sym`'s
    [`CallPrimitive`](#primitive-call-interface) implementation and `"template"`
    phase in `Tmpl`'s, [`ConvertsInto`](#coreconvertsinto) would have to convert
    `"template"` extended types to `"symbolic"` extended types for `Sym(i32)`
    and `"symbolic"` extended types to `"template"` extended types for `Tmpl(U)`
    (including when forwarding through `Apply`). Converting in both directions
    would prevent `Core.ExtType` constraints from distinguishing the two phases
    anyway, whereas allowing implicit `"symbolic"` to `"template"` expression
    conversions in general would silently make checked-generic expressions
    template-dependent without an explicit `template` binding.
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

### `alias` of a reference expression

We considered several alternatives for how [`alias`](#alias) should handle
reference expressions, reflecting a tension between thinking of `alias` as
"giving a name to an existing entity" (where aliasing a variable and aliasing a
function look the same, though pure name aliasing cannot handle expressions like
`buf(i32)`) versus "giving a name to a value" (where we want to distinguish when
the alias's value may change if the underlying storage is mutated):

-   **No `ref` marker (always referentially transparent)**: `alias A = R;`
    automatically preserves the category of the right-hand side (while still
    converting initializing expressions to values, and potentially capturing
    mixed-category tuples). We chose to require `ref` instead so readers can
    tell at the declaration site that the value accessed through `A` may change
    when the underlying storage is mutated through other names.
-   **Explicit `val` marker (`alias A = val E;`)**: bare `alias` follows the
    right-hand side's category, while `val` (or another keyword like `eval`)
    forces conversion to a value (which would be a special case of the previous
    option if Carbon had a general `val` operator). We decided against this
    since we didn't value highlighting the difference between naming and
    evaluating in the source. We have existing ways of converting expressions
    to values, such as `let template` or a `musteval` function, in the rare case
    that conversion is needed.
-   **`alias A = ref R;` instead of `alias ref A = R;`**: putting `ref` on the
    right-hand side would mirror `ref` on call arguments. We chose
    `alias ref A = R;` instead because `alias` acts as an untyped binding
    analogous to `let ref A: T = R;`.
-   **Rejecting all reference expressions without `ref`, even if
    compile-time-readable**: Under this stricter variant, `alias A = B;` would
    forbid any reference expression on the right-hand side rather than
    converting `B` to a value. Instead, we chose to convert `B` to a value
    because we wanted to support initializing expressions as long as the result
    of the conversion is known at compile time. Applying the same rule to
    reference expressions also supports those whose values are known at compile
    time; we want to treat them like compile-time constants and not require a
    `ref` marker since they won't be mutated. The actual way things like global
    constant arrays are declared is outside the scope of this proposal, but we
    imagined they might use `let` instead of `var` to reflect that they are
    immutable. For mutable variables (declared with `var`), converting to a
    value does not produce a compile-time constant, so omitting `ref` is still
    an error rather than silently changing semantics.

This was discussed on
[2026-04-02](https://docs.google.com/document/d/1mjllGO3ZCL4qGt9uJHUtcxKoHAGEY7Y999ie4EtBWB8/edit?tab=t.h6lecvcwu214#heading=h.vvcml6bfo69q)
and
[2026-11-18](https://docs.google.com/document/d/1mjllGO3ZCL4qGt9uJHUtcxKoHAGEY7Y999ie4EtBWB8/edit?tab=t.uvyw2c4zrz9e#heading=h.edvediz60r3d).

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
