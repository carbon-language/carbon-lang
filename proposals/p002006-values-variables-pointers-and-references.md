# Values, variables, pointers, and references

<!--
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

[Pull request](https://github.com/carbon-language/carbon-lang/pull/2006)

<!-- toc -->

## Table of contents

-   [Abstract](#abstract)
-   [Problem](#problem)
    -   [Conceptual integrity between local variables and parameters](#conceptual-integrity-between-local-variables-and-parameters)
-   [Background](#background)
-   [Proposal](#proposal)
    -   [Values, objects, and expressions](#values-objects-and-expressions)
    -   [Patterns, `var`, `let`, and local variables](#patterns-var-let-and-local-variables)
    -   [Pointers, dereferencing, and references](#pointers-dereferencing-and-references)
    -   [Indexing](#indexing)
    -   [`const`-qualified types](#const-qualified-types)
    -   [Interop with `const &` and `const` methods](#interop-with-const--and-const-methods)
    -   [Customization](#customization)
-   [Rationale](#rationale)
    -   [Contrasting the rationale for pointers with `goto`](#contrasting-the-rationale-for-pointers-with-goto)
-   [Alternatives considered](#alternatives-considered)
    -   [Value expression escape hatches](#value-expression-escape-hatches)
    -   [References in addition to pointers](#references-in-addition-to-pointers)
    -   [Syntax-free or automatic dereferencing](#syntax-free-or-automatic-dereferencing)
        -   [Syntax-free address-of](#syntax-free-address-of)
    -   [Exclusively using references](#exclusively-using-references)
    -   [Alternative pointer syntaxes](#alternative-pointer-syntaxes)
        -   [Pointer type alternative syntaxes](#pointer-type-alternative-syntaxes)
        -   [Pointer dereference syntax alternatives](#pointer-dereference-syntax-alternatives)
        -   [Pointer syntax conclusion](#pointer-syntax-conclusion)
    -   [Alternative syntaxes for locals](#alternative-syntaxes-for-locals)
    -   [Make `const` a postfix rather than prefix operator](#make-const-a-postfix-rather-than-prefix-operator)
-   [Appendix: background, context, and use cases from C++](#appendix-background-context-and-use-cases-from-c)
    -   [`const` references versus `const` itself](#const-references-versus-const-itself)
    -   [Pointers](#pointers)
    -   [References](#references)
        -   [Special but critical case of `const T&`](#special-but-critical-case-of-const-t)
        -   [R-value references and forwarding references](#r-value-references-and-forwarding-references)
        -   [Mutable operands to user-defined operators](#mutable-operands-to-user-defined-operators)
        -   [User-defined dereference and indexed access syntax](#user-defined-dereference-and-indexed-access-syntax)
        -   [Member and subobject accessors](#member-and-subobject-accessors)
        -   [Non-null pointers](#non-null-pointers)
        -   [Syntax-free dereference](#syntax-free-dereference)

<!-- tocstop -->

## Abstract

Introduce a concrete design for how Carbon values, objects, storage, variables,
and pointers will work. This includes fleshing out the design for:

-   The expression categories used in Carbon to represent values and objects,
    how they interact, and terminology that anchors on their expression nature.
-   An expression category model for readonly, abstract values that can
    efficiently support input to functions.
-   A customization system for value expression representations, especially as
    seen on function boundaries in the calling convention.
-   An expression category model for references instead of a type system model.
-   How patterns match different expression categories.
-   How initialization works in conjunction with function returns.
-   Specific pointer syntax, semantics, and library customization mechanisms.
-   A `const` type qualifier for use when the value expression category system
    is too abstracted from the underlying objects in storage.

## Problem

Carbon needs a design for how values, variables, objects, pointers, and
references work within the language. These designs are heavily interdependent
and so they are presented together here. The design also needs to provide a
compelling solution for a wide range of use cases in the language:

-   Easy to use, and use correctly, function input and in/out parameters.
-   Clear separation of read-only use cases like function inputs from mutable
    use cases.
-   Local names bound both to fixed values and mutable variables.
-   Indirect (and mutable) access to objects (by way of pointers or references).
-   Extensible models for dereferencing and indexing.
-   A delineated method subset which forms the API for read-only or input
    objects.
-   A delineated method subset which forms the APIs for indirect access to
    shared objects in thread-compatible ways.
-   Interoperation between function input parameters and C++ `const &`
    parameters.
-   Complex type designs which expose both interior pointers and exterior
    pointers transparently for extended dereferencing or indexing.

### Conceptual integrity between local variables and parameters

An additional challenge that this design attempts to address is retaining the
conceptual integrity between local variables and parameters. Two of the most
fundamental refactorings in software engineering are _inlining_ and _outlining_
of regions of code. These operations introduce or collapse one of the most basic
abstraction boundaries in the language: functions. These refactorings translate
between local variables and parameters in both directions. In order to ensure
these translations are unsurprising and don't face significant expressive gaps
or behavioral differences, it is important to have strong semantic consistency
between local variables and function parameters. While there are some places
that these need to differ, there should be a strong overlap of the core
facilities, design, and behavior.

## Background

Much of this is informed by the experience of working with increasingly complex
"value categories" (actually categorizing expressions) and parameter passing in
C++ and how the language arrived there. Some background references on this area
of C++ and the problems encountered:

-   https://en.cppreference.com/w/cpp/language/value_category
-   http://wg21.link/basic.lval
-   https://www.scs.stanford.edu/~dm/blog/decltype.html
-   https://medium.com/@barryrevzin/value-categories-in-c-17-f56ae54bccbe
-   http://isocpp.github.io/CppCoreGuidelines/CppCoreGuidelines#Rf-conventional
-   https://www.youtube.com/watch?v=Tz5drzXREW0

I've also written up a detailed walk-through of the different use cases and
considerations that touch on the space of values, references, function inputs,
and more across C++ in an
[appendix](#appendix-background-context-and-use-cases-from-c).

Leads questions which informed the design proposed here:

-   [What syntax should we use for pointer types? (#523)][#523]

It also builds on the design of the proposal ["Initialization of memory and
variables"][p0257] ([#257]), implementing part of [#1993].

[#257]: https://github.com/carbon-language/carbon-lang/pull/257
[#523]: https://github.com/carbon-language/carbon-lang/issues/523
[#1993]: https://github.com/carbon-language/carbon-lang/issues/1993
[p0257]: /proposals/p0257.md

## Proposal

This section provides a condensed overview of the proposal. The details are
covered in the updated content in the design, and each section links into the
relevant content there. While this overview both duplicates and summarizes
content, it isn't intending anything different from the updates to the design
content, and the design content should be considered authoritative as it will
also continue to be maintained going forward.

### Values, objects, and expressions

Carbon has both abstract _values_ and concrete _objects_. Carbon _values_ are
things like `42`, `true`, and `i32` (a type value). Carbon _objects_ have
_storage_ where values can be read and written. Storage also allows taking the
address of an object in memory in Carbon.

Both objects and values can be nested within each other. For example
`(true, true)` is both a value and also contains two sub-values. When a
two-tuple is stored somewhere, it is both a tuple-typed object and contains two
subobjects.

> Details:
> [Values, objects, and expressions](/docs/design/values.md#values-objects-and-expressions)

Expressions are categorized in a way that explains how they produce values or
refer to objects:

-   [_Value expressions_](/docs/design/values.md#value-expressions) produce
    abstract, read-only _values_ that cannot be modified or have their address
    taken.
-   [_Reference expressions_](/docs/design/values.md#reference-expressions)
    refer to _objects_ with _storage_ where a value may be read or written and
    the object's address can be taken.
    -   [_Durable reference expressions_](/docs/design/values.md#durable-reference-expressions)
        are reference expressions which cannot refer to _temporary_ storage, but
        must refer to some storage that outlives the full expression.
    -   [_Ephemeral reference expressions_](/docs/design/values.md#ephemeral-reference-expressions)
        are reference expressions which _can_ refer to temporary storage.
-   [_Initializing expressions_](/docs/design/values.md#initializing-expressions)
    which require storage to be provided implicitly when evaluating the
    expression. The expression then initializes an object in that storage. These
    are used to model function returns, which can construct the returned value
    directly in the caller's storage.

> Details: [Expression categories](/docs/design/values.md#expression-categories)

### Patterns, `var`, `let`, and local variables

Patterns are by default _value patterns_ and match _value expressions_, but can
be introduced with the `var` keyword to create a _variable pattern_ that has
_storage_ and matches _initializing expressions_. Names bound in value patterns
become value expressions, and names bound in a variable pattern become _durable
reference expressions_ referring to an object in that pattern's storage.

Local patterns can be introduced with `let` to get the default behavior of a
readonly pattern, or they can be directly introduced with `var` to form a
variable pattern and declare mutable local variables.

> Details:
> [Binding patterns and local variables with `let` and `var`](/docs/design/values.md#binding-patterns-and-local-variables-with-let-and-var)

### Pointers, dereferencing, and references

Pointers in Carbon are the primary mechanism for _indirect access_ to storage
containing some object. Dereferencing a pointer forms a
[_durable reference expression_](/docs/design/values.md#durable-reference-expressions)
to the object.

Carbon pointers are heavily restricted compared to C++ pointers -- they cannot
be null and they cannot be indexed or have pointer arithmetic performed on them.
Carbon will have dedicated mechanisms that still provide this functionality, but
those are future work.

> Details: [Pointers](/docs/design/values.md#pointers)

The syntax for working with pointers is similar to C++:

```carbon
var i: i32 = 42;
var p: i32* = &i;

// Form a reference expression `*p` and assign `13` to the referenced storage.
*p = 13;
```

> Details:
>
> -   [Pointer syntax](/docs/design/values.md#pointer-syntax)
> -   [Pointer operators](/docs/design/expressions/pointer_operators.md)

Carbon doesn't have reference types, just reference expressions. API designs in
C++ that use references (outside of a few common cases like `const &` function
parameters) will typically use pointers in Carbon. The goal is to simplify and
focus the type system on a primary model of indirect access to an object.

> Details: [Reference types](/docs/design/values.md#reference-types)

### Indexing

Carbon supports indexing that both accesses directly contained storage like an
array and indirect storage like C++'s `std::span`. As a result, the exact
interfaces used for indexing reflect the expression category of the indexed
operand and the specific interface its type implements. This proposal just
updates and refines this design with the new terminology.

> Details: [Indexing](/docs/design/expressions/indexing.md)

### `const`-qualified types

Carbon provides the ability to qualify a type `T` with the keyword `const` to
get a `const`-qualified type: `const T`. This is exclusively an API-subsetting
feature in Carbon -- for more fundamentally "immutable" use cases, value
expressions and bindings should be used instead. Pointers to `const`-qualified
types in Carbon provide a way to reference an object with an API subset that can
help model important requirements like ensuring usage is exclusively by way of a
_thread-safe_ interface subset of an otherwise _thread-compatible_ type.

> Details:
> [`const`-qualified types](/docs/design/values.md#const-qualified-types)

### Interop with `const &` and `const` methods

Carbon makes value expressions interoperable with `const &` function parameters.
There are two dangers of this approach:

-   Those constructs in C++ _do_ allow the address to be taken, so Carbon will
    synthesize an object and address for value expressions for the duration of
    the C++ call as needed. C++ code already cannot rely on the address of
    `const &` parameters continuing to exist once the call completes.
-   C++ also allows casting away `const`. However, this doesn't allow a method
    to safely _mutate_ a `const &` parameter, except its `mutable` members.

In both cases, correct C++ code should already respect the limitations Carbon
needs for this interop. The same applies to the implicit `this` parameter of
`const` methods in C++.

> Details:
> [Interop with C++ `const &` and `const` methods](/docs/design/values.md#interop-with-c-const--and-const-methods)

### Customization

Carbon will provide a way to customize the implementation representation of a
value expression by nominating some other type to act as its representation.
This will be indicated through an explicit syntactic marker in the class
definition, and will require the type to `impl` a customization interface
`ReferenceImplicitAs` to provide the needed functionality. The result will be to
form this custom type when forming a value expression from a reference
expression, and will restrict the operations on such a value expression to
implicitly converting to the nominated type.

> Details:
> [Value representation and customization](/docs/design/values.md#value-representation-and-customization)

Carbon will also allow customizing the behavior of a dereference operation on a
type to allow building "smart pointers" or other pointer-like types in the
library. This will be done in a similar fashion to overloading other operators,
where the overloaded operation returns some other pointer that is then
dereferenced to form a reference expression.

> Details:
> [Dereferencing customization](/docs/design/values.md#dereferencing-customization)

## Rationale

-   Pointers are a fundamental components of all modern computer hardware --
    they are abstractly random-access machines -- and being able to directly
    model and manipulate this is necessary for
    [performance-critical software](/docs/project/goals.md#performance-critical-software).
-   Simplifying the type system by avoiding both pointers and references is
    expected to make
    [code easier to read, understand, and write](/docs/project/goals.md#code-that-is-easy-to-read-understand-and-write).
-   Creating space in both the syntax and type system to introduce ownership and
    lifetime information is important to be able to address long term
    [safety](/docs/project/goals.md#practical-safety-and-testing-mechanisms)
    needs.
-   Pointers are expected to be deeply familiar to C++ programmers and easily
    [interoperate with C++ code](/docs/project/goals.md#interoperability-with-and-migration-from-existing-c-code).

### Contrasting the rationale for pointers with `goto`

There is an understandable concern about Carbon deeply incorporating pointers
into its design -- pointers in C and C++ have been the root of systematic
security and correctness bugs. There is an amazing wealth of "gotchas" tied to
pointers that it can seem unreasonable to build Carbon on top of these
foundations. By analogy, the `goto` control-flow construct is similarly
considered _deeply_ problematic and there is a wealth of literature in computer
science and programming language design on how and why to build languages on top
of structured control flow instead. It would be very concerning to build `goto`
into the fundamental design of Carbon as an integral and pervasive component of
its control flow. However, there are two important distinctions between pointers
and `goto`.

First, C and C++ pointers are _very_ different from Carbon pointers, and their
rightly earned reputation as a source of bugs and confusion stems precisely from
these differences:

-   C pointers can be null, the ["billion dollar mistake"][null-mistake],
    Carbon's cannot be. Carbon requires null-ness to be explicitly modeled in
    the type system and dealt with prior to dereference.
-   C pointers can be indexed even when not pointing to an array. Carbon's
    pointers are to singular objects. Carbon handles indexing with a separate
    construct and types to distinguish arrays where indexing is semantically
    meaningful from the cases where it is not.
-   C pointers support deeply surprising indexing patterns such as `3[pointer]`.
    Carbon restricts to unsurprising indexing.
-   C pointers don't enforce lifetime or any other memory safety properties.
    While Carbon does not yet support these deeper protections, tackling memory
    safety is an explicit plan of Carbon's and may yet result in changes to the
    pointer type to reflect the safety risks posed by them.

[null-mistake]:
    https://www.infoq.com/presentations/Null-References-The-Billion-Dollar-Mistake-Tony-Hoare/

While these are described in terms of C's pointers, C++ inherits these without
meaningful improvement from C. The core point here is that while Carbon is
building pointers into its foundations, it is building in the least problematic
and most useful aspect of pointers and leaving almost all of the legacy and risk
behind. That is a critical part of what makes pointers viable in Carbon.

The second important contrast with `goto` is the lack of a comprehensive
alternative to pointers. Structured control flow has been thoroughly studied and
shown to address the practical needs of expressing control flow. As a
consequence, we have _good_ tools to replace `goto` within languages, and we
should use them. In contrast, Carbon programs are still expected to map onto
[von Neumann architecture](https://en.wikipedia.org/wiki/Von_Neumann_architecture)
machines and need to model the fundamental construct of a pointer to data. We
have no comprehensive alternative to solve all of the practical needs
surrounding indirect access to data on such an architecture.

So despite including pointers as one of the building blocks of Carbon, we don't
need for `goto` to be the surfaced and visible building block of Carbon's
control flow. Even if we decide to support some limited forms of `goto` to
handle edge cases where structured constructs end up suboptimal, we can still
build the foundations in a structured way without impairing the use of Carbon to
work with the low-level hardware effectively.

## Alternatives considered

### Value expression escape hatches

We could provide escape hatches for value expressions that (unsafely) take the
address or even perform mutations through a value expression. This would more
easily match patterns like `const_cast` in C++. However, there seem to be
effective ways of rewriting the code to avoid this need so this proposal
suggests not adding these escape hatches now. We will instead provide a more
limited escape exclusively for
[interop](#interop-with-const--and-const-methods). We can add more later if
experience proves this is an important pattern to support without the
contortions of manually creating a local copy (or changing to pointers).

### References in addition to pointers

The primary and most obvious alternative to the design proposed here is the one
used by C++: have _references_ in addition to pointers in the type system. This
initially allows zero-syntax modeling of L-values, which can in turn address
many use cases here much as they do in C++. Similarly, adding different kinds of
references can allow modeling more complex situations such as different lifetime
semantics.

However, this approach has two fundamental downsides. First, it would add
overall complexity to the language as references don't form a superset of the
functionality provided by pointers -- there is still no way to distinguish
between the reference and the referenced object. This results in confusion where
references are understood to be syntactic sugar over a pointer, but cannot be
treated as such in several contexts.

Second, this added complexity would reside exactly in the position of the type
system where additional safety complexity may be needed. We would like to leave
this area (pointers and references to non-local objects) as simple and minimal
as possible to ease the introduction of important safety features going forward
in Carbon.

### Syntax-free or automatic dereferencing

One way to make pointers behave very nearly the same as references without
adding complexity to the type system is to automatically dereference them in the
relevant contexts. This can, if done carefully, preserve the ability to
distinguish between the pointer and the pointed-to object while still enabling
pointers to be seamlessly used without syntactic overhead as L-values.

This proposal does not currently provide a way to dereference with zero syntax,
even on function interface boundaries. The presence of a clear level of
indirection can be an important distinction for readability. It helps surface
that an object that may appear local to the caller is in fact escaped and
referenced externally to some degree. However, it can also harm readability by
forcing code that doesn't _need_ to look different to do so anyway. In the worst
case, this can potentially interfere with being generic. Currently, Carbon
prioritizes making the distinction here visible.

Reasonable judgement calls about which direction to prefer may differ, but
Carbon's principle of
[preferring lower context sensitivity](/docs/project/principles/low_context_sensitivity.md)
leans (slightly) toward explicit dereferencing instead. That is the current
proposed direction.

It may prove desirable in the future to provide an ergonomic aid to reduce
dereferencing syntax within function bodies, but this proposal suggests
deferring that in order to better understand the extent and importance of that
use case. If and when it is considered, a direction based around a way to bind a
name to a reference expression in a pattern appears to be a promising technique.
Alternatively, there are various languages with implicit- or
automatic-dereference designs that might be considered in the future such as
Rust.

#### Syntax-free address-of

A closely related concern to syntax-free dereference is syntax-free address-of.
Here, Carbon supports one very narrow form of this: implicitly taking the
address of the implicit object parameter of member functions. Currently that is
the only place with such an implicit affordance. It is designed to be
syntactically sound to extend to other parameters, but currently that is not
planned as we don't yet have enough experience to motivate it and it may prove
surprising.

### Exclusively using references

While framed differently, this is essentially equivalent to automatic
dereferencing of pointers. The key is that it does not add both options to the
type system but addresses the syntactic differences separately and uses
different operations to distinguish between the reference and the referenced
object when necessary.

The same core arguments against automatic dereferencing applies equally to this
alternative -- this would remove the explicit visual marker for where non-local
memory is accessed and potentially mutated.

### Alternative pointer syntaxes

The syntax both for declaring a pointer type and dereferencing a pointer has
been extensively discussed in the leads question [#523].

[#523]: https://github.com/carbon-language/carbon-lang/issues/523

The primary sources of concern over a C++-based syntax:

1.  A prefix dereference operator composes poorly with postfix and infix
    operations.

    -   This is highlighted by the use of `->` for member access due to the poor
        composition with `.`: `(*pointer).member`.

2.  Even without replicating the ["inside out"][inside-out] C/C++ challenges, we
    would end up with a prefix, postfix, and infix operator `*`.

[inside-out]:
    https://faculty.cs.niu.edu/~mcmahon/CS241/Notes/reading_declarations.html

The second issue was resolved in [#520], giving us at least the flexibility to
consider `*` both for dereference and pointer type, but we still considered
numerous alternatives given the first concerns. These were discussed in detail
in a [document][pointer-syntax-doc] but the key syntax alternatives are
extracted with modern syntax below.

[pointer-syntax-doc]:
    https://docs.google.com/document/d/1gsP74fLykZBCWZKQua9VP0GnQcADCkn5-TIC1JbUvdA/edit?resourcekey=0-8MsUybUvHDCuejrzadrbMg

#### Pointer type alternative syntaxes

**Postfix `*`:**

```
var p: i32* = &i;
```

Advantages:

-   Most familiar to C/C++ programmers
    -   Doesn't collide with prefix `*` for dereference so can have that
        familiarity as well.

Disadvantages:

-   Requires the [#520] logic to resolve parsing ambiguities.
-   Visually, `*` is used for both pointers and multiplication.
-   Interacts poorly with prefix array type syntaxes.
-   Looks like a C++ pointer but likely has different semantics.

**Prefix `*`:**

```
var p: *i32 = &i;
```

Advantages:

-   Used by Rust, Go, Zig,
    [Jai](https://github.com/BSVino/JaiPrimer/blob/master/JaiPrimer.md#memory-management).
-   Go in particular has good success transitioning C/C++ programmers to this
    syntax.
-   Allows reading types left-to-right.

Disadvantages:

-   "Pointer to T" and "Dereference T" would not be distinguished grammatically,
    could particularly be a problem in template code
-   `*` used for both pointers and multiplication
-   Uncanny valley -- it is very close to C++'s syntax but not quite the same.

**Prefix `&`:**

```
var p: &i32 = &i;
```

Advantages:

-   "Type of a pointer is a pointer to the type"
-   Allows reading types left-to-right.

Disadvantages:

-   Visual ambiguity:

    ```
    let X:! type = i32;
    // Can't actually write this, but there is a visual collision between
    // whether this is the address of `X` or pointer-to-`i32`.
    var y: auto = &X;
    ```

**Prefix `^`:**

```
var p: ^i32 = &i;
```

Advantages:

-   Used by the Pascal family of languages like
    [Delphi](http://rosettacode.org/wiki/Pointers_and_references#Delphi), and a
    few others like
    [BBC_BASIC](http://rosettacode.org/wiki/Pointers_and_references#BBC_BASIC)
-   `^` looks pointy
-   `^` operator is not heavily used otherwise (as a binary op it could be
    bit-xor or raise-to-power).
-   Allows reading types left-to-right.

**`Ptr(T)`:**

```
var p: Ptr(i32) = &i;
```

Advantages:

-   It is simple and unambiguous.
-   Better supports a variety of pointer types.
-   Allows reading types left-to-right.

Disadvantages:

-   Ends up making common code verbose.
-   Not widely used in other languages.

#### Pointer dereference syntax alternatives

**Prefix `*`:**

```
*p = *p * *p + (*q)[3] + r->x;
```

Advantages:

-   Matches C/C++, and will be familiar to our initial users.
-   When _not_ composed with other operations, prefix may be easier to recognize
    visually than postfix.

Disadvantages:

-   `*` is used for both pointers and multiplication.
-   Need parens or operator `->` to resolve precedence issues when composing
    with postfix and infix operations, which is common.

**Postfix `^`:**

```
p^ = p^ * p^ + q^[3] + r^.x;
```

Advantages:

-   Generally fewer precedence issues.
-   No need for `->` operator.
-   Used by Pascal family of languages.

Disadvantages:

-   Not familiar to C/C++ users.
-   Potential conflict with `^` as xor or exponentiation.
-   Non-ergonomic: would be very frequently typed, but also a stretching motion
    for typing.

**Postfix `[]`:**

```
p[] = p[] * p[] + q[][3] + r[].x;
```

Advantages:

-   Generally fewer precedence issues.
-   No need for `->` operator.
-   Used by Swift.

Disadvantages:

-   Two characters is a bit heavier visually.
-   Possibly confusing given that plain pointers are used for non-array
    pointees.

Maybe should pair this with prefix `[]` to make pointer types? Would also need
to distinguish slice types and maybe dynamically-sized-array types.

#### Pointer syntax conclusion

Ultimately, the decision in [#523] was to keep Carbon's syntax familiar to
C++'s. The challenges that presents and advantages of changes weren't sufficient
to overcome the advantage of familiarity and for the specific challenges we have
effective solutions.

### Alternative syntaxes for locals

Several other options for declaring locals were considered, but none ended up
outweighing the proposed option on balance.

-   `let` and `let mut`, based on the Rust names
    -   While nicely non-inventive, there was both concern that `mut` didn't as
        effectively communicate the requirement of storage and concern around
        not being as obviously or unambiguously a good default with `let`. Put
        differently, the `mut` keyword feels more fitting in its use with
        mutable borrows than as a declaration introducer the way it would work
        in Carbon.
    -   There was a desire, for locals especially, to have both cases be
        similarly easy to spell. While mutation being visible does seem helpful,
        this specific approach seemed to go too far into being a tax.
-   `val` and `var`
    -   Appealing to use `val` instead of `let` given that these form value
        expression bindings. These are also likely to be taught and discussed as
        "local values" which would align well with the `val` introducer.
    -   Not completely inventive, used in Kotlin for example. But rare, overall.
    -   However, _very_ visually similar to `var` which makes code harder to
        read at a glance.
    -   Also difficult when speaking or listening to pronounce or hear the
        distinction.
-   `const` and `var`
    -   Some familiarity for C++ developers given the use of `const` there.
    -   `const` is used by other languages in a similar way to Carbon's `let`.
    -   However, very concerned about reusing `const` but having it mean
        something fairly different from C++ as a declaration introducer. For
        example, nesting a `var` pattern within `const` might be especially
        surprising.

Ultimately, the overwhelming most popular introducer for immutable local values
across popular languages that have such a distinction is `let`. Using that makes
Carbon unsurprising in the world of programming languages which is a good thing.
Using `var` to help signify the allocation of storage and given it also having
widespread usage across popular languages.

### Make `const` a postfix rather than prefix operator

Using a prefix operator for `const` and a postfix operator for `*` causes them
to require parentheses for complex nesting. We could avoid that by using a
postfix `const` and this would also allow more combinations in Carbon to be
valid in C++ with the same meaning as Carbon such as `T* const`.

This direction isn't pursued because:

-   We expect pointers-to-`const` to be significantly more common in code than
    `const`-pointers-to-non-`const`, even more-so than is already the case in
    C++. And for `pointers-to-const`, we lean towards matching more widespread
    convention of `const T*` rather than `T const*`.
    -   We are aware that some developers have a stylistic preference for "East
        `const`" which would write `T const*` in C++. However, that preference
        and advocacy for this style hasn't yet caused it to become more
        widespread or widely adopted style.
-   We don't expect the parenthesized form of `const (T*)` to be _confusing_ to
    C++ developers even though C++ doesn't allow this approach to forming a
    `const`-pointer-to-non-`const`.

This alternative was only lightly considered and can be revisited if we get
evidence that motivates a change here.

## Appendix: background, context, and use cases from C++

This appendix provides an examination of C++'s fundamental facilities in the
space involving `const` qualification, references (including R-value
references), and pointers. Beyond the expression categorization needed to
provide a complete model for the language, these use cases help inform the space
that should be covered by the proposed design.

### `const` references versus `const` itself

C++ provides overlapping but importantly separable semantic models which
interact with `const` references.

1. An _immutable view_ of a value
2. A _thread-safe interface_ of a [thread-compatible type][]

[thread-compatible type]:
    https://abseil.io/blog/20180531-regular-types#:~:text=restrictions%20or%20both,No%20concurrent%20call

Some examples of the immutable view use case are provided below. These include
`const` reference parameters and locals, as well as `const` declared local and
static objects.

```cpp
void SomeFunction(const int &id) {
  // Here `id` is an immutable view of some value provided by the caller.
}

void OtherFunction(...) {
  // ...

  const int &other_id = <some-runtime-expression>;

  // Cannot mutate `other_id` here either, it is just a view of the result of
  // `<some-runtime-expression>` above. But we can pass it along to another
  // function accepting an immutable view:
  SomeFunction(other_id);

  // We can also pass ephemeral values:
  SomeFunction(other_id + 2);

  // Or values that may be backed by read-only memory:
  static const int fixed_id = 42; SomeFunction(fixed_id);
}
```

The _immutable view_ `id` in `SomeFunction` can be thought of as requiring that
the semantics of the program be exactly the same whether it is implemented in
terms of a view of the initializing expression or a copy of that value, perhaps
in a register.

The implications of the semantic equivalence help illustrate the requirements:

-   The input value must not change while the view is visible, or else a copy
    would hide those changes.
-   The view must not be used to mutate the value, or those mutations would be
    lost if made to a copy.
-   The identity of the object must not be relevant, or else inspection of its
    address would reveal whether a copy was used.

Put differently, these restrictions make a copy valid under the
[as-if rule](https://en.cppreference.com/w/cpp/language/as_if).

The _thread-safe interface_ use case is the more prevalent use of `const` in
APIs. It is most commonly seen with code that looks like:

```cpp
class MyThreadCompatibleType {
 public:
  // ...

  int Size() const { return size; }

 private:
  int size;

  // ...
};

void SomeFunction(const MyThreadCompatibleType *thing) {
  // ....

  // Users can expect calls to `Size` here to be correct even if running on
  // multiple threads with a shared `thing`.
  int thing_size = thing->Size();

  // ...
}
```

The first can seem like a subset of the second, but this isn't really true.
There are cases where `const` works for the first use case but doesn't work well
for thread-safety:

```cpp
void SomeFunction(...) {
  // ...

  // We never want to release or re-allocate `data` and `const` makes sure that
  // doesn't happen. But the actual data is completely mutable!
  const std::unique_ptr<BigData> data = ComputeBigData();

  // ...
}
```

These two use cases can also lead to tension between shallow const and deep
const:

-   Immutability use cases will tend towards shallow(-er) const, like pointers.
-   Thread safety use cases will tend towards deep(-er) const.

### Pointers

The core of C++'s indirect access to an object stored somewhere else comes from
C and its lineage of explicit pointer types. These create an unambiguous
separate layer between the pointer object and the pointee object, and introduce
dereference syntax (both the unary `*` operator and the `->` operator).

C++ makes an important extension to this model to represent _smart pointers_ by
allowing the dereference operators to be overloaded. This can be seen across a
wide range of APIs such as `std::unique_ptr`, `std::shared_ptr`,
`std::weak_ptr`, etc. These user-defined types preserve a fundamental property
of C++ pointers: the separation between the pointer object and the pointee
object.

The distinction between pointer and pointee is made syntactically explicit in
C++ both when _dereferencing_ a pointer, and when _forming_ the pointer or
taking an object's address. These two sides can be best illustrated when
pointers are used for function parameters. The caller code must explicitly take
the address of an object to pass it to the function, and the callee code must
explicitly dereference the pointer to access the caller-provided object.

### References

C++ provides for indirection _without_ the syntactic separation of pointers:
references. Because a reference provides no syntactic distinction between the
reference and the referenced object--that is their point!--it is impossible to
refer to the reference itself in C++. This creates a number of restrictions on
their design:

-   They _must_ be initialized when declared
-   They cannot be rebound or unbound.
-   Their address cannot be taken.

References were introduced originally to enable operator overloading, but have
been extended repeatedly and as a consequence fill a wide range of use cases.
Separating these and understanding them is essential to forming a cohesive
proposal for Carbon -- that is the focus of the rest of our analysis of
references here.

#### Special but critical case of `const T&`

As mentioned above, one form of reference in C++ has unique properties:
`const T&` for some type `T`, or a _`const` reference_. The primary use for
these is also the one that motivates its unique properties: a zero-copy way to
provide an input function parameter without requiring the syntactic distinction
in the caller and callee needed when using a pointer. The intent is to safely
emulate passing by-value without the cost of copying. Provided the usage is
immutable, this emulation can safely be done with a reference and so a `const`
reference fits the bill here.

However, to make zero-copy, pass-by-value to work in practice, it must be
possible to pass a _temporary_ object. That works well with by-value parameters
after all. To make this work, C++ allows a `const` reference to bind to a
temporary. However, the rules for parameters and locals are the same in C++ and
so this would create serious lifetime bugs. This is fixed in C++ by applying
_lifetime extension_ to the temporary. The result is that `const` references are
quite different from other references, but they are also quite useful: they are
the primary tool used to fill the _immutable view_ use case of `const`.

One significant disadvantage of `const` references is that they are observably
still references. When used in function parameters, they cannot be implemented
with in-register parameters, etc. This complicates the selection of readonly
input parameter type for functions, as both using a `const` reference and a
by-value parameter force a particular form of overhead. Similarly, range based
`for` loops in C++ have to choose between a reference or value type when each
would be preferable in different situations.

#### R-value references and forwarding references

Another special set of use cases for references are R-value and forwarding
references. These are used to capture _lifetime_ information in the type system
in addition to binding a reference. By doing so, they can allow overload
resolution to select C++'s _move semantics_ when appropriate for operations.

The primary use case for move semantics in function boundaries was to model
_consuming input_ parameters. Because move semantics were being added to an
existing language and ecosystem that had evolved exclusively using copies,
modeling consumption by moving into a by-value parameter would have forced an
eager and potentially expensive copy in many cases. Adding R-value reference
parameters and overloading on them allowed code to gracefully degrade in the
absence of move semantics -- their internal implementation could minimally copy
anything non-movable. These overloads also helped reduce the total number of
moves by avoiding moving first into the parameter and then out of the parameter.
This kind of micro-optimization of moves was seen as important because some
interesting data structures, especially in the face of exception safety
guarantees, either implemented moves as copies or in ways that required
non-trivial work like memory allocation.

Using R-value references and overloading also provided a minor benefit to C++:
the lowest-level mechanics of move semantics such as move construction and
assignment easily fit into the function overloading model that already existed
for these special member functions.

These special member functions are just a special case of a more general pattern
enabled by R-value references: designing interfaces that use _lifetime
overloading_ to _detect_ whether a move would be possible and change
implementation strategy based on how they are called. Both the move constructor
and the move-assignment operator in C++ work on this principle. However, other
use cases for this design pattern are so far rare. For example, Google's C++
style
[forbids](https://google.github.io/styleguide/cppguide.html#Rvalue_references)
R-value references outside of an enumerated set of use cases, which has been
extended incrementally based on demonstrated need, and has now been stable for
some time. While overloading on lifetime is one of the allowed use cases, that
exemption was added almost four years after the initial exemption of move
constructors and move assignment operators.

#### Mutable operands to user-defined operators

C++ user-defined operators have their operands directly passed as parameters.
When these operators require _mutable operands_, references are used to avoid
the syntactic overhead and potential semantic confusion of taking their address
explicitly. This use case stems from the combined design decisions of having
operators that mutate their operands in-place and requiring the operand
expression to be directly passed as a normal function parameter.

#### User-defined dereference and indexed access syntax

C++ also allows user-defined operators that model dereference (or indirecting in
the C++ standard) and indexed access (`*` and `[]`). Because these operators
specifically model forming an L-value and because the return of the operator
definition is directly used as the expression, it is necessary to return a
reference to the already-dereferenced object. Returning a pointer would break
genericity with builtin pointers and arrays in addition to adding a very
significant syntactic overhead.

#### Member and subobject accessors

Another common use of references is in returns from member functions to provide
access to a member or subobject, whether const or mutable. This particular use
case is worth calling out specially as it has an interesting property: this is
often not a fully indirect access. Instead, it is often simply selecting a
particular member, field, or other subobject of the data structure. As a
consequence, making subsequent access transparent seems especially desirable.

However, it is worth noting that this particular use case is also an especially
common source of lifetime bugs. A classic and pervasive example can be seen when
calling such a method on a temporary object. The returned reference is almost
immediately invalid.

#### Non-null pointers

A common reason for using mutable references outside of what has already been
described is to represent _non-null pointers_ with enforcement in the type
system. Because the canonical pointer types in C++ are allowed to be null,
systems that forbid a null in the type system use references to induce any null
checks to be as early as possible. This causes a
"[shift left](https://en.wikipedia.org/wiki/Shift-left_testing)" of handling
null pointers, both moving the error closer to its cause logically and
increasing the chance of moving earlier in the development process by making it
a static property enforced at compile time.

References are imperfectly suited to modeling non-null pointers because they are
missing many of the fundamental properties of pointers such as being able to
rebind them, being able to take their address, etc. Also, references cannot be
safely made `const` in the same places that pointers can because that might
unintentionally change their semantics by allowing temporaries or extending
lifetimes.

#### Syntax-free dereference

Beyond serving as a non-null pointer, the other broad use case for references is
to remove the syntactic overhead of taking an address and dereferencing
pointers. In other words, they provide a way to have _syntax-free dereferences_.
Outside of function parameters, removing this distinction may provide a
genericity benefit, as it allows using the same syntax as would be used with
non-references. In theory code could simply use pointers everywhere, but this
would add syntactic overhead compared to local variables and references. For
immutable accesses, the syntactic overhead seems unnecessary and unhelpful.
However, having distinct syntax for _mutable_ iteration, container access, and
so on often makes code more readable.

There are several cases that have come up in the design of common data
structures where the use of distinct syntaxes immutable and mutable operations
provides clear benefit: copy-on-write containers where the costs are
dramatically different, and associative containers which need to distinguish
between looking up an element and inserting an element. This tension should be
reflected in how we design _indexed access syntax_.

Using mutable references for parameters to reduce syntactic overhead also
doesn't seem particularly compelling. For passing parameters, the caller syntax
seems to provide significant benefit to readability. When using _non-local_
objects in expressions, the fact that there is a genuine indirection into memory
seems to also have high value to readability. These syntactic differences do
make inline code and outlined code look different, but that reflects a behavior
difference in this case.
