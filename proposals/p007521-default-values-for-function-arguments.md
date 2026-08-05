# Default values for function arguments

<!--
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

[Pull request](https://github.com/carbon-language/carbon-lang/pull/7521)

<!-- toc -->

## Table of contents

-   [Abstract](#abstract)
-   [Problem](#problem)
-   [Background](#background)
    -   [General Knowledge](#general-knowledge)
    -   [Vocabulary](#vocabulary)
    -   [Documentation](#documentation)
        -   [Relevant Prior Proposals](#relevant-prior-proposals)
        -   [Meeting Discussions](#meeting-discussions)
        -   [Discord Discussions](#discord-discussions)
        -   [Leads Questions](#leads-questions)
-   [Proposal](#proposal)
-   [Out of Scope](#out-of-scope)
-   [Details](#details)
    -   [Tuple Patterns](#tuple-patterns)
        -   [Right-To-Left Value Elision](#right-to-left-value-elision)
        -   [Pattern Matching](#pattern-matching)
    -   [Struct Patterns](#struct-patterns)
        -   [Out-Of-Order Elision](#out-of-order-elision)
        -   [Interaction Between Defaults and Unspecified Structure Members](#interaction-between-defaults-and-unspecified-structure-members)
        -   [Pattern Matching](#pattern-matching-1)
    -   [Nesting Parameter Defaults](#nesting-parameter-defaults)
    -   [Interaction with Function Overloading](#interaction-with-function-overloading)
    -   [Interaction with Generics](#interaction-with-generics)
    -   [Interaction with `unused`](#interaction-with-unused)
-   [Rationale](#rationale)
-   [Alternatives considered](#alternatives-considered)

<!-- tocstop -->

## Abstract

Proposes adding default values for function arguments to Carbon, along with limited interoperability
support for the same in C++. The proposed syntax and semantics closely emulate C++.

Also outlines future work to provide more general support for defaults in other pattern-matching
contexts such as local variables and `match` alternatives.

## Problem

When interoperating with C++ code, much of that code will use default arguments for functions, for
example when providing default values for `a` and `b` in constructor arguments:

```cpp
class C {
 public:
  C(int a = 0, int b = 0): a_(a), b_(b) {}
 private:
  int a_;
  int b_;
};
```

In projects that are adopting Carbon, developers need to be able to write Carbon objects that
fluently emulate the API of their C++ counterparts. This fluent emulation of C++ necessarily
includes the need for support for default argument values.

In Carbon, a [function parameter list](/docs/design/functions.md#named-and-positional-parameters) is
actually a [_tuple-pattern_](/docs/design/pattern_matching.md#tuple-patterns), and an argument list
is actually a tuple literal expression. When you call a function, the argument tuple is matched
against the parameter pattern. Like any _tuple-pattern_, the parameter list can contain arbitrarily
complex subpatterns, not just name bindings. So we need to design how they fit into the pattern
syntax, and how they affect pattern matching.

A [_struct-pattern_](/docs/design/pattern_matching.md#struct-patterns) is a valid nested _pattern_
to provide to a _tuple-pattern_ in function parameters, so it seems reasonable that the syntax for
declaring default values for each should be the same, as well as the semantics in pattern matching
to provide those default values as needed.

The _tuple-pattern_ and _struct-pattern_ also appear in local binding declarations, as well as in
[_alternative-pattern_](/docs/design/pattern_matching.md#alternative-patterns) arms inside `match`
statements, so while default values for those are outside of this proposal we must ensure that
this design does not unnecessarily encumber any future effort in this space.

There are some decisions to make in the modification of Carbon syntax to support this feature,
discussed in a few [leads questions](#leads-questions) as well as a decision around default
assignment at the syntactic or semantic parameter level (see [vocabulary](#vocabulary)), tentative
answers to all of which are discussed in the proposal [details](#details).

### Out of scope

In order to expedite the implementation work of the C++ interoperability requirements, this
proposal is deliberately scoped to focus on default values for function arguments. What follows is
a non-exhaustive list of elements we will not consider for change here:

-   Function overloading
-   Initializer expressions that are not concrete constants
-   Default argument values that have `ref` bindings
-   Defaults for function positional parameters
-   Defaults for variadic functions
-   Implicit conversions during pattern matching
-   Default values for local bindings, for now the compiler will reject
-   Default values for `match` alternative arms, for now the compiler will reject

## Background

-   Carbon's [C++ interoperability goals](/docs/project/goals.md#interoperability-with-and-migration-from-existing-c-code)
-   [C++ default arguments](https://en.cppreference.com/cpp/language/default_arguments)
-   [p5164](/proposals/p005164-updates-to-pattern-matching-for-objects.md) affirmed that
    `var` must declare a single complete object.
-   [p1084](/proposals/p001084-generics-details-9-forward-declarations.md) in the associated
    [PR](https://github.com/carbon-language/carbon-lang/pull/1084) added support for the
    `where _` syntax when redeclaring constants in generics.

### Vocabulary

For the scope of this document we define certain terms as follows.

-   **Parameters** are the bindings between names and values used in a function declaration,
    declared in Carbon using its pattern-matching system.
-   **Arguments** are the actual values passed by a caller of the function at runtime, and are the
    scrutinee during pattern matching against the function parameter pattern.
-   **Default Argument Values**, sometimes shortened to default arguments or even defaults,
    are values for function parameters defined in the function declaration that will be provided
    by the compiler at the function call site when those arguments are not provided by the
    developer.

### Meeting discussions

We have discussed this proposal on a few occasions at the Carbon open meetings. Links to the
minutes for each date and a short summary of the discussions follow:

-   [2026-07-16](https://docs.google.com/document/d/1mjllGO3ZCL4qGt9uJHUtcxKoHAGEY7Y999ie4EtBWB8/edit?tab=t.k8f9m2vciixb#heading=h.6n4qdhlv19kv):
    Suggestion to broaden the scope to _tuple-pattern_ and _struct-pattern_, discussion of semantic
    and syntactic parameters, proposal of leads questions.
    [#7529](https://github.com/carbon-language/carbon-lang/issues/7529) and
    [#7530](https://github.com/carbon-language/carbon-lang/issues/7530)
-   [2026-07-20](https://docs.google.com/document/d/1mjllGO3ZCL4qGt9uJHUtcxKoHAGEY7Y999ie4EtBWB8/edit?tab=t.k8f9m2vciixb#heading=h.dyf1sdo924kx):
    Exploration of _struct-pattern_ changes and an example of a match _alternative-pattern_ in
    generic code where default values are useful.
-   [2026-07-27](https://docs.google.com/document/d/1mjllGO3ZCL4qGt9uJHUtcxKoHAGEY7Y999ie4EtBWB8/edit?tab=t.k8f9m2vciixb#heading=h.6aq96lu1p4kx)
    Further discussion of semantic versus syntactic defaults support, and some feedback on an early
    draft of this proposal to consider downscoping it back to function default values only.

### Discord discussions

-   [2026-07-24](https://discord.com/channels/655572317891461132/748959784815951963/1530303233942229053)
    Some questions and answers around the match process for patterns with defaults, as well as what
    to do with implicit conversions and defaults in structures.

## Proposal

Following any element of either _struct-pattern_ or _tuple-pattern_, the programmer can declare
an optional default value with the addition of an assignment symbol `=` followed by a
_value-expression_. Note that these defaults cannot appear in `var` patterns, because the
initialization of a `var` pattern must be written in a single place, not interleaved with other
code.

In addition, we propose adding limited interoperation support with C++ for specifying default
values for function arguments. Most of the syntax and rules are designed to mirror that of function
argument default values in C++, to ease "tech transfer" for C++ developers to Carbon as well as
possibly reduce interoperability implementation costs.

For now, default value expressions must be concrete constant expressions.

Default values are only allowed in function declarations for now. This is an intentional
limitation of the scope of this proposal for expediency, and nothing proposed here is intended to
apply, in the long term, to function declarations exclusively.

Here's a simple example declaring default values for parameters in a function definition:

```carbon
import Core library "io";

fn DefaultThree(x: i32 = 3) -> i32 {
  return x;
}

fn Main() {
  Core.Print(DefaultThree());  // `3`
  Core.Print(DefaultThree(7));  // `7`
}
```

## Details

This proposal evolved out of the goal of providing interoperability with C++ code expecting default
values for function arguments, including with function overloading, and as such, we've designed the
proposed syntax changes to be largely unsurprising to C++ developers.

Both _struct-pattern_ and _tuple-pattern_ accept nesting of each other, and so we propose
modifications to both to accept defaults in this context.

Like in C++, defaults can be specified in a Carbon function declaration or definition. Unlike in C++,
Carbon requires defaults to be specified in the first declaration. Subsequent declarations must
either repeat the defaults exactly or provide a `= _` in every place a default value was specified
in the first declaration, following the convention of `where _` for constants established by
[p1084](https://github.com/carbon-language/carbon-lang/pull/1084).

### Tuple patterns

Defaults may be specified as an optional addition for any, all, or none of the elements in a
_tuple-pattern_, at any level of nesting. However, at any particular level of nesting, if a certain
element within the _tuple-pattern_ provides a default value, the developers must provide default
values for every element to the right of that element, or the compiler will issue a diagnostic.

#### Right-to-left value elision

Default values for _tuple-pattern_ elements work similarly to C++ function argument defaults in that
the scrutinee may only elide arguments in right-to-left fashion, avoiding the ambiguity of the\
matching the scrutinee-provided values to the elided parameters in unspecified order. For example:

```carbon
fn TwoDefaults(x: i32 = 0, y: i32 = 1);

TwoDefaults();  // OK, x = 0, y = 1
TwoDefauts(1);  // OK, x = 1, y = 1
TwoDefaults(, 3);  // ERROR, value not specified for first argument
```

Note that the right-to-left elision matching process happens at the individual tuple level, meaning
that nested _tuple-pattern_ scrutinees may also elide values in right-to-left order without being
subject to the constraint of where they are nested in a higher-level containing _tuple-pattern_.
For example the following is valid:

```carbon
fn Nested(a: i32, (b: i32, c: i32 = 1), d: i32);

Nested(1, (2), 3);  // OK, a = 1, b = 2, c = 1, d = 3
```

#### Pattern matching

During the arity check phase of tuple pattern matching, we can any missing elements to the
scrutinee, drawing from the extended type of the defaults provided in the _tuple-pattern_.
Any type or other match issues created by these additions will be caught by further stages in the
match process.

### Struct patterns

We also extend default values to elements of the _struct-pattern_. Since the syntax is a bit more
complex, particularly in the case of the long form, we have a leads question
[#7530](https://github.com/carbon-language/carbon-lang/issues/7530) about this.

The current proposed syntax for both short and long form is to use the same optional assignment `=`
and value pair as proposed for _tuple-pattern_, for example:

```carbon
// The "long" form, allowing member and binding names to differ
fn Long(index: i32 = 0, {.key = k: i32 = 0, .value = v: i32 = 0} = {});

// The "short" form, keeping member and binding names the same
let Short(index: i32 = 0, {key: i32 = 0, value: i32 = 0} = {});
```

#### Out-of-order elision

In a _struct-pattern_, the right-to-left elision requirements don't apply, as each member has an
associated name, thus resolving the order ambiguity. This means that any struct element that has a
provided default may be elided, including all members and the entire structure itself, if defaults
are specified for all named members.

There's a pending leads question [#7529](https://github.com/carbon-language/carbon-lang/issues/7529)
if we should requre the trailing `= {}` to omit the structure entirely.

```carbon
Long();  // OK, index = 0, key = 0, value = 0
Short(1, {.value = 3});  // OK, index = 1, key = 0, value = 3
```

#### Interaction between defaults and unspecified structure members

The _struct-pattern_ allows for the addition of a `, _` as the last element in a struct, indicating
that the scrutinee should match if the struct contains members with names other than those specified
in the pattern. For example:

```carbon
fn Flexible(index: i32 = 0, {key: i32 = 0, value: i32 = 0, _} = {});

Short(1, {.something_else = 42});  // ERROR, no matching member `something_else`
Flexible(1, {.something_else = 42});  // OK, index = 1, key = 0, value = 0
```

This can combine with the default values support to create a range of different matching effects,
which is in part the subject of leads question
[#7529](https://github.com/carbon-language/carbon-lang/issues/7529).

Here are examples of the possible combinations:

```carbon
// Requires a struct with exactly 2 members, named `key` and `value`.
fn D(index: i32, {key: i32, value: i32});

// Requires a struct with 2 or more members, two members must be named `key` and `value`.
fn E(index: i32, {key: i32, value: i32, _});

// Accepts a struct with zero to two members, names must match `key` and `value`.
fn F(index: i32, {key: i32 = 0, value: i32 = 0});

// Accepts a struct with any number of members and names.
fn G(index: i32, {key: i32 = 0, value: i32 = 0, _});
```

#### Pattern matching

For _struct-pattern_, the compiler will supply the default values provided by the pattern for any
member names that are missing from the scrutinee. As with _tuple-pattern_, we expect the rest of the
pattern matching process to proceed as designed.

### Nesting parameter defaults

Defaults supplied at different levels of nesting have different implications for pattern matching.
For example:

```carbon
// Accepts 0 or 1 tuple argument. The supplied tuple can have 0-2 elements, the struct in the
// second position can have 0-2 members but they must be named `j` and/or `k`.
fn Lowest((i: i32 = 1, {j: i32 = 2, k: i32 = 3}));

// Accepts 0 or 1 tuple argument. The supplied tuple can have 0-2 elements. The supplied struct
// must have a `j` and `k` member.
fn Middle((i: i32 = 1, {j: i32, k: i32} = {.j = 2, .k = 3}));

// Accepts 0 or 1 tuple argument. If the tuple is supplied it must be fully-formed, and the struct
// in the second position must provide a `j` and `k` member and have no other members.
fn Highest((i: i32, {j: i32, k: i32}) = (1, {.j = 2, .k = 3}));
```

The developer may supply at most one default for each "primitive" struct or tuple pattern element,
where "primitive" means that it's not a struct or tuple pattern itself. For example, we would
reject the following:

```carbon
fn F({a: i32, .b = {_} = {.x = 1}} = {.a = 1, .b = {.x = 2}});  // ERROR, two defaults for .b.x
```

### Interaction with function overloading

We acknowledge that support for default values in functions, with the accompanying changes in the
pattern matching system, is adding a form of function overloading support to Carbon. However, the
function overloading system in Carbon is under active design and development at this time. Also, we
are intentionally limiting the scope of this proposal to default values for function parameters for
the sake of expedience. Therefore our hope is that this proposal is minimally disruptive to the
overall effort.

### Interaction with generics

Default values should work as expected with generics, affording the developer the same type
programming and flexibility they have in the rest of a generic programming. After monomorphization,
the default value provided for each pattern must be computable at compile time.

### Interaction with `unused`

Although the developer may be sending a mixed message when providing a default value for an
unused parameter, we can imagine situations in code evolution where the combination of the two
are required.

One can also argue that `unused` is a property of the implementation, while defaults are more of
a property of the interface for the function.

Therefore we will not issue a diagnostic when detecting these two in combination.

For future work, we may want to consider the use case in C++ for unused variables in a function
declaration whose default initializers have side effects intended to run in the callee scope.

## Rationale

This proposal advances the following Carbon goals:

-   [Interoperability with and migration from existing C++ code](/docs/project/goals.md#interoperability-with-and-migration-from-existing-c-code):
    Default values for function arguments are a widely-used feature of C++, so Carbon needs some
    utility to express these to support interop.
-   [Code that is easy to read, understand, and write](/docs/project/goals.md#code-that-is-easy-to-read-understand-and-write):
    Allowing default values can increase the brevity and clarity of the calling code, by only
    specifying the arguments that are meaningful in the context of the call.
-   [Fast and scalable development](/docs/project/goals.md#fast-and-scalable-development): Support
    for default values allows some API changes to occur when adding arguments to existing methods,
    requiring no changes to existing client code beyond recompilation.

## Alternatives considered

### Defaults only at highest level of nesting

We considered an implementation of defaults only at the highest level of nesting in a
_tuple-pattern_, for the possibility it might be simpler to implement. We agreed that we preferred
defaults at any arbitrary level of nesting, and so deferred further inquiry. See the
[design details](#nesting-parameter-defaults) for more.

### Downscope to exclude struct patterns

For interoperation with C++ function default arguments, it is plausible we could only add support to
_tuple-pattern_ and skip _struct-pattern_. However, Carbon uses the patterns in other contexts, and
so we wanted to at least specify the defaults for _struct-pattern_, to pave the way for future work.

Furthermore, some felt it would be natural for developers to want to provide _struct-pattern_ as a
nested pattern in a _tuple-pattern_, so we should go ahead and provide it.

### Default support for local patterns and match alternatives

We also considered adding default support to local patterns, but this opened up a number of
questions, for example around ambiguity in implicit casts, and since our immediate need is to
support C++ interop for function defaults, we decided to leave this out of scope of this proposal.

Match alternatives also raised some questions about interactions with generic code, and since the
utility of defaults here seemed less obvious than in the other two use cases, we also decided to
exclude them from this proposal.
