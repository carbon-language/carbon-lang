# Default Values for Function Arguments

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
        -   [Syntactic and Semantic Parameters Example](#syntactic-and-semantic-parameters-example)
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
    -   [Interaction with Function Overloading](#interaction-with-function-overloading)
    -   [Interaction with Generics](#interaction-with-generics)
    -   [Interaction with `unused`](#interaction-with-unused)
-   [Rationale](#rationale)
-   [Alternatives considered](#alternatives-considered)

<!-- tocstop -->

## Abstract

Here we discuss the addition of default values for function arguments in Carbon, along
with limited interoperability support for the same in C++. The proposed Carbon syntax and semantics
closely emulate C++. We finish with some discussion for future work with general
support for defaults in other pattern-matching contexts such as local variables and `match`
alternatives.

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

Carbon establishes named bindings for
[function parameters](/docs/design/functions.md#named-and-positional-parameters) using the
[_tuple-pattern_](/docs/design/pattern_matching.md#tuple-patterns) contained within an outer group
of parenthesis. During pattern matching Carbon converts the _tuple-pattern_ to a tuple extended
type by way of [extended type decomposition](/docs/design/values.md#extended-type-conversions).

Similarly, Carbon converts the [_struct-pattern_](/docs/design/pattern_matching.md#struct-patterns)
to a struct extended type by the same process of extended type decomposition, and given that a
_struct-pattern_ is a valid nested _pattern_ to provide to a _tuple-pattern_ in function
parameters, it follows that the syntax for declaring default values for each should be the same,
as well as the semantics in pattern matching to provide those default values as needed.

The _tuple-pattern_ and _struct-pattern_ also appear in local binding declarations, as well as in
[_alternative-pattern_](/docs/design/pattern_matching.md#alternative-patterns) arms inside `match`
statements, so while default values for those are outside of this proposal we must ensure that
this design does not unnecessarily encumber any future effort in this space.

There are some decisions to make in the modification of Carbon syntax to support this feature,
discussed in a few [leads questions](#leads-questions) as well as a decision around default
assignment at the syntactic or semantic parameter level (see [vocabulary](#vocabulary)), tentative
answers to all of which are discussed in the proposal [details](#details).

## Background

### General Knowledge

Carbon's
[C++ interoperability goals](/docs/project/goals.md#interoperability-with-and-migration-from-existing-c-code)
have bearing on this, as this feature is a necessary part of that overall goal.

In addition, readers should have a basic understanding of
[C++ default arguments](https://en.cppreference.com/cpp/language/default_arguments)
as this proposed Carbon design largely mirrors this.

### Vocabulary

For the scope of this document we define certain terms as follows.

-   **Parameters** are the bindings between names and values used in a function declaration,
    declared in Carbon using its pattern-matching system.
-   **Arguments** are the actual values passed by a caller of the function at runtime, and are the
    scrutinee during pattern matching against the function parameter pattern.
-   **Default Argument Values**, sometimes shortened to default arguments or even defaults,
    are values for function parameters defined in the function declaration that will be provided
    by the compiler at the function call site when those arguments are not provided by the
    programmer.
-   **Syntactic Parameters** are the outermost _pattern_ forms contained in the parenthesis-bound
    _tuple-pattern_ for parameters in the function declaration syntax. They are the parameters at
    the highest level of scope inside the function body. (see example below),
-   **Semantic Parameters** are the leaves of any pattern matching statements in the parameters list,
    and are named left-to-right in the parameters list, regardless of depth of scope.

#### Syntactic and Semantic Parameters Example

Consider the following Carbon function declaration:

```carbon
fn Syn(a: i32, {b: i32, c: i32});
```

In this function call we have two syntactic parameters, the outer-most pattern declarations in the
list `a` and `{b, c}`, and 3 semantic parameters, namely `a`, `b`, and `c`.

### Documentation

#### Relevant Prior Proposals

-   [p5164](/proposals/p005164-updates-to-pattern-matching-for-objects.md) affirmed that
    `var` must declare a single complete object.

#### Meeting Discussions

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

#### Discord Discussions

-   [2026-07-24](https://discord.com/channels/655572317891461132/748959784815951963/1530303233942229053)
    Some questions and answers around the match process for patterns with defaults, as well as what
    to do with implicit conversions and defaults in structures.

#### Leads Questions

The work on this proposal exposed a few questions for Carbon Leads, enumerated here:

-   [#7529](https://github.com/carbon-language/carbon-lang/issues/7529) Do we need to say `= {}` on
    the last syntactic parameter default to make it entirely optional?
-   [#7530](https://github.com/carbon-language/carbon-lang/issues/7530) Which de-structured name
    binding form should accept defaults?

## Proposal

We propose adding initial support for default values for function arguments to Carbon. This will
require modification to the [_tuple-pattern_](/docs/design/pattern_matching.md#tuple-patterns) and
[_struct-pattern_](/docs/design/pattern_matching.md#struct-patterns) elements of pattern matching.

Following a _name-binding-pattern_ in either pattern, the programmer can declare a default value
for named bindings with the addition of an optional assignment symbol `=` followed by a required
_value-expression_.

In addition, we propose adding limited interoperation support with C++ for specifying default
values for function arguments. Most of the syntax and rules are designed to mirror that of function
argument default values in C++, to ease "tech transfer" for C++ developers to Carbon as well as
possibly reduce interoperability implementation costs.

For now, we limit default value expressions to `musteval`, meaning they must be able to evaluate
to a fully-defined value at compile time.

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

## Out of Scope

In order to expedite the implementation work of the C++ interoperability requirements, this
proposal is deliberately scoped to focus on default values for function arguments,. What follows is
a non-exhaustive list of elements we will not consider for change here:

-   Function overloading
-   Initializer expressions that are not `musteval`
-   Default argument values that have `ref` bindings
-   Defaults for function positional parameters
-   Defaults for variadic functions
-   Implicit conversions during pattern matching
-   Default values for local bindings
-   Default values for `match` alternative arms

## Details

This proposal evolved out of the goal of providing interoperability with C++ code expecting default
values for function arguments, including with function overloading, and as such, we've designed the
proposed syntax changes to be largely unsurprising to C++ programmers.

Both _struct-pattern_ and _tuple-pattern_ accept nesting of each other, and so both will need to be
modified to accept defaults in this context.

Like in C++, defaults can be specified in a Carbon function declaration and then do not need to be
repeated in a function definition, but if they are repeated they must be identical to those in the
declaration.

### Tuple Patterns

Function parameters are declared in a _tuple-pattern_ surrounded by parenthesis. Defaults may be
specified as an optional addition for any, all, or none of the named bindings in a tuple in the
function parameters list, at any level of nesting.

#### Right-To-Left Value Elision

Default values for _tuple-pattern_ elements work similarly to C++ function argument default in that
while default values may be provided for any parameter, the callee may only elide arguments in
right-to-left fashion, avoiding the ambiguity of the matching the callee-provided values to the
elided parameters in unspecified order. For example:

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

#### Pattern Matching

During the arity check phase of tuple pattern matching, we can any missing elements to the
scrutinee, drawing from the extended type of the defaults provided in the function parameter
pattern. Any type or other match issues created by these additions will be caught by further stages
in the match process.

### Struct Patterns

It is possible to nest a _struct-pattern_ inside of a _tuple-pattern_ in function parameter lists,
so we must also extend default values to name bindings inside of the _struct-pattern_. Since the
syntax is a bit more complex we have a leads question
[#7530](https://github.com/carbon-language/carbon-lang/issues/7530) about the syntax of the
"long form" of mapping structure members to binding names.

The current proposed syntax for both short and long form is to use the same optional assignment `=`
and value pair as proposed for _tuple-pattern_, for example:

```carbon
// The "long" form, allowing member and binding names to differ
fn Long(index: i32 = 0, {.key = k: i32 = 0, .value = v: i32 = 0} = {});

// The "short" form, keeping member and binding names the same
let Short(index: i32 = 0, {key: i32 = 0, value: i32 = 0} = {});
```

There's a pending leads question [#7529](https://github.com/carbon-language/carbon-lang/issues/7529)
about the trailing `= {}` on both of these examples.

#### Out-Of-Order Elision

In a _struct-pattern_, the right-to-left elision requirements don't apply, as each member has an
associated name, thus resolving the order ambiguity. This means that any struct member binding that
has a provided default may be elided, including all members and the entire structure itself, if
defaults are specified for all named members.

```carbon
Long();  // OK, index = 0, key = 0, value = 0
Short(1, {.value = 3});  // OK, index = 1, k = 0, v = 3
```

#### Interaction Between Defaults and Unspecified Structure Members

The _struct-pattern_ allows for the addition of a `, _` as the last element in a struct, indicating
that the pattern should match if the struct contains members with names other than those specified
in the pattern. For example:

```carbon
fn Flexible(index: i32 = 0, {key: i32 = 0, value: i32 = 0, _} = {});

Short(1, {.something_else = 42});  // ERROR, no matching member `something_else`
Flexible(1, {.something_else = 42});  // OK, index = 1, key = 0, value = 0
```

This can combine with the default values support to create a range of different matching effects,
which is in part the subject of leads question
[#7529](https://github.com/carbon-language/carbon-lang/issues/7529).

Assuming we require the `= {}` to indicate that the specification of the structure itself is
optional, these are examples of some of the possible combinations:

```carbon
// Requires a struct with exactly 2 members, named `key` and `value`.
fn D(index: i32, {key: i32, value: i32});

// Requires a struct with 2 or more members, two members must be named `key` and `value`.
fn E(index: i32, {key: i32, value: i32, _});

// Requires a struct with zero to two members, names must match `key` and `value`.
fn F(index: i32, {key: i32 = 0, value: i32 = 0});

// Requires a struct with any number of members and names.
fn G(index: i32, {key: i32 = 0, value: i32 = 0, _});

// Invalid syntax, default values are not specified for `key` or `value`.
fn H(index: i32, {key: i32, value: i32} = {});  // ERROR

// Invalid syntax, default values are not specified for `key` or `value`.
fn I(index: i32, {key: i32, value: i32, _} = {});  // ERROR

// Accepts an optional struct with zero to two members, names must match `key` and `value`
fn J(index: i32, {key: i32 = 0, value: i32 = 0} = {});

// Accepts an optional struct with any numbers of members and names.
fn K(index: i32, {key: i32 = 0, value: i32 = 0, _} = {});
```

#### Pattern Matching

For _struct-pattern_, the compiler will insert the extended type for default values provided by the
pattern for any member names that are missing from the scrutinee. As with _tuple-pattern_, we expect
the rest of the pattern matching process to proceed as designed.

### Interaction with Function Overloading

We acknowledge that support for default values, with the accompanying changes in the pattern
matching system, is adding a form of function overloading support to Carbon. However, the function
overloading system in Carbon is under active design and development at this time. Also, we are
intentionally limiting the scope of this proposal to default values for function parameters for the
sake of expedience. Therefore our hope is that this proposal is minimally disruptive to the overall
effort.

We will add further content here as needed to clarify the exact relationship between these two
efforts, but it didn't seem prudent to block the forward motion of this proposal on completely
specifying the interaction between these two moving components.

### Interaction with Generics

Default values should work as expected with generics, affording the programmer the same type
programming and flexibility they have in the rest of a generic programming. The `must-eval`
requirement still applies, meaning that, after monomorphization, the default value bound to each
name must be computable at compile time.

### Interaction with `unused`

Providing a default value for a function parameter marked with `unused` sends a mixed message
about the intent of the author, and we propose issuing a diagnostic when detecting this state.

There is a possible future for the `unused` keyword where, because of generic code, the meaning of
`unused` may evolve from "must be unused" to "may be unused." For example, certain specializations
of generic code may not use a certain parameter while others may not. In that situation, we'll
revisit the logic around issuing a diagnostic for `unused` defaults, perhaps by not issuing the
diagnostic for generic code.

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

**TBD** leads questions resolution, further feedback on this proposal.
