# Default Values for Struct and Tuple patterns

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
    -   [Tuple Pattern Default Values](#tuple-pattern-default-values)
        -   [Right-To-Left Value Elision](#right-to-left-value-elision)
    -   [Struct Patterns](#struct-patterns)
        -   [Out-Of-Order Elision](#out-of-order-elision)
    -   [Use Cases For Default Values](#use-cases-for-default-values)
        -   [Default Argument Values for Functions](#default-argument-values-for-functions)
        -   [Default Values for Local Bindings](#default-values-for-local-bindings)
        -   [Alternative Patterns in Match Statements](#alternative-patterns-in-match-statements)
    -   [Interaction with Generics](#interaction-with-generics)
-   [Out of Scope](#out-of-scope)
-   [Details](#details)
    -   [Pattern Matching Changes](#pattern-matching-changes)
        -   [Implicit Conversions in Pattern Matching](#implicit-conversions-in-pattern-matching)
    -   [Generics](#generics)
    -   [Interaction with `unused`](#interaction-with-unused)
-   [Rationale](#rationale)
-   [Alternatives considered](#alternatives-considered)

<!-- tocstop -->

## Abstract

We propose adding initial support for default values for
[_struct-pattern_](/docs/design/pattern_matching.md#struct-patterns) and
[_tuple-pattern_](/docs/design/pattern_matching.md#tuple-patterns) to Carbon. The intent is to
allow the user to specify default values anywhere a `var` pattern would be accepted (per
[p5164](/proposals/p005164-updates-to-pattern-matching-for-objects.md)), with the same
limitations on nesting.

We are specifically focused on changes to named parameters for functions, local
declarations, and for certain cases in `match` alternative patterns in generic code.

In addition, we propose adding limited interoperation support with C++ for specifying default
values for function arguments. Most of the syntax and rules are designed to mirror that of function
argument default values in C++, to ease "tech transfer" for C++ developers to Carbon as well as
possibly reduce interoperability implementation costs.

For now, we limit default value expressions to `musteval`, meaning they must be able to evaluate
to a fully-defined value at compile time.

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
includes the need for support for default arguments.

On the Carbon side, [function parameters](/docs/design/functions.md#named-and-positional-parameters)
are established using the [_tuple-pattern_](/docs/design/pattern_matching.md#tuple-patterns). The
_tuple-pattern_ is converted to a tuple extended type by way of
[extended type decomposition](/docs/design/values.md#extended-type-conversions).

Similarly, the [_struct-pattern_](/docs/design/pattern_matching.md#struct-patterns) is converted to
a struct extended type by the same process of extended type decomposition, and also given that a
_struct-pattern_ is a valid and likely nested _pattern_ to provide to a _tuple-pattern_ in function
parameters, it follows that the syntax for declaring default values for each should be the same.

The _tuple-pattern_ also appears in the
[_alternative-pattern_](/docs/design/pattern_matching.md#alternative-patterns) inside `match`
statements, so whatever changes to _tuple-pattern_ we propose must also be commensurate with this
usage, and the appropriate usage of default values in this context must be clarified.

There are some decisions to make in the modification of Carbon syntax to support this feature,
which have generated a few [leads questions](#leads-questions) as well as a decision around default
assignment at the syntactic or semantic parameter level (see [vocabulary](#vocabulary)), answers
to all of which are proposed in the [proposal body](#proposal).

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

-   **Parameters** are the bindings between names and values used in a function declaration.
-   **Arguments** are the actual values passed by a caller of the function at runtime.
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
    and syntactic parameters, proposal of leads questions
    [#7529](https://github.com/carbon-language/carbon-lang/issues/7529) and
    [#7530](https://github.com/carbon-language/carbon-lang/issues/7530)
-   [2026-07-20](https://docs.google.com/document/d/1mjllGO3ZCL4qGt9uJHUtcxKoHAGEY7Y999ie4EtBWB8/edit?tab=t.k8f9m2vciixb#heading=h.dyf1sdo924kx):
    Exploration of _struct-pattern_ changes and an example of a match _alternative-pattern_ in
    generic code where default values are useful.

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

We propose language syntax and semantic support for providing default values in named bindings for
any occurrence of _tuple-pattern_ and _struct-pattern_ where a `var` statement would be acceptable,
per [p5164](/proposals/p005164-updates-to-pattern-matching-for-objects.md).

### Tuple Pattern Default Values

Following a _name-binding-pattern_ in the _tuple-pattern_ of the function parameter declaration, the
programmer can declare a default value for named bindings with the addition of an optional
assignment symbol `=` followed by a required _value-expression_.

Here's a simple example using the _tuple-pattern_ to declare default values for parameters in a
function definition:

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

Additionally, just like in C++, defaults can be specified in a Carbon function declaration and then
do not need to be repeated in a function definition, but if they are repeated they must be identical
to those in the declaration.

Here's another example to illustrate that default values in _tuple-pattern_ also work with the
same elision rules in local bindings:

```carbon
let (x: i32, y: i32 = 0) = (1);  // OK, x = 1, y = 0
let (a: i32 = 0, b: i32) = (2);  // ERROR, rightmost element can't be elided as it has no default
```

### Struct Patterns

We also extend default values to name bindings inside of the _struct-pattern_, but since the syntax
is a bit more complex we have a leads question
[#7530](https://github.com/carbon-language/carbon-lang/issues/7530) about the syntax of the
"long form" of mapping structure members to binding names.

The current proposed syntax for both short and long form is to use the same optional assignment `=`
and value pair as proposed for _tuple-pattern_, e.g.:

```carbon
// The "long" form, allowing member and binding names to differ
let {.key = k: i32 = 0, .value = v: i32 = 0} = {};
// The "short" form, keeping member and binding names teh same
let {key: i32 = 0, value: i32 = 0} = {};
```

There's a pending leads question [#7529](https://github.com/carbon-language/carbon-lang/issues/7529)
about the trailing `= {}` on both of these examples.

#### Out-Of-Order Elision

In a _struct-pattern_, the right-to-left elision requirements don't apply, as each member has an
associated name, thus resolving the order ambiguity. This means that any struct member binding that
has a provided default may be elided.

```carbon
let {z: i32 = 1, w: i32 = 0} = {};  // OK, z = 1, w = 0
let {i: i32 = 2, j: i32 = 4} = {.j = 3};  // OK, i = 2, j = 4
```

#### Interaction Between Defaults and Unspecified Members

The _struct-pattern_ allows for the addition of a `, _` as the last element in a struct, indicating
that the pattern should match if the struct contains members with names other than those specified
in the pattern. For example:

```carbon
let {x: i32, y: i32} = {.x = 4, .y = 5, .z = 6};  // ERROR, no matching member named `z`
let {x: i32, y: i32, _} = {.x = 4, .y = 5, .z = 6};  // OK
```

This can combine with the default values support to create a range of different matching effects,
all of which are enumerated with examples in the proposal details.

### Use Cases For Default Values

As previously stated, the intent of this proposal is to allow default values anywhere a `var`
statement would be appropriate. This includes function parameters, local bindings, and
(with some limitations) as an _alternative-pattern_ in `match` statements.

#### Default Argument Values for Functions

This proposal evolved out of the goal of providing interoperability with C++ code expecting default
values for function arguments, including with function overloading, and as such, the proposed syntax
changes to _tuple-pattern_ are designed to be largely unsurprising to C++ programmers.

Both _struct-pattern_ and _tuple-pattern_ accept nesting of 

#### Default Values for Local Bindings

#### Alternative Patterns in Match Statements

### Interaction with Generics

Default values should work as expected with generics, affording the programmer the same type
programming and flexibility they have in the rest of a generic programming. The `must-eval`
requirement still applies, meaning that after monomorphization the default value bound to each name
must be computable at compile time.

## Out of Scope

-   Function overloading
-   Initializer expressions that are not `musteval`
-   Default argument values for function parameters that have `ref` bindings
-   Positional parameters - there's nowhere to specify the default values
-   Defaults for variadic functions

Generics? We need to decide whether or not you can `deduce` from a default argument.
Constrain `T` to have the `Default` interface, then you can call `T()` as a default.

Re: the `= =` problem
Is there a structural change to the syntax that would make this more clear?
Can we switch the order - this is based on the assumption that binding of a member name
to a different name is going to be less frequent than the specification of default

## Details

**FIXME** OK I need help with talking about the specific modifications to the parser, because of how vague
_tuple-pattern_ and _struct-pattern_ seem to be.

### Pattern Matching Changes

-   Tuple pattern matching gets a modification to the arity check to provide defaults for elements
    missing at the end from the scrutinee
-   Struct pattern matching gets a modification to the members check to provide defaults for members
    missing (regardless of order) from the scrutinee

#### Implicit Conversions in Pattern Matching

**FIXME**:
Summarize [2026-07-24](https://discord.com/channels/655572317891461132/748959784815951963/1530303233942229053)

### Generics

### Interaction with `unused`

## Rationale

TODO: How does this proposal effectively advance Carbon's goals? Rather than
re-stating the full motivation, this should connect that motivation back to
Carbon's stated goals and principles. This may evolve during review. Use links
to appropriate sections of [`/docs/project/goals.md`](/docs/project/goals.md),
and/or to documents in [`/docs/project/principles`](/docs/project/principles).
For example:

-   [Community and culture](/docs/project/goals.md#community-and-culture)
-   [Language tools and ecosystem](/docs/project/goals.md#language-tools-and-ecosystem)
-   [Performance-critical software](/docs/project/goals.md#performance-critical-software)
-   [Software and language evolution](/docs/project/goals.md#software-and-language-evolution)
-   [Code that is easy to read, understand, and write](/docs/project/goals.md#code-that-is-easy-to-read-understand-and-write)
-   [Practical safety and testing mechanisms](/docs/project/goals.md#practical-safety-and-testing-mechanisms)
-   [Fast and scalable development](/docs/project/goals.md#fast-and-scalable-development)
-   [Modern OS platforms, hardware architectures, and environments](/docs/project/goals.md#modern-os-platforms-hardware-architectures-and-environments)
-   [Interoperability with and migration from existing C++ code](/docs/project/goals.md#interoperability-with-and-migration-from-existing-c-code)

## Alternatives considered

-   We first tried to narrow this proposal to only look at function parameter value defaults,
    but it quickly became clear that because function parameters use the overall pattern
    syntax and semantics, it would be better to generalize this to patterns in general.
