# Updating `self` syntax and adding `static` member variables

<!--
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

[Pull request](https://github.com/carbon-language/carbon-lang/pull/7016)

<!-- toc -->

## Table of contents

-   [Abstract](#abstract)
-   [Problem](#problem)
-   [Background](#background)
-   [Proposal](#proposal)
-   [Details](#details)
    -   [Member variables](#member-variables)
    -   [Member functions](#member-functions)
        -   [Calling without method syntax](#calling-without-method-syntax)
-   [Rationale](#rationale)
-   [Alternatives considered](#alternatives-considered)
    -   [Don't put `self` in either parameter list](#dont-put-self-in-either-parameter-list)
    -   [`self` syntax in the deduced parameter list `[]`](#self-syntax-in-the-deduced-parameter-list-)
    -   [`class` modifier for non-instance member variables and functions](#class-modifier-for-non-instance-member-variables-and-functions)
    -   [Alternative keywords for non-instance member variables](#alternative-keywords-for-non-instance-member-variables)
        -   [`shared`](#shared)
        -   [`global`](#global)
    -   [`static` for non-instance member functions](#static-for-non-instance-member-functions)
    -   [`static` for package- and namespace-scope variables](#static-for-package--and-namespace-scope-variables)
    -   [Distinct `method` introducer](#distinct-method-introducer)

<!-- tocstop -->

## Abstract

Update the syntax for class (and interface/`impl`) methods to move `self` into
the parameter parentheses `()` and make the type in its binding optional
(defaulting to `Self`). Introduce the `static` keyword for non-instance member
variables to indicate static storage. Reflects the decision in leads issue
[#6931](https://github.com/carbon-language/carbon-lang/issues/6931).

## Problem

The placement of `self` in the parameter list has been a repeated source of
debate around the syntax design of Carbon -- grouping it with deduced parameters
like types is surprising for some readers and nearly unprecedented in modern
programming languages. As this is an especially pervasive and impactful aspect
of Carbon's syntax given the prevalence of methods, it is important to revisit
the syntax here and make a durable decision on how to approach this part of
Carbon's syntax.

There is also a problem that we don't have a syntax for representing
_non_-instance data members. Especially as we look at more C++ interop and more
migrations from Carbon to C++ it is important to have a clear and idiomatic
syntax for these constructs.

Key questions this proposal should address include:

-   Where to place `self` (in `[]` versus `()`).
-   How to distinguish instance methods from non-instance functions.
-   How to declare non-instance member variables without overloading keywords or
    creating confusion in interfaces vs classes.

## Background

Leads discussed various syntax options for `self` placement and non-instance
members in issue
[#6931](https://github.com/carbon-language/carbon-lang/issues/6931).

## Proposal

We propose the following updates to the syntax for class members:

1.  **Move `self` to the `()` parameter list**: Instance methods declare `self`
    as the first parameter in the explicit parameter list. This aligns methods
    with a model where method calls are conceptually sugar for function calls
    with an explicit object argument.
2.  **Make `self` type optional**: Optionally allow omitting the type of `self`,
    which results in the type being exactly `Self`. Example:
    `fn MyMethod(self)`.
3.  **Use `static` for non-instance data members**: Use `static var` for data
    members that are part of the type rather than part of instances of the type.
4.  **No keyword for non-instance member functions**: Functions without a `self`
    parameter declared within a `class` are considered non-instance member
    functions without any extra syntax.

Treating methods as functions with an explicit `self` parameter means they can
be called using standard function call syntax (for example,
`MyClass.Method(obj)`), not just method call syntax (`obj.Method()`).

While methods can be invoked as regular functions, method syntax (`obj.Method`)
continues to form a "bound member function" that adapts `obj`. This allows the
call to be resolved appropriately even when `obj` requires conversion to match
the type of `self`. Similarly, we expect `obj.(Class.Method)(args)` to follow
the same model. Fundamentally, our goal is that this does not change the
[instance binding model](/docs/design/expressions/member_access.md#instance-binding).

## Details

### Member variables

_Member variables_ are any `var`s declared within a class context (or in the
future, potentially an interface or `impl` context).

Instance member variables, or _fields_, are unchanged in the current design.

This proposal adds support for non-instance member variables, or _static member
variables_, using the `static` modifier keyword on the `var` declaration. Static
member variables are associated with the type itself rather than instances of
the type, and have static storage duration.

```carbon
class Widget {
  // Non-instance member variable.
  static var count: i32;

  // Non-instance member function.
  fn ResetCount() { count = 0; }
}
```

This does not apply to package- and namespace-scope variables (if we support
them) because static storage duration is the default in those contexts. It is
currently limited to classes, but if we allow variables with static storage
duration in other contexts that don't have a strong default of that storage,
like interfaces and `impl` declarations, they should use the same syntax.

### Member functions

We propose retiring the terminology "class function". It is confusing when used
in contexts like an interface (which is not a class) or where "class" implies
other semantics. Instead, we consider all functions declared within a class,
interface, or `impl` to be _member functions_. Member functions are either
_methods_ (taking a `self` parameter) or _non-instance member functions_.

In practice, we most often are referring to member functions broadly, regardless
of whether they are methods or not, or referring very specifically to methods.
Because referring to the broad set happens much more than referring narrowly to
non-instance member functions, we prioritize terminology that builds on a simple
broad term. We also don't expect this to compound with other adjectives like
"associated" as those will already imply "member", and so would be _associated
functions_ for the broad group and _associated methods_ for the common narrow
group.

Non-instance member functions don't use any special syntax. They don't take a
`self` parameter and so are regular functions. For example:

```carbon
class Point {
  var x: i32;
  var y: i32;

  // No `self` parameter, so it's a regular, non-instance function.
  fn Create() -> Self {
    return {.x = 0, .y = 0};
  }
}

fn F(p: Point) {
  var p2: Point = Point.Create();
}
```

Instance member functions, or _methods_, declare `self` as the first parameter
in the `()` list. Specifying the type of `self` is optional and defaults to
`: Self`.

```carbon
class Point {
  var x: i32;
  var y: i32;

  // Explicit type in the `self` binding.
  fn GetX(self: Point) -> i32 { return self.x; }

  // Omitted type in the `self` binding (defaults to Point).
  fn GetY(self) -> i32 { return self.y; }

  // By-reference `self` binding.
  fn SetX(ref self, new_x: i32) { self.x = new_x; }
}
```

Moving `self` into the parentheses reinforces the conceptual model that method
calls are sugar for function calls with an explicit object argument.
Specifically, `obj.Method(args...)` is equivalent to
`Type.Method(obj, args...)`.

This adopts a **partial-application model** which unifies methods and functions.
A function is an "instance method" if and only if its first parameter is named
`self`. It can always be called directly with the first argument passed to the
`self` parameter. When `Method` names a function with a `self` parameter, the
syntax `obj.Method` is just syntactic sugar for partially applying the `obj` to
the `self` parameter, which can then be called as a function by passing
arguments for the remaining parameters.

For non-instance member functions, the absence of a `self` parameter is
sufficient to distinguish them from methods.

#### Calling without method syntax

Because methods are modeled as functions with an explicit `self` parameter, they
can be called as ordinary functions using their qualified names, with the `self`
argument as the first element of the explicit parameter list:

```carbon
fn F(p: Point) {
  // Standard method call syntax:
  var x1: i32 = p.GetX();

  // Called as an ordinary function:
  var x2: i32 = Point.GetX(p);
}
```

## Rationale

This proposal effectively advances Carbon's goals by focusing on:

-   [Code that is easy to read, understand, and write](/docs/project/goals.md#code-that-is-easy-to-read-understand-and-write):
    Moving `self` to the explicit parameter list within parentheses aligns
    Carbon's syntax with essentially all modern programming languages that use
    an explicit object parameter, notably including C++23 itself, Python, and
    Rust. It also reduces the visual cost of non-generic methods by removing the
    need for an additional delimiter kind in their declaration. Making the
    `: Self` part optional further reduces clutter in the most common cases and
    matches the syntax used in other languages like Rust.
-   [Software and language evolution](/docs/project/goals.md#software-and-language-evolution):
    Focusing `static` on storage (member variables) provides a crisp definition
    that works unambiguously across classes, interfaces, and `impl` blocks.

## Alternatives considered

### Don't put `self` in either parameter list

We considered two approaches to modeling `self` that separated it from
parameters of any kind, implicit or explicit.

1.  We could place `self` in an independent position with a different set of
    delimiters such as:

        ```carbon
        class Point {
          var x: i32;
          var y: i32;

          // `self` as a separate component, likely in between the implicit and
          // explicit parameter lists.
          fn Create[T:! type]<self>(x: T) -> Self {
            return {.x = x as i32, .y = 0};
          }
        }
        ```

2.  We could make `self` implicit, similar to C++.

We have generally been happy with `self` being explicit instead of implicit and
so to an extent we didn't deeply consider (2) as that wasn't part of the problem
we set out to solve. We're still comfortable with the rationale about this
aspect of the syntax from
[#722](/proposals/p000722-nominal-classes-and-methods.md#full-receiver-type).

We did consider (1) but were unable to find a syntax that felt compelling.
Taking a new balanced delimiter is an especially difficult and expensive choice
for the syntax. And without that, most syntaxes felt verbose for something that
we expect to be extremely common and a natural "default".

### `self` syntax in the deduced parameter list `[]`

We considered placing `self` in the deduced parameter list (for example,
`fn F[self: Self](...)`), as this has been the historical syntax in Carbon.

However, users strongly associate the implicit parameter list in `[]`s with
deduced, compile-time parameters, but the `self` parameter is a _runtime_
argument passed explicitly by the caller. While we have syntactic distinctions
such as the `!` and the `self` keyword, the contextual collision still adds some
cognitive load for some readers. We do imagine potential runtime implicit
parameters in the future, but these are expected to be rare enough to be easier
to accommodate and to leverage distinguishing syntax.

In contrast to these challenges, putting `self` in the explicit parameter list
`()` aligns with the semantics of `self` as being an otherwise-normal parameter.

Last but not least, most other languages with explicit object parameters put
them at the start of the parameter list inside `()`s (see for example Python and
Rust). Being consistent with how object parameters are modeled in other
languages is expected to reduce confusion and surprise for readers of Carbon.
This is especially nice here, as we're asking C++ programmers to move from
_implicit_ object parameters to _explicit_ object parameters -- aligning with
other languages' explicit object parameter syntax hopefully minimizes that cost.

There was some concern that removing `self` from the `[]`s would result in
readers assuming that the `[]`s can _only_ contain type parameters, similar to
how Python works. However, that concern didn't carry as much weight as the other
considerations for the leads.

Ultimately, the leads preferred the placement in the `()`s and the associated
tradeoffs with that syntax.

### `class` modifier for non-instance member variables and functions

We considered using `class var` and `class fn` to mark non-instance member
variables and functions. However, `class` is already an introducer in Carbon.
Using it as a modifier adds an overloaded meaning. This meaning would also
diverge from usage in other languages. For example, Swift uses `class` versus
`static` to distinguish between class and struct methods, and has complex
virtual dispatch rules that aren't part of the Carbon design. These members also
appear in interfaces and `impl` blocks where the `class` keyword would be mildly
surprising.

### Alternative keywords for non-instance member variables

The decision to use `static` at all in Carbon is relatively controversial. It
has a history in C++ of being used for a sprawling and seemingly ever-growing
set of things, in addition to the already subtle meanings inherited from C.

We chose `static` for non-instance member variables because of widely Carbon's
major peer languages (notably C++, Rust, Swift, and Java) use the keyword
`static` to mark class member variables with static storage duration. Reusing
this spelling avoids a surprising divergence from the languages our users are
most likely to be familiar with, and provides a clear, specific meaning focused
on storage duration. This aligns with common definitions of
[static variables](https://en.wikipedia.org/wiki/Static_variable) focusing on
lifetime contrasted with scope.

A key aspect of the leads being comfortable with this direction was having the
storage implication of the keyword be fundamental to _any_ usage of it rather
than diluting that meaning with uses in other contexts.

Given that the `static` keyword did come with all of these concerns, we
considered a number of alternatives and the specific rationale for not selecting
those is recorded here.

#### `shared`

We considered using `shared var` to mark non-instance member variables. However,
this term is heavily associated with concurrency and thread safety (for example,
shared ownership, shared pointers). Reserving it for class members would collide
with its use for future concurrency features (see
[concurrency control design](https://docs.google.com/document/d/1WVWcmJdVBlapza_kPj2l3mOO-yw_hNXpb2u-Ren-I5M/)
and
[shared memory in CUDA](https://developer.nvidia.com/blog/using-shared-memory-cuda-cc/)).

The risk of confusion outweighed the benefits of identifying member variables
that are conceptually shared across instances.

#### `global`

We considered using `global var` to mark non-instance member variables. However,
`global` is most widely used to refer to _scope_ (global scope), whereas these
members are explicitly scoped to a type (or potentially an interface). The most
common distinction is that of "global" implicating scope or access, while
"static" more consistently describes the storage model.

### `static` for non-instance member functions

We considered applying `static` to both member variables and functions (for
example, `static fn F()`), for consistency with C++, Java, C#, and Swift.
However, functions do not have instances or storage in the same way data members
do. The absence of a `self` parameter is sufficient to distinguish instance
methods from non-instance functions, making the keyword redundant for functions.

Reusing `static` here would stretch or dilute its meaning in Carbon. Not using
it lets us keep the meaning narrow, and very specific to storage. It also helps
emphasize the underlying unification that we have for member functions but not
for data members: methods _are_ member functions, and can even be called as such
when a viable object is explicitly passed to the `self` parameter. The only
thing special about methods is that they _additionally_ allow the method call
syntax.

### `static` for package- and namespace-scope variables

We considered whether to require or allow the `static` keyword on package- and
namespace-scope variables (if we support them).

The storage of these variables should be described as "static" the same way as
it is for static member variables. However, we propose omitting the keyword in
those contexts because static storage is the strong default. In contrast, the
strong default for class members is instance storage, so a keyword is needed to
opt out of that default.

### Distinct `method` introducer

We considered using a distinct introducer like `method F()` instead of
`fn F(self)`. However, we prefer a uniform `fn` introducer for all functions,
relying on the presence of the `self` parameter to identify methods. This
provides greater consistency across all different kinds of functions, including
lambdas. It also embraces the unification of methods and functions.
