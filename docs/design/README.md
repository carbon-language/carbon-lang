# Language design

<!--
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

<!-- toc -->

## Table of contents

-   [Overview](#overview)
-   [Context and disclaimer](#context-and-disclaimer)
    -   [Example code](#example-code)
-   [Basic syntax](#basic-syntax)
    -   [Code and comments](#code-and-comments)
    -   [Packages, libraries, and namespaces](#packages-libraries-and-namespaces)
    -   [Names and scopes](#names-and-scopes)
        -   [Naming conventions](#naming-conventions)
        -   [Aliases](#aliases)
        -   [Name lookup](#name-lookup)
            -   [Name lookup for common types](#name-lookup-for-common-types)
    -   [Expressions](#expressions)
    -   [Functions](#functions)
    -   [Blocks and statements](#blocks-and-statements)
    -   [Variables](#variables)
    -   [Lifetime and move semantics](#lifetime-and-move-semantics)
    -   [Control flow](#control-flow)
        -   [`if` and `else`](#if-and-else)
        -   [Loops](#loops)
            -   [`while`](#while)
            -   [`for`](#for)
            -   [`break`](#break)
            -   [`continue`](#continue)
        -   [`return`](#return)
-   [Types](#types)
    -   [Primitive types](#primitive-types)
    -   [Composite types](#composite-types)
        -   [Tuples](#tuples)
        -   [Variants](#variants)
        -   [Pointers and references](#pointers-and-references)
        -   [Arrays and slices](#arrays-and-slices)
    -   [User-defined types](#user-defined-types)
        -   [Classes](#classes)
            -   [Assignment, copying](#assignment-copying)
            -   [Member access](#member-access)
            -   [Methods](#methods)
            -   [Allocation, construction, and destruction](#allocation-construction-and-destruction)
            -   [Moving](#moving)
            -   [Comparison](#comparison)
            -   [Implicit and explicit conversion](#implicit-and-explicit-conversion)
            -   [Inline type composition](#inline-type-composition)
        -   [Unions](#unions)
-   [Names](#names)
    -   [Packages, libraries, namespaces](#packages-libraries-namespaces)
    -   [Names and scopes](#names-and-scopes-1)
    -   [Naming conventions](#naming-conventions-1)
    -   [Aliases](#aliases-1)
    -   [Name lookup](#name-lookup-1)
        -   [Name lookup for common types](#name-lookup-for-common-types-1)
    -   [Name visibility](#name-visibility)
-   [Generics](#generics)
    -   [Checked and template parameters](#checked-and-template-parameters)
    -   [Interfaces and implementations](#interfaces-and-implementations)
    -   [Combining constraints](#combining-constraints)
    -   [Generic types](#generic-types)
        -   [Types with template parameters](#types-with-template-parameters)
        -   [Generic choice types](#generic-choice-types)
    -   [Other features](#other-features)
    -   [Operator overloading](#operator-overloading)
        -   [Common type](#common-type)
-   [Bidirectional interoperability with C/C++](#bidirectional-interoperability-with-cc)
-   [Unfinished tales](#unfinished-tales)
    -   [Pattern matching as function overload resolution](#pattern-matching-as-function-overload-resolution)
    -   [Lifetime and move semantics](#lifetime-and-move-semantics-1)
    -   [Metaprogramming](#metaprogramming)
    -   [Execution abstractions](#execution-abstractions)
        -   [Abstract machine and execution model](#abstract-machine-and-execution-model)
        -   [Lambdas](#lambdas)
        -   [Co-routines](#co-routines)

<!-- tocstop -->

## Overview

This documentation describes the design of the Carbon language, and the
rationale for that design.

## Context and disclaimer

Eventually, this document hopes to provide a high-level overview of the design
of the Carbon language. It should summarize the key points across the different
aspects of the language design and link to more detailed and comprehensive
design documents to expand on specific aspects of the design. That means it
isn't and doesn't intend to be complete or stand on its own. Notably, it doesn't
attempt to provide detailed and comprehensive justification for design
decisions. Those should instead be provided by the dedicated and focused designs
linked to from here. However, it should provide an overarching view of the
design and a good basis for diving into specific details.

However, these are extremely early days for Carbon. Currently, this document
tries to capture two things:

1. Initial musings about what _might_ make sense as a basis for Carbon. These
   are largely informed by idle discussions between C++ and Clang developers
   over the years, and should not be given any particular weight.
2. A summary and snapshot of in-progress efforts to flesh out and motivate
   specific designs for parts of the language.

The utility of capturing these at this early stage is primarily to give everyone
a reasonably consistent set of terminology and context as we begin fleshing out
concrete (and well justified) designs for each part of the language. In some
cases, it captures ideas that may be interesting to explore, but isn't meant to
overly anchor on them. Any ideas here need to be fully explored and justified
with a detailed analysis. The context of #1 (directly evolving C++, experience
building Clang, and experience working on C++ codebases including Clang and LLVM
themselves) is also important. It is both an important signal but also a bias.

### Example code

Some syntax used in example code is provisional or placeholder, and may change
later.

## Basic syntax

### Code and comments

> References: [Source files](code_and_name_organization/source_files.md) and
> [lexical conventions](lexical_conventions)
>
> **TODO:** References need to be evolved.

-   All source code is UTF-8 encoded text. For simplicity, no other encoding is
    supported.
-   Line comments look like `// ...`. However, they are required to be the only
    non-whitespace on the line for readability.
-   Block comments look like `//\{ ... //\}`, with each marker on its own line.
    Nested block comments are supported using named regions. For example:

    ```carbon
      live code
    //\{
      commented code
    //\{ nested block
      commented code in nested block
    //\} nested block
    //\}
      live code
    ```

-   Decimal, hexadecimal, and binary integer literals and decimal and
    hexadecimal floating-point literals are supported, with `_` as a digit
    separator. For example, `42`, `0b1011_1101` and `0x1.EEFp+5`. Numeric
    literals are case-sensitive: `0x`, `0b`, `e+`, and `p+` must be lowercase,
    whereas hexadecimal digits must be uppercase. A digit is required on both
    sides of a period.

### Packages, libraries, and namespaces

> References: [Code and name organization](code_and_name_organization)

-   **Files** are grouped into libraries, which are in turn grouped into
    packages.
-   **Libraries** are the granularity of code reuse through imports.
-   **Packages** are the unit of distribution.

Name paths in Carbon always start with the package name. Additional namespaces
may be specified as desired.

For example, this code declares a class `Geometry.Shapes.Flat.Circle` in a
library `Geometry/OneSide`:

```carbon
package Geometry library("OneSide") namespace Shapes;

namespace Flat;
class Flat.Circle { ... }
```

This type can be used from another package:

```carbon
package ExampleUser;

import Geometry library("OneSide");

fn Foo(Geometry.Shapes.Flat.Circle circle) { ... }
```

### Names and scopes

> References: [Lexical conventions](lexical_conventions)
>
> **TODO:** References need to be evolved.

Various constructs introduce a named entity in Carbon. These can be functions,
types, variables, or other kinds of entities that we'll cover. A name in Carbon
is formed from a word, which is a sequence of letters, numbers, and underscores,
and which starts with a letter. We intend to follow Unicode's Annex 31 in
selecting valid identifier characters, but a concrete set of valid characters
has not been selected yet.

#### Naming conventions

> References: [Naming conventions](naming_conventions.md)

Our naming conventions are:

-   For idiomatic Carbon code:
    -   `UpperCamelCase` will be used when the named entity cannot have a
        dynamically varying value. For example, functions, namespaces, or
        compile-time constant values.
    -   `lower_snake_case` will be used when the named entity's value won't be
        known until runtime, such as for variables.
-   For Carbon-provided features:
    -   Keywords and type literals will use `lower_snake_case`.
    -   Other code will use the conventions for idiomatic Carbon code.

#### Aliases

> References: [Aliases](aliases.md)
>
> **TODO:** References need to be evolved.

Carbon provides a facility to declare a new name as an alias for a value. This
is a fully general facility because everything is a value in Carbon, including
types.

For example:

```carbon
alias MyInt = Int;
```

This creates an alias called `MyInt` for whatever `Int` resolves to. Code
textually after this can refer to `MyInt`, and it will transparently refer to
`Int`.

#### Name lookup

> References: [name lookup](name_lookup.md)
>
> **TODO:** References need to be evolved.

Unqualified name lookup will always find a file-local result, including aliases.

##### Name lookup for common types

> References: [Name lookup](name_lookup.md)
>
> **TODO:** References need to be evolved.

Common types that we expect to be used universally will be provided for every
file, including `Int` and `Bool`. These will likely be defined in a special
"prelude" package.

### Expressions

> References: [Lexical conventions](lexical_conventions) and
> [expressions](expressions/)
>
> **TODO:** References need to be evolved.

Expressions describe some computed value. The simplest example would be a
literal number like `42`: an expression that computes the integer value 42.

Some common expressions in Carbon include:

-   Literals: `42`, `3.1419`, `"Hello World!"`
-   Operators:

    -   Increment and decrement: `++i`, `--j`
        -   These do not return any result.
    -   Unary negation: `-x`
    -   Arithmetic: `1 + 2`, `3 - 4`, `2 * 5`, `6 / 3`
    -   Bitwise: `2 & 3`, `2 | 4`, `3 ^ 1`, `~7`
    -   Bit shift: `1 << 3`, `8 >> 1`
    -   Comparison: `2 == 2`, `3 != 4`, `5 < 6`, `7 > 6`, `8 <= 8`, `8 >= 8`
    -   Logical: `a and b`, `c or d`

-   Parenthesized expressions: `(7 + 8) * (3 - 1)`

### Functions

> References: [Functions](functions.md)

Functions are the core unit of behavior. For example:

```carbon
fn Add(a: i64, b: i64) -> i64;
```

Breaking this apart:

-   `fn` is the keyword used to indicate a function.
-   Its name is `Add`.
-   It accepts two `i64` parameters, `a` and `b`.
-   It returns an `i64` result.

You would call this function like `Add(1, 2)`.

### Blocks and statements

> References: [Blocks and statements](blocks_and_statements.md)
>
> **TODO:** References need to be evolved.

The body or definition of a function is provided by a block of code containing
statements. The body of a function is also a new, nested scope inside the
function's scope, meaning that parameter names are available.

Statements within a block are terminated by a semicolon. Each statement can,
among other things, be an expression.

For example, here is a function definition using a block of statements, one of
which is nested:

```carbon
fn Foo() {
  Bar();
  {
    Baz();
  }
}
```

### Variables

> References: [Variables](variables.md)

Blocks introduce nested scopes and can contain local variable declarations that
work similarly to function parameters.

For example:

```carbon
fn DoSomething() {
  var x: i64 = 42;
}
```

Breaking this apart:

-   `var` is the keyword used to indicate a variable.
-   Its name is `x`.
-   Its type is `i64`.
-   It is initialized with the value `42`.

### Lifetime and move semantics

> References: TODO
>
> **TODO:** References need to be evolved.

### Control flow

> References: [Control flow](control_flow/README.md)

Blocks of statements are generally executed sequentially. However, statements
are the primary place where this flow of execution can be controlled.

#### `if` and `else`

> References: [Control flow](control_flow/conditionals.md)

`if` and `else` provide conditional execution of statements. For example:

```carbon
if (fruit.IsYellow()) {
  Print("Banana!");
} else if (fruit.IsOrange()) {
  Print("Orange!");
} else {
  Print("Vegetable!");
}
```

This code will:

-   Print `Banana!` if `fruit.IsYellow()` is `True`.
-   Print `Orange!` if `fruit.IsYellow()` is `False` and `fruit.IsOrange()` is
    `True`.
-   Print `Vegetable!` if both of the above return `False`.

#### Loops

##### `while`

> References: [Control flow](control_flow/loops.md#while)

`while` statements loop for as long as the passed expression returns `True`. For
example, this prints `0`, `1`, `2`, then `Done!`:

```carbon
var x: Int = 0;
while (x < 3) {
  Print(x);
  ++x;
}
Print("Done!");
```

##### `for`

> References: [Control flow](control_flow/loops.md#for)

`for` statements support range-based looping, typically over containers. For
example, this prints all names in `names`:

```carbon
for (var name: String in names) {
  Print(name);
}
```

`PrintNames()` prints each `String` in the `names` `List` in iteration order.

##### `break`

> References: [Control flow](control_flow/loops.md#break)

The `break` statement immediately ends a `while` or `for` loop. Execution will
resume at the end of the loop's scope. For example, this processes steps until a
manual step is hit (if no manual step is hit, all steps are processed):

```carbon
for (var step: Step in steps) {
  if (step.IsManual()) {
    Print("Reached manual step!");
    break;
  }
  step.Process();
}
```

##### `continue`

> References: [Control flow](control_flow/loops.md#continue)

The `continue` statement immediately goes to the next loop of a `while` or
`for`. In a `while`, execution continues with the `while` expression. For
example, this prints all non-empty lines of a file, using `continue` to skip
empty lines:

```carbon
var f: File = OpenFile(path);
while (!f.EOF()) {
  var line: String = f.ReadLine();
  if (line.IsEmpty()) {
    continue;
  }
  Print(line);
}
```

#### `return`

> References: [Control flow](control_flow/return.md)

The `return` statement ends the flow of execution within a function, returning
execution to the caller. If the function returns a value to the caller, that
value is provided by an expression in the return statement. For example:

```carbon
fn Sum(a: Int, b: Int) -> Int {
  return a + b;
}
```

## Types

> References: [Primitive types](primitive_types.md), [tuples](tuples.md), and
> [classes](classes.md)
>
> **TODO:** References need to be evolved.

Carbon's core types are broken down into three categories:

-   Primitive types
-   Composite types
-   User-defined types

The first two are intrinsic and directly built in the language. The last aspect
of types allows for defining new types.

Expressions compute values in Carbon, and these values are always strongly typed
much like in C++. However, an important difference from C++ is that types are
themselves modeled as values; specifically, compile-time constant values.
However, in simple cases this doesn't make much difference.

### Primitive types

> References: [Primitive types](primitive_types.md)
>
> **TODO:** References need to be evolved.

These types are fundamental to the language as they aren't either formed from or
modifying other types. They also have semantics that are defined from first
principles rather than in terms of other operations. These will be made
available through the [prelude package](#name-lookup-for-common-types).

Primitive types fall into the following categories:

-   `Bool` - a boolean type with two possible values: `True` and `False`.
-   `Int` and `UInt` - signed and unsigned 64-bit integer types.
    -   Standard sizes are available, both signed and unsigned, including
        `Int8`, `Int16`, `Int32`, `Int128`, and `Int256`.
    -   Overflow in either direction is an error.
-   `Float64` - a floating point type with semantics based on IEEE-754.
    -   Standard sizes are available, including `Float16`, `Float32`, and
        `Float128`.
    -   [`BFloat16`](primitive_types.md#bfloat16) is also provided.
-   `String` - a byte sequence treated as containing UTF-8 encoded text.
    -   `StringView` - a read-only reference to a byte sequence treated as
        containing UTF-8 encoded text.

### Composite types

#### Tuples

> References: [Tuples](tuples.md)
>
> **TODO:** References need to be evolved.

The primary composite type involves simple aggregation of other types as a
tuple. In formal type theory, tuples are product types.

An example use of tuples is:

```carbon
fn DoubleBoth(x: Int, y: Int) -> (Int, Int) {
  return (2 * x, 2 * y);
}
```

Breaking this example apart:

-   The return type is a tuple of two `Int` types.
-   The expression uses tuple syntax to build a tuple of two `Int` values.

Both of these are expressions using the tuple syntax
`(<expression>, <expression>)`. The only difference is the type of the tuple
expression: one is a tuple of types, the other a tuple of values.

Element access uses subscript syntax:

```carbon
fn DoubleTuple(x: (Int, Int)) -> (Int, Int) {
  return (2 * x[0], 2 * x[1]);
}
```

Tuples also support multiple indices and slicing to restructure tuple elements:

```carbon
// This reverses the tuple using multiple indices.
fn Reverse(x: (Int, Int, Int)) -> (Int, Int, Int) {
  return x[2, 1, 0];
}

// This slices the tuple by extracting elements [0, 2).
fn RemoveLast(x: (Int, Int, Int)) -> (Int, Int) {
  return x[0 .. 2];
}
```

#### Variants

> **TODO:** Needs a feature design and a high level summary provided inline.

#### Pointers and references

> **TODO:** Needs a feature design and a high level summary provided inline.

#### Arrays and slices

> **TODO:** Needs a feature design and a high level summary provided inline.

### User-defined types

#### Classes

> References: [Classes](classes.md)

Classes are a way for users to define their own data strutures or record types.

For example:

```carbon
class Widget {
  var x: Int;
  var y: Int;
  var z: Int;

  var payload: String;
}
```

Breaking apart `Widget`:

-   `Widget` has three `Int` members: `x`, `y`, and `z`.
-   `Widget` has one `String` member: `payload`.
-   Given an instance `dial`, a member can be referenced with `dial.paylod`.

##### Assignment, copying

You may use a _structural data class literal_, also known as a _struct literal_,
to assign or initialize a variable with a class type.

```carbon
var sprocket: Widget = {.x = 3, .y = 4, .z = 5, .payload = "Sproing"};
sprocket = {.x = 2, .y = 1, .z = 0, .payload = "Bounce"};
```

You may also copy one struct into another of the same type.

```carbon
var thingy: Widget = sprocket;
sprocket = thingy;
```

##### Member access

The data members of a variable with a class type may be accessed using dot `.`
notation:

```carbon
Assert(sprocket.x == thingy.x);
```

##### Methods

Class type definitions can include methods:

```carbon
class Point {
  fn Distance[me: Self](x2: i32, y2: i32) -> f32 {
    var dx: i32 = x2 - me.x;
    var dy: i32 = y2 - me.y;
    return Math.Sqrt(dx * dx - dy * dy);
  }
  fn Offset[addr me: Self*](dx: i32, dy: i32);

  var x: i32;
  var y: i32;
}

fn Point.Offset[addr me: Self*](dx: i32, dy: i32) {
  me->x += dx;
  me->y += dy;
}

var origin: Point = {.x = 0, .y = 0};
Assert(Math.Abs(origin.Distance(3, 4) - 5.0) < 0.001);
origin.Offset(3, 4);
Assert(origin.Distance(3, 4) == 0.0);
```

This defines a `Point` class type with two integer data members `x` and `y` and
two methods `Distance` and `Offset`:

-   Methods are defined as functions with a `me` parameter inside square
    brackets `[`...`]` before the regular explicit parameter list in parens
    `(`...`)`.
-   Methods are called using using the member syntax, `origin.Distance(`...`)`
    and `origin.Offset(`...`)`.
-   `Distance` computes and returns the distance to another point, without
    modifying the `Point`. This is signified using `[me: Self]` in the method
    declaration.
-   `origin.Offset(`...`)` does modify the value of `origin`. This is signified
    using `[addr me: Self*]` in the method declaration.
-   Methods may be declared lexically inline like `Distance`, or lexically out
    of line like `Offset`.

##### Allocation, construction, and destruction

> **TODO:** Needs a feature design and a high level summary provided inline.

##### Moving

> **TODO:** Needs a feature design and a high level summary provided inline.

##### Comparison

> **TODO:** Needs a feature design and a high level summary provided inline.

##### Implicit and explicit conversion

> **TODO:** Needs a feature design and a high level summary provided inline.

##### Inline type composition

> **TODO:** Needs a feature design and a high level summary provided inline.

#### Unions

> **TODO:** Needs a detailed design and a high level summary provided inline.

## Names

### Packages, libraries, namespaces

> References:
>
> -   [Code and name organization](code_and_name_organization)
> -   Proposal
>     [#107: Code and name organization](https://github.com/carbon-language/carbon-lang/pull/107)
> -   Proposal
>     [#752: api file default publicn](https://github.com/carbon-language/carbon-lang/pull/752)
> -   Question-for-leads issue
>     [#1136: what is the top-level scope in a source file, and what names are found there?](https://github.com/carbon-language/carbon-lang/issues/1136)

-   **Files** are grouped into libraries, which are in turn grouped into
    packages.
-   **Libraries** are the granularity of code reuse through imports.
-   **Packages** are the unit of distribution.

Name paths in Carbon always start with the package name. Additional namespaces
may be specified as desired.

For example, this code declares a class `Geometry.Shapes.Flat.Circle` in a
library `Geometry/OneSide`:

```carbon
package Geometry library("OneSide") namespace Shapes;

namespace Flat;
class Flat.Circle { ... }
```

This type can be used from another package:

```carbon
package ExampleUser;

import Geometry library("OneSide");

fn Foo(Geometry.Shapes.Flat.Circle circle) { ... }
```

### Names and scopes

> References: [Lexical conventions](lexical_conventions)
>
> **TODO:** References need to be evolved.

Various constructs introduce a named entity in Carbon. These can be functions,
types, variables, or other kinds of entities that we'll cover. A name in Carbon
is formed from a word, which is a sequence of letters, numbers, and underscores,
and which starts with a letter. We intend to follow Unicode's Annex 31 in
selecting valid identifier characters, but a concrete set of valid characters
has not been selected yet.

### Naming conventions

> References:
>
> -   [Naming conventions](naming_conventions.md)
> -   Proposal
>     [#861: Naming conventions](https://github.com/carbon-language/carbon-lang/pull/861)

Our naming conventions are:

-   For idiomatic Carbon code:
    -   `UpperCamelCase` will be used when the named entity cannot have a
        dynamically varying value. For example, functions, namespaces, or
        compile-time constant values.
    -   `lower_snake_case` will be used when the named entity's value won't be
        known until runtime, such as for variables.
-   For Carbon-provided features:
    -   Keywords and type literals will use `lower_snake_case`.
    -   Other code will use the conventions for idiomatic Carbon code.

### Aliases

> References:
>
> -   [Aliases](aliases.md)
> -   Question-for-leads issue
>     [#749: Alias syntax](https://github.com/carbon-language/carbon-lang/issues/749)

> **TODO:** References need to be evolved.

Carbon provides a facility to declare a new name as an alias for a value. This
is a fully general facility because everything is a value in Carbon, including
types.

For example:

```carbon
alias MyInt = i32;
```

This creates an alias called `MyInt` for whatever `i32` resolves to. Code
textually after this can refer to `MyInt`, and it will transparently refer to
`i32`.

### Name lookup

> References:
>
> -   [Name lookup](name_lookup.md)
> -   Proposal
>     [#989: Member access expressions](https://github.com/carbon-language/carbon-lang/pull/989)
>
> **TODO:** References need to be evolved.

Unqualified name lookup will always find a file-local result, including aliases.

#### Name lookup for common types

FIXME: should this be renamed to "The prelude"?

> References: [Name lookup](name_lookup.md)
>
> -   Question-for-leads issue
>     [#750: Naming conventions for Carbon-provided features](https://github.com/carbon-language/carbon-lang/issues/750)
> -   Question-for-leads issue
>     [#1058: How should interfaces for core functionality be named?](https://github.com/carbon-language/carbon-lang/issues/1058)
>
> **TODO:** References need to be evolved.

Common types that we expect to be used universally will be provided for every
file, including `i32` and `Bool`. These will likely be defined in a special
"prelude" package.

### Name visibility

> References:
>
> -   FIXME: Name visibility and access control at file scope
> -   Question-for-leads issue
>     [#665: `private` vs `public` _syntax_ strategy, as well as other visibility tools like `external`/`api`/etc.](https://github.com/carbon-language/carbon-lang/issues/665)
> -   Proposal
>     [#752: api file default public](https://github.com/carbon-language/carbon-lang/pull/752)

> **TODO:**

## Generics

> References: **TODO:** Revisit
>
> -   [Generics: Overview](generics/overview.md)
> -   Proposal
>     [#524: Generics overview](https://github.com/carbon-language/carbon-lang/pull/524)
> -   Proposal
>     [#731: Generics details 2: adapters, associated types, parameterized interfaces](https://github.com/carbon-language/carbon-lang/pull/731)
> -   Proposal
>     [#818: Constraints for generics (generics details 3)](https://github.com/carbon-language/carbon-lang/pull/818)
> -   Proposal
>     [#920: Generic parameterized impls (details 5)](https://github.com/carbon-language/carbon-lang/pull/920)
> -   Proposal
>     [#950: Generic details 6: remove facets](https://github.com/carbon-language/carbon-lang/pull/950)
> -   Proposal
>     [#1013: Generics: Set associated constants using `where` constraints](https://github.com/carbon-language/carbon-lang/pull/1013)
> -   Proposal
>     [#1084: Generics details 9: forward declarations](https://github.com/carbon-language/carbon-lang/pull/1084)

Generics allow Carbon constructs like [functions](#functions) and
[classes](#classes) to have compile-time parameters to allow them to be
applicable to more types. For example, this `Min` function has a type parameter
`T` that can be any type that implements the `Ordered` interface.

```carbon
fn Min[T:! Ordered](x: T, y: T) -> T {
  // Can compare `x` and `y` since they have
  // type `T` known to implement `Ordered`.
  return if x <= y then x else y;
}

var a: i32 = 1;
var b: i32 = 2;
// `T` is deduced to be `i32`
Assert(Min(a, b) == 1);
// `T` is deduced to be `String`
Assert(Min("abc", "xyz") == "abc");
```

Since the `T` type parameter is in the deduced parameter list in square brackets
(`[`...`]`) before the explicit parameter list in parentheses (`(`...`)`), the
value of `T` is determined from the types of the explicit arguments instead of
being passed as a separate explicit argument.

### Checked and template parameters

> References:
>
> -   [Templates](templates.md)
> -   Proposal
>     [#989: Member access expressions](https://github.com/carbon-language/carbon-lang/pull/989)

The `:!` indicates that `T` is a _checked_ parameter passed at compile time.
"Checked" here means that the body of `Min` is type checked when the function is
defined, independent of the specific type values `T` is instantiated with, and
name lookup is delegated to the constraint on `T` (`Ordered` in this case). This
type checking is equivalent to saying the function would pass type checking
given any type `T` that implements the `Ordered` interface. Then calls to `Min`
only need to check that the deduced type value of `T` implements `Ordered`.

Instead, the parameter could be declared to be a _template_ parameter by
prefixing with the `template` keyword, as in `template T:! Type`.

```carbon
fn Convert[template T:! Type](source: T, template U:! Type) -> U {
  var converted: U = source;
  return converted;
}

fn Foo(i: i32) -> f32 {
  // Instantiates with the `T` implicit argument set to `i32` and the `U`
  // explicit argument set to `f32`, then calls with the runtime value `i`.
  return Convert(i, f32);
}
```

Carbon templates follow the same fundamental paradigm as
[C++ templates](<https://en.wikipedia.org/wiki/Template_(C%2B%2B)>): they are
instantiated when called, resulting in late type checking, duck typing, and lazy
binding.

Member lookup into a template type parameter is done in the actual type value
provided by the caller. This means member name lookup and type checking for
anything [dependent](generics/terminology.md#dependent-names) on the template
parameter are delayed until the template is instantiated with a specific
concrete type. This gives semantics similar to

Although generics are generally preferred, templates enable translation of code
between C++ and Carbon, and address some cases where the type checking rigor of
generics are problematic.

### Interfaces and implementations

_Interfaces_ specify a set of requirements that a types might satisfy.
Interfaces act both as constraints on types a caller might supply and
capabilities that may be assumed of types that satisfy that constraint.

```carbon
interface Printable {
  // Inside an interface definition `Self` means
  // "the type implementing this interface".
  fn Print[me: Self]();
}
```

Types only implement an interface if there is an explicit `impl` declaration
that they do. Simply having a `Print` function with the right signature is not
sufficient.

```carbon
class Circle {
  var radius: f32;

  impl as Printable {
    fn Print[me: Self]() {
      Console.WriteLine("Circle with radius: {0}", me.radius);
    }
  }
}
```

### Combining constraints

> References:
>
> -   [Combining interfaces by anding type-of-types](generics/details.md#combining-interfaces-by-anding-type-of-types)
> -   Question-for-leads issue
>     [#531: Combine interfaces with `+` or `&`](https://github.com/carbon-language/carbon-lang/issues/531)

A function can require calling types to implement multiple interfaces by
combining them using an ampersand (`&`):

```carbon
fn PrintMin[T:! Ordered & Printable](x: T, y: T) {
  // Can compare since type `T` implements `Ordered`.
  if (x <= y) {
    // Can call `Print` since type `T` implements `Printable`.
    x.Print();
  } else {
    y.Print();
  }
}
```

The body of the function may call functions that are in either interface, except
for names that are members of both. In that case, use the
[compound member access syntax to qualify the name of the member](generics/details.md#qualified-member-names-and-compound-member-access),
as in:

```carbon
fn DrawTies[T:! Renderable & GameResult](x: T) {
  if (x.(GameResult.Draw)()) {
    x.(Renderable.Draw)();
  }
}
```

### Generic types

> **TODO:**

#### Types with template parameters

> References: [Templates](templates.md)
>
> **TODO:** References need to be evolved.

User-defined types may have template parameters. The resulting type-function may
be used to instantiate the parameterized definition with the provided arguments
in order to produce a complete type. For example:

```carbon
class Stack(template T:! Type) {
  var storage: Array(T);

  fn Push(value: T);
  fn Pop() -> T;
}
```

Breaking apart the template use in `Stack`:

-   `Stack` is a paremeterized type accepting a type `T`.
-   `T` may be used within the definition of `Stack` anywhere a normal type
    would be used, and will only be type checked on instantiation.
-   `var ... Array(T)` instantiates a parameterized type `Array` when `Stack` is
    instantiated.

#### Generic choice types

```carbon
choice Result(T:! Type, Error:! Type) {
  Success(value: T),
  Failure(error: Error)
}
```

### Other features

> References:
>
> -   [Generics details](generics/details.md)

**TODO:**

-   [external impls](generics/details.md#external-impl)
-   [named and template constraints](generics/details.md#named-constraints)
-   [extending interfaces](generics/details.md#interface-extension)
-   [adapter types](generics/details.md#adapting-types)
-   [associated types](generics/details.md#associated-types) and other
    [associated constants](generics/details.md#associated-constants)
-   [generic/parameterized interfaces](generics/details.md#parameterized-interfaces)
-   [`where` constraints](generics/details.md#where-constraints)
-   [implied constraints](generics/details.md#implied-constraints)
-   `observe` declarations:
    [observing types are equal](generics/details.md#observe-declarations),
    [observing types implement an interface](generics/details.md#observing-a-type-implements-an-interface)
-   [generic/parameterized impls](generics/details.md#parameterized-impls)
-   [specialization](generics/details.md#lookup-resolution-and-specialization)
-   [`final` impls](generics/details.md#final-impls)
-   [forward declarations](generics/details.md#forward-declarations-and-cyclic-references)
-   [interface defaults](generics/details.md#interface-defaults)
-   [`final` interface members](generics/details.md#final-members)
-   [dynamic erased types](generics/details.md#runtime-type-fields)
-   [variadics](generics/details.md#variadic-arguments)

### Operator overloading

> References:
>
> -   [Operator overloading](generics/details.md#operator-overloading)
> -   Proposal
>     [#702: Comparison operators](https://github.com/carbon-language/carbon-lang/pull/702)
> -   Proposal
>     [#820: Implicit conversions](https://github.com/carbon-language/carbon-lang/pull/820)
> -   Proposal
>     [#845: as expressions](https://github.com/carbon-language/carbon-lang/pull/845)
> -   Question-for-leads issue
>     [#1058: How should interfaces for core functionality be named?](https://github.com/carbon-language/carbon-lang/issues/1058)
> -   Proposal
>     [#1083: Arithmetic expressions](https://github.com/carbon-language/carbon-lang/pull/1083)
> -   Proposal
>     [#1191: Bitwise operators](https://github.com/carbon-language/carbon-lang/pull/1191)
> -   Proposal
>     [#1178: Rework operator interfaces](https://github.com/carbon-language/carbon-lang/pull/1178)

> **TODO:** Operators are translated into calls into interface methods, so to
> overload an operator for a type, implement the corresponding interface for
> that type.

> **TODO:** `like` for implicit conversions

> **TODO:** Binary operators taking the same or different types

> **TODO:** Change this to a table? Concern: no support for merging cells in a
> markdown table unless you make it using html.

-   [Arithmetic](expressions/arithmetic.md#extensibility):
    -   `-x`: `Negate`
    -   `x + y`: `Add` or `AddWith(U)`
    -   `x - y`: `Sub` or `SubWith(U)`
    -   `x * y`: `Mul` or `MulWith(U)`
    -   `x / y`: `Div` or `DivWith(U)`
    -   `x % y`: `Mod` or `ModWith(U)`
-   [Bitwise and shift operators](expressions/bitwise.md#extensibility):
    -   `^x`: `BitComplement`
    -   `x & y`: `BitAnd` or `BitAndWith(U)`
    -   `x | y`: `BitOr` or `BitOrWith(U)`
    -   `x ^ y`: `BitXor` or `BitXorWith(U)`
    -   `x << y`: `LeftShift` or `LeftShiftWith(U)`
    -   `x >> y`: `RightShift` or `RightShiftWith(U)`
-   Comparison:
    -   `x == y`, `x != y` overloaded by implementing
        [`Eq` or `EqWith(U)`](expressions/comparison_operators.md#equality)
    -   `x < y`, `x > y`, `x <= 8`, `8 >= 8` overloaded by implementing
        [`Ordered` or `OrderedWith(U)`](expressions/comparison_operators.md#ordering)
-   Conversion:
    -   `x as U` is rewritten to use the
        [`As(U)`](expressions/as_expressions.md#extensibility) interface
    -   Implicit conversions use
        [`ImplicitAs(U)`](expressions/implicit_conversions.md#extensibility)
-   **TODO:** Indexing: `a[3]`
-   **TODO:** Function call: `f(4)`

#### Common type

> References:
>
> -   [`if` expressions](expressions/if.md#finding-a-common-type)
> -   Proposal
>     [#911: Conditional expressions](https://github.com/carbon-language/carbon-lang/pull/911)

> **TODO:**

Common type: used to define the result of
[conditional expressions like `if c then t else f`](expressions/if.md) and other
situations where a common type needs to be found for two types, as in:

```carbon
fn F[T:! Type](x: T, y: T);

var a: U;
var b: V;
// Calls `F` with the `T` set to
// the common type of `U` and `V`:
F(a, b);
```

## Bidirectional interoperability with C/C++

> References:
>
> -   [Bidirectional interoperability with C/C++](interoperability/README.md)
> -   Proposal
>     [#175: C++ interoperability goals](https://github.com/carbon-language/carbon-lang/pull/175)
>
> **TODO:** References need to be evolved. Needs a detailed design and a high
> level summary provided inline.

## Unfinished tales

### Pattern matching as function overload resolution

> References: [Pattern matching](pattern_matching.md)
>
> **TODO:** References need to be evolved. Needs a detailed design and a high
> level summary provided inline.

### Lifetime and move semantics

> **TODO:**

### Metaprogramming

> References: [Metaprogramming](metaprogramming.md)
>
> **TODO:** References need to be evolved. Needs a detailed design and a high
> level summary provided inline.

Carbon provides metaprogramming facilities that look similar to regular Carbon
code. These are structured, and do not offer arbitrary inclusion or
preprocessing of source text such as C/C++ does.

### Execution abstractions

Carbon provides some higher-order abstractions of program execution, as well as
the critical underpinnings of such abstractions.

#### Abstract machine and execution model

> **TODO:**

#### Lambdas

> **TODO:**

#### Co-routines

> **TODO:**
