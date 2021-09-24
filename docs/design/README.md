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
-   [Pattern matching](#pattern-matching)
    -   [`match` control flow](#match-control-flow)
    -   [Pattern matching in local variables](#pattern-matching-in-local-variables)
    -   [Pattern matching as function overload resolution](#pattern-matching-as-function-overload-resolution)
-   [Type abstractions](#type-abstractions)
    -   [Interfaces](#interfaces)
    -   [Generics](#generics)
    -   [Templates](#templates)
        -   [Types with template parameters](#types-with-template-parameters)
        -   [Functions with template parameters](#functions-with-template-parameters)
        -   [Overloading](#overloading)
-   [Metaprogramming](#metaprogramming)
-   [Execution abstractions](#execution-abstractions)
    -   [Abstract machine and execution model](#abstract-machine-and-execution-model)
    -   [Lambdas](#lambdas)
    -   [Co-routines](#co-routines)
-   [Bidirectional interoperability with C/C++](#bidirectional-interoperability-with-cc)

<!-- tocstop -->

## Overview

This documentation describes the design of the Carbon language, and the
rationale for that design. The goal is to provide sufficient coverage of the
design to support the following audiences:

-   People who wish to determine whether Carbon would be the right choice to use
    for a project compared to other existing languages.
-   People working on the evolution of the Carbon language who wish to
    understanding the rationale and motivation for existing design decisions.
-   People working on a specification or implementation of the Carbon language
    who need a detailed understanding of the intended design.
-   People writing Carbon code who wish to understand why the language rules are
    the way they are.

For Carbon developers, documentation that is more suitable for learning the
language will be made available separately.

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

In order to keep example code consistent, we are making choices that may change
later. In particular, where `$` is shown in examples, it is a placeholder: `$`
is a well-known bad symbol due to international keyboard layouts, and will be
cleaned up during evolution.

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
>
> **TODO:** References need to be evolved.

Our current proposed naming convention are:

-   `UpperCamelCase` for names of compile-time resolved constants, whether they
    participate in the type system or not.
-   `lower_snake_case` for keywords and names of run-time resolved values.

As a matter of style and consistency, we will follow these conventions where
possible and encourage convergence.

For example:

-   An integer that is a compile-time constant sufficient to use in the
    construction a compile-time array size, such as a template function
    parameter, might be named `N`.
-   A generic function parameter's value can't be used during type-checking, but
    might still be named `N`, since it will be a constant available to the
    compiler at code generation time.
-   Functions and most types will be in `UpperCamelCase`.
-   A type where only run-time type information queries are available would end
    up as `lower_snake_case`.
-   A keyword like `import` uses `lower_snake_case`.

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
> [operators](operators.md)
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
>
> **TODO:** References need to be evolved.

Functions are the core unit of behavior. For example:

```carbon
fn Sum(a: Int, b: Int) -> Int;
```

Breaking this apart:

-   `fn` is the keyword used to indicate a function.
-   Its name is `Sum`.
-   It accepts two `Int` parameters, `a` and `b`.
-   It returns an `Int` result.

You would call this function like `Sum(1, 2)`.

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
fn Foo() {
  var x: Int = 42;
}
```

Breaking this apart:

-   `var` is the keyword used to indicate a variable.
-   Its name is `x`.
-   Its type is `Int`.
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

## Pattern matching

> References: [Pattern matching](pattern_matching.md)
>
> **TODO:** References need to be evolved.

The most prominent mechanism to manipulate and work with types in Carbon is
pattern matching. This may seem like a deviation from C++, but in fact this is
largely about building a clear, coherent model for a fundamental part of C++:
overload resolution.

### `match` control flow

> References: [Pattern matching](pattern_matching.md)
>
> **TODO:** References need to be evolved.

`match` is a control flow similar to `switch` of C/C++ and mirrors similar
constructs in other languages, such as Swift.

An example `match` is:

```carbon
fn Bar() -> (Int, (Float, Float));

fn Foo() -> Float {
  match (Bar()...) {
    case (42, (x: Float, y: Float)) => {
      return x - y;
    }
    case (p: Int, (x: Float, _: Float)) if (p < 13) => {
      return p * x;
    }
    case (p: Int, _: auto) if (p > 3) => {
      return p * Pi;
    }
    default => {
      return Pi;
    }
  }
}
```

Breaking apart this `match`:

-   It accepts a value that will be inspected; in this case, the result of the
    call to `Bar()`.
    -   It then will find the _first_ `case` that matches this value, and
        execute that block.
    -   If none match, then it executes the default block.
-   Each `case` pattern contains a value pattern, such as `(Int p, auto _)`,
    followed by an optional boolean predicate introduced by the `if` keyword.
    -   The value pattern must first match, and then the predicate must also
        evaluate to true for the overall `case` pattern to match.
    -   Using `auto` for a type will always match.

Value patterns may be composed of the following:

-   An expression, such as `42`, whose value must be equal to match.
-   An identifier to bind the value, followed by a `:` and followed by a type,
    such as `Int`.
    -   The special identifier `_` may be used to discard the value once
        matched.
-   A destructuring pattern containing a sequence of value patterns, such as
    `(x: Float, y: Float)`, which match against tuples and tuple-like values by
    recursively matching on their elements.
-   An unwrapping pattern containing a nested value pattern which matches
    against a variant or variant-like value by unwrapping it.

### Pattern matching in local variables

> References: [Pattern matching](pattern_matching.md)
>
> **TODO:** References need to be evolved.

Value patterns may be used when declaring local variables to conveniently
destructure them and do other type manipulations. However, the patterns must
match at compile time, so a boolean predicate cannot be used directly.

An example use is:

```carbon
fn Bar() -> (Int, (Float, Float));
fn Foo() -> Int {
  var (p: Int, _: auto) = Bar();
  return p;
}
```

To break this apart:

-   The `Int` returned by `Bar()` matches and is bound to `p`, then returned.
-   The `(Float, Float)` returned by `Bar()` matches and is discarded by
    `_: auto`.

### Pattern matching as function overload resolution

> References: [Pattern matching](pattern_matching.md)
>
> **TODO:** References need to be evolved. Needs a detailed design and a high
> level summary provided inline.

## Type abstractions

### Interfaces

> **TODO:** Needs a feature design and a high level summary provided inline.

### Generics

> **TODO:** Needs a feature design and a high level summary provided inline.

### Templates

> References: [Templates](templates.md)
>
> **TODO:** References need to be evolved.

Carbon templates follow the same fundamental paradigm as C++ templates: they are
instantiated when called, resulting in late type checking, duck typing, and lazy
binding. Although generics are generally preferred, templates enable translation
of code between C++ and Carbon, and address some cases where the type checking
rigor of generics are problematic.

#### Types with template parameters

> References: [Templates](templates.md)
>
> **TODO:** References need to be evolved.

User-defined types may have template parameters. The resulting type-function may
be used to instantiate the parameterized definition with the provided arguments
in order to produce a complete type. For example:

```carbon
class Stack(T:$$ Type) {
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

#### Functions with template parameters

> References: [Templates](templates.md)
>
> **TODO:** References need to be evolved.

Both implicit and explicit function parameters in Carbon can be marked as
_template_ parameters. When called, the arguments to these parameters trigger
instantiation of the function definition, fully type checking and resolving that
definition after substituting in the provided (or computed if implicit)
arguments. The runtime call then passes the remaining arguments to the resulting
complete definition.

```carbon
fn Convert[T:$$ Type](source: T, U:$$ Type) -> U {
  var converted: U = source;
  return converted;
}

fn Foo(i: Int) -> Float {
  // Instantiates with the `T` implicit argument set to `Int` and the `U`
  // explicit argument set to `Float`, then calls with the runtime value `i`.
  return Convert(i, Float);
}
```

Here we deduce one type parameter and explicitly pass another. It is not
possible to explicitly pass a deduced type parameter; instead the call site
should cast or convert the argument to control the deduction. In this particular
example, the explicit type is passed after a runtime parameter. While this makes
that type unavailable to the declaration of _that_ runtime parameter, it still
is a _template_ parameter and available to use as a type in the remaining parts
of the function declaration.

#### Overloading

> References: [Templates](templates.md)
>
> **TODO:** References need to be evolved.

An important feature of templates in C++ is the ability to customize how they
end up specialized for specific arguments. Because template parameters (whether
as type parameters or function parameters) are pattern matched, we expect to
leverage pattern matching techniques to provide "better match" definitions that
are selected analogously to specializations in C++ templates. When expressed
through pattern matching, this may enable things beyond just template parameter
specialization, but that is an area that we want to explore cautiously.

> **TODO:** lots more work to flesh this out needs to be done...

## Metaprogramming

> References: [Metaprogramming](metaprogramming.md)
>
> **TODO:** References need to be evolved. Needs a detailed design and a high
> level summary provided inline.

Carbon provides metaprogramming facilities that look similar to regular Carbon
code. These are structured, and do not offer arbitrary inclusion or
preprocessing of source text such as C/C++ does.

## Execution abstractions

Carbon provides some higher-order abstractions of program execution, as well as
the critical underpinnings of such abstractions.

### Abstract machine and execution model

> **TODO:** Needs a feature design and a high level summary provided inline.

### Lambdas

> **TODO:** Needs a feature design and a high level summary provided inline.

### Co-routines

> **TODO:** Needs a feature design and a high level summary provided inline.

## Bidirectional interoperability with C/C++

> References:
> [Bidirectional interoperability with C/C++](interoperability/README.md)
>
> **TODO:** References need to be evolved. Needs a detailed design and a high
> level summary provided inline.
