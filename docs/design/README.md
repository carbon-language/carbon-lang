# Language design

<!--
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

<!-- toc -->

## Table of contents

-   [Overview](#overview)
    -   [A note on example code](#a-note-on-example-code)
-   [Hello, Carbon](#hello-carbon)
-   [Code and comments](#code-and-comments)
-   [Build modes](#build-modes)
-   [Types](#types)
    -   [Primitive types](#primitive-types)
        -   [`bool`](#bool)
        -   [Integer types](#integer-types)
            -   [Integer literals](#integer-literals)
        -   [Floating-point types](#floating-point-types)
            -   [Floating-point literals](#floating-point-literals)
        -   [String types](#string-types)
            -   [String literals](#string-literals)
    -   [Composite types](#composite-types)
        -   [Tuples](#tuples)
        -   [Structural and nominal types](#structural-and-nominal-types)
        -   [Struct types](#struct-types)
        -   [Pointer types](#pointer-types)
        -   [Arrays and slices](#arrays-and-slices)
-   [Functions](#functions)
    -   [Blocks and statements](#blocks-and-statements)
    -   [Expressions](#expressions)
    -   [Variables](#variables)
    -   [`let`](#let)
    -   [`auto`](#auto)
    -   [Pattern matching](#pattern-matching)
        -   [Pattern matching in local variables](#pattern-matching-in-local-variables)
    -   [Control flow](#control-flow)
        -   [`if` and `else`](#if-and-else)
        -   [Loops](#loops)
            -   [`while`](#while)
            -   [`for`](#for)
            -   [`break`](#break)
            -   [`continue`](#continue)
        -   [`return`](#return)
            -   [`returned var`](#returned-var)
        -   [`match` control flow](#match-control-flow)
-   [User-defined types](#user-defined-types)
    -   [Classes](#classes)
        -   [Assignment, copying](#assignment-copying)
        -   [Member access](#member-access)
        -   [Methods](#methods)
        -   [Inheritance](#inheritance)
        -   [Access control](#access-control)
        -   [Destructors](#destructors)
    -   [Variants](#variants)
-   [Names](#names)
    -   [Packages, libraries, namespaces](#packages-libraries-namespaces)
    -   [Names and scopes](#names-and-scopes)
    -   [Naming conventions](#naming-conventions)
    -   [Aliases](#aliases)
    -   [Name lookup](#name-lookup)
        -   [Name lookup for common types](#name-lookup-for-common-types)
    -   [Name visibility](#name-visibility)
-   [Generics](#generics)
    -   [Interfaces and implementations](#interfaces-and-implementations)
    -   [Checked and template parameters](#checked-and-template-parameters)
        -   [Templates](#templates)
    -   [Generic functions](#generic-functions)
        -   [Functions with template parameters](#functions-with-template-parameters)
    -   [Generic types](#generic-types)
        -   [Types with template parameters](#types-with-template-parameters)
    -   [Operator overloading](#operator-overloading)
        -   [Implicit and explicit conversion](#implicit-and-explicit-conversion)
        -   [Comparison operators](#comparison-operators)
        -   [Arithmetic operators](#arithmetic-operators)
        -   [Bitwise and shift operators](#bitwise-and-shift-operators)
        -   [Common type](#common-type)
-   [Bidirectional interoperability with C/C++](#bidirectional-interoperability-with-cc)
-   [Unfinished tales](#unfinished-tales)
    -   [Pattern matching as function overload resolution](#pattern-matching-as-function-overload-resolution)
    -   [Lifetime and move semantics](#lifetime-and-move-semantics)
    -   [Metaprogramming](#metaprogramming)
    -   [Execution abstractions](#execution-abstractions)
        -   [Abstract machine and execution model](#abstract-machine-and-execution-model)
        -   [Lambdas](#lambdas)
        -   [Co-routines](#co-routines)

<!-- tocstop -->

## Overview

This documentation describes the design of the Carbon language, and the
rationale for that design. This documentation is an overview of the Carbon
project in its current state, written for the builders of Carbon and for those
interested in learning more about Carbon.

This document is _not_ a complete programming manual, and, nor does it provide
detailed and comprehensive justification for design decisions. These
descriptions are found in linked dedicated designs.

### A note on example code

Some syntax used in example code is provisional or placeholder, and may change
later.

## Hello, Carbon

Here is a simple function showing some Carbon code:

```carbon
import Console;

// Prints the Fibonacci numbers less than `limit`.
fn Fibonacci(limit: i64) {
  var (a, b): (i64, i64) = (0, 1);
  while (a < limit) {
    Console.Print(a, " ");
    (a, b) = (b, a + b);
  }
  Console.Print("\n");
}
```

Carbon is a language that should feel familiar to C++ and C developers. This
example has familiar constructs like imports, function definitions, typed
arguments, and curly braces.

A few other features that are unlike C or C++ may stand out. First, declarations
start with introducer keywords. `fn` introduces a function declaration, and
`var` introduces a variable declaration. You can also see a _tuple_, a composite
type written as a comma-separated list inside parentheses. Unlike, say, Python,
these types are strongly-typed as well.

## Code and comments

> References:
>
> -   [Source files](code_and_name_organization/source_files.md)
> -   [lexical conventions](lexical_conventions)
> -   Proposal
>     [#142: Unicode source files](https://github.com/carbon-language/carbon-lang/pull/142)
> -   Proposal
>     [#198: Comments](https://github.com/carbon-language/carbon-lang/pull/198)

All source code is UTF-8 encoded text. Comments, identifiers, and strings are
allowed to have non-ASCII characters.

```carbon
var résultat: String = "Succès";
```

Comments start with two slashes `//` and go to the end of the line. They are
required to be the only non-whitespace on the line.

```carbon
// Compute an approximation of π
```

## Build modes

The behavior of Carbon programs depends on the _build mode_:

-   In a _development build_, the priority is diagnosing problems and fast build
    time.
-   In a _performance build_, the priority is fastest execution time and lowest
    memory usage.
-   In a _hardened build_, the first priority is safety and second is
    performance.

## Types

Carbon's core types are broken down into three categories:

-   Primitive types
-   Composite types
-   [User-defined types](#user-defined-types)

The first two are intrinsic and directly built in the language and are discussed
in this section. The last category of types allows for defining new types, and
is described [later](#user-defined-types).

Expressions compute values in Carbon, and these values are always strongly typed
much like in C++. However, an important difference from C++ is that types are
themselves modeled as values; specifically, compile-time constant values.
However, in simple cases this doesn't make much difference.

### Primitive types

> References: [Primitive types](primitive_types.md)

These types are fundamental to the language as they aren't either formed from or
modifying other types. They also have semantics that are defined from first
principles rather than in terms of other operations. These will be made
available through the [prelude package](#name-lookup-for-common-types).

Primitive types fall into the following categories:

-   the boolean type `bool`,
-   signed and unsigned integer types,
-   IEEE-754 floating-point types, and
-   string types.

#### `bool`

The type `bool` is a boolean type with two possible values: `true` and `false`.
[Comparison expressions](#expressions) produce `bool` values. The condition
arguments in [control-flow statements](#control-flow), like [`if`](#if-and-else)
and [`while`](#while), and
[`if`-`then`-`else` conditional expressions](#expressions) take `bool` values.

#### Integer types

> References:
>
> -   Question-for-leads issue
>     [#543: pick names for fixed-size integer types](https://github.com/carbon-language/carbon-lang/issues/543)
> -   Proposal
>     [#820: Implicit conversions](https://github.com/carbon-language/carbon-lang/pull/820)
> -   Proposal
>     [#1083: Arithmetic expressions](https://github.com/carbon-language/carbon-lang/pull/1083)

The signed-integer type with bit width `N` may be written `Carbon.Int(N)`. For
convenience and brevity, the common power-of-two sizes may be written with an
`i` followed by the size: `i8`, `i16`, `i32`, `i64`, `i128`, or `i256`.
Signed-integer
[overflow](expressions/arithmetic.md#overflow-and-other-error-conditions) is a
programming error:

-   In a development build, overflow will be caught immediately when it happens
    at runtime.
-   In a performance build, the optimizer can assume that such conditions don't
    occur. As a consequence, if they do, the behavior of the program is not
    defined.
-   In a hardened build, overflow does not result in undefined behavior.
    Instead, either the program will be aborted, or the arithmetic will evaluate
    to a mathematically incorrect result, such as a two's complement result or
    zero.

The unsigned-integer types are: `u8`, `u16`, `u32`, `u64`, `u128`, `u256`, and
`Carbon.UInt(N)`. Unsigned integer types wrap around on overflow, we strongly
advise that they are not used except when those semantics are desired. These
types are intended for [hashing](https://en.wikipedia.org/wiki/Hash_function),
[cryptography](https://en.wikipedia.org/wiki/Cryptography), and
[PRNG](https://en.wikipedia.org/wiki/Pseudorandom_number_generator) use cases.
Values which can never be negative, like sizes, but for which wrapping does not
make sense
[should use signed integer types](/proposals/p1083.md#dont-let-unsigned-arithmetic-wrap).

##### Integer literals

> References:
>
> -   [Integer literals](lexical_conventions/numeric_literals.md#integer-literals)
> -   Proposal
>     [#143: Numeric literals](https://github.com/carbon-language/carbon-lang/pull/143)
> -   Proposal
>     [#144: Numeric literal semantics](https://github.com/carbon-language/carbon-lang/pull/144)
> -   Proposal
>     [#820: Implicit conversions](https://github.com/carbon-language/carbon-lang/pull/820)

Integers may be written in decimal, hexadecimal, or binary:

-   `12345` (decimal)
-   `0x1FE` (hexadecimal)
-   `0b1010` (binary)

Underscores `_` may be as a digit separator, but only in conventional locations.
Numeric literals are case-sensitive: `0x`, `0b` must be lowercase, whereas
hexadecimal digits must be uppercase. Integer literals never contain a `.`.

Unlike in C++, literals do not have a suffix to indicate their type. Instead,
numeric literals have a type derived from their value, and can be implicitly
converted to any type that can represent that value.

#### Floating-point types

> References:
>
> -   Question-for-leads issue
>     [#543: pick names for fixed-size integer types](https://github.com/carbon-language/carbon-lang/issues/543)
> -   Proposal
>     [#820: Implicit conversions](https://github.com/carbon-language/carbon-lang/pull/820)
> -   Proposal
>     [#1083: Arithmetic expressions](https://github.com/carbon-language/carbon-lang/pull/1083)

Floating-point types in Carbon have IEEE 754 semantics, use the round-to-nearest
rounding mode, and do not set any floating-point exception state. They are named
with an `f` and the number of bits: `f16`, `f32`, `f64`, and `f128`.
[`BFloat16`](primitive_types.md#bfloat16) is also provided.

##### Floating-point literals

> References:
>
> -   [Real-number literals](lexical_conventions/numeric_literals.md#real-number-literals)
> -   Proposal
>     [#143: Numeric literals](https://github.com/carbon-language/carbon-lang/pull/143)
> -   Proposal
>     [#144: Numeric literal semantics](https://github.com/carbon-language/carbon-lang/pull/144)
> -   Proposal
>     [#820: Implicit conversions](https://github.com/carbon-language/carbon-lang/pull/820)
> -   Proposal
>     [#866: Allow ties in floating literals](https://github.com/carbon-language/carbon-lang/pull/866)

Decimal and hexadecimal real-number literals are supported:

-   `123.456` (digits on both sides of the `.`)
-   `123.456e789` (optional `+` or `-` after the `e`)
-   `0x1.Ap123` (optional `+` or `-` after the `p`)

Real-number literals always have a period (`.`) and a digit on each side of the
period. When a real-number literal is interpreted as a value of a floating-point
type, its value is the representable real number closest to the value of the
literal. In the case of a tie, the nearest value whose mantissa is even is
selected.

#### String types

There are two string types:

-   `String` - a byte sequence treated as containing UTF-8 encoded text.
-   `StringView` - a read-only reference to a byte sequence treated as
    containing UTF-8 encoded text.

##### String literals

> References:
>
> -   [String literals](lexical_conventions/string_literals.md)
> -   Proposal
>     [#199: String literals](https://github.com/carbon-language/carbon-lang/pull/199)

String literals may be written on a single line using a double quotation mark
(`"`) at the beginning and end of the string, as in `"example"`.

Multi-line string literals, called _block string literals_, begin and end with
three double quotation marks (`"""`), and may have a file type indicator after
the first `"""`.

```carbon
// Block string literal:
var block: String = """
    The winds grow high; so do your stomachs, lords.
    How irksome is this music to my heart!
    When such strings jar, what hope of harmony?
    I pray, my lords, let me compound this strife.
        -- History of Henry VI, Part II, Act II, Scene 1, W. Shakespeare
    """;
```

The indentation of a block string literal's terminating line is removed from all
preceding lines.

Strings may contain
[escape sequences](lexical_conventions/string_literals.md#escape-sequences)
introduced with a backslash (`\`).
[Raw string literals](lexical_conventions/string_literals.md#raw-string-literals)
are available for representing strings with `\`s and `"`s.

### Composite types

#### Tuples

> References: [Tuples](tuples.md)

A tuple is a simple aggregation of types. In formal type theory, tuples are
product types. An example use of tuples is to return multiple values from a
function:

```carbon
fn DoubleBoth(x: i32, y: i32) -> (i32, i32) {
  return (2 * x, 2 * y);
}
```

Breaking this example apart:

-   The return type is a tuple of two `i32` types.
-   The expression uses tuple syntax to build a tuple of two `i32` values.

Both of these are expressions using the tuple syntax
`(<expression>, <expression>)`. The only difference is the type of the tuple
expression: one is a tuple of types, the other a tuple of values.

The components of a tuple are accessed positionally, so element access uses
subscript syntax:

```carbon
fn DoubleTuple(x: (i32, i32)) -> (i32, i32) {
  return (2 * x[0], 2 * x[1]);
}
```

#### Structural and nominal types

Tuple types are _structural_, which means two tuple types are equal if they have
the same components. This is in contrast to _nominal_ types that have a name
that identifies a specific declaration. Two nominal types are equal if their
names resolve to the same declaration.

#### Struct types

> References:
>
> -   [Struct types](classes.md#struct-types)
> -   Proposal
>     [#561: Basic classes: use cases, struct literals, struct types, and future work](https://github.com/carbon-language/carbon-lang/pull/561)
> -   Proposal
>     [#981: Implicit conversions for aggregates](https://github.com/carbon-language/carbon-lang/pull/981)
> -   Proposal
>     [#710: Default comparison for data classes](https://github.com/carbon-language/carbon-lang/issues/710)

The other [structural type](#structural-and-nominal-types) is called a
_structural data class_, also known as a _struct type_ or _struct_. In contrast
to a tuple, a struct's members are identified by name instead of position.

Both struct types and values are written inside curly braces (`{`...`}`). In
both cases, they have a comma-separated list of members that start with a period
(`.`) followed by the field name.

-   In a struct type, the field name is followed by a colon (`:`) and the type,
    as in: `{.name: String, .count: i32}`.
-   In a struct value, called a _structural data class literal_ or a _struct
    literal_, the field name is followed by an equal sign (`=`) and the value,
    as in `{.key = "Joe", .count = 3}`.

#### Pointer types

> References:
>
> -   Question-for-leads issue
>     [#520: should we use whitespace-sensitive operator fixity?](https://github.com/carbon-language/carbon-lang/issues/520)
> -   Question-for-leads issue
>     [#523: what syntax should we use for pointer types?](https://github.com/carbon-language/carbon-lang/issues/523)

The type of pointers-to-values-of-type-`T` is written `T*`. Carbon pointers do
not support
[pointer arithmetic](<https://en.wikipedia.org/wiki/Pointer_(computer_programming)>),
the only pointer [operations](#expressions) are:

-   Dereference: given a pointer `p`, `*p` gives the value `p` points to as an
    [l-value](<https://en.wikipedia.org/wiki/Value_(computer_science)#lrvalue>).
-   Address-of: given an
    [l-value](<https://en.wikipedia.org/wiki/Value_(computer_science)#lrvalue>)
    `x`, `&x` returns a pointer to `x`.

In Carbon, one use of pointers is to pass `&x` into a function that will modify
`x`.

#### Arrays and slices

> **TODO:**

## Functions

> References:
>
> -   [Functions](functions.md)
> -   Proposal
>     [#162: Basic Syntax](https://github.com/carbon-language/carbon-lang/pull/162)
> -   Proposal
>     [#438: Add statement syntax for function declarations](https://github.com/carbon-language/carbon-lang/pull/438)
> -   Question-for-leads issue
>     [#476: Optional argument names (unused arguments)](https://github.com/carbon-language/carbon-lang/issues/476)

Functions are the core unit of behavior. For example, this declares a function
that adds two 64-bit integers:

```carbon
fn Add(a: i64, b: i64) -> i64;
```

Breaking this apart:

-   `fn` is the keyword used to indicate a function.
-   Its name is `Add`.
-   It accepts two `i64` parameters, `a` and `b`.
-   It returns an `i64` result.

You would call this function like `Add(1, 2)`.

This just declares the function, a definition that includes the body that
defines what this function does would follow.

### Blocks and statements

> References:
>
> -   [Blocks and statements](blocks_and_statements.md)
> -   Proposal
>     [#162: Basic Syntax](https://github.com/carbon-language/carbon-lang/pull/162)

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

### Expressions

> References:
>
> -   [Expressions](expressions/)
> -   Proposal
>     [#162: Basic Syntax](https://github.com/carbon-language/carbon-lang/pull/162)
> -   Proposal
>     [#555: Operator precedence](https://github.com/carbon-language/carbon-lang/pull/555)
> -   Proposal
>     [#601: Operator tokens](https://github.com/carbon-language/carbon-lang/pull/601)
> -   Proposal
>     [#680: And, or, not](https://github.com/carbon-language/carbon-lang/pull/680)
> -   Proposal
>     [#702: Comparison operators](https://github.com/carbon-language/carbon-lang/pull/702)
> -   Proposal
>     [#845: as expressions](https://github.com/carbon-language/carbon-lang/pull/845)
> -   Proposal
>     [#911: Conditional expressions](https://github.com/carbon-language/carbon-lang/pull/911)
> -   Proposal
>     [#1083: Arithmetic expressions](https://github.com/carbon-language/carbon-lang/pull/1083)

Expressions describe some computed value. The simplest example would be a
literal number like `42`: an expression that computes the integer value 42.

Some common expressions in Carbon include:

-   Literals: `42`, `3.1419`, `"Hello World!"`
-   Operators:

    -   Increment and decrement: `++i`, `--j`
        -   These do not return any result.
    -   Unary negation: `-x`
    -   Arithmetic: `1 + 2`, `3 - 4`, `2 * 5`, `6 / 3`
    -   Bitwise: `2 & 3`, `2 | 4`, `3 ^ 1`, `^7`
    -   Bit shift: `1 << 3`, `8 >> 1`
    -   Comparison: `2 == 2`, `3 != 4`, `5 < 6`, `7 > 6`, `8 <= 8`, `8 >= 8`
    -   Logical: `a and b`, `c or d`, `not e`
    -   Conditional: `if c then t else f`

-   Parenthesized expressions: `(7 + 8) * (3 - 1)`

### Variables

> References:
>
> -   [Variables](variables.md)
> -   Proposal
>     [#162: Basic Syntax](https://github.com/carbon-language/carbon-lang/pull/162)
> -   Proposal
>     [#257: Initialization of memory and variables](https://github.com/carbon-language/carbon-lang/pull/257)
> -   Proposal
>     [#339: Add `var <type> <identifier> [ = <value> ];` syntax for variables](https://github.com/carbon-language/carbon-lang/pull/339)
> -   Proposal
>     [#618: var ordering](https://github.com/carbon-language/carbon-lang/pull/618)

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

### `let`

> **TODO:**

### `auto`

> References:
>
> -   [Type inference](type_inference.md)
> -   Proposal
>     [#826: Function return type inference](https://github.com/carbon-language/carbon-lang/pull/826)
> -   Proposal
>     [#851: auto keyword for vars](https://github.com/carbon-language/carbon-lang/pull/851)

> **TODO:**

### Pattern matching

> References:
>
> -   [Pattern matching](pattern_matching.md)
> -   Proposal
>     [#162: Basic Syntax](https://github.com/carbon-language/carbon-lang/pull/162)
>
> **TODO:** References need to be evolved.

The most prominent mechanism to manipulate and work with types in Carbon is
pattern matching. This may seem like a deviation from C++, but in fact this is
largely about building a clear, coherent model for a fundamental part of C++:
overload resolution.

#### Pattern matching in local variables

> References: [Pattern matching](pattern_matching.md)
>
> **TODO:** References need to be evolved.

Value patterns may be used when declaring local variables to conveniently
destructure them and do other type manipulations. However, the patterns must
match at compile time, so a boolean predicate cannot be used directly.

An example use is:

```carbon
fn Bar() -> (i32, (f32, f32));
fn Foo() -> i32 {
  var (p: i32, _: auto) = Bar();
  return p;
}
```

To break this apart:

-   The `i32` returned by `Bar()` matches and is bound to `p`, then returned.
-   The `(f32, f32)` returned by `Bar()` matches and is discarded by `_: auto`.

### Control flow

> References:
>
> -   [Control flow](control_flow/README.md)
> -   Proposal
>     [#162: Basic Syntax](https://github.com/carbon-language/carbon-lang/pull/162)
> -   Proposal
>     [#623: Require braces](https://github.com/carbon-language/carbon-lang/pull/623)

Blocks of statements are generally executed sequentially. However, statements
are the primary place where this flow of execution can be controlled.

#### `if` and `else`

> References:
>
> -   [Control flow](control_flow/conditionals.md)
> -   Proposal
>     [#285: if/else](https://github.com/carbon-language/carbon-lang/pull/285)

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

> References: [Control flow](control_flow/loops.md)

##### `while`

> References:
>
> -   [Control flow](control_flow/loops.md#while)
> -   Proposal
>     [#340: Add C++-like `while` loops](https://github.com/carbon-language/carbon-lang/pull/340)

`while` statements loop for as long as the passed expression returns `True`. For
example, this prints `0`, `1`, `2`, then `Done!`:

```carbon
var x: i32 = 0;
while (x < 3) {
  Print(x);
  ++x;
}
Print("Done!");
```

##### `for`

> References:
>
> -   [Control flow](control_flow/loops.md#for)
> -   Proposal
>     [#353: Add C++-like `for` loops](https://github.com/carbon-language/carbon-lang/pull/353)

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

> References:
>
> -   [Control flow](control_flow/return.md)
> -   [`return` statements](functions.md#return-statements)
> -   Proposal
>     [#415: return](https://github.com/carbon-language/carbon-lang/pull/415)
> -   Proposal
>     [#538: return with no argument](https://github.com/carbon-language/carbon-lang/pull/538)

The `return` statement ends the flow of execution within a function, returning
execution to the caller. If the function returns a value to the caller, that
value is provided by an expression in the return statement. For example:

```carbon
fn Sum(a: i32, b: i32) -> i32 {
  return a + b;
}
```

##### `returned var`

> References:
>
> -   [Control flow](control_flow/return.md#returned-var)
> -   Proposal
>     [#257: Initialization of memory and variables](https://github.com/carbon-language/carbon-lang/pull/257)

> **TODO:**

#### `match` control flow

> References: [Pattern matching](pattern_matching.md)
>
> **TODO:** References need to be evolved.

`match` is a control flow similar to `switch` of C/C++ and mirrors similar
constructs in other languages, such as Swift.

An example `match` is:

```carbon
fn Bar() -> (i32, (f32, f32));

fn Foo() -> f32 {
  match (Bar()...) {
    case (42, (x: f32, y: f32)) => {
      return x - y;
    }
    case (p: i32, (x: f32, _: f32)) if (p < 13) => {
      return p * x;
    }
    case (p: i32, _: auto) if (p > 3) => {
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
-   Each `case` pattern contains a value pattern, such as `(p: i32, _: auto)`,
    followed by an optional boolean predicate introduced by the `if` keyword.
    -   The value pattern must first match, and then the predicate must also
        evaluate to true for the overall `case` pattern to match.
    -   Using `auto` for a type will always match.

Value patterns may be composed of the following:

-   An expression, such as `42`, whose value must be equal to match.
-   An identifier to bind the value, followed by a `:` and followed by a type,
    such as `i32`.
    -   The special identifier `_` may be used to discard the value once
        matched.
-   A destructuring pattern containing a sequence of value patterns, such as
    `(x: f32, y: f32)`, which match against tuples and tuple-like values by
    recursively matching on their elements.
-   An unwrapping pattern containing a nested value pattern which matches
    against a variant or variant-like value by unwrapping it.

## User-defined types

### Classes

> References:
>
> -   [Classes](classes.md#nominal-class-types)
> -   Proposal
>     [#722: Nominal classes and methods](https://github.com/carbon-language/carbon-lang/pull/722)

_Nominal classes_, or just _classes_, are a way for users to define their own
data strutures or record types.

For example:

```carbon
class Widget {
  var x: i32;
  var y: i32;
  var z: i32;

  var payload: String;
}
```

Breaking apart `Widget`:

-   `Widget` has three `i32` members: `x`, `y`, and `z`.
-   `Widget` has one `String` member: `payload`.
-   Given an instance `dial`, a member can be referenced with `dial.paylod`.

The order of the member declarations determines the members' memory-layout
order.

Both [structural data classes](#struct-types) and nominal classes are considered
class types, but they are commonly referred to as "structs" and "classes"
respectively when that is not confusing. Like structs, classes refer to their
members by name. Unlike structs, classes are
[nominal types](#structural-and-nominal-types).

#### Assignment, copying

> References: [Classes](classes.md#construction)
>
> -   Proposal
>     [#981: Implicit conversions for aggregates](https://github.com/carbon-language/carbon-lang/pull/981)

You may use a [struct literal](#struct-types), to assign or initialize a
variable with a class type.

```carbon
var sprocket: Widget = {.x = 3, .y = 4, .z = 5, .payload = "Sproing"};
sprocket = {.x = 2, .y = 1, .z = 0, .payload = "Bounce"};
```

You may also copy one struct into another of the same type.

```carbon
var thingy: Widget = sprocket;
sprocket = thingy;
```

#### Member access

> References: Proposal
> [#989: Member access expressions](https://github.com/carbon-language/carbon-lang/pull/989)

The data members of a variable with a class type may be accessed using dot `.`
notation:

```carbon
Assert(sprocket.x == thingy.x);
```

#### Methods

> References:
>
> -   [Methods](classes.md#methods)
> -   Proposal
>     [#722: Nominal classes and methods](https://github.com/carbon-language/carbon-lang/pull/722)

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

#### Inheritance

> References:
>
> -   [Inheritance](classes.md#inheritance)
> -   Proposal
>     [#777: Inheritance](https://github.com/carbon-language/carbon-lang/pull/777)
> -   Proposal
>     [#820: Implicit conversions](https://github.com/carbon-language/carbon-lang/pull/820)

> **TODO:**

#### Access control

> References:
>
> -   [Access control for class members](classes.md#access-control)
> -   Question-for-leads issue
>     [#665: `private` vs `public` _syntax_ strategy, as well as other visibility tools like `external`/`api`/etc.](https://github.com/carbon-language/carbon-lang/issues/665)
> -   Question-for-leads issue
>     [#971: Private interfaces in public API files](https://github.com/carbon-language/carbon-lang/issues/971)

> **TODO:**

#### Destructors

> References:
>
> -   [Destructors](classes.md#destructors)
> -   Proposal
>     [#1154: Destructors](https://github.com/carbon-language/carbon-lang/pull/1154)

> **TODO:**

### Variants

FIXME: Rename to "Choice types"

> References:
>
> -   Proposal
>     [#157: Design direction for sum types](https://github.com/carbon-language/carbon-lang/pull/157)
> -   Proposal
>     [#162: Basic Syntax](https://github.com/carbon-language/carbon-lang/pull/162)

> **TODO:**

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

> **TODO:**

### Interfaces and implementations

> **TODO:**

### Checked and template parameters

> References: Proposal
> [#989: Member access expressions](https://github.com/carbon-language/carbon-lang/pull/989)

> **TODO:**

#### Templates

> References: [Templates](templates.md)
>
> **TODO:** References need to be evolved.

Carbon templates follow the same fundamental paradigm as C++ templates: they are
instantiated when called, resulting in late type checking, duck typing, and lazy
binding. Although generics are generally preferred, templates enable translation
of code between C++ and Carbon, and address some cases where the type checking
rigor of generics are problematic.

### Generic functions

> **TODO:**

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

Here we deduce one type parameter and explicitly pass another. It is not
possible to explicitly pass a deduced type parameter; instead the call site
should cast or convert the argument to control the deduction. In this particular
example, the explicit type is passed after a runtime parameter. While this makes
that type unavailable to the declaration of _that_ runtime parameter, it still
is a _template_ parameter and available to use as a type in the remaining parts
of the function declaration.

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

### Operator overloading

> References:
>
> -   [Operator overloading](generics/details.md#operator-overloading)
> -   Question-for-leads issue
>     [#1058: How should interfaces for core functionality be named?](https://github.com/carbon-language/carbon-lang/issues/1058)
> -   Proposal
>     [#1178: Rework operator interfaces](https://github.com/carbon-language/carbon-lang/pull/1178)

> **TODO:**

#### Implicit and explicit conversion

> References:
>
> -   [Implicit conversions](expressions/implicit_conversions.md#extensibility)
> -   [`as` expressions](expressions/as_expressions.md#extensibility)
> -   Proposal
>     [#820: Implicit conversions](https://github.com/carbon-language/carbon-lang/pull/820)
> -   Proposal
>     [#845: as expressions](https://github.com/carbon-language/carbon-lang/pull/845)

> **TODO:**

#### Comparison operators

> References:
>
> -   [Comparison operators](expressions/comparison_operators.md#extensibility)
> -   Proposal
>     [#702: Comparison operators](https://github.com/carbon-language/carbon-lang/pull/702)
> -   Proposal
>     [#1178: Rework operator interfaces](https://github.com/carbon-language/carbon-lang/pull/1178)

> **TODO:**

#### Arithmetic operators

> References:
>
> -   [Arithmetic expressions](expressions/arithmetic.md#extensibility)
> -   Proposal
>     [#1083: Arithmetic expressions](https://github.com/carbon-language/carbon-lang/pull/1083)

> **TODO:**

#### Bitwise and shift operators

> References:
>
> -   [Bitwise and shift operators](expressions/bitwise.md#extensibility)
> -   Proposal
>     [#1191: Bitwise operators](https://github.com/carbon-language/carbon-lang/pull/1191)

> **TODO:**

#### Common type

> References:
>
> -   [`if` expressions](expressions/if.md#finding-a-common-type)
> -   Proposal
>     [#911: Conditional expressions](https://github.com/carbon-language/carbon-lang/pull/911)

> **TODO:**

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
