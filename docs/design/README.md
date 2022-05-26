# Language design

<!--
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

<!-- toc -->

## Table of contents

-   [Overview](#overview)
    -   [This document is provisional](#this-document-is-provisional)
-   [Hello, Carbon](#hello-carbon)
-   [Code and comments](#code-and-comments)
-   [Build modes](#build-modes)
-   [Types are values](#types-are-values)
    -   [Structural and nominal types](#structural-and-nominal-types)
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
    -   [Struct types](#struct-types)
    -   [Pointer types](#pointer-types)
    -   [Arrays and slices](#arrays-and-slices)
-   [Expressions](#expressions)
-   [Declarations, Definitions, and Scopes](#declarations-definitions-and-scopes)
-   [Functions](#functions)
    -   [Blocks and statements](#blocks-and-statements)
    -   [Variables](#variables)
    -   [`let`](#let)
    -   [`auto`](#auto)
    -   [Pattern matching](#pattern-matching)
    -   [Assignment statements](#assignment-statements)
    -   [Control flow](#control-flow)
        -   [`if` and `else`](#if-and-else)
        -   [Loops](#loops)
            -   [`while`](#while)
            -   [`for`](#for)
            -   [`break`](#break)
            -   [`continue`](#continue)
        -   [`return`](#return)
            -   [`returned var`](#returned-var)
        -   [`match`](#match)
-   [User-defined types](#user-defined-types)
    -   [Classes](#classes)
        -   [Assignment, copying](#assignment-copying)
        -   [Class functions and factory functions](#class-functions-and-factory-functions)
        -   [Methods](#methods)
        -   [Inheritance](#inheritance)
        -   [Access control](#access-control)
        -   [Destructors](#destructors)
    -   [Choice types](#choice-types)
-   [Names](#names)
    -   [Packages, libraries, namespaces](#packages-libraries-namespaces)
    -   [Legal names](#legal-names)
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
        -   [Generic choice types](#generic-choice-types)
    -   [Operator overloading](#operator-overloading)
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

### This document is provisional

This document includes much that is provisional or placeholder. This means that
the syntax used, language rules, standard library, and other aspects of the
design have things that have not been decided through the Carbon process. This
preliminary material fills in gaps until aspects of the design can be filled in.

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

> References:
>
> -   [Source files](code_and_name_organization/source_files.md)
> -   [lexical conventions](lexical_conventions)
> -   Proposal
>     [#142: Unicode source files](https://github.com/carbon-language/carbon-lang/pull/142)
> -   Proposal
>     [#198: Comments](https://github.com/carbon-language/carbon-lang/pull/198)

## Build modes

The behavior of the Carbon compiler depends on the _build mode_:

-   In a _development build_, the priority is diagnosing problems and fast build
    time.
-   In a _performance build_, the priority is fastest execution time and lowest
    memory usage.
-   In a _hardened build_, the first priority is safety and second is
    performance.

> References: [Safety strategy](/docs/project/principles/safety_strategy.md)

## Types are values

Expressions compute values in Carbon, and these values are always strongly typed
much like in C++. However, an important difference from C++ is that types are
themselves modeled as values; specifically, compile-time constant values. This
means that the grammar for writing a type is the expression](#expressions)
grammar. Expressions written where a type is expected must be able to be
evaluated at compile-time and must evaluate to a type value.

### Structural and nominal types

Some types are _structural_, which means they are equal if they have the same
components. This is in contrast to _nominal_ types that have a name that
identifies a specific definition. Two nominal types are equal if their names
resolve to the same definition. If a nominal type is [generic](#generics), and
so has parameters, then the parameters must also be equal for the types to be
equal.

## Primitive types

Some types are used as the building blocks for other types and are made
available through the [prelude package](#name-lookup-for-common-types).

Primitive types fall into the following categories:

-   the boolean type `bool`,
-   signed and unsigned integer types,
-   IEEE-754 floating-point types, and
-   string types.

> References: [Primitive types](primitive_types.md)

### `bool`

The type `bool` is a boolean type with two possible values: `true` and `false`.
[Comparison expressions](#expressions) produce `bool` values. The condition
arguments in [control-flow statements](#control-flow), like [`if`](#if-and-else)
and [`while`](#while), and
[`if`-`then`-`else` conditional expressions](#expressions) take `bool` values.

### Integer types

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

> References:
>
> -   Question-for-leads issue
>     [#543: pick names for fixed-size integer types](https://github.com/carbon-language/carbon-lang/issues/543)
> -   Proposal
>     [#820: Implicit conversions](https://github.com/carbon-language/carbon-lang/pull/820)
> -   Proposal
>     [#1083: Arithmetic expressions](https://github.com/carbon-language/carbon-lang/pull/1083)

#### Integer literals

Integers may be written in decimal, hexadecimal, or binary:

-   `12345` (decimal)
-   `0x1FE` (hexadecimal)
-   `0b1010` (binary)

Underscores `_` may be as a digit separator, but for decimal and hexadecimal
literals, they can only appear in conventional locations. Numeric literals are
case-sensitive: `0x`, `0b` must be lowercase, whereas hexadecimal digits must be
uppercase. Integer literals never contain a `.`.

Unlike in C++, literals do not have a suffix to indicate their type. Instead,
numeric literals have a type derived from their value, and can be
[implicitly converted](expressions/implicit_conversions.md) to any type that can
represent that value.

> References:
>
> -   [Integer literals](lexical_conventions/numeric_literals.md#integer-literals)
> -   Proposal
>     [#143: Numeric literals](https://github.com/carbon-language/carbon-lang/pull/143)
> -   Proposal
>     [#144: Numeric literal semantics](https://github.com/carbon-language/carbon-lang/pull/144)
> -   Proposal
>     [#820: Implicit conversions](https://github.com/carbon-language/carbon-lang/pull/820)

### Floating-point types

Floating-point types in Carbon have IEEE 754 semantics, use the round-to-nearest
rounding mode, and do not set any floating-point exception state. They are named
with an `f` and the number of bits: `f16`, `f32`, `f64`, and `f128`.
[`BFloat16`](primitive_types.md#bfloat16) is also provided.

> References:
>
> -   Question-for-leads issue
>     [#543: pick names for fixed-size integer types](https://github.com/carbon-language/carbon-lang/issues/543)
> -   Proposal
>     [#820: Implicit conversions](https://github.com/carbon-language/carbon-lang/pull/820)
> -   Proposal
>     [#1083: Arithmetic expressions](https://github.com/carbon-language/carbon-lang/pull/1083)

#### Floating-point literals

Decimal and hexadecimal real-number literals are supported:

-   `123.456` (digits on both sides of the `.`)
-   `123.456e789` (optional `+` or `-` after the `e`)
-   `0x1.Ap123` (optional `+` or `-` after the `p`)

Real-number literals always have a period (`.`) and a digit on each side of the
period. When a real-number literal is interpreted as a value of a floating-point
type, its value is the representable real number closest to the value of the
literal. In the case of a tie, the nearest value whose mantissa is even is
selected.

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

### String types

There are two string types:

-   `String` - a byte sequence treated as containing UTF-8 encoded text.
-   `StringView` - a read-only reference to a byte sequence treated as
    containing UTF-8 encoded text.

#### String literals

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

> References:
>
> -   [String literals](lexical_conventions/string_literals.md)
> -   Proposal
>     [#199: String literals](https://github.com/carbon-language/carbon-lang/pull/199)

## Composite types

### Tuples

A tuple is a fixed-size collection of values that can have different types,
where each value is identified by its position in the tuple. An example use of
tuples is to return multiple values from a function:

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
expression: one is a tuple of types, the other a tuple of values. In other
words, a tuple type is a tuple _of_ types.

The components of a tuple are accessed positionally, so element access uses
subscript syntax, but the index must be a compile-time constant:

```carbon
fn DoubleTuple(x: (i32, i32)) -> (i32, i32) {
  return (2 * x[0], 2 * x[1]);
}
```

Tuple types are [structural](#structural-and-nominal-types).

> References: [Tuples](tuples.md)

### Struct types

Carbon also has [structural types](#structural-and-nominal-types) whose members
are identified by name instead of position. These are called _structural data
classes_, also known as a _struct types_ or _structs_.

Both struct types and values are written inside curly braces (`{`...`}`). In
both cases, they have a comma-separated list of members that start with a period
(`.`) followed by the field name.

-   In a struct type, the field name is followed by a colon (`:`) and the type,
    as in: `{.name: String, .count: i32}`.
-   In a struct value, called a _structural data class literal_ or a _struct
    literal_, the field name is followed by an equal sign (`=`) and the value,
    as in `{.key = "Joe", .count = 3}`.

> References:
>
> -   [Struct types](classes.md#struct-types)
> -   Proposal
>     [#561: Basic classes: use cases, struct literals, struct types, and future work](https://github.com/carbon-language/carbon-lang/pull/561)
> -   Proposal
>     [#981: Implicit conversions for aggregates](https://github.com/carbon-language/carbon-lang/pull/981)
> -   Proposal
>     [#710: Default comparison for data classes](https://github.com/carbon-language/carbon-lang/issues/710)

### Pointer types

The type of pointers-to-values-of-type-`T` is written `T*`. Carbon pointers do
not support
[pointer arithmetic](<https://en.wikipedia.org/wiki/Pointer_(computer_programming)>);
the only pointer [operations](#expressions) are:

-   Dereference: given a pointer `p`, `*p` gives the value `p` points to as an
    [l-value](<https://en.wikipedia.org/wiki/Value_(computer_science)#lrvalue>).
    `p->m` is syntactic sugar for `(*p).m`.
-   Address-of: given an
    [l-value](<https://en.wikipedia.org/wiki/Value_(computer_science)#lrvalue>)
    `x`, `&x` returns a pointer to `x`.

There are no [null pointers](https://en.wikipedia.org/wiki/Null_pointer) in
Carbon. To represent a pointer that may not refer to a valid object, use the
type `Optional(T*)`.

Pointers are the main Carbon mechanism for allowing a function to modify a
variable of the caller.

> References:
>
> -   Question-for-leads issue
>     [#520: should we use whitespace-sensitive operator fixity?](https://github.com/carbon-language/carbon-lang/issues/520)
> -   Question-for-leads issue
>     [#523: what syntax should we use for pointer types?](https://github.com/carbon-language/carbon-lang/issues/523)

### Arrays and slices

The type of an array of holding 4 `i32` values is written `[i32; 4]`. There is
an [implicit conversion](expressions/implicit_conversions.md) from tuples to
arrays of the same length as long as every component of the tuple may be
implicitly converted to the destination element type. In cases where the size of
the array may be deduced, it may be omitted, as in:

```carbon
var i: i32 = 1;
// `[i32;]` equivalent to `[i32; 3]` here.
var a: [i32;] = (i, i, i);
```

Elements of an array may be accessed using square brackets (`[`...`]`), as in
`a[i]`:

```carbon
a[i] = 2;
Console.Print(a[0]);
```

> **TODO:** Slices

## Expressions

Expressions describe some computed value. The simplest example would be a
literal number like `42`: an expression that computes the integer value 42.

Some common expressions in Carbon include:

-   Literals:

    -   [boolean](#bool): `true`, `false`
    -   [integer](#integer-literals): `42`, `-7`
    -   [real-number](#floating-point-literals): `3.1419`, `6.022e+23`
    -   [string](#string-literals): `"Hello World!"`
    -   [tuple](#tuples): `(1, 2, 3)`
    -   [struct](#struct-types): `{.word = "the", .count = 56}`

-   [Names](#names) and [member access](expressions/member_access.md)

-   [Operators](expressions#operators):

    -   [Arithmetic](expressions/arithmetic.md): `-x`, `1 + 2`, `3 - 4`,
        `2 * 5`, `6 / 3`, `5 % 3`
    -   [Bitwise](expressions/bitwise.md): `2 & 3`, `2 | 4`, `3 ^ 1`, `^7`
    -   [Bit shift](expressions/bitwise.md): `1 << 3`, `8 >> 1`
    -   [Comparison](expressions/comparison_operators.md): `2 == 2`, `3 != 4`,
        `5 < 6`, `7 > 6`, `8 <= 8`, `8 >= 8`
    -   [Conversion](expressions/as_expressions.md): `2 as i32`
    -   [Logical](expressions/logical_operators.md): `a and b`, `c or d`,
        `not e`
    -   [Indexing](#arrays-and-slices): `a[3]`
    -   [Function](#functions) call: `f(4)`
    -   [Pointer](#pointer-types): `*p`, `p->m`, `&x`

-   [Conditionals](expressions/if.md): `if c then t else f`
-   Parentheses: `(7 + 8) * (3 - 1)`

When an expression appears in a context in which an expression of a specific
type is expected, [implicit conversions](expressions/implicit_conversions.md)
are applied to convert the expression to the target type.

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

## Declarations, Definitions, and Scopes

_Declarations_ introduce a new [name](#names) and say what that name represents.
For some kinds of entities, like [functions](#functions), there are two kinds of
declarations: _forward declarations_ and _definitions_. In this case, there
should be exactly one definition for the name, but there can be additional
forward declarations that introduce the name before it is defined. Forward
declarations allow cyclic references, and can be used to declare a name in an
[api file](#packages-libraries-namespaces) that is defined in an
[impl file](#packages-libraries-namespaces).

A name is valid until the end of the innermost enclosing
[_scope_](<https://en.wikipedia.org/wiki/Scope_(computer_science)>). Except for
the outermost scope, scopes are enclosed in curly braces (`{`...`}`).

## Functions

Functions are the core unit of behavior. For example, this is a
[declaration](#declarations-definitions-and-scopes) of a function that adds two
64-bit integers:

```carbon
fn Add(a: i64, b: i64) -> i64;
```

Breaking this apart:

-   `fn` is the keyword used to introduce a function.
-   Its name is `Add`. This is the name added to the enclosing
    [scope](#declarations-definitions-and-scopes).
-   It accepts two `i64` parameters, `a` and `b`.
-   It returns an `i64` result.

You would call this function like `Add(1, 2)`.

A function definition is a function declaration that has a body block instead of
a semicolon:

```carbon
fn Add(a: i64, b: i64) -> i64 {
  return a + b;
}
```

The names of the parameters are in scope until the end of the definition or
declaration.

> References:
>
> -   [Functions](functions.md)
> -   Proposal
>     [#162: Basic Syntax](https://github.com/carbon-language/carbon-lang/pull/162)
> -   Proposal
>     [#438: Add statement syntax for function declarations](https://github.com/carbon-language/carbon-lang/pull/438)
> -   Question-for-leads issue
>     [#476: Optional argument names (unused arguments)](https://github.com/carbon-language/carbon-lang/issues/476)

### Blocks and statements

A _code block_ or _block_ is a sequence of _statements_. Blocks define a
[scope](#declarations-definitions-and-scopes) and, like other scopes, is
enclosed in curly braces (`{`...`}`). Each statement is terminated by a
semicolon, and can be one of:

-   an [expression](#expressions),
-   a [variable declaration](#variables),
-   a [`let` declaration](#let),
-   an [assignment statement](#assignment-statements), or
-   a [control-flow statement](#control-flow).

Statements within a block are normally executed in the order the appear in the
source code, except when modified by control-flow statements.

The body of a function is defined by a block, and some
[control-flow statements](#control-flow) have their own blocks of code. These
are nested within the enclosing scope. For example, here is a function
definition with a block of statements defining the body of the function, and a
nested block as part of a `while` statement:

```carbon
fn Foo() {
  Bar();
  while (Baz()) {
    Quux();
  }
}
```

> References:
>
> -   [Blocks and statements](blocks_and_statements.md)
> -   Proposal
>     [#162: Basic Syntax](https://github.com/carbon-language/carbon-lang/pull/162)

### Variables

Blocks introduce nested scopes and can contain variable
[declarations](#declarations-definitions-and-scopes) that are local to that
block, similarly to function parameters.

For example:

```carbon
fn DoSomething() -> i64 {
  var x: i64 = 42;
  x = x + 2;
  return x;
}
```

Breaking this apart:

-   `var` is the keyword used to indicate a variable.
-   Its name is `x`. This is the name added to the enclosing
    [scope](#declarations-definitions-and-scopes).
-   Its type is `i64`.
-   It is initialized with the value `42`.

Unlike function parameters, `x` is an
[l-value](<https://en.wikipedia.org/wiki/Value_(computer_science)#lrvalue>),
which means it has storage and an address, and so can be modified.

Note that there are no forward declarations of variables, all variable
declarations are [definitions](#declarations-definitions-and-scopes).

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

### `let`

To bind a name to a value without associating a specific storage location, use
`let` instead of `var`.

```carbon
fn DoSomething() -> i64 {
  let x: i64 = 42;
  return x + 2;
}
```

The `let` binds `x` to the _value_ `42`. `x` is an r-value, so it can not be
modified, for example by being the left side of an assignment statement, and its
address cannot be taken.

Function parameters are passed by value, and so act like they were defined in a
`let` implicitly. **FIXME:** Is this just the default, or can you write `var` in
a function signature to give parameters dedicated storage so they may be
modified?

### `auto`

The keyword `auto` may be used in place of the type in a `var` or `let`
statement in a function body. In this case, the type is the static type of the
initializer expression.

```
var x: i64 = 2;
// The type of `y` is inferred to be `i64`.
let y: auto = x + 3;
// The type of `z` is inferred to be `bool`.
var z: auto = (y > 1);
```

It may also be used as the return type in a function definition. In this case,
the body of the function must have exactly one `return` statement, and the
return type of the function is set to the static type of the expression argument
of that `return`.

```
// Return type is inferred to be `bool`, the type of `a > 0`.
fn Positive(a: i64) -> auto {
  return a > 0;
}
```

Note that `auto` is not allowed in a function declaration without a function
body.

> References:
>
> -   [Type inference](type_inference.md)
> -   [Function return clause](functions.md#return-clause)
> -   Proposal
>     [#826: Function return type inference](https://github.com/carbon-language/carbon-lang/pull/826)
> -   Proposal
>     [#851: auto keyword for vars](https://github.com/carbon-language/carbon-lang/pull/851)

### Pattern matching

Patterns are used in a variety of Carbon language constructs, including
[function parameters](#functions), [variable declarations](#variables), and
[`let` declarations](#let).

The most common pattern is a _binding pattern_, consisting of a new name, a
colon (`:`), and a type. It can only match values that may be
[implicitly converted](expressions/implicit_conversions.md) to that type. An `_`
may be used instead of the name to ignore the value.

> **TODO:** Using `var` before a binding pattern to allocate storage so the new
> name can be modified?

There are also _destructuring patterns_, such as _tuple destructuring_. A tuple
destructuring pattern looks like a tuple of patterns. It may only be used to
match tuple values whose components match the component patterns of the tuple.
An example use is:

```carbon
// `Bar()` returns a tuple consisting of an
// `i32` value and 2-tuple of `f32` values.
fn Bar() -> (i32, (f32, f32));

fn Foo() -> i64 {
  // Pattern in `var` declaration:
  var (p: i64, _: auto) = Bar();
  return p;
}
```

The pattern used in the `var` declaration destructures the tuple value returned
by `Bar()`. The first component pattern, `p: i64`, corresponds to the first
component of the value returned by `Bar()`, which has type `i32`. This is
allowed since there is an implicit conversion from `i32` to `i64`. The result of
this conversion is assigned to the name `p`. The second component pattern,
`_: auto`, matches the second component of the value returned by `Bar()`, which
has type `(f32, f32)`. The [`auto`](#auto) matches any type and the `_` means
the value is discarded.

Additional kinds of patterns are allowed in [`match` statements](#match).

> References:
>
> -   [Pattern matching](pattern_matching.md)
> -   Proposal
>     [#162: Basic Syntax](https://github.com/carbon-language/carbon-lang/pull/162)

### Assignment statements

Assignment statements mutate the value of the
[l-value](<https://en.wikipedia.org/wiki/Value_(computer_science)#lrvalue>)
described on the left-hand side of the assignment.

-   Assignment: `x = y;`. `x` is assigned the value of `y`.
-   Destructuring assignment: `(x, y) = z;`. `z` must be a tuple with the same
    number of compents as the left-hand side. `x` is assigned the value of
    `z[0]` and `y` is assigned the value of `z[1]`.
-   Increment and decrement: `++i;`, `--j;`. `i` is set to `i + 1`, `j` is set
    to `j - 1`.
-   Compound assignment: `x += y;`, `x -= y;`, `x *= y;`, `x /= y;`, `x &= y;`,
    `x |= y;`, `x ^= y;`, `x <<= y;`, `x >>= y;`. `x @= y;` is equivalent to
    `x = x @ y;` for each operator `@`.

Unlike C++, these assignments are statements, not expressions, and don't return
a value.

### Control flow

Blocks of statements are generally executed sequentially. Control-flow
statements give additional control over the flow of execution and which
statements are executed.

Some control-flow statements include [blocks](#blocks-and-statements). Those
blocks will always be within curly braces `{`...`}`.

```carbon
// Curly braces { ... } are required.
if (condition) {
  ExecutedWhenTrue();
} else {
  ExecutedWhenFalse();
}
```

This is unlike C++, which allows control-flow constructs to omit curly braces
around a single statement.

> References:
>
> -   [Control flow](control_flow/README.md)
> -   Proposal
>     [#162: Basic Syntax](https://github.com/carbon-language/carbon-lang/pull/162)
> -   Proposal
>     [#623: Require braces](https://github.com/carbon-language/carbon-lang/pull/623)

#### `if` and `else`

`if` and `else` provide conditional execution of statements. An `if` statement
consists of:

-   An `if` introducer followed by a condition in parentheses. If the condition
    evaluates to `true`, the block following the condition is executed,
    otherwise it is skipped.
-   This may be followed by zero or more `else if` clauses, whose conditions are
    evaluated if all prior conditions evaluate to `false`, with a block that is
    executed if that evaluation is to `true`.
-   A final optional `else` clause, with a block that is executed if all
    conditions evaluate to `false`.

For example:

```carbon
if (fruit.IsYellow()) {
  Console.Print("Banana!");
} else if (fruit.IsOrange()) {
  Console.Print("Orange!");
} else {
  Console.Print("Vegetable!");
}
```

This code will:

-   Print `Banana!` if `fruit.IsYellow()` is `true`.
-   Print `Orange!` if `fruit.IsYellow()` is `false` and `fruit.IsOrange()` is
    `true`.
-   Print `Vegetable!` if both of the above return `false`.

> References:
>
> -   [Control flow](control_flow/conditionals.md)
> -   Proposal
>     [#285: if/else](https://github.com/carbon-language/carbon-lang/pull/285)

#### Loops

> References: [Loops](control_flow/loops.md)

##### `while`

`while` statements loop for as long as the passed expression returns `true`. For
example, this prints `0`, `1`, `2`, then `Done!`:

```carbon
var x: i32 = 0;
while (x < 3) {
  Console.Print(x);
  ++x;
}
Console.Print("Done!");
```

> References:
>
> -   [`while` loops](control_flow/loops.md#while)
> -   Proposal
>     [#340: Add C++-like `while` loops](https://github.com/carbon-language/carbon-lang/pull/340)

##### `for`

`for` statements support range-based looping, typically over containers. For
example, this prints all names in `names`:

```carbon
for (var name: String in names) {
  Console.Print(name);
}
```

This prints each `String` value in `names`.

> References:
>
> -   [`for` loops](control_flow/loops.md#for)
> -   Proposal
>     [#353: Add C++-like `for` loops](https://github.com/carbon-language/carbon-lang/pull/353)

##### `break`

The `break` statement immediately ends a `while` or `for` loop. Execution will
continue starting from the end of the loop's scope. For example, this processes
steps until a manual step is hit (if no manual step is hit, all steps are
processed):

```carbon
for (var step: Step in steps) {
  if (step.IsManual()) {
    Console.Print("Reached manual step!");
    break;
  }
  step.Process();
}
```

> References: [`break`](control_flow/loops.md#break)

##### `continue`

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
  Console.Print(line);
}
```

> References: [`continue`](control_flow/loops.md#continue)

#### `return`

The `return` statement ends the flow of execution within a function, returning
execution to the caller.

```carbon
// Prints the integers 1 .. `n` and then
// returns to the caller.
fn PrintFirstN(n: i32) {
  var i: i32 = 0;
  while (true) {
    i += 1;
    if (i > n) {
      // None of the rest of the function is
      // executed after a `return`.
      return;
    }
    Console.Print(i);
  }
}
```

If the function returns a value to the caller, that value is provided by an
expression in the return statement. For example:

```carbon
fn Sign(i: i32) -> i32 {
  if (i > 0) {
    return 1;
  }
  if (i < 0) {
    return -1;
  }
  return 0;
}

Assert(Sign(-3) == -1);
```

> References:
>
> -   [`return`](control_flow/return.md)
> -   [`return` statements](functions.md#return-statements)
> -   Proposal
>     [#415: return](https://github.com/carbon-language/carbon-lang/pull/415)
> -   Proposal
>     [#538: return with no argument](https://github.com/carbon-language/carbon-lang/pull/538)

##### `returned var`

To avoid a copy when returning a variable, add a `returned` prefix to the
variable's declaration and use `return var` instead of returning an expression,
as in:

```carbon
fn MakeCircle(radius: i32) -> Circle {
  returned var c: Circle;
  c.radius = radius;
  // `return c` would be invalid because `returned` is in use.
  return var;
}
```

This is instead of
[the "named return value optimization" of C++](https://en.wikipedia.org/wiki/Copy_elision#Return_value_optimization).

> References:
>
> -   [`returned var`](control_flow/return.md#returned-var)
> -   Proposal
>     [#257: Initialization of memory and variables](https://github.com/carbon-language/carbon-lang/pull/257)

#### `match`

`match` is a control flow similar to `switch` of C/C++ and mirrors similar
constructs in other languages, such as Swift. The `match` keyword is followed by
an expression in parentheses, whose value is matched against `case` declarations
in order. The code for the first matching `case` is executed. An optional
`default` code block may be placed after the `case` declaratoins, it will be
executed if none of the `case` declarations match.

An example `match` is:

```carbon
fn Bar() -> (i32, (f32, f32));

fn Foo() -> f32 {
  match (Bar()) {
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

A `case` pattern can contain
[binding and destructuring patterns](#pattern-matching). In addition, it can
contain patterns that may or may not match based on the runtime value of the
`match` expression:

-   A _value pattern_ is an expression, such as `42`, whose value must be equal
    to match.
-   An _if pattern_, consisting of an `if` and a boolean predicate at the end of
    the pattern like `if (p < 13)`, matches if the predicate evaluates to
    `true`.

> References:
>
> -   [Pattern matching](pattern_matching.md)
> -   Question-for-leads issue
>     [#1283: how should pattern matching and implicit conversion interact?](https://github.com/carbon-language/carbon-lang/issues/1283)

## User-defined types

> **TODO:** Maybe rename to "nominal types"?

### Classes

_Nominal classes_, or just
[_classes_](<https://en.wikipedia.org/wiki/Class_(computer_programming)>), are a
way for users to define their own
[data strutures](https://en.wikipedia.org/wiki/Data_structure) or
[record types](<https://en.wikipedia.org/wiki/Record_(computer_science)>).

This is an example of a class
[definition](#declarations-definitions-and-scopes):

```carbon
class Widget {
  var x: i32;
  var y: i32;
  var z: i32;

  var payload: String;
}
```

Breaking this apart:

-   This defines a class named `Widget`. `Widget` is the name added to the
    enclosing [scope](#declarations-definitions-and-scopes).
-   The name `Widget` is followed by curly braces (`{`...`}`), making this a
    [definition](#declarations-definitions-and-scopes). A
    [forward declaration](#declarations-definitions-and-scopes) would instead
    have a semicolon(`;`).
-   Those braces delimit the class'
    [scope](#declarations-definitions-and-scopes).
-   `Widget` has three `i32` fields: `x`, `y`, and `z`.
-   `Widget` has one `String` field: `payload`.
-   Given an instance `dial`, a field can be referenced with `dial.payload`.

The order of the field declarations determines the fields' memory-layout order.

Classes may have other kinds of members beyond fields declared in a class scope:

-   [Class functions](#class-functions-and-factory-functions)
-   [Methods](#methods)
-   [`alias`](#aliases)
-   [`let`](#let) to define constants. **TODO:** Are these constants associated
    with the class or the instance? Do we need to another syntax to distinguish
    constants associated with the class like `class let` or `static let`?
-   `class`, to define a
    [_member class_ or _nested class_](https://en.wikipedia.org/wiki/Inner_class)
-   Every class has a constant member named `Self` equal to the class type
    itself.

Both [structural data classes](#struct-types) and nominal classes are considered
_class types_, but they are commonly referred to as "structs" and "classes"
respectively when that is not confusing. Like structs, classes refer to their
members by name. Unlike structs, classes are
[nominal types](#structural-and-nominal-types).

> References:
>
> -   [Classes](classes.md#nominal-class-types)
> -   Proposal
>     [#722: Nominal classes and methods](https://github.com/carbon-language/carbon-lang/pull/722)
> -   Proposal
>     [#989: Member access expressions](https://github.com/carbon-language/carbon-lang/pull/989)

#### Assignment, copying

There is an [implicit conversions](expressions/implicit_conversions.md) defined
between a [struct literal](#struct-types) and a class type with the same fields,
in any scope that has [access](#access-control) to all of the class' fields.
This may be used to assign or initialize a variable with a class type, as in:

```carbon
var sprocket: Widget = {.x = 3, .y = 4, .z = 5, .payload = "Sproing"};
sprocket = {.x = 2, .y = 1, .z = 0, .payload = "Bounce"};
```

You may also copy a value of a class type into a variable of the same type.

```carbon
var thingy: Widget = sprocket;
sprocket = thingy;
Assert(sprocket.x == thingy.x);
```

> References:
>
> -   [Classes: Construction](classes.md#construction)
> -   Proposal
>     [#981: Implicit conversions for aggregates](https://github.com/carbon-language/carbon-lang/pull/981)

#### Class functions and factory functions

Classes may also contain _class functions_. These are functions that are
accessed as members of the type, like
[static member functions in C++](<https://en.wikipedia.org/wiki/Method_(computer_programming)#Static_methods>),
as opposed to [methods](#methods) that are members of instances. They are
commonly used to define a function that creates instances. Carbon does not have
separate
[constructors](<https://en.wikipedia.org/wiki/Constructor_(object-oriented_programming)>)
like C++ does.

```carbon
class Point {
  // Class function that instantiates `Point`.
  // `Self` in class scope means the class currently being defined.
  fn Origin() -> Self {
    return {.x = 0, .y = 0};
  }
  var x: i32;
  var y: i32;
}
```

Note that if the definition of a function is provided inside the class scope,
the body is treated as if it was defined immediately after the outermost class
definition. This means that members such as the fields will be considered
defined even if their definitions are later in the source than the class
function.

The [`returned var` feature](#returned-var) can be used if the address of the
instance being created is needed in a factory function, as in:

```carbon
class Registered {
  fn Create() -> Self {
    returned var result: Self = {...};
    StoreMyPointerSomewhere(&result);
    return var;
  }
}
```

This approach can also be used for types that can't be copied or moved.

#### Methods

Class type definitions can include methods:

```carbon
class Point {
  // Method defined inline
  fn Distance[me: Self](x2: i32, y2: i32) -> f32 {
    var dx: i32 = x2 - me.x;
    var dy: i32 = y2 - me.y;
    return Math.Sqrt(dx * dx - dy * dy);
  }
  // Mutating method
  fn Offset[addr me: Self*](dx: i32, dy: i32);

  var x: i32;
  var y: i32;
}

// Out-of-line definition of method declared inline.
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

-   Methods are defined as class functions with a `me` parameter inside square
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

> References:
>
> -   [Methods](classes.md#methods)
> -   Proposal
>     [#722: Nominal classes and methods](https://github.com/carbon-language/carbon-lang/pull/722)

#### Inheritance

Classes by default are
[_final_](<https://en.wikipedia.org/wiki/Inheritance_(object-oriented_programming)#Non-subclassable_classes>),
which means they may not be extended. A class may be declared as allowing
extension using either the `base class` or `abstract class` introducer instead
of `class`. An `abstract class` is a base class that may not itself be
instantiated.

```carbon
base class MyBaseClass { ... }
```

Either kind of base class maybe _extended_ to get a _derived class_. Derived
classes are final unless they are themselved declared `base` or `abstract`.
Classes may only extend a single class. Carbon only supports single inheritance,
and will use mixins instead of multiple inheritance.

```carbon
base class MiddleDerived extends MyBaseClass { ... }
class FinalDerived extends MiddleDerived { ... }
// ❌ Forbidden: class Illegal extends FinalDerived { ... }
```

A base class may define
[virtual methods](https://en.wikipedia.org/wiki/Virtual_function). These are
methods whose implementation may be overridden in a derived class. By default
methods are _non-virtual_, the declaration of a virtual methods must be prefixed
by one of these three keywords:

-   A method marked `virtual` has a definition in this class but not in any
    base.
-   A method marked `abstract` does not have have a definition in this class,
    but must have a definition in any non-`abstract` derived class.
-   A method marked `impl` has a definition in this class, overriding any
    definition in a base class.

A pointer to a derived class may be cast to a pointer to one of its base
classes. Calling a virtual method through a pointer to a base class will use the
overridden definition provided in the derived class.

For purposes of construction, a derived class acts like its first field is
called `base` with the type of its immediate base class.

```carbon
class MyDerivedType extends MyBaseType {
  fn Create() -> MyDerivedType {
    return {.base = MyBaseType.Create(), .derived_field = 7};
  }
  var derived_field: i32;
}
```

Abstract classes can't be instantiated, so instead they should define class
functions returning `partial Self`. Those functions should be marked
[`protected`](#access-control) so they may only be used by derived classes.

```carbon
abstract class AbstractClass {
  protected fn Create() -> partial Self {
    return {.field_1 = 3, .field_2 = 9};
  }
  // ...
  var field_1: i32;
  var field_2: i32;
}
// ❌ Error: can't instantiate abstract class
var abc: AbstractClass = ...;

class DerivedFromAbstract extends AbstractClass {
  fn Create() -> Self {
    // AbstractClass.Create() returns a
    // `partial AbstractClass` that can be used as
    // the `.base` member when constructing a value
    // of a derived class.
    return {.base = AbstractClass.Create(),
            .derived_field = 42 };
  }

  var derived_field: i32;
}
```

> References:
>
> -   [Inheritance](classes.md#inheritance)
> -   Proposal
>     [#777: Inheritance](https://github.com/carbon-language/carbon-lang/pull/777)
> -   Proposal
>     [#820: Implicit conversions](https://github.com/carbon-language/carbon-lang/pull/820)

#### Access control

Class members are by default publicly accessible. The `private` keyword prefix
can be added to the member's declaration to restrict it to members of the class
or any friends. A `private virtual` or `private abstract` method may be
implemented in derived classes, even though it may not be called.

Friends may be declared using a `friend` declaration inside the class naming an
existing function or type. Unlike C++, `friend` declarations may only refer to
names resolvable by the compiler, and don't act like forward declarations.

`protected` is like `private`, but also gives access to derived classes.

> References:
>
> -   [Access control for class members](classes.md#access-control)
> -   Question-for-leads issue
>     [#665: `private` vs `public` _syntax_ strategy, as well as other visibility tools like `external`/`api`/etc.](https://github.com/carbon-language/carbon-lang/issues/665)
> -   Question-for-leads issue
>     [#971: Private interfaces in public API files](https://github.com/carbon-language/carbon-lang/issues/971)

#### Destructors

A destructor for a class is custom code executed when the lifetime of a value of
that type ends. They are defined with the `destructor` keyword followed by
either `[me: Self]` or `[addr me: Self*]` (as is done with [methods](#methods))
and the block of code in the class definition, as in:

```carbon
class MyClass {
  destructor [me: Self] { ... }
}
```

or:

```carbon
class MyClass {
  // Can modify `me` in the body.
  destructor [addr me: Self*] { ... }
}
```

The destructor for a class is run before the destructors of its data members.
The data members are destroyed in reverse order of declaration. Derived classes
are destroyed before their base classes.

A destructor in a abstract or base class may be declared `virtual` like with
[methods](#inheritance). Destructors in classes derived from one with a virtual
destructor must be declared with the `impl` keyword prefix. It is illegal to
delete an instance of a derived class through a pointer to a base class unless
the base class is declared `virtual` or `impl`. To delete a pointer to a
non-abstract base class when it is known not to point to a value with a derived
type, use `UnsafeDelete`.

> References:
>
> -   [Destructors](classes.md#destructors)
> -   Proposal
>     [#1154: Destructors](https://github.com/carbon-language/carbon-lang/pull/1154)

### Choice types

A _choice type_ is a [tagged union](https://en.wikipedia.org/wiki/Tagged_union),
that can store different types of data in a storage space that can hold the
largest. A choice type has a name, and a list of cases separated by commas
(`,`). Each case has a name and an optional parameter list.

```carbon
choice IntResult {
  Success(value: i32),
  Failure(error: String),
  Cancelled
}
```

The value of a choice type is one of the cases, plus the values of the
parameters to that case, if any. A value can be constructed by naming the case
and providing values for the parameters, if any:

```carbon
fn ParseAsInt(s: String) -> IntResult {
  var r: i32 = 0;
  for (c: i32 in s) {
    if (not IsDigit(c)) {
      // Equivalent to `IntResult.Failure(...)`
      return .Failure("Invalid character");
    }
    // ...
  }
  return .Success(r);
}
```

Choice type values may be consumed using a [`match` statement](#match):

```carbon
match (ParseAsInt(s)) {
  case .Success(value: i32) => {
    return value;
  }
  case .Failure(error: String) => {
    Display(error);
  }
  case .Cancelled => {
    Terminate();
  }
}
```

They can also represent an
[enumerated type](https://en.wikipedia.org/wiki/Enumerated_type), if no
additional data is associated with the choices, as in:

```carbon
choice LikeABoolean { False, True }
```

> References:
>
> -   Proposal
>     [#157: Design direction for sum types](https://github.com/carbon-language/carbon-lang/pull/157)
> -   Proposal
>     [#162: Basic Syntax](https://github.com/carbon-language/carbon-lang/pull/162)

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

### Legal names

> References: [Lexical conventions](lexical_conventions)
>
> **TODO:** References need to be evolved.

Various constructs introduce a named entity in Carbon. These can be functions,
types, variables, or other kinds of entities. A name in Carbon is formed from a
word, which is a sequence of letters, numbers, and underscores, and which starts
with a letter. We intend to follow Unicode's Annex 31 in selecting valid
identifier characters, but a concrete set of valid characters has not been
selected yet.

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

Unqualified name lookup will always find a file-local result, including aliases,
or names that are defined as part of the prelude.

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

#### Generic choice types

```carbon
choice Result(T:! Type, Error:! Type) {
  Success(value: T),
  Failure(error: Error),
  Cancelled
}
```

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

> **TODO:**

> -   [Implicit conversions](expressions/implicit_conversions.md#extensibility)
> -   [`as` expressions](expressions/as_expressions.md#extensibility)
> -   [Comparison operators](expressions/comparison_operators.md#extensibility)
> -   [Arithmetic expressions](expressions/arithmetic.md#extensibility)
> -   [Bitwise and shift operators](expressions/bitwise.md#extensibility)

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
