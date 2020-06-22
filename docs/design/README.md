# Language design overview

<!--
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

## Table of contents

<!-- toc -->

- [Context and disclaimer](#context-and-disclaimer)
  - [Example code](#example-code)
- [Basic syntax](#basic-syntax)
  - [Code and comments](#code-and-comments)
  - [Files, libraries, and packages](#files-libraries-and-packages)
  - [Names and scopes](#names-and-scopes)
    - [Naming conventions](#naming-conventions)
    - [Aliases](#aliases)
    - [Name lookup](#name-lookup)
      - [Name lookup for common types](#name-lookup-for-common-types)
  - [Expressions](#expressions)
  - [Functions](#functions)
  - [Blocks and statements](#blocks-and-statements)
  - [Variables](#variables)
  - [Lifetime and move semantics](#lifetime-and-move-semantics)
  - [Control flow](#control-flow)
    - [`if`/`else`](#ifelse)
    - [`loop`, `break`, and `continue`](#loop-break-and-continue)
    - [`return`](#return)
- [Types](#types)
  - [Primitive types](#primitive-types)
  - [Composite types](#composite-types)
    - [Tuples](#tuples)
    - [Variants](#variants)
    - [Pointers and references](#pointers-and-references)
    - [Arrays and slices](#arrays-and-slices)
  - [User-defined types](#user-defined-types)
    - [Structs](#structs)
      - [Allocation, construction, and destruction](#allocation-construction-and-destruction)
      - [Assignment, copying, and moving](#assignment-copying-and-moving)
      - [Comparison](#comparison)
      - [Implicit and explicit conversion](#implicit-and-explicit-conversion)
      - [Inline type composition](#inline-type-composition)
    - [Unions](#unions)
- [Pattern matching](#pattern-matching)
  - [`match` control flow](#match-control-flow)
  - [Pattern matching in local variables](#pattern-matching-in-local-variables)
  - [Pattern matching as function overload resolution](#pattern-matching-as-function-overload-resolution)
- [Type abstractions](#type-abstractions)
  - [Interfaces](#interfaces)
  - [Generics](#generics)
  - [Templates](#templates)
    - [Types with template parameters](#types-with-template-parameters)
    - [Functions with template parameters](#functions-with-template-parameters)
    - [Specialization](#specialization)
    - [Constraining templates with interfaces](#constraining-templates-with-interfaces)
- [Metaprogramming](#metaprogramming)
- [Execution abstractions](#execution-abstractions)
  - [Abstract machine and execution model](#abstract-machine-and-execution-model)
  - [Lambdas](#lambdas)
  - [Co-routines](#co-routines)
- [Carbon &lt;-> C/C++ interoperability](#carbon-lt--cc-interoperability)

<!-- tocstop -->

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

> References: [Lexical conventions](lexical_conventions.md)
>
> **TODO:** References need to be evolved.

- All source code is UTF-8 encoded text. For simplicity, no other encoding is
  supported.
- Line comments look like `// ...`. However, they are required to be the only
  non-whitespace on the line for readability.
- Block comments begin with `//\{`, and end with `//\}`. The begin and end
  markers must be the only things on their lines.
  - Nested block comments will be supported.

### Files, libraries, and packages

> References: [Files, libraries and packages](files_libraries_and_packages.md)
>
> **TODO:** References need to be evolved.

Carbon code is organized into files, libraries, and packages:

- A **file** is the unit of compilation.
- A **library** can be made up of multiple files, and is the unit whose public
  interface can be imported.
- A **package** is a collection of one or more libraries, typically ones with a
  single common source and with some close association.

A file belongs to precisely one library, and a library belongs to precisely one
package.

Files have a `.6c` extension. They must start with a declaration of their
package and library. They may import both other libraries from within their
package, as well as libraries from other packages. For example:

```carbon
// This is a file in the "Eucalyptus" library of the "Koala" package.
package Koala library Eucalyptus;

// Import the "Wombat" library from the "Widget" package.
import Widget library Wombat;

// Import the "Container" library from the "Koala" package.
import library Container;
```

### Names and scopes

> References: [Lexical conventions](lexical_conventions.md)
>
> **TODO:** References need to be evolved.

Various constructs introduce a named entity in Carbon. These can be functions,
types, variables, or other kinds of entities that we'll cover. A name in Carbon
is always formed out of an "identifier", or a sequence of letters, numbers, and
underscores which starts with a letter. As a regular expression, this would be
`/[a-zA-Z][a-zA-Z0-9_]*/`. Eventually we may add support for further unicode
characters as well.

#### Naming conventions

> References: [Naming conventions](naming_conventions.md)
>
> **TODO:** References need to be evolved.

Our current proposed naming convention are:

- `UpperCamelCase` for names of compile-time resolved constants, such that they
  can participate in the type system and type checking of the program.
- `lower_snake_case` for keywords and names of run-time resolved values.

As a matter of style and consistency, we will follow these conventions where
possible and encourage convergence.

For example:

- An integer that is a compile-time constant sufficient to use in the
  construction a compile-time array size might be named `N`.
- An integer that is not available as part of the type system would be named
  `n`, even if it happened to be immutable or only take on a single value.
- Functions and most types will be in `UpperCamelCase`.
- A type where only run-time type information queries are available would end up
  as `lower_snake_case`.
- A keyword like `import` uses `lower_snake_case`.

#### Aliases

> References: [Aliases](aliases.md)
>
> **TODO:** References need to be evolved.

Carbon provides a facility to declare a new name as
an alias for a value. This is a fully general
facility because everything is a value in Carbon, including types.

For example:

```carbon
alias MyInt = Int;
```

This creates an alias called `MyInt` for whatever `Int` resolves to. Code
textually after this can refer to `MyInt`, and it will transparently refer to
`Int`.

#### Name lookup

> References: [Name lookup](name_lookup.md)
>
> **TODO:** References need to be evolved.

Names are always introduced into some scope which defines where they can be
referenced. Many of these scopes are themselves named. `namespace` is used to
introduce a dedicated named scope, and we traverse nested names in a uniform way
with `.`-separated names. Unqualified name lookup will always find a file-local
result, including aliases.

For example:

```carbon
package Koala library Eucalyptus;

namespace Leaf {
  namespace Vein {
    fn Count() -> Int;
  }
}
```

`Count` may be referred to as:

- `Count` from within the `Vein` namespace.
- `Vein.Count` from within the `Leaf` namespace.
- `Leaf.Vein.Count` from within this file.
- `Koala.Leaf.Vein.Count` from any arbitrary location.

Note that libraries do **not** introduce a scope; they share the scope of their
package.

##### Name lookup for common types

> References: [Name lookup](name_lookup.md)
>
> **TODO:** References need to be evolved.

Common types that we expect to be used universally will be provided for every
file, including `Int` and `Bool`. These will likely be defined in a `Carbon`
package, and be treated as if always imported and aliased by every file.

### Expressions

> References: [Lexical conventions](lexical_conventions.md),
> [operators](operators.md)
>
> **TODO:** References need to be evolved.

Expressions describe
some computed value. The simplest example would be a literal number like `42`:
an expression that computes the integer value 42.

Some common expressions in Carbon include:

- Literals: `42`, `-13`, `3.1419`, `"Hello World!"`
- Operators:

  - Increment and decrement: `++i`, `--j`
    - These do not return any result.
  - Unary negation: `-x`
  - Arithmetic: `1 + 2`, `3 - 4`, `2 * 5`, `6 / 3`
  - Bitwise: `2 & 3`, `2 | 4`, `3 ^ 1`, `~7`
  - Bit sequence: `1 << 3`, `8 >> 1`
  - Comparison: `2 == 2`, `3 != 4`, `5 < 6`, `7 > 6`, `8 <= 8`, `8 >= 8`
  - Logical: `a && b`, `c || d`

- Parenthesized expressions: `(7 + 8) * (3 - 1)`

### Functions

> References: [Functions](functions.md)
>
> **TODO:** References need to be evolved.

Functions are the core unit of behavior. For example:

```carbon
fn Sum(Int: a, Int: b) -> Int;
```

Breaking this apart:

- `fn` is the keyword used to indicate a function.
- Its name is `Sum`.
- It accepts two `Int` parameters, `a` and `b`.
- It returns an `Int` result.

Calling functions involves a new form of expression, for example, `Sum(1, 2)`.

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
>
> **TODO:** References need to be evolved.

Blocks introduce nested scopes and can contain local variable declarations that
work similarly to function parameters.

For example:

```carbon
fn Foo() {
  var Int: x = 42;
}
```

Breaking this apart:

- `var` is the keyword used to indicate a variable.
- Its name is `x`.
- Its type is `Int`.
- It is initialized with the value `42`.

### Lifetime and move semantics

> References: TODO
>
> **TODO:** References need to be evolved.

### Control flow

> References: [Control flow](control_flow.md)
>
> **TODO:** References need to be evolved.

Blocks of statements are generally executed sequentially. However, statements are
the primary place where this flow of execution can be controlled.

#### `if`/`else`

> References: [Control flow](control_flow.md)
>
> **TODO:** References need to be evolved.

`if` and `else` are common flow control keywords, which can result in
conditional execution of statements.

For example:

```carbon
fn Foo(Int: x) {
  if (x < 42) {
    Bar();
  } else if (x > 77) {
    Baz();
  }
}
```

Breaking the `Foo` function apart:

- `Bar()` is invoked if `x` is less than `42`.
- `Baz()` is invoked if `x` is greater than `77`.
- Nothing happens if `x` is between `42` and `77`.

#### `loop`, `break`, and `continue`

> References: [Control flow](control_flow.md)
>
> **TODO:** References need to be evolved.

Loops will be supported with a low-level primitive `loop` statement which loops
unconditionally. `break` will be a way to exit the `loop` directly, while
`continue` will skip the rest of the current loop iteration.

For example:

```carbon
fn Foo() {
  var Int: x = 0;
  loop (x < 42) {
    if (ShouldStop()) break;
    if (ShouldSkip(x)) {
      ++x;
      continue;
    }
    Bar(x);
    ++x;
  }
}
```

Breaking the `Foo` function apart:

- The loop body is normally executed for all values of `x` in [0, 42).
  - The increment of x at the end causes this.
- If `ShouldStop()` returns true, the `break` causes the `loop` to exit early.
- If `ShouldSkip()` returns true, the `continue` causes the `loop` to restart
  early.
- Otherwise, `Bar(x)` is called for values of `x` in [0, 42).

#### `return`

> References: [Control flow](control_flow.md)
>
> **TODO:** References need to be evolved.

The `return` statement ends the flow of execution within a function, returning
execution to the caller. If the function returns a value to the caller, that
value is provided by an expression in the return statement. This allows us to
complete the definition of our `Sum` function from earlier as:

```carbon
fn Sum(Int: a, Int: b) -> Int {
  return a + b;
}
```

## Types

> References: [Primitive types](primitive_types.md), [tuples](tuples.md), and
> [structs](structs.md)
>
> **TODO:** References need to be evolved.

Carbon's core types are broken down into three categories:

- Primitive types
- Composite types
- User-defined types

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
principles rather than in terms of other operations. Even though these are
special, their names are not keywords or reserved; they are just names in the
global scope.

Primitive types fall into the following categories:

- `Void` - a type with only one possible value: empty.
- `Bool` - a boolean type with two possible values: `True` and `False`.
- `Int` and `UInt` - signed and unsigned 64-bit integer types.
  - Standard sizes are available, both signed and unsigned, including `Int8`,
    `Int16`, `Int32`, `Int128`, and `Int256`.
  - Overflow in either direction is an error.
- `Float64` - a floating point type with semantics based on IEEE-754.
  - Standard sizes are available, including `Float16`, `Float32`, and
    `Float128`.
  - [`BFloat16`](primitive_types.md#bfloat16) is also provided.
- `String` - a byte sequence treated as containing UTF-8 encoded text.
  - `StringView` - a read-only reference to a byte sequence treated as
    containing UTF-8 encoded text.

### Composite types

#### Tuples

> References: [Tuples](tuples.md)
>
> **TODO:** References need to be evolved.

The primary composite type involves simple aggregation of other types as a
tuple. In formal type theory, tuples are product types. A tuple of a single
value is special and collapses to the single value.

An example use of tuples is:

```carbon
fn DoubleBoth(Int: x, Int: y) -> (Int, Int) {
  return (2 * x, 2 * y);
}
```

Breaking this tuple:

- The return type is a tuple of two `Int` types.
- The expression uses tuple syntax to build a tuple of two `Int` values.

Both of these are expressions using the tuple syntax
`(<expression>, <expression>)`. The only difference is the type of the tuple
expression: one is a tuple of types, the other a tuple of values.

Element access uses subscript syntax:

```carbon
fn DoubleTuple((Int, Int): x) -> (Int, Int) {
  return (2 * x[0], 2 * x[1]);
}
```

Tuples also support multiple indices and slicing to restructure tuple elements:

```carbon
// This reverses the tuple using multiple indices.
fn Reverse((Int, Int, Int): x) -> (Int, Int, Int) {
  return x[2, 1, 0];
}

// This slices the tuple by extracting elements [0, 2).
fn RemoveLast((Int, Int, Int): x) -> (Int, Int) {
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

#### Structs

> References: [Structs](structs.md)
>
> **TODO:** References need to be evolved.

`struct`s are a way for users to define their own data strutures or named
product types.

For example:

```carbon
struct Widget {
  var Int: x;
  var Int: y;
  var Int: z;

  var String: payload;
}
```

Breaking apart `Widget`:

- `Widget` has three `Int` members: `x`, `y`, and `z`.
- `Widget` has one `String` member: `payload`.
- Given an instance `dial`, a member can be referenced with `dial.paylod`.

More advanced `struct`s may be created:

```carbon
struct AdvancedWidget {
  // Do a thing!
  fn DoSomething(AdvancedWidget: self, Int: x, Int: y);

  // A nested type.
  struct Subtype {
    // ...
  }

  private var Int: x;
  private var Int: y;
}

fn Foo(AdvancedWidget: thing) {
  thing.DoSomething(1, 2);
}
```

Breaking apart `AdvancedWidget`:

- `AdvancedWidget` has a public object method `DoSomething`.
  - `DoSomething` explicitly indicates how the `AdvancedWidget` is passed to it,
    and there is no automatic scoping - `self` must be specified as an input.
    The `self` name is also a keyword that explains how to invoke this method
    on an object.
  - `DoSomething` accepts `AdvancedWidget` _by value_, which is easily expressed
    here along with other constraints on the object parameter.
- `AdvancedWidget` has two private data members: `x` and `y`.
  - Private methods and data members are restricted to use by `AdvancedWidget`
    only, providing a layer of easy validation of the most basic interface
    constraints.
- `Subtype` is a nested type, and can be accessed as `AdvancedWidget.Subtype`.

##### Allocation, construction, and destruction

> **TODO:** Needs a feature design and a high level summary provided inline.

##### Assignment, copying, and moving

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
  match (Bar()) {
    case (42, (Float: x, Float: y)) => {
      return x - y;
    }
    case (Int: p, (Float: x, Float: _)) if (p < 13) => {
      return p * x;
    }
    case (Int: p, auto: _) if (p > 3) => {
      return p * Pi;
    }
    default => {
      return Pi;
    }
  }
}
```

Breaking apart this `match`:

- It accepts a value that will be inspected; in this case, the result of the
  call to `Bar()`.
  - It then will find the _first_ `case` that matches this value, and execute
    that block.
  - If none match, then it executes the default block.
- Each `case` pattern contains a value pattern, such as `(Int: p, auto: _)`,
  followed by an optional boolean predicate introduced by the `if` keyword.
  - The value pattern must first match, and then the predicate must also
    evaluate to true for the overall `case` pattern to match.
  - Using `auto` for a type will always match.

Value patterns may be composed of the following:

- An expression, such as `42`, whose value must be equal to match.
- An optional type, such as `Int`, followed by a `:` and an identifier to bind
  the value.
  - The special identifier `_` may be used to discard the value once matched.
- A destructuring pattern containing a sequence of value patterns, such as
  `(Float: x, Float: y)`, which match against tuples and tuple-like values by
  recursively matching on their elements.
- An unwrapping pattern containing a nested value pattern which matches against
  a variant or variant-like value by unwrapping it.

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
  var (Int: p, auto: _) = Bar();
  return p;
}
```

To break this apart:

- The `Int` returned by `Bar()` matches and is bound to `p`, then returned.
- The `(Float, Float)` returned by `Bar()` matches and is discarded by
  `auto: _`.

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
instantiated, resulting in late type checking, duck typing, and lazy binding.
Although generics are generally preferred, templates enable translation of code
between C++ and Carbon, and address some cases where the type checking rigor of
generics are problematic.

#### Types with template parameters

> References: [Templates](templates.md)
>
> **TODO:** References need to be evolved.

User-defined types may have _template_ parameters. The resulting type-function
may be used to instantiate the parameterized definition with the provided
arguments in order to produce a complete type. For example:

```carbon
struct Stack(Type:$$ T) {
  var Array(T): storage;

  fn Push(T: value);
  fn Pop() -> T;
}
```

Breaking apart the template use in `Stack`:

- `Stack` is a paremeterized type accepting a type `T`.
- `T` may be used within the definition of `Stack` anywhere a normal type would
  be used, and will only be type checked on instantiation.
- `var Array(T)` instantiates a parameterized type `Array` when `Stack` is
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
fn Convert[Type:$$ T](T: source, Type:$$ U) -> U {
  var U: converted = source;
  return converted;
}

fn Foo(Int: i) -> Float {
  // Instantiates with the `T` implicit argument set to `Int` and the `U`
  // explicit argument set to `Float`, then calls with the runtime value `i`.
  return Convert(i, Float);
}
```

Here we deduce one type parameter and explicitly pass another. It is not
possible to explicitly pass a deduced type parameter; instead the call site
should cast or convert the argument to control the deduction. The explicit type
is passed after a runtime parameter. While this makes that type unavailable to
the declaration of _that_ runtime parameter, it still is a _template_ parameter
and available to use as a type even within the remaining parts of the function
declaration.

#### Specialization

> References: [Templates](templates.md)
>
> **TODO:** References need to be evolved.

An important feature of templates in C++ is the ability to customize how they
end up specialized for specific arguments. Because template parameters (whether as
type parameters or function parameters) are pattern matched, we expect to
leverage pattern matching techniques to provide "better match" definitions that
are selected analogously to specializations in C++ templates. When expressed
through pattern matching, this may enable things beyond just template parameter
specialization, but that is an area that we want to explore cautiously.

> **TODO:** lots more work to flesh this out needs to be done...

#### Constraining templates with interfaces

> References: [Templates](templates.md)
>
> **TODO:** References need to be evolved.

These generic interfaces also provide a mechanism to constrain fully
instantiated templates to operate in terms of a restricted and explicit API
rather than being fully duck typed. This falls out of the template type produced
by the interface declaration. A template can simply accept one of those:

```carbon
template fn TemplateRender[Type: T](Point(T): point) {
  ...
}
```

Here, we accept the specific interface wrapper rather than the underlying `T`.
This forces the interface of `T` to match that of `Point`. It also provides only
this restricted interface to the template function.

This is designed to maximize the programmer's ability to move between different
layers of abstraction, from fully generic to a generically constrained template.

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

## Carbon &lt;-> C/C++ interoperability

> References: [Carbon &lt;-> C/C++ interoperability](interoperability/README.md)
>
> **TODO:** References need to be evolved. Needs a detailed design and a high
> level summary provided inline.
