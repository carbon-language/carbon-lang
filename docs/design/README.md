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
  - [Tuples](#tuples)
  - [Variants](#variants)
  - [Pointers and references](#pointers-and-references)
  - [Arrays and slices](#arrays-and-slices)
  - [User-defined types, both structs and unions](#user-defined-types-both-structs-and-unions)
    - [Allocation, construction, and destruction](#allocation-construction-and-destruction)
    - [Assignment, copying, and moving](#assignment-copying-and-moving)
    - [Comparison](#comparison)
    - [Implicit and explicit conversion](#implicit-and-explicit-conversion)
    - [Inline type composition](#inline-type-composition)
    - [User-defined unions](#user-defined-unions)
- [Pattern matching](#pattern-matching)
  - [Pattern match control flow](#pattern-match-control-flow)
  - [Pattern matching in local variables](#pattern-matching-in-local-variables)
  - [Pattern matching as function overload resolution](#pattern-matching-as-function-overload-resolution)
- [Type abstractions](#type-abstractions)
  - [Interfaces and generics](#interfaces-and-generics)
  - [Templates](#templates)
    - [Types with template parameters](#types-with-template-parameters)
    - [Functions with template parameters](#functions-with-template-parameters)
    - [Specialization](#specialization)
    - [Constraining templates with interfaces](#constraining-templates-with-interfaces)
- [Execution abstractions](#execution-abstractions)
  - [Metaprogramming](#metaprogramming)
  - [Abstract machine and execution model](#abstract-machine-and-execution-model)
  - [Lambdas](#lambdas)
  - [Co-routines](#co-routines)
- [C/C++ interoperability](#cc-interoperability)

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
> **TODO**: References need to be evolved.

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
> **TODO**: References need to be evolved.

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
> **TODO**: References need to be evolved.

Various constructs introduce a named entity in Carbon. These can be functions,
types, variables, or other kinds of entities that we'll cover. A name in Carbon
is always formed out of an "identifier", or a sequence of letters, numbers, and
underscores which starts with a letter. As a regular expression, this would be
`/[a-zA-Z][a-zA-Z0-9_]*/`. Eventually we may add support for further unicode
characters as well.

#### Naming conventions

> References: [Naming conventions](naming_conventions.md)
>
> **TODO**: References need to be evolved.

Our current proposed naming convention are:

- `UpperCamelCase` for names of compile-time resolved constants, such that they
  can participate in the type system and type checking of the program.
- `lower_snake_case` for names of run-time resolved values.

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

#### Aliases

> References: [Aliases](aliases.md)
>
> **TODO**: References need to be evolved.

Carbon provides a fully general name aliasing facility to declare a new name as
an alias for a value; everything is a value in Carbon. This is a fully general
facility because everything is a value in Carbon, including types.

For example:

```
alias ??? MyInt = Int;
```

This creates an alias called `MyInt` for whatever `Int` resolves to. Code
textually after this can refer to `MyInt`, and it will transparently refer to
`Int`.

#### Name lookup

> References: [Name lookup](name_lookup.md)
>
> **TODO**: References need to be evolved.

Names are always introduced into some scope which defines where they can be
referenced. Many of these scopes are themselves named. `namespace` is used to
introduce a dedicated named scope, and we traverse nested names in a uniform way
with `.`-separated names. Unqualified name lookup will always find a file-local
result, including aliases.

For example:

```
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
> **TODO**: References need to be evolved.

Common types that we expect to be used universally will be provided for every
file, including `Int` and `Bool`. These will likely be defined in a `Carbon`
package, and be treated as if always imported and aliased by every file.

### Expressions

> References: [Lexical conventions](lexical_conventions.md),
> [operators](operators.md)
>
> **TODO**: References need to be evolved.

The most pervasive part of the Carbon language are "expressions". These describe
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
> **TODO**: References need to be evolved.

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
> **TODO**: References need to be evolved.

The body or definition of a function is provided by a block of code containing
statements. The body of a function is also a new, nested scope inside the
function's scope, meaning that parameter names are available.

Statements within a block are terminated by a semicolon. Each statement can,
among other things, be an expression.

For example, here is a function definition using a block of statements, one of
which is nested:

```
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
> **TODO**: References need to be evolved.

Blocks introduce nested scopes and can contain local variable declarations that
work similarly to function parameters.

For example:

```
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
> **TODO**: References need to be evolved.

### Control flow

> References: [Control flow](control_flow.md)
>
> **TODO**: References need to be evolved.

Blocks of statements are generally executed linearly. However, statements are
the primary place where this flow of execution can be controlled.

#### `if`/`else`

> References: [Control flow](control_flow.md)
>
> **TODO**: References need to be evolved.

`if` and `else` are common flow control keywords, which can result in
conditional execution of statements.

For example:

```
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
- Nothing happens if x is between `42` and `77`.

#### `loop`, `break`, and `continue`

> References: [Control flow](control_flow.md)
>
> **TODO**: References need to be evolved.

Loops will be supported with a low-level primitive `loop` statement which loops
unconditionally. `break` will be a way to exit the `loop` directly, while
`continue` will skip the rest of the `loop`.

For example:

```
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
> **TODO**: References need to be evolved.

The `return` statement ends the flow of execution within a function, returning
execution to the caller. If the function returns a value to the caller, that
value is provided by an expression in the return statement. This allows us to
complete the definition of our `Sum` function from earlier as:

```
fn Sum(Int: a, Int: b) -> Int {
  return a + b;
}
```

## Types

Carbon's core types are broken down into three categories:

- Primitive types
- Composite types
- User-defined types

The first two are intrinsic and directly built in the language. The last aspect
of types allows for defining new types.

Let's walk through the core types in Carbon. These are broken down into three
categories: primitive types, composite types, and user defined types. The first
two are intrinsic and directly built into the language because they don't have
any reasonable way to be expressed on top of the language. The last aspect of
types allows for defining new types.

Expressions compute values in Carbon, and these values are always strongly typed
much like in C++. However, an important difference from C++ is that types are
themselves modeled as values; specifically, compile-time constant values.
However, in simple cases this doesn't make much difference.

We'll cover more [types] as we go through the language, but the most basic types
are the following:

- `Int` - a signed 64-bit 2’s-complement integer
- `Bool` - a boolean type that is either `True` or `False`.
- `String` - a byte sequence suitable for storing UTF-8 encoded text (and by
  convention assumed to contain such text)

The [primitive types] section outlines other fundamentals such as other sized
integers, floating point numbers, unsigned integers, etc.

### Primitive types

> **TODO:** Need a comprehensive design document to underpin these, and then
> link to it here.

These types are fundamental to the language as they aren't comprised of other
types (or modifying other types) and have semantics that are defined from first
principles rather than in terms of other operations. Even though these are
special, their names are not keywords or reserved in any sense, they are just
names in the global scope.

> **Note:** there are open questions about the extent to which these types
> should be defined in Carbon code rather than special. Clearly they can't be
> directly implemented w/o help, but it might still be useful to force the
> programmer-observed interface to reside in code. However, this can cause
> difficulty with avoiding the need to import things gratuitously.

They in turn can be decomposed into the following categories:

- A monotype `Void` (or possibly `()`, the empty tuple) that has only one
  possible value (empty).
- A boolean type `Bool` that has two possible values: `True` and `False`.
- Integer types
- Floating point types
- A string view type which is a read-only reference to a sequence of bytes
  typically representing (UTF-8 encoded) text.

  > **Note:** The right model of a string view vs. an owning string is still
  > very much unsettled.

Integer types can be either signed or unsigned, much like in C++. Signed
integers are represented using 2's complement and notionally modeled as
unbounded natural numbers. Overflow in either direction is an error. Unsigned
integer types provide modular arithmetic based on the bit width of the integer,
again like C++. The default size for both is 64-bits: `Int` and `UInt`. Specific
sizes are also available, for example: `Int8`, `Int16`, `Int32`, `Int128`,
`UInt256`. Arbitrary powers of two above `8` are supported for both (although
perhaps we'll want to avoid _huge_ values for implementation simplicity).

> **Note:** Open question around allowing special syntax for wrapping operations
> (even on signed types) and/or requiring such syntax for wrapping operations on
> unsigned types.

> **Note:** Supporting non-power-of-two sizes is likely needed to have a clean
> model for bitfields, but requires more details to be worked out around memory
> access.

Floating point types are based on the binary floating point formats provided by
IEEE-754. `Float16`, `Float32`, `Float64` and `Float128` correspond exactly to
those sized IEEE-754 formats, and have the semantics defined by IEEE-754. Carbon
also supports the
`[BFloat16](https://en.wikipedia.org/wiki/Bfloat16_floating-point_format)`
format, a 16-bit truncation of a "binary32" IEEE-754 format floating point
number.

### Tuples

> **TODO(joshl):** Link to tuple and struct design (even in draft) when
> available, and sync any of this section with it.

The primary composite type involves simple aggregation of other types as a tuple
(called a "product type" in formal type theory):

```
fn DoubleBoth(Int: x, Int: y) -> (Int, Int) {
  return (2 * x, 2 * y);
}
```

This function returns a tuple of two integers represented by the type
`(Int, Int)`. The expression to return it uses a special tuple syntax to build a
tuple within an expression: `(<expression>, <expression>)`. This is actually the
same syntax in both cases. The return type is a tuple expression, and the first
and second elements are expressions referring to the `Int` type. The only
difference is the type of these expressions. Both are tuples, but one is a tuple
of types.

Element access uses subscript syntax:

```
fn Bar(Int: x, Int: y) -> Int {
  var (Int, Int): t = (x, y);
  return t[0] + t[1];
}
```

Tuples also support multiple indices and slicing to restructure tuple elements:

```
fn Baz(Int: x, Int: y, Int: z) -> (Int, Int) {
  var (Int, Int, Int): t1 = (x, y, z);
  var (Int, Int, Int): t2 = t1[2, 1, 0];
  return t2[0 .. 2];
}
```

This code first reverses the tuple, and then extracts a slice using a half-open
range of indices.

> **Note:** we will likely want to restrict these indices to compile-time
> constants. Without that, run-time indexing would need to suddenly switch to a
> variant-style return type to handle heterogeneous tuples. This would both be
> surprising and complex for little or no value.

> **Note:** using multiple indices in this way is a bit questionable. If we end
> up wanting to support multidimensional arrays / slices (a likely selling point
> for the scientific world), a sequence of indices seems a likely desired
> facility there. We'd either need to find a different syntax there, change this
> syntax, or cope with tuples and arrays having different semantics for multiple
> indices (which seems really bad).

> **Note:** the intent of `0 .. 2` is to be syntax for forming a sequence of
> indices based on the half-open range. There are a bunch of questions we'll
> need to answer here. Is this valid anywhere? Only some places? What _is_ the
> sequence? If it is a tuple of indices, maybe that solves the above issue, and
> unlike function call indexing with multiple indices is different from indexing
> with a tuple of indexes. Also, do we need syntax for a closed range (`...`
> perhaps, unclear if that ends up _aligned_ or in _conflict_ with other likely
> uses of `...` in pattern matching)? All of these syntaxes are also very close
> to `0.2`, is that similarity of syntax OK? Do we want to require the `..` to
> be surrounded by whitespace to minimize that collision?

A tuple of a single value is special and simply collapses to the single value.

> **Note:** this remains an area of active investigation. There are serious
> problems with all approaches here. Without the collapse of one-tuples to
> scalars we need to distinguish between a parenthesized expression (`(42)`) and
> a one tuple (in Python or Rust, `(42,)`), and if we distinguish them then we
> cannot model a function call as simply a function name followed by a tuple of
> arguments; one of `f(0)` and `f(0,)` becomes a special case. With the
> collapse, we either break genericity by forbidding `(42)[0]` from working, or
> it isn't clear what it means to access a nested tuple's first element from a
> parenthesized expression: `((1, 2))[0]`.

Generally, functions pattern match a single tuple value of the arguments (with
some important questions above around single-value tuples) in order to bind
their parameters. However, when _calling_ a function, we insist on using
explicit parentheses to have clear and distinct syntax that matches common
conventions in C++ as well as other programming languages around function
notation.

> **Note:** there are some interesting corner cases we need to expand on to
> fully and more precisely talk about the exact semantic model of function calls
> and their pattern match here, especially to handle variadic patterns and
> forwarding of tuples as arguments. We are hoping for a purely type system
> answer here without needing templates to be directly involved outside the type
> system as happens in C++ variadics.

### Variants

> **TODO:** Needs a detailed design and a high level summary provided inline.

### Pointers and references

> **TODO:** Needs a detailed design and a high level summary provided inline.

### Arrays and slices

> **TODO:** Needs a detailed design and a high level summary provided inline.

### User-defined types, both structs and unions

> **TODO(joshl):** Link to tuple and struct design (even in draft) when
> available, and sync any of this section with it.

Beyond simple tuples, Carbon of course allows defining named product types. This
is the primary mechanism for users to extend the Carbon type system and
fundamentally is deeply rooted in C++ and its history (C and Simula). We simply
call them `struct`s rather than other terms as it is both familiar to existing
programmers and accurately captures their essence: they are a mechanism for
structuring data:

```
struct Widget {
  var Int: x;
  var Int: y;
  var Int: z;

  var String: payload;
}
```

Most of the core features of structures from C++ remain present in Carbon, but
often using different syntax:

```
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

Here we provide a public object method and two private data members. The method
explicitly indicates how the object parameter is passed to it, and there is no
automatic scoping - you have to use `self` here. The `self` name is also a
keyword, though, that explains how to invoke this method on an object. This
member function accepts the object _by value_, which is easily expressed here
along with other constraints on the object parameter. Private members work the
same as in C++, providing a layer of easy validation of the most basic interface
constraints.

> **Note:** requiring the type of `self` makes method declarations quite
> verbose. Unclear what is the best way to mitigate this, there are many
> options. One is to have a special `Self` type.

> **Note:** it may be interesting to consider separating the `self` syntax from
> the rest of the parameter pattern as it doesn't seem necessary to inject all
> of the special rules (covariance vs. contravariance, special pointer handling)
> for `self` into the general pattern matching system.

> **Note:** the default access control level (and the options for access
> control) are a pretty large open question. Swift and C++ (especially w/
> modules) provide a lot of options and a pretty wide space to explore here. if
> the default isn't right most of the time, access control runs the risk of
> becoming a significant ceremony burden that we may want to alleviate with
> grouped access regions instead of per-entity specifiers. Grouped access
> regions have some other advantages in terms of pulling the public interface
> into a specific area of the type.

The type itself is a compile-time constant value. All name access is done with
the `.` notation. Constant members (including member types and member functions
which do not need an implicit object parameter) can be accessed via that
constant: `AdvancedWidget.Subtype`. Other members and member functions needing
an implicit object parameter (or "methods") must be accessed from an object of
the type.

Some things in C++ are notably absent or orthogonally handled:

- No need for `static` functions, they simply don't accept an implicit object
  parameter.
- No `static` variables because there are no global variables. Instead, can have
  scoped constants.

#### Allocation, construction, and destruction

#### Assignment, copying, and moving

#### Comparison

#### Implicit and explicit conversion

#### Inline type composition

#### User-defined unions

> **TODO:** Needs a detailed design and a high level summary provided inline.

## Pattern matching

> **TODO:** Publish draft design and link to it here. Update summary to match.

The most prominent mechanism to manipulate and work with types in Carbon is
pattern matching. This may seem like a deviation from C++, but in fact this is
largely about building a clear, coherent model for a fundamental part of C++:
overload resolution.

### Pattern match control flow

The most powerful form and easiest to explain form of pattern matching is a
dedicated control flow construct that subsumes the `switch` of C and C++ into
something much more powerful, `match`. This is not a novel construct, and is
widely used in existing languages (Swift and Rust among others) and is currently
under active investigation for C++. Carbon's `match` can be used as follows:

```
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

There is a lot going on here. First, let's break down the core structure of a
`match` statement. It accepts a value that will be inspected, here the result of
the call to `Bar()`. It then will find the _first_ `case` that matches this
value, and execute that block. If none match, then it executes the default
block.

Each `case` contains a pattern. The first part is a value pattern
(`(Int: p, auto: _)` for example) followed by an optional boolean predicate
introduced by the `if` keyword. The value pattern has to match, and then the
predicate has to evaluate to true for the overall pattern to match. Value
patterns can be composed of the following:

- An expression (`42` for example), whose value must be equal to match.
- An optional type (`Int` for example), followed by a `:` and either an
  identifier to bind to the value or the special identifier `_` to discard the
  value once matched.
- A destructuring pattern containing a sequence of value patterns
  (`(Float: x, Float: y)`) which match against tuples and tuple like values by
  recursively matching on their elements.
- An unwrapping pattern containing a nested value pattern which matches against
  a variant or variant-like value by unwrapping it.

  > **Note:** an open question is how to effectively fit a "slice" or "array"
  > pattern into this (or whether we shouldn't do so).

> **Note:** an open question is going beyond a simple "type" to things that
> support generics and/or templates.

In order to match a value, whatever is specified in the pattern must match.
Using `auto` for a type will always match, making `auto: _` the wildcard
pattern.

### Pattern matching in local variables

Value patterns may be used when declaring local variables to conveniently
destructure them and do other type manipulations. However, the patterns must
match at compile time which is why the boolean predicate cannot be used
directly.

```
fn Bar() -> (Int, (Float, Float));
fn Foo() -> Int {
  var (Int: p, auto: _) = Bar();
  return p;
}
```

This extracts the first value from the result of calling `Bar()` and binds it to
a local variable named `p` which is then returned.

### Pattern matching as function overload resolution

> **TODO:** Need to flesh out specific details of how overload selection
> leverages the pattern matching machinery, what (if any) restrictions are
> imposed, etc.

## Type abstractions

Carbon's type abstraction systems are centered around a core set of concepts:
generics, interfaces, and facet types. We extend these with pure templates
(similar to C++ templates) and inheritance to build a cohesive and powerful set
of abstractions that both covers existing paradigms in C++ code as well as
providing a cleaner and stronger model going forward. In fact, we use generics
and facet types widely in Carbon to provide a unified framework of type
abstraction.

> **TODO:** Update all of this to match the generics proposal when we have a
> draft published, and link to it.

> **TODO:** Add a minimal introduction to the ideas of parameterized types and
> implicit/deduced function parameters and how it is these parameters that are
> potentially generic or templated.

### Interfaces and generics

> **TODO:** Add a (very) high level summary of interfaces and generics.

### Templates

Carbon templates follow the same fundamental paradigm as C++ templates: they are
instantiated, resulting in late type checking, duck typing, and lazy binding.
They both enable interoperability between Carbon and C++ and address some
(hopefully limited) use cases where the type checking rigor imposed by generics
isn't helpful.

> **TODO:** Link these terms into the terminology document when available.

#### Types with template parameters

When parameterizing a user-defined type, the parameters can be marked as
_template_ parameters. The resulting type-function will instantiate the
parameterized definition with the provided arguments to produce a complete type
when used. Note that only the parameters marked as having this _template_
behavior are subject to full instantiation -- other parameters will be type
checked and bound early to the extent possible. For example:

```
struct Stack(Type:$$ T) {
  var Array(T): storage;

  fn Push(T: value);
  fn Pop() -> T;
}
```

This both defines a parameterized type (`Stack`) and uses one (`Array`). Within
the definition of the type, the _template_ type parameter `T` can be used in all
of the places a normal type would be used, and it will only by type checked on
instantiation.

#### Functions with template parameters

Both implicit and explicit function parameters in Carbon can be marked as
_template_ parameters. When called, the arguments to these parameters trigger
instantiation of the function definition, fully type checking and resolving that
definition after substituting in the provided (or computed if implicit)
arguments. The runtime call then passes the remaining arguments to the resulting
complete definition.

```
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
possible to explicitly pass a deduced type parameter, instead the call site
should cast or convert the argument to control the deduction. The explicit type
is passed after a runtime parameter. While this makes that type unavailable to
the declaration of _that_ runtime parameter, it still is a _template_ parameter
and available to use as a type even within the remaining parts of the function
declaration.

#### Specialization

An important feature of templates in C++ is the ability to customize how they
end up specialized for specific types. Because template parameters (whether as
type parameters or function parameters) are pattern matched, we expect to
leverage pattern matching techniques to provide "better match" definitions that
are selected analogously to specializations in C++ templates. When expressed
through pattern matching, this may enable things beyond just template parameter
specialization, but that is an area that we want to explore cautiously.

> **TODO:** lots more work to flesh this out needs to be done...

#### Constraining templates with interfaces

These generic interfaces also provide a mechanism to constrain fully
instantiated templates to operate in terms of a restricted and explicit API
rather than being fully duck typed. This falls out of the template type produced
by the interface declaration. A template can simply accept one of those:

```
template fn TemplateRender[Type: T](Point(T): point) {
  ...
}
```

Here, we accept the specific interface wrapper rather than the underlying `T`.
This forces the interface of `T` to match that of `Point`. It also provides only
this restricted interface to the template function.

This is designed to maximize the programmer's ability to move between different
layers of abstraction, from fully generic to a generically constrained template.

## Execution abstractions

Carbon also provides some higher-order abstractions of program execution, as
well as the critical underpinnings of such abstractions: an execution model,
abstract machine, etc.

### Metaprogramming

There is no support for textual inclusion or preprocessing of the source text in
any form. Instead, metaprogramming facilities should be provided to more
clearly, cleanly, and directly express the real and important use cases
typically covered by such facilities in C++.

### Abstract machine and execution model

> **TODO:** Needs a detailed design and a high level summary provided inline.

### Lambdas

> **TODO:** Needs a detailed design and a high level summary provided inline.

### Co-routines

> **TODO:** Needs a detailed design and a high level summary provided inline.

## C/C++ interoperability

> **TODO:** Publish draft design and link to it here. Add relevant summary and
> examples.
