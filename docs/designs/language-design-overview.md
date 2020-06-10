# Language Design Overview

<!--
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

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

## Basics

The goal of this section is not to be comprehensive but to quickly cover the
most basic concepts as we walk through the most basic examples of Carbon code.

### Source code

All source code is UTF-8 encoded text. For simplicity, no other encoding is
supported. The language only uses widely available characters and symbols with
low risk of conflict. For example, `$` is a poor symbol due to international
keyboard layouts making it less available and its widespread use in text
processing and templating systems that may be combined with source code.

When represented as a file, source code should use the extension `.6c`. There is
no support for textual inclusion or preprocessing of the source text in any
form. Instead, metaprogramming facilities should be provided to more clearly,
cleanly, and directly express the real and important use cases typically covered
by such facilities in C++.

> **Note**: The use of `6c` as a short file extension or top-level CLI (with
> subcommands below it similar to `git` or `go`) has some drawbacks. There are
> several other possible extensions / commands:
>
> - `cb`: This collides with several acronyms and may not be especially
>   memorable as referring to Carbon.
> - `c6`: This seems a weird incorrect ordering of the atomic number and has
>   unfortunate (if very bizarre) slang associations (NSFW, use caution if
>   searching, as with too much Internet slang).
> - `carbon`: This is an obvious and unsurprising choice, but also quite long.
>
> This seems fairly easy for us to change as we go along, but we should at some
> point do a formal proposal to gather other options and let the core team try
> to find the set that they feel is close enough to be a bikeshed.

Carbon supports C++-style line comments with `// ...`, but they are required to
be the only non-whitespace on the line. It will also have some block-comment
syntax. The current direction is attempting to support nesting and have clean
and unsurprising interactions with other lexical constructs, but is still under
active discussions. All comments are treated as whitespace.

> **Note**: The details here are being actively developed and still very much
> subject to change.
>
> **TODO(zygoloid)**: Sync this with
> [p0016](https://github.com/carbon-language/carbon-lang/pull/17) and link to
> its docs when they land.

### Basic files, libraries, and packages

> **TODO(chandlerc)**: Upstream the proposal covering the details of files,
> libraries, and package oraganization and link to it here.

Carbon code is organized into files, libraries, and packages. A Carbon file is
the unit of compilation. A library can be made up of multiple files, and is the
unit whose public interface can be imported. A Carbon package is a collection of
one or more libraries, typically ones with a single common source and with some
close association. A file belongs to precisely one library, and a library
belongs to precisely one package.

Files start with a declaration of their package and library:

```
package Abseil library Container;
```

They can import both other libraries from within their package, as well as
libraries from other packages:

```
package Abseil library Time;

// Importing a library from another package.
import Widget library Wombat;

// Importing one of my own package's libraries.
import library Container;
```

We'll get into more detail about what these mean and how they all work as we go
along.

### Basic names and scopes

Various constructs introduce a named entity in Carbon. These can be functions,
types, variables, or other kinds of entities that we'll cover. A name in Carbon
is always formed out of an "identifier", or a sequence of letters, numbers, and
underscores which starts with a letter. As a regular expression, this would be
`/[a-zA-Z][a-zA-Z0-9_]*/`. Eventually we may add support for further unicode
characters as well.

> **TODO(zygoloid)**: Sync this with
> [p0016](https://github.com/carbon-language/carbon-lang/pull/17) and link to
> its docs when they land.

#### Naming conventions

We would like to have widespread and consistent naming conventions across Carbon
code to the extent possible. This is for the same core reason as naming
conventions are provided in most major style guides. Even migrating existing C++
code at-scale presents a significant opportunity to converge even more broadly
and we're interested in pursuing this if viable.

Our current proposed naming convention, which we at least are attempting to
follow within Carbon documentation in order to keep code samples as consistent
as possible:

- `UpperCamelCase` for names of compile-time resolved constants, such that they
  can participate in the type system and type checking of the program.
- `lower_snake_case` for names of run-time resolved values.

As an example, an integer that is a compile-time constant sufficient to use in
the construction a compile-time array size might be named `N`, where an integer
that is not available as part of the type system would be named `n`, even if it
happened to be immutable or only take on a single value. Functions and most
types will be in `UpperCamelCase`, but a type where only run-time type
information queries are available would end up as `lower_snake_case`.

We only use `UpperCamelCase` and `lower_snake_case` (skipping other variations
on both snake-case and camel-case naming conventions) because these two have the
most significant visual separation. For example, the value of adding
`lowerCamelCase` for another set seems low given the small visual difference
provided.

The rationale for the specific division between the two isn't a huge or
fundamental concept, but it stems from a convention in Ruby where constants are
named with a leading capital letter. The idea is that it mirrors the English
language capitalization of proper nouns: the name of a constant refers to a
_specific_ value that is precisely resolved at compile time, not just to _some_
value. For example, there are many different _shires_ in Britain, but Frodo
comes from the _Shire_ -- a specific fictional region.

> **Note**: We need some consist pattern while writing documentation, but the
> specific one proposed here still needs to be fully considered.

#### Aliasing of names

Naming is one of the things that most often requires careful management over
time -- things tend to get renamed and moved around. Carbon provides a fully
general name aliasing facility to declare a new name as an alias for a value.
This is a fully general facility because everything is a value in Carbon,
including types. For example:

```
alias ??? MyInt = Int;
```

This creates an alias for whatever `Int` resolves to called `MyInt`. Code
textually after this can refer to `MyInt` and it will transparently refer to
`Int`.

> **Note**: the syntax here is not at all in a good state yet. We've considered
> a few alternatives, but they all end up being confusing in some way. We need
> to figure out a good and clean syntax that can be used here.

#### Scopes and name lookup

Names are always introduced into some scope which defines where they can be
referenced. Many of these scopes are themselves named. Carbon has a special
facility for introducing a dedicated named scope just like C++, but we traverse
nested names in a uniform way with `.`-separated names:

```
namespace Foo {
  namespace Bar {
    alias ??? MyInt = Int;
  }
}

fn F(Foo.Bar.MyInt: x);
```

Carbon packages are also namespaces so to get to an imported name from the
`Abseil` package you would write `Abseil.Foo`. The "top-level" file scope is
that of the Carbon package containing the file, meaning that there is no
"global" scope. Dedicated namespaces can be reopened within a package, but there
is no way to reopen a package without being a library and file _within_ that
package.

Note that libraries (unlike packages) do **not** introduce a scope, they share
the scope of their package. This is based on the observation that in practice, a
fairly coarse scoping tends to work best, with some degree of global registry to
establish a unique package name.

The Carbon standard library is in the `Carbon` package. A very small subset of
this standard library is provided implicitly in every file's scope as-if it were
first imported and then every name in it aliased into that file's package scope.
This makes the names from this part of the standard library nearly the same as
keywords, and so it is expected to be extremely small but contain the very
fundamentals that essentially every file of Carbon code will need (`Int`,
`Bool`, etc.).

Unqualified name lookup in Carbon will always find a file-local result, and
other than the implicit "prelude" of importing and aliasing the fundamentals of
the standard library, there will be an explicit mention of the name in the file
that declares it in that scope.

> **Note**: this implies that other names within your own package but not
> declared within the file must be found via the package name. It isn't clear if
> this is the desirable end state. We need to consider alternatives where names
> from the same library or any library in the same package are made immediately
> visible within the package scope for unqualified name lookup.

Carbon also disallows the use of shadowed unqualified names, but not the
_declaration_ of shadowing names in different named scopes:

Because all unqualified name lookup is locally controlled, shadowing isn't
needed for robustness and is a long and painful source of bugs over time.
Disallowing it provides simple, predictable rules for name lookup. However, it
is important that adding names to the standard library or importing a new
package (both of which bring new names into the current package's scope) doesn't
force renaming interfaces that may have many users. To accomplish this, we allow
code to declare shadowing names, but references to that name must be qualified.
For package-scope names, this can be done with an explicit use of the current
package name: `PackageName.ShadowingName`.

```
package Foo library MyLib;

// Consider an exported function named `Shadow`.
fn Shadow();

// The package might want to import some other package named `Shadow`
// as part of its implementation, but cannot rename its exported
// `Shadow` function:
import Shadow library OtherLib;

// We can reference the imported library:
alias ??? OtherLibType = Shadow.SomeType;

// We can also reference the exported function and provide a new alias by
// using our current package name as an explicitly qualified name.
alias ??? NewShadowFunction = Foo.Shadow;
```

> **Note:** it may make sense to restrict this further to only allowing
> shadowing for exported names as internal names should be trivially renamable,
> and it is only needed when the source is already changing to add a new import.
> Or we may want to completely revisit the rules around shadowing.

For more details on all of this, see the later
[section on name organization](#code-and-name-organization).

### Basic expressions

The most pervasive part of the Carbon language are "expressions". These describe
some computed value. The simplest example would be a literal number like `42`.
This is an expression that computes the integer value 42. Some common
expressions in Carbon include the following constructs:

- Literals: `42`, `-13`, `3.1419`, `"Hello World!"`

  > **TODO(zygoloid)**: Sync this with
  > [p0016](https://github.com/carbon-language/carbon-lang/pull/17) and link to
  > its docs when they land.

- Most operators from C++, but some differences:

  - Only one form of increment and decrement, without returning the result:
    `++i`, `--j`
  - Unary negation: `-x`
  - Arithmetic binary: `1 + 2`, `3 - 4`, `2 * 5`, `6 / 3`
  - Bitwise: `2 & 3`, `2 | 4`, `3 ^ 1`, `1 <&lt; 3`, `8 >> 1`, `~7`
    - Note that these are candidates for keywords instead of high-value
      punctuation.
  - Relational: `2 == 2`, `3 != 4`, `5 &lt; 6`, `7 > 6`, `8 &lt;= 8`, `8 >= 8`
  - No short-circuiting operators from C/C++ (`||`, `&&`, `?:`) -- they have
    custom semantics and are not just operators. We'll get to these later.
  - See the draft
    [operator design](https://github.com/carbon-language/carbon-proposals/pull/5)
    for more details including precedence.

    > **TODO(zygoloid)**: Update this summary to reflect the in-progress
    > operator design.

- Parenthesized expressions: `(7 + 8) * (3 - 1)`

### Basic types and values

> **TODO:** Need a comprehensive design document to underpin these, and then
> link to it here.

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

### Basic functions

> **TODO:** Need a comprehensive design document to underpin these, and then
> link to it here.

Programs written in Carbon, much like those written in other languages, are
primarily divided up into "functions" (or "procedures", "subroutines", or
"subprograms"). These are the core unit of behavior for the programming
language. Let's look at a simple example to understand how these work:

```
fn Sum(Int: a, Int: b) -> Int;
```

This declares a function called `Sum` which accepts two `Int` parameters, the
first called `a` and the second called `b`, and returns an `Int` result. C++
might declare the same thing:

```
std::int64_t Sum(std::int64_t a, std::int64_t b);

// Or with the new trailing return type syntax:
auto Sum(std::int64_t a, std::int64_t b) -> std::int64_t;
```

> **Note**: While we are currently keeping types first matching C++, there is
> significant uncertainty around the right approach here. While adding the colon
> improves the grammar by unambiguously marking the transition from type to a
> declared identifier, in essentially every other language with a colon in a
> similar position, the identifier is first and the type follows. However, that
> ordering would be very _inconsistent_ with C++.
>
> One very important consideration here is the fundamental approach to type
> inference. Languages which use the syntax `<identifier>: <type>` typically
> allow completely omitting the colon and the type to signify inference. With
> C++, inference is achieved with a placeholder keyword `auto`, and Carbon is
> currently being consistent there as well with `auto: <identifier>`. For
> languages which simply allow omission, this seems an intentional incentive to
> encourage inference. On the other hand, there has been strong advocacy in the
> C++ community to not overly rely on inference and to write the explicit type
> whenever convenient. Being consistent with the _ordering_ of identifier and
> type may ultimately be less important than being consistent with the
> incentives and approach to type inference. What should be the default that we
> teach? Teaching to avoid inference unless it specifically helps readability by
> avoiding a confusing or unhelpfully complex type name, and incentivizing that
> by requiring `auto` or another placeholder, may cause as much or more
> inconsistency with languages that use `<identifier: <type>` as retaining the
> C++ ordering.
>
> That said, all of this is largely unknown. It will require a significant
> exploration of the trade-offs and consistency differences. It should also
> factor in further development of pattern matching generally and whether that
> has an influence on one or another approach. Last but not least, while this
> may seem like something that people will get used to with time, it may be
> worthwhile to do some user research to understand the likely reaction
> distribution, strength of reaction, and any quantifiable impact these options
> have on measured readability. We have only found one _very_ weak source of
> research that focused on the _order_ question (rather than type inference vs.
> explicit types or other questions in this space). That was a very limited PhD
> student's study of Java programmers that seemed to indicate improved latency
> for recalling the type of a given variable name with types on the left (as in
> C++). However, those results are _far_ from conclusive.
>
> **TODO**: Get a useful link to this PhD research (a few of us got a copy from
> the professor directly).

Let's look at how some specific parts of this work. The function declaration is
introduced with a keyword `fn` followed by the name of the function `Sum`. This
declares that name in the surrounding scope and opens up a new scope for this
function. We declare the first parameter as `Int: a`. The `Int` part is an expression
(here referring to a constant) that computes the type of the parameter. The `:`
marks the end of the type expression and introduces the identifier for the
parameter, `a`. The parameter names are introduced into the function's scope and
can be referenced immediately after they are introduced. The return type is
indicated with `-> Int`, where again `Int` is just an expression computing the
desired type. The return type can be completely omitted in the case of functions
which do not return a value.

Calling functions involves a new form of expression: `Sum(1, 2)` for example.
The first part, `Sum`, is an expression referring to the name of the function.
The second part, `(1, 2)` is a parenthesized list of arguments to the function.
The juxtaposition of one expression with parentheses forms the core of call
expression, similar to a postfix operator.

### Blocks and statements

> **TODO:** Need a comprehensive design document to underpin these, and then
> link to it here.

The body or definition of a function is provided by a block of code containing
statements, much like in C or C++. The body of a function is also a new, nested
scope inside the function's scope (meaning that parameter names are available).
Statements within a block are terminated by a semicolon. Each statement can,
among other things, be an expression. Here is a trivial example of a function
definition using a block of statements:

```
fn Foo() {
  Bar();
  Baz();
}
```

Statements can also themselves be a block of statements, which provide scopes
and nesting:

```
fn Foo() {
  Bar();
  {
    Baz();
  }
}
```

### Variables

> **TODO:** Need a comprehensive design document to underpin these, and then
> link to it here.

Blocks introduce nested scopes and can contain local variable declarations that
work similarly to function parameters:

```
fn Foo() {
  var Int: x = 42;
}
```

This introduces a local variable named `x` into the block's scope. It has the
type `Int` and is initialized with the value `42`. These variable declarations
(and function declarations) have a lot more power than what we're covering just
yet, but this gives you the basic idea.

> **Note:** an open question is what syntax (and mechanism) to use for declaring
> constants. There are serious problems with the use of `const` in C++ as part
> of the type system, so alternatives to pseudo-types (type qualifiers) are
> being explored. The obvious syntax is `let` from Swift, although there are
> some questions around how intuitive it is for this to introduce a constant.
> Another candidate is `val` from Kotlin. Another thing we need to contend with
> is the surprise of const and reference (semantic) types.

While there can be global constants, there are no global variables.

> **Note:** we are exploring several different ideas for how to design less
> bug-prone patterns to replace the important use cases programmers still have
> for global variables. We may be unable to fully address them, at least for
> migrated code, and be forced to add some limited form of global variables
> back. We may also discover that their convenience outweighs any improvements
> afforded.

### Lifetime and move semantics

> **TODO:** Need a document to explore this fully.

### Basic control flow

> **TODO:** Need a comprehensive design document to underpin these, and then
> link to it here.

Blocks of statements are generally executed linearly. However, statements are
the primary place where this flow of execution can be controlled. Carbon's
control flow constructs are mostly similar to those in C, C++, and other
languages.

```
fn Foo(Int: x) {
  if (x < 42) {
    Bar();
  } else if (x > 77) {
    Baz();
  }
}
```

> **Note:** It is an open question whether a block is required or a single
> statement may be nested in an `if` statement. Similarly, it is an open
> question whether `else if` is a single keyword versus a nested `if` statement,
> and if it is a single construct whether it should be spelled `elif` or
> something else.

Loops will at least be supported with a low-level primitive `loop` statement
which loops unconditionally, with `break` and `continue` statements which work
the same as in C++.

> **Note:** if and how to support a "labeled break" or "labeled continue" is
> still a point of open discussion.

Last but not least, for the basics we need to include the `return` statement.
This statement ends the flow of execution within a function, returning it to the
caller. If the function returns a value to the caller, that value is provided by
an expression in the return statement. This allows us to complete the definition
of our `Sum` function from earlier as:

```
fn Sum(Int: a, Int: b) -> Int {
  return a + b;
}
```

### Programs and "Hello World!"

This is enough for us to describe complete programs in Carbon:

```
package MyProgram library Entry;

fn Run() -> Bool {
  return True;
}
```

The entry point is the function called `Run` because running the program is
modeled as calling that function. This function is in the normal package scope,
and the build system is responsible for selecting the package which will provide
the entry point. It returns `Bool` to hide the platform specific management of
error codes as results of programs. The often used "Hello World!" example
becomes:

```
package MyProgram library Entry;

fn Run() -> Bool {
  Print("Hello World!");
  return True;
}
```

The `Print` function here is a hypothetical part of the standard library and
simply writes to whatever is configured as output for the program when it is
run. This isn't a concrete suggestion of what the I/O interfaces should look
like, or which (if any) part of them should be automatically made available
similar to the `Int` type. That needs to be fleshed out as part of the design of
the standard I/O library for Carbon.

## Syntax and source code

> **TODO:** Needs a detailed design and a high level summary provided inline.
> The below organization may not precisely make sense.

### Lexical conventions

### Declarations

### Blocks, statements, expressions, and operators

### Names

> **TODO:** Publish draft design and link to it here. Add relevant summary and
> examples.

### Code organization

> **TODO:** Publish draft design and link to it here. Add relevant summary and
> examples.

## Control structures

> **TODO:** Needs a detailed design and a high level summary provided inline.
> The below organization may not precisely make sense.

### Conditionals

> **TODO:** At least summarize `if` and `else` to cover basics. Especially
> important to surface the idea of using basic conditionals as both expressions
> and statements to avoid needing conditional operators.

### Looping

> **TODO:** Looping is an especially interesting topic to explore as there are
> lots of challenges posed by the C++ loop structure. Even C++ itself has been
> seeing significant interest and pressure to improve its looping facilities.

## Types

Let's walk through the core types in Carbon. These are broken down into three
categories: primitive types, composite types, and user defined types. The first
two are intrinsic and directly built into the language because they don't have
any reasonable way to be expressed on top of the language. The last aspect of
types allows for defining new types.

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
tuple within an expression: `(<expression>, <expression>)`. This is
actually the same syntax in both cases. The return type is a tuple expression,
and the first and second elements are expressions referring to the `Int` type.
The only difference is the type of these expressions. Both are tuples, but one
is a tuple of types.

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

### Abstract machine and execution model

> **TODO:** Needs a detailed design and a high level summary provided inline.

### Lambdas

> **TODO:** Needs a detailed design and a high level summary provided inline.

### Co-routines

> **TODO:** Needs a detailed design and a high level summary provided inline.

## C/C++ interoperability

> **TODO:** Publish draft design and link to it here. Add relevant summary and
> examples.
