# Carbon metaprogramming

<!--
Part of the Carbon Language, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

## Table of contents

<!-- toc -->

- [Goals](#goals)
- [Example use cases](#example-use-cases)
- [Background](#background)
  - [Macros](#macros)
  - [Structured metaprogramming](#structured-metaprogramming)
  - [Compile-time partial evaluation](#compile-time-partial-evaluation)
- [Proposal](#proposal)
  - [Case study: tuple struct](#case-study-tuple-struct)

<!-- tocstop -->

Result of a discussion with: [josh11b](https://github.com/josh11b),
[zygoloid](https://github.com/zygoloid),
[chandlerc](https://github.com/chandlerc),
[noncombatant](https://github.com/noncombatant),
[mconst](https://github.com/mconst).

## Goals

We would like to support a number of use cases for functions and types that
could be solved by generating code at build time via built-in Carbon language
features. Today, some of these use cases are addressed in C++ using preprocessor
macros and template metaprogramming.

**Desiderata:**

- Expressive: support many of the use cases where users would otherwise be
  tempted to write code that generates code, including those outlined in the
  next section.
- Clarity: for readers and writers, at both definition and use of
  metaprogramming constructs. This means that things that do not act like
  ordinary (non-meta) behavior do not look ordinary. And things that are
  ordinary still look ordinary even when written in a metaprogramming context.
- Consistency: Less to learn and more predictability. (Example problem: C++
  template metaprogramming ends up being a very different language from the rest
  of C++, where the syntax doesn't even parallel normal code.)
- Support for tools. This includes allowing metaprogramming constructs to be
  selectively expanded on an as-needed basis, potentially at a variable level of
  detail. Also, when a user asks to go to where a symbol is defined, we need a
  good place in the source to point them to.
- Integrated: Metaprogramming features should work well with other language
  features. For example, we should avoid the pitfalls of C/C++ where
  preprocessor macros don't obey rules for namespacing, name collisions, etc.
- Safe: There are two parts to this. (a) We would like to avoid pitfalls that
  make the natural way users write code using this feature violate their
  expectations. This includes problems with hygiene, multiple evaluation,
  precedence, etc. Of course, metaprogramming is powerful and the wide range of
  use cases we want to be able to express means that it will likely be possible
  to use metaprogramming features to do surprising things, but this should not
  be the default. (b) We don't want to compromise the build process itself. In
  particular, builds should be reproducible and only use known, specified
  inputs.

If meta-programming is easier, more people will be able to employ it as a
strategy for making high-performance code.

**Non-goal:** Allow arbitrary changes to Carbon syntax such as for the creation
of DSLs.

## Example use cases

- Handle variable numbers of arguments to a function
- Add serialize method / deserialize factory function to a struct/class (using
  compile-time reflection capabilities). TODO: expand this into a case study
  since it is both difficult and real-world, then compare with Rust Serde.
- Enum val -> string (or string -> enum val)
- Code specialization for (time or memory) performance
- Define members of a struct/class using code (e.g. what the proto compiler
  does)
- Assert function that prints out useful information on failure (text of the
  expression that failed, file & line it occurred on, etc.)
- Array of structures -> structure of arrays conversion (e.g.
  [https://www.circle-lang.org/s2a.html](https://www.circle-lang.org/s2a.html))
- Convert class -> version of class that caches the results of (some) function
  calls. For example, maybe each function call is an expensive step in a data
  processing pipeline.

## Background

Different languages use different approaches to support metaprogramming, and
sometimes use a combination of approaches. I'm going to divide these approaches
into three camps:

### Macros

Here I'm going to include anything that operates at a source text or token
level, at an early stage in the compilation process, particularly if can take
source as input and transform it in some way. Examples include:

- C/C++ preprocessor macros
- Rust macros, of which there are multiple flavors, and are a work in progress
  (to add features such as namespacing)
- Lisp macros which take code-as-lists and produces code-as-lists.
- Writing a program that outputs source that you run at build time (which can be
  supported in a language's ecosystem, as in Go:
  [https://blog.golang.org/generate](https://blog.golang.org/generate)).

There are many known pitfalls when using C/C++ preprocessor macros. Some
examples are listed
[here](https://gcc.gnu.org/onlinedocs/cpp/Macro-Pitfalls.html) and
[here](https://stackoverflow.com/questions/14041453/why-are-preprocessor-macros-evil-and-what-are-the-alternatives).
Some macro systems fix some of these problems (like some are
[hygienic](https://en.wikipedia.org/wiki/Hygienic_macro) or provide something
like Lisp's gensym facility, avoid precedence problems, etc.), but some problems
are essential:

- The code as written does not conform to the grammar the compiler ultimately
  consumes. This can be hard for code readers as well as tools such as syntax
  highlighters, refactoring tools, linters, code formatters, etc. For example, a
  macro can transform illegal looking tokens sequences into legal code. Or it
  could return a token sequence that represents a fragment of code like "3 +"
  instead of a complete expression.
- Difficulty understanding compiler errors that result from macro expansion.
- Pitfalls writing macros in a way that matches the author's intentions that
  result from operating at a low (non-semantic) level.
- Macro code looks very different from non-macro code due to
  quoting/unquoting/quasi-quoting, gensym, etc. In particular, inside a macro
  definition ordinary operations (like using the value of an argument passed in)
  look different than they would in a function body.
- Outside of languages that have a very simple syntax (e.g. Lisp variants),
  there can be a lot of complexity taking code as input. This comes up, for
  example, in Rust's proc macros, which expose a lot of internal compiler APIs
  for operating on code.

As a general rule, macros score poorly on clarity, consistency, tool support,
integration, and easy misuse.

There are some good points though:

- Macros can be very expressive.
- Macros can require very few language features / mechanisms / syntax, instead
  using ordinary language features to write code-manipulating-code.
- Sometimes you want non-hygienic behavior, like Rust's `try!` macro that can
  affect the caller's control flow.
- Macros are _old_, so they are pretty familiar to a number of programmers. That
  being said, as well understood as they are, the path to including them in a
  language is not clear. This can be seen in new languages that adopt macros
  such as Rust, which are still iterating in efforts to address these concerns.

In short, macros are arguably are _too powerful_, allowing a wider range of uses
than is actually desirable, with little to keep you on a good path. As a result,
we aim to support metaprogramming in Carbon without macros.

### Structured metaprogramming

This bucket includes anything that shows up in the language's grammar. Examples:

- C++ template metaprogramming;
- [Circle](https://github.com/seanbaxter/circle/blob/master/examples/README.md) -
  features using @ minus the macro stuff.
- ["inline for"](https://ziglang.org/documentation/master/#inline-for),
  ["inline while"](https://ziglang.org/documentation/master/#inline-while), and
  ["comptime" code blocks](https://ziglang.org/documentation/master/#comptime)
  in [Zig](https://ziglang.org/)

Unlike with macros, there are no "functions that map code -> code". Instead the
grammar includes meta constructs that produce AST. There can be code that is
executed at compile-time that affects the process and that code can do things
like compile-time reflection on types, but without the complexity of directly
operating on the code itself. The AST produced represents a complete grammatical
concept like an expression, statement, or declaration, rather than a fragment
like "3 +" that you can get from a sequence of tokens.

This approach has some nice properties:

- Syntax that is explicitly modeled in the grammar is better for tooling.
- Instead of one very powerful mechanism, we compose together several
  more-specific mechanisms.
- Integrated: Uses the same braces as the non-meta language, so braces always
  match; obeys namespacing, etc.

A lot hinges, though, on the quality of the meta features provided. Done well
(not like C++ template metaprogramming), it _can_ be clear and safer than
macros. This safety is achieved by making each individual meta feature, which by
itself is less powerful, ergonomic and safe.

To make it consistent, we can follow some rules:

- A similar programming model for writing metaprogramming vs. regular code
  (contrast with C++ where metaprogramming is mostly done in a functional,
  recursive style using partial specialization, templates, inheritance, etc.).
- Consistently introduce metaprogramming & compile-time constructs using the
  same syntax (the `@` symbol in Circle).
- Make metaprogramming constructs parallel regular constructs. For example,
  meta-`for` should have the same form as regular `for`, similarly with
  meta-`if` and regular `if`. The rules for meta scopes should be consistent
  across all meta control flow constructs.

The simplest version of the programming model is that there are conceptually two
compiler passes. The first pass runs meta code and the result is an AST without
any meta constructs that is the input to the second compiler pass. Template
instantiation can add some complexity to this model. It is desirable that the
meta constructs used in the body of a templated function get evaluated when that
function is instantiated. This raises questions about the order that meta code
gets evaluated, particularly if there are any side effects that could affect the
interpretation of any meta constructs outside of that function.

Note that the PL community has less experience with this style of
metaprogramming, so less is known about the pitfalls and best practices for
adding this to a programming language. The good news is Circle is already paving
a path forward here, demonstrating something better than C++ template
metaprogramming. Unlike Circle, though, we don't want to allow unrestricted
access to things like the clock or I/O during compilation, both for security and
reproducibility.

### Compile-time partial evaluation

This bucket includes mechanisms that use exactly the same syntax as runtime
code, but instead evaluates at compile time. This can arise as the natural
evolution of partial evaluation and inlining done in an optimizer. It can
include things like:

- Functions that can be executed at compile time or runtime without changing
  code.
- Compile-time vs. runtime evaluation controlled by types rather than syntax.
- Compile-time operations use the same syntax as runtime operations, but are
  more dynamic, less static type checking. Examples:
  - reflection information about a type exposed via ordinary method calls on
    compile-time objects. That reflection information can return a type that can
    e.g. be used to define a variable
  - compile-time duck-typing by instantiating a function for each type used to
    call it
- Compile-time requirement -- assert that something is known at compile time.

In [Zig](https://ziglang.org/):

- <code>[comptime](https://ziglang.org/documentation/master/#comptime)</code>.
  - "This works because Zig implicitly inlines <code>if</code> expressions when
    the condition is known at compile-time, and the compiler guarantees that it
    will skip analysis of the branch not taken."
  - "A programmer can use a comptime expression to guarantee that the expression
    will be evaluated at compile-time. If this cannot be accomplished, the
    compiler will emit an error."
  - "This means that a programmer can create a function which is called both at
    compile-time and run-time, with no modification to the function required."
  - "In the global scope (outside of any function), all expressions are
    implicitly comptime expressions. This means that we can use functions to
    initialize complex static data."
- [What is Zig's Comptime?](https://kristoff.it/blog/what-is-zig-comptime/)
- Zig
  [Generic data structures and functions](https://ziglang.org/#Generic-data-structures-and-functions)
- Zig
  [Compile-time reflection and compile-time code execution](https://ziglang.org/#Compile-time-reflection-and-compile-time-code-execution)

One concern with these sorts of features is how to reconcile calling a function
in another library at compile time with separate compilation (part of Carbon's
scalability goals). One aid here is that we can at least indicate, as part of a
function's signature, which of its arguments are required to be known statically
/ at compile time at the call site. Racket addressed this issue by
[having a different way of importing other modules](https://beautifulracket.com/explainer/importing-and-exporting.html)
(also see
[here](https://www.greghendershott.com/fear-of-macros/Transform_.html)):
<code>(require (for-syntax ...))</code> instead of <code>(require ...)</code>.

Furthermore, there is a general concern that a function's signature will only
capture type information, and not the additional requirements needed to evaluate
code during compilation.

## Proposal

**Overall approach:** Structured metaprogramming and compile-time partial
evaluation are generally speaking compatible with each other. Frequently the
structured metaprogramming approach will be more explicit, and we should use it
when we think a construct should be highlighted in the code. Other cases we will
decide to do something more implicit, because the compiler can straightforwardly
disambiguate and adding syntax would only add clutter. This clutter could make
something safe look similar to things that we really want to highlight, and
should be avoided.

**Principles:**

- Limit power by only implementing constructs we are comfortable with.
- Avoid clutter in the form of switching back and forth between meta code and
  regular code for mundane tasks.
- Be consistent about using a single symbol to signal metaprogramming
  constructs, if we decide that any distinction is needed.
- ...

**Note:** In this document I'm using the symbol `@` to introduce
meta-constructs, but this should be unified with the `:$` and `:$$` syntax used
in the
[Carbon templates and generics (TODO)](#broken-links-footnote)<!-- T:Carbon templates and generics -->
and
[Carbon pattern matching](https://github.com/josh11b/carbon-lang/blob/pattern-matching/docs/design/pattern-matching.md)
docs.

### Case study: tuple struct

**Problem:** Create a struct with `n` integer fields, where `n` is a
compile-time integer.

This problem is meant to be representative of the general problem of adding
fields to a struct (or synthesizing a struct wholesale) from some compile-time
metadata.

**Proposal:** We follow Circle's approach of having a "meta for" loop, where:

- This `for` loop can be used in contexts where a regular `for` loop can not,
  like to generate declarations in a struct.
- We use an `@` to indicate this is a special meta construct (structured
  metaprogramming approach).
- The iteration condition will be interpreted at compile time without any
  additional syntax.
- The body of the loop is non-meta code, that is regular code that will be part
  of the result of executing the meta for loop. It should be possible to parse
  it into an AST once.
- The result of executing the loop is the code in the body of the loop is
  repeated `n` times and inlined in the containing scope. In particular the
  meta-scope does not introduce a new scope in the resulting AST.
- Inside the body of the loop you may use the loop variable as an integer
  directly and it will be treated as a constant (though a different constant in
  each repetition).
- If you want to synthesize an identifier using the loop variable you need to
  use a meta syntax (`@(...)` below, using Circle's syntax).
- Any expression defining the type of a variable will be evaluated at compile
  time, no meta syntax needed.

      ```

  // To use n in a meta-construct, it has to be a template argument. // To be
  consistent, we'd likely spell this `Int:@@ n`, but I'm using // `Int:$$ n` to
  match other Carbon docs. struct NTuple(Int:\$\$ n) { @meta for (int i = 0; i <
  n; ++i) { var Int: @("field\_" + Str(i)) = 2 \* i; } }

var NTuple(3): x;

```



is evaluated at compile time to be essentially equivalent to:


```

struct NTuple*3 { var Int: field_0 = 2 * 0; var Int: field*1 = 2 * 1; var Int:
field_2 = 2 \* 2; }

var NTuple_3: x;

```


**Note:** the `@meta` syntax above matches Circle. Circle uses `@meta` consistently to indicate a **statement** (or block of statements) is to be executed at compile time, and other `@` constructs for computing values at compile time. `@meta` applied to `for`, along with other control-flow statements, only applies to the `for` itself, not the block of code it repeats, which is treated as runtime code unless annotated otherwise.

This is the clearest expression of the programmer's intent we have found. It uses the normal `struct` syntax where possible, and a `for` loop construct that parallels the `for` loop used in non-meta code.

**Question:** Do we want to allow modularity here; in particular, is there a way to write a function that returns a declaration?

This seems like something we can add later if desired. The syntax would be something like:


```

@mdecl Foo() { // meta declaration return @decl { var Int: x; }; }

```


**Alternative considered:** A compile-time partial evaluation API would be to define an API for constructing types available to code running at compile-time. For example:


```

// Ordinary function, no meta syntax in sight. fn NTuple(Int: n) -> Type { var
StructBuilder: result*struct; for (Int: i = 0; i < n; ++i) {
result_struct.add_data_field(Int, "field*" + Str(i), .default = 2 \* i) } //
StructBuilder.finalize() returns the completed Type value. return
result_struct.finalize(); }

// Compile-time evaluation triggered by using the function as the type of a //
variable. var NTuple(3): x;

```


One concern with this approach is that the API for dynamically defining struct fields looks nothing like the normal way of defining struct fields.

**Alternative considered:** This was suggested as a possibility arising from the discussions about how this might be done in [Template Haskell](https://downloads.haskell.org/~ghc/latest/docs/html/users_guide/glasgow_exts.html#template-haskell) (or some C++ proposals?):


```

fn NTuple(Int: n) -> Type { return {type| struct { @{ for (int i = 0; i < n;
++i) { co*yield {| var Int: @("field*" + Str(i)) = 2 \* i; |}; } } } |}; }

var NTuple(3): x;

```


(Note: the following is [josh11b](https://github.com/josh11b)'s likely mistaken interpretation of what is going on here.)

The idea here is that we support types representing various grammatical types like "type declaration", "id", "variable declaration", etc. Each of these would have a construct for producing values of that type, containing an AST branch, using the `{optional-specifier| ... |}` syntax. Inside that syntax, you would write code as if it were a normal declaration (or whatever), but you can escape to meta code using `@{...}`. In that meta code block you can emit regular code using the `co_yield` keyword followed by an expression returning syntax, like another `{optional-specifier|...|}` block or a function returning one of the AST types.

The good news here is that a struct definition is being assembled from pieces that actually use the normal struct definition syntax. The concerns here have to do with noise coming from switching between meta and non-meta execution more than is desirable, hiding what is ordinary and what is meta.

**Observation:** I believe the root problem here is trying to use an ordinary `for` loop by providing ways of switching between meta and non-meta context (such as quasiquote and unquote operators), instead of a dedicated meta-`for` construct. Using an ordinary `for` loop is noisier, less clear, and can't have semantics tailored to the needs of writing meta code that a meta-specific `for` loop operator can.


### Case study: assert

**Problem:** Define a function corresponding to `assert()` macro in C/C++.

In particular, the C/C++ version of `assert()` makes use of:



*   Getting the expression passed to `assert` as a string to include in the failure message.
*   Getting the source location of the call.
*   Conditional compilation.
*   Lazy evaluation of the argument.

**Rejected alternative:** If we just want to be able to print a message that includes the text, file, and line of the expression passed in, we will provide meta operators (using a `@`) for getting that information from a templated argument. These special capabilities do not need to be reflected in the calling code, just the signature of the `Assert` function.

**Proposal:** We have a meta-`if` that works like regular `if` except:



*   The condition has to be a compile-time value.
*   The branch taken is inlined, so the meta-`if` scope is not treated as a non-meta scope.
*   The branch not taken will be parsed but not compiled, so it need not type check and will be guaranteed to produce no code.

This is analogous to and consistent with the meta-`for` used above.

**Rejected alternative:**


```

fn Assert(Bool:@@@ condition) { @meta if (!NDEBUG) { // If in NDEBUG mode, the
body here would be parsed but not compiled. if (!condition) {
Log.Fatal(StrCat("Assertion failure: ", @arg_str(condition), " in file: ",
@arg_file(condition), " line: ", @arg_line(condition))); } } }

Assert(f());

```


Here we are using `@@@` to indicate the `condition` argument gets special treatment. Like `@@`/`$$`, as a result of this argument we are in "template mode" which means that we need to make the body of this function available to the caller so the caller can expand it based on the call site. However, with `@@` we would only be able to use the value of `condition` in the body of the function, and that value would have to be known statically / at compile time. The extra level of meta here is to instead of using the value, we would use the expression itself from the callsite. So the levels are:



*   0- (`:`) normal argument: value is passed at runtime. Meta-expansion occurs at function definition time.
*   1- (`:@`) generic argument: the value needs to be known to the caller at compile time, and will be used to instantiate/specialize the function body for that value, but the value can't be used in type-checking of the function. Meta-expansion occurs at function definition time.
*   2- (`:@@`) template argument: the value needs to be known to the caller at compile time, and will be used to instantiate/specialize the function body for that value, and that instantiation will happen before type-checking. To do this, the body of templated functions needs to be visible to callers. Meta-expansion happens at instantiation time, and can use the value of template arguments.
*   3- macro argument (`:@@@`): like template arguments, except in some sense the expression itself used by the caller is passed in, not the value per-se, and so the value need not be known at compile time.

**Proposal:**  Note that unlike the C/C++ `assert` macro, the condition is still evaluated even if `NDEBUG` is `True` -- that is the function `f` will get called. The compiler might be able to optimize it away, but any side effects of computing `condition` would still have to happen.

Our judgement is that `f()` not being evaluated would be too big of a change from normal function invocation to be done silently. It would have to be reflected in how the function was called. We may reconsider this decision, see the alternative below.

**Proposal:** If an argument may be evaluated zero or more than once, that should be reflected by some syntax at the call site, like being put in a lambda.

We did not see a need to support things like the short-circuit evaluation semantics of operators `||` and `&&` in user-defined constructs.

**Concern:** This would require a very convenient lambda syntax in Carbon so that it was not much burden on callers.

**Alternative:** In general, arguments may be marked "lazy" -- the caller's expression is not evaluated until explicitly requested inside the body of function.


```

fn Assert(@autoclosure(Bool): condition) { ... }

```


This is [available in Swift](https://docs.swift.org/swift-book/LanguageGuide/Closures.html#ID543). There it is called `@autoclosure`, see for example: [Using @autoclosure when designing Swift APIs](https://www.swiftbysundell.com/articles/using-autoclosure-when-designing-swift-apis/). That link lists a few different use cases.

**Proposal:** The `Assert` function could be made into something that could be compiled separately if all the meta dependencies on arguments were moved into the signature.


```

fn Assert(Bool: condition, .text = String: text = @arg_str(condition), .file =
String: file = @calling_file, .line = Int: line = @calling_line) { @meta if
(!NDEBUG) { // Do nothing in NDEBUG mode. if (!condition) {
Log.Fatal(StrCat("Assertion failure: ", text, " in file: ", file, " line: ",
line)); } } }

Assert(f());

```


**Note:** Here `condition` is no longer a template/meta argument. In this case, you could make an ordinary function call by supplying all the optional arguments. This call might be equivalent to something like:


```

Assert(f(), .text = "f()", .file = "my_file.carbon", .line = 42);

```


One advantage of this approach is to allow the `file` and `line` arguments to be overridden in generated code to point to something more user friendly.

**Note:** The semantics of `@arg_str(x)` are:



*   `x` is required to exactly match the name of an argument to this function;
*   the result of `@arg_str(x)` is a string with the source code text that results in the value of `x` in the caller, possibly with whitespace normalized and comments removed.

**Note:** For `Assert`, it is better to have the file name and line number of the function call rather than the argument, hence the `@calling_file` and `@calling_line` constructions in the example above. This is likely to be more useful in general, and would also work for functions that don't take arguments. If necessary, we could also have `@arg_file(x)` and `@arg_line(x)` that take the name of a previous argument, like `@arg_str(x)`.


### Case study: variadics

**Context:** see [Carbon tuples and variadics (TODO)](#broken-links-footnote)<!-- T:Carbon tuples and variadics --> and [Carbon pattern matching](https://github.com/josh11b/carbon-lang/blob/pattern-matching/docs/design/pattern-matching.md), in particular [the variadics section](https://github.com/josh11b/carbon-lang/blob/pattern-matching/docs/design/pattern-matching.md#variadics-).

In short: A function taking a variadic argument (`...`) is treated as a templated function in that it is instantiated at each call site. The arguments themselves are passed in a tuple, with varying number and types matching whatever is passed at the callsite. There is a postfix `...` operator for expanding the tuple when calling a function.

**Problem:** How would you write `StrCat`? We need to be able to do things like define local arrays with size matching the number of arguments, iterate through the arguments, etc.

**Proposal:** Can use meta-`for` to iterate through a tuple.

One advantage of meta-`for` over regular `for` is it more naturally handles the fact that the element types vary.

**Question:** How do you get the length of a tuple? the types? the values? the element names?

In particular, do we use a meta operator introduced with a `@` or do tuples have member functions that are only available at compile time? Note that for the types, the type of a tuple (possibly accessible via a `typeof` operator?) is a tuple of the types.

My inclination is that tuples and types should be thought of as generally compile-time things, and can have member methods that expose compile-time information.


```

fn StrCat(...: args) -> String { var Int: reserved = 0; var
Optional(StringView)[args.length()]: views; // ?? @meta for (var (Int: i, auto:
a) in Enumerate(args)) { // ? // Add the length of `a` as a string to
`reserved`. // For some types it is efficient to compute views[i] at the same
time. // Use a `match` here to run different code based on the type of `a`. }
var String: ret; ret.reserve(reserved); @meta for (var (Int: i, auto: a) in
Enumerate(args)) { // Append the string representation of `a` to `ret`. match
(a) { ... } } return ret; }

```


**Question:** Should the type of `...` matching only positional arguments be different from the type of `...` matching keyword arguments?

True for Python, but I think we want to use tuples for both.

**Proposal:** We should support a variety of variadic patterns, in addition to the positional vs. keyword distinction.



*   Varying types vs. all the same type.
*   Within all the same type: type is a template argument, a generic argument, or fixed (`Int`).
*   Within all the same type: the function may either be instantiated for each length used at compile time or the length could be passed in at runtime.

**Concern:** The syntax proposed above does not obviously generalize to something that supports these alternative variadic patterns. Maybe:



*   `... PositionalArgs: args` -> tuple with positional args of any type
*   `... KeywordArgs: args` -> tuple with keyword args of any type
*   `... KeywordsOfType(T): args` -> all the same type, only keyword arguments
*   `... KeywordsOfType(Int): args` -> all Int, only keyword args
*   `... PositionalOfType(T): args` -> all the same type, only positional arguments

**Concern:** In the "all the same type" case, if the length is zero, it may be awkward to say what the member type is. We may want to restrict these to the "length >=1" case.

**Concern:** Implicit conversions could complicate the process of deducing a consistent type.

**Proposal:** In the "all the same types and the length passed in at runtime" case, could potentially support passing in a sequence type (like `vector`). Or if the type is fixed, we might not even need to do template/generic instantiation (at least not for that argument).


### Case study: reflection

**Question:** Should reflection be done via meta operators (explicitly using something with `@` syntax) or via ordinary methods on types, enums, etc. that are only available at compile time?

Circle uses meta operators:



*   <code>[@type_name](https://github.com/seanbaxter/circle/blob/master/examples/README.md#introspection-keywords)</code> mapping <code>Type</code> -> <code>String</code>.
*   [type/class/struct reflection/introspection:](https://github.com/seanbaxter/circle/blob/master/examples/README.md#introspection-keywords) <code>@member_count</code>, <code>@member_name</code>, <code>@member_ptr</code>, <code>@member_ref</code>, <code>@member_type</code>
*   <code>[@meta for enum](https://github.com/seanbaxter/circle/blob/master/examples/README.md#for-enum-statements)</code>, <code>[@enum_count, @enum_value, @enum_name](https://github.com/seanbaxter/circle/blob/master/examples/README.md#introspection-on-enums)</code>, <code>[@enum_type](https://github.com/seanbaxter/circle/blob/master/examples/README.md#introspection-keywords)</code>
*   <code>[@is_class_template](https://github.com/seanbaxter/circle/blob/master/examples/README.md#template-metaprogramming)</code>

However that may in some cases be just to distinguish Circle from ordinary C++ code, since you end up using ordinary C++ constructs (to be fair, template metaprogramming) in a similar way. For example, the Circle doc has examples using <code>std::is_enum&lt;></code>, <code>std::is_class&lt;></code>, <code>std::underlying_type&lt;></code>, <code>std::is_void&lt;></code>, <code>std::is_integral&lt;></code>, <code>std::is_floating_point&lt;></code>, <code>std::is_array&lt;></code>, <code>std::extent&lt;></code>.

Furthermore, many of the reflection/introspection operators would naturally be members of the `Type` or `Enum` APIs instead of in a global namespace. My inclination would be to make any  compile-time-only API that is naturally a member into a function that doesn't need an `@`, but I don't feel strongly (saying all compile-time APIs have an `@` would be a consistent position and would potentially add clarity by being explicit). Possibly we could all provide these at runtime via an RTTI interface? Or perhaps we have an `@` member syntax (like `my_type.@Name` or `my_type.@MemberCount`)?


### Debugging compile-time code execution

You have written your compile-time meta code, now how do you debug it?



*   `printf` debugging using a compile-time logging statement `@meta Log.Info("...");`.
*   Ability to dump the output of the meta-compilation step for a given function, struct, etc.
*   Attaching a debugger to the compiler. The compiler should do whatever is needed to allow you to set breakpoints in meta code, query meta variables, etc.

**Rejected alternative:** 2-step build. The idea is the first step compiles the meta-statements to produce an executable. The second step executes that program, executing the meta-statements, to produce a non-meta AST, that then gets compiled. To debug your meta code, you would attach a debugger to this second program. Its main entry point would be your meta code and the compiler would be a library invoked by it.

This model seems simple at first, but gets more complicated when you consider that templated functions will execute their meta-code at instantiation time (likely in other modules entirely), see [below](#rules-for-meta-scopes).


### Partial expansion

The Carbon compiler will have to execute/expand all of the meta code in the files it compiles, but other tools, particularly IDEs may not need to.

This could be for speed. For example, an IDE might not need to examine the bodies of functions, but that might be tricky if meta-code inside a function has side effects on the meta-environment that can alter future declarations that the IDE does care about. Pre-expanding the code could be quite expensive, too, since it could multiple the amount of code to consider by a large factor.

IDEs traditionally have been able to do this by matching braces and ignoring scopes at some level of detail the IDE does not need. This technique could also be used to support graceful degradation in the face of incomplete or ill formed code.

A big benefit of not using traditional macros is we can still match up braces -- there is just a bit more complexity to handle the fact that some braces are meta. Consider an example:


```

struct Foo(Int:@@ n) { @meta for (var Int: i = 0; i < n; ++i) { fn @("Member" +
Str(i))(Int: x) { // _lots_ of code @meta for (var Int: j = 0; j <= i; j += 2) {
// more code } } } }

```


Let's say we want to just figure out the names and signatures of all the members of `Foo`. We parse the source code into an AST under the language grammar that includes meta constructs. This parse will match up curly braces (`{...}`) for meta constructs and non-meta constructs. You can then throw away any non-meta code inside a (non-meta) declaration to get the desired level of detail. In our example:


```

struct Foo(Int:@@ n) { @meta for (var Int: i = 0; i < n; ++i) { fn @("Member" +
Str(i))(Int: x) { // _lots_ of code @meta for (var Int: j = 0; j <= i; j += 2) {
// more code } } } }

```


Since we haven't executed the meta code, we have not expanded any meta-for statements, which should help keep things manageable. Furthermore, the meta constructs inside the declarations can in all likely be recognized as dead code:


```

struct Foo(Int:@@ n) { @meta for (var Int: i = 0; i < n; ++i) { fn @("Member" +
Str(i))(Int: x) { @meta for (var Int: j = 0; j <= i; j += 2) { } } } }

```


Leaving us with just the meta code we need to expand to get the desired declarations:


```

struct Foo(Int:@@ n) { @meta for (var Int: i = 0; i < n; ++i) { fn @("Member" +
Str(i))(Int: x) { /_ successfully elided _/ } } }

```


Then a particular use of `Foo` with a specific `n`, we perform the expansion by actually executing the meta code:


```

var Foo(3): a;

a.<request completion> // Members of Foo(3): // Member0(Int: x) // Member1(Int:
x) // Member2(Int: x)

```


A tricky case is if some inner scope mutates some state of an outer scope:


```

struct Foo(Int:@@ n) { @meta for (var Int: i = 0; i < n; ++i) { fn @("Member" +
Str(i))(Int: x) { // _lots_ of code @meta for (var Int: j = 0; j <= i; j += 2) {
@meta if (j == i) { @meta ++i; // Question: is this an error? } // more code } }
} }

```


The plan is the "dead code" detection would see a side effect it could not eliminate. It would then do the slower-but-correct thing of executing the inner loop.

**Rejected alternative:** Meta classes. The idea is to define the API for a type, and then allow users to implement that API directly. The intent is to be more reactive -- the type only provides the information in response to a query.

Example use case: There is some proposal to generate a permutation of some data by the name of the accessor used to access it:


```

struct Color { var Int: r; var Int: g; var Int: b; var Int: a; } var Color[64]:
c = ...; Print(MagicPermute(c).argb())

```


You could have a function that returned a lambda given a member name. For IDEs, would also need to know the legal completions given a prefix.


### Rules for meta scopes

**Rule:** meta-expand generic and regular functions (only `:$` and regular args) when defined.

**Rule:** meta-expand templates (anything with an `:$$` or `...` arg) when instantiated by a caller.

Since template instantiation happens in the middle of processing other files, a variable number of times, it needs to be reproducible and can't have side effects outside the scope of the instantiated function. Probably we need to save the meta environment at the point of a templated definition for use when it is later expanded. When later instantiating the templated function, would then mark this meta environment as read-only, triggering an error at any attempt to modify a meta variable while executing the meta code in the body of the function.

In addition, Circle uses these rules (from [the Circle README](https://github.com/seanbaxter/circle/blob/master/examples/README.md#tldr)):



*   Meta object declarations have automatic storage duration over the lifetime of the enclosing scope. This scope may be a namespace, a class-specifier, an enum-specifier or a block scope. For example, you can create meta objects inside a class definition to help define the class; these meta objects are torn down when the class definition is complete.
*   Meta control flow is executed at compile time. These loops are structurally unrolled, and the child statement is translated into AST at each iteration. Programs are grown algorithmically as if by deposition: each real statement in a meta scope deposits itself in the innermost enclosing scope. By controlling how real statements are visited with meta control flow, we can guide the program's creation.

See more details [here](https://github.com/seanbaxter/circle/blob/master/examples/README.md#same-language-reflection).


### Other questions about what gets executed at compile time

**Proposal:** Let's say you write a non-meta for loop on static data. Question: Does it get unrolled? Answer: that is up to the optimizer. It would not be guaranteed by the language spec, but would be a reasonable optimization that the optimizer might perform depending on factors such as the amount of code generated.

**Question:** In various contexts, what will be the language contract w.r.t. whether we will evaluate an expression (e.g. call a function) at compile time without `@`?



*   Expressions where a type is expected: probably yes we will decide that the language will guarantee those expressions will be evaluated at compile time.
*   Expressions where a compile time value is required, like a template argument in a call to a function, or the size of an array: likely yes
*   Expressions where the inputs are known at compile time: up to the optimizer, no language contract
*   Operations which would only be allowed at compile time: not sure, maybe all those operations should have an `@`?


### Beef Mixins

[https://www.beeflang.org/docs/language-guide/datatypes/members/#mixins](https://www.beeflang.org/docs/language-guide/datatypes/members/#mixins)

Idea here is that "break" and "return" in a mixin actually execute in the caller's context, so you can write an error-handling function that returns *from the caller* when there is an error.



---



## C++ template metaprogramming uses

CppCon 2014: Walter E. Brown "Modern Template Metaprogramming: A Compendium, Part I"

[https://www.youtube.com/watch?v=Am2is2QCvxY](https://www.youtube.com/watch?v=Am2is2QCvxY)

Use cases:



*   Running pure functions (abs, gcd) at compile time.
*   Functions operating on types (sizeof, rank of an array type, is_integral, is_void, is_same).
*   Something returning a type (remove const from a type).
*   Machinery for doing ordinary things except in meta-programming (type_is, integral_constant, conditional_t).
*   SFINAE: Substitution Failure Is Not An Error, enable_if, for specifying which overloads to consider (`if` in ["Carbon pattern matching"](https://github.com/josh11b/carbon-lang/blob/pattern-matching/docs/design/pattern-matching.md))

CppCon 2014: Walter E. Brown "Modern Template Metaprogramming: A Compendium, Part II"

[https://www.youtube.com/watch?v=a0FliKwcwXE](https://www.youtube.com/watch?v=a0FliKwcwXE)



*   Operating on lists of types (parameter packs in C++): is_one_of
*   Pattern matching, using best match not first match, is main tool
*   Machinery: sizeof, typeid, decltype, noexcept are not evaluated
*   What is the return type after overload resolution of a function call? std::declval&lt;T>() not implemented, just declared, gives something invalid for evaluation of type T
*   is_copy_assignable, is move_assignable: is an expression legal? ("something of type U& = something of type U const &") Doesn't require any values, can just use types. Can also test if the expression returns a specific type.
*   void_t: is type well formed?
*   has_type_member: Does class have a type member named "type"?


## Broken links footnote

Some links in this document aren't yet available,
and so have been directed here until we can do the
work to make them available.

We thank you for your patience.
```
