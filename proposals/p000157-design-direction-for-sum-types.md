# Design direction for sum types

<!--
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

[Pull request](https://github.com/carbon-language/carbon-lang/pull/157)

## Table of contents

<!-- toc -->

## Table of contents

-   [Problem](#problem)
-   [Background](#background)
-   [Proposal](#proposal)
-   [Shareable storage](#shareable-storage)
-   [User-defined pattern matching](#user-defined-pattern-matching)
-   ["Bare" designator syntax](#bare-designator-syntax)
    -   [Alternatives considered](#alternatives-considered)
        -   [Placeholder keywords](#placeholder-keywords)
        -   [Designator types](#designator-types)
-   [Distinguishing pattern and expression semantics](#distinguishing-pattern-and-expression-semantics)
    -   [Alternatives considered](#alternatives-considered-1)
        -   [Separate syntaxes](#separate-syntaxes)
        -   [Disambiguation by fixed priority](#disambiguation-by-fixed-priority)
        -   [Disambiguation based on pattern content](#disambiguation-based-on-pattern-content)
-   [Choice types](#choice-types)
    -   [Alternatives considered](#alternatives-considered-2)
        -   [Different spelling for `choice`](#different-spelling-for-choice)
-   [Alternatives considered](#alternatives-considered-3)
    -   [`choice` types only](#choice-types-only)
    -   [Indexing by type](#indexing-by-type)
    -   [Pattern matching proxies](#pattern-matching-proxies)
    -   [Pattern functions](#pattern-functions)
-   [Rationale](#rationale)

<!-- tocstop -->

## Problem

Many important programming use cases involve values that are most naturally
represented as having one of several alternative forms (called _alternatives_
for short). For example,

-   Optional values, which are pervasive in computing, take the form of either
    values of some underlying type, or a special "not present" value.
-   Functions that cannot throw exceptions often use a return type that can
    represent either a successfully computed result, or some description of how
    the computation failed.
-   Nodes of a parse tree often take different forms depending on the grammar
    production that generated them. For example, a node of a parse tree for
    simple arithmetic expressions might represent either a sum or product
    expression with two child nodes representing the operands, or a
    parenthesized expression with a single child node representing the contents
    of the parentheses.
-   The error codes returned by APIs like POSIX have a fixed set of named
    values.

What unites these use cases is that the set of alternatives is fixed by the API,
it is possible for user code to determine which alternative is present, and
there is little or nothing you can usefully do with such a value without first
making that determination.

Carbon needs to support defining and working with types representing such
values. Following Carbon's principles, these types need to be easy to define,
understand, and use, and they need to be safe -- in ordinary usage, the type
system should ensure that user code cannot accidentally access the wrong
alternative. These types should be writeable as well as readable, and writing
should be type-safe and efficient. In particular, it should be possible to
mutate a single sub-field of an alternative, without having to overwrite the
entire alternative, and without a risk of accidentally doing so when that
alternative is not present.

Furthermore, it needs to be possible for type owners to customize the
representations of these types. For example:

-   Most sum types need a "discriminator" field to indicate which alternative is
    present, but since it typically has very few possible values, it can often
    be packed into padding, or even the low-order bits of a pointer.
-   Other sum types avoid an explicit discriminator, and instead reserve certain
    values to indicate separate alternatives. For example, a typical C-style
    pointer can be thought of as an optional type, with a special null value
    indicating that no pointer is present, because the platform guarantees that
    the null byte pattern is never the representation of a valid pointer.

It must be possible to implement such customizations without changing the type's
API, and hence without altering the static safety guarantees for users of the
type.

## Background

The terminology in this space is quite fragmented and inconsistent. This
proposal will use the term _sum types_ to refer to types of the kind described
in the problem statement. Note that "sum type" is not being proposed as a
specific Carbon feature, or even as a precisely defined term of art; it is
merely an informal way for this proposal to refer to its motivating use cases,
in much the same way that a structs proposal might refer to "value types".

Carbon as currently envisioned is already capable of approximating support for
sum types. In particular, [pattern matching](/docs/design/pattern_matching.md)
gives us a natural way to express querying which alternative is active, and then
performing computations on that active alternative, which as discussed above is
the primary way of interacting with a sum type. For example, a value-or-error
type `Result(T, Error)` could be implemented like so:

```
// Result(T, Error) holds either a successfully-computed value of type T,
// or metadata about a failure during that computation, or a singleton
// "cancelled" state indicating that the computation successfully complied with
// a request to halt before completion.
struct Result(Type:$$ T, Type:$$ Error) {
  // 0 if this represents a value, 1 if this represents an error, 2 if this
  // represents the cancelled state.
  var Int: discriminator;
  var T: value;
  var Error: error;

  fn Success(T: value) -> Result(T, Error) {
    return (.discriminator = 0, .value = value, .error = Error());
  }

  fn Failure(Error: error) -> Result(T, Error) {
    return (.discriminator = 1, .value = T(), .error = error);
  }

  var Result(T, Error):$$ Cancelled =
    (.discriminator = 2, .value = T(), .error = Error());
}
```

A typical usage might look like:

```
fn ParseAsInt(String: s) -> Result(Int, String) {
  var Int: result = 0;
  var auto: it = s.begin();
  while (it != s.end()) {
    if (*it < '0' || *it > '9') {
      return Result(Int, String).Failure("String contains non-digit");
    }
    result += *it - '0';
    result *= 10;
  }
  return Result(Int, String).Success(result);
}

fn GetIntFromUser() -> Int {
  while(True) {
    var String: s = UserPrompt("Please enter a number");
    match (ParseAsInt(s)) {
      case (.discriminator = 0, .value = Int: value, .error = String: _) => {
        return value;
      }
      case (.discriminator = 1, .value = Int: _, .error = String: error) => {
        Display(error);
      }
      case .Cancelled => {
        // We didn't request cancellation, so something is very wrong.
        Terminate();
      }
      default => {
        // Can't happen, because the above cases are exhaustive.
        Assert(False);
      }
    }
  }
}
```

However, this code has several functional deficiencies:

-   `.value` and `.error` must both be live throughout the `Result`'s lifetime,
    even when they are not meaningful. Consequently, `Success` must populate
    `.error` with a default-constructed dummy value (and so it won't work if
    `Error` is not default-constructible), `Failure` must do the same for
    `.value`, and `Cancelled` must do the same for both. Furthermore, `Result`
    is bloated by the fact that the two fields must have separately-allocated
    storage, even though at most one at a time actually stores any data.
-   The implementation details of `Result` are not encapsulated. This makes the
    `Result` API unsafe: nothing prevents client code from accessing `.value`
    even when `.discriminator` is not 0. This also makes the patterns extremely
    verbose.
-   `.discriminator` should never have any value other than 0, 1, or 2, but the
    compiler can't enforce that property when `Result`s are created, or exploit
    it when `Result`s are used. So, for example, the `match` must have a
    `default` case in order for the compiler and other tools to consider it
    exhaustive, even though that default case should never be entered.

It also has a couple of ergonomic problems:

-   The definition of `Result` is largely boilerplate. Conceptually, the only
    information needed to specify this type is the names and parameter types of
    the two factory functions, the name of the static member `Cancelled`, plus
    the fact that every possible value of `Result` is uniquely described by the
    name and parameter values of a call to one of those two functions, or else
    equal to `Cancelled`. Given that information, the compiler could easily
    generate the rest of the struct definition. This generated implementation
    may not always be as efficient as a hand-coded one could be, but in a lot of
    cases that may not matter.
-   The `return` statements in `ParseAsInt` are quite verbose, due to the need
    to explicitly qualify the function calls with `Result(Int, String)`. In
    fact, developers might prefer to avoid having a function call at all,
    especially in the success case, and instead rely on implicit conversions to
    write something like `return result;`.

## Proposal

To summarize, the previous section identified several missing features in
Carbon, which together would enable Carbon to support efficient and ergonomic
sum types:

-   There's no way to manually control the lifetimes of subobjects, or enable
    them to share storage.
-   There's no way for pattern matching to operate through an encapsulation
    boundary.
-   There's no way for a type to specify that a given set of patterns is
    exhaustive.
-   There's no concise way to define a sum type based on the form of its
    alternatives.
-   There's no way to return a specific alternative without restating the return
    type of the function.

I propose supporting sum types by introducing several language features to
supply the missing functionality identified above. These features are largely
separable, although there are some dependencies between them, so their detailed
design will be addressed in future proposals, and the details discussed here
should be considered provisional. This proposal merely establishes the overall
design direction for sum types, in the same way that [#83](p0083.md) established
the overall design direction for the language as a whole.

To support manual lifetime control and storage sharing, I propose introducing at
least one and preferably both of the following:

-   A `Storage` type, which represents a fixed-size buffer of untyped memory and
    provides operations for creating and destroying objects within it.
-   A typed `union` facility, such as the one described in proposal
    [0139](https://github.com/carbon-language/carbon-lang/pull/139).

To support encapsulation in pattern matching, I propose introducing a
`Matchable` interface, which a type can implement in order to specify how it
behaves in pattern matching, including what (if anything) constitutes an
exhaustive set of patterns for that type.

To allow users to concisely specify a sum type when they don't need to
micro-optimize the implementation details, I propose a `choice` syntax that
specifies a sum type purely in terms of its alternatives, which acts as a sugar
syntax for those lower-level features.

To avoid redundant boilerplate in functions that return sum types, I propose
allowing statements of the form `return .X;` and `return .F();`, which are
interpreted as if the function's return type appeared immediately prior to the
`.` character.

Cumulatively, these features will allow `Status` to be defined so that the usage
portions of the above example can be rewritten as follows:

```
fn ParseAsInt(String: s) -> Result(Int, String) {
  var Int: result = 0;
  var auto: it = s.begin();
  while (it != s.end()) {
    if (*it < '0' || *it > '9') {
      return .Failure("String contains non-digit");
    }
    result += *it - '0';
    result *= 10;
  }
  return .Success(result);
}

fn GetIntFromUser() -> Int {
  while(True) {
    var String: s = UserPrompt("Please enter a number");
    match (ParseAsInt(s)) {
      case .Success(var Int: value) => {
        return value;
      }
      case .Failure(var String: error) => {
        Display(error);
      }
      case .Cancelled => {
        // We didn't request cancellation, so something is very wrong.
        Terminate();
      }
    }
  }
}
```

> **Open question:** How will user-defined sum types (and pattern matching in
> general) support name bindings that allow you to mutate the underlying object?

## Shareable storage

This approach to sum types imposes relatively few requirements on the language
features used to implement shareable storage (meaning, storage that can be
inhabited by different objects at different times), and so this proposal doesn't
describe them in much detail. The primary options are typed unions along the
lines of proposal
[0139](https://github.com/carbon-language/carbon-lang/pull/139), and untyped
byte buffers along the lines described below. Typed unions are somewhat safer
and more readable, but less general. For example, they can't support use cases
like implementing a small-object-optimized version of
[`std::any`](https://en.cppreference.com/w/cpp/utility/any), because the set of
possible types is not known in advance.

This proposal takes the position that Carbon must have at least one of these two
features. Whether it should have only untyped byte buffers, only typed unions,
or both, is left as an **open question**, because the answer is orthogonal to
the overall design direction that is the focus of this proposal. Consequently, I
will not further discuss the tradeoffs between the two features. This proposal's
examples focus on untyped byte buffers because they are simpler to describe, and
aren't already covered by another proposal.

Regardless of the form that shareable storage takes, it won't be able to
intrinsically keep track of whether it currently holds any objects, or the types
or offsets of those objects, because that would require it to maintain
additional hidden storage, and a major goal of this design is to give the
developer explicit control of the object representation. Consequently, it is not
safe to copy, move, assign to, or destroy shareable storage unless it is known
not to be inhabited by an object.

This means that in the general case, the compiler will not be able to generate
safe default implementations for any special member functions of types that have
shareable storage members.

For purposes of illustration in this proposal, I will treat
`Storage(SizeT size, SizeT align)` as a library template representing an untyped
buffer of `size` bytes, aligned to `align`. It provides `Create`, `Read`, and
`Destroy` methods which create, access, and destroy an object of a specified
type within the buffer.

Using `Storage`, we can redefine `Status` as follows:

```
struct Result(Type:$$ T, Type:$$ Error) {
  var Int: discriminator;

  var Storage(Max(Sizeof(T), Sizeof(Error)),
              Max(Alignof(T), Alignof(Error))): storage;

  fn Success(T: value) -> Self {
    Self result = (.discriminator = 0);
    result.storage->Create(T, value);
    return result;
  }

  fn Failure(Error: error) -> Self {
    Self result = (.discriminator = 1);
    result.storage->Create(Error, error);
    return result;
  }

  var Self:$$ Cancelled = (.discriminator = 2);

  // Copy, move, assign, destroy, and similar operations need to be defined
  // explicitly, but are omitted for brevity.
}
```

## User-defined pattern matching

As seen in the example above, we want to allow pattern-matching on `Status` to
look like this:

```
match (ParseAsInt(s)) {
  case .Success(var Int: value) => {
    return value;
  }
  case .Failure(var String: error) => {
    Display(error);
  }
  case .Cancelled => {
    // We didn't request cancellation, so something is very wrong.
    Terminate();
  }
}
```

For this to work, `Status` needs to specify two things:

-   The set of all possible alternatives, including their names and parameter
    types, so that the compiler can typecheck the `match` body, identify any
    unreachable `case`s, and determine whether any `case`s are missing.
-   The algorithm that, given a `Status` object, determines which alternative is
    present, and specifies the values of its parameters.

Here's how `Status` can do that under this proposal:

```
struct Result(Type:$$ T, Type:$$ Error) {
  var Int: discriminator;

  var Storage(Max(Sizeof(T), Sizeof(Error)),
              Max(Alignof(T), Alignof(Error))): storage;

  interface MatchContinuation {
    var Type:$$ ReturnType;
    fn Success(T: value) -> ReturnType;
    fn Failure(Error: error) -> ReturnType;
    fn Cancelled() -> ReturnType;
  }

  impl Matchable(MatchContinuation) {
    method (Ptr(Self): this) Match[MatchContinuation:$ Continuation](
        Ptr(Continuation): continuation) -> Continuation.ReturnType {
      match (discriminator) {
        case 0 => {
          return continuation->Success(this->storage.Read(T));
        }
        case 1 => {
          return continuation->Failure(this->storage.Read(Error));
        }
        case 2 => {
          return continuation->Cancelled();
        }
        default => {
          Assert(false);
        }
      }
    }
  }

  // Success() and Failure() factory functions, and the Cancelled static
  // constant, are defined as above.

  // Copy, move, assign, destroy, and similar operations need to be defined
  // explicitly, but are omitted for brevity.
}
```

In this code, `Result` makes itself available for use in pattern matching by
declaring that it implements the `Matchable` interface. `Matchable` takes an
interface argument, `MatchContinuation` in this case, which specifies the set of
possible alternatives by declaring a method for each one. I'll call that
argument the _continuation interface_, for reasons that are about to become
clear.

The `Match` method of the `Matchable` interface. This method takes two
parameters: the value being matched against, and an instance of the continuation
interface, which the compiler generates from the `match` expression being
evaluated, with method bodies that correspond to the bodies of the corresponding
`case`s. Once `Match` has determined which alternative is present, and the
values of its parameters, it invokes the corresponding method of the
continuation object. `Match` is required to invoke exactly one continuation
method, and to do so exactly once.

This proposal assumes that Carbon will have support for defining and
implementing generic interfaces, including interfaces that take interface
parameters, and uses `interface`, `impl`, `method` etc. as **placeholder**
syntax. It can probably be revised to work if interfaces can't be parameterized
that way, or if we don't have a feature like this at all, but it might be
somewhat more awkward.

Notice that the names `Success`, `Failure`, and `Cancelled` are defined twice,
once as factory functions of `Result` and once as methods of
`MatchContinuation`, with the same parameter types in each case. The two
effectively act as inverses of each other: the factory functions compute a
`Result` from their parameters, and the methods are used to report the parameter
values that compute a given `Result`. This mirroring between expression and
pattern syntax is ultimately a design choice by the type author; there is no
language-level requirement that the alternatives correspond to the factory
functions.

This approach can be extended to support non-sum patterns as well. For example,
a type that wants to match a tuple-shaped pattern like `(Int: i, String: s)`
could define a continuation interface like

```
interface MatchContinuation {
  fn operator()(Int: i, String: s);
}
```

A type's continuation interface can also include a function named
`operator default`, which implements the `default` case of the `match`.
Consequently, any `match` on that type will be required to have a `default`
case. This is valuable because it protects the type author's ability to add new
alternatives in the future, without causing build failures in client code.

> **Open question:** Can `Match` actually invoke `operator default`? It might
> sometimes be useful to define a type that can't be matched exhaustively, and
> the mere possibility that the `default` case could actually run might help
> encourage client code to implement it robustly, rather than blindly providing
> something like `Assert(False)`. However, if there's a language-level guarantee
> that the `default` case is unreachable if all other alternatives are handled,
> then we can allow code in the same library as the type to omit the `default`
> case. That's desirable because code in the same library doesn't pose an
> evolutionary risk, and it's often valuable to have a build-time guarantee that
> code within the type's own API explicitly handles every case.

Note that `Match`'s continuation parameter type must be generic rather than
templated (`:$` rather than `:$$`). Template specialization is driven by the
concrete values of the template arguments, but the type of the continuation
parameter may depend on code generation details that aren't yet known when
template specialization takes place. By the same token, we won't support
overloading `Match`, because overload resolution is likewise driven by the
concrete types of the function arguments, which may not be known at that point.

## "Bare" designator syntax

A _designator_ is a token consisting of `.` followed by an identifier. The
canonical use case for designators is member access, as in `foo.bar`, where the
designator applies to the preceding expression. However, we expect Carbon will
also have some use cases for "bare" designators, where there is no preceding
expression. In particular, we expect to use bare designators to initialize named
tuple fields, as in `(.my_int = 42, .my_str = "Foo")`, which produces a tuple
with fields named `my_int` and `my_str`.

I propose to also permit using bare designators to refer to the alternatives of
a sum type, in cases where the sum type is clear from context. In particular, I
propose that in statements of the form `return R.Alt;` or
`return R.Alt(<args>);`, where `R` is the function return type, the `R` can be
omitted. Similarly, I propose that patterns of the form `S.Alt` or
`S.Alt(<subpatterns>)`, where `S` is the type being matched against, the `S` can
be omitted. Note that both of these shorthands are allowed only at top level,
not as subexpressions or subpatterns.

> **Open question:** Can we also permit these shorthands to be nested? This
> would be more consistent and less surprising, but could create ambiguity with
> the use of bare designators in tuple initialization: does `case (.Foo, .Bar)`
> match a tuple of two fields named `Foo` and `Bar`, or does it match a tuple of
> two positional fields, of sum types that respectively have `.Foo` and `.Bar`
> as alternatives? Furthermore, allowing such nested usages may conflict with
> the
> [proposed principle](https://github.com/carbon-language/carbon-lang/pull/103)
> that type information should propagate only from an expression to its
> enclosing context, and not vice-versa.
>
> The issue of type propagation is particularly acute if we want to allow
> nesting of alternatives, such as `case .Foo(.Bar(_))`. The problem there is
> that we are relying on the type of `Foo`'s parameter to tell us the type of
> the argument expression `.Bar(_)`, but if `Foo` is overloaded, we can't
> determine the type of the parameter until the overload is resolved, and to do
> that we first need to know the type of the argument expression. And even if
> `Foo` is not currently overloaded, an overload might be added in the future
> (at least if the sum type has an `operator default`). Adding overloads is a
> canonical example of the kind of software evolution that Carbon is intended to
> allow, so the validity of this code can't depend on whether or not `Foo` is
> overloaded.

### Alternatives considered

#### Placeholder keywords

Rather than allowing code to omit the type altogether, we could allow code to
replace the type with a placeholder keyword, for example `case auto.Foo`. This
would avoid the ambiguity with tuple initialization, but would still mean that
type information is propagating into the expression from its surrounding context
rather than vice-versa (which also means that `auto` may not be an appropriate
spelling). Furthermore, this could wind up feeling fairly boilerplate-heavy,
even if the keyword is very short.

#### Designator types

We could treat each bare designator as essentially defining its own type, which
may then implicitly convert to a suitable sum type. For example, we could think
of `.Some(42)` as having a type `DesignatedTuple("Some", (Int))`, and
`Optional(Int)` would define an implicit conversion from that type. This would
ensure that type information does not propagate into the expression from the
context, but would not resolve the ambiguity with tuple initialization.
Furthermore, this would mean that the names of bare designators do not need to
be declared before they are used, and in fact can't be meaningfully declared at
all.

This could have very surprising consequences. For example, a typo in a line of
code like `var auto: x = .Sone(42);` cannot be diagnosed at that line. Instead,
the problem can't be diagnosed until `x` is used, and even then it will show up
as an overload resolution error rather than a name lookup error. This is
particularly problematic because it is much harder to provide useful, actionable
diagnostics for overload resolution failure than for name lookup failure.
Relatedly, in the case of bare designators, IDEs would not be able to implement
tab-completion using Carbon's name lookup rules, but would effectively have to
invent their own ad hoc name lookup rules.

Note that this approach would work well with the "Indexing by type" alternative
discussed below, with instances of `DesignatedTuple` (or whatever we call it)
acting as tagged wrapper types.

## Distinguishing pattern and expression semantics

As discussed above, we expect a well-behaved sum type to define factory
functions that correspond to each of the alternatives in its continuation
interface. This creates some potential ambiguity about whether a given use of an
alternative name refers to the factory function or the continuation interface.
For example, consider the following code:

```
var Result(Int, String): r = ...;
match (r) {
  case Result(Int, String).Value(0) => ...
```

This could potentially be interpreted two ways:

-   `Result(Int, String).Value(0)` is evaluated as an ordinary function call,
    and the result is compared with `r` to see if they match, presumably using
    the `==` operator.
-   The whole `match` expression is evaluated by invoking
    `Result(Int, String)`'s implementation of `Matchable`, with a continuation
    object whose `.Value(Int)` method compares the `Int` parameter with 0.

This can also be thought of as a name lookup problem: is `.Value` looked up in
`Result(Int, String)`, or in `Result(Int, String)`'s implementation of
`Matchable`?

I propose to leave this choice unspecified, so that the compiler may validly
generate code either way. This gives the compiler more freedom to optimize, and
perhaps more importantly, it helps discourage sum type authors from
intentionally making its factory function behavior inconsistent with its pattern
matching behavior. By extension, I propose treating the shorthand syntax
`case .Value(0)` the same way.

### Alternatives considered

#### Separate syntaxes

We could instead avoid the ambiguity by providing separate syntaxes for the two
semantics. The more straightforward version of this would be to say that
`Result(Int, String).Value(0)` is always interpreted as an ordinary function
call, and patterns like `Result(Int, String).Value(Int: i)` are ill-formed
because they cannot be interpreted as function calls. However, this would
require us to have a syntax for matching alternatives that is disjoint from the
syntax for constructing alternatives. This would be at odds with existing
practice in languages like Rust, Swift, Haskell, and ML, all of which use the
same syntax for constructing and matching alternatives.

Note that it is tempting to use bare designators as the syntax for matching
alternatives, so that `Result(Int, String).Value(0)` is an expression, but
`.Value(0)` is a pattern. However, that is unlikely to be sufficient on its own,
because bare designators rely on type information that may not always be
available, especially in nested patterns. Hence, in order for this approach to
work, we would need to introduce a separate syntax for specifying the type of a
bare designator, such as `.Value(0) as Result(Int, String)`.

Alternatively, we could say that code in a pattern-matching context is always
interpreted as a pattern, even if it could otherwise be interpreted as an
expression. We would then need to introduce a pattern operator for explicitly
evaluating a subpattern as an expression, such as
`is Result(Int, String).Value(0)` or `== Result(Int, String).Value(0)`. However,
this would impose an educational and cognitive burden on users: FAQ entries like
"What's the difference between `case .Foo` and `case is .Foo`" and "How do I
choose between `case .Foo` and `case is .Foo`" seem inevitable, and would
require fairly nuanced answers. It would also add syntactic noise to the use
cases that correspond to C++ `switch` statements, where all of the cases are
fixed values.

#### Disambiguation by fixed priority

We could instead specify that, when `Result(Int, String).Value(0)` appears in a
context where a pattern is expected, the name `.Value` is looked up as both an
ordinary function call and as a use of the `Matchable` interface, and specify
one of the two as the "winner" in the case where both lookups succeed. This is
effectively a variant of the previous option, except that some usages that would
be build errors under that approach would be saved by the fallback
interpretation here. In particular, it would still require us to introduce a
second syntax for the case where the programmer wants the lower-priority of the
two behaviors. Thus, it would carry largely the same drawbacks as the previous
option, with the additional drawback that there wouldn't be a consistent
correspondence between syntax and semantics.

#### Disambiguation based on pattern content

We could instead specify that `Result(Int, String).Value(0)` is always
interpreted as an ordinary function call, but patterns like
`Result(Int, String).Value(Int: i)` are evaluated using the `Matchable`
interface, because no other implementation is possible. However, this would mean
that the name-lookup behavior of a function call depends on code that can be
arbitrarily deeply nested within it, which seems likely to be hostile to both
programmers and tools.

## Choice types

To allow users to define sum types without micromanaging the implementation
details, I propose introducing `choice` as a convenient syntax for defining a
sum type by specifying only the declarations of the set of alternatives. From
that information, the compiler generates an appropriate object representation,
and synthesizes definitions for the alternatives and special member functions.

Our manual implementation of `Result` above doesn't really benefit from having
direct control of the object representation, and doesn't seem to have any
additional API surfaces, so it's well-suited to being defined as a choice type
instead:

```
choice Result(Type:$$ T, Type:$$ Error) {
  Success(T: value),
  Failure(Error: error),
  Cancelled
}
```

The body of a `choice` type definition consists of a comma-separated list of
alternatives. These have the same syntax as a function declaration, but with
`fn` and the return type omitted. If there are no arguments, the parentheses may
also be omitted, as with `Cancelled`. `default` may also be included in the
list, with the same meaning as `operator default` in a continuation interface:
it means that pattern-matching operations on this type must be prepared to
handle alternatives other than those explicitly listed.

The choice type will have a static factory function corresponding to each of the
alternatives with parentheses, and a static data member corresponding to each
alternative with no parentheses. The choice type will also implement the
`Matchable` interface, and its continuation interface will have a method
corresponding to each alternative.

In short, this definition of `Result` as a choice type will have the same
semantics as the earlier definition of it as a struct. It will probably also
have the same implementation, with a discriminator field and a storage buffer
large enough to hold the argument values of the alternatives. Any alternative
parameter types that are incomplete (or have unknown size for any other reason)
will be represented using owning pointers; among other things, this will allow
users to define recursive choice types. The implementation will be hidden, of
course, and the compiler may be able to generate better code, but we will design
this feature to support at least that baseline implementation strategy.

One consequence is that although the alternatives of a choice type can be
overloaded (as in the `Variant` example below), they cannot be templates. More
precisely, the parameter types of a pattern function must be fixed without
knowing the values of any of the arguments. To see why, consider a choice type
like the following, which attempts to emulate `std::any`:

```
choice Any {
  Value[Type:$$ T](T: value)
}
```

The problem is that since `T` could be any type, and a single `Any` object could
hold values of different types throughout its lifetime, `Any` can't be
implemented using a storage buffer within the `Any` object. Instead, the storage
buffer for the `T` object would have to be allocated on the heap, but then the
compiler would need to decide whether to apply a small buffer optimization, and
if so what size threshold to use, etc. Allowing choice types to be implemented
in terms of heap allocation would make their performance far less predictable,
contrary to Carbon's performance goals, and would have little offsetting
benefit: these sorts of types appear to be rare, and when needed they should be
implemented in library code, where the performance tradeoffs are explicit and
under programmer control.

It may be possible to relax this restriction when and if we have a design for
supporting non-fixed-size types, although it's worth noting that even that would
not give us a way for `Any` to support assignment.

Carbon will probably have some mechanism for allowing a struct to have
compiler-generated default implementations of operations such as copy, move,
assignment, hashing, and equality comparison, so long as the struct's members
support those operations. Assuming that mechanism exists, choice types will
support it as well, with the parameter types of the pattern functions taking the
place of the member types. However, there are a couple of special cases:

-   choice types cannot be default constructible, unless we provide a separate
    mechanism for specifying which alternative is the default.
-   choice types can be assignable, regardless of whether the parameter types
    are assignable, because assigning to a choice type always destroys the
    existing alternative, rather than assigning to it.

A future proposal for this mechanism will need to consider whether to require an
explicit opt-in to generate these operations.

**Open question:** Should `choice` provide a way to directly access the
discriminator? Correspondingly, should it provide a way to specify the
discriminator type, and which discriminator values correspond to which
alternatives? These features would enable choice types to support all the same
use cases as C++ `enum`s, and permit zero-overhead conversion between the two at
language boundaries.

### Alternatives considered

#### Different spelling for `choice`

The Rust and Swift counterparts of `choice` are spelled `enum`. I have avoided
this because these types are not really "enumerated types" in the sense of all
values being explicitly enumerated in the code. I chose the spelling `choice`
because "choice type" is one of the only available synonyms for "sum type" that
doesn't have any potentially-misleading associations.

## Alternatives considered

### `choice` types only

Rather than layering `choice` types on top of lower level features, we could
make them a primitive language feature, and simply not provide a way for user
code to customize the representation of sum types. However, this would mean that
users who encounter performance problems with the compiler-generated code for a
`choice` type would have no way to address those problems without rewriting all
code that uses that type. This would be contrary to Carbon's performance and
evolvability goals. Furthermore, the rewritten code would probably be
substantially less readable, and less safe, because it wouldn't be able to use
pattern matching.

### Indexing by type

Rather than requiring each alternative to have a distinct name (or at least a
distinct function signature), we could pursue a design that requires each
alternative to have a distinct type. With this approach, which I'll call
"type-indexed" as opposed to "name-indexed", Carbon sum types would much more
closely resemble C++'s `std::variant`, rather than Swift and Rust's `enum` or
the sum types of various functional programming languages.

Either approach can be emulated in terms of the other. For example, we don't yet
have enough of a design for variadics to give an example of a Carbon counterpart
for `std::variant<Ts...>`, but a variant with exactly three alternative types
could be written like so:

```
choice Variant(Type:$$ T1, Type:$$ T2, Type:$$ T3) {
  Value(T1: value),
  Value(T2: value),
  Value(T3: value)
}
```

Conversely a type-indexed type like `std::variant` can model a name-indexed type
like `Result(T,E)` by introducing a wrapper type for each name, leading to
something like `std::variant<Value<T>, Error<E>, Cancelled>` (note that
`std::variant<T,E>` would not work, because `T` and `E` can be the same type).
In either case, emulating the other model introduces some syntactic overhead:
with name-indexing, `Variant`'s factory functions must be given a name (`Value`)
even though it doesn't really convey any information, and emulating
`Result(T,E)` in terms of type-indexing requires separately defining the tagged
wrapper templates `Value` and `Error`.

The distinction between these two models of sum types seems analogous the
distinction between the tuple and struct models of product types. Tuples and
type-indexed sum types treat the data structurally, in terms of types and
positional indices, but structs and name-indexed sum types require the
components of the data to have names, which contributes to both readability and
type-safety by attaching higher-level semantics to the data.

It is possible that both models of sum types could coexist in Carbon, just as
structs and tuples do. However, that seems unlikely to be a good idea: the
coexistence of tuples and structs is necessitated by the fact that it is quite
difficult to emulate either of them in terms of the other in a type-safe way,
but as we've seen, it's fairly straightforward to emulate either model of sum
types in terms of the other.

Use cases that work best with type-indexing appear to be quite rare, just as use
cases for tuples appear to be quite rare compared to use cases for structs.
Consequently, if Carbon has only one form of sum types, it should probably be
the name-indexed form, as proposed here.

> **Open question:** Should Carbon have a native syntax for pattern matching on
> the dynamic type of an object? If so, should types like `Variant` be able to
> use it, instead of having the `.Value` boilerplate in every pattern? Should
> this mechanism be aware of subtype relationships (so that a subtype pattern is
> a better match than a supertype pattern)? If so, how are those subtype
> relationships defined?

### Pattern matching proxies

As a variant of the previous approach, we could allow types to specify their
pattern-matching behavior in terms of a proxy type that Carbon "natively" knows
how to pattern-match against. In the case of a sum type, this proxy would be a
`choice` type, which means that `choice` needs to be a fundamental part of the
language, rather than syntactic sugar for a sum type struct. Returning once more
to the `Result` example:

```
struct Result(Type:$$ T, Type:$$ Error) {
  // Data members, factories, and special members same as above

  choice Choice {
    Success(T: value),
    Failure(Error: error),
    Cancelled
  }

  fn operator match(Ptr(Self): this) -> Choice {
    match (discriminator) {
      case 0 => {
        return .Success(this->storage.Read(T));
      }
      case 1 => {
        return .Failure(this->storage.Read(Error));
      }
      case 2 => {
        return .Cancelled;
      }
      default => {
        Assert(False);
      }
    }
  }
}
```

This approach has several advantages:

-   It's somewhat simpler, because it uses return values instead of
    continuation-passing.
-   It will be easier for the compiler to reason about, because of that
    simplicity and the somewhat narrower API surface. This may lead to better
    compiler performance, and better generated code.
-   It could generalize more easily to allow things like user-defined types that
    can match list patterns (if Carbon has those).

However, it also has several drawbacks:

-   It forces us to treat `choice` as a fundamental part of the language: in
    order to implement a sum type, you have to work with an object type whose
    layout and implementation is inherently opaque. This would be a substantial
    departure from C++, and it's difficult to foresee the consequences of that.
    Possibly the closest analogy in C++ is virtual calls, and especially virtual
    base classes, where fundamental operations like `->` and pointer casts can
    involve nontrivial generated code, and some aspects of object layout are
    required to be hidden from the user. However, Carbon seems to be moving away
    from C++ in precisely those ways.
-   Although both approaches carry a risk of confusion due to duplicate names,
    the risk is somewhat greater here: a naive reader might think that, for
    example, `.Success` in the body of `operator match` refers to `Self.Success`
    rather than `Self.Choice.Success`. Relatedly, there's some risk that the
    author may omit the leading `.`, and thereby invoke `Self.Success` instead
    of `Self.Choice.Success`. This will probably fail to build, but the errors
    may be confusing.
-   It has less expressive power, because there's no straightforward way for the
    library type to specify code that should run after pattern matching is
    complete. For example, the callback approach could allow `Match` to create a
    mutable local variable, pass it to the callback by pointer/reference, and
    then write the (possibly modified) contents of the variable back into the
    sum type object.
-   Extending this approach to support in-place mutation during pattern matching
    is likely to require Carbon to support reference _types_, whereas the
    primary proposal would probably only require reference _patterns_, which are
    substantially less problematic. This is a consequence of using a return type
    rather than a set of callback parameters (which are patterns) to define the
    type's pattern matching interface.

I very tentatively recommend the callback approach rather than this one,
primarily because of the last point above: the Carbon type system is likely to
be dramatically simplified if there are no reference types, but I think the
proxy approach will make reference types all but unavoidable.

### Pattern functions

Rather than require user types to define both the pattern-matching logic and the
factory functions, with the expectation that they will be inverses of each
other, we could instead enable them to define the set of alternatives as factory
functions that the compiler can invert automatically. This approach, like the
primary proposal, would consist of several parts.

A _pattern function_ is a function that can be invoked as part of a pattern,
even with arguments that contain placeholders. Pattern functions use the
introducer `pattern` instead of `fn`, and can only contain the sort of code that
could appear directly in a pattern. This lets us define reusable pattern
syntaxes that can do things like encapsulate hidden implementation details of
the object they're matching.

Next, we would introduce the concept of an `alternatives` block, which groups
together a set of factory functions and designates them as a set of
alternatives. They are required to be both exhaustive and unambiguous, meaning
that for any possible value of the type, it must be possible to obtain it as the
return value of exactly one of the alternatives. Alternatives that take no
arguments, which represent singleton states such as `Cancelled`, can instead be
written as static constants. An `alternatives` block can be marked `closed`,
which plays the same role as omitting `operator default` in the primary
proposal: it indicates that client code need not be prepared to accept
alternatives other than the ones currently present.

As with the primary proposal, `Storage` is used to represent a span of memory
that the user can create objects within. However, with this approach we also
need it to support initialization from a `ToStorage` factory function, because
pattern functions can't contain procedural code. Note that for the same reason,
`ToStorage` will probably need to be a language intrinsic, or implemented in
terms of one.

Using these features, the `Result(T, Error)` example can be written as follows:

```
struct Result(Type:$$ T, Type:$$ Error) {
  var Int: discriminator;

  var Storage(Max(Sizeof(T), Sizeof(Error)),
              Max(Alignof(T), Alignof(Error))): storage;

  closed alternatives {
    pattern Success(T: value) -> Self {
      return (.discriminator = 0, .storage = ToStorage(value));
    }

    pattern Failure(Error: error) -> Self {
      return (.discriminator = 1, .storage = ToStorage(error));
    }

    var Self:$$ Cancelled = (.discriminator = 2);
  }
}
```

Notice that with this approach, we do not need to define the special member
functions of `Result` manually. The compiler can infer appropriate definitions
in the same way that it infers how to invert these functions during pattern
matching.

As with the primary proposal, `choice` would be available as syntactic sugar,
but its syntax would mirror the syntax of an `alternatives` block:

```
closed choice Result(Type:$$ T, Type:$$ Error) {
  pattern Success(T: value) -> Self;
  pattern Failure(Error: error) -> Self;
  var Self:$$ Cancelled;
}
```

This ensures that a sum type's API is defined using essentially the same syntax,
regardless of how the type author chooses to implement it.

This approach is described in much more detail in an
[earlier draft](https://github.com/carbon-language/carbon-lang/blob/4dbd31d71e02895892f97a211df4b5fff8cae5c3/proposals/p0157.md)
of this document, where it was the primary proposal. It has a number of
advantages over the primary proposal:

-   Manually-defined sum types could probably be implemented with dramatically
    less code, because both the special member functions and the code that
    implements pattern matching can be generated automatically. This would not
    only make these types less tedious to implement, it would probably also
    reduce the risk of bugs, and avoid readability problems arising from the
    duplication of names between factory functions and continuation interface
    methods.
-   Programmers would be able to define functions that encapsulate complex
    pattern-matching logic behind simple interfaces.
-   Pattern matching would not require the special overload resolution rules
    that are needed to translate a pattern-matching operation into a `Match`
    call.
-   User-defined sum types would default to being open, whereas they default to
    being closed under the primary proposal. This is probably a better default,
    because an open sum type can always be closed, but a closed sum type can't
    be opened (much less extended with new alternatives) without the risk of
    breaking user code.

However, this approach also carries some substantial drawbacks:

-   Types can't use the full power of the Carbon language when defining their
    pattern-matching behavior, or the corresponding factory functions. Instead,
    they are restricted to a very narrow subset of the language that is valid in
    both patterns and procedural code. This forces us to introduce intrinsics
    like `ToStorage` to make that subset even minimally usable, and creates a
    substantial risk that some user-defined sum types just won't be able to
    support pattern matching.
-   The language rules are substantially more complicated and harder to explain,
    because we need to define the language subset that is usable in pattern
    functions, and define both "forward" and "reverse" semantics for it.
    Relatedly, it means that during pattern matching, there will be no
    straightforward correspondence between the Carbon code and the generated
    assembly, which could substantially complicate things like debugging.
-   Carbon's pattern language can't allow you to write _underconstrained_
    patterns, which are patterns where the values of all bindings aren't
    sufficient to uniquely determine the state of the object being matched. This
    rules out things like the `|` pattern operator, and even prevents us from
    using pattern matching with types that have non-salient state, like the
    capacity of a `std::vector`.

These issues, especially the overall complexity of this approach, leads me to
recommend against adopting it.

## Rationale

This proposal gives us solid ground on which to continue developing features
that rely on sum types and pattern matching thereof. The approach taken seems
plausible, and while there's a good chance that we will revise it significantly
before we finish our first pass over the complete Carbon design (for example, by
switching to pattern matching proxies or by supporting mutation of matched
elements), we don't think it will anchor us too much on this one particular
direction.

With regard to Carbon's goals:

-   Performance: while the approach taken herein potentially has some
    performance cost for a common operation that is likely to appear in
    performance-critical code (requiring an indirect call and the generation of
    continuations for user-defined pattern matching), such cost should be
    practically removable by inlining. We'll need to take care to ensure this
    abstraction penalty is reliably removed in common cases, but this seems
    sufficiently feasible to be worth attempting.
-   Evolution: software evolution is supported by allowing user-defined pattern
    matching to specify (by way of the presence/absence of operator default)
    whether the set of patterns is intended to be extended in the future.
-   Ergonomics: custom pattern matching for user-defined types promotes language
    consistency and removes boilerplate

Note: At the decision meeting, it was stated that geoffromer will update the
proposal to add a rationale to address austern's questions about the layered
approach.
