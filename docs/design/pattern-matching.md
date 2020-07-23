# Carbon pattern matching

<!--
Part of the Carbon Language, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

## Table of contents

<!-- toc -->

- [Goals](#goals)
- [Uses](#uses)
  - [Pattern match control flow](#pattern-match-control-flow)
  - [Pattern matching in local variables](#pattern-matching-in-local-variables)
  - [Pattern matching as function overload resolution](#pattern-matching-as-function-overload-resolution)
  - [Differences](#differences)
- [Features](#features)
  - [Positional & keyword arguments](#positional--keyword-arguments)
  - [Defaults / optional arguments](#defaults--optional-arguments)
  - [Deduced arguments](#deduced-arguments)
  - [Variadics (...)](#variadics-)
  - [Conditions](#conditions)
- [Specification](#specification)
  - [Pattern syntax](#pattern-syntax)
  - [Value specification match rules](#value-specification-match-rules)
  - [Deduced specification match rules](#deduced-specification-match-rules)
  - [Destructuring rules](#destructuring-rules)
  - [Condition](#condition)
- [Broken links footnote](#broken-links-footnote)

<!-- tocstop -->

## Goals

Pattern matching is a general Carbon mechanism for handling structured values
like
[tuples (TODO)](https://github.com/josh11b/carbon-lang/blob/tuples/docs/design/tuples.md).
Given a value and a pattern, there are two steps:

1. Matching: Does this value match the pattern?
2. Destructuring: If it does match, bind names to the components of the value.

Desirable properties:

- Consistent: One mechanism with one set of rules. As much as possible it should
  use the same syntax and semantics, unifying multiple parts of the Carbon
  language. Less to learn.
- Predictable: Users can easily determine the resulting behavior.
- Expressive: Straightforward to express the range of expected use cases.

## Uses

We have a few places where we would like to use pattern matching in Carbon:

- Carbon's `match` statement, replacing C++'s `switch`.
- May use a pattern on the left side of a variable initialization, to
  destructure a tuple value on the right.
- To define function arguments, and support function overloading.

### Pattern match control flow

The most powerful form and easiest to explain form of pattern matching is a
dedicated control flow construct that subsumes the `switch` of C and C++ into
something much more powerful, `match`. This is not a novel construct, and is
widely used in existing languages (Swift and Rust among others) and is currently
under active investigation for C++. Carbon's `match` can be used as follows:

```
fn SimpleSwitch(Int: i) -> String {
  match (i) {
    case (0) => {
      return "Zero";
    }
    case (1) => {
      return "One";
    }
    default => {
      return "Negative or more than one";
    }
  }
  return "Unreachable";
}

fn MultipleArgs(Int: j, Bool: b) -> String {
  match (j, b) {
    case (0, True) => {
      return "j == 0, b == True";
    }
    // `Bool: _` means match any value with type `Bool`.
    // The `_` means throw the value away.
    case (1, Bool: _) => {
      return "j == 1, b anything";
    }
    // `auto: k` means match any value bind it to the name `k`.
    // We can then add conditions on the bound value or use `k`
    // in the body of the case.
    case (auto: k, True) if (k >= 0) => {
      // j == 0, b == True already handled above.
      return "j > 0, b == True";
    }
    // This is equivalent to "default", since `...` matches
    // multiple arguments, as a tuple.
    case (... auto: _) => {

      return "Nothing else matched";
    }
  }
  return "Unreachable";
}

fn Bar() -> (Int, (Float, Float));
fn UnpackTuple() -> Float {
  // This uses tuple unpacking in the match expression,
  // so the cases don't need extra parens.
  match (Bar()...) {
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

Note: we may want to say `where` instead of `if`, I've seen both suggested in
various docs.

There is a lot going on here. First, let's break down the core structure of a
`match` statement. It accepts a value or list of values that will be inspected.
It then will find the _first_ `case` that matches this
value, and execute that block. If none match, then it executes the default
block.

Each `case` contains a pattern. The first part is a value pattern
(`(Int: p, auto: _)` for example) followed by an optional boolean predicate
introduced by the `if` keyword. The value pattern has to match, and then the
predicate has to evaluate to true for the overall pattern to match. Value
patterns can be composed of the following:

- An expression (`42` for example), whose value must be equal to match.
- A type (`Int` for example), followed by a `:` and either an
  identifier to bind to the value or the special identifier `_` to discard the
  value once matched.
- Three dots (`...`) followed by a type, `:`, and the identifier (or `_`) to
  match multiple arguments, as a tuple.
- A destructuring pattern containing a sequence of value patterns
  (`(Float: x, Float: y)`) which match against tuples and tuple like values by
  recursively matching on their elements.
- An unwrapping pattern containing a nested value pattern which matches against
  a variant or variant-like value by unwrapping it.

In order to match a value, whatever is specified in the pattern must match.
Using `auto` for a type will always match, making `auto: _` the wildcard
pattern matching a single value.

**Open question:** How do we effectively fit a "slice" or "array" pattern into
this (or whether we shouldn't do so)?

**Context**: Similar pattern matching control flow constructs in other
languages:

- [Rust](https://doc.rust-lang.org/book/ch18-03-pattern-syntax.html),
- [Circle](https://github.com/seanbaxter/circle/blob/master/pattern/pattern.md),
- ... TODO

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
a local variable named `p` which is then returned. The `(Float, Float)` returned
by `Bar()` matches and is discarded by `auto: _`.

#### Constants

The `:$` and `:$$` forms can be used to defined constants.

```
fn Constants(Int:$ N, Int: m) -> Int {
  var Int:$$ Two = 2;
  var Int:$ TwoN = Two * N;
  return TwoN + m;
}
```

The `:$$` form defines a constant whose value is available during typechecking,
which would normally be preferred unless you are setting it to an expression
that depends on a generic input value (`N` in the example above) and so is not
known when typechecking.

### Pattern matching as function overload resolution

The argument list to a function in Carbon is a pattern. Functions may be
overloaded by defining functions with the same name but different patterns.
Example:

```
fn ToString(Int: x) -> String { return "A" }
fn ToString(Bool: x) -> String { return "B" }
fn ToString[Serializable:$ T](T: x) -> String { return "C" }

var String: result = (
    "3: " + ToString(3) + ", " +
    "False: " + ToString(False) + ", " +
    "m_s_o: " + ToString(my_serializable_object));
assert(result == "3: A, False: B, m_s_o: C")
```

Any given call to `ToString` will take its argument list and try matching it to
the argument pattern defined for each overload. Assuming there is a single
match, that defines which function implementation is actually called.

**Proposal:** The order of patterns / overloads should not matter.

In particular, Carbon won't use "which was declared first" to resolve
ambiguities about which pattern / overload to use. This avoids questions about
what happens when the overload definitions are split across multiple files, or a
particular overload has a separate forward declaration and implementation.

**Problem:** Can the set of overloads associated with a name be different in
different files or different positions within the same file?

It would be nice if the answer was no. In particular it would be useful to make
things like type functions deterministic (avoiding a source of
[ODR](https://en.wikipedia.org/wiki/One_Definition_Rule) problems), and
refactorings that move code around safer. However, that is contrary to the goal
that code later in a file does not affect the interpretation of code earlier in
the file.

We should consider if this is too strict. A weaker property that we could aim
for instead is: the overload selected only depends on the arguments, not the
location of the call site. In particular if we define the overload of `f` taking
a `Foo` together with the definition of `Foo`, then any call site that can have
a value of type `Foo` to pass to `f` would also see the overload of `f` taking
`Foo`. This leads to this possibility...

**Alternative considered:** All overloads for a given function name should be in
the library defining the name or in a library defining a type used in the
argument pattern.

The idea is that you could overload a function in another namespace, but you
would only ever allow a call to a function if you could see both the library
defining the function's name and those defining all argument types.

Some languages (Swift, C++) are much looser than this.

**Concern:** This could create ambiguity in the presence of generics and
interfaces. Consider this situation:

- Library `A` defines a function `A.F` taking an `Int`.
- Library `B` defines a type `T`.
- Library `C` defines an interface `Interface1` and an implementation of
  `Interface1` for `B.T`. This is legal since implementations of interface can
  either be defined with the type or the interface (to address dependency issues
  and the expression problem). Library `C` also defines an overload for `A.F`
  for types implementing `Interface1`.
- Library `D` defines an interface `Interface2`, an implementation of
  `Interface2` for `B.T`, and an overload for `A.F` for type implementing
  `Interface2`. Same deal as with library `C`.

Code importing libraries `A` and `B` will find calling `A.F` with a value of
type `T` generates a compiler error. Code importing `A`, `B`, and `C` will see
the call as legal, but will see different behavior than code importing `A`, `B`,
and `D`. And code importing all 4 libraries will see the call as ambiguous. This
does not seem like a desirable state of affairs and not in keeping with the
desire to keep the behavior of a call to a function consistent independent of
other context.

**Proposal:** All overloads of a function have to be defined in the library
where the name is defined.

The use cases that need implementations defined in multiple libraries are better
served using interfaces and generics, which allow lookup based on the type being
operated on. This position is espoused in the
["Carbon closed function overloading" proposal (TODO)](#broken-links-footnote)<!-- T:Carbon closed function overloading proposal -->.

**Problem:** Is there a rule for selecting an overload if there are multiple
patterns that match?

Seems like a conservative choice would be to just give a compile error in this
situation, but C++ found it necessary to define an ordering of matches so that
the "most specific" match would win. This would make varying the overload set
depending on context even more dangerous, and another reason to prefer closed
overloading for functions.

**Proposal:** If multiple overloads match, one must be strictly more specific
from all the others.

This matches what we want to happen with implementations of interfaces. There we
want to provide a default implementation via a broad match (say, using a
templated implementation), while still allowing overrides for specific types.

**Proposal:** Overloads must be resolved at compile time.

This may mean that some arguments may have to be known at compile time at the
caller. For example:

```
fn Foo(True);
fn Foo(False);
fn Foo(Int: x);

// Okay
Foo(True);
Foo(3);

fn Bar() -> Bool;
if (Bar()) { Foo(True); } else { Foo(False); }

// Illegal! Compile error:
Foo(Bar());
```

**Note:** A consequence of this is the value of `ToString` above doesn't have a
single signature and so is not an ordinary function. This means it can't for
example be saved in a runtime variable (with a function type). Instead
`ToString` is an "overloaded function set", which is a value that is only
meaningful at compile time. [Same is true of any generic/templated function!]

**Question:** How would you get a function representing one overload from an
"overloaded function set"?

- Maybe the members of the set each have a name, like `ToString(Int)` or
  `ToString(MySerializableType)`?
- Maybe you have to use a lambda? Like:

```
  var auto: f = fn(Int: x) -> String { return ToString(x); };
  var auto: g = fn(MySerializableType: x) -> String { return ToString(x); };
```

We would like a good story here so that adding an overload to an existing
function name doesn't break existing code.

**Note:** For functions argument lists, we also support template / generic
arguments, but that is detailed in
[another document (TODO)](https://github.com/josh11b/carbon-lang/blob/generics-docs/docs/designs/generics-overview.md).

### Differences

There are a few differences between these uses:

- The `match` statement has several patterns and determines which applies at run
  time. If multiple patterns match, it uses the first match.
- An assignment statement will have a single pattern, and it will be a compile
  error unless it can be shown to match at compile time.
- Function overloading allows multiple patterns, but at compile time any
  function call must be resolved to select a single pattern. If a call matches
  multiple overloads, there must be a most specific match that wins.

Both the `match` statement and function calls take a comma-separated list of
values inside parentheses. Similarly, the `case` expressions and functions
overloads have a comma-separated list of patterns inside parentheses.

## Features

### Positional & keyword arguments

**Rationale:** In general, positional arguments are great when either there is
an obvious order to provide the arguments (e.g. coordinates in x, y, z order) or
the order doesn't matter (e.g. `max()` is commutative). Otherwise, keyword
arguments can (a) prevent mistakes (mixing up source & destination) and (b)
provide documentation (what does this "True" argument mean?).

Context:
[Be wary of functions which take several parameters of the same type](https://news.ycombinator.com/item?id=21086666).

Context:
[Google C++ Style Guide](https://google.github.io/styleguide/cppguide.html#Structs_vs._Tuples)
says "Prefer to use a struct instead of a pair or a tuple whenever the elements
can have meaningful names."

Named components are also naturally a part of tuples so they can be used to
initialize struct variables. When a struct type is changed to use a factory
function / constructor for initialization, we should be able to use the same
initialization syntax. The tuple with named components used to initialize will
become the arguments to the factory function, which the factory function will
declare as keyword arguments in order to match. See
["Carbon struct types"](https://github.com/josh11b/carbon-lang/blob/structs/docs/design/structs.md).

**Proposal:** Furthermore, I believe we should use keywords in overload
resolution. Since the keywords are written in both the caller and the function
signature, there is no ambiguity about which function should be called
(considerably less than using types in overload resolution, which can be much
less visible in the source code). Furthermore, there are cases where using the
types of the argument to disambiguate is less clear or even ambiguous. For
example, let us say we see a `Array(Int)` being constructed with the arguments
`(3, 5)`. Is that creating a array containing three 5s or the two values `3, 5`?
Different names for the arguments would resolve the question of which
constructor should be used.

**Note:** Even when using keyword arguments, we are currently saying that the
order still needs to match between the call site and the function signature.
This is to simplify our order-of-evaluation story. It is expected that IDEs (and
other tooling) will make it easy to update the code whenever there is a
mismatch.

### Defaults / optional arguments

**Rationale:** By providing a facility for specifying defaults for arguments in
the function signature and then making those arguments optional at the call
site, we provide a number of benefits:

- Brevity for common cases where the defaults are what the user wants.
- Allow many combinations of options to be specified at the caller without a
  combinatorial explosion of function definitions.
- Allow evolution of an API. A new argument can be added to an existing function
  as long as it has a default that preserves the old behavior.

Note that this feature is more powerful when combined with keyword arguments. In
C++ which only uses positional arguments, you always have to specify a prefix of
the argument list at the call site, and so only the tail of the argument list in
the signature is allowed to have defaults. Within the keyword arguments, any may
have defaults, and the caller may skip specifying any optional argument and
still specify later arguments.

### Deduced arguments

**Rationale:** The primary use case for deduced arguments is for type inference,
but it could also be used to infer the size of an array argument. This is mainly
for brevity, so you don't need to specify redundant information at the call
site. It also provides continuity with C++ which has had this feature for a
while. TODO: add ways this is used to enhance expressivity.

Note that the rules in C++ for deduced type arguments end up complex due to
dealing with `const` and references (both `&` and `&&`).

```
// Templated function that works for any value with an integer `.x` member.
fn DeducesType[Type:$$ T](T: a) -> Int {
  // `T`, the type of `a`, is deduced to be the type of whatever value
  // is passed to this function.
  return a.x;
}

struct S {
  var Int: x;
};
var S: named_struct = (.x = 2);
// Here, `T` is deduced to be the named struct `S`.
Assert(DeduceType(named_struct) == 2);

fn DeducesSize[Int: size](FixedLenArray(Int, size): a) -> Int {
  return size;
}
var FixedLenArray(Int, 3): array = (1, 2, 3);
Assert(DeducesSize(array) == 3);
```

#### Deducing parameters of types

*Context:*
[Carbon chat Oct 31, 2019: Higher-kinded types, normative types, and type deduction (TODO)](#broken-links-footnote)<!-- T:Carbon chat Oct 31, 2019: Higher-kinded types, normative types, and type deduction --><!-- A:#heading=h.r48w6htktgjf -->

Since parameterized types have names that include their parameters, this means
that parameter may be deduced when calling a function. For example,

```
struct Vec(Type: T) { ... }

fn F[Type: T](Vec(T): v) { ... }
// More commonly, this would be:
// fn F[Type:$ T](Vec(T): v) { ... }

var Vec(Int): x;
F(x);  // `T` is deduced to be `Int`.
```

In addition, a parameterized type can actually be thought of as a function,
which can also be deduced:

```
// Continued from above
fn G[fn(Type)->Type: V](V(Int): v) { ... }
G(x);  // `V` is deduced to be `Vec`

fn H[Type: T, fn(Type)->Type: V](V(T): v) { ... }
H(x);  // `T` is deduced to be `Int` and `V` is deduced to be `Vec`.
```

This would be used in the same situations as
[C++'s template-template parameters](https://stackoverflow.com/questions/213761/what-are-some-uses-of-template-template-parameters).

**Proposal:** The above deductions are only available based on a type's name,
not arbitrary functions returning types.

If we write some other function that returns a type:

```
fn I(Type: T) -> Type {
  if (T != Bool) {
    return Vec(T);
  } else {
    return BitVec;
  }
}
```

In theory, since the function is injective, it might be possible to deduce its
input from a specific output value, as in:

```diff
  // Not allowed: `I` is an arbitrary function, can't be involved in deduction:
- fn J[Type: T](I(T): z) { ... }
  var I(Int): y;
  // We do not attempt to figure out that if `T` was `Int`, then `I(T)` is equal
  // to the type of `y`.
  J(y);
```

If we wanted to support this case, we would require the type function to satisfy
two conditions:

- It must be _injective_, so different inputs are guaranteed to produce
  different outputs. For example, `F(T) = Pair(T, T)` is injective but
  `F(T) = Int` is not.
- It must not have any conditional logic depending on the input. We could
  enforce this by requiring it to take arguments generically using the `:$`
  syntax.

If both those conditions are met, then in principle the compiler can
symbolically evaluate the function. The result should be a type expression we
could pattern match with the input type to determine the inputs to the function.
In general, this might be difficult so we need to determine if this feature is
important and possibly some other restrictions we may want to place. A hard
example would be deducing `N` in `F(T, N) = Array(T, N * N * N)` given
`Array(Int, 27)`. This makes me think this feature should be deprioritized until
there are compelling use cases which can guide what sort of restrictions would
make sense.

One use case we do want to support for [variadics (below)](#variadics-), is the
function `NTuple(N, value)` that returns a tuple with `N` components, all of
which are equal to `value`. If `value` is a type `T`, since the type of a tuple
is the tuple of the component types, then `NTuple(N, T)` is a type. For example,
the type of `(1, 2, 3)` is `(Int, Int, Int) == NTuple(3, Int)`. However, note
that `NTuple` isn't even injective when `N == 0`!

We are considering theee approaches for representing `NTuple`:
- It could be something built-in and not writable in user code. This is
  undesirable because there will likely be variations on `NTuple` that
  user's will need and won't be provided.
- We could provide a special "type function that supports deduction" facillity
  that explicitly provides two functions:
  - A forward function, in this case taking `N = 3` and `value = Int` to
    `(Int, Int, Int)`.
  - A deduction function, taking a resulting type (like `(Int, Int, Int)`)
    and optional values for the parameters and returns values for the
    parameters that were not specified. It would return an error if there
    are no parameters that would cause the forward function to produce that
    output (like `(Int, Bool)`). It would also return an error if there was
    no unique way of deducing an argument (you can't deduce `value` from the
    empty tuple `()`).
  A flaw with this approach is that even if we support deduction, there is
  no clear way to tell if a match using one of these is more specific than
  another for purposes of function overload resolution.
- We could provide a "tuple comprehension" syntax, that is simple enough for
  the compiler to analyze to be able to perform deductions and tell when one
  expression is more specific than another, but expressive enough to cover most
  use cases. For reference, the Python-like syntax for comprehensions would be
  something like `NTuple(N, value) = (value for _ in 0..N)`, but we might want
  to make changes to put uses of names after they are declared.

### Variadics (...)

Variadics are when a pattern can match any number of arguments.

**Rationale:** Variadics are important for expressiveness. There are a number of
use cases:

- Functions like `Max` or array constructors take a variable number of
  positional arguments, as long as they all are of the same type.
- Functions like `StrCat` take a variable number of positional arguments, and
  different arguments may have different types as long as they can all be
  converted to strings.
- Functions like `EncodeAsJSON`, like
  [this C++ library](https://github.com/nlohmann/json), would take a variable
  number of keyword arguments, with different types as long as they can be
  converted to JSON.
- Some functions forward to other functions, passing all the positional and
  keyword arguments after a certain point along.

**Proposal:** A variadic element of the pattern follows a "max munch" rule,
matching the maximum sequence of arguments not matched by other elements of the
pattern that may be assigned to the type of the variadic.

In these examples, we define variations on `Max` which take a variable number of
positional arguments, all of which must have the same type:

```
// `Array(Int)` can be constructed by a tuple containing only integers.
// This only matches positional arguments, not keyword arguments.
fn MaxV1(... Array(Int): args) -> Int;

// `Array(T)` can be constructed by a tuple containing elements of type `T`.
// This only matches positional arguments, not keyword arguments.
// `MaxV2` is instantiated once per distinct type `T`.
 fn MaxV2[Comparable:$ T](... Array(T): args) -> T;

// Positional arguments are matched by `args`, keyword argument `compare`
// is separate.
// `MaxV3` is instantiated once per distinct type `T`.
fn MaxV3[Type:$ T](
    ... Array(T): args,
    .compare = fn(T:_, T:_)->Int: compare) -> T;

// `NTuple(N, T)` is the tuple with `N` components all of which are `T`.
// This example is similar to `MaxV2`, except `MaxV4` instantiated per number of
// arguments, in addition to per distinct type. `N` must be at least 1, or there
// is no way to deduce `T`.
fn MaxV4[Int:$$ N, Comparable:$ T](... NTuple(N, T): args) -> T;

// In this case, `T` is deduced from the first argument, so `N` can be 0.
fn MaxV5[Int:$$ N, Comparable:$ T](T: first, ... NTuple(N, T): rest) -> T;
```

**Concern:**

> The intent of this code is to use the constructor for `Array(T)` that accepts
> an n-tuple of values of type `T`. But what if `Array(T)` has an alternate
> constructor that takes a count and a value to repeat using keyword arguments?
> Naively, this would allow you to call
> `MaxV1(.count = 3, .repeated_value = 7)`. This is both surprising and exposes
> an implementation detail that you wouldn't want callers to rely on. Seems like
> we need this to express our intent more clearly.

The `MaxV3` example demonstrates a case where the variadic should not consume
all remaining arguments. There are a few cues in this example:

- `Array(T)` can only match positional arguments of type `T`
- Clearly an argument using the `.compare` keyword belongs to the next
  parameter.

I think we likely want to support all three of these stopping criteria:

- If the type changes to something the variadic won't accept, it should stop
  consuming arguments. This would allow you to write something that takes a
  variable number of integers followed by a variable number of strings.
- If the arguments switch from positional to keyword and the variadic can only
  accept positional arguments, it should stop consuming arguments. This is used
  in the `ForwardAllV1` example below.
- The variadic should not consume the next argument if it uses a keyword
  matching a later parameter. Otherwise it is difficult for something that would
  otherwise consume all keyword arguments to know when to stop.

**Concern:**

> These rules would all become more complicated and less efficient to implement
> if we don't make keyword arguments ordered. On the other hand, there are
> forwarding use cases where you want to consume or add a keyword argument when
> forwarding that are very difficult to do unless keyword arguments can be in
> any order.

Variadic patterns may also be used with the `match` statement. For example, here
is an implementation of `MaxV4` from above:

```
fn MaxV4[Int:$$ N, Comparable:$ T](... NTuple(N, T): args) -> T {
  match (args...) {
    case (T: x) => return x;
    // Using the same variadic pattern matching syntax in a `match` statement.
    case (T: first, ... NTuple(N-1, T): rest) => {
      var T: max_of_rest = MaxV4(rest...);
      if (first < max_of_rest) {
        return max_of_rest;
      } else {
        return first;
      }
    }
  }
}
```

To support `StrCat`, we need to say "the type of `args` is a tuple of types that
are not necessarily all the same but all implement the `ToString` interface." If
that type is `TupleOfNTypes`, then the type of _that_ is a tuple with elements
that are all `ToString`:

```
fn StrCat[Int:$$ N, NTuple(N, ToString):$$ TupleOfNTypes]
    (... TupleOfNTypes: args) -> String;
```

For the forwarding use case, we can forward positional arguments using the same
approach as for `StrCat` above:

```
fn ForwardPositional[Int:$$ N, NTuple(N, Type):$$ TupleOfNTypes]
    (... TupleOfNTypes: args);
```

To forward keyword arguments, we need `NamedTuple(T)`, which is a type whose
values are tuples with any set of named arguments, whose values are all `T`.

```
fn ForwardKeyword[NamedTuple(Type):$$ NamedTupleofTypes]
    (... NamedTupleOfTypes: args);
```

This construct also allows us to implement `EncodeAsJSON`, assuming we implement
interface `JSONType` for any supported type:

```
fn EncodeAsJSON[NamedTuple(JSONType):$$ NamedTupleofJSONTypes]
    (... NamedTupleOfJSONTypes: args) -> String;

```

We could combine these two approaches to forward all arguments:

```
fn ForwardAllV1
    [Int:$$ N, NTuple(N, Type):$$ TupleOfNTypes,
     NamedTuple(Type):$$ NamedTupleofTypes]
    (... TupleOfNTypes: positional, ... NamedTupleOfTypes: keyword);
```

This is an interesting case, where the compiler will have to recognize that
`positional` can only match positional arguments, much like the `MaxV3` case
above, and it should leave the remaining arguments to match the second variadic
argument.

I suspect this will be common enough of a case that I think we should support it
directly with a `MixedTuple(T)` type whose values are any tuple with positional
and named members equal to `T`.

```
// `T` can be any tuple containing types, so both positional and keyword
// arguments are allowed.
fn ForwardAllV2[MixedTuple(Type):$$ TupleofTypes](... TupleOfTypes: args);
```

### Conditions

**Rationale:** Conditions are most helpful when used in a `match` statement. In
that context, we are expressing a condition and the condition part is needed for
completeness. In a `match` statement the conditions have an order in which they
are tested, so something like:

```
match (a) {
  case (P1) if (C1) => { ... }
  case (P2) => { ... }
  case (P1) => { ... }
}
```

would be hard to rewrite to avoid using conditions.

We still intend to support conditions in function overload resolution for
consistency, but the use cases are less clear. Note that in order to satisfy the
"overloads must be resolved at compile time" plan, these would have to be
conditions that could be evaluated at compile time, and so generally can't use
anything about the dynamic values being supplied as arguments. Restricting to
generic and template arguments of the function, they could be used as part of
the generic constraints story.

**Speculation:** For example they may be useful to express a type satisfies
multiple constraints. Or to express that different implementations are used for
forward and random-access iterators. They may also be needed to make overload
resolution less ambiguous in a case where otherwise two signatures would match.

**Proposal:** Conditions will not be used to affect deductions -- since
conditions are arbitrary expressions they will be evaluated as a black box
returning "true" or "false", not as information that can disambiguate tricky
deduction cases.

## Specification

### Pattern syntax

A pattern consists of three pieces, in sequence:

- Deduced specification
- Value specification
- Condition specification

Only the value specification is required.

**Deduced specification:** [ `[`&lt;type> `:` [ `$` | `$$` ] &lt;id>`,` ... `]`
]

**Value specification:** `(` &lt;value pattern>`,` ... `,` `...` &lt;value
pattern> , `.` &lt;id> `=` &lt;value pattern>`,` ... `)`

The initial &lt;value pattern>s are called "positional"; ones starting with
`...` are called "variadic"; and the ones that start with "`.` &lt;id> `=`" are
called "keyword". Positional, variadic, and keyword patterns are optional -- you
can have zero, one, two, or all three kinds. Positional &lt;value pattern>s must
always appear before keyword &lt;value pattern>s when both are present. Variadic
patterns can be mixed in with positional and keyword patterns.

In all three cases, a &lt;value pattern> can either be:

- &lt;value>
- (&lt;type> | `auto`) `:` [ `$` | `$$` ] (&lt;id> | `_`) [ `=` &lt;default
  value> ]
- &lt;value specification>

Positional &lt;value pattern>s without defaults (if any) must appear before
positional &lt;value pattern>s with defaults. Keywords &lt;value pattern>s can
have defaults or not, in any order.

**Condition specification:** [ `if` `(` &lt;expression> `)` ]

**TODO:** We need some mechanism for capturing the value of a match and binding
it to a name. [zygoloid](https://github.com/zygoloid) has suggested using the
`@` symbol. Another idea is to define a monotype `Only(v)` that only matches the
value `v`. Then `Only(F()) : x` would match if the value matches `F()`, in which
case the value would be bound to the name `x`.

### Value specification match rules

- **Positional:** a tuple value `(a, b)` matches value specification `(A, B)`
  iff `a` matches `A` and `b` matches `B`. It will not match either value
  specification `(A)` or `(A, B, C)`.
- **Positional defaults:** a tuple value `(a, b)` matches value specification
  `(A, B = C, D = E)` iff `a` matches `A`, `b` matches `B`, and `E` matches `D`.
  The default value for `B` of `C` is ignored because `b` is present in the
  input. `D` is not required in the input since it has a default, as long as the
  default value matches the pattern (see the
  [deduced specification section below](#deduced-specification-match-rules) for
  why it might not). The same value specification could match `(a)` or
  `(a, b, d)` but not `()` or `(a, b, d, g)`.
- **Keyword:** a tuple value `(.x = a, .y = b)` matches value specification
  `(.x = A, .y = B)` iff `a` matches `A` and `b` matches `B`. Order does matter
  for keywords so `(.y = b, .x = a)` is different from `(.y = B, .x = A)`.
- **Positional does not match keyword:** a tuple value `(a, b)` does _not_ match
  value specification `(.x = A, .y = B)` since positional inputs do not match
  keyword value patterns. Similarly a tuple value `(.x = a, .y = b)` does _not_
  match value specification `(A, B)`.
- **Keyword defaults:** a tuple value `(.x = a, .y = b)` matches value
  specification `(.x = A, .y = B = C, .z = D = E)` iff `a` matches `A`, `b`
  matches `B`, and `E` matches `D`. Here `.x` is required and `.y` & `.z` are
  optional, so the same pattern could match `(.x = a)` or `(.x = a, .z = d)` but
  not `(.y = b, .z = d)`.
- **Positional & keyword combination:** A tuple value `(a, b, .x = f, .y = g)`
  matches a value specification
  `(A, B = C, D = E, .x = F, .y = G = H, .z = I = J)` iff `a`, `b`, `f`, `g`
  match `A`, `B`, `F`, `G`.
- **Value:** A tuple value `(7, True)` matches value specification `(7, True)`
  but not `(7, False)`, `(8, True)`, or `(True, 7)`.
- **Type:** A tuple value `(7, True)` matches value specification
  `(Int: a, Bool: b)` or `(Int: _, Bool: _)`, but not `(Bool: a, Int: b)` or
  `(Bool: _, Int: _)`.
- **Recursive:** A tuple value `((7, .x = True), .y = (8, .z = False))` matches
  value specification `((Int: _, .x = Bool: _), .y = (Int: _, .z = Bool: _))`.
- **Generic** (`:$`) and **Template** (`:$$`): These both require that the value
  of the argument is available at compile time -- these match compile-time
  constants. The difference is that in the template (`:$$`) case, the value is
  available as part of type-checking, whereas the value of generic (`:$`)
  constants is only known at code generation time. In the case of generic and
  template arguments to functions, the function body to be instantiated once for
  every combination of values to those arguments. Template arguments prevent
  type checking until the function is instantiated so the value is known.
  Generic arguments allow the function to be type checked once when the body is
  defined. See the
  [generics overview (TODO)](https://github.com/josh11b/carbon-lang/blob/generics-docs/docs/designs/generics-overview.md)
  for more details.
- **Variadic** (`...`): These match multiple arguments which are packed into a
  tuple. So a tuple value `(1, 2, 3, 4)` matches `(... Array(Int): x)` --
  assuming that `Array(Int)` has a constructor that will accept an `Int`
  4-tuple.

### Deduced specification match rules

The rule is that a value will match `[A: a, B: b](...)` if there is a unique
value of `a` with type `A` and unique value of `b` with type `B` that when
substituted into the value specification (the `...` part), there is a match.

- `(True, True)` and `(False, False)` match `[Bool: b](b, b)`. Neither
  `(True, False)` nor `(False, True)` match that pattern, though.
- `(True, False)` and `(7, 8)` match `[Type: T](T: a, T: b)` (with `T = Bool`
  and `T = Int` respectively). Neither `(True, 8)` nor `(7, False)` match that
  pattern.
- `(MakeFixedArray(1, 2, 3))` matches `[Type: T, Int: N](FixedArray(T, N): x)`
  with `T = Int` and `N = 3`.
- `(True, False)` and `(7)` matches `[Type: T](T: a, T: b = 3)` with `T = Bool`
  and `T = Int` respectively. Note that `(True)` does not match the pattern:
  since no value is provided for `b`, it gets its default value of `3` and there
  is no consistent assignment of `T` for `(True, 3)`.

**Generic** (`:$`) and **Template** (`:$$`) are handled the same way as in a
value specification (see above).

**Question:** Do we want to require that all deductions are resolved at compile
time? This would only be relevant to the `match` statement:

```
fn f(Bool: a, Bool: b) -> String {
  match (a, b) {
    case [Bool: c](c, c) => {
      return "same";
    }
    default => {
      return "different";
    }
  }
}
```

This would be equivalent to:

```
fn f(Bool: a, Bool: b) -> String {
  match (a, b) {
    case (Bool: c, Bool: d) if (c == d) => {
      return "same";
    }
    default => {
      return "different";
    }
  }
}
```

### Destructuring rules

- Once you match `(2, False)` to `(Int: x, Bool: y)`, this will bind `x = 2` and
  `y = False`.
- Matching `(2, False)` to `(2, False)` won't bind any names to values.
- Once you match `(2, False)` to `(Int: x, Bool: _)`, will just bind `x = 2`.
  The pattern `(Int: _, Bool: _)` won't bind any names to values.
- Matching `(2)` to `(Int: x = 3, Int: y = 4)` will bind `x = 2` (ignoring the
  default) and `y = 4` (using the default).

### Condition

The semantics of matching a value to the pattern `[A: a](B: b, C: c) if (X)` is to:

1. match without considering the `if (X)` clause
2. tentatively bind `a`, `b`, and `c` as a result of that match
3. evaluate `X`, which may use the names `a`, `b`, and `c`
4. if the result is `True`, the match succeeds and those tentative bindings
   become real bindings
5. otherwise the match fails

**Examples:**

- `(3, 4)` would match `(Int: x, Int: y) if (x < y)` but `(4, 3)` does not.
- `(3, 4)` would match
  `[Type: T](T: x, T: y) if (T implements ComparableInterface)`.

## Broken links footnote

Some links in this document aren't yet available, and so have been directed here
until we can do the work to make them available.

We thank you for your patience.
