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

<!-- tocstop -->

NOTE: this doc has a lot of overlap with the
[tuples (TODO)](#broken-links-footnote)<!-- T:Carbon tuples and variadics -->
doc; we should do some consolidation!

## Goals

Pattern matching is a general Carbon mechanism for handling structured values
like
[tuples (TODO)](#broken-links-footnote)<!-- T:Carbon tuples and variadics -->.
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

In other languages:
[Rust](https://doc.rust-lang.org/book/ch18-03-pattern-syntax.html),
[Circle](https://github.com/seanbaxter/circle/blob/master/pattern/pattern.md),
... TODO

[NOTE: This text also appears in
["Carbon language design / Pattern matching"](https://github.com/jonmeow/carbon-lang/blob/proposal-design-overview/docs/design/README.md#pattern-matching).]

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

Note: we may want to say `where` instead of `if`, I've seen both suggested in
various docs.

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

  **Note:** an open question is how to effectively fit a "slice" or "array"
  pattern into this (or whether we shouldn't do so).


    **Note:** an open question is going beyond a simple "type" to things that support generics and/or templates.

In order to match a value, whatever is specified in the pattern must match.
Using `auto` for a type will always match, making `auto: _` the wildcard
pattern.

### Pattern matching in local variables

[NOTE: This text also appears in
["Carbon language design / Pattern matching"](https://github.com/jonmeow/carbon-lang/blob/proposal-design-overview/docs/design/README.md#pattern-matching).]

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

The argument list to a function in Carbon is a pattern. Functions may be
overloaded by defining functions with the same name but different patterns.
Example:

```
fn ToString(Int: x) -> String { return "A" }
fn ToString(Bool: x) -> String { return "B" }
fn ToString[Type:$ T](T: x) if (T implements SerializableInterface) -> String {
  return "C"
}

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

**Question:** Can the set of overloads associated with a name be different in
different files or different positions within the same file?

It would be nice if the answer was no. In particular it would be useful to make
things like type functions deterministic (avoiding a source of
[ODR](https://en.wikipedia.org/wiki/One_Definition_Rule) problems), and
refactorings that move code around safer. However, that is contrary to the goal
that code later in a file does not affect the interpretation of code earlier in
the file.

Also, I think this is too strict. A weaker property that we could aim for
instead is: the overload selected only depends on the arguments, not the
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

- Library `A` defines a function `A.f` taking an `Int`.
- Library `B` defines a type `T`.
- Library `C` defines an interface `Interface1` and an implementation of
  `Interface1` for `B.T`. This is legal since implementations of interface can
  either be defined with the type or the interface (to address dependency issues
  and the expression problem). Library `C` also defines an overload for `A.f`
  for types implementing `Interface1`.
- Library `D` defines an interface `Interface2`, an implementation of
  `Interface2` for `B.T`, and an overload for `A.f` for type implementing
  `Interface2`. Same deal as with library `C`.

Code importing libraries `A` and `B` will find calling `A.f` with a value of
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

**Question:** Is there a rule for selecting an overload if there are multiple
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

  var auto: f = fn(Int: x) -> String { return ToString(x); }; var auto: g =
  fn(MySerializableType: x) -> String { return ToString(x); };

```



**Note:** For functions argument lists, we also support template / generic arguments, but that is detailed in [another document (TODO)](#broken-links-footnote)<!-- T:Carbon templates and generics -->.


### Differences

There are a few differences between these uses:



*   The `match` statement has several patterns and determines which applies at run time.
*   An assignment statement will have a single pattern, and it will be a compile error unless it can be shown to match at compile time.
*   Function overloading allows multiple patterns, but at compile time any function call must be resolved to match a single pattern. Right now this is the only case that uses the generic (:$) and template (:$$) syntax.


## Features


### Positional & keyword arguments

**Rationale:** In general, positional arguments are great when either there is an obvious order to provide the arguments (e.g. coordinates in x, y, z order) or the order doesn't matter (e.g. `max()` is commutative). Otherwise, keyword arguments can (a) prevent mistakes (mixing up source & destination) and (b) provide documentation (what does this "True" argument mean?).

Context: [Be wary of functions which take several parameters of the same type](https://news.ycombinator.com/item?id=21086666).

Context: [Google C++ Style Guide](https://google.github.io/styleguide/cppguide.html#Structs_vs._Tuples) says "Prefer to use a struct instead of a pair or a tuple whenever the elements can have meaningful names."

Named components are also naturally a part of tuples so they can be used to initialize struct variables. When a struct type is changed to use a factory function / constructor for initialization, we should be able to use the same initialization syntax. The tuple with named components used to initialize will become the arguments to the factory function, which the factory function will declare as keyword arguments in order to match. See ["Carbon struct types" (TODO)](#broken-links-footnote)<!-- T:Carbon struct types -->.

**Proposal:** Furthermore, I believe we should use keywords in overload resolution. Since the keywords are written in both the caller and the function signature, there is no ambiguity about which function should be called (considerably less than using types in overload resolution, which can be much less visible in the source code). Furthermore, there are cases where using the types of the argument to disambiguate is less clear or even ambiguous. For example, let us say we see a `Vector(Int)` being constructed with the arguments `(3, 5)`. Is that creating a vector containing three 5s or the two values `3, 5`? Different names for the arguments would resolve the question of which constructor should be used.

**Note:** Even when using keyword arguments, we are currently saying that the order still needs to match between the call site and the function signature. This is to simplify our order-of-evaluation story. It is expected that IDEs (and other tooling) will make it easy to update the code whenever there is a mismatch.


### Defaults / optional arguments

**Rationale:** By providing a facility for specifying defaults for arguments in the function signature and then making those arguments optional at the call site, we provide a number of benefits:



*   Brevity for common cases where the defaults are what the user wants.
*   Allow many combinations of options to be specified at the caller without a combinatorial explosion of function definitions.
*   Allow evolution of an API. A new argument can be added to an existing function as long as it has a default that preserves the old behavior.

Note that this feature is more powerful when combined with keyword arguments. In C++ which only uses positional arguments, you always have to specify a prefix of the argument list at the call site, and so only the tail of the argument list in the signature is allowed to have defaults. Within the keyword arguments, any may have defaults, and the caller may skip specifying any optional argument and still specify later arguments.


### Deduced arguments

**Rationale:** The primary use case for deduced arguments is for type inference, but it could also be used to infer the size of an array argument. This is mainly for brevity, so you don't need to specify redundant information at the call site. It also provides continuity with C++ which has had this feature for a while. TODO: add ways this is used to enhance expressivity.

Note that the rules in C++ for deduced type arguments end up complex due to dealing with `const` and references (both `&` and `&&`).


### Variadics (...)

**Rationale:** Variadics are important for expressiveness. Functions like `StrCat` and `max` are significantly more convenient if they work with a variable number of arguments.

**Question:** Do we want there to be some convenient way to say that all values matched by a variadic have to be the same type?

Seems like this would be a reasonably common case for variadics. For example, `max` could take a variable number of arguments, as well as a function to create a fixed-sized array from an argument list. In this case, the compiler might be able to coerce the arguments to a common type when that was harmless, or give a clear error when the arguments were incompatible.

This can introduce special cases, though. For example, do we allow a function to take two variadics in a row? This might be okay if the first one was restricted to always be `Int` values and the second to always be `String`s. But less okay if those types were deduced types.

**Proposal:** There should be a way to control whether `...` matches positional vs. keyword arguments.

For example, `max` would only take positional arguments via `...`, and may have a non-variadic keyword argument to provide the comparison function. On the other hand, an [encode_as_json(...)](https://github.com/nlohmann/json) function should only take keyword arguments. Python has two different syntaxes for matching the two different cases, which I think is reasonable. I'm not sure what the two different syntaxes should be, though.


### Conditions

**Rationale:** Conditions are most helpful when used in a `match` statement. In that context, we are expressing a condition and the condition part is needed for completeness. In a `match` statement the conditions have an order in which they are tested, so something like:


```

match (a) { case P1 if C1 => { ... } case P2 => { ... } case P1 => { ... } }

```


would be hard to rewrite to avoid using conditions.

We still intend to support conditions in function overload resolution for consistency, but the use cases are less clear. Note that in order to satisfy the "overloads must be resolved at compile time" plan, these would have to be conditions that could be evaluated at compile time, and so generally can't use anything about the dynamic values being supplied as arguments. Restricting to generic and template arguments of the function, they could be used as part of the generic constraints story.

**Speculation:** For example they may be useful to express a type satisfies multiple constraints. Or to express that different implementations are used for forward and random-access iterators. They may also be needed to make overload resolution less ambiguous in a case where otherwise two signatures would match.

**Proposal:** Conditions will not be used to affect deductions -- since conditions are arbitrary expressions they will be evaluated as a black box returning "true" or "false", not as information that can disambiguate tricky deduction cases.


## Specification


### Pattern syntax

A pattern consists of three pieces, in sequence:



*   Deduced specification
*   Value specification
*   Condition specification

Only the value specification is required.

**Deduced specification:** [ `[`&lt;type> `:` [ `$` | `$$` ] &lt;id>`,` ... `]` ]

**Value specification:** `(` &lt;value pattern>`,` ... `,` `.` &lt;id> `=` &lt;value pattern>`,` ... `)`

The initial &lt;value pattern>s are called "positional"; the ones that start with "`.` &lt;id> `=`" are called "keyword". Positional &lt;value pattern>s must always appear before keyword &lt;value pattern>s. Here a &lt;value pattern> can either be:



*   &lt;value>
*   (&lt;type> | `auto`) `:` [ `$` | `$$` ] (&lt;id> | `_`) [ `=` &lt;default value> ]
*   &lt;value specification>

Positional &lt;value pattern>s without defaults (if any) must appear before positional &lt;value pattern>s with defaults. Keywords &lt;value pattern>s can have defaults or not, in any order.

**Condition specification:** [ `if` `(` &lt;expression> `)` ]

**TODO:** We need some mechanism for capturing the value of a match and binding it to a name. [zygoloid](https://github.com/zygoloid) has suggested using the `@` symbol.


### Value specification match rules



*   **Positional:** a tuple value `(a, b)` matches value specification `(A, B)` iff `a` matches `A` and `b` matches `B`. It will not match either value specification `(A)` or `(A, B, C)`.
*   **Positional defaults:** a tuple value `(a, b)` matches value specification `(A, B = C, D = E)` iff `a` matches `A`, `b` matches `B`, and `E` matches `D`. The default value for `B` of `C` is ignored because `b` is present in the input. `D` is not required in the input since it has a default, as long as the default value matches the pattern (see the [deduced specification section below](#deduced-specification-match-rules) for why it might not). The same value specification could match `(a)` or `(a, b, d)` but not `()` or `(a, b, d, g)`.
*   **Keyword:** a tuple value `(.x = a, .y = b)` matches value specification `(.x = A, .y = B)` iff `a` matches `A` and `b` matches `B`. Order does matter for keywords so `(.y = b, .x = a)` is different from `(.y = B, .x = A)`.
*   **Positional does not match keyword:** a tuple value `(a, b)` does *not* match value specification `(.x = A, .y = B)` since positional inputs do not match keyword value patterns. Similarly a tuple value `(.x = a, .y = b)` does *not* match value specification `(A, B)`.
*   **Keyword defaults:** a tuple value `(.x = a, .y = b)` matches value specification `(.x = A, .y = B = C, .z = D = E)` iff `a` matches `A`, `b` matches `B`, and `E` matches `D`. Here `.x` is required and `.y` & `.z` are optional, so the same pattern could match `(.x = a)` or `(.x = a, .z = d)` but not `(.y = b, .z = d)`.
*   **Positional & keyword combination:** A tuple value `(a, b, .x = f, .y = g)` matches a value specification `(A, B = C, D = E, .x = F, .y = G = H, .z = I = J)` iff `a`, `b`, `f`, `g` match `A`, `B`, `F`, `G`.
*   **Value:** A tuple value `(7, True)` matches value specification `(7, True)` but not `(7, False)`, `(8, True)`, or `(True, 7)`.
*   **Type:** A tuple value `(7, True)` matches value specification `(Int: a, Bool: b)` or `(Int: _, Bool: _)`, but not `(Bool: a, Int: b)` or `(Bool: _, Int: _)`.
*   **Recursive:** A tuple value `((7, .x = True), .y = (8, .z = False))` matches value specification `((Int: _, .x = Bool: _), .y = (Int: _, .z = Bool: _))`.
*   **Generic** (`:$`) and **Template** (`:$$`): These both require that the value of the argument is available at compile time, and cause the function to be instantiated once for each (combination of) value(s) to generic/template arguments. In the template (`:$$`) case, this body of the function will only be type checked once it is instantiated with this value (ignoring any control branches where that are unreachable as a result of substituting this value), whereas with the generic (`:$`) case the body of the function will be type checked when the body is defined.


### Deduced specification match rules

The rule is that a value will match `[A: a, B: b](...)` if there is a unique value of `a` with type `A` and unique value of `b` with type `B` that when substituted into the value specification (the `...` part), there is a match.



*   `(True, True)` and `(False, False)` match `[Bool: b](b, b)`. Neither `(True, False)` nor `(False, True)` match that pattern, though.
*   `(True, False)` and `(7, 8)` match `[Type: T](T: a, T: b)` (with `T = Bool` and `T = Int` respectively). Neither `(True, 8)` nor `(7, False)` match that pattern.
*   `(MakeFixedArray(1, 2, 3))` matches `[Type: T, Int: N](FixedArray(T, N): x)` with `T = Int` and `N = 3`.
*   `(True, False)` and `(7)` matches `[Type: T](T: a, T: b = 3)` with `T = Bool` and `T = Int` respectively. Note that `(True)` does not match the pattern: since no value is provided for `b`, it gets its default value of `3` and there is no consistent assignment of `T` for `(True, 3)`.

**Generic** (`:$`) and **Template** (`:$$`) are handled the same way as in a value specification (see above).

**Question:** Do we want to require that all deductions are resolved at compile time? This would only be relevant to the `match` statement:


```

fn f(Bool: a, Bool: b) -> String { match (a, b) { case [Bool: c](c, c) => {
return "same"; } default => { return "different"; } } }

```


This would be equivalent to:


```

fn f(Bool: a, Bool: b) -> String { match (a, b) { case (Bool: c, Bool: d) if (c
== d) => { return "same"; } default => { return "different"; } } }

```



### Destructuring rules



*   Once you match `(2, False)` to `(Int: x, Bool: y)`, this will bind `x = 2` and `y = False`.
*   Matching `(2, False)` to `(2, False)` won't bind any names to values.
*   Once you match `(2, False)` to `(Int: x, Bool: _)`, will just bind `x = 2`. The pattern `(Int: _, Bool: _)` won't bind any names to values.
*   Matching `(2)` to `(Int: x = 3, Int: y = 4)` will bind `x = 2` (ignoring the default) and `y = 4` (using the default).


### Condition

The semantics of `[A: a](B: b, C: c) if (X)` is to



1. match without considering the `if (X)` clause
2. tentatively bind `a`, `b`, and `c` as a result of that match
3. evaluate `X`, which may use the names `a`, `b`, and `c`
4. if the result is `True`, the match succeeds and those tentative bindings become real bindings
5. otherwise the match fails

**Examples:**



*   `(3, 4)` would match `(Int: x, Int: y) if (x &lt; y)` but `(4, 3)` does not.
*   `(3, 4)` would match `[Type: T](T: x, T: y) if (T implements ComparableInterface)`.


## Broken links footnote

Some links in this document aren't yet available,
and so have been directed here until we can do the
work to make them available.

We thank you for your patience.
```
