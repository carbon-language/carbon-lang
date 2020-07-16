# Carbon tuples and variadics

<!--
Part of the Carbon Language, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

## Table of contents

<!-- toc -->

- [Overview](#overview)
- [Basic design](#basic-design)
- [Question: Tuple types](#question-tuple-types)
- [Member access](#member-access)
- [Multiple indices](#multiple-indices)
- [Slicing](#slicing)
- [Destructuring](#destructuring)
- [Named members](#named-members)
  - [Proposed syntax](#proposed-syntax)
  - [Order matters](#order-matters)
  - [Struct conversion](#struct-conversion)
- [Function calling](#function-calling)
  - [Keyword arguments](#keyword-arguments)
  - [Unpacking](#unpacking)
  - [Variadic function arguments](#variadic-function-arguments)
  - [Concern](#concern)
- [Equality](#equality)
- [Rejected alternative: Parens](#rejected-alternative-parens)

<!-- tocstop -->

## Overview

Tuples in Carbon play a role in several parts of the language:

- They are a light-weight product type.
- They support multiple return values from functions.
- They provide a way to specify a literal that will convert into a struct value.
- They are involved in pattern matching use cases (such as function signatures)
  particularly for supporting varying numbers of arguments, called "variadics."
- A tuple may be unpacked into multiple arguments of a function call.

## Basic design

A tuple value is created by combining values with commas inside one of the
[compound brackets](https://github.com/zygoloid/carbon-lang/blob/p0016/docs/proposals/p0016.md#brackets),
say `(|...|)`:

```
var auto: a = (|1, 2, 3|);
```

**Concern:** `(|` and `|)` are not very convenient to type. See
[the "rejected alternative: parens" section below](#rejected-alternative-parens)
for why we chose this syntax.

A tuple type is just a tuple of types:

```
var (|Int, Int, Int|): b = (|1, 2, 3|);
```

Tuples support having elements with different types:

```
var (|Int, Float64, Bool|): c = (|1, 2.0, true|);
```

For uniformity, we support tuples with 0 or 1 component (call 0-tuples or
1-tuples):

```
// 1-tuple
var (|Int|): d = (|1|);
Assert(typeof(d) != typeof(Int));

// 0-tuple: type and only value are both spelled (||) (equivalently `(| |)`)
var (||): e = (||);
```

**Oddity:** The 0-tuple type should also be able to be written `struct {}`, but
that isn't a way of writing the 0-tuple value.

## Question: Tuple types

One thing we have considered is using a type constructor to name tuple types, so
the type of `(|1, 2.0, true|)` would be `Tuple(Int, Float64, Bool)` instead of
`(|Int, Float64, Bool|)`. Another way we could spell these types is by equating
tuples and unnamed structs, but this would require a syntax for defining unnamed
fields in structs accessed positionally.

Conversely, we could get rid of the anonymous struct syntax so that there is
only one representation for anonymous product types.

## Member access

**Question:** To access a member of a tuple we are considering two possible
syntaxes:

```
// Option 1: subscripting tuples using `t[i]` like arrays
var (|Float64, Int, Bool|): x1 = (|c[1], c[0], c[2]|);
// Option 2: using a tuple-specific operator, perhaps `t[|i|]`
var (|Float64, Int, Bool|): x2 = (|c[|1|], c[|0|], c[|2|]|);
```

The basic question here is tuple enough like an array to use the same operator
to access its elements?

davidstone argues they are indeed similar enough:

> Indexing into a tuple seems fundamentally the same to me as indexing into an
> array. The operation is just more generic, which means that it is less
> powerful -- the special case of arrays can promise more about their contents,
> namely that they are all the same type.
>
> A tuple has a size that is fixed at compile time, and types can be different.
> Array is the special case of that where all types must be the same.
>
> I would want to just use `tuple[index]`, where `index` is either a
> compile-time value or, if it can uniquely identify an element, a type.

Similarly, since we plan on supporting slicing a tuple just like we will for
arrays, that suggests we would want to use the same syntax, just to avoid a
proliferation of parallel ways of doing things for tuples vs. arrays.

The argument that they are different has to do with the signature being
different. In particular, two things:

- The argument must be template-const, that is either a literal or a `:$$`
  value.
- The return type varies with the value of the argument.

One thing to think about is whether this `[|index|]` operator would also be used
with
[structs](https://github.com/josh11b/carbon-lang/blob/structs/docs/design/structs.md)?
You could imagine it would accept a compile-time ("template") constant string
that matched a field name, as in:

```
struct S {
  var Int: a = 1;
}

var S: s;
Assert(s.a == s[|"a"|]);
```

If we provide some way of defining positional fields for structs (which we
haven't done so far, but would allow us to make tuples a special case of
structs), then it would be convenient to access those positional field using
`[|index|]` to avoid interfering with any operator `[]` you might want to define
for that struct type.

A more practical concern is that `[|`, `|]` is harder to type than `[`, `]`. If
this is going to be a common operation in code, we may want the more convenient
choice. The counter argument is that we expect tuples to most commonly be used
to represent multiple return values from a function, and that case is most
naturally handled by immediately
[destructuring the value into separately named variables](#destructuring). In
fact, this may not be a common enough operation to have its own dedicated
syntax.

**Tentative proposal:** Option 1, we will use the `[index]` form.

**Rejected alternative:**

> Accessing tuple members uses their index with a dot.

```
var (|Float64, Int, Bool|): z = (|x.1, x.0, x.2|);
```

> **Concern:** Visual ambiguity with floating point numbers, as in `I.0` and
> `l.0` vs. `1.0`.
>
> **Concern:** This syntax is awkward when supplying a template-constant value
> that is not a literal. We expect to need this when doing things like iterating
> through the components of a tuple when meta-programming. We could possibly
> have a meta-programming-specific operator to substitute the value of an
> expression to handle this.

## Multiple indices

Pass tuple of indices to a tuple to get another tuple:

```
// This reverses the tuple using multiple indices.
fn Reverse((|Int, Int, Int|): x) -> (|Int, Int, Int|) {
  return x[(|2, 1, 0|)];
}
Assert(Reverse( (| 10, 20, 30 |) ) == (| 30, 20, 10 |) ));

Assert( (| 10, 20, 30 |)[(| 1, 1, 0, 0 |)] == (| 20, 20, 10, 10 |) );
```

**Rejected alternative:** The problem with passing a comma-separated list
directly to the indexing operator, as in `x[2, 1, 0]`, is that it doesn't
generalize to the behavior of one index. That is, it makes it ambiguous whether
`x[1]` should be a scalar or a 1-tuple. It also conflicts with wanting to
support multidimensional arrays / slices (important to the scientific world).

**Future extension:** The syntax as proposed could be generalized to arguments
with a more general structure, like:

```
Assert( (| 10, 20, 30 |)[(| (| 1 |), (| 0 |) |)] == (| (| 20 |), (| 10 |) |) );
```

## Slicing

Slicing allows you to get a subrange of a tuple:

```
// This slices the tuple by extracting elements [0, 2).
fn RemoveLast((|Int, Int, Int|): x) -> (|Int, Int|) {
  return x[0 .. 2];
}
```

## Destructuring

A destructuring pattern can decompose tuples into separately named variables:

```
var (|Int: x, Int: y, Int: z|) = (|1, 2, 3|);

var Int: a;
var Int: b;
var Int: c;
(|a, b, c|) = (|7, 8, 9|);
```

This naturally supports the use case of representing multiple return values from
a function

```
fn Position(Point: p) -> (|Float32, Float32|) {
  return (|p.x, p.y|);
}

var (|Float32: x, Float32: y|) = Position(this_point);
// ...
(|x, y|) = Position(that_point);
```

## Named members

In addition to supporting members accessed by position, we also support naming
the members of a tuple. This is useful in its own right, the
[Google C++ style guide](https://google.github.io/styleguide/cppguide.html#Structs_vs._Tuples)
recommends structs over C++'s pair & tuple precisely because naming the
components is so important. In addition, we have two applications for tuples
that benefit from having named members:

- Providing literal values that can be used to initialize a struct (with named
  fields).
- Representing the arguments to a function with named parameters.

### Proposed syntax

```
var auto: proposed_value_syntax = (|.x = 1, .y = 2|);
var (|.x = Int, .y = Int|): proposed_type_syntax = (|.x = 3, .y = 4|);
var (|.z = Int|): tuple_with_single_named_field = (|.z = 7|);
```

The `.` here means "in the tuple's namespace"; so it is clear that `x` and `y`
in that first line don't refer to anything in the current namespace. The intent
is that `.x` reminds us of field access, as in `foo.x = 3`, and indeed that is
how you would access the named members of a tuple:

```
Assert(proposed_value_syntax.x == 1);
Assert(tuple_with_single_named_field.z == 7);
```

Notice that the type of `(|.x = 3, .y = 4|)` is represented by another tuple,
`(|.x = Int, .y = Int|)`, containing types instead of values. This is to mirror
the same pattern for tuples with unnamed components: `(|Int, Int|)` (a tuple
containing two types) is the type of `(|3, 4|)`, as seen [above](#basic-design).

### Order matters

Tuples may contain a mix of positional and named members; the positional members
must always come first.

**Proposal:** Order still matters when using named fields.

Note that my first design was that two tuples with the same (name, type) pairs
(as a set) were compatible no matter the order they were listed. In particular,
these statements would not be errors:

```
// Compile error! Order doesn't match.
Assert(proposed_type_syntax == (|.y = 4, .x = 3|));
// Compile error! Order doesn't match.
proposed_type_syntax = (|.y = 7, .x = 8|);
```

This has a problem though, when the order of fields of the type doesn't match
the order of the initializer, what order are the components evaluated? A reader
of the code will expect the evaluation to match the most visible order written
in the code: the order used for the initializer. But the ordering of the fields
of the type is what will determine the destruction order -- which really should
be the opposite of the order that values are constructed.

**Concern:** Field order affects hidden implementation details like alignment
and padding, and yet the field order is also API-visible in a way that clients
almost can't avoid depending on.

We may need to provide a way for users to explicitly specify a struct layout and
an explicit (different) order of initialization for the rare structs where this
matters.

### Struct conversion

We allow tuples with named fields to be converted to structs with a compatible
set of fields. This would be used, for example, to provide initial values for
struct variables, or a value for assignment statements.

```
struct Point {
  var Int: x;
  var Int: y;
}

var Point: p = (| .x = 1, .y = 2 |);
p = (| .x = p.x+1, .y = p.y+2 |);
Assert(p == (| .x = 2, .y = 4 |));
```

This is covered in more detail in
[the structs design doc](https://github.com/josh11b/carbon-lang/blob/structs/docs/design/structs.md#simple-initialization-from-a-tuple).

## Function calling

Tuples can represent function arguments. The idea is that the function calling
protocol is equivalent to packaging up all the arguments to the function at the
call site into a tuple and then using tuple destructuring according to the
pattern from the function's signature to determine the parameters passed into
the function body.

This consistency allows us to convert between tuples and function arguments in
both directions.

### Keyword arguments

Just as tuples without names are used to provide positional arguments when
calling a function, tuples with names are used to call functions that take
keyword arguments. Just as tuples may contain a mix of positional and named
members, functions can take a combination of positional and keyword arguments.
Just as the positional members will always come first in tuples, positional
arguments will always come before keyword arguments in functions.

Here is an example showing the syntax:

```
  // Define a function that takes keyword arguments.
  // Keyword arguments must be defined after positional arguments.
  fn f(Int: p1, Int: p2, .key1 = Int: key1, .key2 = Int: key2) { ... }
  // Call the function.
  f(positional_1, positional_2, .key1 = keyword_1, .key2 = keyword_2);
  // Keyword arguments must be provided in the same order.
  // ERROR! Order mismatch:
- f(positional_1, positional_2, .key2 = keyword_2, .key1 = keyword_1);
```

We are currently making order always matter for [consistency](#order-matters),
even though the implementation concerns for function calling may not require
that particular constraint.

Keyword arguments to functions are covered in more detail in
[the pattern matching design doc](https://github.com/josh11b/carbon-lang/blob/pattern-matching/docs/design/pattern-matching.md#positional-amp-keyword-arguments).

### Unpacking

We allow expanding a tuple with the `...` postfix operator when calling a
function. For example:

```
fn Overload(Int: s1, Int: s2, Int: s3) -> Int { return 1; }
fn Overload(Int: s1, (|Int, Int|): tuple) -> Int { return 2; }
fn Overload((|Int, Int|): tuple, Int: s3) -> Int { return 3; }
fn Overload((|Int, Int, Int|): tuple) -> Int { return 4; }

var (|Int, Int, Int|): tuple3 = (|10, 20, 30|);
// Overload is passed a single argument that is a 3-tuple.
Assert(Overload(tuple3) == 4);
// Overload is passed three separate integer arguments.
Assert(Overload(tuple3...) == 1);

var (|Int, Int|): tuple2 = (|40, 50|);
Assert(Overload(42, tuple2) == 2);
Assert(Overload(42, tuple2...) == 1);
Assert(Overload(tuple2, 42) == 3);
Assert(Overload(tuple2..., 42) == 1);
```

When unpacking, the named members of a tuple will match to the keyword arguments
of a function. For example:

```
fn WithKeywordParameters(Int: x, .a = Int: a, .b = Int: b) -> Int {
  return x + a + b;
}
fn TupleArgument(

var auto: ab_tuple = (| .a = 2, .b = 3 |);
Assert(WithKeywordParameters(1, ab_tuple...) == 6);

var auto: mixed_tuple = (| 4, .a = 5, .b = 6 |);
Assert(WithKeywordParameters(mixed_tuple...) == 15);
```

You can also unpack tuple when creating another tuple:

```
var (|Int, Int|): a = (| 1, 2 |);
var (|Bool, Int, Int|): b = (| true, a... |);
var (|Int, Int, Bool, Int, Int|): c = (| a..., b... |);
```

As a general rule, we should allow unpacking any place that takes a
comma-separated list.

### Variadic function arguments

General syntax for pattern matching some number of arguments is "`...`
&lt;Type>`:` &lt;Name>". The named parameter is set to a tuple containing the
matched arguments. Since we are using this mechanism to match an unknown number
of arguments, typically we will need a template type. For this we provide a
type-type named `TupleType` whose values are the set of all tuple types.

```
fn Overload(Int: s1, Int: s2, Int: s3) -> Int {
  return 1;
}
fn Overload((|Int, Int, Int|): tuple) -> Int {
  return 2;
}
// T can be any tuple type.
fn Forward1[TupleType:$$ T](... T: args) -> Int {
  // Using the unpacking syntax for tuples.
  return Overload(args...);
}
fn Forward2[TupleType:$$ T](... T: args) -> Int {
  // `args` is a tuple so this calls the second `Overload`.
  return Overload(args);
}

Assert(Forward1(1, 2, 3) == 1);
Assert(Forward1( (| 1, 2, 3 |) ) == 2);
Assert(Forward2(1, 2, 3) == 2);
```

We'll also have type-types `UnnamedTupleType` and `NamedTupleType` for matching
just positional or keyword arguments.

**Question:** We need to decide on the mechanism for determining how many
arguments are matched by the `...`. Is it influenced by &lt;Type>? Or the next
thing in the parameter list?

We shall also define a type `NTuple(Int:$$ N, Type:$$ T)` which is equivalent to
the tuple type `(|T, T, ...|)` with `N` components. This lets us declare
variadic arguments that all have to be the same type, and the `Max` function:

```
fn Max[Int:$$ N, Comparable:$$ T](... NTuple(N, T): args) -> T {
  match (args) {
    case (| T: x |) => return x;
    // Using the same variadic pattern matching syntax in a `match` statement.
    case (| T: first, ... NTuple(N-1, T): rest |) => {
      var T: max_of_rest = Max(rest...);
      if (first < max_of_rest) {
        return max_of_rest;
      } else {
        return first;
      }
    }
  }
}

// `N` == 3, `T` == Int
Assert(Max(1, 3, 2) == 3);
```

Here we'd probably restrict `N > 0` to avoid problems deducing `T` in the `N==0`
case. Another way of writing `Max` would allow `N==0`, since `T` is determined
by another argument:

```
fn Max[Int:$$ N, Comparable:$$ T](T: first, ... NTuple(N, T): rest) -> T {
  // Note: This is `@meta if` so the `else` branch is not type-checked
  // when `N == 0`.
  @meta if (N == 0) {
    return first;
  } else {
    var T: max_of_rest = Max(rest...);
    if (first < max_of_rest) {
      return max_of_rest;
    } else {
      return first;
    }
  }

// `N` == 2, `T` == Int
Assert(Max(1, 3, 2) == 3);
```

Potentially we could allow this to be passed things with dynamic lengths,
avoiding the need to instantiate a different version of the function for each
number of arguments:

```
fn Max[Comparable:$ T](... DynamicLengthArray(T): args) -> T {
  Assert(args.length() > 0);
  var T: biggest_so_far = args[0];
  var Int: i = 1;
  loop {
    if (i >= length) break;
    if (args[i] > biggest_so_far) {
      biggest_so_far = args[i];
    }
    ++i;
  }
  return biggest_so_far;
}

// `T` == Int
Assert(Max(1, 3, 2) == 3);
// `T` == Int as well, same instantiation of `Max`.
Assert(Max(1, 2, 4, 8, 16) == 16);
```

Again, there is an issue with deducing the type `T` if there are no arguments to
match, in addition to it being an error case for this particular example.

**Question:** Sometimes it seems like `...` matching no arguments is an awkward
edge case where things break, other times it seems fine. How should we handle
that?

### Concern

[geoffromer](https://github.com/geoffromer) says:

> It feels weird to use the same syntax [`...`] to turn a sequence into a tuple
> (in pattern matching), and to turn a tuple into a sequence (in procedural
> code)
>
> This means we have two operations that have the same spelling, but exactly
> opposite semantics. That feels confusing, in a way that sticks out because so
> many other confusing aspects of C++ variadics have been cleaned up.
>
> The pattern-matching syntax seems closer to the natural-language meaning of an
> ellipsis, so I'd keep that, but I'd prefer something else for the expression
> syntax. I don't know of any good precedents, unfortunately- `**` seems
> unworkable because of ambiguity with double-dereferencing.

## Equality

Tuple types don't have names, so they are compared structurally. That means two
tuple types are equal if they have the same member types and names (if present)
in the same order. Two tuples may be compared if they have the same type and all
of their component types define a comparison operator.

**Note:** I assume we do not want to also support order comparisons, like C++20
does if the components do. But we should provide some easy way to get one,
possibly using
[an adapting type](https://github.com/josh11b/carbon-lang/blob/generics-docs/docs/designs/generics-interface-type-types.md#adapting-types).

## Rejected alternative: Parens

We have also considered letting a tuple value be created by combining values
with commas inside of parentheses

```
var auto: x = (1, 2, 3);
```

There are two problems with this.

First, how do you write a 1-tuple? You can add a trailing comma to distinguish
between a regular parenthesized expression and a 1-tuple, but it ends up being
an inconsistency.

Secondly, it means that when encountering an opening parenthesis `(` in the
source code, the parser doesn't know whether it is defining a tuple or an
ordinary parenthesized expression until it sees a comma `,` or the matching
closing parenthesis `)`.

**Rejected proposal:**

Formally, tuples represent product types. However, in Carbon we allow for
one-tuples to exist in order to support preserving important information when
modeling variadic functions. We want to be able to distinguish between the case
of a one-tuple and the case of a scalar when writing code that is necessarily
generic over the arity of a tuple.

A one-tuple can be written explicitly using a trailing comma within parentheses:

```
var auto: tuple = (42,);
```

Purely for aesthetic and consistency reasons (we want exactly one way to spell
things wherever reasonable), a trailing comma is not allowed for tuples with
arity greater than one:

```
var auto: invalid = (1, 2, 3,);
```

Because function calls are always tuples, the trailing comma is unnecessary and
invalid even for one-tuples:

```
DoSomething(42);
```

This call forms a one-tuple to represent the argument to the function. This in
turn matches the parameter syntax which is required to destructure the argument
tuple and doesn't require a comma there either:

```
fn DoSomething(Int: x);
```

Other pattern matching (which doesn't involve an intrinsic destructuring as in
the parameter list) works the same as expressions to destructure one tuples by
using a single comma:

```
template fn F(Int: x) {
  // Match against the one-tuple containing `x`.
  match (x,) {
    // Destructure the matched one-tuple and bind the value to `y`.
    case (Int: y,) {
      // ...
    }
    // ...
  }

  // Without a comma, the parentheses are just parentheses here as
  // this isn't a function call or declaration.
  var ((((Int)): z)) = ((42));
}
```

While requiring a trailing comma in one-tuple expressions and one-element
destructuring patterns outside of function calls and parameters is somewhat
awkward, it is expected to be very rarely used in practice and provides for
clear, unambiguous, and consistent syntax in the common uses of parentheses.

> **Open question:** should we allow spelling the unit type `()` as well as
> `Void` outside of a pattern? Should we only allow the spelling of the unit
> type to be `()`? Doing so is annoying as the idea is that "`,`" is what
> signifies a tuple. But we basically have to allow a pattern matching the unit
> type as `()` to make generic things sensible, and it would seem somewhat more
> orthogonal to allow that to simply _be_ the unit type which makes it a normal
> pattern spelling.
