# Comparison operators

<!--
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

<!-- toc -->

## Table of contents

-   [Overview](#overview)
-   [Details](#details)
    -   [Precedence](#precedence)
    -   [Associativity](#associativity)
    -   [Built-in comparisons and implicit conversions](#built-in-comparisons-and-implicit-conversions)
        -   [Consistency with implicit conversions](#consistency-with-implicit-conversions)
        -   [Comparisons with constants](#comparisons-with-constants)
    -   [Extensibility](#extensibility)
        -   [Equality](#equality)
        -   [Ordering](#ordering)
        -   [Compatibility of equality and ordering](#compatibility-of-equality-and-ordering)
        -   [Custom result types](#custom-result-types)
    -   [Default implementations for basic types](#default-implementations-for-basic-types)
-   [Open questions](#open-questions)
-   [Alternatives considered](#alternatives-considered)
-   [References](#references)

<!-- tocstop -->

## Overview

Carbon provides equality and relational comparison operators, each with a
standard mathematical meaning:

| Category   | Operator | Example  | Mathematical meaning | Description                |
| ---------- | -------- | -------- | -------------------- | -------------------------- |
| Equality   | `==`     | `x == y` | =                    | Equality or equal to       |
| Equality   | `!=`     | `x != y` | ≠                    | Inequality or not equal to |
| Relational | `<`      | `x < y`  | <                    | Less than                  |
| Relational | `<=`     | `x <= y` | ≤                    | Less than or equal to      |
| Relational | `>`      | `x > y`  | >                    | Less than                  |
| Relational | `>=`     | `x >= y` | ≥                    | Less than or equal to      |

Comparison operators all return a `bool`; they evaluate to `true` when the
indicated comparison is true. All comparison operators are infix binary
operators.

These operators have predefined meanings for some of Carbon's
[built-in types](#built-in-comparisons-and-implicit-conversions), as well as for
simple ["data" types](#default-implementations-for-basic-types) like structs and
tuples.

User-defined types can define the meaning of these operations by
[implementing an interface](#extensibility) provided as part of the Carbon
standard library.

## Details

### Precedence

The comparison operators are all at the same precedence level. This level is
lower than operators used to compute (non-`bool`) values, higher than the
[logical operators](logical_operators.md) `and` and `or`, and incomparable with
the precedence of `not`.

For example:

```carbon
// ✅ Valid: precedence provides order of evaluation.
if (n + m * 3 < n * n and 3 < m and m < 6) {
  ...
}
// The above is equivalent to:
if (((n + (m * 3)) < (n * n)) and ((3 < m) and (m < 6))) {
  ...
}

// ❌ Invalid due to ambiguity: `(not a) == b` or `not (a == b)`?
if (not a == b) {
  ...
}
// ❌ Invalid due to precedence: write `a == (not b)`.
if (a == not b) {
  ...
}
// ❌ Invalid due to precedence: write `not (f < 5.0)`.
if (not f < 5.0) {
  ....
}
```

### Associativity

The comparison operators are non-associative. For example:

```carbon
// ❌ Invalid: write `3 < m and m < 6`.
if (3 < m < 6) {
  ...
}
// ❌ Invalid: write `a == b and b == c`.
if (a == b == c) {
  ...
}
// ❌ Invalid: write `(m > 1) == (n > 1)`.
if (m > 1 == n > 1) {
  ...
}
```

### Built-in comparisons and implicit conversions

Built-in comparisons are permitted in three cases:

1.  When both operands are of standard Carbon integer types (`Int(n)` or
    `Unsigned(n)`).
2.  When both operands are of standard Carbon floating-point types (`Float(n)`).
3.  When one operand is of floating-point type and the other is of integer type,
    if all values of the integer type can be exactly represented in the
    floating-point type.

In each case, the result is the mathematically-correct answer. This applies even
when comparing `Int(n)` with `Unsigned(m)`.

For example:

```carbon
// ✅ Valid: Fits case #1. The value of `compared` is `true` because `a` is less
// than `b`, even though the result of a wrapping `i32` or `u32` comparison
// would be `false`.
fn Compare(a: i32, b: u32) -> bool { return a < b; }
let compared: bool = Compare(-1, 4_000_000_000);

// ❌ Invalid: Doesn't fit case #3 because `i64` values in general are not
// exactly representable in the type `f32`.
let float: f32 = 1.0e18;
let integer: i64 = 1_000_000_000_000_000_000;
let eq: bool = float == integer;
```

Comparisons involving integer and floating-point constants are not covered by
these rules and are [discussed separately](#comparisons-with-constants).

#### Consistency with implicit conversions

We support the following [implicit conversions](implicit_conversions.md):

-   From `Int(n)` to `Int(m)` if `m > n`.
-   From `Unsigned(n)` to `Int(m)` or `Unsigned(m)` if `m > n`.
-   From `Float(n)` to `Float(m)` if `m > n`.
-   From `Int(n)` to `Float(m)` if `Float(m)` can represent all values of
    `Int(n)`.

These rules can be summarized as: a type `T` can be converted to `U` if every
value of type `T` is a value of type `U`.

Implicit conversions are also supported from certain kinds of integer and
floating-point constants to `Int(n)` and `Float(n)` types, if the constant can
be represented in the type.

All built-in comparisons can be viewed as performing implicit conversions on at
most one of the operands in order to reach a suitable pair of identical or very
similar types, and then performing a comparison on those types. The target types
for these implicit conversions are, for each suitable value `n`:

-   `Int(n)` versus `Int(n)`
-   `Unsigned(n)` versus `Unsigned(n)`
-   `Int(n)` versus `Unsigned(n)`
-   `Unsigned(n)` versus `Int(n)`
-   `Float(n)` versus `Float(n)`

There will in general be multiple combinations of implicit conversions that will
lead to one of the above forms, but we will arrive at the same result regardless
of which is selected, because all comparisons are mathematically correct and all
implicit conversions are lossless. Implementations are expected to do whatever
is most efficient: for example, for `u16 < i32` it is likely that the best
choice would be to promote the `u16` to `i32`, not `u32`.

Because we only ever convert at most one operand, we never use an intermediate
type that is larger than both input types. For example, both `i32` and `f32` can
be implicitly converted to `f64`, but we do not permit comparisons between `i32`
and `f32` even though we could perform those comparisons in `f64`. If such
comparisons were permitted, the results could be surprising:

```carbon
// `i32` can exactly represent this value.
var integer: i32 = 2_000_000_001;
// This value is within the representable range for `f32`, but will be rounded
// to 2_000_000_000.0 due to the limited precision of `f32`.
var float: f32 = 2_000_000_001.0;

// ❌ Invalid: `f32` cannot exactly represent all values of `i32`.
if (integer == float) {
  ...
}

// ✅ Valid: An explicit cast to `f64` on either side makes the code valid, but
// will compare unequal because `float` was rounded to 2_000_000_000.0
// but `integer` will convert to exactly 2_000_000_001.0.
if (integer == float as f64) {
  ...
}
if (integer as f64 == float) {
  ...
}
```

The two kinds of mixed-type comparison may be
[less efficient](/proposals/p0702.md#performance) than the other kinds due to
the slightly wider domain.

Note that this approach diverges from C++, which would convert both operands to
a common type first, sometimes performing a lossy conversion potentially giving
an incorrect result, sometimes converting both operands, and sometimes using a
wider type than either of the operand types.

#### Comparisons with constants

We permit the following comparisons involving constants:

-   A constant can be compared with a value of any type to which it can be
    implicitly converted.
-   Any two constants can be compared, even if there is no type that can
    represent both.

As described in [implicit conversions](implicit_conversions.md#data-types),
integer constants can be implicitly converted to any integer or floating-point
type that can represent their value, and floating-point constants can be
implicitly converted to any floating-point type that can represent their value.

Note that this disallows comparisons between, for example, `i32` and an integer
literal that cannot be represented in `i32`. Such comparisons would always be
tautological. This decision should be revisited if it proves problematic in
practice, for example in templated code where the literal is sometimes in range.

### Extensibility

User-defined types can extend the behavior of the comparison operators by
implementing interfaces. In this section, various properties are specified that
such implementations "should" satisfy. These properties are not enforced in
general, but the standard library might detect violations of some of them in
some circumstances. These properties may be assumed by generic code, resulting
in unexpected behavior if they are violated.

#### Equality

Comparison operators can be provided for user-defined types by implementing the
`EqWith` and `OrderedWith` interfaces.

The `EqWith` interface is used to define the semantics of the `==` and `!=`
operators for a given pair of types:

```
interface EqWith(U:! type) {
  fn Equal[self: Self](u: U) -> bool;
  default fn NotEqual[self: Self](u: U) -> bool {
    return not (self == u);
  }
}
constraint Eq {
  extends EqWith(Self);
}
```

Given `x: T` and `y: U`:

-   The expression `x == y` calls `x.(EqWith(U).Equal)(y)`.
-   The expression `x != y` calls `x.(EqWith(U).NotEqual)(y)`.

```
class Path {
  private var drive: String;
  private var path: String;
  private fn CanonicalPath[self: Self]() -> String;

  external impl as Eq {
    fn Equal[self: Self](other: Self) -> bool {
      return (self.drive, self.CanonicalPath()) ==
             (other.drive, other.CanonicalPath());
    }
  }
}
```

The `EqWith` overload is selected without considering possible implicit
conversions. To permit implicit conversions in the operands of an `==` overload,
the
[`like` operator](/docs/design/generics/details.md#like-operator-for-implicit-conversions)
can be used:

```
class MyInt {
  var value: i32;
  fn Value[self: Self]() -> i32 { return self.value; }
}
external impl i32 as ImplicitAs(MyInt);
external impl like MyInt as EqWith(like MyInt) {
  fn Equal[self: Self](other: Self) -> bool {
    return self.Value() == other.Value();
  }
}
fn CompareBothWays(a: MyInt, b: i32, c: MyInt) -> bool {
  // OK, calls above implementation three times.
  return a == a and a != b and b == c;
}
```

The behavior of `NotEqual` can be overridden separately from the behavior of
`Equal` to support cases like floating-point NaN values, where two values can
compare neither equal nor not-equal, and thus both functions would return
`false`. However, an implementation of `EqWith` should _not_ allow both `Equal`
and `NotEqual` to return `true` for the same pair of values. Additionally, these
operations should have no observable side-effects.

```
external impl like MyFloat as EqWith(like MyFloat) {
  fn Equal[self: MyFloat](other: MyFloat) -> bool {
    if (self.IsNaN() or other.IsNaN()) {
      return false;
    }
    return self.Representation() == other.Representation();
  }
  fn NotEqual[self: MyFloat](other: MyFloat) -> bool {
    if (self.IsNaN() or other.IsNaN()) {
      return false;
    }
    return self.Representation() != other.Representation();
  }
}
```

Heterogeneous comparisons must be defined both ways around:

```
external impl like MyInt as EqWith(like MyFloat);
external impl like MyFloat as EqWith(like MyInt);
```

**TODO:** Add an adapter to the standard library to make it easy to define the
reverse comparison.

#### Ordering

The `OrderedWith` interface is used to define the semantics of the `<`, `<=`,
`>`, and `>=` operators for a given pair of types.

```
choice Ordering {
  Less,
  Equivalent,
  Greater,
  Incomparable
}
interface OrderedWith(U:! type) {
  fn Compare[self: Self](u: U) -> Ordering;
  default fn Less[self: Self](u: U) -> bool {
    return self.Compare(u) == Ordering.Less;
  }
  default fn LessOrEquivalent[self: Self](u: U) -> bool {
    let c: Ordering = self.Compare(u);
    return c == Ordering.Less or c == Ordering.Equivalent;
  }
  default fn Greater[self: Self](u: U) -> bool {
    return self.Compare(u) == Ordering.Greater;
  }
  default fn GreaterOrEquivalent[self: Self](u: U) -> bool {
    let c: Ordering = self.Compare(u);
    return c == Ordering.Greater or c == Ordering.Equivalent;
  }
}
constraint Ordered {
  extends OrderedWith(Self);
}

// Ordering.Less < Ordering.Equivalent < Ordering.Greater.
// Ordering.Incomparable is incomparable with all three.
external impl Ordering as Ordered;
```

**TODO:** Revise the above when we have a concrete design for enumerated types.

Given `x: T` and `y: U`:

-   The expression `x < y` calls `x.(OrderedWith(U).Less)(y)`.
-   The expression `x <= y` calls `x.(OrderedWith(U).LessOrEquivalent)(y)`.
-   The expression `x > y` calls `x.(OrderedWith(U).Greater)(y)`.
-   The expression `x >= y` calls `x.(OrderedWith(U).GreaterOrEquivalent)(y)`.

For example:

```
class MyWidget {
  var width: i32;
  var height: i32;

  fn Size[self: Self]() -> i32 { return self.width * self.height; }

  // Widgets are normally ordered by size.
  external impl as Ordered {
    fn Compare[self: Self](other: Self) -> Ordering {
      return self.Size().(Ordered.Compare)(other.Size());
    }
  }
}
fn F(a: MyWidget, b: MyWidget) -> bool {
  return a <= b;
}
```

As for `EqWith`, the
[`like` operator](/docs/design/generics/details.md#like-operator-for-implicit-conversions)
can be used to permit implicit conversions when invoking a comparison, and
heterogeneous comparisons must be defined both ways around:

```
fn ReverseOrdering(o: Ordering) -> Ordering {
  return Ordering.Equivalent.(Ordered.Compare)(o);
}
external impl like MyInt as OrderedWith(like MyFloat);
external impl like MyFloat as OrderedWith(like MyInt) {
  fn Compare[self: Self](other: Self) -> Ordering {
    return Reverse(other.(OrderedWith(Self).Compare)(self));
  }
}
```

The default implementations of `Less`, `LessOrEquivalent`, `Greater`, and
`GreaterOrEquivalent` can be overridden if a more efficient version can be
implemented. The behaviors of such overrides should follow those of the above
default implementations, and the members of an `OrderedWith` implementation
should have no observable side-effects.

`OrderedWith` implementations should be _transitive_. That is, given `V:! type`,
`U:! OrderedWith(V)`, `T:! OrderedWith(U) & OrderedWith(V)`, `a: T`, `b: U`,
`c: V`, then:

-   If `a <= b` and `b <= c` then `a <= c`, and moreover if either `a < b` or
    `b < c` then `a < c`.
-   If `a >= b` and `b >= c` then `a >= c`, and moreover if either `a > b` or
    `b > c` then `a > c`.
-   If `a` and `b` are equivalent, then `a.Compare(c) == b.Compare(c)`.
    Similarly, if `b` and `c` are equivalent, then
    `a.Compare(b) == a.Compare(c)`.

`OrderedWith` implementations should also be _consistent under reversal_. That
is, given types `T` and `U` where `T is OrderedWith(U)` and
`U is OrderedWith(T)`, and values `a: T` and `b: U`:

-   If `a.(OrderedWith.Compare)(b)` is `Ordering.Greater`, then
    `b.(OrderedWith.Compare)(a)` is `Ordering.Less`, and the other way around.
-   Otherwise, `a.(OrderedWith.Compare)(b)` returns the same value as
    `b.(OrderedWith.Compare)(a)`.

There is no expectation that an `Ordered` implementation be a total order, a
weak order, or a partial order, and in particular the implementation for
floating-point types is none of these because NaN values do not compare less
than or equivalent to themselves.

**TODO:** The standard library should provide a way to specify that an ordering
is a weak, partial, or total ordering, and a way to request such an ordering in
a generic.

#### Compatibility of equality and ordering

There is no requirement that a pair of types that implements `OrderedWith` also
implements `EqWith`. If a pair of types does implement both, however, the
equality relation provided by `x.(EqWith.Equal)(y)` should be a refinement of
the equivalence relation provided by
`x.(OrderedWith.Compare)(y) == Ordering.Equivalent`.

#### Custom result types

**TODO:** Support a lower-level extensibility mechanism that allows a result
type other than `bool`.

### Default implementations for basic types

In addition to being defined for standard Carbon numeric types, equality and
relational comparisons are also defined for all "data" types:

-   [Tuples](../tuples.md)
-   [Struct types](../classes.md#struct-types)
-   [Classes implementing an interface that identifies them as data classes](../classes.md#interfaces-implemented-for-data-classes)

Relational comparisons for these types provide a lexicographical ordering. In
each case, the comparison is only available if it is supported by all element
types.

Because implicit conversions between data classes can reorder fields, the
implementations for data classes do not permit implicit conversions on their
arguments in general. Instead:

-   Equality comparisons are permitted between any two data classes that have
    the same _unordered set_ of field names, if each corresponding pair of
    fields has an `EqWith` implementation. Fields are compared in the order they
    appear in the left-hand operand.
-   Relational comparisons are permitted between any two data classes that have
    the same _ordered sequence_ of field names, if each corresponding pair of
    fields has an `OrderedWith` implementation. Fields are compared in order.

Comparisons between tuples permit implicit conversions for either operand, but
not both.

## Open questions

The `bool` type should be treated as a choice type, and so should support
equality comparisons and relational comparisons if and only if choice types do
in general. That decision is left to a future proposal.

## Alternatives considered

-   [Alternative symbols](/proposals/p0702.md#alternative-symbols)
-   [Chained comparisons](/proposals/p0702.md#chained-comparisons-1)
-   [Convert operands like C++](/proposals/p0702.md#convert-operands-like-c)
-   [Provide a three-way comparison operator](/proposals/p0702.md#provide-a-three-way-comparison-operator)
-   [Allow comparisons as the operand of `not`](/proposals/p0702.md#allow-comparisons-as-the-operand-of-not)
-   [Rename `OrderedWith` to `ComparableWith`](/proposals/p1178.md#use-comparablewith-instead-of-orderedwith)

## References

-   Proposal
    [#702: Comparison operators](https://github.com/carbon-language/carbon-lang/pull/702)
-   Proposal
    [#1178: Rework operator interfaces](https://github.com/carbon-language/carbon-lang/pull/1178)
-   Issue
    [#710: Default comparison for data classes](https://github.com/carbon-language/carbon-lang/issues/710)
