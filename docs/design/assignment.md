# Assignment

<!--
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

<!-- toc -->

## Table of contents

-   [Overview](#overview)
-   [Syntax](#syntax)
-   [Simple assignment semantics](#simple-assignment-semantics)
-   [Compound assignment semantics](#compound-assignment-semantics)
-   [Built-in types](#built-in-types)
-   [Tuples, structs, choice types, and data classes](#tuples-structs-choice-types-and-data-classes)
-   [Extensibility](#extensibility)
    -   [Simple assignment](#simple-assignment)
    -   [Arithmetic](#arithmetic)
    -   [Bitwise and bit-shift](#bitwise-and-bit-shift)
    -   [Defaults](#defaults)
-   [Alternatives considered](#alternatives-considered)
-   [References](#references)

<!-- tocstop -->

## Overview

Values can be assigned to variables using the `=` operator:

```
var a: i32 = 5;
a = 6;
```

For each binary [arithmetic](expressions/arithmetic.md) or
[bitwise](expressions/bitwise.md) operator `$`, a corresponding compound
assignment `$=` is provided that performs the operation in-place:

```
// Same as `a = a + 1;`
a += 1;
// Same as `a = a << 3;`
a <<= 3;
```

In addition, increment and decrement operators are provided:

```
// Same as `a = a + 1;`
++a;
// Same as `a = a - 1;`
--a;
```

These simple assignment, compound assignment, increment, and decrement operators
can only be used as complete statements, not as subexpressions of other
operators, even when parenthesized:

```
var n: i32;
// Error, assignment is not permitted as a subexpression.
if (F() and (n = GetValue()) > 5) {
}
```

User-defined types can define the meaning of these operations by
[implementing an interface](#extensibility) provided as part of the Carbon
standard library.

## Syntax

The operands of these operators can be any [expression](expressions/README.md).
However, the first operand must be modifiable because it is passed to an
`[addr self: Self*]` parameter, which disallows most expression forms other
than:

-   The name of a `var` binding.
-   A dereference of a pointer.
-   Array indexing that produces a modifiable result.
-   Member access naming a field, where the object is one of these expressions.

## Simple assignment semantics

A simple assignment statement is intended to exactly mirror the semantics of
initialization. The following two code snippets should have the same meaning if
they are both valid:

```
// Declare and initialize.
var v: T = init;
```

```
// Declare separately from initialization.
// Requires that `T` has an unformed state.
var v: T;
v = init;
```

This equivalence is not enforced, but when an object is in an unformed state,
running the assignment function is _optional_, just like running the destructor
is. If the assignment function is not run, the object will be directly
initialized from the right-hand side instead. The type is still required to
implement `AssignWith` for the assignment to be valid.

```
class C { ... }
fn F() -> C {
  returned var c: C = {...};
  // `&c` here is `&x` for the first call to `F()`.
  // `&c` here can be `&y` for the second call  to `F()`.
  return var;
}
fn G() {
  var x: C = F();
  var y: C;
  y = F();
}
```

## Compound assignment semantics

The syntax `a $= b;` is intended to be syntactic sugar for `a = a $ b;`, except
as follows:

-   A type might be able to provide a more efficient implementation for the
    compound assignment form than for the uncombined form.
-   A type might not be able to, or might not want to, provide the uncombined
    form at all, for example because creating a new instance requires additional
    resources that might not be available, such as a context object or an
    allocator.

The syntactic sugar is implemented by a [default implementation](#defaults) of
`$=` in terms of `$` and `=`.

In contrast, `++a;` and `--a;` are not simply syntactic sugar for `a = a + 1;`
and `a = a - 1;`. Instead, we interpret these operators as meaning "move to the
next value" and "move to the previous value". These operations may be available
and meaningful in cases where adding an integer is not a desirable operation,
such as for an iterator into a linked list, and may not be available in cases
where adding an integer is meaningful, such as for a type representing a
rational number.

## Built-in types

Integers and floating-point types, `bool`, and pointer types support simple
assignment with `=`. The right-hand operand is implicitly converted to the type
of the left-hand operand, and the converted value replaces the value of that
operand.

Compound assignment `$=` for integer and floating point types is
[provided automatically](#defaults) for each supported operator `$`.

For integer types, `++n;` and `--n;` behave the same as `n += 1;` and `n -= 1;`
respectively. For floating-point types, these operators are not provided.

## Tuples, structs, choice types, and data classes

_TODO_: Describe the rules for assignment in these cases.

See leads issue
[#686: Operation order in struct/class assignment/initialization](https://github.com/carbon-language/carbon-lang/issues/686)

## Extensibility

Assignment operators can be provided for user-defined types by implementing the
following families of interfaces. Implementations of these interfaces are
provided for built-in types as necessary to give the semantics described above.

### Simple assignment

```
package Core;

// Simple `=`.
interface AssignWithPrimitive
    (implicit_into U:! NoRefForm) {
  fn Op[ref self: Self](other:? U);
}
constraint AssignWith(U:! type) {
  extend require impls
      AssignWithPrimitive(form(let U));
}
constraint Assign {
  extend require impls AssignWith(Self);
}
```

The statement `x = y;` is rewritten to
`x.(AssignWithPrimitive(formof(y)).Op)(y);`.

### Arithmetic

```
// Compound `+=`.
interface AddAssignWithPrimitive
    (implicit_into U:! NoRefForm) {
  fn Op[ref self: Self](other:? U);
}
constraint AddAssignWith(U:! type) {
  extend require impls
      AddAssignWithPrimitive(form(let U));
}
constraint AddAssign {
  extend require impls AddAssignWith(Self);
}
```

```
// Compound `-=`.
interface SubAssignWithPrimitive
    (implicit_into U:! NoRefForm) {
  fn Op[ref self: Self](other:? U);
}
constraint SubAssignWith(U:! type) {
  extend require impls
      SubAssignWithPrimitive(form(let U));
}
constraint SubAssign {
  extend require impls SubAssignWith(Self);
}
```

```
// Compound `*=`.
interface MulAssignWithPrimitive
    (implicit_into U:! NoRefForm) {
  fn Op[ref self: Self](other:? U);
}
constraint MulAssignWith(U:! type) {
  extend require impls
      MulAssignWithPrimitive(form(let U));
}
constraint MulAssign {
  extend require impls MulAssignWith(Self);
}
```

```
// Compound `/=`.
interface DivAssignWithPrimitive
    (implicit_into U:! NoRefForm) {
  fn Op[ref self: Self](other:? U);
}
constraint DivAssignWith(U:! type) {
  extend require impls
      DivAssignWithPrimitive(form(let U));
}
constraint DivAssign {
  extend require impls DivAssignWith(Self);
}
```

```
// Compound `%=`.
interface ModAssignWithPrimitive
    (implicit_into U:! NoRefForm) {
  fn Op[ref self: Self](other:? U);
}
constraint ModAssignWith(U:! type) {
  extend require impls
      ModAssignWithPrimitive(form(let U));
}
constraint ModAssign {
  extend require impls ModAssignWith(Self);
}
```

```
// Increment `++`.
interface Inc { fn Op[ref self: Self](); }
// Decrement `++`.
interface Dec { fn Op[ref self: Self](); }
```

These assignment statements are rewritten as follows:

-   The statement `x += y;` is rewritten to
    `x.(AddAssignWithPrimitive(formof(y)).Op)(y);`.
-   The statement `x -= y;` is rewritten to
    `x.(SubAssignWithPrimitive(formof(y)).Op)(y);`.
-   The statement `x *= y;` is rewritten to
    `x.(MulAssignWithPrimitive(formof(y)).Op)(y);`.
-   The statement `x /= y;` is rewritten to
    `x.(DivAssignWithPrimitive(formof(y)).Op)(y);`.
-   The statement `x %= y;` is rewritten to
    `x.(ModAssignWithPrimitive(formof(y)).Op)(y);`.
-   The statement `++x;` is rewritten to `x.(Inc.Op)();`.
-   The statement `--x;` is rewritten to `x.(Dec.Op)();`.

### Bitwise and bit-shift

```
// Compound `&=`.
interface BitAndAssignWithPrimitive
    (implicit_into U:! NoRefForm) {
  fn Op[ref self: Self](other:? U);
}
constraint BitAndAssignWith(U:! type) {
  extend require impls
      BitAndAssignWithPrimitive(form(let U));
}
constraint BitAndAssign {
  extend require impls BitAndAssignWith(Self);
}
```

```
// Compound `|=`.
interface BitOrAssignWithPrimitive
    (implicit_into U:! NoRefForm) {
  fn Op[ref self: Self](other:? U);
}
constraint BitOrAssignWith(U:! type) {
  extend require impls
      BitOrAssignWithPrimitive(form(let U));
}
constraint BitOrAssign {
  extend require impls BitOrAssignWith(Self);
}
```

```
// Compound `^=`.
interface BitXorAssignWithPrimitive
    (implicit_into U:! NoRefForm) {
  fn Op[ref self: Self](other:? U);
}
constraint BitXorAssignWith(U:! type) {
  extend require impls
      BitXorAssignWithPrimitive(form(let U));
}
constraint BitXorAssign {
  extend require impls BitXorAssignWith(Self);
}
```

```
// Compound `<<=`.
interface LeftShiftAssignWithPrimitive
    (implicit_into U:! NoRefForm) {
  fn Op[ref self: Self](other:? U);
}
constraint LeftShiftAssignWith(U:! type) {
  extend require impls
      LeftShiftAssignWithPrimitive(form(let U));
}
constraint LeftShiftAssign {
  extend require impls LeftShiftAssignWith(Self);
}
```

```
// Compound `>>=`.
interface RightShiftAssignWithPrimitive
    (implicit_into U:! NoRefForm) {
  fn Op[ref self: Self](other:? U);
}
constraint RightShiftAssignWith(U:! type) {
  extend require impls
      RightShiftAssignWithPrimitive(form(let U));
}
constraint RightShiftAssign {
  extend require impls
    RightShiftAssignWith(Self);
}
```

These assignment statements are rewritten as follows:

-   The statement `x &= y;` is rewritten to
    `x.(BitAndAssignWithPrimitive(formof(y)).Op)(y);`.
-   The statement `x |= y;` is rewritten to
    `x.(BitOrAssignWithPrimitive(formof(y)).Op)(y);`.
-   The statement `x ^= y;` is rewritten to
    `x.(BitXorAssignWithPrimitive(formof(y)).Op)(y);`.
-   The statement `x <<= y;` is rewritten to
    `x.(LeftShiftAssignWithPrimitive(formof(y)).Op)(y);`.
-   The statement `x >>= y;` is rewritten to
    `x.(RightShiftAssignWithPrimitive(formof(y)).Op)(y)`;.

Implementations of these interfaces are provided for built-in types as necessary
to give the semantics described above.

### Defaults

When a type provides both an assignment and a binary operator `$`, so that
`a = a $ b;` is valid, Carbon provides a default `$=` implementation so that
`a $= b;` is valid and has the same meaning as `a = a $ b;`.

This defaulting is accomplished by a parameterized implementation of
`OpAssignWith(U)` defined in terms of `AssignWith` and `OpWith`:

```
impl forall [U:! type, T:! OpWith(U) where .Self impls AssignWith(.Self.Result)]
    T as OpAssignWith(U) {
  fn Op[ref self: Self](other: U) {
    // Here, `$` is the operator described by `OpWith`.
    self = self $ other;
  }
}
```

If a more efficient form of compound assignment is possible for a type, a more
specific `impl` can be provided:

```
impl anchor MyString as AddWith(anchor MyString) {
  // Allocate new memory and perform addition.
}

impl MyString as AddAssignWith(MyString) {
  // Reuse existing storage where possible.
}
```

## Alternatives considered

-   [Allow assignment as a subexpression](/proposals/p2511.md#allow-assignment-as-a-subexpression)
-   [Allow chained assignment](/proposals/p2511.md#allow-chained-assignment)
-   [Do not provide increment and decrement](/proposals/p2511.md#do-not-provide-increment-and-decrement)
-   [Treat increment as syntactic sugar for adding `1`](/proposals/p2511.md#treat-increment-as-syntactic-sugar-for-adding-1)
-   [Define `$` in terms of `$=`](/proposals/p2511.md#define--in-terms-of-)
-   [Do not allow overloading the behavior of `=`](/proposals/p2511.md#do-not-allow-overloading-the-behavior-of-)
-   [Treat the left hand side of `=` as a pattern](/proposals/p2511.md#treat-the-left-hand-side-of--as-a-pattern)
-   [Different names for interfaces](/proposals/p2511.md#different-names-for-interfaces)

## References

-   Leads issue
    [#451: Do we want variable-arity operators?](https://github.com/carbon-language/carbon-lang/issues/451)
-   Proposal
    [#257: Initialization of memory and variables](https://github.com/carbon-language/carbon-lang/pull/257)
-   Proposal
    [#1083: Arithmetic](https://github.com/carbon-language/carbon-lang/pull/1083)
-   Proposal
    [#1178: Rework operator interfaces](https://github.com/carbon-language/carbon-lang/pull/1178)
-   Proposal
    [#1191: Bitwise and shift operators](https://github.com/carbon-language/carbon-lang/pull/1191)
-   Proposal
    [#2511: Assignment statements](https://github.com/carbon-language/carbon-lang/pull/2511)
-   Proposal
    [#5389: Generic across forms](https://github.com/carbon-language/carbon-lang/pull/5389)
