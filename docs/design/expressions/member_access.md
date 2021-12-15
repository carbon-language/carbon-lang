# Qualified names and member access

<!--
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

<!-- toc -->

## Table of contents

-   [Overview](#overview)
-   [Package and namespace members](#package-and-namespace-members)
-   [Lookup within values](#lookup-within-values)
    -   [Templates and generics](#templates-and-generics)
-   [Member access](#member-access)
-   [Precedence and associativity](#precedence-and-associativity)
-   [Alternatives considered](#alternatives-considered)
-   [References](#references)

<!-- tocstop -->

## Overview

A _qualified name_ is a [word](../lexical_conventions/words.md) that is preceded
by a period. The name is found within a contextually-determined entity:

-   In a member access expression, this is the entity preceding the period.
-   For a designator in a struct literal, the name is found or declared within
    the struct.

A _member access expression_ allows a member of a value, type, interface,
namespace, etc. to be accessed by specifying a qualified name for the member. A
member access expression is either a _direct_ member access expression of the
form:

> _member-access-expression_ ::= _expression_ `.` _word_

or an _indirect qualified name_ of the form:

> _member-access-expression_ ::= _expression_ `.` `(` _member-access-expression_ `)`

The meaning of a qualified name in a member access expression depends on the
first operand, which can be:

-   A [package or namespace name](#package-and-namespace-members).
-   A [value of some type](#lookup-within-values).

## Package and namespace members

If the first operand is a package or namespace name, the member access must be
direct. The _word_ must name a member of that package or namespace, and the
result is the entity with that name.

An expression that names a package or namespace can only be used as the first
operand of a member access or as the target of an `alias` declaration.

```
namespace N;
fn N.F() {}

// OK, can alias a namespace.
alias M = N;
fn G() { M.F(); }

// Error: a namespace is not a value.
let M2:! auto = N;

fn H() {
  // Error: cannot perform indirect member access into a namespace.
  N.(N.F());
}
```

## Lookup within values

When the first operand is not a package or namespace name, there are three
remaining cases we wish to support:

-   The first operand is a type-of-type, and lookup should consider members of
    that type-of-type. For example, `Addable.Add` should find the member
    function `Add` of the interface `Addable`.
-   The first operand is a type, and lookup should consider members of that
    type. For example, `i32.Least` should find the member constant `Least` of
    the type `i32`.
-   The first operand is a value, and lookup should consider members of the
    value's type; if that type is a type parameter, lookup should consider
    members of the type-of-type, treating the type parameter as an archetype.

Note that because a type is a value, and a type-of-type is a type, these cases
are overlapping and not entirely separable.

For a direct member access, the word is [looked up](#type-members) in the
following types:

-   If the first operand can be evaluated and evaluates to a type, that type.
-   If the type of the first operand can be evaluated, that type.
-   If the type of the first operand is a generic type parameter, and the type
    of that generic type parameter can be evaluated, that type-of-type.

The results of these lookups are combined. If more than one distinct entity is
found, the qualified name is invalid.

For an indirect member access, the second operand is evaluated to determine the
member being accessed.

For example:

```
class A {
  var x: i32;
}
interface I {
  fn F[me: Self]();
}
impl A as I {
  fn F[me: Self]() {}
}
fn Test(a: A) {
  // OK, `x` found in type of `a`, namely `A`.
  a.x = 1;
  // OK, `x` found in the type `A`.
  a.(A.x) = 1;

  // OK, `F` found in type of `a`, namely `A`.
  a.F();
  // OK, `F` found in the type `I`.
  a.(I.F)();
}
fn GenericTest[T: I](a: T) {
  // OK, type of `a` is the type parameter `T`;
  // `F` found in the type of `T`, namely `I`.
  a.F();
}
fn CallGenericTest(a: A) {
  GenericTest(a);
}
```

The resulting member is then [accessed](#member-access) within the value denoted
by the first operand.

### Templates and generics

If the value or type of the first operand depends on a template or generic
parameter, the lookup is performed from a context where the value of that
parameter is unknown. Evaluation of an expression involving the parameter may
still succeed, but will result in a symbolic value involving that parameter.

If the value or type depends on any template parameters, the lookup is redone
from a context where the values of those parameters are known, but where the
values of any generic parameters are still unknown. The lookup results from
these two contexts are combined, and if more than one distinct entity is found,
the qualified name is invalid.

The lookup for a member name never considers the values of any generic
parameters that are in scope at the point where the member name appears.

```
class A { fn F[me: Self]() {} }
interface I { fn F[me: Self](); }
impl A as I { fn F[me: Self]() {} }
fn UseDirect(a: A) { a.F(); }
fn UseGeneric[T:! I](a: T) { a.F(); }
fn UseTemplate[template T:! I](a: T) { a.F(); }

fn Use(a: A) {
  // Calls member of `A`.
  UseDirect(a);
  // Calls member of `impl A as I`.
  UseGeneric(a);
  // Error: ambiguous.
  UseTemplate(a);
}

class B {
  fn F[me: Self]() {}
  impl as I {
    alias F = Self.F;
  }
}
class C {
  fn F[me: Self]() {}
}
impl C as I {
  alias F = C.F;
}

fn UseBC(b: B, c: C) {
  // OK, lookup in type and in type-of-type find the same entity.
  UseTemplate(b);

  // OK, lookup in type and in type-of-type find the same entity.
  UseTemplate(c);
}
```

## Member access

A member `M` is accessed within a value `V` as follows:

-   If the member is an instance member -- a field or a method -- and was either
    named indirectly or was named directly within the type or type-of-type of
    the value, the result is:

    -   For a field member, the corresponding subobject within the value.
    -   For a method, a _bound method_, which is a value `F` such that a
        function call `F(args)` behaves the same as a call to `M(args)` with the
        `me` parameter initialized by `V`.

-   Otherwise, the member access must be direct, and the result is the member,
    but evaluating the member access expression still evaluates `V`. An
    expression that names an instance member can only be used as the second
    operand of a member access or as the target of an `alias` declaration.

```
class A {
  fn F[me: Self]();
  fn G();
  var v: i32;
  class B {};
}
fn H(a: A) {
  // OK, calls `A.F` with `me` initialized by `a`.
  a.F();

  // OK, same as above.
  var bound_f: auto = a.F;
  bound_f();

  // OK, calls `A.G`.
  A.G();
  // OK, evaluates expression `a` then calls `A.G`.
  a.G();

  // Error: name of instance member `A.v` can only be used in a
  // member access or alias.
  A.v = 1;
  // OK
  a.v = 1;

  // OK
  let T:! Type = A.B;
  // Error: value of `:!` binding is not constant because it
  // refers to local variable `a`.
  let U:! Type = a.B;
}
```

## Precedence and associativity

Member access expressions associate left-to-right:

```
class A {
  class B {
    fn F();
  }
}
interface B {
  fn F();
}
impl A as B;

fn Use(a: A) {
  // Calls member `F` of class `A.B`.
  (a.B).F();
  // Calls member `F` of interface `B`, as implemented by type `A`.
  a.(B.F)();
  // Same as `(a.B).F()`.
  a.B.F();
}
```

Member access has lower precedence than primary expressions, and higher
precedence than all other expression forms.

```
// OK, `*` has lower precedence than `.`.
var p: A.B*;
// OK, `1 + (X.Y)` not `(1 + X).Y`.
var n: i32 = 1 + X.Y;
```

## Alternatives considered

-   [...](/proposals/p0123.md#foo)

## References

-   Proposal
    [#123: title](https://github.com/carbon-language/carbon-lang/pull/123)
