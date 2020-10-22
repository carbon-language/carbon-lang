<!--
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

# Carbon: Separate implementation of parameterized interfaces problem

## Background

Interface implementation is Carbon's only language construct that allows open
extension, and this sort of open extension is needed to address the "expression
problem" in programming language design. However, we need to limit which
libraries can implement an interface for a type so we can be guaranteed to see
the implementation when we try and use it.

## Question

Can we allow an implementation of a parameterized interface `I(T)` for a type
`A` to be in the same library as `T`, or can it only be provided with `I` or
`A`?

## Answer

It can only be provided with `I` or `A`, unless `T` can not be deduced, and then
it could be provided with `T` as well. Recall for interfaces where we want to
allow the same interface to be implemented for a type with two different
parameters, we can't deduce `T`. For example, we may want to support
`ComparableTo(Float32)` and `ComparableTo(Complex128)` for type `Complex64`. So
the argument to `ComparableTo` can't be deduced and it wouldn't break anything
to allow those implementations to be defined with the argument (`Float32` or
`Complex128`).

## Problem

Consider this collection of libraries, where there are implementations for an
interface `I(T)` for a type `A`, and those implementations are in the libraries
defining the type parameter:

```
package X library "I and A" api;

interface I(Type:$ T) { ... }

struct A { ... }
```

```
package Y library "T1" api;

import X library "I and A";

struct T1 { ... }

// Type `X.A` has an implementation for `X.I(T)` for `T == Y.T1`.
impl X.I(T1) for X.A { ... }
```

```
package Z library "T2" api;

import X library "I and A";

struct T2 { ... }

// Type `X.A` has an implementation for `X.I(T)` for `T == Z.T2`.
impl X.I(T2) for X.A { ... }
```

```
package Main api;

import X library "I and A";
// Consider what happens if we include different combinations
// of the following two statements:
// import Y library "T1";
// import Z library "T2";

// Function `F` is called with value `a` with type `U`,
// where `U` implements interface `X.I(T)` for some type `T`.
fn F[Type:$ T, X.I(T):$ U](U:$ a) { ... }

fn Main() {
  var X.A: a = X.A.Init();
  F(a);
}
```

(I have placed the libraries in different packages, but the packages are not the
important part here.)

The `F(a)` call triggers a lookup for implementations of the interface `X.I(T)`
for some `T`. There exists such implementations in both libraries `Y.T1` and
`Z.T2` for different values of `T`. This has a number of sad consequences:

-   "Import what you use" is hard to measure: libraries `Y.T1` and `Z.T2` are
    important/used even though `Y` and `Z` are not mentioned outside the
    `import` statement.
-   The call `F(a)` has different interpretations depending on what libraries
    are imported:
    -   If neither is imported, it is an error.
    -   If both are imported, it is ambiguous.
    -   If only one is imported, you get totally different code executed
        depending on which it is.
-   We have no way of enforcing a "one implementation per interface" rule that
    would prevent the call to `F` from being ambiguous.

Basically, there is nothing guaranteeing that we import libraries defining the
types that are used as interface parameters if we allow the interface parameters
to be deduced. For cases where we do _not_ allow interface parameters to be
deduced, because we want to allow multiple implementations of that interface for
a single type (such as `ComparableTo(T)`), this problem doesn't arise.

## Conclusion

It appears we need to require all implementations of interface `I(...)` for type
`A(...)` to live in the same library as either the definition of `I`, `A`, or a
parameter of `A`. Being in the same library as a parameter of `I` is
insufficient, unless it can't be deduced.
