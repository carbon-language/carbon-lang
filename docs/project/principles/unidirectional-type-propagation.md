# Principle: unidirectional type propagation

<!--
Part of the Carbon Language, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

## Table of contents

<!-- toc -->

- [Principle](#principle)
- [Detailed rationale](#detailed-rationale)
- [Caveats](#caveats)
- [Open questions](#open-questions)
- [Applications of these principles](#applications-of-these-principles)
  - [Overload selection for address of function](#overload-selection-for-address-of-function)
  - [Overload selection based on type context of use](#overload-selection-based-on-type-context-of-use)
  - [Types of literals](#types-of-literals)
  - [Designator expressions](#designator-expressions)
- [Proposals relevant to these principles](#proposals-relevant-to-these-principles)
- [References](#references)

<!-- tocstop -->

## Principle

> Type information in Carbon propagates only in one direction: from expressions
> to the context in which they appear, never in the opposite direction.

Here, "type information" should be interpreted broadly. This covers the formal
type of an expression, but also any side information that might be required to
match between an expression and the context in which it appears (such as
mutability, overflow semantics, and so on) even if those things are not modeled
as part of a type.

One intended consequence is that the interpretation of a particular syntactic
construct does not depend on the context in which it appears.

This supports the following Carbon goals:

- Code that is simple and easy to read, understand, and write
- Fast and scalable development

## Detailed rationale

The other natural and principled choice would be to perform type unification
between a type of a context and a type of an expression, allowing the context to
influence the type of the expression and vice versa. This enables some
additional expressive power (see examples below, in the "Applications of these
principles" section) but comes at some cost:

- The ease with which humans can reason about types is reduced: the locus of
  information that must be considered when determining the type of an expression
  includes its complete context, making reasoning about types more challenging.
- The language and implementation become more complex: a type inference engine
  must be built into the implementation, and its behavior must be understood by
  programmers.
- Diagnostic quality is degraded: producing high-quality, precise diagnostics
  explaining the locus of a problem in a Hindley-Milner-derived type system is
  an open research problem.
- Toolability is degraded: when an expression can have a polytype even in a
  non-generic setting, tooling is less able to describe and use properties of
  that expression, particularly in an incomplete program.
- Compiler performance suffers: solving a set of type equations is more complex
  than locally computing the type of each expression in turn. Depending on the
  type system in use, solving the type equations may be exponential-time or
  worse. (For example, Swift's type checker suffers at least exponential
  complexity in the worst-case when type checking large container literals and
  certain mixed-type arithmetic expressions; such complexity can be exhibited by
  relatively simple expressions.)
- Expressions cannot freely compute their types: any expression with an
  arbitrary type computation (as would be common when interfacing with ad-hoc
  C++ code) blocks bidirectional type inference, as it is not in general
  possible to determine the inputs to an arbitrary function that would result in
  a given type.
- Forwarding of function arguments through call wrappers can no longer be done
  in general, as the expression's type needs to be resolved in the caller where
  the eventual type context may be unknown.

## Caveats

We will need to assess the impact of this decision on common C++ idioms.

C++ overload resolution is a form of local bidirectional type inference: the
function selected depends on the type of the arguments, and the (converted) type
of the arguments depends on the function selected. This functionality will
remain available, but the mechanism by which it is obtained may change (eg,
explicit pattern matching on arguments with known types).

C++ also permits the result type of a conversion function to be deduced based on
the context in which an expression appears. Such functionality is used as the
mechanism behind libraries such as
[Boost.Assignment](https://www.boost.org/doc/libs/release/libs/assign/doc/index.html)
and a somewhat-similar Google container literal library that provide expressions
whose types appear to differ based on how they are used:

```c++
std::vector<int> vi = container_literal(1, 2, 3, 4);
std::vector<float> vf = container_literal(1, 2, 3, 4);
```

Such functionality is an edge case for this principle: no original source
expression has a type that depends on the context in which it appears, but the
semantics of an expression do depend on type information determined from its
context. However, providing such functionality is not a violation of this
principle, as every expression in the program still has a monomorphic type that
is computed based on the types of its subexpressions and not that of its
context.

## Open questions

To what extent does this principle prohibit implicit conversions? Some examples
to consider: derived-to-base conversions, user-defined implicit conversions, or
a conversion that allows the “0” literal to initialize both integer and
floating-point variables.

## Applications of these principles

### Overload selection for address of function

We will not support the mechanism in C++ whereby an overloaded function name can
be resolved to a particular overload based on the type of the destination:

```c++
void f(int);
void f(float);
void (*p)(int) = &f; // valid C++, no corresponding mechanism in Carbon
```

### Overload selection based on type context of use

In languages such as Rust and Haskell, the appropriate implementation of a trait
/ type class can be determined based on details of the point of use. Such
functionality will not be available in this form in Carbon.

For example, in Rust, the type of a matched expression can be inferred based on
the type of the cases it is matched against:

```rust
trait MatchableAs<T> {
  fn as_matchable(self) -> T;
}
enum B { True, False }
impl MatchableAs<B> for bool {
  fn as_matchable(self) -> B { if self { B::True } else { B::False } }
}
impl MatchableAs<i32> for bool {
  fn as_matchable(self) -> i32 { if self { 1 } else { 0 } }
}

match true.as_matchable() {     // the type of this expression...
  B::True => println!("True"),  // ... is determined by the type of these patterns
  B::False => println!("False"),
}

match true.as_matchable() {
  1 => println!("1"),
  0 => println!("0"),
  _ => println!("???"),
}
```

Here, the type of `true.as_matchable()` is the polytype "any `T` such that
`bool : MatchableAs<T>`", which is resolved to the monotype `Bool` or `i32`
based on the context in which that expression appears.

In Carbon, we might permit an implicit conversion from a matched expression to
the type of the cases, but we would not permit the type and interpretation of
the expression itself (for example, which `as_matchable` function is invoked) to
depend on the types of the cases.

Similarly, in Haskell, a type annotation or constraint on the result of an
expression can affect the interpretation of subexpressions of that expression:

```haskell
-- Multiplication is done in destination type, resulting in overflow
a :: Int
a = 1000000000000 * 1000000000000

-- Multiplication is done in destination type, resulting in correct answer
b :: Float
b = 1000000000000 * 1000000000000

main = do
  print a  -- prints 2003764205206896640
  print b  -- prints 1.0e24
```

In Carbon, the expression `1000000000000 * 1000000000000` can have only one
value and only one meaning. When that expression is assigned to an `Int` or to a
`Float`, it will be converted to the corresponding types, which may result in a
local difference of behavior, but the multiplication will be performed in the
same way irrespective of the destination type.

### Types of literals

As shown above in the case of Haskell, Swift numeric literals get their type
from the context, and default to `Int` for integer literals and `Double` for
floating-point literals when the context does not mandate a specific type:

```swift
var x = 10 // OK, `x` is an `Int`
var y: Int8 = 10 // OK, `10` and `y` are `Int8`

func takeInt8(_ x: Int8) { ... }
takeInt8(10) // OK, `10` is an `Int8`
takeInt8(1234567) // error, `1234567` is too big to be an `Int8`
```

The above example demonstrates an issue with simply using `Int` as the type of
the literal `10`: either we would accept implicit conversions from `Int` to
`Int8`, or reject use of the literal `10` to initialize an `Int8` object, or we
would need some non-type-based rule to allow `10` but disallow `1234567`.

There are nonetheless ways to give much the same benefits that Swift derives
from giving literals a polytype with a fallback, but without introducing
polytypes or bidirectional type inference into the Carbon type system. One
possibility would be to give a literal a type that depends on its value, such
that literals with different values have different types, with implicit
conversions provided from literal types to the types that can represent those
values. Note that there is some tension between this idea and the principle that
[built-in types are not privileged](https://github.com/carbon-language/carbon-lang/pull/57).

### Designator expressions

In Swift, enum cases can be looked up in the type specified contextually:

```swift
enum Device {
  case keyboard
  case mouse
}
var x1 = Device.keyboard // OK, `x1` is a `Device` with the value `Device.keyboard`
var x2: Device = Device.keyboard // same
var x3: Device = .keyboard // same
var x4 = .keyboard // error, don't know where to look up `.keyboard`

func takeValue(_ device: Device) // overload #1
func takeValue(_ integer: Int) // overload #2
takeDevice(.keyboard) // calls overload #1 because `Device.keyboard` exists, and `Int.keyboard' does not

func getDevice() -> Device
switch getDevice() {
case .keyboard: // `.keyboard` is looked up in the type of the scrutinee expression of the switch statement, that is, the type of the expression `getDevice`, that is, `Device`.
  ...
case .mouse:
  ...
}
```

If we support such a mechanism in Carbon, this principle would require that
`.keyboard` has a type that is not dependent on its context. Similar to the
example of literals above, we could imagine giving `.keyboard` a type that
depends on the identifier `keyboard`, but not on the context of use, and
supporting an implicit conversion from values of that type to enumerations
containing an enumerator with that name.

## Proposals relevant to these principles

- TODO

## References

- [The Hindley-Milner type system](https://en.wikipedia.org/wiki/Hindley-Milner_type_system)

- [Diagnosing Type Errors with Class (Zhang et al, 2015)](http://www.cse.psu.edu/~dbz5017/pub/pldi15.pdf)

- [Type inference and type error diagnosis for Hindley/Milner with extensions (Wazny, J. R., 2006)](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.110.5050&rep=rep1&type=pdf)
