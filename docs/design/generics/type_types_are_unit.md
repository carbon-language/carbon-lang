# Types of types are unit types

<!--
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

## Table of contents

<!-- toc -->

-   [Question](#question)
-   [Background](#background)
-   [Resolution](#resolution)

<!-- tocstop -->

## Question

Types in Carbon are values. So `Int32` is a type, but it is also a value, and so
it has its own type. What can we say about the type of `Int32`? Do any other
types have the same type?

## Background

In this example:

```
var Int32: x = 7;
```

we say `x` has type `Int32` and value `7`. When writing a function with generic
parameters, we need to talk about sets of types, such as the set of types that
are legal inputs. For example, a `Max` function may be able to take any two
values of the same type, as long as that type implements the `Comparable`
interface:

```
fn Max[Comparable:$ T](T: x, T: y) -> T;
```

In this example, `T` is the type of the inputs and outputs, and we constrain
that type to just be types that implement `Comparable` by writing that `T` has
type `Comparable`. `Int32` would be an example of a type that implements
`Comparable`, so you should be able to call `Max` with arguments of type
`Int32`.

To address
[the expression problem](https://eli.thegreenplace.net/2016/the-expression-problem-and-its-solutions),
we allow an implementation of an interface to be defined either with the type or
the interface. That is if I define a new type `Foo`, I can also define an
implementation for an existing interface `Bar` for `Foo` at the same time.
Similarly, if I define a new interface `Baz`, I can define implementations for
existing types like `Int32` of `Baz` with `Baz`. See
[the generics proposal](https://github.com/josh11b/carbon-lang/blob/generics-docs/docs/design/generics/facet-type-types.md#implementing-interfaces)
for details.

## Resolution

Let us call the type of `Int32` `X`. What set of types have `X` as their type?
Well, we know `X` is a subtype of `Comparable` since `Int32` implements
`Comparable`. This allows `T` to be `Int32` in the definition of `Max` above. In
fact, `X` is a subtype of _every_ interface `Int32` implements and _none_ of the
interfaces `Int32` doesn't implement. The problem is that we never have a
comprehensive list of all the interfaces that `Int32` implements, just which of
the interfaces that are imported into any particular file.

So imagine there is another type `T` that also has type `X`. It would have to
implement exactly the same interfaces as `Int32`. But we can create a new
interface, call it `Int32ButNotT`, in a new library. Since we are defining a new
interface, the only types that will implement that interface are those with
implementations in the same library as `Int32ButNotT`. So if we implement
`Int32ButNotT` for `Int32` but not `T`, we arrive at a contradiction. Since the
compiler won't see every interface definition, it will have to conservatively
assume that there are interfaces out there that separate any pair of types.

We conclude that every type that can have interface implementations can't share
its type with any other type.
