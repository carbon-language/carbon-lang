# Templates

<!--
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

<!-- toc -->

## Table of contents

-   [TODO](#todo)
-   [Overview](#overview)
    -   [Types with template parameters](#types-with-template-parameters)
    -   [Functions with template parameters](#functions-with-template-parameters)
    -   [Overloading](#overloading)
    -   [Constraining templates with interfaces](#constraining-templates-with-interfaces)

<!-- tocstop -->

## TODO

This is a skeletal design, added to support [the overview](README.md). It should
not be treated as accepted by the core team; rather, it is a placeholder until
we have more time to examine this detail. Please feel welcome to rewrite and
update as appropriate.

## Overview

Carbon templates follow the same fundamental paradigm as C++ templates: they are
instantiated, resulting in late type checking, duck typing, and lazy binding.
They both enable interoperability between Carbon and C++ and address some
(hopefully limited) use cases where the type checking rigor imposed by generics
isn't helpful.

### Types with template parameters

When parameterizing a user-defined type, the parameters can be marked as
template parameters. The resulting type-function will instantiate the
parameterized definition with the provided arguments to produce a complete type
when used. Note that only the parameters marked as having this template behavior
are subject to full instantiation -- other parameters will be type checked and
bound early to the extent possible. For example:

```
class Stack(template T:! type) {
  var storage: Array(T);

  fn Push[addr self: Self*](value: T);
  fn Pop[addr self: Self*]() -> T;
}
```

This both defines a parameterized type (`Stack`) and uses one (`Array`). Within
the definition of the type, the template type parameter `T` can be used in all
of the places a normal type would be used, and it will only by type checked on
instantiation.

### Functions with template parameters

Both deduced and explicit function parameters in Carbon can be marked as
template parameters. When called, the arguments to these parameters trigger
instantiation of the function definition, fully type checking and resolving that
definition after substituting in the provided (or computed if deduced)
arguments. The runtime call then passes the remaining arguments to the resulting
complete definition.

```
fn Convert[template T:! type](source: T, template U:! type) -> U {
  var converted: U = source;
  return converted;
}

fn Foo(i: i32) -> f32 {
  // Instantiates with the `T` deduced argument set to `i32` and the `U`
  // explicit argument set to `f32`, then calls with the runtime value `i`.
  return Convert(i, f32);
}
```

Here we deduce one type parameter and explicitly pass another. It is not
possible to explicitly pass a deduced type parameter, instead the call site
should cast or convert the argument to control the deduction. The explicit type
is passed after a runtime parameter. While this makes that type unavailable to
the declaration of _that_ runtime parameter, it still is a template parameter
and available to use as a type even within the remaining parts of the function
declaration.

### Overloading

An important feature of templates in C++ is the ability to customize how they
end up specialized for specific types. Because template parameters (whether as
type parameters or function parameters) are pattern matched, we expect to
leverage pattern matching techniques to provide "better match" definitions that
are selected analogously to specializations in C++ templates. When expressed
through pattern matching, this may enable things beyond just template parameter
specialization, but that is an area that we want to explore cautiously.

### Constraining templates with interfaces

Because we consider only specific _parameters_ to be templated and they could be
individually migrated to a constrained interface using the
[generics system](README.md#generics), constraining templates themselves may be
less critical. Instead, we expect parameterized types and functions may use a
mixture of generic parameters and templated parameters based on where they are
constrained.

However, if there are still use cases, we would like to explore applying the
interface constraints of the generics system directly to template parameters
rather than create a new constraint system.
