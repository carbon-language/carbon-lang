# Templates

<!--
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

## Table of contents

<!-- toc -->

- [TODO](#todo)
- [Overview](#overview)
  - [Types with template parameters](#types-with-template-parameters)
  - [Functions with template parameters](#functions-with-template-parameters)
  - [Specialization](#specialization)
  - [Constraining templates with interfaces](#constraining-templates-with-interfaces)

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
_template_ parameters. The resulting type-function will instantiate the
parameterized definition with the provided arguments to produce a complete type
when used. Note that only the parameters marked as having this _template_
behavior are subject to full instantiation -- other parameters will be type
checked and bound early to the extent possible. For example:

```
struct Stack(Type:$$ T) {
  var Array(T): storage;

  fn Push(T: value);
  fn Pop() -> T;
}
```

This both defines a parameterized type (`Stack`) and uses one (`Array`). Within
the definition of the type, the _template_ type parameter `T` can be used in all
of the places a normal type would be used, and it will only by type checked on
instantiation.

### Functions with template parameters

Both implicit and explicit function parameters in Carbon can be marked as
_template_ parameters. When called, the arguments to these parameters trigger
instantiation of the function definition, fully type checking and resolving that
definition after substituting in the provided (or computed if implicit)
arguments. The runtime call then passes the remaining arguments to the resulting
complete definition.

```
fn Convert[Type:$$ T](T: source, Type:$$ U) -> U {
  var U: converted = source;
  return converted;
}

fn Foo(Int: i) -> Float {
  // Instantiates with the `T` implicit argument set to `Int` and the `U`
  // explicit argument set to `Float`, then calls with the runtime value `i`.
  return Convert(i, Float);
}
```

Here we deduce one type parameter and explicitly pass another. It is not
possible to explicitly pass a deduced type parameter, instead the call site
should cast or convert the argument to control the deduction. The explicit type
is passed after a runtime parameter. While this makes that type unavailable to
the declaration of _that_ runtime parameter, it still is a _template_ parameter
and available to use as a type even within the remaining parts of the function
declaration.

### Specialization

An important feature of templates in C++ is the ability to customize how they
end up specialized for specific types. Because template parameters (whether as
type parameters or function parameters) are pattern matched, we expect to
leverage pattern matching techniques to provide "better match" definitions that
are selected analogously to specializations in C++ templates. When expressed
through pattern matching, this may enable things beyond just template parameter
specialization, but that is an area that we want to explore cautiously.

### Constraining templates with interfaces

These generic interfaces also provide a mechanism to constrain fully
instantiated templates to operate in terms of a restricted and explicit API
rather than being fully duck typed. This falls out of the template type produced
by the interface declaration. A template can simply accept one of those:

```
template fn TemplateRender[Type: T](Point(T): point) {
  ...
}
```

Here, we accept the specific interface wrapper rather than the underlying `T`.
This forces the interface of `T` to match that of `Point`. It also provides only
this restricted interface to the template function.

This is designed to maximize the programmer's ability to move between different
layers of abstraction, from fully generic to a generically constrained template.
