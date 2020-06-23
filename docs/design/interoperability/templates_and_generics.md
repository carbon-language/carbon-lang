# Templates and generics

<!--
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

## Table of contents

<!-- toc -->

- [C++ templates](#c-templates)
- [Using Carbon generics/templates with C++ types in Carbon code](#using-carbon-genericstemplates-with-c-types-in-carbon-code)
- [Using Carbon templates from C++](#using-carbon-templates-from-c)
  - [Alternative: Require bridge code](#alternative-require-bridge-code)
- [Using Carbon generics from C++](#using-carbon-generics-from-c)

<!-- tocstop -->

## C++ templates

Simple C++ class templates are directly made available as Carbon templates. For
example, ignoring allocators and their associated complexity, `std::vector<T>`
in C++ would be available as `Cpp.std.vector(T)` in Carbon.

Instantiating C++ templates with a Carbon type requires that type to be made
available to C++ code and the instantiation occurs against the C++ interface to
that Carbon type. For example:

```carbon
package Art;

import Cpp "<vector>";

// Extern the interface so that template code can see it:
$extern("Cpp") interface Shape {
  fn Draw();
}

// Then instantiate the template:
var Cpp.std.vector(Shape): shapes;
```

More complex C++ class templates may need to be explicitly instantiated using
bridge C++ code to explicitly provide Carbon types visible within C++ to the
appropriate C++ template parameters. The key principle is that C++ templates are
instantiated within C++ against a C++-visible API for a given Carbon type.

## Using Carbon generics/templates with C++ types in Carbon code

Any C++ type can be used as a type parameter in Carbon. However, it will be
interpreted as Carbon code; for example, if there are any requirements for
Carbon interfaces, [bridge code will be required](#bookmark=kix.8fx2t4lplthb).

## Using Carbon templates from C++

We plan to modify Clang to allow for extensions that will use Carbon to compile
the template then insert the results into Clang's AST for expansion.

This assumes low-level modifications to LLVM. We acknowledge this would be
necessary, and may gate such a feature.

### Alternative: Require bridge code

We could require bridge code that explicitly instantiates versions of the
template for use with C++ types.

Pros:

- Avoids modifications to Clang.

Cons:

- Requires extra code to use templates from C++, making it harder to migrate
  code to Carbon.

## Using Carbon generics from C++

Using Carbon generics from C++ code will require bridge Carbon code that hides
the generic. Note this could be wrapping the generic with a template.

For example, given the Carbon code:

```carbon
fn GenericAPI[Foo:$ T](T*: x) { ... }

fn TemplateAPI[Foo:$$ T](T* x) { GenericAPI(x); }
```

We could have C++ code that uses the template wrapper to use the generic:

```cc
CppType y;
::Carbon::TemplateAPI(&y);
```
