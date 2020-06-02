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
- [Using Carbon generics from C++](#using-carbon-generics-from-c)
- [Alternatives](#alternatives)
  - [Require bridge code](#require-bridge-code)
  - [Translate C++ template code to Carbon](#translate-c-template-code-to-carbon)

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
Carbon interfaces,
[bridge code will be required](user_defined_types.md#implementing-carbon-interfaces-in-c).

## Using Carbon templates from C++

We plan to modify Clang to allow for extensions that will use Carbon to compile
the template then insert the results into Clang's AST for expansion.

This assumes low-level modifications to LLVM. We acknowledge this would be
necessary, and may gate such a feature.

## Using Carbon generics from C++

Carbon generics will require bridge code that hides the generic. This bridge
code may be written using a Carbon template, changing compatibility constraints
to match.

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

## Alternatives

### Require bridge code

Instead of using Clang extensions to support Carbon templates, we could require
bridge code that explicitly instantiates versions of the template for use with
C++ types.

Pros:

- Avoids modifications to Clang.

Cons:

- Requires extra code to use templates from C++, making it harder to migrate
  code to Carbon.

### Translate C++ template code to Carbon

Instead of requiring Carbon types be externed for use with C++ templates, we
could instead translate the implementation of a C++ template to Carbon, then
compile it as normal.

This would rely on translation support being written for C/C++, and additionally
that it support translating partial files.

Pros:

- Builds on existing plans to build a translation tool, creating more consistent
  behavior.

Cons:

- Makes the translation tool a compilation dependency.
  - Creates new feature requirements for the translation tool.
- Limits compatibility to what the translation tool supports.
  - C++ templates may be particularly likely to use edge-case language features.
  - Translation errors may be difficult to catch.

The use of
$extern should be low cost, and the risks of relying on the
translation tool could be high, and so we expect to use $extern.
It will always be possible for a project to run the translation tool themselves
if it is preferred by a code owner.
