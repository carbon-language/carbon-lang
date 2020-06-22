# Name mapping

<!--
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

## Table of contents

<!-- toc -->

- [C/C++ names in Carbon](#cc-names-in-carbon)
  - [Alternative: Providing C calls in a separate C package](#alternative-providing-c-calls-in-a-separate-c-package)
  - [Defining types across multiple imports](#defining-types-across-multiple-imports)
- [Incomplete types](#incomplete-types)
  - [Alternative: Have incomplete types in the Cpp package behave slightly differently.](#alternative-have-incomplete-types-in-the-cpp-package-behave-slightly-differently)
  - [Alternative: Don't map incomplete types](#alternative-dont-map-incomplete-types)
- [Carbon names in C/C++](#carbon-names-in-cc)

<!-- tocstop -->

## C/C++ names in Carbon

C/C++ names are mapped into the `Cpp` Carbon package. C++ namespaces work the
same fundamental way as Carbon namespaces within the `Cpp` package name. Dotted
names are used when referencing these names from Carbon code. For example,
`std::exit` becomes `Cpp.std.exit`.

For example, given the C code `widget/knob.h`:

```
extern void turn_knob(void);
```

We would expect the calling Carbon code:

```
package Widget library Knob;

import Cpp "widget/knob.h";

fn Call() {
  Cpp.turn_knob();
}
```

For example, given the C++ code `widget/knob.h`:

```
namespace widget {
namespace knob {
extern "C" {
void turn_knob();
}  // extern "C"
}  // namespace knob
}  // namespace widget
```

We would expect the calling Carbon code to work:

```
package Widget library Knob;

import Cpp "widget/knob.h";

fn Call() {
  // Uses the C++ namespaced version.
  Cpp.widget.knob.turn_knob();
  // Uses the extern "C" version.
  Cpp.turn_knob();
}
```

### Alternative: Providing C calls in a separate C package

We could provide C APIs in a separate `C` package (either in addition to, or in
place of, the `Cpp` package). We plan to not do this because it's not clear
there's a significant advantage to the split.

For example, given the C++ code `widget/knob.h`:

```
namespace widget {
namespace knob {
extern "C" {
void turn_knob();
}  // extern "C"
}  // namespace knob
}  // namespace widget
```

We would expect the calling Carbon code to work, using the `C` package for the C
extern:

```
package Widget library Knob;

import Cpp "widget/knob.h";

fn Call() {
  Cpp.widget.knob.turn_knob();
  C.turn_knob();
}
```

Pros:

- Makes it clear when C vs C++ code is being invoked.
- Allows for language-specific name lookup rules within a given package space.

Cons:

- Extra overhead for engineers to think about which language a given call is
  coming from.
- If C APIs are in both `C` and `Cpp` packages, creates a divergent syntax for
  equivalent calls.

### Defining types across multiple imports

Carbon will allow C/C++ imports to fill in type definitions of other C/C++
imports as if they were all #included together. Due to this, the order of C/C++
imports will matter.

For example, if one header defines a template and another header defines
specializations of that template, they may both be included in order for Carbon
code to get a fully correct view of the template.

In other words, this is a meaningful combination of imports:

```
import Cpp "template.h"
import Cpp "template-specializations.h"
```

## Incomplete types

C++ incomplete types will be mirrored into Carbon's incomplete type behavior.
Users wanting to avoid differences in incomplete type behaviors should fully
define the C++ types using repeated imports.

For example, given `factory.h` with a partial definition:

```
struct Foo;
Foo* CreateFoo();
void Process(Foo* foo);
```

And `foo.h` with a definition of Foo:

```
struct Foo {
  int i;
};
```

A Carbon file importing only `factory.h` will be able to access the incomplete
type:

```
package FactoryUser;

import Cpp "factory.h"

var Cpp.Foo*?: x = CreateFoo();
Process(x);
```

A Carbon file importing both `factory.h` and `foo.h` will see the full type
information:

```
package Factoryuser;

import Cpp "factory.h"
import Cpp "foo.h"

var Cpp.Foo*?: x = CreateFoo();
DoSomethingWith(x->i);
```

### Alternative: Have incomplete types in the Cpp package behave slightly differently.

It's possible that the C++ semantics of incomplete types may differ from the
Carbon semantics. To address this, we could work to offer slightly different
behavior for incomplete types when dealing with the `Cpp` package.

Pros:

- Maintains consistent use of C++ APIs.

Cons:

- Creates an inconsistency in how the `Cpp` package functions.

### Alternative: Don't map incomplete types

If the type is left incomplete, Carbon could instead import pointers to
incomplete types as `OpaquePointer`.

Pros:

- Avoids skew in cross-language semantics of incomplete types.

Cons:

- Forces users to provide the full list of headers for incomplete types.
  - Makes it harder to migrate code that's using C++ incomplete type semantics
    in ways that's fully compatible with Carbon.
  - May lead to unintended consequences with recursive include problems.

For example, given the above `factory.h` with a partial definition, a Carbon
file importing only `factory.h` will see `OpaquePointer`:

```
package FactoryUser;

import Cpp "factory.h"

var Cpp.OpaquePointer*?: x = CreateFoo();
// It is unsafe for this code to call Process(x) because the type is opaque.
```

## Carbon names in C/C++

Carbon names which are mapped into C++ will use a top-level namespace of
`Carbon`, with the package name and namespaces represented as namespaces below
that. For example, the `Widget` Carbon package with a namespace `Foo` would
become `::Carbon::Widget::Foo` in C++.

For example, given the Carbon code:

```
package Widget library Knob;

$extern("Cpp") fn Turn() { ... }
```

This may be callable from C++ with (including a compiler-generated
`knob.carbon.h`):

```
#include "widget/knob.carbon.h"

void Call() {
  ::Carbon::Widget::Knob::Turn();
}
```

Users will be allowed to override the default namespace and name mapping in
order to support migration of C++ code. The intent is that it should be easy to
migrate users from C++ to Carbon, without needing to touch all call sites. C++
has limited support for aliasing, so we prefer to support this from the Carbon
side.

For example, given the Carbon code:

```
package Widget library Knob;

$extern("Cpp", namespace="::widget::knob", name="Twist") fn Turn() { ... }
```

This may be callable from C++ with (including a compiler-generated
`knob.carbon.h`):

```
#include "widget/knob.carbon.h"

void Call() {
  ::widget::knob::Twist();
}
```

It should also be easy to extern multiple declarations as long as `name` is not
specified. For example:

```
package Widget library Knob;

$extern("Cpp", namespace="::widget::knob") {
  fn Push() { ... }
  fn Rotate() { ... }
}
```
