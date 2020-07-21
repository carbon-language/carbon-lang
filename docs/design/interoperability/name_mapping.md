# Name mapping

<!--
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

## Table of contents

<!-- toc -->

- [C/C++ names in Carbon](#cc-names-in-carbon)
- [Incomplete types](#incomplete-types)
- [Carbon names in C/C++](#carbon-names-in-cc)
- [Open questions](#open-questions)
  - [Syntax for importing C++ code](#syntax-for-importing-c-code)
  - [Provide flexibility for where C++ APIs are imported to](#provide-flexibility-for-where-c-apis-are-imported-to)
- [Alternatives](#alternatives)
  - [Providing C calls in a separate C package](#providing-c-calls-in-a-separate-c-package)
  - [Don't map incomplete types](#dont-map-incomplete-types)

<!-- tocstop -->

## C/C++ names in Carbon

C/C++ names are mapped into the `Cpp` Carbon package. C++ namespaces work the
same fundamental way as Carbon namespaces within the `Cpp` package name. Dotted
names are used when referencing these names from Carbon code. For example,
`std::exit` becomes `Cpp.std.exit`.

For example, given the C code `widget/knob.h`:

```cc
extern void turn_knob(void);
```

We would expect the calling Carbon code:

```carbon
package Widget library Knob;

import Cpp "widget/knob.h";

fn Call() {
  Cpp.turn_knob();
}
```

For example, given the C++ code `widget/knob.h`:

```cc
namespace widget {
namespace knob {
extern "C" {
void turn_knob();
}  // extern "C"
}  // namespace knob
}  // namespace widget
```

We would expect the calling Carbon code to work:

```carbon
package Widget library Knob;

import Cpp "widget/knob.h";

fn Call() {
  // Uses the C++ namespaced version.
  Cpp.widget.knob.turn_knob();
  // Uses the extern "C" version.
  Cpp.turn_knob();
}
```

## Incomplete types

C++ incomplete types will be mirrored into Carbon's incomplete type behavior.
Users wanting to avoid differences in incomplete type behaviors should fully
define the C++ types using repeated imports.

For example, given `factory.h` with a partial definition:

```cc
struct Foo;
Foo* CreateFoo();
void Process(Foo* foo);
```

And `foo.h` with a definition of Foo:

```cc
struct Foo {
  int i;
};
```

A Carbon file importing only `factory.h` will be able to access the incomplete
type:

```carbon
package FactoryUser;

import Cpp "factory.h"

var Cpp.Foo*?: x = CreateFoo();
Process(x);
```

A Carbon file importing both `factory.h` and `foo.h` will see the full type
information:

```carbon
package Factoryuser;

import Cpp "factory.h"
import Cpp "foo.h"

var Cpp.Foo*?: x = CreateFoo();
DoSomethingWith(x->i);
```

## Carbon names in C/C++

Carbon names which are mapped into C++ will use a top-level namespace of
`Carbon` by default, with the package name and namespaces represented as
namespaces below that. For example, the `Widget` Carbon package with a namespace
`Foo` would become `::Carbon::Widget::Foo` in C++.

For example, given the Carbon code:

```carbon
package Widget library Knob;

$extern("Cpp") fn Turn() { ... }
```

This may be callable from C++ through a compiler-generated `knob.6c.h`:

```cc
#include "widget/knob.6c.h"

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

```carbon
package Widget library Knob;

$extern("Cpp", namespace="::widget::knob", name="Twist") fn Turn() { ... }
```

This may be callable from C++ through a compiler-generated `knob.6c.h`:

```cc
#include "widget/knob.6c.h"

void Call() {
  ::widget::knob::Twist();
}
```

It should also be easy to extern multiple declarations as long as `name` is not
specified. For example:

```carbon
package Widget library Knob;

$extern("Cpp", namespace="::widget::knob") {
  fn Push() { ... }
  fn Rotate() { ... }
}
```

## Open questions

### Syntax for importing C++ code

Right now, these proposals apply, and use in examples,
`import Cpp "project/file.h"`. This may not be the ideal syntax, particularly as
it limits us to `Cpp` as the language identifier, versus `C++` or `c++`.

Syntaxes we could consider are:

- `import Cpp "project/file.h"`: Adds `Cpp` to the import to make it clear this
  is an interop import. The current proposed syntax.

  - Pro: Reuses the `import` keyword for similar functionality.
  - Con: Limits us to the `Cpp` identifier on imports, and implicitly also on
    exports for symmetry.

- `import("c++") "project/file.h"`: This moves the name of the language into an
  argument, allowing for more flexibility.

  - Could also do `import("c++") Cpp "project/file.h"` to make the base import
    path explicit.
  - Pro: Allows for use of flexible language identifiers.
  - Pro: Straightforward way to add more parameters to `import()`.
  - Con: Makes the `import` keyword more multi-purpose and ambiguous.
    - Could change `import` syntax to always look like
      `import("Carbon.Library")`, `import("project/file.h", lang="c++")`.

- `interop("c++", "project/file.h")`: This makes the interop import more
  function-like.
  - Pro: Allows for use of flexible language identifiers.
  - Pro: As syntax drifts further from standard Carbon `import` syntax, switches
    keywords.
  - Pro: Straightforward way to add more parameters to `interop()`.
  - Con: May not be clear that this is doing import-like behavior.
    - Could adopt instead `import interop("c++", "project/file.h")`, or
      `import library("project/file.h", lang="c++")`.

### Provide flexibility for where C++ APIs are imported to

Right now, we plan to always import C++ APIs to the top-level `Cpp` package,
with C++ namespaces appended, resulting in names like `Cpp.Namespace.API`. Users
may still `alias` these into Carbon locations.

We could instead add flexibility in where C++ APIs are imported, rewriting
either `Cpp` or the full `Cpp.Namespace`.

Pros:

- Increased user flexibility for naming.
  - Mirrors similar flexibility in `extern` syntax.

Cons:

- Users could end up importing a single C++ API under different names.
  - In other words, `Cpp.Namespace.Type` and `Carbon.RenamedNamespace.Class`
    would be different types.
  - To avoid having this be a problem, we could essentially make this be
    shorthand for an `alias` of the type in `Cpp.Namespace`; however, that would
    imply the feature is fully redundant with `alias`.
  - We could also instead use a C++ attribute, so that the remapping is
    specified with the declaration.
- While we expect to add support for arbitrary namespaces when externing Carbon
  code, that's to support migration and backwards compatibility. Doing similar
  for C++ would stand to encourage continued and new use of C++ code, a
  different value call.

## Alternatives

### Providing C calls in a separate C package

We could provide C APIs in a separate `C` package; this could be either in
addition to, or in place of, the `Cpp` package. We plan to not do this because
it's not clear there's a significant advantage to the split.

For example, given the C++ code `widget/knob.h`:

```cc
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

```carbon
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

### Don't map incomplete types

If the type is left incomplete, Carbon could instead import pointers to
incomplete types as `OpaquePointer`.

Pros:

- Avoids skew in cross-language semantics of incomplete types.

Cons:

- Forces users to provide the full list of headers for incomplete types.
  - Makes it harder to migrate code that's using C++ incomplete type semantics
    in ways that are fully compatible with Carbon.
  - May lead to unintended consequences with recursive include problems.

For example, given the above `factory.h` with a partial definition, a Carbon
file importing only `factory.h` might see `OpaquePointer`:

```carbon
package FactoryUser;

import Cpp "factory.h"

var Cpp.OpaquePointer*?: x = CreateFoo();
// This works, but is unsafe because there's effectively no type checking.
Process(x);
```

This unsafe behavior could be avoided by instead not exposing calls. That would
shift the cons, as it would also reduce the available C++ APIs further.
