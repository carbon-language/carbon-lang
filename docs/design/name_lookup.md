# Name lookup

<!--
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

## Table of contents

<!-- toc -->

- [TODO](#todo)
- [Overview](#overview)
  - [Unqualified name lookup](#unqualified-name-lookup)
    - [Alternatives](#alternatives)
  - [Name lookup common, standard types](#name-lookup-common-standard-types)
  - [Shadowing](#shadowing)
  - [Alternatives](#alternatives-1)

<!-- tocstop -->

## TODO

This is a skeletal design, added to support [the overview](README.md). It should
not be treated as accepted by the core team; rather, it is a placeholder until
we have more time to examine this detail. Please feel welcome to rewrite and
update as appropriate.

## Overview

Names are always introduced into some scope which defines where they can be
referenced. Many of these scopes are themselves named. Carbon has a special
facility for introducing a dedicated named scope just like C++, but we traverse
nested names in a uniform way with `.`-separated names:

```
namespace Foo {
  namespace Bar {
    alias ??? MyInt = Int;
  }
}

fn F(Foo.Bar.MyInt: x);
```

Carbon packages are also namespaces so to get to an imported name from the
`Abseil` package you would write `Abseil.Foo`. The "top-level" file scope is
that of the Carbon package containing the file, meaning that there is no
"global" scope. Dedicated namespaces can be reopened within a package, but there
is no way to reopen a package without being a library and file _within_ that
package.

Note that libraries (unlike packages) do **not** introduce a scope, they share
the scope of their package. This is based on the observation that in practice, a
fairly coarse scoping tends to work best, with some degree of global registry to
establish a unique package name.

### Unqualified name lookup

Unqualified name lookup in Carbon will always find a file-local result, and
other than the implicit "prelude" of importing and aliasing the fundamentals of
the standard library, there will be an explicit mention of the name in the file
that declares it in that scope.

#### Alternatives

This implies that other names within your own package but not declared within
the file must be found via the package name. It isn't clear if this is the
desirable end state. We need to consider alternatives where names from the same
library or any library in the same package are made immediately visible within
the package scope for unqualified name lookup.

### Name lookup common, standard types

The Carbon standard library is in the `Carbon` package. A very small subset of
this standard library is provided implicitly in every file's scope as-if it were
first imported and then every name in it aliased into that file's package scope.
This makes the names from this part of the standard library nearly the same as
keywords, and so it is expected to be extremely small but contain the very
fundamentals that essentially every file of Carbon code will need (`Int`,
`Bool`, etc.).

### Shadowing

Carbon also disallows the use of shadowed unqualified names, but not the
_declaration_ of shadowing names in different named scopes:

Because all unqualified name lookup is locally controlled, shadowing isn't
needed for robustness and is a long and painful source of bugs over time.
Disallowing it provides simple, predictable rules for name lookup. However, it
is important that adding names to the standard library or importing a new
package (both of which bring new names into the current package's scope) doesn't
force renaming interfaces that may have many users. To accomplish this, we allow
code to declare shadowing names, but references to that name must be qualified.
For package-scope names, this can be done with an explicit use of the current
package name: `PackageName.ShadowingName`.

```
package Foo library MyLib;

// Consider an exported function named `Shadow`.
fn Shadow();

// The package might want to import some other package named `Shadow`
// as part of its implementation, but cannot rename its exported
// `Shadow` function:
import Shadow library OtherLib;

// We can reference the imported library:
alias ??? OtherLibType = Shadow.SomeType;

// We can also reference the exported function and provide a new alias by
// using our current package name as an explicitly qualified name.
alias ??? NewShadowFunction = Foo.Shadow;
```

### Alternatives

It may make sense to restrict this further to only allowing shadowing for
exported names as internal names should be trivially renamable, and it is only
needed when the source is already changing to add a new import. Or we may want
to completely revisit the rules around shadowing.

The written approach may also have issues: e.g., is the `alias` referring to the
`Shadow` function or package? As a result, we may need to work on the approach.
