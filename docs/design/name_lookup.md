# Name lookup

<!--
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

<!-- toc -->

## Table of contents

-   [TODO](#todo)
-   [Overview](#overview)
    -   [Unqualified name lookup](#unqualified-name-lookup)
        -   [Alternatives](#alternatives)
    -   [Name lookup for common, standard types](#name-lookup-for-common-standard-types)
-   [Open questions](#open-questions)
    -   [Shadowing](#shadowing)

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

fn F(x: Foo.Bar.MyInt);
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

Unqualified name lookup in Carbon will always find a file-local result, other
than the implicit "prelude" of importing and aliasing the fundamentals of the
standard library. There will be an explicit mention of the name in the file that
declares the name in the current or enclosing scope, which must also precede the
reference.

#### Alternatives

This implies that other names within your own package but not declared within
the file must be found by way of the package name. It isn't clear if this is the
desirable end state. We need to consider alternatives where names from the same
library or any library in the same package are made immediately visible within
the package scope for unqualified name lookup.

### Name lookup for common, standard types

The Carbon standard library is in the `Carbon` package. A very small subset of
this standard library is provided implicitly in every file's scope. This is
called the "prelude".

Names in the prelude will be available without a package qualifier. For example,
the name `Type` can be directly used in code without a `Carbon.` qualifier, even
though it belongs to the `Carbon` package, and no import is necessary to use the
name `Type`.

## Open questions

### Shadowing

We can probably disallow the use of shadowed unqualified names, but the actual
design for such needs to be thought through.
