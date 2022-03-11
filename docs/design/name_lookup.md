# Name lookup

<!--
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

<!-- toc -->

## Table of contents

-   [Overview](#overview)
    -   [Scopes](#scopes)
    -   [Same name](#same-name)
    -   [Unqualified name lookup](#unqualified-name-lookup)
        -   [Alternatives](#alternatives)
    -   [Name lookup for common, standard types](#name-lookup-for-common-standard-types)
-   [Open questions](#open-questions)
    -   [Shadowing](#shadowing)

<!-- tocstop -->

## Overview

Names are always introduced into some scope which defines where they can be
referenced. Many of these scopes are themselves named. Carbon has a special
facility for introducing a dedicated named scope just like C++, but we traverse
nested names in a uniform way with `.`-separated names:

```
namespace Foo;
namespace Foo.Bar;
alias Foo.Bar.MyInt = i32;

fn F(x: Foo.Bar.MyInt);
fn Foo.G(y: Bar.MyInt);
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

### Scopes

### Same name

Two _word_\s represent the _same name_ if they are identical after conversion
into
[Unicode Normalization Form C](https://unicode.org/reports/tr15/#Norm_Forms).

Two _word_\s represent _similar names_ if they are
[confusable](http://www.unicode.org/reports/tr39/#Confusable_Detection), that
is, they have the same skeleton as defined in Unicode UAX #39. Note that words
that represent the same name also represent similar names.

Any lookup into a scope looks for not only declarations of the same name being
queried, but also for similar names. If any lookup finds a name that is similar
to that being looked up, but not the same, the program is invalid.

```
// The name of this function contains a single code point:
// U+00C5 LATIN CAPITAL LETTER A WITH RING ABOVE
fn Å() {}

// The name of this function contains two code points:
// U+00E5 LATIN CAPITAL LETTER A
// U+030A COMBINING RING ABOVE
// ... which normalize in NFC to U+00C5.
// This redeclares the function declared above.
fn Å() {}

class X {
  // The name of this function contains two code points:
  // U+0410 CYRILLIC CAPITAL LETTER A
  // U+030A COMBINING RING ABOVE
  // ... which are unchanged by normalization into NFC.
  // This name is similar to that of the previous functions, but not the same.
  fn А̊() {}

  fn B() {
    // This is U+00C5. However, this call is invalid because name lookup
    // searches the scope of class `X` which contains the similar name
    // U+0410 U+030A.
    Å();
  }
}
```

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
called the "prelude" package.

Names in the prelude package will be available without scoping names. For
example, `Bool` will be the commonly used name in code, even though the
underlying type may be `Carbon::Bool`. Also, no `import` will be necessary to
use `Bool`.

## Open questions

### Shadowing

We can probably disallow the use of shadowed unqualified names, but the actual
design for such needs to be thought through.
