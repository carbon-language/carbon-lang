# Code and name organization

<!--
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

## Table of contents

<!-- toc -->

-   [Goals and philosophy](#goals-and-philosophy)
-   [Overview](#overview)
    -   [Name paths](#name-paths)
    -   [Named scopes](#named-scopes)
    -   [Imports](#imports)
    -   [File extensions](#file-extensions)
-   [Details](#details)
    -   [Name paths](#name-paths-1)
        -   [Disallowing name path conflicts](#disallowing-name-path-conflicts)
    -   [Package keyword](#package-keyword)
    -   [Libraries](#libraries)
    -   [ORIGINAL libraries](#original-libraries)
    -   [Namespaces](#namespaces)
        -   [Using imported namespaces](#using-imported-namespaces)
        -   [Aliasing](#aliasing)
    -   [Imports](#imports-1)
        -   [Name conflicts of imports](#name-conflicts-of-imports)
-   [Alternatives](#alternatives)
    -   [Aliasing namespace names](#aliasing-namespace-names)
    -   [Allow shadowing of names](#allow-shadowing-of-names)
    -   [Broader imports, either all names or arbitrary code](#broader-imports-either-all-names-or-arbitrary-code)
    -   [Different file extensions](#different-file-extensions)
    -   [Imports from URLs](#imports-from-urls)
    -   [Prevent libraries from crossing package boundaries](#prevent-libraries-from-crossing-package-boundaries)
    -   [Scoped namespaces](#scoped-namespaces)
    -   [Strict association between the filesystem path and library/namespace](#strict-association-between-the-filesystem-path-and-librarynamespace)

<!-- tocstop -->

## Goals and philosophy

Important Carbon goals for code and name organization are:

-   [Software and language evolution](/docs/project/goals.md#software-and-language-evolution):

    -   We should support API libraries adding new structs, functions or other
        identifiers without those new identifiers being able to shadow or break
        existing users that already have identifiers with conflicting names.

    -   We should make it as easy as possible to refactor code between files.

-   [Fast and scalable development](/docs/project/goals.md#fast-and-scalable-development):

    -   It should be easy for IDEs and other tooling to parse code, without
        needing to parse imports for context.

    -   Structure should be provided for large projects to opt into features
        which will help scale features, while not adding burdens to small
        projects that don't need it.

## Overview

### Name paths

A name path is the dot-separated identifier list that indicates the full path of
a name as it corresponds to any named scope. For example, while both `Geometry`
and `Shapes` are names of identifiers, `Geometry.Shapes` is a name path
indicating `Shapes` is an identifier in the `Geometry` named scope.

### Named scopes

Carbon has two categories of named scopes used by code and name organization:

-   **Library** scopes, used at compile-time to determine which library
    dependencies to include. These scopes enable separate compilation of
    dependencies.

-   **Namespace** scopes, used by [name lookup](name_lookup.md) to choose which
    entity to use for a given piece of code.

Every **file** must specify both library and namespace scopes that will be used
for all APIs that it contains. The names of these scopes may differ. For
example, `Geometry.TwoDimensional` may be the library scope, while
`Geometry.Shapes` may be the namespace scope.

The `package` keyword is required for every file. It contributes a common name
path to both the library and namespace scope. Additional scoping is optional,
although a file may only contribute to a single library, even if its names are
in multiple namespaces.

For example, to set use `Geometry` as both the library scope and namespace
scope, a file might contain:

```carbon
package Geometry;

struct Point { ... }
```

If distinct scope name paths are desired, such as a `Geometry.TwoDimensional`
library scope and `Geometry.Shapes` namespace scope, a file might contain:

```carbon
package Geometry library TwoDimensional namespace Shapes;

struct Circle { ... }
```

The `namespace` keyword allows specifying additional namespace scopes within a
file. For example, this is an alternate way to declare `Geometry.Shapes.Circle`:

```carbon
package Geometry;

namespace Shapes;
struct Shapes.Circle { ... }
```

### Imports

The `import` keyword supports reusing code from other files and libraries.

For example, to import the `Geometry.Shapes` namespace:

```carbon
import("Geometry", "Shapes");

fn Area(Shapes.Circle circle) { ... };
```

Specific identifiers may also be imported, for example:

```carbon
import("Geometry.Shapes", "Circle");

fn Area(Circle circle) { ... };
```

### File extensions

Carbon files use the `.6c` extension as a matter of convention. This comes from
how the atomic number notation for
[Carbon](https://en.wikipedia.org/wiki/Carbon), "<sub>6</sub>C". In the
language's case, we need a reasonably unique extension, and this is what we've
chosen.

## Details

### Name paths

[Name paths](#name-paths) are defined above as dot-separated identifiers. This
syntax can be expressed as a rough regular expression:

```regex
IDENTIFIER(\.IDENTIFIER)*
```

#### Disallowing name path conflicts

Carbon will disallow name conflicts, even those that may be treated as
[shadowing](https://en.wikipedia.org/wiki/Variable_shadowing) in other
languages. In Carbon, a name conflict can be considered as any case where two
identically named entities can see each other; that is, if they are in either in
the same scope, or one is at a parent scope of the other.

For example, all cases of `Triangle` shadow each other:

```
package Example;

import("Geometry.Shapes", "Triangle");
import("Music.Instrument", "Triangle");

namespace Triangle { ... }
fn Triangle() { ... }
struct Triangle { ... }

fn Foo() { var Int: Triangle = 3; }
fn Bar(var Int: Triangle) { ... }

namespace Baz {
  fn Triangle() { ... }
}
```

Rather than trying to resolve shadowing, Carbon will reject code until there is
only _one_ possible result for a `Triangle` lookup for any given scope. Renaming
and [aliasing](aliases.md) are standard solutions to avoid this problem.

Names in scopes that do _not_ have a parent-child relationship will not result
in a name conflict. In this example, `Foo` is not shadowed because because the
name paths are distinct:

```
package Example;

namespace Bar {
  fn Foo() { ... }
}

namespace Baz {
  fn Foo() { ... }
}
```

### Package keyword

The first non-comment, non-whitespace line of a Carbon file will be the
`package` keyword. The `package` keyword's syntax, combined with the optional
`library` keyword, may be expressed as a rough regular expression:

```regex
package NAME_PATH (library NAME_PATH)? (namespace NAME_PATH)?;
```

For example:

```carbon
package Geometry library Objects.Flat namespace Shapes
```

Breaking this apart:

-   The first name passed to the `package` keyword, `Geometry`, is a name path
    prefix that will be used for both the file's library and namespace scopes.
-   When the optional `library` keyword is specified, its name path is combined
    with the package to generate the library scope. In this example, the
    `Geometry.Objects.Flat` library will be used.
-   When the optional `namespace` keyword is specified, its name path is
    combined with the package to generate the namespace scope. In this example,
    the `Geometry.Shapes` namespace will be used.

It's possible that files contributing to the `Geometry.Objects.Flat` may use
different `package` arguments. These examples vary only on the resulting
namespace, which will use only the `package` name path:

```carbon
package Geometry.Objects.Flat;
package Geometry.Objects library Shapes;
package Geometry library Objects.Flat;
```

Because the `package` keyword must be specified in all files, there are a couple
important and deliberate side-effects:

-   Every file will be in precisely one library.
-   Every name in Carbon will be in a namespace due to the `package`. There is
    no "global" namespace.
    -   Names within a file may have additional namespaces specified,
        [as detailed below](#namespaces).

### Libraries

Every Carbon library consists of one or more files.

### ORIGINAL libraries

This approach for libraries would be more inspired by C++ modules.

Every Carbon library has a primary file that defines its exported interface.
This file must start off with the canonical form of the top-level declaration:

```
package Geometry library Shapes;
```

The library may contain additional files as part of its implementation. These
all must start with a special top-level declaration:

```
package Geometry library Shapes impl;
```

In C++, this approach is taken to address the problems of recursive includes.
Separating the interface and implementation allows for less evaluation of

Within a library, files can import other files from that library using a special
syntax (see below). Every file in a library which exports an interface to users
of the library must be transitively imported into the primary interface file and
only those files transitively imported are allowed to export an interface. Files
which are not transitively imported into the primary interface file (and thus
also do not export any interfaces) are allowed to import the primary file of the
library in addition to importing other implementation files. Imports must always
form a directed acyclic graph (DAG), including these file-based imports.

    **Open question:** We may find that it is too burdensome to insist on the imports within a library forming a DAG and/or allow cyclic references within files and libraries. Relaxing this would add complexity to the compilation (no trivial parallel or incremental compiles between files within a library) but allow a simpler intra-library import model where we simply list the files providing exported interfaces in the main file. Then the interface is just the concatenation of those files, and the implementation is the further concatenation of the rest of the files. We're starting with the more restrictive model in part because it seems easier to relax this later if desired.

The result is that the set of files which must be examined to find the complete
exported interface of a library is the transitive closure of imported files
starting from the primary interface file.

### Namespaces

Namespaces offer named scopes for names. Namespaces may be nested. Multiple
libraries may contribute to the same namespace. In practice, packages may have
namespaces such as `Testing` containing names that benefit from an isolated
space but are present in many libraries.

Syntax for the `namespace` keyword may loosely be expressed as a regular
expression:

```regex
namespace NAME_PATH;
```

A namespace is used by first declaring it, then using it when declaring a name.
For example:

```
namespace Foo.Bar;
struct Foo.Bar.Baz { ... }

fn Wiz(Foo.Bar.Baz x);
```

Namespaces declared and added to within a file must always be children of the
file-level namespace. For example, this declares `Geometry.Shapes.Triangle`:

```
package Geometry;

namespace Shapes;

struct Shapes.Triangle { ... }
```

#### Using imported namespaces

If a namespace is imported, it may also be used to declare new items, so long as
it's a child of the file-level namespace.

For example, this code declares `Geometry.Shapes.Triangle`:

```
package Geometry;

import("Geometry", "Shapes");

struct Shapes.Triangle { ... }
```

On the other hand, this is invalid code because the `Instruments` namespace is
not under `Geometry`, and so `Triangle` cannot be added to it from this file:

```
package Geometry;

import("Music", "Instruments");

struct Instruments.Triangle { ... }
```

This is also invalid code because the `Geometry.Volumes` namespace is a sibling
of the `Geometry.Areas` namespace used for this file, and so `Sphere` cannot be
added to it from this file:

```
package Geometry namespace Areas;

import("Geometry", "Volumes");

struct Circle { ... }

struct Volumes.Sphere { ... }
```

However, with a higher-level file namespace of `Geometry`, a single file could
still add to both `Volumes` and `Areas`:

```
package Geometry;

import("Geometry", "Volumes");

namespace Areas;
struct Areas.Circle { ... }

struct Volumes.Sphere { ... }
```

#### Aliasing

Carbon's [alias keyword](aliases.md) will support aliasing namespaces. For
example, this would be valid code to complement the above example:

```
alias FB = Foo.Bar;

fn WizAlias(FB.Baz x);
```

### Imports

The `import` keyword supports reusing code from other files and libraries. All
imports for a file must have only whitespace and comments between the `package`
declaration and them. No other code can be interleaved.

One or more names may be imported with a single statement. When multiple names
are being imported, they should be specified using a [tuple](tuples.md). For
example:

```carbon
import("Geometry.Shapes", "Circle");
import("Geometry.Shapes", ("Circle", "Triangle", "Square"));
```

When importing from another file or library that is in the current namespace
scope, the namespace may be omitted from the import. For example, an import
`Geometry.Point` may look like:

```carbon
package Geometry;

import("Point");
```

#### Name conflicts of imports

It's expected that multiple libraries will end up exporting identical names.
Importing them would result in
[shadowing](https://en.wikipedia.org/wiki/Variable_shadowing), which Carbon
treats as a conflict. For example, this would be a rejected name conflict
because it redefines `Triangle`:

```carbon
import("Geometry.Shapes", "Triangle");
import("Music.Instruments", "Triangle");
```

In cases such as this, one or more of the conflicting names must be
[aliased](aliases.md) such that the resulting names are unique. This only works
when importing one symbol in the `import`. For example, this would be allowed:

```carbon
import("Geometry.Shapes", "Triangle");
alias MusicTriangle = import("Music.Instruments", "Triangle");
```

## Alternatives

### Aliasing namespace names

It's been proposed that we could alias a namespace _name_, allowing code like:

```
$namespace_alias FBB = Foo.Bar.Baz

namespace FBB {
  struct Wiz { ... };
}
```

This is distinct from normal aliasing of the namespace itself, which would look
like:

```
alias FBB = Foo.Bar.Baz;

fn Fez(FBB.Wiz x);
```

In the latter example, although `FBB` can be referenced in code, `namespace FBB`
wouldn't be valid because `FBB` would be treated as the namespace name in that
context.

It's not clear that this is worthwhile; no evidence has been gathered to support
or contradict the use-case. Until and unless there is supporting evidence for a
real need for this, we should decline the feature due to its inherent complexity
and potential overlap with metaprogramming features.

### Allow shadowing of names

Disallowing shadowing of names may turn out to be costly, even for its
advantages of reducing bugs and confusion. While we could relax it later,
potentially with special syntax to handle edge cases, we don't have evidence at
present that it's necessary. As a result, we prefer the simplicity in reading of
disallowing shadowing.

### Broader imports, either all names or arbitrary code

Carbon imports require specifying individual names to import. We could support
broader imports, for example by pulling in all names from a library. In C++, the
`#include` preprocessor directive even supports inclusion of arbitrary code.

Pros:

-   Reduces boilerplate code specifying individual names.

Cons:

-   Loses out on parser benefits of knowing which identifiers are being
    imported.
-   Increases the risk of adding new features to APIs, as they may immediately
    get imported by a user and conflict with a pre-existing name, breaking code.
-   Readability problems arise because it's not clear where a given name may be
    coming from.
-   Arbitrary code inclusion can result in unexpected code execution, a way to
    create obfuscated code and a potential security risk.

We particularly value the parser benefits of knowing which identifiers are being
imported, and so we require individual names for imports.

### Different file extensions

The use of `.6c` as a short file extension or CLI has some drawbacks for
typability. There are several other possible extensions / commands:

-   `.cb` or `.cbn`: These collide with several acronyms and may not be
    especially memorable as referring to Carbon.
-   `.c6`: This seems a weird incorrect ordering of the atomic number and has a
    bad and NSFW, if _extremely_ obscure, Internet slang association.
-   `.carbon`: This is an obvious and unsurprising choice, but also quite long
    for a file extension.

This seems fairly easy for us to change as we go along, but we should at some
point do a formal proposal to gather other options and let the core team try to
find the set that they feel is close enough to be a bikeshed.

### Imports from URLs

In the future, we may want to support importing from a URL that identifies the
repository where that package can be found. This can be used to help drive
package management tooling and to support providing a non-name identity for a
package that is used to enable handling conflicted package names.

Although we're not designing this right now, it could easily fit into the
proposed syntax. For example:

```carbon
import("Foo.Bar", "Baz", .url = "https://foo.com")
```

### Prevent libraries from crossing package boundaries

Carbon's packages as proposed are effectively shorthand for a combination of:

```carbon
library PACKAGE
namespace PACKAGE { ... }
```

We could instead enforce package boundaries, preventing libraries from spanning
packages. Pragmatically, this would mean that `package Geometry library Shapes`
and `package Geometry.Shapes` would be different libraries, or perhaps packages
could only be a single identifier to prevent overlap.

This had been the original proposal, combined with package-granularity imports;
in other words, `import Geometry library Shapes` would be distinct from
`import Geometry.Shapes`. This approach didn't address name-level imports, which
would presumably become an additional argument for the `import` keyword, such as
`import Geometry library Shapes { Circle, Square }`.

Pros:

-   Finer-grained imports reduce the chance of conflicts between packages.

Cons:

-   Adds an extra layer of boundaries that isn't clearly necessary.
-   Increased complexity for users who must explicitly address libraries in more
    cases.

By avoiding creation of a package boundary separate from library and namespace
boundaries, we hope to simplify use of Carbon. It should still be possible for
name lookup to determine which dependency a name comes from without issue
because two distinct dependencies using an identical library name can still be
treated as a conflict.

### Scoped namespaces

Instead of including additional namespace information per-name, we could have
scoped namespaces, similar to C++. For example:

```carbon
namespace Foo {
  namespace Bar.Baz {
    struct Wiz { ... }
  }
}

fn Fez(Foo.Bar.Baz.Wiz x);
```

Pros:

-   Makes it easy to write many things in the same namespace.

Cons:

-   Can be hard to find the end of a namespace. For examples addressing this,
    end-of-namespace comments are called for by both the
    [Google](https://google.github.io/styleguide/cppguide.html#Namespaces) and
    [Boost](https://github.com/boostorg/geometry/wiki/Guidelines-for-Developers)
    style guides.
    -   Carbon may disallow the same-line-as-code comment style used for this.
        Even if not, if we acknowledge it's a problem, we should address it
        structurally for
        [readability](/docs/projects/goals.md#code-that-is-easy-to-read-understand-and-write).
    -   This is less of a problem for other scopes because they can often be
        broken apart.

There are other ways to address the con, such as adding syntax to indicate the
end of a namespace:

```carbon
namespace Foo {
  namespace Bar.Baz {
    struct Wiz { ... }
  } namespace Bar.Baz
} namespace Foo

fn Fez(Foo.Bar.Baz.Wiz x);
```

While we could consider such alternative approaches, we believe the proposed
contextless namespace approach is better, as it reduces information that
developers will need to remember when reading/writing code.

### Strict association between the filesystem path and library/namespace

Several languages create a strict association between the filesystem path, and
the method for pulling in an API. For example:

-   In C++, `#include` refers to specific files.
-   In Java, `package` and `import` both reflect filesystem structure.
-   In Python, `import` requires matching filesystem structure.
-   In TypeScript, `import` refers to specific files.

For contrast:

-   In Go, `package` uses an arbitrary name.

Pros:

-   A strict association between filesystem path and import path makes it easier
    to find source files. This is used by some languages for compilation.

Cons:

-   The strict association makes it harder to move names between files without
    updating callers.

We are choosing to avoid the strict association with filesystem paths in order
to ease refactoring. With this approach, more refactorings will not need changes
to API users.
