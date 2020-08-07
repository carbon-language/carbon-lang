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
    -   [Named scopes](#named-scopes)
    -   [Imports](#imports)
-   [Details](#details)
    -   [Package keyword](#package-keyword)
        -   [File extension](#file-extension)
-   [ORIGINAL Libraries](#original-libraries)
-   [ORIGINAL Namespaces](#original-namespaces)
-   [ORIGINAL Packages](#original-packages)
-   [ORIGINAL Imports](#original-imports)
-   [ORIGINAL Shadowing of names](#original-shadowing-of-names)
-   [ORIGINAL Standard library names](#original-standard-library-names)
-   [Alternatives](#alternatives)
    -   [File extensions](#file-extensions)
    -   [Implementation vs interface files](#implementation-vs-interface-files)
    -   [Strict association between the filesystem path and library/namespace](#strict-association-between-the-filesystem-path-and-librarynamespace)
    -   [Broader imports, either all names or arbitrary code](#broader-imports-either-all-names-or-arbitrary-code)

<!-- tocstop -->

## Goals and philosophy

Important Carbon goals for code and name organization are:

-   [Software and language evolution](/docs/project/goals.md#software-and-language-evolution):

    -   We should support API libraries adding new structs, functions or other
        identifiers without those new identifiers being able to shadow or break
        existing client identifiers with identical names.

    -   We should make it as easy as possible to refactor code between files.

-   [Fast and scalable development](/docs/project/goals.md#fast-and-scalable-development):

    -   It should be easy for IDEs and other tooling to parse code, without
        needing to parse imports for context.

    -   Structure should be provided for large projects to opt into features
        which will help scale features, while not adding burdens to small
        projects that don't need it.

## Overview

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

A **package** is also required for every file. This contributes a common element
to both the library and namespace scope. A file may optionally append to the
package to build the library scope, to separate compilation and dependencies of
the file within its package. It may also optionally wrap code in one or more
namespace scopes, to separate names.

For example, to set use `Geometry` as both the library scope and namespace
scope, a file might contain:

```carbon
package Geometry;

struct Point { ... }
```

If separate scopes are desired, such as a `Geometry.TwoDimensional` library
scope and `Geometry.Shapes` namespace scope, a file might contain:

```carbon
package Geometry library TwoDimensional;

namespace Shapes {
  struct Circle { ... }
}
```

### Imports

Imports are the method for importing names from other namespace scopes. Imports
will always name what they are importing;

For example, to import the `Geometry.Shapes` namespace:

```carbon
import("Geometry", "Shapes");
```

This results in a usable name of `Shapes`, through which members of the
namespace can be accessed, such as `Shapes.Circle`. Specific identifiers may
also be imported, for example:

```carbon
import("Geometry.Shapes", "Circle");
```

Multiple names may be imported using a single statement:

```carbon
import("Geometry.Shapes", ("Circle", "Triangle", "Square"));
```

In the case of a shadowed name, the imported name may also be
[aliased](aliases.md):

```carbon
alias GeometryCircle = import("Geometry.Shapes", "Circle");
```

When importing from another file or library that is in the current namespace
scope, the namespace may be omitted from the import. For example, an import
`Geometry.Point` may look like:

```carbon
package Geometry;

import("Point");
```

## Details

### Package keyword

The first non-comment, non-whitespace line of a Carbon file will be the
`package` keyword. The `package` keyword's syntax, combined with the optional
`library` keyword, may be expressed as a regular expression:

```
package IDENTIFIER(\.IDENTIFIER)* (library IDENTIFIER(\.IDENTIFIER)*)?;
```

For example, the following are all valid package keyword uses:

```
package Geometry;
package Geometry.Shapes;
package Geometry.Shapes.Flat;
package Geometry library Shapes;
package Geometry library Shapes.Flat;
```

The first name passed to the `package` keyword is a prefix that will be used for
both the file's library and namespace scopes. When the `library` keyword is
specified, its identifiers are combined with the package to generate the library
scope.

For example, `package Geometry library Shapes.Flat` defines a file that is in
the library `Geometry.Shapes.Flat`, and uses the `Geometry` namespace.

Because the `package` keyword must be specified in all files, there are a couple
important and deliberate side-effects:

-   Every name in Carbon will be in a namespace corresponding to the `package`.
    There is no "global" namespace.
-   Every file will be in precisely one library.

#### File extension

We use the `.6c` extension as a matter of convention. This comes from how the
element Carbon may be written as "<sub>6</sub>C". In the language's case, we
need a reasonably unique extension, and this is what we've chosen.

## ORIGINAL Libraries

Carbon's libraries are based very directly on the design of C++ modules.

Every Carbon library has a primary file that defines its exported interface.
This file must start off with the canonical form of the top-level declaration:

```
package Abseil library Container;
```

The library may contain additional files as part of its implementation. These
all must start with a special top-level declaration:

```
package Abseil library Container impl;
```

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

## ORIGINAL Namespaces

Carbon namespaces work essentially the same way as C++ namespaces. They form
named scopes for names. They can nest and be reopened and aren't restricted to
library boundaries. A good example in practice will be things like `testing` or
`internal` namespaces below a package. These often will have names that benefit
from an isolated space but are spread across several libraries.

Namespaces are navigated using dot notation just like names in other scopes in
Carbon:

```
namespace Foo {
  namespace Bar {
    namespace Baz {
      alias ??? MyInt = Int;
    }
  }
}

fn F(Foo.Bar.Baz.MyInt x);
```

    **Note**: the syntax for aliases is not at all in a good state yet. We've considered a few alternatives, but they all end up being confusing in some way. We need to figure out a good and clean syntax that can be used here.

There are no wildcard aliases (or complex partial wildcard aliases like
`using namespace foo;` in C++). Alias declarations can alias namespace names, so
there is no need for custom namespace alias support.

**Possible extension**: If it proves to be a common case, we could support
aliasing a set of names in one namespace into another and provide special syntax
for it. However, it would force us to have syntax introducing many (top-level)
names in a single declaration, and presents other syntactic challenges. Unclear
this will be worth it or how best to design it. Very vague syntax sketch:

```
// Normal alias of a namespace.
alias Foo.Bar.Baz: FBB;

namespace FBB {
  alias Int: OtherInt;
}

fn G(Foo.Bar.Baz.OtherInt y);

namespace Elsewhere {
  // Quickly create aliases in this namespace for two names
  // (`MyInt`, and `OtherInt`) in some other namespace.
  alias ??? from Foo.Bar.Baz ??? MyInt, OtherInt;
}
```

## ORIGINAL Packages

Carbon's packages are specialized top-level namespaces. All code in Carbon is
written within a package and thus within a top-level namespace (and thus
inherently not in the global namespace). Libraries cannot span packages. This
restricts our library abstraction boundary to fit within the top-level name
abstraction boundary. This is the only place where we intersect our name
abstractions and our library abstractions.

All names introduced in a Carbon file are fundamentally within that file's
package. When they are accessed from outside of that package, they have to be
qualified with the package name. From inside the package, however, names from
that package can be used without qualification, as a matter of convenience.
Conceptually, we consider all code within one package to belong together.

## ORIGINAL Imports

Carbon files access interfaces declared in other files by importing them. All
imports for a file must be immediately after the package declaration and in a
contiguous block. No other code can be interleaved with the imports. There are
also three forms of import based on the distance between the files:

```
package Abseil library Time;

// Importing a library from some other package.
import Widget library Wombat;

// Importing another library from the same package.
import library Container;

// Importing another file from the same library.
import file "internal/Duration.cb";

...
```

Two of these imports, the one from a file and the one from another library in
the same package, are fundamentally importing _local_ code -- all of this code
resides in a single package. As such, they directly import names into the
package's namespace and those names are visible with unqualified name lookup.

    **Open question:** The importing of a file using a quoted string to name it is an interesting but potentially risky approach. We suggested similar techniques for C++ modules and so have done some work to understand the implications and it at least looks reasonable to implement and use. But it may be too surprising for programmers, so this may be an area we revisit. Certainly, the C++ committee was quite unhappy with taking advantage of the fact that some of the organization and naming has already been done with filenames, requiring "module partitions" which have a custom name syntax.

The more general form of import is importing external code into the package. As
a consequence, the only name that becomes visible is the name of that package,
and everything else is found under it using the standard namespace dotted
traversal (`Widget.Stuff`).

Each import brings in a specific library (or file), not the entire package. When
importing an unnamed library from a package (which must be the only library in
the package), these are indistinguishable and so the library component can be
simply omitted:

```
import SimplePackage;
```

This imports the entire package because that consists of a single library.

A single import can name multiple libraries within the package to import:

```
import Abseil libraries: Container, Time;
```

    **Open question:** We might investigate some different syntax alternatives for multi-import. Automatic formatting should be able to make the above at least somewhat cleaner for long lists:

```
import Abseil libraries:
    ModuleNumber1,
    ModuleNumber2,
    ModuleNumber3,
    ModuleNumber4,
    ModuleNumber5,
    ModuleNumber6,
    ModuleNumber7,
    ModuleNumber8,
    ModuleNumber9,
    ModuleNumber10;
```

    But maybe people would prefer other more structured syntaxes like `(a, b, c, ...)`. We can investigate these and see what sticks.

When importing a named package, the imported name must exactly match that
package's declared package name -- no renaming is permitted. Aliases plus a
wrapping package can be used to incrementally migrate uses of a package to a new
name, then to a wrapping package using the new name, and finally remove the old
package, but this will remain a non-trivial refactoring. The reason this is
important is that the name of the package essentially forms the identity of
every exported interface component and in the face of templates must be stable
for the purpose of linking together code.

    **Future work:** Carbon should also support importing packages from a URL that identifies the repository where that package can be found. This can be used to help drive package management tooling and to support providing a non-name identity for a package that is used to enable handling conflicted package names.

## ORIGINAL Shadowing of names

Carbon completely forbids shadowing of names. Because there is no global
namespace and no using directive, unqualified names are entirely within local
control and so should not require shadowing for maintenance or development.
Allowing shadowing has a long history of bugs and confusion. While it is used in
some clear and readable places in C++, those situations are likely better
addressed with language facilities than cleverly shadowed names.

    **Open question:** This is an extremely restrictive stance. It has many advantages, but may end up being too costly. We can easily relax it if needed, potentially with special syntax to handle specific edge cases or as an interim measure to aid migration. Or if data emerges, we can simply revisit it entirely.

## ORIGINAL Standard library names

When discussing standard library names, we mean the core of the standard library
that is generically useful and applicable to code and application in nearly
every domain, such as the STL in C++. More domain-specific libraries, while we
may choose to standardize them for Carbon, don't need any special consideration
and should be organized into domain-relevant packages like any other libraries.
We also don't think this core part of the standard library would meaningfully be
represented as a single Carbon library -- Carbon's libraries are more fine
grained.

The package named "Carbon" will provide this core part of the standard "library"
as a collection of Carbon libraries providing similar facilities to what is
found in the C++ STL. Other than the package name, it would work just like any
other package of libraries.

However, there are some specific wrinkles with some things provided by this
package that motivate some special facilities.

The first interesting aspect is that we would like to make even fundamental and
primitive types in Carbon use interfaces that are no more special than user
defined types. This means `Int32` should be a "normal" type rather than a
keyword, and be provided by the standard library.

This raises the second interesting aspect: it doesn't seem reasonable for Carbon
code to need to import some library in order to access `Int` or `Bool` types,
despite wanting those interfaces to be defined using normal Carbon syntax and
facilities. Further, they would be imported as the `Carbon` package requiring
them to be qualified with a package name: `Carbon.Int`, etc. This seems
significantly too onerous.

To address these issues, all Carbon files have an implicit import of the
`Fundamentals` library from the `Carbon` package as if the file contained the
line:

```
import Carbon library Fundamentals;
```

Further, all Carbon files have an implicit alias of every exported name in this
library into their file's scope.

    **Open question:** We could add an automatic syntax for creating aliases while importing, and reduce this to a single implicit import using that facility:

```
import Carbon library Fundamentals with aliases: Int32, Int64, Bool, ...;
```

    It isn't clear whether this is an important convenience to expose outside of the implicit case so we've simply used an implicit alias for now, but we can revisit this if needed.

This makes the `Fundamentals` library extremely special in one final way: adding
new exported names to this library is a potentially globally breaking change and
would require shipping some update tooling to users. Because of the lack of
shadowing, every name in this library is nearly as disruptive as a keyword.
Fortunately, we expect this library to be both fairly small and extremely
unlikely to change. The kinds of names exported from it often _are_ keywords in
other languages (including C++).

However, this does mean that a collection of standard library names (those from
the `Fundamentals` library) will be used unqualified in essentially all cases in
Carbon code. For consistency, we should recommend that names from standard
libraries in the Carbon package are typically aliased and used unqualified for
consistency. Skipping this and using the qualified form is always available to
work around collisions. The fact that the package name isn't extra short may
help incentivise consistent use of aliases here. Because each aliased name is
explicitly introduced in the code, these don't cause any evolution problems for
Carbon libraries -- newly added names won't be implicitly aliased for any
library outside of `Fundamentals`.

    **Open question:** If we use an embedded alias-in-the-import syntax, we could additionally require using that for any imports of standard libraries, and make the `Carbon` package an unnamed and unnamable package where every usable name must be aliased into the importer's scope. This would ensure consistency, remove the visible name `Carbon` but would make it more complicated to handle name collisions and add yet more special syntax usage.

## Alternatives

### File extensions

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

### Implementation vs interface files

We could separate implementation and interface at the file level. The idea would
be that an implementation file could not export any interface from the library.

Pros:

-   Increases the ability for parallel compilation, as we would be able to skip
    parsing of implementation-only files when compiling a different library that
    depends on the former.

Cons:

-   Parallel compilation may end up expecting the full, compiled library, not
    just an interface. Separating out the interface may not yield significant
    benefit.
-   This functionality is redundant and adds syntax complexity. We will want
    similar "implementation" marking at the namespace and possibly individual
    function/struct level, and that should suffice.
-   We plan on per-file parsing to be fast, so marking a file as
    implementation-only versus marking every possibly exported name in that file
    as implementation-only may end up having negligible performance benefit.

This isn't clearly a large enough benefit to add the complexity.

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
