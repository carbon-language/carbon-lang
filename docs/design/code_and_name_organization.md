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
    -   [Sizing packages and libraries](#sizing-packages-and-libraries)
    -   [Imports](#imports)
-   [Details](#details)
    -   [Name paths](#name-paths)
    -   [Packages](#packages)
        -   [Shorthand notation for libraries in packages](#shorthand-notation-for-libraries-in-packages)
        -   [Package name conflicts](#package-name-conflicts)
    -   [Libraries](#libraries)
        -   [Exporting entities from an API file](#exporting-entities-from-an-api-file)
        -   [Granularity of libraries](#granularity-of-libraries)
    -   [Namespaces](#namespaces)
        -   [Re-declaring imported namespaces](#re-declaring-imported-namespaces)
        -   [Aliasing](#aliasing)
    -   [Imports](#imports-1)
        -   [Imported name conflicts](#imported-name-conflicts)
        -   [Imports from the current package](#imports-from-the-current-package)
-   [Caveats](#caveats)
    -   [Package and library name conflicts](#package-and-library-name-conflicts)
    -   [Potential refactorings](#potential-refactorings)
        -   [Update imports](#update-imports)
        -   [Between `api` and `impl` files](#between-api-and-impl-files)
        -   [Other refactorings](#other-refactorings)
    -   [Preference for few child namespaces](#preference-for-few-child-namespaces)
    -   [Redundant markers](#redundant-markers)
-   [Open questions](#open-questions)
    -   [Different file extensions](#different-file-extensions)
    -   [Imports from other languages](#imports-from-other-languages)
    -   [Imports from URLs](#imports-from-urls)
    -   [Test file type](#test-file-type)
-   [Alternatives](#alternatives)
    -   [Packages](#packages-1)
        -   [Name paths for package names](#name-paths-for-package-names)
        -   [Referring to the package as `package`](#referring-to-the-package-as-package)
        -   [Remove the `library` keyword from `package` and `import`](#remove-the-library-keyword-from-package-and-import)
        -   [Remove the `namespace` keyword from `package`](#remove-the-namespace-keyword-from-package)
        -   [Rename package concept](#rename-package-concept)
        -   [Strict association between the filesystem path and library/namespace](#strict-association-between-the-filesystem-path-and-librarynamespace)
    -   [Libraries](#libraries-1)
        -   [Allow importing implementation files from within the same library](#allow-importing-implementation-files-from-within-the-same-library)
        -   [Alternative library separators and shorthand](#alternative-library-separators-and-shorthand)
            -   [`/` separators](#-separators)
            -   [Single-word libraries](#single-word-libraries)
        -   [Collapse API and implementation file concepts](#collapse-api-and-implementation-file-concepts)
            -   [Automatically generating the API separation](#automatically-generating-the-api-separation)
        -   [Collapse file and library concepts](#collapse-file-and-library-concepts)
        -   [Collapse the library concept into packages](#collapse-the-library-concept-into-packages)
        -   [Collapse the package concept into libraries](#collapse-the-package-concept-into-libraries)
        -   [Different file type labels](#different-file-type-labels)
        -   [Function-like syntax](#function-like-syntax)
        -   [Inlining from implementation files](#inlining-from-implementation-files)
        -   [Library-private access controls](#library-private-access-controls)
        -   [Managing API versus implementation in libraries](#managing-api-versus-implementation-in-libraries)
        -   [Multiple API files](#multiple-api-files)
        -   [Name paths as library names](#name-paths-as-library-names)
    -   [Namespaces](#namespaces-1)
        -   [Coarser namespace granularity](#coarser-namespace-granularity)
        -   [Scoped namespaces](#scoped-namespaces)
    -   [Imports](#imports-2)
        -   [Block imports](#block-imports)
        -   [Block imports of libraries of a single package](#block-imports-of-libraries-of-a-single-package)
        -   [Broader imports, either all names or arbitrary code](#broader-imports-either-all-names-or-arbitrary-code)
        -   [Direct name imports](#direct-name-imports)
        -   [Optional package names](#optional-package-names)

<!-- tocstop -->

## Goals and philosophy

Important Carbon goals for code and name organization are:

-   [Language tools and ecosystem](#language-tools-and-ecosystem)

    -   Tooling support is important for Carbon, including the possibility of a
        package manager.

    -   Developer tooling, including both IDEs and refactoring tools, are
        expected to exist and be well-supported.

-   [Software and language evolution](/docs/project/goals.md#software-and-language-evolution):

    -   We should support libraries adding new structs, functions or other
        identifiers without those new identifiers being able to shadow or break
        existing users that already have identifiers with conflicting names.

    -   We should make it easy to refactor code, including moving code between
        files. This includes refactoring both by humans and by developer
        tooling.

-   [Fast and scalable development](/docs/project/goals.md#fast-and-scalable-development):

    -   It should be easy for developer tooling to parse code, without needing
        to parse imports for context.

    -   Structure should be provided for large projects to opt into features
        which will help maintain scaling of their codebase, while not adding
        burdens to small projects that don't need it.

## Overview

Carbon files have a `.carbon` extension, such as `geometry.carbon`. These files
are the basic unit of compilation.

Each file begins with a declaration of which
_package_<sup><small>[[define](/docs/guides/glossary.md#package)]</small></sup>
it belongs in. The package is the unit of _distribution_. The package name is a
single identifier, such as `Geometry`. An example API file in the `Geometry`
package would start with `package Geometry api;`.

A tiny package may consist of a single library with a single file, and not use
any further features of the `package` keyword.

It is often useful to have a physical separation the API from its
implementation. This may help organize code as a library grows, or to let the
build system distinguish between the dependencies of the API itself and its
underlying implementation. Implementation files allow for code to be extracted
out from the API file, while only being callable from other files within the
library, including both API and implementation files. Implementation files are
marked by both naming the file to use an extension of `.impl.carbon` and
changing the package to `package Geometry impl`.

However, as a package adds more files, it will probably want to separate out
into multiple
_libraries_<sup><small>[[define](/docs/guides/glossary.md#library)]</small></sup>.
A library is the basic unit of _dependency_. Separating code into multiple
libraries can speed up the overall build while also making it clear which code
is being reused. An example API file in the library of `Shapes` in the
`Geometry` package would look like `package Geometry library("Shapes") api;`.
This library can also be referred to as `Geometry/Shapes` as shorthand in text.

As code becomes more complex, and users pull in more code, it may also be
helpful to add
_namespaces_<sup><small>[[define](/docs/guides/glossary.md#namespace)]</small></sup>
to give related entities consistently structured names. A namespace affects the
_name
path_<sup><small>[[define](/docs/guides/glossary.md#name-path)]</small></sup>
used when calling code. For example, with no namespace, if a `Geometry` library
defines `Circle` then the name path will be `Geometry.Circle`. However, an
example namespace of `TwoDimensional` in the `Geometry` package would look like
`package Geometry library("Shapes") namespace TwoDimensional api;`, and result
in a name path of `Geometry.TwoDimensional.Circle`.

This scaling of packages into libraries and namespaces is how Carbon supports
both small and large codebases.

### Sizing packages and libraries

A different way to think of the sizing of packages and libraries is:

-   A package is a GitHub repository.
    -   Small and medium projects that fit in a single repository will typically
        have a single package. For example, a medium-sized project like
        [Abseil](https://github.com/abseil/abseil-cpp/tree/master/absl) could
        still use a single `Abseil` package.
    -   Large projects will have multiple packages. For example,
        [Boost.Geometry](https://github.com/boostorg/geometry) might be a
        `BoostGeometry` package, with other packages for other Boost
        repositories.
-   A library is a few files that provide an interface and implementation, and
    should remain small.
    -   Small projects will have a single library when it's easy to maintain all
        code in a few files.
    -   Medium and large projects will have multiple libraries. For example,
        [Boost.Geometry's Distance](https://github.com/boostorg/geometry/blob/develop/include/boost/geometry/algorithms/detail/distance/interface.hpp)
        interface and implementation might be its own library within
        `BoostGeometry`, with dependencies on other libraries in `BoostGeometry`
        and potentially other packages from Boost.
        -   Library names could be named after the feature, such as
            `library("Distance")`, or include part of the path to reduce the
            chance of name collisions, such as `library("Algorithms.Distance")`.

Packages may choose to expose libraries that expose unions of interfaces from
other libraries within the package. However, doing so would also provide the
transitive closure of build-time dependencies, and is likely to be discouraged
in many cases.

### Imports

The `import` keyword supports reusing code from other files and libraries.

For example, to use `Geometry.Circle` from the `Geometry/Shapes` library:

```carbon
import Geometry library("Shapes");

fn Area(Geometry.Circle circle) { ... };
```

The `library` keyword is optional for `import`, and its use should parallel that
of `library` on the `package` of the code being imported.

Imports may also be
[renamed to prevent name conflicts](#name-conflicts-of-imports). For example:

```carbon
import Geometry library("Shapes") as Geo;

fn Area(Geo.Circle circle) { ... };
```

If multiple imports are made from the same package, each import may be named
differently. This allows code to be explicit about which library is being used,
which some engineers may prefer to make it explicit which library an API comes
from. Also, if a `api` file wants to re-export portions of a package, it may be
helpful to be cautious about what is being exported.

For example:

```carbon
package Math library("Interface") api;

import Math library("Statistics");
import Math library("Internal") as MathInternal;

// This would export the `Functions` namespace in "Math/Statistics", and not
// "Math/Internal".
api alias Functions = Math.Functions;
```

## Details

### Name paths

[Name paths](#name-paths) are defined above as sequences of identifiers
separated by dots. This syntax can be expressed as a rough regular expression:

```regex
IDENTIFIER(\.IDENTIFIER)*
```

Name conflicts are addressed by [name lookup](name_lookup.md).

### Packages

The first non-comment, non-whitespace line of a Carbon file will be the
`package` keyword. The `package` keyword's syntax may be expressed as a rough
regular expression:

```regex
package IDENTIFIER (library\(STRING\))? (namespace NAME_PATH)? (api|impl);
```

For example:

```carbon
package Geometry library("Objects.FourSides") namespace TwoDimensional api;
```

Breaking this apart:

-   The identifier passed to the `package` keyword, `Geometry`, is the package
    name and will prefix both library and namespace paths.
    -   The `package` keyword also declares a namespace entity matching the
        package name. In other words, if the file declares `struct Line`, that
        may be used from within the file as both `Line` directly and
        `Geometry.TwoDimensional.Line` using the `Geometry` declaration created
        by the `package` keyword.
-   When the optional `library` keyword is specified, its name path argument is
    combined with the package to generate the library path. In this example, the
    `Geometry/Objects.FourSides` library path will be used.
-   When the optional `namespace` keyword is specified, its name path argument
    is combined with the package to generate the namespace path. In this
    example, the `Geometry.TwoDimensional` namespace will be used.
-   The use of the `api` keyword indicates this is an API files as described
    under [libraries](#libraries). If it instead had `impl`, this would be an
    implementation file.

Because the `package` keyword must be specified exactly once in all files, there
are a couple important and deliberate side-effects:

-   Every file will be in precisely one library.
    -   A library still exists even when there is no explicit library argument,
        such as `package Geometry api;`. This could be considered equivalent to
        `package Geometry library("") api;`, although we should not allow that
        specific syntax as error-prone.
-   Every entity in Carbon will be in a namespace, even if its namespace path
    consists of only the package name. There is no "global" namespace.
    -   Every entity in a file will be defined within the namespace described in
        the `package` statement.
    -   Entities within a file may be defined in
        [child namespaces](#namespaces).

Files contributing to `Geometry/Objects.FourSides` must all start with
`package Geometry library("Object.FourSides")`, but will differ on `api`/`impl`
types, and may have differing `namespace` arguments.

There is no restriction that a namespace only come from one library in a
package. It's possible that files contributing to the `Geometry.TwoDimensional`
namespace may use different `library` arguments.

#### Shorthand notation for libraries in packages

Library names may also be referred to as `PACKAGE/LIBRARY` as shorthand in text.
`PACKAGE/default` will refer to the name of the library used when no `library`
argument is specified, although `PACKAGE` may also be used in situations where
it is unambiguous that it still refers to the default library.

The `/` character is used as a separator because it is not a valid character for
identifiers, particularly the package name. Note that library names are strings
and may include any character, including `/` if desired.

#### Package name conflicts

Because the package also declares a namespace entity with the same name,
conflicts with the package name are possible. We do not support packages
providing entities with the same name as the package.

For example, this is a conflict for `DateTime`:

```carbon
package DateTime api;

struct DateTime { ... };
```

This declaration is important for [implementation files](#libraries), which
implicitly import the library's API, because it keeps the package name as an
explicit entity in source files.

### Libraries

Every Carbon library consists of one or more files. Each Carbon library has a
primary file that defines its API, and may optionally contain additional files
that are implementation.

-   An API file's `package` will have `api`. For example,
    `package Geometry library("Shapes") api;`
    -   API filenames must have the `.carbon` extension. They must not have a
        `.impl.carbon` extension.
-   An implementation file's `package` will have `impl`. For example,
    `package Geometry library("Shapes") impl;`.
    -   Implementation filenames must have the `.impl.carbon` extension.
    -   Implementation files implicitly import the library's API. Implementation
        files cannot import each other. There is no facility for file or
        non-`api` imports.

The difference between API and implementation will act as a form of access
control. API files must compile independently of implementation, only importing
from APIs from other libraries. API files are also visible to all files and
libraries for import. Implementation files only see API files for import, not
other implementation files.

When any file imports a library's API, it should be expected that the transitive
closure of imported files from the primary API file will be a compilation
dependency. The size of that transitive closure affects compilation time, so
libraries with complex implementations should endeavor to minimize their API
imports.

Libraries also serve as a critical unit of compilation. Dependencies between
libraries must be clearly marked, and the resulting dependency graph will allow
for separate compilation.

#### Exporting entities from an API file

In order to actually be part of a library's API, entities must both be in the
API file and explicitly marked as an API. This is done using the `api` keyword,
which is only allowed in the API file. For example:

```carbon
package Geometry library("Shapes") api;

// Circle is marked as an API, and will be available to other libraries as
// Geometry.Circle.
api struct Circle { ... }

// CircleHelper is not marked as an API, and so will not be available to other
// libraries.
fn CircleHelper(Circle circle) { ... }

// Only entities in namespaces should be marked as an API, not the namespace
// itself.
namespace Operations;

// Operations.GetCircumference is marked as an API, and will be available to
// other libraries as Geometry.Operations.GetCircumference.
api fn Operations.GetCircumference(Circle circle) { ... }
```

This means that an API file can contain all implementation code for a library.
However, separate implementation files are still desirable for a few reasons:

-   It will be easier for readers to quickly scan an API-only file for API
    documentation.
-   Reducing the amount of code in an API file can speed up compilation,
    especially if fewer imports are needed. This can result in transitive
    compilation performance improvements for files using the library.
-   From a code maintenance perspective, having smaller files can make a library
    more maintainable.

Use of the `api` keyword is not allowed within files marked as `impl`.

#### Granularity of libraries

Conceptually, we expect libraries to be very small, possibly containing only a
single class. This will be pressured because of the limitation of a single API
file per library. We expect that keeping libraries small will enable better
parallelism of compilation.

### Namespaces

Namespaces offer named paths for entities. Namespaces may be nested. Multiple
libraries may contribute to the same namespace. In practice, packages may have
namespaces such as `Testing` containing entities that benefit from an isolated
space but are present in many libraries.

While the `namespace` keyword syntax should be similar to the `namespace`
argument to the `package` keyword, the `namespace` keyword is its own statement.
Its syntax may loosely be expressed as a regular expression:

```regex
namespace NAME_PATH;
```

Whereas a `package`-line `namespace` changes the namespace for the entire file,
the separate `namespace` keyword only declares a namespace. It is then applied
to specified entities by including it as a prefix when declaring a name. For
example:

```carbon
package Time;

namespace Timezones.Internal;
struct Timezones.Internal.RawData { ... }

fn ParseData(Timezones.Internal.RawData data);
```

A namespace declaration adds the first identifier in the name path as a name in
the file's namespace. In the above example, after declaring
`namespace Timezones.Internal;`, `Timezones` is available as an identifier and
`Internal` is reached through `Timezones`.

Namespaces declared and added to within a file must always be children of the
file-level namespace. For example, this declares
`Geometry.Shapes.Flat.Triangle`:

```carbon
package Geometry namespace Shapes api;

namespace Flat;

struct Flat.Triangle { ... }
```

#### Re-declaring imported namespaces

Namespaces may be imported, in addition to being declared. However, the
namespace must still be declared locally in order to add symbols to it.

For example, if the `Geometry/Shapes.ThreeSides` library provides the
`Geometry.Shapes` namespace, this code is still valid:

```carbon
package Geometry library("Shapes.FourSides") api;

import Geometry library("Shapes.ThreeSides");

// This does not conflict with the existence of `Geometry.Shapes` from
// `Geometry/Shapes.ThreeSides`, even though the name path is identical.
namespace Shapes;

// This requires the above 'namespace Shapes' declaration.
struct Shapes.Square { ... };
```

#### Aliasing

Carbon's [alias keyword](aliases.md) will support aliasing namespaces. For
example, this would be valid code:

```carbon
namespace Timezones.Internal;
alias TI = Timezones.internal;

struct TI.RawData { ... }
fn ParseData(TI.RawData data);
```

### Imports

The `import` keyword supports reusing code from other files and libraries. The
`import` keyword's syntax may be expressed as a rough regular expression:

```regex
import IDENTIFIER (library NAME_PATH)? (as IDENTIFIER)?;
```

All imports for a file must have only whitespace and comments between the
`package` declaration and them. If [metaprogramming](metaprogramming.md) code
generates imports, it must only generate imports following the
non-metaprogramming imports. No other code can be interleaved.

All imports are done at the library level. They provide the package's namespace
path for use. Child namespaces or entities must separately be
[aliased](aliases.md) if desired.

For example:

```carbon
package Geometry api;

// This imports the Math/default library, and provides a Math namespace
// identifier for use.
import Math;
// This imports the Math/Trigonometry library, and resuses the Math namespace.
import Math library("Trigonometry");

fn DoSomething() {
  ...
  // The compiler will determine which Math library Sin comes from.
  Math.Sin(...);
  ...
}
```

Imports do not elide child namespaces; they are always referenced from the
package. For example, given this library:

```carbon
package Unicode library("conversions.utf32") namespace Conversions;

api fn ToUtf32(String str) -> Bytes { ... }
```

Calling could should still be rooted at the package being imported:

```carbon
// The namespace here is ignored when addressing imported APIs.
package Unicode library("conversions") namespace Conversions;

import Unicode library("utf32");

fn ToUnicode(String str, Int size) -> Bytes {
  ...
  if (size == 32) {
    return Unicode.Conversions.Utf32(str);
  }
  ...
}
```

#### Imported name conflicts

It's possible that an imported package will have the same name as an entity
within the file doing the import. Importing them would result in name conflicts.
For example, this would be a rejected name conflict because it redefines
`Geometry`:

```carbon
import Geometry;

fn Geometry(Geometry.Circle: circle) { ... }
```

In cases such as this, `as` can be used to rename the entity used to access the
imported package. For example, this would be allowed:

```carbon
import Geometry as Geo;

fn Geometry(Geo.Circle: circle) { ... }
```

The `as` keyword only renames the imported package identifier; it does not
rename namespaces within the package.

#### Imports from the current package

Entities defined in the current file may be used without mentioning the package
prefix. However, other symbols from the package must be imported and accessed
through the package namespace just like symbols from any other package.

For example:

```carbon
package Geometry api;

// This is required even though it's still in the Geometry package.
import Geometry library("Shapes");

// Circle must be referenced using the Geometry namespace of the import.
fn GetArea(Geometry.Circle: c) { ... }
```

## Caveats

### Package and library name conflicts

Library name conflicts should not occur, because it's expected that a given
package is maintained by a single organization. It's the responsibility of that
orgnaization to maintain unique library names within their package.

There is a greater risk of package name conflicts where two organizations use
the same package name. We will encourage a unique package naming scheme, such as
maintaining a name server for open source packages. Conflicts can also be
addressed by renaming one of the packages, either at the source, or as a local
modification.

The `as` keyword of `import` does not address package name conflicts because,
while it supports renaming a package to avoid intra-file name conflicts, it
would not be able to differentiate between two identically named packages.

### Potential refactorings

These are potential refactorings that we consider important to make it easy to
automate.

#### Update imports

Imports will frequently need to be updated as part of refactorings.

When code is deleted, it should be possible to parse the remaining code, parse
the imports, and determine which entities in imports are referred to. Unused
imports can then be removed.

When code is moved, it's similar to deletion in the originating file. For the
destination file, the moved code should be parsed to determine which entities it
referred to from the originating file's imports, and these will need to be
included in the destination file: either reused if already present, or added.

When new code is added, existing imports can be checked to see if they provide
the symbol in question. There may also be heuristics which can be implemented to
check build dependencies for where imports should be added from, such as a
database of possible entities and their libraries. However, adding references
may require manually adding imports.

#### Between `api` and `impl` files

-   Move an implementation of an API from an `api` file to an `impl` file, while
    leaving a declaration behind.

    -   This should be a local change that will not affect any calling code.
    -   Inlining will be affected because the implementation won't be visible to
        callers.
    -   [Update imports](#update-imports).

-   Split an `api` and `impl` file.

    -   This is a repeated operation of individual API moves, as noted above.

-   Move an implementation of an API from an `impl` file to an `api` file.

    -   This should be a local change that will not affect any calling code.
    -   Inlining will be affected because the implementation becomes visible to
        callers.
    -   [Update imports](#update-imports).

-   Combine an `api` and `impl` file.

    -   This is a repeated operation of individual API moves, as noted above.

-   Remove the `api` label from a declaration.

    -   Search for library-external callers, and fix them first.

-   Add the `api` label to a declaration.

    -   This should be a local change that will not affect any calling code.

-   Move a non-`api`-labelled declaration from an `api` file to an `impl` file.

    -   The declaration must be moved to the same file as the implementation of
        the declaration.
    -   The declaration can only be used by the `impl` file that now contains
        it. Search for other callers within the library, and fix them first.
    -   [Update imports](#update-imports).

-   Move a non-`api`-labelled declaration from an `impl` file to an `api` file.

    -   This should be a local change that will not affect any calling code.
    -   [Update imports](#update-imports).

-   Move a declaration and implementation from one `impl` file to another.

    -   Search for any callers within the source `impl` file, and either move
        them too, or fix them first.
    -   [Update imports](#update-imports).

#### Other refactorings

-   Rename a package.

    -   The imports of all calling files must be updated accordingly.
    -   Either `as` can be used on the import to keep the old name in order to
        avoid changing call sites, or call sites will need to be changed.
    -   [Update imports](#update-imports).

-   Move an `api`-labelled declaration and implementation between libraries in
    the same package.

    -   The imports of all calling files must be updated accordingly.
    -   As long as the namespaces remain the same, no call sites will need to be
        changed.
        -   There is an exception to this if the new library is already imported
            in the calling file using a different value for `as` than the old
            library's import. In that case, the named entity for the import will
            change and needs to be updated when the imports are updated.
    -   [Update imports](#update-imports).

-   Rename a library.

    -   This is equivalent to a repeated operation of moving an `api`-labelled
        declaration and implementation between libraries in the same package.

-   Move a declaration and implementation from one namespace to another.

    -   Ensure the new namespace is declared for the declaration and
        implementation.
    -   Update the namespace used by call sites.
    -   The imports of all calling files may remain the same.

-   Rename a namespace.

    -   This is equivalent to a repeated operation of moving a declaration and
        implementation from one namespace to another.

-   Rename a file, or move a file between directories.

    -   Build configuration will need to be updated.
    -   Carbon code will not change. The `package` keyword determines how a file
        is imported, so the library is unaffected by filesystem location.

### Preference for few child namespaces

We expect that most code should use a package and library, but avoid specifying
namespaces beneath the package. The package name itself should typically be
sufficient distinction for names.

Child namespaces create longer names, which engineers will dislike typing. Based
on experience, we expect to start seeing aliasing even at name lengths around
six characters long. With longer names, we should expect more aliasing, which in
turn will reduce code readability because more types will have local names.

We believe it's feasible for even large projects to collapse namespaces down to
a top level, avoiding internal tiers of namespaces.

We understand that child namespaces are sometimes helpful, and will robustly
support them for that. However, we will model code organization to encourage
fewer namespaces.

### Redundant markers

We use a few possibly redundant markers for packages and libraries:

-   The `package` keyword requires one of `api` and `impl`, rather than
    excluding either or both.
-   The filename repeats the `api` versus `impl` choice.
-   The `import` keyword requires the full library.

These choices are made to assist human readability and tooling:

-   Being explicit about imports creates the opportunity to generate build
    dependencies from files, rather than having them maintained separately.
-   Being explicit about `api` versus `impl` makes it easier for both humans and
    tooling to determine what to expect.
-   Repeating the type in the filename makes it possible to check the type
    without reading file content.
-   Repeating the type in the file content makes non-filesystem-based builds
    possible.

## Open questions

These open questions are expected to be revisited by future proposals.

### Different file extensions

Currently, we're using `.carbon` and `.impl.carbon`. In the future, we may want
to change the extension, particularly because Carbon may be renamed.

There are several other possible extensions / commands that we've considered in
coming to the current extension:

-   `.carbon`: This is an obvious and unsurprising choice, but also quite long
    for a file extension.
-   `.6c`: This sounds a little like 'sexy' when read aloud.
-   `.c6`: This seems a weird incorrect ordering of the atomic number and has a
    bad, if obscure, Internet slang association.
-   `.cb` or `.cbn`: These collide with several acronyms and may not be
    especially memorable as referring to Carbon.
-   `.crb`: This has a bad Internet slang association.

### Imports from other languages

Currently, we do not support cross-language imports. In the future, we will
likely want to support imports from other languages, particularly for C++
interoperability.

Although we're not designing this right now, it could fit into the proposed
syntax. For example:

```carbon
import Cpp file("myproject/myclass.h");

fn MyCarbonCall(var Cpp.MyProject.MyClass: x);
```

### Imports from URLs

Currently, we don't support any kind of package management with imports. In the
future, we may want to support tagging imports with a URL that identifies the
repository where that package can be found. This can be used to help drive
package management tooling and to support providing a non-name identity for a
package that is used to enable handling conflicted package names.

Although we're not designing this right now, it could fit into the proposed
syntax. For example:

```carbon
import Carbon library("Utilities")
    url("https://github.com/carbon-language/carbon-libraries");
```

### Test file type

Similar to `api` and `impl`, we may eventually want a type like `test`. This
should be part of a larger testing plan.

## Alternatives

### Packages

#### Name paths for package names

Right now, we only allow a single identifier for the package name. We could
allow a full name path without changing syntax.

Advantages:

-   Allow greater flexibility and hierarchy for related packages, such as
    `Database.Client` and `Database.Server`.
-   Would allow using GitHub repo names as package names. For example,
    `carbon-language/carbon-toolchain` could become
    `carbon_language.carbon_toolchain`.

Disadvantages:

-   Multiple identifiers is more complex.
-   Other languages with similar distribution packages don't have a hierarchy,
    and so it may be unnecessary for us.
    -   In other languages that use packages for distribution, they apply
        similar restrictions. For example,
        [Node.JS/NPM](https://www.npmjs.com/), [Python PyPi](https://pypi.org/),
        or [Rust Crates](https://crates.io/).
    -   In [Rust Crates](https://crates.io/), we can observe an example
        `winapi-build` and `winapi-util`.
-   We can build a custom system for reserving package names in Carbon.

At present, we are choosing to use single-identifier package names because of
the lack of clear advantage towards a more complex name path.

#### Referring to the package as `package`

Right now, we plan to refer to the package containing the current file by name.
What's important in the below example is the use of `Math.Stats`:

```carbon
package Math library("Stats") api;
api struct Stats { ... }
struct Quantiles {
  fn Stats();
  fn Build() {
    ...
    var Math.Stats: b;
    ...
  }
}
```

We could instead use `package` as an identifier within the file to refer to the
package, giving `package.Stats`.

It's important to consider how this behaves for `impl` files, which expect an
implicit import of the API. In other words, for `impl` files, this can be
compared to an implicit `import Math;` versus an implicit
`import Math as package;`. However, there may also be _explicit_ imports from
the package, such as `import Math library("Trigonometry");`, which may or may
not be referrable to using `package`, depending on the precise option used.

Advantages:

-   Gives a stable name to refer to the current library's package.
    -   This reduces the amount of work necessary if the current library's
        package is renamed, although imports and library consumers may still
        need to be updated. If the library can also refer to the package by the
        package name, even with imports from other libraries within the package,
        work may not be significantly reduced.
-   The same syntax can be used to refer to entities with the same name as the
    package.
    -   For example, in a
        [package named `DateTime`](https://docs.python.org/3/library/datetime.html#datetime-objects),
        `package.DateTime` is unambiguous, whereas `DateTime.DateTime` could be
        confusing.

Disadvantages:

-   Reuses the `package` keyword with a significantly different meaning,
    changing from a prefix for the required declaration at the top of the file,
    to an identifier within the file.
    -   We don't need to have a special way to refer to the package to
        disambiguate duplicate names. In other words, there is likely to be
        other syntax for referring to an entity `DateTime` in the package
        `DateTime`.
-   Creates inconsistencies as compared to imports from other packages, such as
    `import Widget;`, and imports from the current package, such as
    `import Foo library("Wiz");`.
    -   Option 1: Require `package` to be used to refer to all imports from
        `Foo`, including the current file. This gives consistent treatment for
        the `Foo` package, but not for other imports.
    -   Option 2: Require `package` be used for the current library's entities,
        but not other imports. This gives consistent treatment for imports, but
        not for the `Foo` package as a whole.
    -   Option 3: Allow either `package` or the full package name to refer to
        the current package. This allows code to say either `package` or `Foo`,
        with no enforcement for consistency.

As part of pushing library authors to consider how their package will be used,
we require them to specify the package by name where desired.

#### Remove the `library` keyword from `package` and `import`

Right now, we have syntax like:

```carbon
package Foo library("Bar") api;
package Foo library("Bar") namespace Baz api;
import Foo library("Bar");
```

We could remove `library`, resulting in:

```carbon
package Foo.Bar api;
package Foo.Bar namespace Foo.Baz api;
import Foo.Bar;
```

Advantages:

-   Reduces redundant syntax in library declarations.
    -   We expect libraries to be common, so this may add up.

Disadvantages:

-   Reduces explicitness of package vs library concepts.
-   Creates redundancy of the package name in the namespace declaration.
    -   Instead of `package Foo.Bar namespace Foo.Baz`, could instead use `Baz`,
        or `this.Baz` to elide the package name.

#### Remove the `namespace` keyword from `package`

Right now, we allow syntax like:

```carbon
package Foo namespace Bar.Baz;
fn Wiz() { ... }
```

We could remove `namespace` from the `package` keyword, which would mean there
would no longer be file-scoped namespaces. Specifying child namespaces would
always be required, for example:

```carbon
package Foo;

namespace Bar.Baz;
fn Bar.Baz.Wiz() { ... }
```

Regarding short names, there is agreement that brevity of code is useful. If
this alternative were adopted, it may encourage avoidance of the namespace
feature due to the resulting length of declarations, which would yield desirable
short names for library callers. However, larger libraries and packages are more
likely to run into issues with name collisions on internal names if everything
is in the same namespace, and this alternative would make it more verbose to
write code that avoids name collisions.

Advantages:

-   Encourages short names.
-   Reduces complexity of the `package` keyword.
-   Provides a single mechanism for declaring namespaces in a package, with only
    one meaning for the `namespace` keyword.
    -   Eliminates the file-level namespace context, leaving only the package.
-   Requires that the code as written by the library maintainer more closely
    match how it would be called.
    -   In other words, an `import` will always need some reference, even an
        alias, rooted at the package name. This places a similar requirement on
        the library itself.
    -   This alignment is not a precise match: where `Bar.Baz.Wiz` is declared
        in package `Foo`, the package isn't part of the name path. Library
        consumers would need to use the full `Foo.Bar.Baz.Wiz`.
        -   We could change this, and require package `Foo` always include `Foo`
            in declarations too, including `namespace Foo.Bar.Baz;` instead of
            `namespace Bar.Baz;`.

Disadvantages:

-   Increases verbosity for libraries which use namespaces, as every identifier
    must have the namespace specified.
    -   While this verbosity is partially aligned with what library consumers
        would see, large libraries and packages may be more highly
        self-referential.
        -   Additionally, the package author may want to make some functionality
            accessible but distinctly separate and/or discouraged through a
            namespace. This may improve library usability, but at a verbosity
            cost to the library author.
    -   Although library authors could address this by omitting the namespace,
        that may in turn lead to more name collisions for large packages.
    -   This includes entities declared in `impl` files and package-internal
        libraries that aren't part of the package's API, but may still lead to
        name collisions within the package.
-   May cause significant amounts of aliasing in order to reduce verbosity,
    hindering readability.
    -   Users are known to alias long names, where "long" may be considered
        anything over six characters.
    -   This is a risk for any package that uses namespaces, as importers may
        also need to address it.

Overall, the desire to make it easy to avoid name collisions means we should
allow `namespace` at the file level. We also want to avoid the potential
readability problems of users frequently aliasing name paths.

#### Rename package concept

In other languages, a "package" is equivalent to what we call the name path
here, which includes the `namespace`. We may want to rename the `package`
keyword to avoid conflicts in meaning.

Alternative names could be 'bundle', 'universe', or something similar to Rust's
'crates'; perhaps 'compound' or 'molecule'.

Advantages:

-   Avoids conflicts in meaning with other languages.
    -   [Java](https://www.oracle.com/java/technologies/glossary.html), similar
        to a namespace path.
    -   [Go](https://golang.org/doc/effective_go.html#package-names), similar to
        a namespace path.

Disadvantages:

-   The meaning of `package` also overlaps a fair amount, and we would lose that
    context.
    -   [NPM/Node.js](https://www.npmjs.com/), as a distributable unit.
    -   [Python](https://packaging.python.org/tutorials/installing-packages/),
        as a distributable unit.
    -   [Rust](https://doc.rust-lang.org/book/ch07-01-packages-and-crates.html),
        as a collection of crates.
    -   [Swift](https://developer.apple.com/documentation/swift_packages), as a
        distributable unit.

#### Strict association between the filesystem path and library/namespace

Several languages create a strict association between the filesystem path, and
the method for pulling in an API. For example:

-   In C++, `#include` refers to specific files.
-   In Java, `package` and `import` both reflect filesystem structure.
-   In Python, `import` requires matching filesystem structure.
-   In TypeScript, `import` refers to specific files.

For contrast:

-   In Go, `package` uses an arbitrary name.

Advantages:

-   A strict association between filesystem path and import path makes it easier
    to find source files. This is used by some languages for compilation.

Disadvantages:

-   The strict association makes it harder to move names between files without
    updating callers.
-   If there were a strict association of paths, it would also need to handle
    filesystem-dependent casing behaviors.
    -   For example, on Windows, `project.carbon` and `Project.carbon` are
        conflicting filenames. This is exacerbated by paths, wherein a file
        `config` and a directory `Config/` would conflict, even though this
        would be a valid structure on Unix-based filesystems.

We are choosing to avoid the strict association with filesystem paths in order
to ease refactoring. With this approach,
[more refactorings](#potential-refactorings) will not need changes to imports of
callers.

### Libraries

#### Allow importing implementation files from within the same library

The current proposal is that implementation files in a library implicitly import
their API, and that they cannot import other implementation files in the same
library.

We could instead allow importing implementation files from within the same
library. There are two ways this could be done:

-   We could add a syntax for importing symbols from other files in the same
    library. This would make it easy to identify a directed acyclic graph
    between files in the library. For example:

    ```carbon
    package Geometry;

    import file("point.6c");
    ```

-   We could automatically detect when symbols from elsewhere in the library are
    referenced, given an import of the same library. For example:

    ```carbon
    package Geometry;

    import this;
    ```

Advantages:

-   Allows more separation of implementation between files within a library.

Disadvantages:

-   Neither approach is quite clean:
    -   Using filenames creates a common case where filenames _must_ be used,
        breaking away from name paths.
    -   Detecting where symbols exist may cause separate parsing, compilation
        debugging, and compilation parallelism problems.
-   Libraries are supposed to be small, and we've chosen to only allow one API
    file per library to promote that concept. Encouraging implementation files
    to be inter-dependent appears to support a more complex library design
    again, and may be better addressed through inter-library ACLs.
-   Loses some of the ease-of-use that some other languages have around imports,
    such as Go.
-   Part of the argument towards `api` and `impl`, particularly with a single
    `api`, has been to mirror C++ `.h` and `.cc`. Wherein a `.cc` `#include`-ing
    other `.cc` files is undesirable, allowing a `impl` to import another `impl`
    could be considered similarly.

The problems with these approaches, and encouragement towards small libraries,
is how we reach the current approach of only importing APIs, and automatically.

#### Alternative library separators and shorthand

Examples are using `.` to separator significant terms in library names, and `/`
to separate the package name in shorthand. For example,
`package Time library("Timezones.Internal");` with shorthand
`Time/Timezones.Internal`.

Note that, because the library is an arbitrary string and shorthand is not a
language semantic, this won't affect much. However, users should be expected to
treat examples as best practice.

##### `/` separators

We could instead use `/` for both separators. For example,
`package Foo library("Bar/Baz");` with shorthand `Foo/Bar/Baz`.

Advantages:

-   Only uses one separator, so users don't need to switch.
-   Creates an intuition that libraries are like filesystem paths.

Disadvantages:

-   Obscures distinction between the package and library, reducing readability.
-   We have chosen not to
    [enforce filesystem paths](#strict-association-between-the-filesystem-path-and-librarynamespace)
    in order to ease refactoring, and encouraging a mental model where they may
    match could confuse users.

In this case, the understandability advantages of distinct separators outweighs
the benefit of any singular separator.

##### Single-word libraries

We could stick to single word libraries in examples, such as replacing
`library("Algorithms.Distance")` with `library("Distance")`.

Advantages:

-   Encourages short library names.

Disadvantages:

-   Users are likely to end up doing some hierarchy, and we should address it.
    -   Consistency will improve code understandability.

We might list this as a best practice, and have Carbon only expose libraries
following it. However, some hierarchy from users can be expected, and so it's
worthwhile to include a couple examples to nudge users towards consistency.

#### Collapse API and implementation file concepts

We could remove the distinction between API and implementation files.

Advantages:

-   Removing the distinction between API and implementation would be a language
    simplification.
-   Developers will not need to consider build performance impacts of how they
    are distributing code between files.

Disadvantages:

-   Serializes compilation across dependencies.
    -   May be exacerbated because developers won't be aware of when they are
        adding a dependency that affects imports.
    -   In large codebases, it's been necessary to abstract out API from
        implementation in languages that similarly consolidate files, such as
        Java. However, the lack of language-level support constrains potential
        benefit and increases friction for a split.
-   Whereas an `api`/`impl` hierarchy gives a structure for compilation, if
    there are multiple files we will likely need to provide a different
    structure, perhaps explicit file imports, to indicate intra-library
    compilation dependencies.
    -   We could also effectively concatenate and compile a library together,
        reducing build parallelism options differently.
-   Makes it harder for users to determine what the API is, as they must read
    all the files.

Requiring users to manage the `api`/`impl` split allows us to speed up
compilation for large codebases. This is important for large codebases, and
shouldn't directly affect small codebases that choose to only use `api` files.

##### Automatically generating the API separation

We could try to address the problems with collapsing API and implementation
files by automatically generating an API file from the input files for a
library.

For example, it may preprocess files to split out an API, reducing the number of
imports propagated for _actual_ APIs. For example:

1. Extract `api` declarations within the `api` file.
2. Remove all implementation bodies.
3. Add only the imports that are referenced.

Even under the proposed model, compilation will do some of this work as an
optimization. However, determining which imports are referenced requires
compilation of all imports that _may_ be referenced. When multiple libraries are
imported from a single package, it will be ambiguous which imports are used
until all have been compiled. This will cause serialization of compilation that
can be avoided by having a developer split out the `impl`, either manually or
with developer tooling.

The `impl` files may make it easier to read code, but they will also allow for
better parallelism than `api` files alone can. This does not mean the compiler
will or will not add optimizations -- it only means that we cannot wholly rely
on optimizations by the compiler.

Automatically generating the API separation would only partly mitigate the
serialization of compilation caused by collapsing file and library concepts.
Most of the build performance impact would still be felt by large codebases, and
so the mitigation does not significantly improve
[the alternative](#collapse-api-and-implementation-file-concepts).

#### Collapse file and library concepts

We could collapse the file and library concepts. What this implies is:

-   [Collapse API and implementation file concepts](#collapse-api-and-implementation-file-concepts).
    -   As described there, this approach significantly reduces the ability to
        separate compilation.
-   Only support having one file per library.
    -   The file would need to contain both API and implementation together.

This has similar advantages and disadvantages to
[collapse API and implementation file concepts](#collapse-api-and-implementation-file-concepts).
Differences follow.

Advantages:

-   Offers a uniformity of language usage.
    -   Otherwise, some developers will use only `api` files, while others will
        always use `impl` files.
-   The structure of putting API and implementation in a single file mimics
    other modern languages, such as Java.
-   Simplifies IDEs and refactoring tools.
    -   Otherwise, these systems will need to understand the potential for
        separation of interface from implementation between multiple files.
    -   For example, see [potential refactorings](#potential-refactorings).

Disadvantages:

-   Avoids the need to establish a hierarchy between files in a library, at the
    cost of reducing build parallelism options further.
-   While both API and implementation is in the same file, it can be difficult
    to visually identify the API when it's mixed with a lengthy implementation.

As with
[collapse API and implementation file concepts](#collapse-api-and-implementation-file-concepts),
we consider the split to be important for large codebases. The additional
advantages of a single-file restriction do not outweigh the disadvantages
surrounding build performance.

#### Collapse the library concept into packages

We could only have packages, with no libraries. Some other languages do this;
for example, in Node.JS, a package is often similar in size to what we currently
call a library.

If packages became larger, that would lead to compile-time bottlenecks. Thus, if
Carbon allowed large packages without library separation, we would undermine our
goals for fast compilation. Even if we combined the concepts, we should expect
it's by turning the "package with many small libraries" concept into "many small
packages".

Advantages:

-   Simplification of organizational hierarchy.
    -   Less complexity for users to think about on imports.

Disadvantages:

-   Coming up with short, unique package names may become an issue, leading to
    longer package names that overlap with the intent of libraries.
    -   These longer package names would need to be used to refer to contained
        entities in code, affecting brevity of Carbon code. The alternative
        would be to expect users to always alias packages on import using `as`;
        some organizations anecdotally see equivalent happen for C++ once names
        get longer than six characters.
    -   For example, [boost](https://github.com/boostorg) could use
        per-repository packages like `BoostGeometry` and child libraries like
        `algorithms-distance` under the proposed approach. Under the alternative
        approach, it would use either a monolithic package that could create
        compile-time bottlenecks, or packages like
        `BoostGeometryAlgorithmsDistance` for uniqueness.
-   While a package manager will need a way to specify cross-package version
    compatibility, encouraging a high number of packages puts more weight and
    maintenance cost on the configuration.
    -   We expect libraries to be versioned at the package-level.

We prefer to keep the library separation to enable better hierarchy for large
codebases, plus encouraging small units of compilation. It's still possible for
people to create small Carbon packages, without breaking it into multiple
libraries.

#### Collapse the package concept into libraries

Versus
[collapse the library concept into packages](#collapse-the-library-concept-into-packages),
we could have libraries without packages. Under this model, we still have
libraries of similar granularity as what's proposed. However, there is no
package grouping to them: there are only libraries which happen to share a
namespace.

For example:

-   `package` vs `library`:
    -   Trivial:
        -   Proposal: `package Foo;`
        -   Alternative: `library "Foo" namespace Foo;`
    -   Multi-layer library:
        -   Proposal: `package Foo library "Bar";`
        -   Alternative: `library "Foo/Bar" namespace Foo;`
    -   Specifying namespaces:
        -   Proposal: `package Foo namespace Baz;`
        -   Alternative: `library "Foo" namespace Foo.Baz;`
    -   Combined:
        -   Proposal: `package Foo library("Bar") namespace Baz;`
        -   Alternative: `library "Foo/Bar" namespace Foo.Baz;`
-   `import` changes:
    -   Trivial:
        -   Proposal: `import Foo;`
        -   Alternative: `import "Foo";`
    -   Multi-layer library:
        -   Proposal: `import Foo library("Bar");`
        -   Alternative: `import "Foo/Bar";`
    -   Namespaces have no effect on `import` under both approaches.

References to imports from other top-level namespaces would need to be prefixed
with a '`.`' in order to make it clear which symbols were from imports. For
example:

```carbon
library "Foo" namespace Foo;

import "Bar";

fn DoSomething(.Bar.Baz x) { ... }
```

This `.` is required because it indicates the symbol comes from a different
top-level namespace. Imports from the same top-level namespace do not need the
dot. For example:

```carbon
library "Foo/Bar" namespace Foo.Child;

import "Foo/Wiz";

// Only valid if Baz is declared in Foo/Bar.
fn DoSomething(Baz x) { ... }

// May come from either Foo/Bar or Foo/Wiz.
fn DoSomething(Foo.Child.Baz x) { ... }
```

We assume that the compiler will enforce that the root namespace must either
match or be a prefix of the library name, followed by a `/`. For example, `Foo`
in `Foo.Baz` must either match a `library "Foo"` or prefix as
`library "Foo/..."`; `library "FooBar"` does not match because it's missing the
`/`.

There are several approaches which might remove this duplication, but each has
been declined due to flaws:

-   We could have `library "Foo";` imply `namespace Foo`. However, we want name
    paths to use things listed as identifiers in files. We specifically do not
    want to use strings to generate identifiers in order to maintain
    readability.
-   We could alternately have `namespace Foo;` syntax imply
    `library "Foo" namespace Foo;`.
    -   This approach only helps with single-library namespaces. While this
        would be common enough that a special syntax would help some developers,
        we are likely to encourage multiple libraries per namespace as part of
        best practices. We would then expect that the quantity of libraries in
        multi-library namespaces would dominate cost-benefit, leaving this to
        address only an edge-case of duplication issues.
    -   This would create an ambiguity between the file-level `namespace` and
        other `namespace` keyword use. We could then rename the `namespace`
        argument for `library` to something like `file-namespace`.
    -   It may be confusing as to what `namespace Foo.Bar;` does. It may create
        `library "Foo/Bar"` because `library "Foo.Bar"` would not be legal, but
        the change in characters may in turn lead to developer confusion.
        -   We could change the library specification to use `.` instead of `/`
            as a separator, but that may lead to broader confusion about the
            difference between libraries and namespaces.

Advantages:

-   Avoids introducing the "package" concept to code and name organization.
    -   Retains the key property that library and namespace names have a prefix
        that is intended to be globally unique.
    -   Avoids coupling package management to namespace structure. For example,
        it would permit a library collection like Boost to be split into
        multiple repositories and multiple distribution packages, while
        retaining a single top-level namespace.
-   The library and namespace are pushed to be more orthogonal concepts than
    packages and namespaces.
    -   Although some commonality must still be compiler-enforced.
-   For the common case where packages have multiple libraries, removing the
    need to specify both a package and library collapses two keywords into one
    for both `import` and `package`.
-   It makes it easier to draw on C++ intuitions, because all the concepts have
    strong counterparts in C++.
-   The prefix `.` on imported name paths can help increase readability by
    making it clear they're from imports, so long as those imports aren't from
    the current top-level namespace.
-   Making the `.` optional for imports from the current top-level namespace
    eliminates the boilerplate character when calling within the same library.

Disadvantages:

-   The use of a leading `.` to mark absolute paths may conflict with other
    important uses, such as designated initializers and named parameters.
-   Declines an opportunity to align code and name organization with package
    distribution.
    -   Alignment means that if a developer sees `package Foo library("Bar");`,
        they know installing a package `Foo` will give them the library.
        Declining this means that users seeing `library "Foo/Bar"`, they will
        still need to do research as to what package contains `Foo/Bar` to
        figure out how to install it because that package may not be named
        `Foo`.
    -   Package distribution is a
        [project goal](/docs/project/goals.md#language-tools-and-ecosystem), and
        cannot be avoided indefinitely.
    -   This also means multiple packages may contribute to the same top-level
        namespace, which would prevent things like tab-completion in IDEs from
        optimizing based on the knowledge that modified packages cannot add to a
        given top-level namespace.
        -   For example, if a user is editing a package `Foo`, package
            boundaries would mean they could not add to namespace `Bar`. Under
            this alternative, that guarantee only exists at library granularity,
            meaning that IDEs will need to be able to combine information from
            multiple packages to determine which libraries contribute to
            namespace `Bar`.
-   The string prefix enforcement between `library` and `namespace` forces
    duplication between both, which would otherwise be handled by `package`.
-   For the common case of packages with a matching namespace name, increases
    verbosity by requiring the `namespace` keyword.
-   The prefix `.` on imported name paths will be repeated frequently through
    code, increasing overall verbosity, versus the package approach which only
    affects import verbosity.
-   Making the `.` optional for imports from the current top-level namespace
    hides whether an API comes from the current library or an import.

We are declining this approach because we desire package separation, and because
of concerns that this will lead to an overall increase in verbosity due to the
[preference for few child namespaces](#preference-for-few-child-namespaces),
whereas this alternative benefits when `namespace` is specified more often.

#### Different file type labels

We're using `api` and `impl` for file types, and have `test` as an open
question.

We've considered using `interface` instead of `api`, but that introduces a
terminology collision with interfaces in the type system.

We've considered dropping `api` from naming, but that creates a definition from
absence of a keyword. It also would be more unusual if both `impl` and `test`
must be required, that `api` would be excluded. We prefer the more explicit
name.

We could spell out `impl` as `implementation`, but are choosing the abbreviation
for ease of typing. We also don't think it's an unclear abbreviation.

We expect `impl` to be used for implementations of `interface`. This isn't quite
as bad as if we used `interface` instead of `api` because of the `api` export
syntax on entities, such as `api fn Foo()`, which could create ambiguities as
`interface fn Foo()`. It may still confuse people to see an `interface impl` in
an `api` file. However, we're touching on related concepts and don't see a great
alternative.

#### Function-like syntax

We could consider more function-like syntax for `import`, and possibly also
`package`.

For example, instead of:

```carbon
import Foo library("Bar");
import Baz as B;
```

We could do:

```carbon
import("Foo", "Bar").Foo;
alias B = import("Baz").Baz;
```

Advantages:

-   Allows straightforward reuse of `alias` for language consistency.
-   Easier to add more optional arguments, which we expect to need for
    [interoperability](#imports-from-other-languages) and
    [URLs](#imports-from-urls).
-   Avoids defining keywords for optional fields: `library`, `as`, and possibly
    more long-term.

Disadvantages:

-   It's unusual for a function-like syntax to produce identifiers for name
    lookup.
    -   This could be addressed by _requiring_ alias, but that becomes verbose.
    -   There's a desire to explicitly note the identifier being imported some
        way, as with `.Foo` and `.Baz` above. However, this complicates the
        resulting syntax.

The preference is for keywords.

#### Inlining from implementation files

An implicit reason for keeping code in an `api` file is that it makes it
straightforward to inline code from there into callers.

We could explicitly encourage inlining from `impl` files as well, making the
location of code unimportant during compilation. Alternately, we could add an
`inline` file type which explicitly supports separation of inline code from the
`api` file.

Advantages:

-   Allows moving code out of the main API file for easier reading.

Disadvantages:

-   Requires compilation of `impl` files to determine what can be inlined from
    the `api` file, leading to the transitive closure dependency problems which
    `impl` files are intended to avoid.

We expect to only support inlining from `api` files in order to avoid confusion
about dependency problems.

#### Library-private access controls

We currently have no special syntax for library-private APIs. However,
non-exported APIs are essentially library-private, and may be in the `api` file.
It's been suggested that we could either provide a special syntax or a new file
type, such as `shared_impl`, to support library-private APIs.

Advantages:

-   Allows for better separation of library-private APIs.

Disadvantages:

-   Increases language complexity.
-   Dependencies are still an issue for library-private APIs.
    -   If used from the `api` file, the dependencies are still in the
        transitive closure of client libraries, and any separation may confuse
        users about the downsides of the extra dependencies.
    -   If only used from `impl` files, then they could be in the `impl` file if
        there's only one, or shared from a separate library.
-   Generalized access controls may provide overlapping functionality.

At this point in time, we prefer not to provide specialized access controls for
library-private APIs.

#### Managing API versus implementation in libraries

At present, we plan to have `api` versus `impl` as a file type, and also
`.carbon` versus `.impl.carbon` as the file extension. We chose to use both
together, rather than one or the other, because we expect some parties to
strongly want file content to be sufficient for compilation, while others will
want file extensions to be meaningful for the syntax split.

Instead of the file type split, we could drift further and instead have APIs in
any file in a library, using the same kind of
[API markup](#exporting-entities-from-an-api-file).

Advantages:

-   May help users who have issues with cyclical code references.
-   Improves compiler inlining of implementations, because the compiler can
    decide how much to actually put in the generated API.

Disadvantages:

-   While allowing users to spread a library across multiple files can be
    considered an advantage, we see the single API file as a way to pressure
    users towards smaller libraries, which we prefer.
-   May be slower to compile because each file must be parsed once to determine
    APIs.
-   For users that want to see _only_ APIs in a file, they would need to use
    tooling to generate the API file.
    -   Auto-generated documentation may help solve this problem.

#### Multiple API files

The proposal also presently suggests a single API file. Under an explicit API
file approach, we could still allow multiple API files.

Advantages:

-   More flexibility when writing APIs; could otherwise end up with one gigantic
    API file.

Disadvantages:

-   Encourages larger libraries by making it easier to provide large APIs.
-   Removes some of the advantages of having an API file as a "single place" to
    look, suggesting more towards the markup approach.
-   Not clear if API files should be allowed to depend on each other, as they
    were intended to help resolve cyclical dependency issues.

We particularly want to discourage large libraries, and so we're likely to
retain the single API file limit.

#### Name paths as library names

We're proposing strings for library names. We've discussed also using name paths
(`My.Library`) and also restricting to single identifiers (`Library`).

Advantages:

-   Shares the form between packages (identifiers) and namespaces (name paths).
-   Enforces a constrained set of names for libraries for cross-package
    consistency of naming.

Disadvantages:

-   Indicates that a library may be referred to in code, when only the package
    and namespace are used for name paths of entities.
-   The constrained set of names may also get in the way for some packages that
    can make use of more flexibility in naming.

We've decided to use strings primarily because we want to draw the distinction
that a library is not something that's used when referring to an entity in code.

### Namespaces

#### Coarser namespace granularity

It's been discussed whether we need to provide namespaces outside of
package/file granularity. In other words, if a file is required to only add to
one namespace, then there's no need for a `namespace` keyword or similar.

Advantages:

-   Requiring files to contribute to only one namespace offers a language
    simplification.

-   Library interface vs implementation separation may be used to address some
    problems that namespaces have been used for in C++.

Disadvantages:

-   One point made was the difficulty in C++ of doing `friend` declarations for
    template functions, making ACL controls difficult. Putting template
    functions in a namespace such as `internal` allows for an implicit warning
    about access misuse. We may end up with similar problems and solutions in
    Carbon. It's preferable that both the template and the functions it calls be
    allowed to be in the same file, and we don't want to prevent that.

    -   Namespaces named `internal` or similar may also be used to hide certain
        calls from IDEs.

-   It's not clear that file-granularity namespaces would make it easy to
    address potential circular references in code.

-   Makes it more difficult to migrate C++ code, which should be expected to
    sometimes have multiple namespaces within a file.

-   Combined with the restriction that a library only provides a single API
    file, it means APIs will only be able to add to a single namespace. For
    example, a library adding to `Geometry.Shapes` could not also add to
    `Geometry.Shapes.Flat`.

We believe that while we may find better solutions for _some_ use-cases of
fine-grained namespaces, but not all. The proposed `namespace` syntax is a
conservative solution to the problem which we should be careful doesn't add too
much complexity.

#### Scoped namespaces

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

Advantages:

-   Makes it easy to write many things in the same namespace.

Disadvantages:

-   It's not clear which namespace an identifier is in without scanning to the
    start of the file.
-   It can be hard to find the end of a namespace. For examples addressing this,
    end-of-namespace comments are called for by both the
    [Google](https://google.github.io/styleguide/cppguide.html#Namespaces) and
    [Boost](https://github.com/boostorg/geometry/wiki/Guidelines-for-Developers)
    style guides.
    -   Carbon may disallow the same-line-as-code comment style used for this.
        Even if not, if we acknowledge it's a problem, we should address it
        structurally for
        [readability](/docs/projects/goals.md#code-that-is-easy-to-read-understand-and-write).
    -   This is less of a problem for other scopes, such as functions, because
        they can often be broken apart until they fit on a single screen.

There are other ways to address the con, such as adding syntax to indicate the
end of a namespace, similar to block comments. For example:

```carbon
{ namespace Foo
  { namespace Bar.Baz
    struct Wiz { ... }
  } namespace Bar.Baz
} namespace Foo

fn Fez(Foo.Bar.Baz.Wiz x);
```

While we could consider such alternative approaches, we believe the proposed
contextless namespace approach is better, as it reduces information that
developers will need to remember when reading/writing code.

### Imports

#### Block imports

Rather than requiring an `import` keyword per line, we could support block
imports, as can be found in languages like Go.

In other words, instead of:

```carbon
import Bar;
import Foo;
```

We could have:

```carbon
imports {
  Bar,
  Foo,
}
```

Advantages:

-   Allows repeated imports with less typing.

Disadvantages:

-   Makes it harder to find files importing a package or library using tools
    like `grep`.

One concern has been that a mix of `import` and `imports` syntax would be
confusing to users: we should only allow one.

This alternative has been declined because retyping `import` statements is
low-cost, and `grep` is useful.

#### Block imports of libraries of a single package

We could allow block imports of librarys from the same package. For example:

```carbon
import Foo libraries({
  "Bar",
  "Baz",
})
```

This may help readability of code around `alias`. In particular, consider a
package trying to alias the API of another package:

-   Package X:

    ```carbon
    package X;
    import Foo library "blah1";
    import Foo library "blah2";
    api alias Bar = Foo;
    ```

-   Package Y:

    ```carbon
    package Y;
    import X;
    fn Run() { X.Bar.Baz(); }
    ```

The result of this `api alias` allowing `X.Bar.Baz()` to work regardless of
whether `Baz` is in `"blah1"` or `"blah2"` may be clearer if both `import Foo`
statements were a combined `import Foo libraries({"blah1", "blah2"});`. Note
that this is also a case where allowing different `as` values for different
libraries of the same package may be helpful, as it would allow `alias` to
provide `"blah1"` while allowing private, local use of `"blah2"`.

The advantages/disadvantages are similar to [block imports](#block-imports).
Additional advantages/disadvantages are:

Advantages:

-   Avoids repeating `as` for a package.
-   If we limit to one import per library, then any `alias` of the package `Foo`
    is easier to understand as affecting all libraries.

Disadvantages:

-   If we allow both `library` and `libraries` syntax, it's two was of doing the
    same thing.
    -   Can be addressed by _always_ requiring `libraries`, removing `library`,
        but that diverges from `package`'s `library` syntax.

This alternative has been declined for similar reasons to block imports; the
additional advantages/disadvantages don't substantially shift the cost-benefit
argument.

#### Broader imports, either all names or arbitrary code

Carbon imports require specifying individual names to import. We could support
broader imports, for example by pulling in all names from a library. In C++, the
`#include` preprocessor directive even supports inclusion of arbitrary code. For
example:

```carbon
import Geometry library("Shapes") names *;

// Triangle was imported as part of "*".
fn Draw(var Triangle: x) { ... }
```

Advantages:

-   Reduces boilerplate code specifying individual names.

Disadvantages:

-   Loses out on parser benefits of knowing which identifiers are being
    imported.
-   Increases the risk of adding new features to APIs, as they may immediately
    get imported by a user and conflict with a pre-existing name, breaking code.
-   As the number of imports increases, it can become difficult to tell which
    import a particular symbol comes from, or how imports are being used.
-   Arbitrary code inclusion can result in unexpected code execution, a way to
    create obfuscated code and a potential security risk.

We particularly value the parser benefits of knowing which identifiers are being
imported, and so we require individual names for imports.

#### Direct name imports

We could allow direct imports of names from libraries. For example, under the
current setup we might see:

```carbon
import Foo library("Bar");
alias Baz = Foo.Baz;
alias Wiz = Foo.Wiz;
```

We could simplify this syntax by augmenting `import`:

```carbon
import Foo library("Bar") name Baz;
import Foo library("Bar") name Wiz;
```

Or more succinctly with block imports of names:

```carbon
import Foo library("Bar") names {
  Baz,
  Wiz,
}
```

Advantages:

-   Avoids an additional `alias` step.

Disadvantages:

-   With a single name, this isn't a significant improvement in syntax.
-   With multiple names, this runs into similar issues as
    [block imports](#block-imports).

#### Optional package names

We could allow a short syntax for imports from the current library. For example,
this code imports `Geometry.Shapes`:

```carbon
package Geometry library("Operations") api;

import library("Shapes");
```

Advantages:

-   Reduces typing.

Disadvantages:

-   Makes it harder to find files importing a package or library using tools
    like `grep`.
-   Creates two syntaxes for importing libraries from the current package.
    -   If we instead disallow `import Geometry library("Shapes")` from within
        `Geometry`, then we end up with a different inconsistency.

Overall, consistent with the decision to disallow
[block imports](#block-imports), we are choosing to require the package name.
