# Code and name organization

<!--
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

<!-- toc -->

## Table of contents

-   [Goals and philosophy](#goals-and-philosophy)
-   [Overview](#overview)
    -   [Sizing packages and libraries](#sizing-packages-and-libraries)
    -   [Imports](#imports)
-   [Details](#details)
    -   [Source file introduction](#source-file-introduction)
    -   [Name paths](#name-paths)
        -   [`package` syntax](#package-syntax)
    -   [Packages](#packages)
        -   [Shorthand notation for libraries in packages](#shorthand-notation-for-libraries-in-packages)
        -   [Package name conflicts](#package-name-conflicts)
    -   [Libraries](#libraries)
        -   [Exporting entities from an API file](#exporting-entities-from-an-api-file)
        -   [Granularity of libraries](#granularity-of-libraries)
        -   [Exporting namespces](#exporting-namespces)
    -   [Imports](#imports-1)
        -   [Imports from the current package](#imports-from-the-current-package)
    -   [Namespaces](#namespaces)
        -   [Re-declaring imported namespaces](#re-declaring-imported-namespaces)
        -   [Aliasing](#aliasing)
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
        -   [Rename package concept](#rename-package-concept)
        -   [No association between the file system path and library/namespace](#no-association-between-the-file-system-path-and-librarynamespace)
    -   [Libraries](#libraries-1)
        -   [Allow exporting namespaces](#allow-exporting-namespaces)
        -   [Allow importing implementation files from within the same library](#allow-importing-implementation-files-from-within-the-same-library)
        -   [Alternative library separators and shorthand](#alternative-library-separators-and-shorthand)
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
    -   [Imports](#imports-2)
        -   [Block imports](#block-imports)
        -   [Block imports of libraries of a single package](#block-imports-of-libraries-of-a-single-package)
        -   [Broader imports, either all names or arbitrary code](#broader-imports-either-all-names-or-arbitrary-code)
        -   [Direct name imports](#direct-name-imports)
        -   [Optional package names](#optional-package-names)
    -   [Namespaces](#namespaces-1)
        -   [File-level namespaces](#file-level-namespaces)
        -   [Scoped namespaces](#scoped-namespaces)

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

Carbon [source files](source_files.md) have a `.carbon` extension, such as
`geometry.carbon`. These files are the basic unit of compilation.

Each file begins with a declaration of which
_package_<sup><small>[[define](/docs/guides/glossary.md#package)]</small></sup>
it belongs in. The package is the unit of _distribution_. The package name is a
single identifier, such as `Geometry`. An example API file in the `Geometry`
package would start with:

```
package Geometry api;
```

A tiny package may consist of a single library with a single file, and not use
any further features of the `package` keyword.

It is often useful to use separate files for the API and its implementation.
This may help organize code as a library grows, or to let the build system
distinguish between the dependencies of the API itself and its underlying
implementation. Implementation files allow for code to be extracted out from the
API file, while only being callable from other files within the library,
including both API and implementation files. Implementation files are marked by
both naming the file to use an extension of `.impl.carbon` and instead start
with:

```
package Geometry impl;
```

However, as a package adds more files, it will probably want to separate out
into multiple
_libraries_<sup><small>[[define](/docs/guides/glossary.md#library)]</small></sup>.
A library is the basic unit of _dependency_. Separating code into multiple
libraries can speed up the overall build while also making it clear which code
is being reused. For example, an API file adding the library `Shapes` to the
`Geometry` package, or `Geometry//Shapes` in
[shorthand](#shorthand-notation-for-libraries-in-packages), would start with:

```
package Geometry library "Shapes" api;
```

As code becomes more complex, and users pull in more code, it may also be
helpful to add
_namespaces_<sup><small>[[define](/docs/guides/glossary.md#namespace)]</small></sup>
to give related entities consistently structured names. A namespace affects the
_name
path_<sup><small>[[define](/docs/guides/glossary.md#name-path)]</small></sup>
used when calling code. For example, with no namespace, if a `Geometry` package
defines `Circle` then the name path will be `Geometry.Circle`. However, it can
be named `Geometry.TwoDimensional.Circle` with a `namespace`; for example:

```
package Geometry library "Shapes" api;
namespace TwoDimensional;
struct TwoDimensional.Circle { ... };
```

This scaling of packages into libraries and namespaces is how Carbon supports
both small and large codebases.

### Sizing packages and libraries

A different way to think of the sizing of packages and libraries is:

-   A package is a GitHub repository.
    -   Small and medium projects that fit in a single repository will typically
        have a single package. For example, a medium-sized project like
        [Abseil](https://github.com/abseil/abseil-cpp/tree/master/absl) could
        still use a single `Abseil` package.
    -   Large projects will have multiple packages. For example, Mozilla may
        have multiple packages for Firefox and other efforts.
-   A library is a few files that provide an interface and implementation, and
    should remain small.
    -   Small projects will have a single library when it's easy to maintain all
        code in a few files.
    -   Medium and large projects will have multiple libraries. For example,
        [Boost Geometry's Distance](https://github.com/boostorg/geometry/blob/develop/include/boost/geometry/algorithms/detail/distance/interface.hpp)
        interface and implementation might be its own library within `Boost`,
        with dependencies on other libraries in `Boost` and potentially other
        packages from Boost.
        -   Library names could be named after the feature, such as
            `library "Algorithms"`, or include part of the path to reduce the
            chance of name collisions, such as `library "Geometry/Algorithms"`.

Packages may choose to expose libraries that expose unions of interfaces from
other libraries within the package. However, doing so would also provide the
transitive closure of build-time dependencies, and is likely to be discouraged
in many cases.

### Imports

The `import` keyword supports reusing code from other files and libraries.

For example, to use `Geometry.Circle` from the `Geometry//Shapes` library:

```carbon
import Geometry library "Shapes";

fn Area(Geometry.Circle circle) { ... };
```

The `library` keyword is optional for `import`, and its use should parallel that
of `library` on the `package` of the code being imported.

## Details

### Source file introduction

Every source file will consist of, in order:

1. One `package` statement.
2. A section of zero or more `import` statements.
3. Source file body, with other code.

Comments and blank lines may be intermingled with these sections.
[Metaprogramming](/docs/design/metaprogramming.md) code may also be
intermingled, so long as the outputted code is consistent with the enforced
ordering. Other types of code must be in the source file body.

### Name paths

[Name paths](#name-paths) are defined above as sequences of identifiers
separated by dots. This syntax may be loosely expressed as a regular expression:

```regex
IDENTIFIER(\.IDENTIFIER)*
```

Name conflicts are addressed by [name lookup](/docs/design/name_lookup.md).

#### `package` syntax

### Packages

The `package` keyword's syntax may be loosely expressed as a regular expression:

```regex
package IDENTIFIER (library STRING)? (api|impl);
```

For example:

```carbon
package Geometry library "Objects/FourSides" api;
```

Breaking this apart:

-   The identifier passed to the `package` keyword, `Geometry`, is the package
    name and will prefix both library and namespace paths.
    -   The `package` keyword also declares a package entity matching the
        package name. A package entity is almost identical to a namespace
        entity, except with some package/import-specific handling. In other
        words, if the file declares `struct Line`, that may be used from within
        the file as both `Line` directly and `Geometry.TwoDimensional.Line`
        using the `Geometry` package entity created by the `package` keyword.
-   When the optional `library` keyword is specified, sets the name of the
    library within the package. In this example, the
    `Geometry//Objects/FourSides` library will be used.
-   The use of the `api` keyword indicates this is an API files as described
    under [libraries](#libraries). If it instead had `impl`, this would be an
    implementation file.

Because the `package` keyword must be specified exactly once in all files, there
are a couple important and deliberate side-effects:

-   Every file will be in precisely one library.
    -   A library still exists even when there is no explicit library argument,
        such as `package Geometry api;`. This could be considered equivalent to
        `package Geometry library "" api;`, although we should not allow that
        specific syntax as error-prone.
-   Every entity in Carbon will be in a namespace, even if its namespace path
    consists of only the package name. There is no "global" namespace.
    -   Every entity in a file will be defined within the namespace described in
        the `package` statement.
    -   Entities within a file may be defined in
        [child namespaces](#namespaces).

Files contributing to the `Geometry//Objects/FourSides` library must all start
with `package Geometry library "Objects/FourSides"`, but will differ on
`api`/`impl` types.

#### Shorthand notation for libraries in packages

Library names may also be referred to as `PACKAGE//LIBRARY` as shorthand in
text. `PACKAGE//default` will refer to the name of the library used when no
`library` argument is specified, although `PACKAGE` may also be used in
situations where it is unambiguous that it still refers to the default library.

It's recommended that libraries use a single `/` for separators where desired,
in order to distinguish between the `//` of the package and `/` separating
library segments. For example, `Geometry//Objects/FourSides` uses a single `/`
to separate the `Object/FourSides` library name.

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

Note that [imported name conflicts](#imported-name-conflicts) are handled
differently.

### Libraries

Every Carbon library consists of one or more files. Each Carbon library has a
primary file that defines its API, and may optionally contain additional files
that are implementation.

-   An API file's `package` will have `api`. For example,
    `package Geometry library "Shapes" api;`
    -   API filenames must have the `.carbon` extension. They must not have a
        `.impl.carbon` extension.
    -   API file paths will correspond to the library name.
        -   The precise form of this correspondence is undetermined, but should
            be expected to be similar to a "Math/Algebra" library being in a
            "Math/Algebra.carbon" file path.
        -   The package will not be used when considering the file path.
-   An implementation file's `package` will have `impl`. For example,
    `package Geometry library "Shapes" impl;`.
    -   Implementation filenames must have the `.impl.carbon` extension.
    -   Implementation file paths need not correspond to the library name.
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
package Geometry library "Shapes" api;

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

The compilation graph of Carbon will generally consist of `api` files depending
on each other, and `impl` files depending only on `api` files. Compiling a given
file requires compiling the transitive closure of `api` files first.
Parallelization of compilation is then limited by how large that transitive
closure is, in terms of total volume of code rather than quantity. This also
affects build cache invalidation.

In order to maximize opportunities to improve compilation performance, we will
encourage granular libraries. Conceptually, we want libraries to be very small,
possibly containing only a single class. The choice of only allowing a single
`api` file per library should help encourage developers to write small
libraries.

#### Exporting namespces

Any entity may be marked with `api` except for namespace and package entities.
That is, `api namespace Sha256;` is invalid code. Instead, namespaces are
implicitly exported based on the name paths of other entities marked as `api`.

For example, given this code:

```carbon
package Checksums library "Sha" api;

namespaces Sha256;

api fn Sha256.HexDigest(Bytes data) -> String { ... }
```

Calling code may look like:

```carbon
package Caller api;

import Checksums library "Sha";

fn Process(Bytes data) {
  ...
  var String digest = Checksums.Sha256.HexDigest(data);
  ...
}
```

In this example, the `Sha256` namespace is exported as part of the API
implicitly.

### Imports

The `import` keyword supports reusing code from other files and libraries. The
`import` keyword's syntax may be loosely expressed as a regular expression:

```regex
import IDENTIFIER (library NAME_PATH)?;
```

An import declares a package entity named after the imported package, and makes
`api`-tagged entities from the imported library through it. The full name path
is a concatenation of the names of the package entity, any namespace entities
applied, and the final entity addressed. Child namespaces or entities may be
[aliased](/docs/design/aliases.md) if desired.

For example, given a library:

```carbon
package Math api;
namespace Trigonometry;
api fn Trigonometry.Sin(...);
```

Calling code would import it and use it like:

```carbon
package Geometry api;

import Math;

fn DoSomething() {
  ...
  Math.Trigonometry.Sin(...);
  ...
}
```

Repeat imports from the same package reuse the same package entity. For example,
this produces only one `Math` package entity:

```carbon
import Math;
import Math library "Trigonometry";
```

#### Imports from the current package

Entities defined in the current file may be used without mentioning the package
prefix. However, other symbols from the package must be imported and accessed
through the package namespace just like symbols from any other package.

For example:

```carbon
package Geometry api;

// This is required even though it's still in the Geometry package.
import Geometry library "Shapes";

// Circle must be referenced using the Geometry namespace of the import.
fn GetArea(Geometry.Circle c) { ... }
```

### Namespaces

Namespaces offer named paths for entities. Namespaces may be nested. Multiple
libraries may contribute to the same namespace. In practice, packages may have
namespaces such as `Testing` containing entities that benefit from an isolated
space but are present in many libraries.

The `namespace` keyword's syntax may loosely be expressed as a regular
expression:

```regex
namespace NAME_PATH;
```

The `namespace` keyword declares a namespace entity. The namespace is applied to
other entities by including it as a prefix when declaring a name. For example:

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

#### Re-declaring imported namespaces

Namespaces may exist on imported package entities, in addition to being declared
in the current file. However, even if the namespace already exists in an
imported library from the current package, the namespace must still be declared
locally in order to add symbols to it.

For example, if the `Geometry//Shapes/ThreeSides` library provides the
`Geometry.Shapes` namespace, this code is still valid:

```carbon
package Geometry library "Shapes/FourSides" api;

import Geometry library "Shapes/ThreeSides";

// This does not conflict with the existence of `Geometry.Shapes` from
// `Geometry//Shapes/ThreeSides`, even though the name path is identical.
namespace Shapes;

// This requires the above 'namespace Shapes' declaration. It cannot use
// `Geometry.Shapes` from `Geometry//Shapes/ThreeSides`.
struct Shapes.Square { ... };
```

#### Aliasing

Carbon's [alias keyword](/docs/design/aliases.md) will support aliasing
namespaces. For example, this would be valid code:

```carbon
namespace Timezones.Internal;
alias TI = Timezones.internal;

struct TI.RawData { ... }
fn ParseData(TI.RawData data);
```

## Caveats

### Package and library name conflicts

Library name conflicts should not occur, because it's expected that a given
package is maintained by a single organization. It's the responsibility of that
organization to maintain unique library names within their package.

A package name conflict occurs when two different packages use the same name,
such as two packages named `Stats`. Versus libraries, package name conflicts are
more likely because two organizations may independently choose identical names.
We will encourage a unique package naming scheme, such as maintaining a name
server for open source packages. Conflicts can also be addressed by renaming one
of the packages, either at the source, or as a local modification.

We do need to address the case of package names conflicting with other entity
names. It's possible that a pre-existing `api` entity will conflict with a new
import, and that the `api` is infeasible to rename due to existing callers.
Alternately, the `api` entity may be using an idiomatic name that it would
contradict naming conventions to rename. In either case, this conflict may exist
in a single file without otherwise affecting users of the API. This will be
addressed by [name lookup](/docs/design/name_lookup.md).

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

-   Move a non-`api`-labeled declaration from an `api` file to an `impl` file.

    -   The declaration must be moved to the same file as the implementation of
        the declaration.
    -   The declaration can only be used by the `impl` file that now contains
        it. Search for other callers within the library, and fix them first.
    -   [Update imports](#update-imports).

-   Move a non-`api`-labeled declaration from an `impl` file to an `api` file.

    -   This should be a local change that will not affect any calling code.
    -   [Update imports](#update-imports).

-   Move a declaration and implementation from one `impl` file to another.

    -   Search for any callers within the source `impl` file, and either move
        them too, or fix them first.
    -   [Update imports](#update-imports).

#### Other refactorings

-   Rename a package.

    -   The imports of all calling files must be updated accordingly.
    -   All call sites must be changed, as the package name changes.
    -   [Update imports](#update-imports).

-   Move an `api`-labeled declaration and implementation between different
    packages.

    -   The imports of all calling files must be updated accordingly.
    -   All call sites must be changed, as the package name changes.
    -   [Update imports](#update-imports).

-   Move an `api`-labeled declaration and implementation between libraries in
    the same package.

    -   The imports of all calling files must be updated accordingly.
    -   As long as the namespaces remain the same, no call sites will need to be
        changed.
    -   [Update imports](#update-imports).

-   Rename a library.

    -   This is equivalent to a repeated operation of moving an `api`-labeled
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
    -   This additionally requires the steps to rename a library, because
        library names must correspond to the renamed paths.

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
-   Repeating the type in the file content makes non-file-system-based builds
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

fn MyCarbonCall(Cpp.MyProject.MyClass x);
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
import Carbon library "Utilities"
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
-   Would allow using GitHub repository names as package names. For example,
    `carbon-language/carbon-lang` could become `carbon_language.carbon_lang`.

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
package Math library "Stats" api;
api struct Stats { ... }
struct Quantiles {
  fn Stats();
  fn Build() {
    ...
    var Math.Stats b;
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
the package, such as `import Math library "Trigonometry";`, which may or may not
be referable to using `package`, depending on the precise option used.

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

-   We are likely to want a more fine-grained, file-level approach proposed by
    [name lookup](/docs/design/name_lookup.md).
-   Allows package owners to name their packages things that they rarely type,
    but that importers end up typing frequently.
    -   The existence of a short `package` keyword shifts the balance for long
        package names by placing less burden on the package owner.
-   Reuses the `package` keyword with a significantly different meaning,
    changing from a prefix for the required declaration at the top of the file,
    to an identifier within the file.
    -   We don't need to have a special way to refer to the package to
        disambiguate duplicate names. In other words, there is likely to be
        other syntax for referring to an entity `DateTime` in the package
        `DateTime`.
    -   Renaming to a `library` keyword has been suggested to address concerns
        with `package`. Given that `library` is an argument to `package`, it
        does not significantly change the con.
-   Creates inconsistencies as compared to imports from other packages, such as
    `package Math; import Geometry;`, and imports from the current package, such
    as `package Math; import Math library "Stats";`.
    -   Option 1: Require `package` to be used to refer to all imports from
        `Math`, including the current file. This gives consistent treatment for
        the `Math` package, but not for other imports. In other words,
        developers will always write `package.Stats` from within `Math`, and
        `Math.Stats` will only be written in _other_ packages.
    -   Option 2: Require `package` be used for the current library's entities,
        but not other imports. This gives consistent treatment for imports, but
        not for the `Math` package as a whole. In other words, developers will
        only write `package.Stats` when referring to the current library,
        whether in `api` or `impl` files. `Math.Stats` will be used elsewhere,
        including from within the `Math` package.
    -   Option 3: Allow either `package` or the full package name to refer to
        the current package. This allows code to say either `package` or `Math`,
        with no enforcement for consistency. In other words, both
        `package.Stats` and `Math.Stats` are valid within the `Math` package.

Because name lookup can be expected to address the underlying issue differently,
we will not add a feature to support name lookup. We also don't want package
owners to name their packages things that even _they_ find difficult to type. As
part of pushing library authors to consider how their package will be used, we
require them to specify the package by name where desired.

#### Remove the `library` keyword from `package` and `import`

Right now, we have syntax such as:

```carbon
package Math library "Median" api;
package Math library "Median" namespace Stats api;
import Math library "Median";
```

We could remove `library`, resulting in:

```carbon
package Math.Median api;
package Math.Median namespace Math.Stats api;
import Math.Median;
```

Advantages:

-   Reduces redundant syntax in library declarations.
    -   We expect libraries to be common, so this may add up.

Disadvantages:

-   Reduces explicitness of package vs library concepts.
-   Creates redundancy of the package name in the namespace declaration.
    -   Instead of `package Math.Median namespace Math.Stats`, could instead use
        `Stats`, or `this.Stats` to elide the package name.
-   Potentially confuses the library names, such as `Math.Median`, with
    namespace names, such as `Math.Stats`.
-   Either obfuscates or makes it difficult to put multiple libraries in the
    top-level namespace.
    -   This is important because we are interested in encouraging such
        behavior.
    -   For example, if `package Math.Median api;` uses the `Math` namespace,
        the presence of `Median` with the same namespace syntax obfuscates the
        actual namespace.
    -   For example, if `package Math.Median namespace Math api` is necessary to
        use the `Math` namespace, requiring the `namespace` keyword makes it
        difficult to put multiple libraries in the top-level namespace.

As part of avoiding confusion between libraries and namespaces, we are declining
this alternative.

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
    -   [Package management systems](https://en.wikipedia.org/wiki/List_of_software_package_management_systems)
        in general.
    -   [NPM/Node.js](https://www.npmjs.com/), as a distributable unit.
    -   [Python](https://packaging.python.org/tutorials/installing-packages/),
        as a distributable unit.
    -   [Rust](https://doc.rust-lang.org/book/ch07-01-packages-and-crates.html),
        as a collection of crates.
    -   [Swift](https://developer.apple.com/documentation/swift_packages), as a
        distributable unit.

#### No association between the file system path and library/namespace

Several languages create a strict association between the method for pulling in
an API and the path to the file that provides it. For example:

-   In C++, `#include` refers to specific files without any abstraction.
    -   For example, `#include "PATH/TO/FILE.h"` means there's a file
        `PATH/TO/FILE.h`.
-   In Java, `package` and `import` both reflect file system structure.
    -   For example, `import PATH.TO.FILE;` means there's a file
        `PATH/TO/FILE.java`.
-   In Python, `import` requires matching file system structure.
    -   For example, `import PATH.TO.FILE` means there's a file
        `PATH/TO/FILE.py`.
-   In TypeScript, `import` refers to specific files.
    -   For example, `import {...} from 'PATH/TO/FILE';` means there's a file
        `PATH/TO/FILE.ts`.

For contrast:

-   In Go, `package` uses an arbitrary name.
    -   For example, `import "PATH/TO/NAME"` means there is a directory
        `PATH/TO` that contains one or more files starting with `package NAME`.

In Carbon, we are using a strict association to say that
`import PACKAGE library "PATH/TO/LIBRARY"` means there is a file
`PATH/TO/LIBRARY.carbon` under some package root.

Advantages:

-   The strict association makes it harder to move names between files without
    updating callers.
-   If there were a strict association of paths, it would also need to handle
    file system dependent casing behaviors.
    -   For example, on Windows, `project.carbon` and `Project.carbon` are
        conflicting filenames. This is exacerbated by paths, wherein a file
        `config` and a directory `Config/` would conflict, even though this
        would be a valid structure on Unix-based filesystems.

Disadvantages:

-   A strict association between file system path and import path makes it
    easier to find source files. This is used by some languages for compilation.
-   Allows getting rid of the `package` keyword by inferring related information
    from the file system path.

We are choosing to have some association between the file system path and
library for API files to make it easier to find a library's files. We are not
getting rid of the `package` keyword because we don't want to become dependent
on file system structures, particularly as it would increase the complexity of
distributed builds.

### Libraries

#### Allow exporting namespaces

We propose to not allow exporting namespaces as part of library APIs. We could
either allow or require exporting namespaces. For example:

```carbon
package Checksums;

api namespace Sha256;
```

While this approach would mainly be syntactic, a more pragmatic use of this
would be in refactoring. It implies that an aliased namespace could be marked as
an `api`. For example, the below could be used to share an import's full
contents:

```carbon
package Translator library "Interface" api;

import Translator library "Functions" as TranslatorFunctions;

api alias Functions = TranslatorFunctions;
```

Advantages:

-   Avoids any inconsistency in how entities are handled.
-   Reinforces whether a namespace may contain `api` entities.
-   Enables new kinds of refactorings.

Disadvantages:

-   Creates extra syntax for users to remember, and possibly forget, when
    declaring `api` entities.
    -   Makes it possible to have a namespace marked as `api` that doesn't
        contain any `api` entities.
-   Allowing aliasing of entire imports makes it ambiguous which entities are
    being passed on through the namespace.
    -   This may impair refactoring.
    -   This can be considered related to
        [broader imports, either all names or arbitrary code](#broader-imports-either-all-names-or-arbitrary-code).

This alternative is declined because it's not sufficiently clear it'll be
helpful, versus impairment of refactoring.

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

Examples are using `/` to separator significant terms in library names, and `//`
to separate the package name in shorthand. For example,
`package Time library "Timezones/Internal";` with shorthand
`Time//Timezones/Internal`.

Note that, because the library is an arbitrary string and shorthand is not a
language semantic, this won't affect much. However, users should be expected to
treat examples as best practice.

We could instead use `.` for library names and `/` for packages, such as
`Time/Timezones.Internal`.

Advantages:

-   Clearer distinction between the package and library, increasing readability.
-   We have chosen not to
    [enforce file system paths](#strict-association-between-the-file-system-path-and-librarynamespace)
    in order to ease refactoring, and encouraging a mental model where they may
    match could confuse users.

Disadvantages:

-   Uses multiple separators, so people need to type different characters.
-   There is a preference for thinking of libraries like file system paths, even
    if they don't actually correspond.

People like `/`, so we're going with `/`.

##### Single-word libraries

We could stick to single word libraries in examples, such as replacing
`library "Algorithms/Distance"` with `library "Distance"`.

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
        would be to expect users to always rename packages on import; some
        organizations anecdotally see equivalent happen for C++ once names get
        longer than six characters.
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

References to imports from other top-level namespaces would need to be prefixed
with a '`.`' in order to make it clear which symbols were from imports.

For example, suppose `Boost` is a large system that cannot be distributed to
users in a single package. As a result, `Random` functionality is in its own
distribution package, with multiple libraries contained. The difference between
approaches looks like:

-   `package` vs `library`:
    -   Trivial:
        -   Proposal: `package BoostRandom;`
        -   Alternative: `library "Boost/Random" namespace Boost;`
    -   Multi-layer library:
        -   Proposal: `package BoostRandom library "Uniform";`
        -   Alternative: `library "Boost/Random.Uniform" namespace Boost;`
    -   Specifying namespaces:
        -   Proposal: `package BoostRandom namespace Distributions;`
        -   Alternative:
            `library "Boost/Random.Uniform" namespace Boost.Random.Distributions;`
    -   Combined:
        -   Proposal:
            `package BoostRandom library "Uniform" namespace Distributions;`
        -   Alternative:
            `library "Boost/Random.Uniform" namespace Boost.Random.Distributions;`
-   `import` changes:
    -   Trivial:
        -   Proposal: `import BoostRandom;`
        -   Alternative: `import "Boost/Random";`
    -   Multi-layer library:
        -   Proposal: `import BoostRandom library "Uniform";`
        -   Alternative: `import "Boost/Random.Uniform";`
    -   Namespaces have no effect on `import` under both approaches.
-   Changes to use an imported entity:
    -   Proposal: `BoostRandom.UniformDistribution`
    -   Alternative:
        -   If the code is in the `Boost.Random` namespace: `Uniform`
        -   If the code is in the `Boost` package but a different namespace:
            `Random.Uniform`
        -   If the code is outside the `Boost` package: `.Boost.Random.Uniform`

We assume that the compiler will enforce that the root namespace must either
match or be a prefix of the library name, followed by a `/` separator. For
example, `Boost` in the namespace `Boost.Random.Uniform` must either match a
`library "Boost"` or prefix as `library "Boost/..."`; `library "BoostRandom"`
does not match because it's missing the `/` separator.

There are several approaches which might remove this duplication, but each has
been declined due to flaws:

-   We could have `library "Boost/Random.Uniform";` imply `namespace Boost`.
    However, we want name paths to use things listed as identifiers in files. We
    specifically do not want to use strings to generate identifiers in order to
    support understandability of code.
-   We could alternately have `namespace Boost;` syntax imply
    `library "Boost" namespace Boost;`.
    -   This approach only helps with single-library namespaces. While this
        would be common enough that a special syntax would help some developers,
        we are likely to encourage multiple libraries per namespace as part of
        best practices. We would then expect that the quantity of libraries in
        multi-library namespaces would dominate cost-benefit, leaving this to
        address only an edge-case of duplication issues.
    -   This would create an ambiguity between the file-level `namespace` and
        other `namespace` keyword use. We could then rename the `namespace`
        argument for `library` to something like `file-namespace`.
    -   It may be confusing as to what `namespace Boost.Random;` does. It may
        create `library "Boost/Random"` because `library "Boost.Random"` would
        not be legal, but the change in characters may in turn lead to developer
        confusion.
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
    -   Alignment means that if a developer sees
        `package BoostRandom library "Uniform";`, they know installing a package
        `BoostRandom` will give them the library. Declining this means that
        users seeing `library "Boost/Random.Uniform"`, they will still need to
        do research as to what package contains `Boost/Random.Uniform` to figure
        out how to install it because that package may not be named `Boost`.
    -   Package distribution is a
        [project goal](/docs/project/goals.md#language-tools-and-ecosystem), and
        cannot be avoided indefinitely.
    -   This also means multiple packages may contribute to the same top-level
        namespace, which would prevent things like tab-completion in IDEs from
        producing cache optimizations based on the knowledge that modified
        packages cannot add to a given top-level namespace. For example, the
        ability to load less may improve performance:
        -   As proposed, a package `BoostRandom` only adds to a namespace of the
            same name. If a user is editing libraries in a package
            `BoostCustom`, then `BoostRandom` may be treated as unmodifiable. An
            IDE could optimize cache invalidation of `BoostRandom` at the
            package level. As a result, if a user types `BoostRandom.` and
            requests a tab completion, the system need only ensure that
            libraries from the `BoostRandom.` package are loaded for an accurate
            result.
        -   Under this alternative, a library `Boost.Random` similarly adds to
            the namespace `Boost`. However, if a user is editing libraries, the
            IDE needs to support them adding to both `Boost` and `MyProject`
            simultaneously. As a result, if a user types `Boost.` and requests a
            tab completion, the system must have all libraries from all packages
            loaded for an accurate result.
        -   Although many features can restricted to _current_ imports, some
            features, such as
            [auto-imports](https://www.jetbrains.com/help/idea/creating-and-optimizing-imports.html),
            examine _possible_ imports. Large codebases may have a
            memory-constrained quantity of possible imports.
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
syntax on entities, such as `api fn DoSomething()`, which could create
ambiguities as `interface fn DoSomething()`. It may still confuse people to see
an `interface impl` in an `api` file. However, we're touching on related
concepts and don't see a great alternative.

#### Function-like syntax

We could consider more function-like syntax for `import`, and possibly also
`package`.

For example, instead of:

```carbon
import Math library "Stats";
import Algebra as A;
```

We could do:

```carbon
import("Math", "Stats").Math;
alias A = import("Algebra").Algebra;
```

Or some related variation.

Advantages:

-   Allows straightforward reuse of `alias` for language consistency.
-   Easier to add more optional arguments, which we expect to need for
    [interoperability](#imports-from-other-languages) and
    [URLs](#imports-from-urls).
-   Avoids defining keywords for optional fields, such as `library`.
    -   Interoperability and package management may add more fields long-term.

Disadvantages:

-   It's unusual for a function-like syntax to produce identifiers for name
    lookup.
    -   This could be addressed by _requiring_ alias, but that becomes verbose.
    -   There's a desire to explicitly note the identifier being imported some
        way, as with `.Math` and `.Algebra` above. However, this complicates the
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

### Imports

#### Block imports

Rather than requiring an `import` keyword per line, we could support block
imports, as can be found in languages like Go.

In other words, instead of:

```carbon
import Math;
import Geometry;
```

We could have:

```carbon
imports {
  Math,
  Geometry,
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

We could allow block imports of libraries from the same package. For example:

```carbon
import Containers libraries({
  "FlatHashMap",
  "FlatHashSet",
})
```

The result of this `api alias` allowing `Containers.HashSet()` to work
regardless of whether `HashSet` is in `"HashContainers"` or `"Internal"` may be
clearer if both `import Containers` statements were a combined
`import Containers libraries({"HashContainers", "Internal"});`.

The advantages/disadvantages are similar to [block imports](#block-imports).
Additional advantages/disadvantages are:

Advantages:

-   If we limit to one import per library, then any `alias` of the package
    `Containers` is easier to understand as affecting all libraries.

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
import Geometry library "Shapes" names *;

// Triangle was imported as part of "*".
fn Draw(Triangle x) { ... }
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
import Math library "Stats";
alias Median = Stats.Median;
alias Mean = Stats.Mean;
```

We could simplify this syntax by augmenting `import`:

```carbon
import Math library "Stats" name Median;
import Math library "Stats" name Mean;
```

Or more succinctly with block imports of names:

```carbon
import Math library "Stats" names {
  Median,
  Mean,
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
package Geometry library "Operations" api;

import library "Shapes";
```

Advantages:

-   Reduces typing.

Disadvantages:

-   Makes it harder to find files importing a package or library using tools
    like `grep`.
-   Creates two syntaxes for importing libraries from the current package.
    -   If we instead disallow `import Geometry library "Shapes"` from within
        `Geometry`, then we end up with a different inconsistency.

Overall, consistent with the decision to disallow
[block imports](#block-imports), we are choosing to require the package name.

### Namespaces

#### File-level namespaces

We are providing entity-level namespaces. This is likely necessary to support
migrating C++ code, at a minimum. It's been discussed whether we should _also_
support file-level namespaces.

For example, this is the current syntax for defining `Geometry.Shapes.Circle`:

```carbon
package Geometry library "Shapes" api;

namespace Shapes;
struct Shapes.Circle;
```

This is the proposed alternative syntax for defining `Geometry.Shapes.Circle`,
and would put all entities in the file under the `Shapes` namespace:

```carbon
package Geometry library "Shapes" namespace Shapes api;

struct Circle;
```

Advantages:

-   Reduces repetitive syntax in the file when every entity should be in the
    same, child namespace.
    -   Large libraries and packages are more likely to be self-referential, and
        may pay a disproportionate ergonomics tax that others wouldn't see.
    -   Although library authors could also avoid this repetitive syntax by
        omitting the namespace, that may in turn lead to more name collisions
        for large packages.
    -   Note that syntax can already be reduced with a shorter namespace alias,
        but the redundancy cannot be _eliminated_.
-   Reduces the temptation of aliasing in order to reduce verbosity, wherein
    it's generally agreed that aliasing creates inconsistent names which hinder
    readability.
    -   Users are known to alias long names, where "long" may be considered
        anything over six characters.
    -   This is a risk for any package that uses namespaces, as importers may
        also need to address it.

Disadvantages:

-   Encourages longer namespace names, as they won't need to be retyped.
-   Increases complexity of the `package` keyword.
-   Creates two ways of defining namespaces, and reuses the `namespace` keyword
    in multiple different ways.
    -   We generally prefer to provide one canonical way of doing things.
    -   Does not add functionality which cannot be achieved with entity-level
        namespaces. However, the converse is not true: entity-level control
        allows a single file to put entities into multiple namespaces.
-   Creates a divergence between code as written by the library maintainer and
    code as called.
    -   Calling code would need to specify the namespace, even if aliased to a
        shorter name. Library code gets to omit this, essentially getting a free
        alias.

We are choosing not to provide this for now because we want to provide the
minimum necessary support, and then see if it works out. It may be added later,
but it's easier to add features than to remove them.

#### Scoped namespaces

Instead of including additional namespace information per-name, we could have
scoped namespaces, similar to C++. For example:

```carbon
namespace absl {
  namespace numbers_internal {
    fn SafeStrto32Base(...) { ... }
  }

  fn SimpleAtoi(...) {
    ...
    return numbers_internal.SafeStrto32Base(...);
    ...
  }
}
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
{ namespace absl
  { namespace numbers_internal
    fn SafeStrto32Base(...) { ... }
  } namespace numbers_internal

  fn SimpleAtoi(...) {
    ...
    return numbers_internal.SafeStrto32Base(...);
    ...
  }
} namespace absl
```

While we could consider such alternative approaches, we believe the proposed
contextless namespace approach is better, as it reduces information that
developers will need to remember when reading/writing code.
