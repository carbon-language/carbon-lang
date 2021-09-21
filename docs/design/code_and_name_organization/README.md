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
        -   [`package` directives](#package-directives)
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
-   [Alternatives considered](#alternatives-considered)
-   [References](#references)

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

1. One `package` directive.
2. A section of zero or more `import` directives.
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

#### `package` directives

### Packages

The `package` directive's syntax may be loosely expressed as a regular
expression:

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

Because every file must have exactly one `package` directive, there are a couple
important and deliberate side-effects:

-   Every file will be in precisely one library.
    -   A library still exists even when there is no explicit library argument,
        such as `package Geometry api;`. This could be considered equivalent to
        `package Geometry library "" api;`, although we should not allow that
        specific syntax as error-prone.
-   Every entity in Carbon will be in a namespace, even if its namespace path
    consists of only the package name. There is no "global" namespace.
    -   Every entity in a file will be defined within the namespace described in
        the `package` directive.
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

-   An API file's `package` directive will have `api`. For example,
    `package Geometry library "Shapes" api;`
    -   API filenames must have the `.carbon` extension. They must not have a
        `.impl.carbon` extension.
    -   API file paths will correspond to the library name.
        -   The precise form of this correspondence is undetermined, but should
            be expected to be similar to a "Math/Algebra" library being in a
            "Math/Algebra.carbon" file path.
        -   The package will not be used when considering the file path.
-   An implementation file's `package` directive will have `impl`. For example,
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

Entities in the API file are part of the library's public API by default. They
may be marked as `private` to indicate they should only be visible to other
parts of the library.

```carbon
package Geometry library "Shapes" api;

// Circle is an API, and will be available to other libraries as
 Geometry.Circle.
struct Circle { ... }

// CircleHelper is private, and so will not be available to other libraries.
private fn CircleHelper(Circle circle) { ... }

// Only entities in namespaces should be marked as an API, not the namespace
// itself.
namespace Operations;

// Operations.GetCircumference is an API, and will be available to
// other libraries as Geometry.Operations.GetCircumference.
fn Operations.GetCircumference(Circle circle) { ... }
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

Entities in the `impl` file should never have visibility keywords. If they are
forward declared in the `api` file, they use the declaration's visibility; if
they are only present in the `impl` file, they are implicitly `private`.

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

fn Sha256.HexDigest(Bytes data) -> String { ... }
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

`import` directives supports reusing code from other files and libraries. The
`import` directive's syntax may be loosely expressed as a regular expression:

```regex
import IDENTIFIER (library NAME_PATH)?;
```

An import declares a package entity named after the imported package, and makes
API entities from the imported library available through it. The full name path
is a concatenation of the names of the package entity, any namespace entities
applied, and the final entity addressed. Child namespaces or entities may be
[aliased](/docs/design/aliases.md) if desired.

For example, given a library:

```carbon
package Math api;
namespace Trigonometry;
fn Trigonometry.Sin(...);
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

NOTE: A library must never import itself. Any `impl` files in a library
automatically import the `api`, so a self-import should never be required.

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
names. It's possible that a pre-existing entity will conflict with a new import,
and that renaming the entity is infeasible to rename due to existing callers.
Alternately, the entity may be using an idiomatic name that it would contradict
naming conventions to rename. In either case, this conflict may exist in a
single file without otherwise affecting users of the API. This will be addressed
by [name lookup](/docs/design/name_lookup.md).

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

## Alternatives considered

-   Packages
    -   [Name paths for package names](/proposals/p0107.md#name-paths-for-package-names)
    -   [Referring to the package as `package`](/proposals/p0107.md#referring-to-the-package-as-package)
    -   [Remove the `library` keyword from `package` and `import`](/proposals/p0107.md#remove-the-library-keyword-from-package-and-import)
    -   [Rename package concept](/proposals/p0107.md#rename-package-concept)
    -   [No association between the file system path and library/namespace](/proposals/p0107.md#no-association-between-the-file-system-path-and-librarynamespace)
-   Libraries
    -   [Allow exporting namespaces](/proposals/p0107.md#allow-exporting-namespaces)
    -   [Allow importing implementation files from within the same library](/proposals/p0107.md#allow-importing-implementation-files-from-within-the-same-library)
    -   [Alternative library separators and shorthand](/proposals/p0107.md#alternative-library-separators-and-shorthand)
        -   [Single-word libraries](/proposals/p0107.md#single-word-libraries)
    -   [Collapse API and implementation file concepts](/proposals/p0107.md#collapse-api-and-implementation-file-concepts)
        -   [Automatically generating the API separation](/proposals/p0107.md#automatically-generating-the-api-separation)
    -   [Collapse file and library concepts](/proposals/p0107.md#collapse-file-and-library-concepts)
    -   [Collapse the library concept into packages](/proposals/p0107.md#collapse-the-library-concept-into-packages)
    -   [Collapse the package concept into libraries](/proposals/p0107.md#collapse-the-package-concept-into-libraries)
    -   [Default `api` to private](/proposals/p0752.md#default-api-to-private)
    -   [Default `impl` to public](/proposals/p0752.md#default-impl-to-public)
    -   [Different file type labels](/proposals/p0107.md#different-file-type-labels)
    -   [Function-like syntax](/proposals/p0107.md#function-like-syntax)
    -   [Inlining from implementation files](/proposals/p0107.md#inlining-from-implementation-files)
    -   [Library-private access controls](/proposals/p0107.md#library-private-access-controls)
    -   [Make keywords either optional or required in separate definitions](/proposals/p0752.md#make-keywords-either-optional-or-required-in-separate-definitions)
    -   [Managing API versus implementation in libraries](/proposals/p0107.md#managing-api-versus-implementation-in-libraries)
    -   [Multiple API files](/proposals/p0107.md#multiple-api-files)
    -   [Name paths as library names](/proposals/p0107.md#name-paths-as-library-names)
-   Imports
    -   [Block imports](/proposals/p0107.md#block-imports)
    -   [Block imports of libraries of a single package](/proposals/p0107.md#block-imports-of-libraries-of-a-single-package)
    -   [Broader imports, either all names or arbitrary code](/proposals/p0107.md#broader-imports-either-all-names-or-arbitrary-code)
    -   [Direct name imports](/proposals/p0107.md#direct-name-imports)
    -   [Optional package names](/proposals/p0107.md#optional-package-names)
-   Namespaces
    -   [File-level namespaces](/proposals/p0107.md#file-level-namespaces)
    -   [Scoped namespaces](/proposals/p0107.md#scoped-namespaces)

## References

-   Proposal
    [#107: Code and name organization](https://github.com/carbon-language/carbon-lang/pull/107)
