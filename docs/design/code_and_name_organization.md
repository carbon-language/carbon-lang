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
    -   [Imports](#imports)
-   [Details](#details)
    -   [Name paths](#name-paths)
        -   [Disallowing name conflicts](#disallowing-name-conflicts)
    -   [Packages](#packages)
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
    -   [Moving code between between files](#moving-code-between-between-files)
    -   [Package and library name conflicts](#package-and-library-name-conflicts)
-   [Open questions](#open-questions)
    -   [Different file extensions](#different-file-extensions)
    -   [Imports from other languages](#imports-from-other-languages)
    -   [Imports from URLs](#imports-from-urls)
    -   [Test file type](#test-file-type)
-   [Alternatives](#alternatives)
    -   [Name paths](#name-paths-1)
        -   [Allow shadowing of names](#allow-shadowing-of-names)
    -   [Packages](#packages-1)
        -   [Name paths for package names](#name-paths-for-package-names)
        -   [Remove the `library` keyword from `package` and `import`](#remove-the-library-keyword-from-package-and-import)
        -   [Rename package concept](#rename-package-concept)
        -   [Strict association between the filesystem path and library/namespace](#strict-association-between-the-filesystem-path-and-librarynamespace)
    -   [Libraries](#libraries-1)
        -   [Different file type labels](#different-file-type-labels)
        -   [Allow importing implementation files from within the same library](#allow-importing-implementation-files-from-within-the-same-library)
    -   [Function-like syntax](#function-like-syntax)
        -   [Managing API versus implementation in libraries](#managing-api-versus-implementation-in-libraries)
        -   [Multiple API files](#multiple-api-files)
    -   [Namespaces](#namespaces-1)
        -   [Coarser namespace granularity](#coarser-namespace-granularity)
        -   [Scoped namespaces](#scoped-namespaces)
    -   [Imports](#imports-2)
        -   [Block imports](#block-imports)
        -   [Block imports of libraries of a single package](#block-imports-of-libraries-of-a-single-package)
        -   [Broader imports, either all names or arbitrary code](#broader-imports-either-all-names-or-arbitrary-code)
        -   [Direct name imports](#direct-name-imports)

<!-- tocstop -->

## Goals and philosophy

Important Carbon goals for code and name organization are:

-   [Language tools and ecosystem](#language-tools-and-ecosystem)

    -   Tooling support is important for Carbon, including the possibility of a
        package manager.

-   [Software and language evolution](/docs/project/goals.md#software-and-language-evolution):

    -   We should support libraries adding new structs, functions or other
        identifiers without those new identifiers being able to shadow or break
        existing users that already have identifiers with conflicting names.

    -   We should make it easy to refactor code between files.

-   [Fast and scalable development](/docs/project/goals.md#fast-and-scalable-development):

    -   It should be easy for IDEs and other tooling to parse code, without
        needing to parse imports for context.

    -   Structure should be provided for large projects to opt into features
        which will help maintain scaling of their codebase, while not adding
        burdens to small projects that don't need it.

## Overview

Carbon files have a `.carbon` extension, such as `geometry.carbon`. Carbon files
are the basic unit of compilation.

Each file begins with a declaration of which
_package_<sup><small>[[define](/docs/guides/glossary.md#package)]</small></sup>
it belongs in. The package will be a single identifier, such as `Geometry`. An
example file in the `Geometry` package would start with `package Geometry api;`.

A tiny package may consist of a single file, and not use any further features of
the `package` keyword.

However, as a package adds more files, it will probably want to separate out
into multiple
_libaries_<sup><small>[[define](/docs/guides/glossary.md#library)]</small></sup>.
A library is the basic unit of _code reuse_. Separating code into multiple
libraries can speed up the overall build while also making it clear which code
is being reused. An example library of `Shapes` in the `Geometry` package would
look like `package Geometry library Shapes api;`

It is often useful to have a physical separation the API of a library from its
implementation. This may help organize code as the library becomes larger, or to
let the build system distinguish between the dependencies of the API itself and
its underlying implementation. Implementation files allow for code to be
extracted out from the API file, while only being callable from other files
within the library, including both API and implementation files. Implementation
files are marked by both naming the file to use an extension of `.impl.carbon`
and changing the package to `package Geometry library Shapes impl`.

As code becomes more complex, and users pull in more code, it may also be
helpful to add
_namespaces_<sup><small>[[define](/docs/guides/glossary.md#namespace)]</small></sup>
to give related entities consistently structured names. A namespace affects the
_name
path_<sup><small>[[define](/docs/guides/glossary.md#name-path)]</small></sup>
used when calling code. For example, with no namespace, if a `Geometry` library
defines `Circle` then the name path will be `Geometry.Circle`. However, an
example namespace of `TwoDimensional` in the `Geometry` package would look like
`package Geometry library Shapes namespace TwoDimensional api;`, and result in a
name path of `Geometry.TwoDimensional.Circle`.

This scaling of packages into libraries and namespaces is how Carbon supports
both small and large codebases.

### Imports

The `import` keyword supports reusing code from other files and libraries.

For example, to use `Geometry.Circle` from the `Geometry.Shapes` library:

```carbon
import Geometry library Shapes;

fn Area(Geometry.Circle circle) { ... };
```

The `library` keyword is optional for `import`, and its use should parallel that
of `library` on the `package` of the code being imported.

Imports may also be
[renamed to prevent name conflicts](#name-conflicts-of-imports). For example:

```carbon
import Geometry library Shapes as Geo;

fn Area(Geo.Circle circle) { ... };
```

## Details

### Name paths

[Name paths](#name-paths) are defined above as dot-separated identifiers. This
syntax can be expressed as a rough regular expression:

```regex
IDENTIFIER(\.IDENTIFIER)*
```

#### Disallowing name conflicts

Carbon will disallow name conflicts when two identical names are declared within
the same name path. Identical names in different namespaces will be allowed,
although [name lookup](name_lookup.md) will produce an error if there is
ambiguous shadowing; that is to say, a name is used that could match two
different entities. [`as`](#import-name-conflicts) may be used to address name
conflicts caused by imports, and developers will be expected to avoid name
conflicts in other entities that they define.

For example, all cases of `Geometry` conflict because they're in the same scope:

```carbon
package Example api;

import Geometry;

namespace Geometry;
fn Geometry() { ... }
struct Geometry { ... }
```

In this below example, the declaration of `Foo.Geometry` shadows `Geometry`, but
is not inherently a conflict. A conflict arises when trying to use it from
`Foo.Geometry` because the `Geometry` lookup from within `Foo` shadows the
imported `Geometry`, and so produces a name lookup error.

```carbon
package Example api;

import Geometry;

namespace Foo;
struct Foo.Geometry { ... }
fn Foo.GetArea(var Geometry.Circle: t) { ... }
```

We expect some shadowing like this to occur, particularly during refactoring.
However, it remains important that code uniquely refer to which entity it uses
when shadowing is an issue.

### Packages

The first non-comment, non-whitespace line of a Carbon file will be the
`package` keyword. The `package` keyword's syntax may be expressed as a rough
regular expression:

```regex
package IDENTIFIER (library NAME_PATH)? (namespace NAME_PATH)? (api|impl);
```

For example:

```carbon
package Geometry library Objects.FourSides namespace TwoDimensional api;
```

Breaking this apart:

-   The identifier passed to the `package` keyword, `Geometry`, is the package
    name and will prefix both library and namespace paths.
-   When the optional `library` keyword is specified, its name path argument is
    combined with the package to generate the library path. In this example, the
    `Geometry.Objects.FourSides` library path will be used.
-   When the optional `namespace` keyword is specified, its name path argument
    is combined with the package to generate the namespace path. In this
    example, the `Geometry.TwoDimensional` namespace will be used.
-   The use of the `api` keyword indicates this is an API files as described
    under [libraries](#libraries). If it instead had `impl`, this would be an
    implementation file.

It's possible that files contributing to the `Geometry.TwoDimensional` namespace
may use different `library` arguments. Similarly, files contributing to the
`Geometry.Objects.FourSides` library may use different `namespace` arguments.
However, they will all be in the `Geometry` package.

Because the `package` keyword must be specified in all files, there are a couple
important and deliberate side-effects:

-   Every file will be in precisely one library, even if it's a library path
    that consists of only the package name.
-   Every entity in Carbon will be in a namespace, even if it's a namespace path
    that consists of only the package name. There is no "global" namespace.
    -   Entities within a file may have additional namespaces specified,
        [as detailed below](#namespaces).

### Libraries

Every Carbon library consists of one or more files. Each Carbon library has a
primary file that defines its API, and may optionally contain additional files
that are implementation.

-   An API file's `package` will have `api`. For example,
    `package Geometry library Shapes api;`
    -   API filenames must have the `.carbon` extension. They must not have a
        `.impl.carbon` extension.
-   An implementation file's `package` will have `impl`. For example,
    `package Geometry library Shapes impl;`.
    -   Implementation filenames must have the `.impl.carbon` extension.
    -   Implementation files automatically import the library's API, and may not
        import each other.

The difference between API and implementation will act as a form of access
control. API files must compile independently of implementation, only importing
from external libraries. Implementation files inside the library may consume
from either API or implementation. Files outside the library may only consume
the API.

When any file imports a library's API, it should be expected that the transitive
closure of imported files from the primary API file will be used. The size of
that transitive closure will affect compilation time, so libraries with complex
implementations should endeavor to minimize the API imports.

Libraries also serve as a critical unit of compilation. Dependencies between
libraries must be clearly marked, and the resulting dependency graph will allow
for separate compilation.

#### Exporting entities from an API file

In order to actually be part of a library's API, entities must both be in the
API file and explicitly marked as an API. This is done using the `api` keyword,
which is only allowed in the API file. For example:

```carbon
package Geometry library Shapes api;

// Circle is marked as an API, and will be available to other libraries.
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

Syntax for the `namespace` keyword may loosely be expressed as a regular
expression:

```regex
namespace NAME_PATH;
```

A namespace is used by first declaring it, then using it when declaring a name.
For example:

```carbon
namespace Foo.Bar;
struct Foo.Bar.Baz { ... }

fn Wiz(Foo.Bar.Baz x);
```

Only the first identifier in the name path becomes available for direct use;
other identifiers in the name path must be accessed through that identifier. In
other words, after declaring `namespace Foo.Bar;` in the above example, `Foo` is
available as an identifier and `Bar` must be reached through `Foo`; `Bar.Baz` is
invalid code because `Bar` would be unknown.

Namespaces declared and added to within a file must always be children of the
file-level namespace. For example, this declares
`Geometry.Shapes.ThreeSides.Triangle`:

```carbon
package Geometry namespace Shapes api;

namespace ThreeSides;

struct ThreeSides.Triangle { ... }
```

#### Re-declaring imported namespaces

Namespaces may be imported, in addition to being declared. However, the
namespace must still be declared locally in order to add symbols to it.

For example, if the `Geometry.Shapes.ThreeSides` library provides the
`Geometry.Shapes` namespace, this code is still valid:

```carbon
package Geometry library Shapes.FourSides;

import Geometry library Shapes.ThreeSides;

// This does not conflict with the existence of `Geometry.Shapes` from
// `Geometry.Shapes.ThreeSides`, even though the name path is identical.
namespace Shapes;

// This requires the above 'namespace Shapes' declaration.
struct Shapes.Square { ... };
```

#### Aliasing

Carbon's [alias keyword](aliases.md) will support aliasing namespaces. For
example, this would be valid code:

```carbon
namespace Foo.Bar;
alias FB = Foo.Bar;

struct FB.Baz { ... }
fn WizAlias(FB.Baz x);
```

### Imports

The `import` keyword supports reusing code from other files and libraries. The
`import` keyword's syntax may be expressed as a rough regular expression:

```regex
import (IDENTIFIER)? (library NAME_PATH)? (as IDENTIFIER)?;
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
package Geometry;

// This imports Math's default library, and provides a Math namespace
// identifier for use.
import Math;
// This imports Math's Trigonometry library, and resuses the Math namespace.
import Math library Trigonometry;

fn Foo() {
  ...
  // The compiler will determine which Math library Sin comes from.
  Math.Sin(...);
  ...
}
```

#### Imported name conflicts

It's possible that an imported package will have the same name as an entity
within the file doing the import. Importing them would result in
[shadowing](https://en.wikipedia.org/wiki/Variable_shadowing), which Carbon
treats as a conflict. For example, this would be a rejected name conflict
because it redefines `Geometry`:

```carbon
import Geometry;

fn Geometry(var Geometry.Circle: circle) { ... }
```

In cases such as this, `as` can be used to rename the import. For example, this
would be allowed:

```carbon
import Geometry as Geo;

fn Geometry(var Geo.Circle: circle) { ... }
```

#### Imports from the current package

The package identifier is optional when the package is intended to be the same
as the current file. However, the import will still define a namespace
identifier of the same name as the package for lookup of members, as usual. This
is so that it's unambiguous where names in the file are coming from; otherwise,
it would require compilation of imported files to determine during parsing
whether a name was undefined or imported.

For example:

```carbon
package Geometry;

// This imports Geometry's Shapes library, and provides a Geometry namespace
// identifier for use.
import library Shapes;

// Circle must be referenced using the Geometry namespace of the import.
fn GetArea(var Geometry.Circle: c) { ... }
```

Note this means the `import library Shapes` does still declare a `Geometry`
namespace that wasn't there previously.
[Imported name conflicts](#imported-name-conflicts) may still occur if
`Geometry.Geometry` is defined in the current file, and would still be fixed by
`as`.

## Caveats

### Moving code between between files

Moving code between two files in the same library is a local change that will
not affect calling code, so long as appropriate API endpoints remain.

Moving code between two libraries requires that imports be updated accordingly.
However, the namespace should remain the same, and so call sites would not need
to change.

Moving code between two namespaces always requires call sites be updated,
although it will not affect the actual import statement.

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
import Foo library Baz url("https://foo.com")
```

### Test file type

Similar to `api` and `impl`, we may eventually want a type like `test`. This
should be part of a larger testing plan.

## Alternatives

### Name paths

#### Allow shadowing of names

Disallowing shadowing of names may turn out to be costly, even for its
advantages of reducing bugs and confusion. While we could relax it later,
potentially with special syntax to handle edge cases, we don't have evidence at
present that it's necessary. As a result, we prefer the simplicity in reading of
disallowing shadowing.

### Packages

#### Name paths for package names

Right now, we only allow a single identifier for the package name. We could
allow a full name path without changing syntax.

Pros:

-   Allow greater flexibility and hierarchy for related packages, such as
    `Database.Client` and `Database.Server`.
-   Would allow using GitHub repo names as package names. For example,
    `carbon-language/carbon-toolchain` could become
    `carbon_language.carbon_toolchain`.

Cons:

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

#### Remove the `library` keyword from `package` and `import`

Right now, we have syntax like:

```carbon
package Foo library Bar;
package Foo library Bar namespace Baz;
import Foo library Bar;
```

We could remove `library`, resulting in:

```carbon
package Foo.Bar;
package Foo.Bar namespace Foo.Baz;
import Foo.Bar;
```

Pros:

-   Reduces redundant syntax in library declarations.
    -   We expect libraries to be vcommon, so this may add up.

Cons:

-   Reduces explicitness of package vs library concepts.
-   Creates redundancy of the package name in the namespace declaration.
    -   Instead of `package Foo.Bar namespace Foo.Baz`, could instead use `Baz`,
        or `this.Baz` to elide the package name.

#### Rename package concept

In other languages, a "package" is equivalent to what we call the name path
here, which includes the `namespace`. We may want to rename the `package`
keyword to avoid conflicts in meaning.

Alternative names could be 'bundle', 'universe', or something similar to Rust's
'crates'; perhaps 'compound' or 'molecule'.

Pros:

-   Avoids conflicts in meaning with other languages.
    -   [Java](https://www.oracle.com/java/technologies/glossary.html), similar
        to a namespace path.
    -   [Go](https://golang.org/doc/effective_go.html#package-names), similar to
        a namespace path.

Cons:

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

Pros:

-   A strict association between filesystem path and import path makes it easier
    to find source files. This is used by some languages for compilation.

Cons:

-   The strict association makes it harder to move names between files without
    updating callers.

We are choosing to avoid the strict association with filesystem paths in order
to ease refactoring. With this approach, more refactorings will not need changes
to API users.

### Libraries

#### Different file type labels

We're using `api` and `impl` for file types.

We've considered using `interface` instead of `api`, but that introduces a
terminology collision with interfaces in the type system.

We've considered dropping `api` from naming, but that creates a definition from
absence of a keyword. We prefer the more explicit name.

We could spell out `impl` as `implementation`, but are choosing the abbreviation
for ease of typing. We also don't think it's an unclear abbreviation.

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

Pros:

-   Allows more separation of implementation between files within a library.

Cons:

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

The problems with these approaches, and encouragement towards small libraries,
is how we reach the current approach of only importing APIs, and automatically.

### Function-like syntax

We could consider more function-like syntax for `import`, and possibly also
`package`.

For example, instead of:

```carbon
import Foo library Bar;
import Baz as B;
```

We could do:

```carbon
import("Foo", "Bar").Foo;
alias B = import("Baz").Baz;
```

Pros:

-   Allows straightforward reuse of `alias` for language consistency.
-   Easier to add more optional arguments, which we expect to need for
    [interoperability](#imports-from-other-languages) and
    [URLs](#interop-from-urls).
-   Avoids defining keywords for optional fields: `library`, `as`, and possibly
    more long-term.

Cons:

-   It's unusual for a function-like syntax to produce identifiers for name
    lookup.
    -   This could be addressed by _requiring_ alias, but that becomes verbose.
    -   There's a desire to explicitly note the identifier being imported some
        way, as with `.Foo` and `.Baz` above. However, this complicates the
        resulting syntax.

The preference is for keywords.

#### Managing API versus implementation in libraries

At present, we plan to have `api` versus `impl` as a file type, and also
`.carbon` versus `.impl.carbon` as the file extension. We chose to use both
together, rather than one or the other, because we expect some parties to
strongly want file content to be sufficient for compilation, while others will
want file extensions to be meaningful for the syntax split.

Instead of the file type split, we could drift further and instead have APIs in
any file in a library, using the same kind of
[API markup](#exporting-entities-from-an-api-file).

-   Pros:

    -   May help users who have issues with cyclical code references.
    -   Improves compiler inlining of implementations, because the compiler can
        decide how much to actually put in the generated API.

-   Cons:

    -   While allowing users to spread a library across multiple files can be
        considered an advantage, we see the single API file as a way to pressure
        users towards smaller libraries, which we prefer.
    -   May be slower to compile because each file must be parsed once to
        determine APIs.
    -   For users that want to see _only_ APIs in a file, they would need to use
        tooling to generate the API file.
        -   Auto-generated documentation may help solve this problem.

#### Multiple API files

The proposal also presently suggests a single API file. Under an explicit API
file approach, we could still allow multiple API files.

Pros:

-   More flexibility when writing APIs; could otherwise end up with one gigantic
    API file.

Cons:

-   Encourages larger libraries by making it easier to provide large APIs.
-   Removes some of the advantages of having an API file as a "single place" to
    look, suggesting more towards the markup approach.
-   Not clear if API files should be allowed to depend on each other, as they
    were intended to help resolve cyclical dependency issues.

We particularly want to discourage large libraries, and so we're likely to
retain the single API file limit.

### Namespaces

#### Coarser namespace granularity

It's been discussed whether we need to provide namespaces outside of
package/file granularity. In other words, if a file is required to only add to
one namespace, then there's no need for a `namespace` keyword or similar.

Pros:

-   Requiring files to contribute to only one namespace offers a language
    simplification.

-   Library interface vs implementation separation may be used to address some
    problems that namespaces have been used for in C++.

Cons:

-   One point made was the difficulty in C++ of doing `friend` declarations for
    template functions, making ACL controls difficult. Putting template
    functions in a namespace such as `internal` allows for an implicit warning
    about access misuse. It's preferable that both the template and the
    functions it calls be in the same file.

-   It's not clear that file-granularity namespaces would make it easy to
    address potential circular references in code.

-   `internal` namespaces and similar may also be used to hide certain calls
    from IDEs.

-   Makes it more difficult to migrate C++ code, which should be expected to
    frequently have multiple namespaces within a file.

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

Pros:

-   Allows repeated imports with less typing.

Cons:

-   Makes it harder to list imports using tools like `grep`.

One concern has been that a mix of `import` and `imports` syntax would be
confusing to users: we should only allow one.

In the end, most of the decision for `import` leans on the ease of retyping the
keyword, as well as `grep`-ability.

#### Block imports of libraries of a single package

We could allow block imports of librarys from the same package. For example:

```carbon
import Foo libraries {
  Bar,
  Baz,
}
```

This is similar to [block imports](#block-imports), and should be handled the
same. It has the additional problem that if we allow both `library` and
`libraries` syntaxes, it's a divergence that could be even more difficult to
handle.

#### Broader imports, either all names or arbitrary code

Carbon imports require specifying individual names to import. We could support
broader imports, for example by pulling in all names from a library. In C++, the
`#include` preprocessor directive even supports inclusion of arbitrary code. For
example:

```carbon
import Geometry library Shapes names *;

// Triangle was imported as part of "*".
fn Draw(var Triangle: x) { ... }
```

Pros:

-   Reduces boilerplate code specifying individual names.

Cons:

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
import Foo library Bar;
alias Baz = Foo.Baz;
alias Wiz = Foo.Wiz;
```

We could simplify this syntax by augmenting `import`:

```carbon
import Foo library Bar name Baz;
import Foo library Bar name Wiz;
```

Or more succinctly with block imports of names:

```carbon
import Foo library Bar names {
  Baz,
  Wiz,
}
```

Pros:

-   Avoids an additional `alias` step.

Cons:

-   With a single name, this isn't a significant improvement in syntax.
-   With multiple names, this runs into similar issues as
    [block imports](#block-imports).
