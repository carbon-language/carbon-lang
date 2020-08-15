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
    -   [Package keyword](#package-keyword)
    -   [Libraries](#libraries)
    -   [Namespaces](#namespaces)
        -   [Using imported namespaces](#using-imported-namespaces)
        -   [Aliasing](#aliasing)
    -   [Imports](#imports-1)
        -   [Imported name conflicts](#imported-name-conflicts)
    -   [Library name conflicts](#library-name-conflicts)
    -   [Moving APIs between between files](#moving-apis-between-between-files)
-   [Open questions to resolve](#open-questions-to-resolve)
    -   [Managing interface versus implementation in libraries](#managing-interface-versus-implementation-in-libraries)
        -   [Multiple interface files](#multiple-interface-files)
    -   [Require files in a library be imported by filename](#require-files-in-a-library-be-imported-by-filename)
    -   [Function-like `package` and `import` syntax](#function-like-package-and-import-syntax)
    -   [Quoting names in imports](#quoting-names-in-imports)
    -   [Reducing arguments for single name imports](#reducing-arguments-for-single-name-imports)
    -   [Rename package concept](#rename-package-concept)
-   [Alternatives](#alternatives)
    -   [Allow shadowing of names](#allow-shadowing-of-names)
    -   [Broader imports, either all names or arbitrary code](#broader-imports-either-all-names-or-arbitrary-code)
    -   [Coarser namespace granularity](#coarser-namespace-granularity)
    -   [Different file extensions](#different-file-extensions)
    -   [Imports from URLs](#imports-from-urls)
    -   [Namespace syntax](#namespace-syntax)
    -   [Prevent libraries from crossing package boundaries](#prevent-libraries-from-crossing-package-boundaries)
    -   [Scoped namespaces](#scoped-namespaces)
    -   [Strict association between the filesystem path and library/namespace](#strict-association-between-the-filesystem-path-and-librarynamespace)

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

When Carbon code is placed in files, we need to have some way of organizing that
code for reuse.

When creating a new file, files will choose a name with a `.carbon` extension,
such as `geometry.carbon`. Carbon files are the basic unit of compilation.

Each file begins with a declaration of which
_package_<sup><small>[[define](/docs/guides/glossary.md#package)]</small></sup>
it belongs in. The package will be a single identifier, such as `Geometry`. An
example file in the `Geometry` package would start with `package Geometry;`.

A tiny package may consist of a single file, and not use any further features of
the `package` keyword.

However, as a package adds more files, it will probably want to separate out
into multiple
_libaries_<sup><small>[[define](/docs/guides/glossary.md#library)]</small></sup>.
A library is the basic unit of *code reuse*. Separating code
into multiple libraries can speed up the overall build while also making it clear
which code is being reused. An example library of `Shapes` in the `Geometry`
package would look like `package Geometry library Shapes;`

It is often useful to have a physical separation the API of a library from its implementation.
This may help organize code as the library becomes larger, or to let the build system distinguish between the dependencies of the API itself and its underlying implementation.
Implementation files allow for code to be extracted out from the API
file, while only being callable from other files within the library, including
both API and implementation files. Implementation files are marked by both
naming the file to use an extension of `.impl.carbon` and changing the package
to `package Geometry library Shapes impl`.

As code becomes more complex, and users pull in more code, it may also be
helpful to add
_namespaces_<sup><small>[[define](/docs/guides/glossary.md#namespace)]</small></sup>
to give related entities consistently structured names. A namespace affects the _name
path_<sup><small>[[define](/docs/guides/glossary.md#name-path)]</small></sup>
used when calling code. For example, with no namespace, if a `Geometry` library
defines `Circle` then the name path will be `Geometry.Circle`. However, an
example namespace of `TwoDimensional` in the `Geometry` package would look like
`package Geometry library Shapes namespace TwoDimensional;`, and result in a
name path of `Geometry.TwoDimensional.Shapes`.

This scaling of packages into libraries and namespaces is how Carbon supports
both small and large codebases.

### Imports

The `import` keyword supports reusing code from other files and libraries.

For example, to import the `Geometry.Shapes` namespace:

```carbon
import("Geometry", "Shapes");

fn Area(Shapes.Circle circle) { ... };
```

Specific identifiers may also be imported. For example:

```carbon
import("Geometry.Shapes", "Circle");

fn Area(Circle circle) { ... };
```

Imports may also be [aliased](#name-conflicts-of-imports). For example:

```carbon
alias C = import("Geometry.Shapes", "Circle");
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
different entities. [Aliasing](aliases.md) may be used to avoid name conflicts,
[as described below](#import-name-conflicts).

For example, all cases of `Triangle` conflict because they're in the same scope:

```carbon
package Example;

import("Geometry.Shapes", "Triangle");
import("Music.Instrument", "Triangle");

namespace Triangle;
fn Triangle() { ... }
struct Triangle { ... }
```

In this below example, the declaration of `Foo.Triangle` shadows `Triangle`, but
is not inherently a conflict. A conflict arises when trying to use it from
`Foo.GetArea` because the `Triangle` lookup from within `Foo` shadows the
imported `Triangle`, and so produces a name lookup error.

```carbon
package Example;

import("Geometry.Shapes", "Triangle");

namespace Foo;
struct Foo.Triangle { ... }
fn Foo.GetArea(var Triangle: t) { ... }
```

We expect some shadowing like this to occur, particularly during refactoring.
However, it remains important that code uniquely refer to which entity it uses
when shadowing is an issue.

### Package keyword

The first non-comment, non-whitespace line of a Carbon file will be the
`package` keyword. The `package` keyword's syntax, combined with the optional
`library` keyword, may be expressed as a rough regular expression:

```regex
package NAME_PATH (library NAME_PATH)? (namespace NAME_PATH)? (impl)?;
```

For example:

```carbon
package Geometry library Objects.Flat namespace Shapes
```

Breaking this apart:

-   The first name passed to the `package` keyword, `Geometry`, is a name path
    prefix that will be used for both the file's library and namespace paths.
-   When the optional `library` keyword is specified, its name path is combined
    with the package to generate the library path. In this example, the
    `Geometry.Objects.Flat` library will be used.
-   When the optional `namespace` keyword is specified, its name path is
    combined with the package to generate the namespace path. In this example,
    the `Geometry.Shapes` namespace will be used.
-   The optional `impl` keyword, which is not present in this example, would
    make this an implementation file as described under [libraries](#libraries).

It's possible that files contributing to the `Geometry.Objects.Flat` may use
different `package` arguments. These examples vary only on the resulting
namespace, which will use only the `package` name path:

```carbon
package Geometry.Objects.Flat;
package Geometry.Objects library Flat;
package Geometry library Objects.Flat;
```

Because the `package` keyword must be specified in all files, there are a couple
important and deliberate side-effects:

-   Every file will be in precisely one library.
-   Every entity in Carbon will be in a namespace due to the `package`. There is
    no "global" namespace.
    -   Entities within a file may have additional namespaces specified,
        [as detailed below](#namespaces).

### Libraries

Every Carbon library consists of one or more files. Each Carbon library has a
primary file that defines its interface, and may optionally contain additional
files that are implementation.

-   An implementation file will have `impl` in the `package` declaration. For
    example, `package Geometry library Shapes impl;`.
-   An interface file will not have `impl`. For example,
    `package Geometry library Shapes;`

The difference between interface and implementation will act as a form of access
control. Interface files must compile independently of implementation, only
importing from external libraries. Implementation files inside the library may
consume from either interface or implementation. Files outside the library may
only consume the interface.

When importing a library's interface, it should be expected that the transitive
closure of imported files from the primary interface file will be used. The size
of that transitive closure will affect compilation time, so libraries with
complex implementations should endeavor to minimize the interface imports.

Libraries also serve as a critical unit of compilation. Dependencies between
libraries must be clearly marked, and the resulting dependency graph will allow
for separate compilation.

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
file-level namespace. For example, this declares `Geometry.Shapes.Triangle`:

```carbon
package Geometry;

namespace Shapes;

struct Shapes.Triangle { ... }
```

#### Using imported namespaces

If a namespace is imported, it may also be used to declare new items, so long as
it's a child of the file-level namespace.

For example, this code declares `Geometry.Shapes.Triangle`:

```carbon
package Geometry;

import("Geometry", "Shapes");

struct Shapes.Triangle { ... }
```

On the other hand, this is invalid code because the `Instruments` namespace is
not under `Geometry`, and so `Triangle` cannot be added to it from this file:

```carbon
package Geometry;

import("Music", "Instruments");

struct Instruments.Triangle { ... }
```

This is also invalid code because the `Geometry.Volumes` namespace is a sibling
of the `Geometry.Areas` namespace used for this file, and so `Sphere` cannot be
added to it from this file:

```carbon
package Geometry namespace Areas;

import("Geometry", "Volumes");

struct Circle { ... }

struct Volumes.Sphere { ... }
```

However, with a higher-level file namespace of `Geometry`, a single file could
still add to both `Volumes` and `Areas`:

```carbon
package Geometry;

import("Geometry", "Volumes");

namespace Areas;
struct Areas.Circle { ... }

struct Volumes.Sphere { ... }
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

The `import` keyword supports reusing code from other files and libraries. All
imports for a file must have only whitespace and comments between the `package`
declaration and them. If [metaprogramming](metaprogramming.md) code generates
`import`s, it must only generate imports and immediately follow the explicit
`import`s. No other code can be interleaved.

One or more entities may be imported with a single statement. When multiple
entities are being imported, they should be specified using a
[tuple](tuples.md). For example:

```carbon
import("Geometry.Shapes", "Circle");
import("Geometry.Shapes", ("Circle", "Triangle", "Square"));
```

When importing from another file or library that is in the current namespace
path, the namespace may be omitted from the import. For example, an import of
`Geometry.Point` may look like:

```carbon
package Geometry;

import("Point");
```

#### Imported name conflicts

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

### Library name conflicts

It's possible that two libraries will use the same namespace paths, and have
overlapping entity names. For example, someone may want to use two separate
geometry-related packages that both use the `Geometry` namespace and have
`Geometry.Point`.

To cover this case, we can allow users to specify which interface library a
given import should read from. For example:

```carbon
alias FooPoint = import("Geometry", "Point", .library_path = "path/to/foo.lib");
alias BarPoint = import("Geometry", "Point", .library_path = "path/to/bar.lib");
```

### Moving APIs between between files

When moving APIs between two files, the design of `package`, `namespace`, and
`import` is intended to minimize the amount of additional work needed. The
expected work depends on which library and namespace paths the files are in:

-   If both files are part of the same library and namespace paths:

    -   If the source was an interface, the destination file must not be an
        interface because there is only one interface file per library. Some
        declaration must remain in the interface file in order to keep the APIs
        exposed.

    -   If the source was not an interface, then no further changes are needed.

-   If the library path differs:

    -   If the destination is an interface, then no further code changes need to
        be made. However, build dependencies may need to be updated to ensure
        library dependencies are correct for callers.

    -   If the source was an interface, the destination must have an interface
        provided to support callers unless the destination library is the _only_
        caller.

    -   If the source was not an interface, and the source library still has
        calls to the API, the destination must have an interface added. If the
        source library has no remaining calls to the API, then no further
        changes are needed.

-   If the namespace path differs:

    -   The `namespace` keyword might be useable to expose the moved APIs at the
        same name path.

    -   Otherwise, callers need to be updated to the new name path.

-   If both the library and namespace paths differ, then both above sections
    apply for fixes.

## Open questions to resolve

On all of these open questions, thoughts on pros/cons would be helpful to come
to a tentative resolution. Essentially, I have thoughts on each, but I'd like
more feedback before coming to a conclusion.

> **NOTE:** Each of these open questions will be resolved before asking for a
> comment deadline; they will not be presented as "open questions" to the core
> team, or for later resolution, unless replies head in that direction. Either
> they will be adopted or not, possibly partially, and "Alternatives" will be
> updated accordingly.

### Managing interface versus implementation in libraries

The proposal currently suggests having an `impl` flag in the `package` marker to
separate implementation from interface. Is this the right way to manage the
split?

A few alternatives:

-   Add `interface` instead of, or in addition to, `impl`.

    -   This could also take the form of `package Geometry library Shapes` for
        interfaces and `package Geometry impl Shapes` for implementation. If
        there's no additional library name path, `package Geometry library`
        versus `package Geometry impl`.

    -   Pros:
        -   Increases explicitness of an interface file, which is important
            because it becomes the API.
        -   If libraries typically consist of two or more files, `impl` would
            come up more frequently, and so the less frequent syntax should be
            emphasized.
    -   Cons:
        -   This may end up being the more verbose syntax. An interface is
            _always_ required, whereas `impl` files are optional.

-   Use a different file extension. For example, `.6c` for implementation and
    `.6ch` for interface.

    -   Pros:
        -   Increases explicitness of an interface file, which is important
            because it becomes the API.
            -   Can find the interface without opening files.
    -   Cons:
        -   Adds another file extension; adds some overhead to switching which
            file in a library is the interface.

-   Instead of having special files, add a markup to names to indicate whether
    they should be treated as an interface. Have tooling determine what the
    interface is. For example, to make `Foo` an interface:

    ```carbon
    $interface
    struct Foo { ... }
    ```

    -   Pros:
        -   Avoids forcing users to separate their interface into one file.
            -   This may be considered a manual maintenance problem.
        -   May help users who have issues with cyclical code references.
        -   Improves compiler inlining of implementations, because the compiler
            can decide how much to actually put in the generated interface.
    -   Cons:
        -   May be slower to compile, as each file must be parsed once to
            determine interfaces.
        -   For users that want to see _only_ interfaces in a file, they would
            need to use tooling to generate the interface file.
            -   Auto-generated documentation may help solve this problem.

-   Use a hybrid solution with `$interface` recommended, but allow interface
    files to be specified optionally to improve compilation performance.

    -   Pros:
        -   Allows users to use `$interface` when they find it easier, without
            giving up performance options.
    -   Cons:
        -   Creates language complexity with two different approaches for
            similar issues.

#### Multiple interface files

The proposal also presently suggests a single interface file. Under an explicit
interface file approach, we could still allow multiple interface files.

Pros:

-   More flexibility when writing interfaces; could otherwise end up with one
    gigantic interface file.

Cons:

-   Removes some of the advantages of having an interface file as a "single
    place" to look, suggesting more towards the markup approach.
-   Not clear if interface files should be allowed to depend on each other, as
    they were intended to help resolve cyclical dependency issues.

### Require files in a library be imported by filename

We could add a syntax for importing symbols from other files in the same
library, such as:

```carbon
package Geometry;

import("point.6c", "Point");
```

This would be instead of other possible syntaxes:

```carbon
package Geometry;

import("Geometry", "Point");
import("Point");
```

Pros:

-   Eases enforcement of a DAG (directed acyclic graph) between files in a
    library.
    -   Do we need this to have the option of a DAG, versus using parsing to
        find which file to import from?

Cons:

-   Creates a common case where filenames _must_ be used, breaking away from
    namespace names on imports.
-   Loses some of the ease-of-use that some other languages have around imports,
    such as Go.

The choice here feels related to how we manage interface versus implementation,
although it may actually be distinct.

### Function-like `package` and `import` syntax

The proposal currently suggests function-like syntax for `import()`, and more
specialized syntax for `package`.

We could:

-   Keep this approach.
-   Make both function-like. That is, adopt `package` syntax like:
    ```carbon
    package("Foo.Bar");
    package("Foo.Bar", .library = "Bar", .namespace = "Wiz");
    ```
-   Make both specialized syntax. That is, adopt `import` syntax like:

    ```carbon
    import Geometry.Shapes names Triangle;
    import Geometry.Shapes names Triangle, Square;
    alias MusicTriangle = import MusicalInstruments names Triangle;

    // For possible interop:
    import Cpp.mynamespace names MyClass lang C++ file "myproject.h";

    // For possible URLs:
    import Foo.Bar names Baz url "https://foo.com";
    ```

### Quoting names in imports

Note the use of quotes may be optional here. Given constraints on inputs, maybe
we can already switch `import("Geometry.Shapes", "Triangle");` to
`import(Geometry.Shapes, Triangle);`. We could additionally prepend an `@` for
`@import`, assuming [metaprogramming](metaprogramming.md) keeps that syntax.

Pros:

-   Creates a consistent structure for optional arguments, aligning with
    [pattern matching](pattern_matching.md).

-   Avoids creating new keywords for new optional arguments.

Cons:

-   May require that we quote strings for consistency, adding incremental burden
    when writing imports and making them look visually inconsistent with code
    using the import.

-   Other languages, such as Java or Python, prefer the specialized syntax for
    equivalent keywords.

### Reducing arguments for single name imports

The three basic forms of `import` are:

-   `import("Foo")`: imports `Foo` from the current namespace.
-   `import("Bar", "Baz")`: imports `Bar.Baz`.
-   `import(("Foo", "Fob"))`: imports `Foo` and `Fob` from the current
    namespace.
-   `import("Bar", ("Baz", "Wiz"))`: imports `Bar.Baz` and `Bar.Wiz`.

We could allow syntax like `import("Bar.Baz")`, but that creates ambiguity with
`import("Foo")`, which could then mean either a top-level `Foo` or the current
namespace's `Foo`. That suggests at a distinctly separate syntax approach:

-   `import("this.Foo")`: imports `Foo` from the current namespace.
-   `import("Bar.Baz")`: imports `Bar.Baz`.
-   `import("this", .names=("Foo", "Fob"))`: imports `Foo` and `Fob` from the
    current namespace.
-   `import("Bar", .names=("Baz", "Wiz"))`: imports `Bar.Baz` and `Bar.Wiz`.

Pros:

-   Eliminates an argument in the common case of importing a single name.
-   Increases syntax consistency for single name imports.
-   Reduces potential confusion between importing multiple names and importing a
    single name from a different namespace.
-   Considering cross-language consistency, Java is closer to the latter syntax
    using a full name path, including resulting in being able to use only the
    last identifier for calls.

Cons:

-   Adds a `this` namespace alias, which may cause issues.
-   Increases verbosity and decreases syntax consistency for multiple name
    imports.
-   The resulting name of an import is simply the last element; the former
    syntax is clearer about this, the latter less so.

Currently, I'm wary of `this` syntax.

### Rename package concept

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

## Alternatives

### Allow shadowing of names

Disallowing shadowing of names may turn out to be costly, even for its
advantages of reducing bugs and confusion. While we could relax it later,
potentially with special syntax to handle edge cases, we don't have evidence at
present that it's necessary. As a result, we prefer the simplicity in reading of
disallowing shadowing.

### Broader imports, either all names or arbitrary code

Carbon imports require specifying individual names to import. We could support
broader imports, for example by pulling in all names from a library. In C++, the
`#include` preprocessor directive even supports inclusion of arbitrary code. For
example:

```carbon
import("Geometry.Shapes", "*");

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

### Coarser namespace granularity

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

### Different file extensions

The use of `.6c` as a short file extension or CLI has some drawbacks for
typability. There are several other possible extensions / commands:

-   `.6c`: Sounds a little like 'sexy' when read aloud.
-   `.cb` or `.cbn`: These collide with several acronyms and may not be
    especially memorable as referring to Carbon.
-   `.c6`: This seems a weird incorrect ordering of the atomic number and has a
    bad and NSFW, if _extremely_ obscure, Internet slang association.
-   `.crb`: Bad slang association.
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

### Namespace syntax

The proposed `namespace` syntax had a few alternatives considered.

-   C++-style, such as:

    ```carbon
    namespace foo {
      struct Bar { ... }
    }
    ```

    -   Pros:

        -   Supports clusters of items in the same namespace.

        -   Consistent with C++.

    -   Cons:

        -   Given a file with a long namespace, it can be difficult to identify
            which namespace names are in. This is exacerbated because the
            end-of-namespace `}` may easily be misinterpreted for the end of a
            different namespace.

            -   Some style guides recommend end-of-namespace comments.
                `} // namespace foo`. This includes
                [Google](https://google.github.io/styleguide/cppguide.html#Namespaces)
                and
                [Boost](https://github.com/boostorg/geometry/wiki/Guidelines-for-Developers).
                The use of comments indicates a syntax problem. Additionally,
                Carbon may disallow comments on the same line as code.

-   Alternative syntax with clearer start and end markers, such as:

    ```carbon
    namespace foo {
      struct Bar { ... }
    } namespace foo
    ```

    -   Pros:

        -   Supports clusters of items in the same namespace.

        -   Makes it easy to identify the start and end.

    -   Cons:

        -   Given a piece of code with a long namespace, it's still difficult to
            identify which namespace names are in.

        -   Introduces unusual markup around `}`.

Overall, we expect the proposed `namespace` syntax offers equivalent
expressiveness with improved readability and a reasonable amount of extra
typing.

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
