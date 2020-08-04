# Code and name organization

<!--
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

## Table of contents

<!-- toc -->

-   [Overview](#overview)
-   [Files](#files)
-   [Libraries](#libraries)
-   [Namespaces](#namespaces)
-   [Packages](#packages)
-   [Imports](#imports)
-   [Shadowing of names](#shadowing-of-names)
-   [Standard library names](#standard-library-names)
-   [Alternatives](#alternatives)
    -   [File extensions](#file-extensions)

<!-- tocstop -->

## Overview

Carbon code is organized into files, libraries, and packages in increasing
scope. Names within Carbon code are organized into a hierarchy of named scopes,
the outermost of which are namespaces.

-   **File:** This is the basic unit of Carbon compilation. Each file can be
    compiled separately (but with access to imports), and parsed (without
    semantic analysis) in isolation. These correspond to translation units in
    (modular) C++. For example, the
    `[flat_hash_map.h](https://github.com/abseil/abseil-cpp/blob/master/absl/container/flat_hash_map.h)`
    header would be a single module interface unit with C++ modules, and it
    would be a single file in Carbon.

-   **Library:** This is the basic unit of Carbon interfaces that can be
    imported. In other words, they form the link-time abstraction boundary. A
    library consists of one or more Carbon files. These correspond exactly to
    C++ modules or Bazel `cc_library` targets. For example, if all the code in
    the Abseil `[container](https://abseil.io/docs/cpp/guides/container)`
    library were in a single `cc_library`, that library would map to a Carbon
    library (and C++ module).

-   **Package:** This is a collection of one or more Carbon libraries that share
    a top-level namespace. They don't introduce new constructs, but rather are a
    restricted and special kind of namespace. All Carbon code is in a named
    package and cannot be in a "global" namespace. Neither libraries nor
    (nested) namespaces can span package boundaries. Typically, all the
    libraries within a package are developed and distributed together. These are
    expected to correspond to a repository on GitHub or a top level namespace in
    an organization's codebase. For example, Abseil would likely map to a Carbon
    package much like it resides in the top level C++ namespace `::absl`.

-   **Namespace:** This is the basic unit organizing Carbon names and represents
    the _name_ abstraction boundary. They introduce (potentially nested) named
    scopes. They can be navigated using dot-syntax (`Foo.Bar.Baz`). Libraries,
    notably, are not namespaces and are orthogonal to them.

This organization is specifically designed to scale up to relatively complex
structures with multiple files in a library and multiple libraries in a package.
However, it also keeps simple cases simple. For example, when a package consists
of a single library, or a library consists of a single file. The goal is to
avoid unnecessary syntax and ceremony in these cases while having consistent and
graceful scaling up of syntax to support the more complex cases.

## Files

All Carbon files are part of exactly one library (and all libraries are part of
exactly one package). There are also two kinds of files: ones that contribute to
the interface of a library, and ones that only contain implementation details of
that library. The latter cannot export any interfaces from the library, and that
kind is designed to facilitate more efficient parallel and distributed
compilation strategies.

The first (non-comment, non-whitespace) line of a Carbon file declares what
library and package the file is a part of, as well as whether the file is
implementation-only:

```
package Abseil library Container;
```

```
package Abseil library Container impl;
```

Here, `package`, `library`, and `impl` are all keywords (at least in this
context). The package name is `Abseil`, and the library name is `Container`.

There are some more details of these file declarations that will be explained as
we get into the details of Carbon libraries.

## Libraries

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

## Namespaces

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

## Packages

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

## Imports

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

## Shadowing of names

Carbon completely forbids shadowing of names. Because there is no global
namespace and no using directive, unqualified names are entirely within local
control and so should not require shadowing for maintenance or development.
Allowing shadowing has a long history of bugs and confusion. While it is used in
some clear and readable places in C++, those situations are likely better
addressed with language facilities than cleverly shadowed names.

    **Open question:** This is an extremely restrictive stance. It has many advantages, but may end up being too costly. We can easily relax it if needed, potentially with special syntax to handle specific edge cases or as an interim measure to aid migration. Or if data emerges, we can simply revisit it entirely.

## Standard library names

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

The use of `6c` as a short file extension or top-level CLI (with subcommands
below it similar to `git` or `go`) has some drawbacks. There are several other
possible extensions / commands:

-   `cb`: This collides with several acronyms and may not be especially
    memorable as referring to Carbon.
-   `c6`: This seems a weird incorrect ordering of the atomic number and has a
    bad (if _extremely_ obscure) Internet slang association (NSFW, use caution
    if searching, as with too much Internet slang).
-   `carbon`: This is an obvious and unsurprising choice, but also quite long.

This seems fairly easy for us to change as we go along, but we should at some
point do a formal proposal to gather other options and let the core team try to
find the set that they feel is close enough to be a bikeshed.
