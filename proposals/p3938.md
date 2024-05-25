# Exporting imported names

<!--
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

[Pull request](https://github.com/carbon-language/carbon-lang/pull/3938)

<!-- toc -->

## Table of contents

-   [Abstract](#abstract)
-   [Problem](#problem)
-   [Background](#background)
    -   [Carbon exports](#carbon-exports)
    -   [Other languages](#other-languages)
-   [Proposal](#proposal)
    -   [Source file introduction](#source-file-introduction)
-   [Future work](#future-work)
    -   [Namespaces](#namespaces)
-   [Rationale](#rationale)
-   [Alternatives considered](#alternatives-considered)
    -   [Other `export` syntax structures](#other-export-syntax-structures)
    -   [Other `export name` placements](#other-export-name-placements)
    -   [Re-exporting cross-package](#re-exporting-cross-package)

<!-- tocstop -->

## Abstract

In order to support exporting imported names, add
`export import library <library>` and `export <name reference>` syntax.

## Problem

As we develop libraries such as the prelude, we want the ability to indicate
that an imported name should be re-exported for indirect use. At present, we can
use the prototype `alias` to expose names on a case-by-case basis
(`alias Foo = Bar;`), but it doesn't work to export the _same_ name
(`alias Foo = Foo;` is a name conflict), and we want to be able to more broadly
forward a library's exported names.

For example:

```
package Foo library "internal";

// Declare C.
class C;
```

```
package Foo;

// We want the ability to expose everything imported here.
import library "internal";
```

```
import library Foo;

// Uses C by way of the default library.
var c: Foo.C;
```

## Background

Some of the syntax options were discussed
[on Discord](https://discord.com/channels/655572317891461132/1217182321933815820/1234534411350048810).

### Carbon exports

Names declared in a Carbon file are currently exported by default. A `private`
keyword may be used to prevent that export. Note C++ and TypeScript use
private-by-default behavior, so the syntax choices that make sense elsewhere may
not make as much sense for Carbon.

As described in the problem statement, `alias` offers incomplete re-export
support. However, `alias` is not fully designed; it's modeled on informal
discussions.

### Other languages

In [C++ modules](https://en.cppreference.com/w/cpp/language/modules), this is
`export import ...`.

In
[TypeScript modules](https://www.typescriptlang.org/docs/handbook/2/modules.html),
similar syntax might look like:

```typescript
import * from '<module>'
export * from '<module>'
```

In Python, names in a module are generally public, and imported names are
accessible too. For example, given `import datetime`, the module makes the name
`datetime` available to clients. There is interest in
[more explicit `export` syntax](https://discuss.python.org/t/add-the-export-keyword-to-python/28444).

In other languages, such as Java, Kotlin, or Go, direct re-exports aren't
supported. Instead, the expectation seems to be that either a copy of the entity
would be exported, or it should just be moved.

## Proposal

Support the `export` keyword as a modifier to `import library <library>`
(excluding cross-package imports). This is `export import` for short. For
example:

```carbon
export import library "lib";
```

Additionally, support the `export` keyword on individual, file-scoped or
namespace-scoped entities (excluding entities in other packages, and namespaces
themselves). This is `export name` for short. For example:

```carbon
// Export an entity:
export Foo;
// Export an entity inside a namespace:
export NS.Bar;

// Invalid: exporting namespaces is disallowed.
export NS;
```

Although exporting cross-package names is disallowed, note that `alias` can be
used to add a package-local name that originates from another package, which
then is valid for export. For example:

```carbon
import package Other;

// Invalid: cross-package exports are disallowed.
export Other.Obj;

// This introduces a package-local name. The alias is exported, and other
// libraries importing this library may export `Obj`.
alias Obj = Other.Obj;
```

The `export` keyword is only valid in files which are valid to import. It is
invalid in files which cannot be imported: implementation files and
`Main//default`.

### Source file introduction

In the
[source file introduction](/docs/design/code_and_name_organization/README.md#source-file-introduction),
`export import` directives are intermingled with other `import` directives.
`export name` directives are normal code and cannot be intermingled with any
`import` directives, including `export import` directives.

This allows:

```
import library "foo";
export import library "wiz";
import library "bar";

export FooType;

class C { ... };

export BarType;
```

This disallows:

```
import library "foo";
// Invalid: All `import` directives must come before other code, including
// `export name`.
export FooType;

import library "bar";

class C { ... };

// Invalid: `export import` must be grouped with `import` directives.
export import library "wiz";
```

## Future work

### Namespaces

Namespaces are not valid arguments to `export`; entities in namespaces must be
individually exported.

This keeps open a future design option of having `export` on a namespace export
all imported names inside the namespace, such as `export NS;`. This could also
be achieved with `*` syntax, such as `export NS.*;`. There hasn't been
discussion of this option, and this proposal takes no stance on the option.

## Rationale

-   [Software and language evolution](/docs/project/goals.md#software-and-language-evolution)
    -   `export <name>` allows moving entities between libraries without needing
        to make modifications to clients, enabling more incremental
        refactorings.
-   [Code that is easy to read, understand, and write](/docs/project/goals.md#code-that-is-easy-to-read-understand-and-write)
    -   Export logic in general is intended to support factoring large or
        complex APIs into multiple, smaller files. For example, with the
        prelude, we'll have many types, interfaces, and implementations: concise
        re-exporting logic will make it easy to provide a singular
        `prelude.carbon` that exports all related functionality.

## Alternatives considered

### Other `export` syntax structures

We discussed several different syntax choices.

A couple placement alternatives discussed were:

-   Put `export` before `library`. For example, `import export library "lib"`.
    -   An advantage of this is that if we support cross-package re-exports,
        `import Foo export library "lib"` could make it clearer the library is
        being re-exported, rather than the package.
    -   A disadvantage is that we would probably not put other keywords between
        the package and library.
-   Put `export` as a suffix. For example, `import library "lib" export`.
    -   An advantage of this is that it makes `import` statements line up better
        when some may not have the `export` modifier.

The current design uses `export` as a prefix. This is for consistency with how
we put other modifier keywords, such as `private` or `extern`, prior to the
introducer keyword.

A couple keyword alternatives discussed (alongside placement options) were:

-   `reexport`
    -   An advantage noted is that it may read more intuitively for some
        developers.
    -   This proposal suggests `export` because it mirrors `import`, and it's
        consistent with multiple other languages. It's also shorter, and Carbon
        often chooses keywords for shortness.
-   `exported` and `reexported`
    -   These didn't seem to read as clearly as `export` or `reexport`.

### Other `export name` placements

We see several options for `export name` placement. This compares them, focusing
on advantages and disadvantages for each option.

1. `export name` with `import`s

    `export name` can (only) appear in the preamble, with the imports, and
    cannot appear with the other declarations in the library. Note this option
    could either have `export name` refer to earlier `import`s (creating an
    ordering consistency issue), or expend additional effort in order to track
    whether a name was already imported at the site of the `export`.

    Advantages:

    - No need to teach developers they cannot (don't need to) `export` locally
      introduced names.

    Disadvantages:

    - Although the restricted placement might imply placement is tied to
      specific libraries, that's not the case. This could mislead developers.
        - In theory, we could enforce this, but then we could end up breaking
          code if the path a name is imported through changes.

2. `export name` with other declarations

    `export name` can only appear after imports. This means that all names valid
    for `export` will already be made available.

    Advantages:

    - `import` remains very special.
    - Makes it unambiguous that names valid for `export` are already imported.

    Disadvantages:

    - Prevents placing `export name` next to the import that is expected to add
      the name.
    - Means `export import` and `export name` will be in different sections: no
      single place to look for re-exports.

3. No ordering for `export name`

    Let developers choose what the prefer.

    Advantages:

    - Maximum flexibility, HOA rule.

    Disadvantages:

    - Most inconsistent with the desire to treat `import` as special.

We're choosing option (2). The name lookup issues avoided by requiring `export`
be below `import` directives seem worthwhile.

A possible option to (2) might be to create an additional section dedicated to
`export name` below the `import` section. This proposal suggests avoiding that
in order to avoid increasing the amount of enforced ordering in Carbon files.

### Re-exporting cross-package

As proposed, re-exporting names from other packages will not be supported. This
is done to continue maintaining package boundaries, and so that names aren't
unexpectedly introduced. For example:

```carbon
package Foo;

class C;
```

```carbon
package Bar;

export import Foo;
```

```carbon
package Wiz;

import Bar;
```

In the last package `Wiz`, it might be confusing if the name `Foo.C` should be
introduced: typically importing `Bar` would put everything under the `Bar`
namespace.

We may choose to re-examine this choice, but this proposal does not include
support.
