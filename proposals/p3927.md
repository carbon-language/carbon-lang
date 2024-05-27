# More consistent package syntax

<!--
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

[Pull request](https://github.com/carbon-language/carbon-lang/pull/3927)

<!-- toc -->

## Table of contents

-   [Abstract](#abstract)
-   [Problem](#problem)
-   [Background](#background)
-   [Proposal](#proposal)
-   [Details](#details)
-   [Rationale](#rationale)
-   [Alternatives considered](#alternatives-considered)
    -   [Mandatory `api` or `impl` as suffix](#mandatory-api-or-impl-as-suffix)
    -   [Put the `impl` modifier before `library`](#put-the-impl-modifier-before-library)

<!-- tocstop -->

## Abstract

Change the syntax for `package` declarations from:

```carbon
[package Foo] [library "bar"] api;
[package Foo] [library "bar"] impl;
```

to

```carbon
[package Foo] [library "bar"];
impl [package Foo] [library "bar"];
```

## Problem

The `package` declaration is currently inconsistent with other Carbon
declarations:

-   Modifier keywords for other declarations precede the introducer keyword.
    However, for package declarations, the `api` / `impl` modifier appears at
    the end of the declaration.
-   For most declarations describing the public interface of a library, we
    default to `public` because we prioritize the readability of the public
    interface over other concerns. However, the `api` tag in a `package` API
    declaration is mandatory, making the library interface more verbose than
    necessary.

## Background

[Proposal #107: Code and name organization](/proposals/p0107.md) introduced the
current syntax. It did
[consider the possibility of omitting the `api` keyword](https://github.com/carbon-language/carbon-lang/blob/trunk/proposals/p0107.md#different-file-type-labels):

> We've considered dropping `api` from naming, but that creates a definition
> from absence of a keyword. It also would be more unusual if both `impl` and
> `test` must be required, that api would be excluded. We prefer the more
> explicit name.

However, this argument did not and could not consider the inconsistencies
between the choice made for package declaration and the choices made for other
declarations, because those inconsistencies were created by later changes:

-   #107 used the `api` keyword as a marker for exporting names from an API
    file. Later, [proposal #752: api file default public](/proposals/p0752.md)
    removed this use of the `api` keyword, with the new rule being that
    declarations are in the public API by default, with an explicit keyword used
    to mark non-public declarations. This removed all uses of the `api` keyword
    other than in package declarations.
-   Rules for modifier keywords were added incrementally by various proposals,
    with the common syntactic approach that modifier keywords precede an
    introducer keyword in a declaration.

In addition, the idea of having `test` files in addition to `api` and `impl`
files has not yet been realised, and we have no current plans to add such a
feature. While that may be an interesting avenue to pursue in future, using
`test` as a modifier keyword may also be an interesting avenue to explore in
that case too -- for example, to allow functions and types within an API file to
be declared as test-only with a `test` modifier -- and so the possibility of
`test` files isn't a robust rationale for choosing the syntax for `api` and
`impl`.

## Proposal

Remove the `api` keyword from `package` and `library` declarations. Consistent
with #752, the way we define a public interface is by saying nothing at all.

Turn the `impl` keyword on such declarations into a modifier keyword, consistent
with its use in `impl fn` declarations. This reorders the `impl` keyword to the
start of the declaration.

In documentation, we refer to API files as "API files", not as "`api` files".
For consistency, we also refer to implementation files as "implementation
files", not as "`impl` files". We were previously inconsistent and used both
terms.

## Details

See design changes.

## Rationale

-   [Code that is easy to read, understand, and write](/docs/project/goals.md#code-that-is-easy-to-read-understand-and-write)
    -   Small readability and writability gain for API files.
    -   More consistency between different kinds of declaration.
    -   Minor risk that an `impl` file will be interpreted as an API file due to
        missing the `impl` modifier. However, this is likely to be caught
        quickly, whether by file naming conventions, the lack of an implicit
        import of the "real" API file, or by duplicate API file detection in the
        toolchain.
-   [Interoperability with and migration from existing C++ code](/docs/project/goals.md#interoperability-with-and-migration-from-existing-c-code)
    -   Marginally more consistent with C++ modules, which use `module Foo` vs
        `export module Foo` for the two cases -- with a leading keyword.
        However, C++ defaults to not exporting, so the case with a keyword is
        reversed, both here and in all other declarations.

## Alternatives considered

### Mandatory `api` or `impl` as suffix

The rationale for changing from the status quo ante of having a mandatory `impl`
or `api` keyword as a suffix is documented above.

This proposal also harmonizes the `impl package` syntax with the
`import package` syntax:

```carbon
impl package Foo library "bar";

import package Baz library "quux";
```

We consistently use a prefix for the package declaration for both of these
cases. We also anticipate doing so for the `import reexport package ...` or
`export import package ...` syntax that is currently under discussion.

As a trivial side benefit, the degenerate case of the package declaration for
the default library in the Main package would now be expressed as simply `;`
rather than `api;`. We retain the rule that the package declaration is omitted
entirely in this case, which is slightly easier to justify given that the
omitted declaration would comprise only a semicolon.

### Put the `impl` modifier before `library`

Because an implementation file is implementing part of a library, we can
consider placing the `impl` keyword before the `library` keyword -- as a
modifier on the `library` portion of the declaration -- instead of at the start
of the overall package declaration.

This leads to constructs such as:

```carbon
package Foo impl library "bar";
package Foo impl;
impl library "bar";
```

While there is a logical rationale and consistent explanation for this approach,
it results in the `impl` keyword's positioning appearing inconsistent: sometimes
at the start, middle, or end of the declaration. This also doesn't make the
`impl` declaration consistent with `import`, where the same argument can be
made: an `import` imports the library, not the package.

As a result, we do not pursue this direction.
