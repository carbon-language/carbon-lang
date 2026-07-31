# Principle: File concatenation should preserve meaning

<!--
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

<!-- toc -->

## Table of contents

-   [Abstract](#abstract)
-   [Background](#background)
-   [Principle](#principle)
-   [Applications of these principles](#applications-of-these-principles)
-   [Alternatives considered](#alternatives-considered)

<!-- tocstop -->

## Abstract

Within a package, it is always possible to replace the import of a library with
the contents of that library's api file, without changing the meaning of or
diagnostics applied to the code.

## Background

In C++, it is a common practice to concatenate files together in order to
recreate and distribute a bug repro or to build a piece of C++ code in an
environment without the requirement of a build system. This allows code taken
from any build environment to be easily compiled by a developer in any other
environment, through a single compiler invocation. And it enables minimization
tools to mechanically move a single compilation unit into a single file, and
then work to minimize within that file.

The reason this works is largely historical, as `#include` performs a textual
inclusion. But it can be at least partially maintained without textual includes
as well.

## Principle

When one library imports another within a package, that import can be replaced
by the contents of the imported library's api file without changing the meaning
of or diagnostics applied to the importing library in isolation.

Then it follows that in a package with at most one impl file, all api files can
be written together into a single api file. It would be required to order the
contents of the api files so that dependencies are ordered first, in order to
maintain the information accumulation principle. Due to rejecting cyclical
imports, there is always such an ordering.

This rule applies to compiling a single library in isolation, for purposes such
as building a minimal repro, or collapsing a package into a single library. It
is not valid to copy imported definitions into multiple libraries and expect
them to work together.

## Applications of these principles

While the language may require code to appear in the same library or file, it
will never require code to be split between two libraries, and thus between two
files. Doing so would prevent the inlining of an api file into its importing
file.

Prioritization of search results for entities must be based on rules that do not
involve which file or library an entity is written in. For example, in proposal
[#5337: Interface extension and `final impl` update](/proposals/p005337-interface-extension-and-final-impl-update.md)
we considered a rule for `final impl` that would prioritize `final impl`s based
on which file they were written in. But then combining files, such as when
creating a reproduction of a bug, would require non-trivial changes to the code
in order to maintain the same behaviour.

## Alternatives considered

This rule was originally stated that any two files could be concatenated in some
order without changing the meaning of the code. However this creates (at least) two problems:

-   Under separate compilation, `impl` declarations in an impl file are not
    visible to other Carbon files. Concatenating them into another file would make
    them visible, and could change the meaning of code that can now find them.
-   Packages introduce a named scope, so the symbols within the package are
    qualified by the package name. Concatenating the contents of one package into
    another would change the name by which any moved entities would be found. This
    would necessitate changes to the code to resolve name lookups.
