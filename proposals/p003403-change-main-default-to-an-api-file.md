# Change Main//default to an api file

<!--
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

[Pull request](https://github.com/carbon-language/carbon-lang/pull/3403)

<!-- toc -->

## Table of contents

-   [Abstract](#abstract)
-   [Problem](#problem)
-   [Background](#background)
-   [Proposal](#proposal)
-   [Rationale](#rationale)
-   [Alternatives considered](#alternatives-considered)
    -   [Default to `Main//default impl` instead of `Main//default api`](#default-to-maindefault-impl-instead-of-maindefault-api)

<!-- tocstop -->

## Abstract

When there is no `package` directive, default to `Main//default api` instead of
`Main//default impl`. This has two user-visible consequences:

-   The extension will be `.carbon`, not `.impl.carbon`.
-   Only one `Main//default api` is allowed.

## Problem

[Code and name organization](/docs/design/code_and_name_organization/#libraries)
says:

-   API filenames must have the `.carbon` extension. They must not have a
    `.impl.carbon` extension.
-   Implementation filenames must have the `.impl.carbon` extension.

The `Main//default` library is currently specified as an `impl` file. This means
that it should be something like `main.impl.carbon`.

## Background

More generally in Carbon, a single-file library can be an `api` file, not an
`impl` file. This comes as a side-effect from `impl` files implicitly importing
the `api`, which would fail if there were an `impl` file without an `api`. On
the other hand, there is no requirement that an `api` file have an `impl`.

Proposal
[#2550: Simplified package declaration for the `Main` package](https://github.com/carbon-language/carbon-lang/pull/2550)
chose `impl` as the default, providing an empty `api` file that cannot otherwise
be imported. The proposal doesn't provide rationale for this choice; it likely
wasn't considered key to the proposal.

In C++, we often see `main.cpp`. This might be where the `impl` choice for
`Main//default` comes from, as it has a more equivalent feel.

## Proposal

Omitting the package directive means `Main//default api`, rather than `impl`. As
a consequence:

-   The `.carbon` extension applies instead of the `.impl.carbon` extension.
-   An `api` can only be defined once, so there is a limit of one such file per
    executable.

Mentions of `Main//default api` as being an empty file are removed.

The rules preventing use of `Main//default` in `package`, `library`, or `import`
remain. The library can only be defined by omission of `package` and `library`,
and cannot be imported.

## Rationale

-   [Code that is easy to read, understand, and write](/docs/project/goals.md#code-that-is-easy-to-read-understand-and-write)

    -   Writing `Run` logic in a `main.carbon` file is expected to be more
        intuitive than `main.impl.carbon`.

## Alternatives considered

### Default to `Main//default impl` instead of `Main//default api`

`Main//default impl`, the status quo, is now a declined alternative. Key
considerations are:

-   Using `Main//default api` is consistent with single-file libraries in other
    contexts.
-   `main.carbon` is preferred over `main.impl.carbon`, and using
    `Main//default api` allows the `main.carbon` extension without
    special-casing the file extension choices.
-   The special-case definition of `Main//default api` as an empty file is no
    longer needed.
-   Being able to have multiple `Main//default impl` files is of limited
    utility. `Run` could only be defined in one such file, and the
    `Main//default api` was defined as empty so no sharing was allowed.
