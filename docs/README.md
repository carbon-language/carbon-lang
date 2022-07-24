# Docs

<!--
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

This directory contains current, accepted documentation underpinning Carbon.
These documents cover all aspects of Carbon ranging from the project down to
detailed designs for specific language features.

If you're trying to learn more about Carbon, we recommend starting at
[`/README.md`](/README.md).

## Design

Carbon language's design and rationale are documented in the
[`design/` directory](design/README.md). This documentation is intended to
support the following audiences:

-   People who wish to determine whether Carbon would be the right choice for a
    project compared to other existing languages.
-   People working on the evolution of the Carbon language who wish to
    understand the rationale and motivation for existing design decisions.
-   People working on a specification or implementation of the Carbon language
    who need a detailed understanding of the intended design.
-   People writing Carbon code who wish to understand why the language rules are
    the way they are.

This is in contrast to [proposals](/proposals/README.md), which document the
individual decisions that led to this design (along with other changes to the
Carbon project), including the rationale and alternatives considered.

## Project

The [`project/` directory](project/README.md) contains project-related
documentation for Carbon, including:

-   [goals](project/goals.md), and the
    [principles](project/principles/README.md) and [roadmap](project/roadmap.md)
    derived from those goals,
-   how the project works, and
-   how to contribute.

## Guides

The [`guides/` directory](guides/README.md) contains **to-be-written** end-user
documentation for developers writing programs in Carbon.

## Spec

The [`spec/` directory](spec/) contains the **to-be-written** formal
specification of the Carbon language. This is for implementers of compilers or
other tooling. This is intended to complement the interactive
[language explorer tool](/explorer/).
