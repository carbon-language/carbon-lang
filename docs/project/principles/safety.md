# Safety

<!--
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

<!-- toc -->

## Table of contents

-   [Static Checking](#static-checking)
-   [Strict and Permissive Carbon](#strict-and-permissive-carbon)
-   [Partial Safety and Gradual Ramp-up](#partial-safety-and-gradual-ramp-up)
-   [Simplicity and Ease of Understanding](#simplicity-and-ease-of-understanding)

<!-- tocstop -->

## Static Checking

For any sufficiently expressive programming language, it is undecidable whether
the execution of programs in that language will have interesting properties.

_Static checking_ is a compile-time method to ensure program properties through
analysis and annotations that describe intended behavior.

Carbon aims to provides rigorous memory safety guarantees through static
checking. This requires navigating a field of tension between safety and
expressivity. Carbon's statically checked safety guarantees should not come at
the price of losing possibility to cover realistic programming use cases.

Static checking cannot fully eliminate the need for dynamic techniques for
safety (for example bounds checks) but it can provide constraints on runtime
program behaviors to minimize such checks.

## Strict and Permissive Carbon

The goals of Carbon include incremental migration from C++, which does not come
with an overall statically checked safety guarantee. Further, interactions with
system components not written in Carbon place a limit on safety guarantees.

For this reason, Carbon's design distinguishes between a _strict_ and a
_permissive_ variant of the language and clarifies the interaction and boundary
between these variants.

A strict-Carbon program fragment is only accepted by the compiler if execution
is guaranteed to be entirely free of safety-related execution errors and its
behavior with respect to safety is predictable. Where strict Carbon may call
operations where the compiler cannot prove ("unsafe operations"), a marker and
an alternative safety argument must be provided in some form.

A permissive-Carbon program is accepted without an overall safety guarantee, so
its execution may lead to safety errors. Permissive Carbon should still perform
safety checks wherever possible, and signal errors. A permissive-Carbon program
fragment can call C++ code without a safety argument of any form. Where
additional information can enable safety checks, permissive Carbon can provide
ways to perform such check and thus obtain partial (weak) safety. Crucially, the
programmer is not obliged to give additional information to the compiler which
would be necessary for checking partial safety.

## Partial Safety and Gradual Ramp-up

While absence of safety-related errors is a property of a program as a whole,
the design must specify the boundary and interaction between strict-Carbon and
permissive-Carbon fragments in a way that benefits from partial safety
guarantees.

A strict-Carbon program fragment may safely interact with a permissive-Carbon
fragment if the conditions for safe execution are met.

It will be possible and necessary to _assume_ such safety conditions. In
permissive code, the responsibility for safety lies with the programmer.
Carbon's safety model may support communicating part or all of the assumed
information to the compiler which can make use of it for safety checking.

It must further be possible to migrate permissive-Carbon code into strict-Carbon
by gradual adoption of safety-related annotations, until it is fully annotated
and benefits from statically checked overall safety guarantee of strict Carbon.

## Simplicity and Ease of Understanding

Annotations for static checking need to be
[easy to read and write](../goals.md#code-that-is-easy-to-read-understand-and-write),
and program safety must be attainable without being forced to fully rewrite
migrated C++ and without performance cost.

The rules that determine whether a strict-Carbon program is accepted should be
documented and easy to understand.

Different build modes never change the rules that make safe-Carbon safe
(safe-Carbon is safe in any build mode). They may affect behavior and
performance characteristics of Carbon programs as a whole.
