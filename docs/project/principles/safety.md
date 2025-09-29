# Safety

<!--
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

<!-- toc -->

## Distinction between Strict and Permissive Carbon

For any sufficiently expressive programming language, it is 
undecidable whether the execution of programs in that language
will have interesting properties.

Any language that aims to provides rigorous memory safety guarantees 
thus needs to navigate a tension between safety and 
expressivity.

For Carbon, we distinguish between a *strict* and a *permissive*
variant of the Carbon language, which provides rigorous memory safety guarantees,
and a *permissive* variant that provides no such guarantees.

A strict-Carbon program fragment is only accepted by the compiler
if execution is guaranteed to be free of safety-related execution errors 
and its behavior with respect to safety is predictable.

A permissive-Carbon program is accepted independent of whether it may 
or may not have safety errors. In particular, a permissive-Carbon program
can call C++ code without the programmer having to give any additional
information that may be required for checking safety.

## Partial Safety and Gradual Ramp-up

While absence of safety-related errors is a property of a program as
a whole, the design must specify the boundary and interaction between 
strict-Carbon and permissive-Carbon fragments in a way that benefits
from partial safety guarantees.

In particular, it must be possible to turn permissive-Carbon code that 
calls C++ or was migrated from C++ into safe-Carbon by gradual adoption
of safety-related annotations.

Annotations for static checking need to be [easy to read and write](../goals.md#code-that-is-easy-to-read-understand-and-write), and program safety must
be attainable without being forced to fully rewrite migrated C++ and
without performance cost.

Annotations may include programmer adding assumptions that enable
the compiler to apply the rules of safe-Carbon to obtain a proof of 
safety.

## Simplicity

The rules that determine whether a safe-Carbon program is accepted
should be easy to understand and documented.

Different build modes never change the rules that make
safe-Carbon safe (safe-Carbon is safe in any build mode), but 
they may affect behavior and performance characteristics of 
Carbon programs as a whole.

