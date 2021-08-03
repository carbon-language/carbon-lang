# Principle: Success criteria

<!--
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

<!-- toc -->

## Table of contents

-   [Principle](#principle)
-   [Applications of these principles](#applications-of-these-principles)
    -   [Modern OS platforms, hardware architectures, and environments](#modern-os-platforms-hardware-architectures-and-environments)
        -   [OS platforms](#os-platforms)
        -   [Hardware architectures](#hardware-architectures)
        -   [Historical platforms](#historical-platforms)
    -   [Interoperability with and migration from existing C++ code](#interoperability-with-and-migration-from-existing-c-code)
        -   [Migration tooling](#migration-tooling)

<!-- tocstop -->

## Principle

Carbon's goals set a high-level path for where Carbon should head. However,
given priorities, it's not always clear how specific features or details may end
up being evaluated.

Carbon's success criteria are specific, measurable, key results that we expect
to use to see how Carbon is doing against its goals. Success criteria will be
considered as part of Carbon's [roadmap process](../roadmap_process.md), missing
them will be considered significant, and extra scrutiny will be applied on
proposals that would require diminishing them. These success criteria are not
exhaustive, but they are a bar that we aim to _exceed_.

## Applications of these principles

> TODO: Add more metrics for various goals.

### Modern OS platforms, hardware architectures, and environments

> References:
> [goal](../goals.md#modern-os-platforms-hardware-architectures-and-environments)

This should not be considered an exhaustive list of important platforms.

#### OS platforms

Our priority OS platforms are modern versions of:

-   Linux, including common distributions, Android and ChromeOS
-   FreeBSD
-   Windows
-   macOS and iOS
-   Fuchsia
-   WebAssembly
-   Bare metal

#### Hardware architectures

We expect to prioritize 64-bit little endian hardware, including:

-   x86-64
-   AArch64, also known as ARM 64-bit
-   PPC64LE, also known as Power ISA, 64-bit, Little Endian
-   RV64I, also known as RISC-V 64-bit

We believe Carbon should strive to support some GPUs, other restricted
computational hardware and environments, and embedded environments. While this
should absolutely include future and emerging hardware and platforms, those
shouldn't disproportionately shape the fundamental library and language design
while they remain relatively new and rapidly evolving.

#### Historical platforms

Example historical platforms that we will not prioritize support for are:

-   Byte sizes other than 8 bits, or non-power-of-two word sizes.
-   Source code encodings other than UTF-8.
-   Big- or mixed-endian, at least for computation; accessing encoded data
    remains useful.
-   Non-2's-complement integer formats.
-   Non-IEEE 754 binary floating point format and semantics for default single-
    and double-precision floating point types.
-   Source code in file systems that don’t support file extensions or nested
    directories.

### Interoperability with and migration from existing C++ code

> References:
> [goal](../goals.md#interoperability-with-and-migration-from-existing-c-code)

#### Migration tooling

Migrations must be mostly automatic. To that end, given an arbitrary large
codebase following best practices, we aim to have less than 2% of files require
human interaction.

This criterion includes:

-   Addressing performance bugs unique to Carbon, introduced by migration
    tooling.
-   Converting complex code which migration tooling does not handle.

This criterion does not include:

-   Cleaning up coding style to idiomatic Carbon.
    -   For example, heavy use of C++ preprocessor macros may result in expanded
        code where there is no equivalent Carbon metaprogramming construct.
