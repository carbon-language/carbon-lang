# Principle: Metrics

<!--
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

## Table of contents

<!-- toc -->

- [Principle](#principle)
- [Applications of these principles](#applications-of-these-principles)
  - [Modern OS platforms, hardware architectures, and environments](#modern-os-platforms-hardware-architectures-and-environments)
    - [OS platforms](#os-platforms)
    - [Hardware architectures](#hardware-architectures)
    - [Historical platforms](#historical-platforms)

<!-- tocstop -->

## Principle

Carbon's goals set a high-level path for where Carbon should head. However,
given priorities, it's not always clear how specific features or details may end
up being prioritized.

In order to add some clarity to how we interpret Carbon's goals, we present
these metrics to define some key outcomes corresponding to the goals.

## Applications of these principles

> TODO: Add more metrics for various goals.

### Modern OS platforms, hardware architectures, and environments

#### OS platforms

Our priority OS platforms are:

- Linux
- Android
- Windows
- macOS, iOS, watchOS, tvOS
- Fuchsia
- WebAssembly
- Bare metal

#### Hardware architectures

We expect to prioritize 64-bit little endian hadware, including:

- x86-64
- AArch64, also known as ARM 64-bit
- PPC64LE, also known as ISA, 64-bit, Little Endian
- RV64I, also known as RISC-V 64-bit

We believe Carbon should strive to support some GPUs, other restricted
computational hardware and environments, and embedded environments. While this
should absolutely include future and emerging hardware and platforms, those
shouldn't disproportionately shape the fundamental library and language
design—they remain relatively new and narrow in user base at least initially.

#### Historical platforms

Example historical platforms that we will not prioritize support for are:

- Byte sizes other than 8-bits, or non-power-of-two word sizes.
- Source code encodings other than UTF-8.
- Big- or mixed-endian, at least for computation; accessing encoded data remains
  useful.
- Non-2's-complement integer formats.
- Non-IEEE 754 floating point format as default floating point types.
- Source code in file systems that don’t support file extensions or nested
  directories.
