# Decision for: C++ interoperability goals

<!--
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

Proposal accepted on 2020-11-24

Affirming:

-   [austern](https://github.com/austern)
-   [chandlerc](https://github.com/chandlerc)
-   [geoffromer](https://github.com/geoffromer)
-   [gribozavr](https://github.com/gribozavr)
-   [josh11b](https://github.com/josh11b)
-   [zygoloid](https://github.com/zygoloid)

Abstaining:

-   [noncombatant](https://github.com/noncombatant)
-   [tituswinters](https://github.com/tituswinters)

## Open questions

All open questions in the document are intended to be answered later after we
have more experience.

## Rationale

The core team agrees with the mapping of the proposed interoperability goals to
the overall Carbon goals as rationale.

Embedding of C++ bridge code inside Carbon code is inspired in part by
experience with CUDA and SYCL, where users provided feedback that these were
preferred over alternatives such as OpenCL due to the ability to embed bridge
code and device kernels in the same source file as host C++ code. It also
matches the approach of bridging between C++ and C where bridge code is often
embedded as `extern "C"` within C++ source files.

The non-goals seem appropriately motivated by the non-interoperability
priorities, without undermining the effectiveness of the interoperability as a
whole.

We agree with ensuring some level of C interoperability in order to effectively
interoperate with both some parts of the C++ ecosystem that rely on the C ABI as
well as all other languages which target C-based interoperability.
