# Generics

<!--
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

This directory contains the collection of documents describing the generics
feature of Carbon:

-   [Overview](overview.md) - A high-level description of the generics design,
    with pointers to other design documents that dive deeper into individual
    topics
-   [Goals](goals.md) - The motivation and principles guiding the design
    direction
-   [Terminology](terminology.md) - A glossary establishing common terminology
    for describing the design
-   [Detailed design](details.md) - In-depth description
    -   [Appendix: Coherence](appendix-coherence.md) - Describes the rationale
        for Carbon's choice to have coherent generics, and the alternatives.
    -   [Appendix: Rewrite constraints](appendix-rewrite-constraints.md) -
        Describes the detailed rules governing rewrite constraints, and why
        resolving them terminates.
    -   [Appendix: Witness tables](appendix-witness.md) - Describes an
        implementation strategy for checked generics, and Carbon's rationale for
        only using it for dynamic dispatch.
