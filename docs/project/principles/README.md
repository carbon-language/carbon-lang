# Principles

<!--
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

Some language [goals](../goals.md) will have widely-applicable, high-impact, and
sometimes non-obvious corollaries. We collect concrete language design
principles in this directory as a way to document and clarify these. Principles
clarify, but do not supersede, goals and priorities. Principles should be used
as a tool in making decisions, and to clarify to contributors how decisions are
expected to be made.

A key difference between a principle and the design of a language feature is
that a principle should inform multiple designs, whereas a feature's design is
typically more focused on achieving a specific goal or set of goals. The
principle can help achieve consistency across those multiple designs.

Note that these principles seek to establish both the approaches the project
wants to pursue, as well as those we want to exclude.

-   [Errors are values](error_handling.md)
-   [Information accumulation](information_accumulation.md)
-   [Low context-sensitivity](low_context_sensitivity.md)
-   [Prefer providing only one way to do a given thing](one_way.md)
-   [Safety strategy](safety_strategy.md)
-   [One static open extension mechanism](static_open_extension.md)
-   [Success criteria](success_criteria.md)
