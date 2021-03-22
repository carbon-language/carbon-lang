# Decision for: if/else

<!--
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

Proposal accepted on 2021-03-16

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

## Rationale

This proposal focuses on an uncontroversial piece that we are going to carry
from C++, as a baseline for future Carbon evolution. It serves our migration
goals (especially "Familiarity for experienced C++ developers with a gentle
learning curve") by avoiding unnecessary deviation from C++, and instead
focusing on subsetting the C++ feature. While we expect this feature to evolve
somewhat, the changes we're likely to want can easily be applied incrementally,
and this is a fine starting point that anchors us on favoring syntax familiar to
C++ developers.
