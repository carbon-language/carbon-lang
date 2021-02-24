# Decision for: Numeric literals

<!--
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

Proposal accepted on 2020-09-15

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

Placement restrictions of digit separators:

-   The core team had consensus for the proposed restricted placement rules.

Use `_` or `'` as the digit separator character:

-   The core team deferred this decision to the painter.
-   The painter selected `_`.

## Rationale

The proposal provides a syntax that is sufficiently close to that used both by
C++ and many other languages to be very familiar. However, it selects a
reasonably minimal subset of the syntaxes. This minimal approach provides
benefits directly in line with both the simplicity and readability goals of
Carbon:

-   Reduces unnecessary choices for programmers.
-   Simplifies the syntax rules of the language.
-   Improves consistency of written Carbon code.

That said, it still provides sufficient variations to address important use
cases for the goal of not leaving room for a lower level language:

-   Hexadecimal and binary integer literals.
-   Scientific notation floating point literals.
-   Hexadecimal (scientific) floating point literals.

### Painter rationale

The primary aesthetic benefit of `'` to the painter is consistency with C++.
However, its rare usage in C++ at this point reduces this advantage to a very
small one, while there is broad convergence amongst other languages around `_`.
The choice here has no risk of significant meaning or building up patterns of
reading for users that might be disrupted by the change, and so it seems
reasonable to simply converge with other languages to end up in the less
surprising and more conventional syntax space.
