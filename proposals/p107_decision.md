# Decision for: Code and name organization

<!--
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

Proposal accepted on 2020-10-06

Affirming:

-   [austern](https://github.com/austern)
-   [chandlerc](https://github.com/chandlerc)
-   [josh11b](https://github.com/josh11b)
-   [tituswinters](https://github.com/tituswinters)
-   [zygoloid](https://github.com/zygoloid)

Abstaining:

-   [geoffromer](https://github.com/geoffromer)
-   [gribozavr](https://github.com/gribozavr)
-   [noncombatant](https://github.com/noncombatant)

## Open questions

### TODO question?

Should we switch to a library-oriented structure that's package-agnostic?

-   **Decision:** No.
-   **Rationale:** While this would simplify the overall set of constructs
    needed, removing the concept of a global namespace remained desirable and
    would require re-introducing much of the complexity around top-level
    namespaces. Overall, the simplification trade-off didn't seem significantly
    better.

Should there be a tight association between file paths and packages/libraries?

-   **Decision:** Yes, for the API files in libraries. Specifically, the library
    name should still be written in the source, but it should be checked to
    match--after some platform-specific translation--against the path.
-   **Note:** Sufficient restrictions to result in a portable and simple
    translation on different filesystems should be imposed, but the Core team
    was happy for these restrictions to be developed as part of implementation
    work.
-   **Rationale:** This will improve usability and readability for users by
    making it obvious how to find the files that are being imported. Similarly,
    this will improve tooling by increasing the ease with which tools can find
    imported APIs.

## Rationale

This proposal provides an organizational structure that seems both workable and
aligns well with Carbon's goals:

-   Distinct and required top-level namespace -- "package"s from the proposal --
    both matches software best practices for long-term evolution, and avoids
    complex and user-confusing corner cases.
-   Providing a fine-grained import structure as provided by the "library"
    concept supports scalable build system implementations while ensuring
    explicit dependencies.
-   The structured namespace facilities provide a clear mechanism to migrate
    existing hierarchical naming structures in C++ code.
