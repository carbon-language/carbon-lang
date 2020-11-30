# Decision for: Unicode source files

<!--
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

Proposal accepted on 2020-10-13

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

We need to specify the source file encoding in order to move on to higher-level
lexical issues, and the proposal generally gives sound rationales for the
specific choices it makes. UTF-8 and Unicode Annex 31 are non-controversial
choices for a new language in 2020. The proposal provides a good rationale for
requiring NFC.

We support treating homoglyphs and confusability as out of scope. We favor of
accepting of a wide range of scripts to start, in order to be welcoming to as
wide a community as possible, assuming that underhanded code concerns can be
adequately addressed outside of the compiler proper.
