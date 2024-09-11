# Lower

<!--
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

<!-- toc -->

## Table of contents

-   [Overview](#overview)
-   [Mangling](#mangling)

<!-- tocstop -->

## Overview

Lowering takes the SemIR and produces LLVM IR. At present, this is done in a
single pass, although it's possible we may need to do a second pass so that we
can first generate type information for function arguments.

Lowering is done per `SemIR::InstBlock`. This minimizes changes to the
`IRBuilder` insertion point, something that is both expensive and potentially
fragile.

## Mangling

Part of lowering is choosing deterministically unique identifiers for each
lowered entity to use in platform object files. Any feature of an entity (eg:
the namespace it appears in, parameters for overloaded functions, etc) that
would create a distinct entity must be included in some way in the generated
identifier.

The current rudimentary name mangling scheme is as follows:

-   Start with `_C`.
-   Then the unqualified function name (function name mangling is the only thing
    implemented at the moment).
-   Then `.` separated scopes (namespaces/classes), most nested first, outermost
    last.
-   Or, if the function is in an `impl`:
    -   the implementing type, per the scope mangling above.
    -   the interface type, per the scope mangling above.
