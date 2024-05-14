# Naming conventions

<!--
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

<!-- toc -->

## Table of contents

-   [Overview](#overview)
-   [Details](#details)
    -   [Constants](#constants)
    -   [Carbon-provided item naming](#carbon-provided-item-naming)
-   [Alternatives considered](#alternatives-considered)
-   [References](#references)

<!-- tocstop -->

## Overview

Our naming conventions are:

-   For idiomatic Carbon code:
    -   `UpperCamelCase` will be used when the named entity cannot have a
        dynamically varying value. For example, functions, namespaces, or
        compile-time constant values.
    -   `lower_snake_case` will be used when the named entity's value won't be
        known until runtime, such as for variables.
-   For Carbon-provided features:
    -   Keywords and type literals will use `lower_snake_case`.
    -   Other code will use the guidelines for idiomatic Carbon code.

In other words:

| Item                      | Convention         | Explanation                                                                                |
| ------------------------- | ------------------ | ------------------------------------------------------------------------------------------ |
| Packages                  | `UpperCamelCase`   | Used for compile-time lookup.                                                              |
| Types                     | `UpperCamelCase`   | Resolved at compile-time.                                                                  |
| Functions                 | `UpperCamelCase`   | Resolved at compile-time.                                                                  |
| Methods                   | `UpperCamelCase`   | Methods, including virtual methods, are equivalent to functions.                           |
| Compile-time parameters   | `UpperCamelCase`   | May vary based on inputs, but are ultimately resolved at compile-time.                     |
| Compile-time constants    | `UpperCamelCase`   | Resolved at compile-time. See [constants](#constants) for more remarks.                    |
| Variables                 | `lower_snake_case` | May be reassigned and thus require runtime information.                                    |
| Member variables          | `lower_snake_case` | Behave like variables.                                                                     |
| Keywords                  | `lower_snake_case` | Special, and developers can be expected to be comfortable with this casing cross-language. |
| Type literals             | `lower_snake_case` | Equivalent to keywords.                                                                    |
| Boolean type and literals | `lower_snake_case` | Equivalent to keywords.                                                                    |
| Other Carbon types        | `UpperCamelCase`   | Behave like normal types.                                                                  |
| `Self` and `Base`         | `UpperCamelCase`   | These are similar to type members on a class.                                              |

We only use `UpperCamelCase` and `lower_snake_case` in naming conventions in
order to minimize the variation in rules.

## Details

### Constants

Consider the following code:

```carbon
package Example;

let CompileTimeConstant: i32 = 7;

fn RuntimeFunction(runtime_constant: i32);
```

In this example, `CompileTimeConstant` has a singular value (`7`) which is known
at compile-time. As such, it uses `UpperCamelCase`.

On the other hand, `runtime_constant` may be constant within the function body,
but it is assigned at runtime when `RuntimeFunction` is called. Its value is
only known in a given runtime invocation of `RuntimeFunction`. As such, it uses
`lower_snake_case`.

### Carbon-provided item naming

Carbon-provided items are split into a few categories:

-   Keywords; for example, `for`, `fn`, and `var`.
-   Type literals; for example, `i<digits>`, `u<digits>`, and `f<digits>`.
-   Boolean type and literals; for example, `bool`, `true`, and `false`.
    -   The separate categorization of booleans should not be taken as a rule
        that only booleans would use lowercase; it's just the only example right
        now.
-   `Self` and `Base`.
-   Other Carbon types; for example, `Int`, `UInt`, and `String`.

Note that while other Carbon types currently use `UpperCamelCase`, that should
not be inferred to mean that future Carbon types will do the same. The leads
will make decisions on future naming.

## Alternatives considered

-   [Other naming conventions](/proposals/p0861.md#other-naming-conventions)
-   [Other conventions for naming Carbon types](/proposals/p0861.md#other-conventions-for-naming-carbon-types)

## References

-   Proposal
    [#861: Naming conventions](https://github.com/carbon-language/carbon-lang/pull/861)
