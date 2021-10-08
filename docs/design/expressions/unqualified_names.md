# Unqualified names

<!--
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

<!-- toc -->

## Table of contents

-   [Overview](#overview)
-   [Alternatives considered](#alternatives-considered)
-   [References](#references)

<!-- tocstop -->

## Overview

An _unqualified name_ is a [word](../lexical_conventions/words.md) other than a
keyword that is not preceded by a `.`. An unqualified name is looked up in the
enclosing scopes, and refers to the entity found by
[unqualified name lookup](../name_lookup.md). If the lookup finds more than one
entity in the enclosing scopes, the program is invalid due to ambiguity.

## Alternatives considered

-   [FIXME](/docs/proposals/p0845.md#FIXME)

## References

-   Proposal
    [#845: unqualified names](https://github.com/carbon-language/carbon-lang/pull/845).
