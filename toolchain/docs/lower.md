# Lower

<!--
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

<!-- toc -->

## Table of contents

-   [Overview](#overview)

<!-- tocstop -->

## Overview

Lowering takes the SemIR and produces LLVM IR. At present, this is done in a
single pass, although it's possible we may need to do a second pass so that we
can first generate type information for function arguments.

Lowering is done per `SemIR::InstBlock`. This minimizes changes to the
`IRBuilder` insertion point, something that is both expensive and potentially
fragile.
