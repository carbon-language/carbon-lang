# Pattern matching

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

-   Two-phase model (Draw mainly on Richard's doc)
    -   top-down vs bottom-up
    -   Pattern insts and procedural insts
-   Creating phase 2 insts during phase 1 (mostly no doc currently)
    -   bind_name
    -   var
    -   type expressions and speculative pushing. Note issue #5351 -- some of
        this might exist for historical reasons.
-   Discuss precise pattern block pushing? (draw mainly on my doc)
-   Function parameters
    -   Syntactic params versus `Call` params
    -   Modeling return slot as output param
