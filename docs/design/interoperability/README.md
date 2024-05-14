# Bidirectional interoperability with C and C++

<!--
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

<!-- toc -->

## Table of contents

-   [Philosophy and goals](#philosophy-and-goals)
-   [Overview](#overview)

<!-- tocstop -->

## Philosophy and goals

The C++ interoperability layer of Carbon allows a subset of C++ APIs to be
accessed from Carbon code, and similarly a subset of Carbon APIs to be accessed
from C++ code. This requires expressing one language as a subset of the other.
Bridge code may be needed to map some APIs into the relevant subset, but the
constraints on expressivity should be loose enough to keep the amount of such
bridge code sustainable.

The [interoperability philosophy and goals](philosophy_and_goals.md) provide
more detail.

## Overview

TODO
