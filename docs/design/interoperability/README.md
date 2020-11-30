# Bidirectional interoperability with C/C++

<!--
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

<!-- toc -->

## Table of contents

-   [Philsophy and goals](#philsophy-and-goals)
-   [Overview](#overview)

<!-- tocstop -->

## Philsophy and goals

The C++ interoperability layer of Carbon is the section wherein a specific,
restricted set of C++ APIs can be expressed in a way that's callable from
Carbon, and similar for calling Carbon from C++. This requires expressing one
language as a subset of the other. The constraint of expressivity should be
loose enough that the resulting amount of bridge code is sustainable.

The [interoperability goals](goals.md) provide more detail.

## Overview

TODO
