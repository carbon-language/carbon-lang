# C++ migration tooling

<!--
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

<!-- toc -->

## Table of contents

-   [Overview](#overview)
-   [Structure](#structure)

<!-- tocstop -->

## Overview

`migrate_cpp` assists in migration of C++ code to Carbon. It's currently being
assembled; more documentation will be added later.

## Structure

The `migrate_cpp` tool uses a `clang::RecursiveASTVisitor` to traverse Clang's
AST and, to each node, associate replacements. Each node's replacement is a
sequence of text, or a reference to some other node that should be used to
replace it.
