# Metaprogramming

<!--
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

<!-- toc -->

## Table of contents

-   [TODO](#todo)
-   [Overview](#overview)

<!-- tocstop -->

## TODO

This is a skeletal design, added to support [the overview](README.md). It should
not be treated as accepted by the core team; rather, it is a placeholder until
we have more time to examine this detail. Please feel welcome to rewrite and
update as appropriate.

See [proposal PR 89](https://github.com/carbon-language/carbon-lang/pull/89) for
context -- that proposal may replace this.

## Overview

Carbon provides metaprogramming facilities that look similar to regular Carbon
code. These are structured, and do not offer inclusion or arbitrary
preprocessing of source text such as C/C++ does.
