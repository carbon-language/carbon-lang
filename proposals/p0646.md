# Low context-sensitivity principle

<!--
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

[Pull request](https://github.com/carbon-language/carbon-lang/pull/646)

<!-- toc -->

## Table of contents

-   [Problem](#problem)
-   [Background](#background)
-   [Proposal](#proposal)
-   [Rationale based on Carbon's goals](#rationale-based-on-carbons-goals)

<!-- tocstop -->

## Problem

Carbon needs a consistent policy on how willing we are to make syntax dependent
on context.

## Background

Rust has
[a statement of their ergonomic principles](https://blog.rust-lang.org/2017/03/02/lang-ergonomics.html)
that talks about their approach to context dependence.

## Proposal

We propose adding
(`docs/project/principles/low_context_sensitivity.md`)[/docs/project/principles/low_context_sensitivity.md],
which has the details.

## Rationale based on Carbon's goals

This proposal supports the goal of making Carbon code
[easy to read, understand, and write](/docs/project/goals.md#code-that-is-easy-to-read-understand-and-write)
particularly prioritizing "read and understand" over "write".

It supports
[software evolution](/docs/project/goals.md#software-and-language-evolution) by
allowing code to be moved or refactored with fewer changes.

Further it supports developing
[performance-critical software](/docs/project/goals.md#performance-critical-software)
in Carbon by making performance predictable.
