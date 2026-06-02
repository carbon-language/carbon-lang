# Clang IRGen in Carbon

<!--
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

[Pull request](https://github.com/carbon-language/carbon-lang/pull/6641)

<!-- toc -->

## Table of contents

-   [Abstract](#abstract)
-   [Problem](#problem)
-   [Background](#background)
-   [Proposal](#proposal)
-   [Rationale](#rationale)
-   [Alternatives considered](#alternatives-considered)
    -   [PR5543 More closely mimic the Clang compilation](#pr5543-more-closely-mimic-the-clang-compilation)
    -   [Status Quo with Improvements](#status-quo-with-improvements)
    -   [Upstream Clang Changes to use Phase Based Lowering](#upstream-clang-changes-to-use-phase-based-lowering)

<!-- tocstop -->

## Abstract

Document a principled and robust approach to Clang interop with respect to the
conflict between Clang's continuous lowering approach and Carbon's phase based
lowering.

## Problem

Clang performs the equivalent of Carbon's `lower` progressively, interleaved
with parsing/semantic analysis. This is in conflict with Carbon's phase-based
approach and leads to bugs in missing functionality in Clang's generated IR
during Carbon/C++ interop.

## Background

A review of different ways Carbon and other tools use Clang APIs is written up
[here](../toolchain/docs/design/clang_api.md). While Swift looks like the most
comparable, its hand-crafted reimplementation of the Clang Sema/IRGen
interaction seems like a maintenance risk.

## Proposal

Carbon should use Clang is such a way that Clang can have the
`clang::CodeGenerator` attached throughout Clang's Sema phase, to ensure parity
with existing Clang usage/behavior.

This means Clang will diverge from Carbon's strict phase-based approach (Clang
will be creating LLVM IR during `check` despite Carbon deferring all LLVM IR
lowering for Carbon itself to the `lower` phase). This divergence seems
worthwhile to keep Carbon as compatible with Clang's functionality as possible
now and in the face of possible changes to Clang in the future.

(practically speaking, this is implemented in
[PR6569](https://github.com/carbon-language/carbon-lang/pull/6569))

## Rationale

-   [Interoperability with and migration from existing C++ code](/docs/project/goals.md#interoperability-with-and-migration-from-existing-c-code)
    -   Establishing a design that leaves us as consistent with Clang's C++
        behavior as possible both now, and in the future, with as little
        maintenance needed when Clang changes are made.

## Alternatives considered

### [PR5543](https://github.com/carbon-language/carbon-lang/pull/5543) More closely mimic the Clang compilation

This looks closer/identical to Clang's API usage. But the problem is that
Clang’s `FrontendAction` API (down through… `CreateFrontendBaseAction`,
`EmitObjAction`, `ASTFrontendAction`, `ParseAST`) is a closed system (does all
the work from the start to the end) whereas Carbon wants to incrementally use
Clang while parsing more Carbon, calling back into C++, etc, before finishing
the C++ parsing. To address that atomicity, PR5543 uses a background thread
(without actual concurrency) \- doing part of the `FrontendAction` work,
pausing, doing some Carbon work, then finishing up in lowering:

-   `check` executes the `clang::FrontendAction` on a background thread
    -   This runs up until `handleTranslationUnit` and blocks
-   `check` does things to the AST, trigger template instantiation, etc
-   `lower` triggers the background thread to continue to IR generation from the
    AST

However, using a background thread to achieve this requires a great deal of
complexity. We have to both spawn and maintain the background thread, as well as
inject cross-thread synchronization to orchestrate each phase of Clang's
execution. Especially with many different C++ compilations, this complexity and
overhead would be increasingly concerning.

### Status Quo with Improvements

Keep Clang parsing without an attached `clang::CodeGenerator`, use our own
ASTConsumer to gather whatever we seem to need during parsing/sema that will be
needed during lowering. The code snippets above show some functionality we’re
missing based on the callbacks that aren’t implemented/replayed in Carbon’s
current implementation.

Risk: Missing Clang features because we didn’t save the right things for
lowering either now or with Clang changes in the future.

### Upstream Clang Changes to use Phase Based Lowering

Change Clang to no longer lower during parsing/sema, but in a single pass after
that work.

This could benefit Swift and similar API users, might simplify Clang/improve its
performance/make Clang’s IRGen more flexible (it wouldn’t struggle so much when
AST properties chang (currently that’s not allowed, but even when a definition
is found after a previous declaration - updates have to be made, etc - that
wouldn’t be a problem if all IRGen was done late)).

Risk: Upstream changes are slower, require more social engineering, buy
in/consensus building, and the benefit to Carbon (being able to do Clang
lowering at the same time as Carbon lowering) seems limited.
