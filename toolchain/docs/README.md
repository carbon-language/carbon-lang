# Toolchain architecture

<!--
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

<!-- toc -->

## Table of contents

-   [Goals](#goals)
-   [High-level architecture](#high-level-architecture)
    -   [Design patterns](#design-patterns)
-   [Adding features](#adding-features)

<!-- tocstop -->

## Goals

The toolchain represents the production portion of Carbon. At a high level, the
toolchain's top priorities are:

-   Correctness.
-   Quality of generated code, including performance.
-   Compilation performance.
-   Quality of diagnostics for incorrect or questionable code.

TODO: Add an expanded document that details the goals and priorities and link to
it here.

## High-level architecture

The main components are:

-   [Driver](driver.md): Provides commands and ties together compilation flow.
-   [Diagnostics](diagnostics.md): Produces diagnostic output.
-   Compilation flow:

    1. Source: Load the file into a
       [SourceBuffer](/toolchain/source/source_buffer.h).
    2. [Lex](lex.md): Transform a SourceBuffer into a
       [Lex::TokenizedBuffer](/toolchain/lex/tokenized_buffer.h).
    3. [Parse](parse.md): Transform a TokenizedBuffer into a
       [Parse::Tree](/toolchain/parse/tree.h).
    4. [Check](check): Transform a Tree to produce
       [SemIR::File](/toolchain/sem_ir/file.h).
    5. [Lower](lower.md): Transform the SemIR to an
       [LLVM Module](https://llvm.org/doxygen/classllvm_1_1Module.html).
    6. CodeGen: Transform the LLVM Module into an Object File.

### Design patterns

A few common design patterns are:

-   Distinct steps: Each step of processing produces an output structure,
    avoiding callbacks passing data between structures.

    -   For example, the parser takes a `Lex::TokenizedBuffer` as input and
        produces a `Parse::Tree` as output.

    -   Performance: It should yield better locality versus a callback approach.

    -   Understandability: Each step has a clear input and output, versus
        callbacks which obscure the flow of data.

-   Vectorized storage: Data is stored in vectors and flyweights are passed
    around, avoiding more typical heap allocation with pointers.

    -   For example, the parse tree is stored as a
        `llvm::SmallVector<Parse::Tree::NodeImpl>` indexed by `Parse::Node`
        which wraps an `int32_t`.

    -   Performance: Vectorization both minimizes memory allocation overhead and
        enables better read caching because adjacent entries will be cached
        together.

-   Iterative processing: We rely on state stacks and iterative loops for
    parsing, avoiding recursive function calls.

    -   For example, the parser has a `Parse::State` enum tracked in
        `state_stack_`, and loops in `Parse::Tree::Parse`.

    -   Scalability: Complex code must not cause recursion issues. We have
        experience in Clang seeing stack frame recursion limits being hit in
        unexpected ways, and non-recursive approaches largely avoid that risk.

See also [Idioms](idioms.md) for abbreviations and more implementation
techniques.

## Adding features

We have a [walkthrough for adding features](adding_features.md).
