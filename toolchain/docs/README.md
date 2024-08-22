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
-   [Main components](#main-components)
    -   [Driver](#driver)
    -   [Diagnostics](#diagnostics)
    -   [Lex](#lex)
        -   [Bracket matching](#bracket-matching)
    -   [Parse](#parse)
    -   [Check](#check)
-   [Adding features](#adding-features)
-   [Alternatives considered](#alternatives-considered)
    -   [Bracket matching in parser](#bracket-matching-in-parser)
    -   [Using a traditional AST representation](#using-a-traditional-ast-representation)

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

The default compilation flow is:

1. Load the file into a [SourceBuffer](/toolchain/source/source_buffer.h).
2. Lex a SourceBuffer into a
   [Lex::TokenizedBuffer](/toolchain/lex/tokenized_buffer.h).
3. Parse a TokenizedBuffer into a [Parse::Tree](/toolchain/parse/tree.h).
4. Check a Tree to produce [SemIR::File](/toolchain/sem_ir/file.h).
5. Lower the SemIR to an
   [LLVM Module](https://llvm.org/doxygen/classllvm_1_1Module.html).
6. CodeGen turns the LLVM Module into an Object File.

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

## Main components

### Driver

The driver provides commands and ties together the toolchain's flow. Running a
command such as `carbon compile --phase=lower <file>` will run through the flow
and print output. Several dump flags, such as `--dump-parse-tree`, print output
in YAML format for easier parsing.

### Diagnostics

The diagnostic code is used by the toolchain to produce output.

See [Diagnostics](diagnostics.md) for details.

### Lex

Lexing converts input source code into tokenized output. Literals, such as
string literals, have their value parsed and form a single token at this stage.

#### Bracket matching

The lexer handles matching for `()`, `[]`, and `{}`. When a bracket lacks a
match, it will insert a "recovery" token to produce a match. As a consequence,
the lexer's output should always have matched brackets, even with invalid code.

While bracket matching could use hints such as contextual clues from
indentation, that is not yet implemented.

### Parse

Parsing uses tokens to produce a parse tree that faithfully represents the tree
structure of the source program, interpreted according to the Carbon grammar. No
semantics are associated with the tree structure at this level, and no name
lookup is performed.

See [Parse](parse.md) for details.

### Check

Check takes the parse tree and generates a semantic intermediate representation,
or SemIR. This will look closer to a series of instructions, in preparation for
transformation to LLVM IR. Semantic analysis and type checking occurs during the
production of SemIR. It also does any validation that requires context.

See [Check](check.md) for details.

## Adding features

We have a [walkthrough for adding features](adding_features.md).

## Alternatives considered

### Bracket matching in parser

Bracket matching could have also been implemented in the parser, with some
awareness of parse state. However, that would shift some of the complexity of
recovery in other error situations, such as where the parser searches for the
next comma in a list. That needs to skip over bracketed ranges. We don't think
the trade-offs would yield a net benefit, so any change in this direction would
need to show concrete improvement, for example better diagnostics for common
issues.

### Using a traditional AST representation

Clang creates an AST as part of compilation. In Carbon, it's something we could
do as a step between parsing and checking, possibly replacing the SemIR. It's
likely that doing so would be simpler, amongst other possible trade-offs.
However, we think the SemIR approach is going to yield higher performance,
enough so that it's the chosen approach.
