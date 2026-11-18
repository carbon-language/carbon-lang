# Move toolchain alternatives to proposals

<!--
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

[Pull request](https://github.com/carbon-language/carbon-lang/pull/6716)

<!-- toc -->

## Table of contents

-   [Abstract](#abstract)
-   [Problem](#problem)
-   [Proposal](#proposal)
-   [Rationale](#rationale)
-   [Alternatives considered](#alternatives-considered)
    -   [Lex](#lex)
        -   [Bracket matching in parser](#bracket-matching-in-parser)
    -   [Parse](#parse)
        -   [Restrictive parsing](#restrictive-parsing)
    -   [Check](#check)
        -   [Using a traditional AST representation](#using-a-traditional-ast-representation)
    -   [Coalescing generic functions emitted when lowering to LLVM IR](#coalescing-generic-functions-emitted-when-lowering-to-llvm-ir)
        -   [Coalescing in the front-end vs back-end?](#coalescing-in-the-front-end-vs-back-end)
        -   [When to do coalescing in the front-end?](#when-to-do-coalescing-in-the-front-end)
        -   [Compile-time trade-offs](#compile-time-trade-offs)
        -   [Coalescing duplicate non-specific functions](#coalescing-duplicate-non-specific-functions)

<!-- tocstop -->

## Abstract

As part of using the evolution process with the toolchain, alternatives should
be in proposals. This proposal migrates existing alternatives here.

## Problem

Leads want to use the evolution process with more of the toolchain changes.
Historically, alternatives were documented as part of the toolchain design, not
going through evolution. Switching leaves those older alternatives in the
toolchain design, while putting newer alternatives in the proposals directory.

## Proposal

Move existing alternatives to this proposal. Future proposals will more
naturally have alternatives in the proposal document itself.

## Rationale

This is in support of the
[evolution process](/docs/project/goals.md#software-and-language-evolution),
aligning the toolchain documentation with design documentation.

## Alternatives considered

### Lex

#### Bracket matching in parser

Bracket matching could have also been implemented in the parser, with some
awareness of parse state. However, that would shift some of the complexity of
recovery in other error situations, such as where the parser searches for the
next comma in a list. That needs to skip over bracketed ranges. We don't think
the trade-offs would yield a net benefit, so any change in this direction would
need to show concrete improvement, for example better diagnostics for common
issues.

### Parse

#### Restrictive parsing

The toolchain will often parse code that could theoretically be rejected,
instead allowing the check phase to reject incorrect structures.

For example, consider the code `abstract var x: i32 = 0;`. When parsing the
`abstract` modifier, parse could do single-token lookahead to see `var`, and
error in the parse (`abstract var` is never valid). Instead, we save the
modifier and diagnose it during check.

The problem is that code isn't always this simple. Considering the above
example, there could be other modifiers, such as
`abstract private returned var x: i32 = 0;`, so single-token lookahead isn't a
general solution. Some modifiers are also contextually valid; for example,
`abstract fn` is only valid inside an `abstract class` scope. As a consequence,
a form of either arbitrary lookahead or additional context would be necessary in
parse in order to reliably diagnose incorrect uses of `abstract`. In contrast
with parse, check will have that additional context.

Rejecting incorrect code during parsing can also have negative consequences for
diagnostics. The additional information that check has about semantics may
produce better diagnostics. Alternately, sometimes check will produce
diagnostics equivalent to what parse could, but with less work overall.

As a consequence, at times we will defer to the check phase to produce
diagnostics instead of trying to produce those same diagnostics during parse.
Some examples of why we might diagnose in check instead of parse are:

-   To issue better diagnostics based on semantic information.
-   To diagnose similar invalid uses in one place, versus partly in check and
    partly in parse.
-   To support syntax highlighting for IDEs in near-correct code, still being
    typed.

Some examples of why we might diagnose in parse are:

-   When it's important to distinguish between multiple possible syntaxes.
-   When permitting the syntax would require more work than rejecting it.

A few examples of parse designs to avoid are:

-   Using arbitrary lookahead.
    -   Looking ahead one or two tokens is okay. However, we should never have
        arbitrary lookahead.
    -   This includes approaches which would require using the mapping of
        opening brackets to closing brackets that is produced by
        `TokenizedBuffer`. Those are helpful for error recovery.
-   Building complex context.
    -   We want parsing to be faster and lighter weight than check.
-   Duplicating diagnostics between parse and check.
    -   When there are closely related invalid variants of syntax, only some of
        which can be diagnosed during parse, consider diagnosing all variants
        during check.

This is a balance. We don't want to unnecessarily shift costs from parse onto
check, and we don't try to allow clearly invalid constructs. Parse still tries
to produce a reasonable parse tree. However, parse leans more towards a
permissive parse, and an error-free parse tree does not mean the code is
grammatically correct.

### Check

#### Using a traditional AST representation

Clang creates an AST as part of compilation. In Carbon, it's something we could
do as a step between parsing and checking, possibly replacing the SemIR. It's
likely that doing so would be simpler, amongst other possible trade-offs.
However, we think the SemIR approach is going to yield higher performance,
enough so that it's the chosen approach.

### Coalescing generic functions emitted when lowering to LLVM IR

#### Coalescing in the front-end vs back-end?

An alternative considered was not doing any coalescing in the front-end and
relying on LLVM to make the analysis and optimization. The current choice was
made based on the expectation that such an
[LLVM pass](https://llvm.org/docs/MergeFunctions.html) would be more costly in
terms of compile-time. The relative cost has not yet been evaluated.

#### When to do coalescing in the front-end?

The analysis and coalescing could be done prior to lowering, after
specialization. The advantage of that choice would be avoiding to lower
duplicate LLVM functions and then removing the duplicates. The disadvantage of
that choice would be duplicating much of the lowering logic, currently necessary
to make the equivalence determination.

#### Compile-time trade-offs

Not doing any coalescing is also expected to increase the back-end codegen time
more than performing the analysis and deduplication. This can be evaluated in
practice and the feature disabled if found to be too costly.

#### Coalescing duplicate non-specific functions

We could coalesce duplicate functions in non-specific cases, similar to lld's
[Identical Code Folding](https://lld.llvm.org/NewLLD.html#glossary) or LLVM's
[MergeFunctions pass](https://llvm.org/docs/MergeFunctions.html). This would
require fingerprinting all instructions in all functions, whereas specific
coalescing can focus on cases that only Carbon's front-end knows about. Carbon
would also be restricted to coalescing functions in a single compilation unit,
which would require replacing function definitions that allow external calls
with a placeholder that calls the coalesced definition. We don't expect
sufficient advantages over existing support.
