# Create initial rough framework for specification

<!--
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

[Pull request](https://github.com/carbon-language/carbon-lang/pull/140)

## Table of contents

<!-- toc -->

## Table of contents

-   [Problem](#problem)
-   [Proposal](#proposal)
-   [Details](#details)
    -   [Conventions](#conventions)
-   [Alternatives considered](#alternatives-considered)
    -   [Maintain the specification in a different language.](#maintain-the-specification-in-a-different-language)

<!-- tocstop -->

## Problem

We need a rough layout for our specification so that we can start adding details
to it once they're decided.

## Proposal

Split the specification into a language and a library section. In the language
section, use one file per broad area of functionality. Divide the language up
based on the intended layering of the language design.

For now, maintain the specification sources in Markdown.

## Details

Proposed top-level structure of the `spec/` directory as of this pull request:

-   `README.md` Introduction to the specification
-   `lang`
    -   `README.md` Language specification overview and basics
    -   `execution.md` Execution semantics
    -   `lex.md` Lexical analysis
    -   `libs.md` Libraries and packages
    -   `names.md` Names and name binding / lookup
    -   `parsing.md` Parsing
    -   `semantics.md` Semantic analysis
-   `lib`
    -   `README.md` Library specification overview and basics

This is only a starting point; the structure should be expected to change and
grow as the specification is filled out. Most of the proposed files are empty or
nearly-empty placeholders.

### Conventions

All paragraphs within the specification are numbered so that they can be
referenced more easily.

Defined terms are introduced in italics.

Hyperlinks between sections of the specification are used liberally.

## Alternatives considered

### Maintain the specification in a different language.

Advantages:

-   An alternative language may provide better support for custom typesetting,
    representing grammars, linking to definitions, and so on.

Disadvantages:

-   Using a different language would add complexity and inconsistency to our
    documentation.
-   There is unlikely to be any existing documentation language that is
    well-suited to our needs without significant customization.
-   Conversion from a more sophisticated language is likely to be more complex
    than converting from Markdown.

Conversion of Markdown to another language at a later point (either manually or
using a tool like Sphinx) is expected to remain a relatively low-cost option,
due to the relative simplicity of Markdown-formatted documents.
