# Language design style guide

<!--
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

<!-- toc -->

## Table of contents

-   [Background](#background)
-   [General](#general)
-   [Linking](#linking)
-   [Document structure](#document-structure)
    -   [Overview and detailed design](#overview-and-detailed-design)
    -   [Alternatives considered](#alternatives-considered)
    -   [References](#references)

<!-- tocstop -->

## Background

The [language design](/docs/design) documentation in the Carbon project should
use a consistent style and tone, and should read as if it were written by a
single author. This document describes structural, stylistic, and formatting
conventions for the language design, where they have been established.

## General

The language design documentation follows the
[style conventions](/CONTRIBUTING.html#google-docs-and-markdown) for Carbon
documentation.

## Linking

-   Links to issues and to complete proposals should use the text `#nnnn`, where
    `nnnn` is the issue number, and should link to the issue or pull request on
    GitHub. It can be helpful to mention the proposal title. For example, `[#123](https://github.com/carbon-language/carbon-lang/pull/123): Widget painting`.
-   Links to specific sections of a proposal should use section title and link to the repository copy of the proposal file. For example,
    `[Painting details](/docs/proposals/p0123.md#painting-details)`

## Document structure

Each document within the language design should be divided into the following
sections, with suitable level-two (`##`) headings:

-   **Table of contents** (auto-generated)
-   **TODO** (optional)
-   **Overview**
-   Zero or more detailed design sections
-   **Alternatives considered**
-   **References**

### Overview and detailed design

The overview should describe the high-level concepts of this area of the design,
following BLUF principles. Where the overview does not fully cover the detailed
design, additional sections can be added as needed to more completely describe
the design.

The aim of these sections is to describe the design choices that have been made,
how those choices fit into the overall design of Carbon, the rationale for those
choices, and how and why those choices differ from other languages to which
Carbon is likely to be compared, particularly C++, Rust, and Swift.

### Alternatives considered

This section should provide bullet points briefly describing alternative designs
that were considered, along with references to the proposals in which those
designs were discussed. For example:

```md
-   [Paint widgets from bottom to top](/docs/proposals/p999999.md#alternatives-considered).
```

### References

This section should provide bullet points linking to the following:

-   External documents providing background on the topic or additional useful
    information.
-   Each proposal that contributed to the design described in this document.

For example:

```md
-   [Wikipedia example page](https://en.wikipedia.org/wiki/Wikipedia:Example)
-   Proposal
    [#999999](https://github.com/carbon-language/carbon-lang/pull/999999):
    widget painting.
```

Links to related parts of the design should be included inline, where relevant,
not in the references section.
