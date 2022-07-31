# Contributing

<!--
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

<!-- toc -->

## Table of contents

-   [Overview](#overview)
-   [Ways to contribute](#ways-to-contribute)
    -   [Help comment on proposals](#help-comment-on-proposals)
    -   [Help contribute ideas to Carbon](#help-contribute-ideas-to-carbon)
    -   [Help implement Carbon's design](#help-implement-carbons-design)
    -   [Help address bugs](#help-address-bugs)
    -   [Good first issues](#good-first-issues)
-   [How to become a contributor to Carbon](#how-to-become-a-contributor-to-carbon)
    -   [Contributor License Agreements (CLAs)](#contributor-license-agreements-clas)
        -   [Future CLA plans](#future-cla-plans)
    -   [Collaboration systems](#collaboration-systems)
    -   [Contribution tools](#contribution-tools)
    -   [Contribution guidelines and standards](#contribution-guidelines-and-standards)
        -   [Guidelines and philosophy for contributions](#guidelines-and-philosophy-for-contributions)
-   [Style](#style)
    -   [Google Docs and Markdown](#google-docs-and-markdown)
    -   [Other files](#other-files)
-   [License](#license)
    -   [Google Docs](#google-docs)
    -   [Markdown](#markdown)
    -   [Other files](#other-files-1)
-   [Workflow](#workflow)
-   [Acknowledgements](#acknowledgements)

<!-- tocstop -->

## Overview

Thank you for your interest in contributing to Carbon! There are many ways to
contribute, and we appreciate all of them. If you have questions, please feel
free to ask on Discord or
[GitHub](https://github.com/carbon-language/carbon-lang/discussions).

Everyone who contributes to Carbon is expected to:

-   Read and follow the [Code of Conduct](CODE_OF_CONDUCT.md). We expect
    everyone in our community to be welcoming, helpful, and respectful.
-   Ensure you have signed the
    [Contributor License Agreement (CLA)](https://cla.developers.google.com/).
    We need this to cover some legal bases.

We also encourage anyone interested in contributing to check out all the
information here in our contributing guide, especially the
[guidelines and philosophy for contributions](#guidelines-and-philosophy-for-contributions)

## Ways to contribute

### Help comment on proposals

If you're looking for a quick way to contribute, commenting on proposals is a
way to provide proposal authors with a breadth of feedback.
[Issues for leads](https://github.com/carbon-language/carbon-lang/projects/2)
has questions the community is looking for a decision on. The
[list of open proposals](https://github.com/carbon-language/carbon-lang/issues?q=is%3Aopen+label%3Aproposal+draft%3Afalse)
will have more mature proposals that are nearing a decision. For more about the
difference, see the [evolution process](docs/project/evolution.md).

When giving feedback, please keep comments positive and constructive. Our goal
is to use community discussion to improve proposals and assist authors.

### Help contribute ideas to Carbon

If you have ideas for Carbon, we encourage you to discuss it with the community,
and potentially prepare a proposal for it. Ultimately, any changes or
improvements to Carbon will need to turn into a proposal and go through our
[evolution process](docs/project/evolution.md).

If you do start working on a proposal, keep in mind that this requires a time
investment to discuss the idea with the community, get it reviewed, and
eventually implemented. A good starting point is to read through the
[evolution process](docs/project/evolution.md). We encourage discussing the idea
early, before even writing a proposal, and the process explains how to do that.

### Help implement Carbon's design

Eventually, we will also be working toward a reference implementation of Carbon,
and are very interested in folks joining in to help us with it.

### Help address bugs

As Carbon's design and eventually implementation begin to take shape, we'll
inevitably end up with plenty of bugs. Helping us triage, analyze, and address
them is always a great way to get involved. See
[open issues on GitHub](https://github.com/carbon-language/carbon-lang/issues).

### Good first issues

Some issues have been marked as
["good first issues"](https://github.com/carbon-language/carbon-lang/labels/good%20first%20issue).
These are intended to be a good place to start contributing.

## How to become a contributor to Carbon

### Contributor License Agreements (CLAs)

We'd love to accept your documentation, pull requests, and comments! Before we
can accept them, we need you to cover some legal bases.

Please fill out either the individual or corporate CLA.

-   If you are an individual contributing to spec discussions or writing
    original source code and you're sure you own the intellectual property, then
    you'll need to sign an
    [individual CLA](https://code.google.com/legal/individual-cla-v1.0.html).
-   If you work for a company that wants to allow you to contribute your work,
    then you'll need to sign a
    [corporate CLA](https://code.google.com/legal/corporate-cla-v1.0.html).

Follow either of the two links above to access the appropriate CLA and
instructions for how to sign and return it. Once we receive it, we'll be able to
accept your documents, comments and pull requests.

**_NOTE_**: Only original content from you and other people who have signed the
CLA can be accepted as Carbon contributions: this covers GitHub (including both
code and discussion), Google Docs, and Discord.

#### Future CLA plans

Initially, Carbon is bootstrapping using Google's CLA. We are planning to create
an open source foundation and transfer all Carbon-related rights to it; our goal
is for the foundation setup to be similar to other open source projects, such as
LLVM or Kubernetes.

### Collaboration systems

We use a few systems for collaboration which contributors should be aware of.

Before using these systems, everyone must sign the CLA. They are all governed by
the Code of Conduct.

-   [The GitHub carbon-language organization](https://github.com/carbon-language)
    is used for our repositories.
-   [Discord](https://discord.gg/ZjVdShJDAs) is used for chat.
-   [A shared Google Drive](https://drive.google.com/corp/drive/folders/18AFPsUWNXfAloZx0tRHTsrdCWUlJLpeW)
    is used for all of our Google Docs, particularly proposal drafts.
-   [Google Calendar](https://calendar.google.com/calendar/embed?src=c_07td7k4qjq0ssb4gdl6bmbnkik%40group.calendar.google.com)
    is used for meeting invites and project reminders. Contributors may add
    calendar entries for meetings added to discuss details. Standard entries
    are:

    -   The
        [weekly sync](https://docs.google.com/document/d/1dwS2sJ8tsN3LwxqmZSv9OvqutYhP71dK9Dmr1IXQFTs/edit?resourcekey=0-NxBWgL9h05yD2GOR3wUisg),
        where contributors are welcome.
    -   [Open discussions](https://docs.google.com/document/d/1tEt4iM6vfcY0O0DG0uOEMIbaXcZXlNREc2ChNiEtn_w/edit),
        which are unstructured meeting slots used for discussing proposals,
        tooling, and other Carbon topics based on who attends.

Note that commenting on Google Docs, attending meetings, and some label changes
in GitHub will require some contributor access: make sure you've
[signed the CLA](#contributor-license-agreements-clas) then ask for access on
[#getting-started](https://discord.com/channels/655572317891461132/655577725347561492)
on Discord.

### Contribution tools

Please see our [contribution tool](/docs/project/contribution_tools.md)
documentation for information on setting up a git client for Carbon development,
as well as helpful tooling that will ease the contribution process. For example,
[pre-commit](/docs/project/contribution_tools.md#pre-commit) is used to simplify
[code review](/docs/project/code_review.md).

### Contribution guidelines and standards

All documents and pull requests must be consistent with the guidelines and
follow the Carbon documentation and coding styles.

#### Guidelines and philosophy for contributions

-   For **both** documentation and code:

    -   When the Carbon team accepts new documentation or features, to Carbon,
        by default they take on the maintenance burden. This means they'll weigh
        the benefit of each contribution against the cost of maintaining it.
    -   The appropriate [style](#style) is applied.
    -   The [license](#license) is present in all contributions.
    -   [Code review](/docs/project/code_review.md) is used to improve the
        correctness, clarity, and consistency of all contributions.
        -   Please avoid rebasing PRs after receiving comments; it can break
            viewing of the comments in files.

-   For documentation:

    -   All documentation is written for clarity and readability. Beyond fixing
        spelling and grammar, this also means content is worded to be accessible
        to a broad audience.
    -   Substantive changes to Carbon follow the
        [evolution process](docs/project/evolution.md). Pull requests are only
        sent after the documentation changes have been accepted by the reviewing
        team.
    -   Typos or other minor fixes that don't change the meaning of a document
        do not need formal review, and are often handled directly as a pull
        request.

-   For code:

    -   New features should have a documented design that has been approved
        through the [evolution process](docs/project/evolution.md). This
        includes modifications to preexisting designs.
    -   Bug fixes and mechanical improvements don't need this.
    -   All new features include unit tests, as they help to (a) document and
        validate concrete usage of the feature and its edge cases, and (b) guard
        against future breaking changes to lower the maintenance cost.
    -   Bug fixes also generally include unit tests, because the presence of
        bugs usually indicates insufficient test coverage.
    -   Unit tests must pass with the changes.
    -   If some tests fail for unrelated reasons, we wait until they're fixed.
        It helps to contribute a fix!
    -   Code changes should be made with API compatibility and evolvability in
        mind.
    -   Keep in mind that code contribution guidelines are incomplete while we
        start work on Carbon, and may change later.

## Style

### Google Docs and Markdown

Changes to Carbon documentation follow the
[Google developer documentation style guide](https://developers.google.com/style).

Markdown files should additionally use
[Prettier](/docs/project/contribution_tools.md#prettier) for formatting.

Other style points to be aware of are:

-   Whereas the Google developer documentation style guide
    [says to use an em dash](https://developers.google.com/style/dashes)
    (`text—text`), we are using a double-hyphen with surrounding spaces
    (`text -- text`). We are doing this because we frequently read Markdown with
    fixed-width fonts where em dashes are not clearly visible.
-   Prefer the term "developers" when talking about people who would write
    Carbon code. We expect the Carbon's community to include people who think of
    themselves using many titles, including software developers, software
    engineers, systems engineers, reliability engineers, data scientists,
    computer scientists, programmers, and coders. We're using "developers" to
    succinctly cover the variety of titles.

### Other files

If you're not sure what style to use, please ask on Discord or GitHub.

## License

A license is required at the top of all documents and files.

### Google Docs

Google Docs all use
[this template](https://docs.google.com/document/d/1sqEnIWWZKTrtMz2XgD7_RqvogwbI0tBQjAZIvOabQsw/template/preview).
It puts the license at the top of every page if printed.

### Markdown

Markdown files always have at the top:

```
# DOC TITLE

<!--
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->
```

For example, see the top of
[CONTRIBUTING.md](https://github.com/carbon-language/carbon-lang/raw/trunk/CONTRIBUTING.md)'s
raw content.

### Other files

Every file type uses a variation on the same license text ("Apache-2.0 WITH
LLVM-exception") with similar formatting. If you're not sure what text to use,
please ask on Discord or GitHub.

## Workflow

Carbon repositories all follow a common
[pull-request workflow](docs/project/pull_request_workflow.md) for landing
changes. It is a trunk-based development model that emphasizes small,
incremental changes and preserves a simple linear history.

## Acknowledgements

Carbon's Contributing guidelines are based on
[Tensorflow](https://github.com/tensorflow/tensorflow/blob/master/CONTRIBUTING.md)
and [Flutter](https://github.com/flutter/flutter/blob/master/CONTRIBUTING.md)
guidelines. Many thanks to these communities for their help in providing a
basis.
