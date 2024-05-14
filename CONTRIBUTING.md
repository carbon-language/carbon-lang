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
    -   [Contributing to the language design](#contributing-to-the-language-design)
        -   [Comment on proposals](#comment-on-proposals)
        -   [Contribute design ideas to Carbon](#contribute-design-ideas-to-carbon)
    -   [Contributing to the language implementation](#contributing-to-the-language-implementation)
        -   [Review and comment on Pull Requests (no code)](#review-and-comment-on-pull-requests-no-code)
        -   [Implement Carbon's design](#implement-carbons-design)
        -   [Triage, analyze or address bugs](#triage-analyze-or-address-bugs)
-   [How to become a contributor to Carbon](#how-to-become-a-contributor-to-carbon)
    -   [Contributor License Agreements (CLAs)](#contributor-license-agreements-clas)
        -   [Future CLA plans](#future-cla-plans)
    -   [Collaboration systems](#collaboration-systems)
        -   [Getting access](#getting-access)
    -   [Contribution tools](#contribution-tools)
    -   [Contribution guidelines and standards](#contribution-guidelines-and-standards)
        -   [Guidelines and philosophy for contributions](#guidelines-and-philosophy-for-contributions)
        -   [How to say things](#how-to-say-things)
            -   [Make your point concisely](#make-your-point-concisely)
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
free to ask on [Discord](https://discord.gg/ZjVdShJDAs) `#contributing-help`
channel or [GitHub](https://github.com/carbon-language/carbon-lang/discussions).

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

### Contributing to the language design

#### Comment on proposals

If you're looking for a quick way to contribute, commenting on proposals is a
way to provide proposal authors with a breadth of feedback. The
["leads questions" label](https://github.com/carbon-language/carbon-lang/issues?q=is%3Aissue+is%3Aopen+label%3A%22leads+question%22)
has questions the community is looking for a decision on. The
[list of open proposals](https://github.com/carbon-language/carbon-lang/pulls?q=is%3Apr+is%3Aopen+label%3A%22proposal+rfc%22)
will have more mature proposals that are nearing a decision. For more about the
difference, see the [evolution process](docs/project/evolution.md).

When giving feedback, please keep comments positive, constructive, and
[concise](#make-your-point-concisely). Our goal is to use community discussion
to improve proposals and assist authors.

#### Contribute design ideas to Carbon

If you have ideas for Carbon, we encourage you to discuss it with the community,
and potentially prepare a proposal for it. Ultimately, any changes or
improvements to Carbon will need to turn into a proposal and go through our
[evolution process](docs/project/evolution.md).

If you do start working on a proposal, keep in mind that this requires a time
investment to discuss the idea with the community, get it reviewed, and
eventually implemented. A good starting point is to read through the
[evolution process](docs/project/evolution.md). We encourage discussing the idea
early, before even writing a proposal, and the process explains how to do that.

### Contributing to the language implementation

#### Review and comment on Pull Requests (no code)

Helping with
[pull requests](https://github.com/carbon-language/carbon-lang/pulls) review is
a good way to provide feedback, while getting a acquainted with the code base.

#### Implement Carbon's design

The implementation of the Carbon language design is currently focused on the
[Carbon toolchain](/toolchain/) (see Carbon
[toolchain issues](https://github.com/carbon-language/carbon-lang/issues?q=is%3Aissue+is%3Aopen+label%3Atoolchain))

**Some issues are also marked as
["good first issues"](https://github.com/carbon-language/carbon-lang/labels/good%20first%20issue)**.
These are intended to be a good place to start contributing.

To pick up a "good first issue", check to make sure there's no in-flight pull
request on the issue, and then start working on it. We don't assign issues to
new contributors because some people have different time constraints, and we
want new contributors to feel welcome to pick the issue up when it may not be
making progress. Even if someone else merges a fix before you, these issues
should be a quick and helpful way to start learning how Carbon is built,
building towards larger contributions.

#### Triage, analyze or address bugs

As Carbon's design and implementation take shape, we'll inevitably encounter
plenty of bugs. Helping us triage, analyze, and address them is always a great
way to get involved. See
[open issues on GitHub](https://github.com/carbon-language/carbon-lang/issues).

When triaging issues, we typically won't assign issues because we want to be
confident that contributors who have an issue assigned to them are planning for
the amount of time it will take, which requires familiarity. Contributors with
write access are expected to have that familiarity and may assign issues to
themselves.

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
-   [A shared Google Drive](https://drive.google.com/drive/folders/1aC5JJ5EcI8B7cgVDrLvO7WNw97F0LpS2)
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

#### Getting access

Our collaboration systems are all viewable publicly, and most can be joined
without particular requests. However, some require extra permissions, such as
editing Google Docs, joining meetings, or some details of the proposal process.

When requesting any of the following access, please provide a reason for the
access. All requests require a
[signed CLA](#contributor-license-agreements-clas).

-   Google Docs/Calendar access groups:
    -   **Commenter** access:
        [join group](https://groups.google.com/a/carbon-lang.dev/g/commenters/about)
        -   [Google Docs](https://drive.google.com/drive/folders/1aC5JJ5EcI8B7cgVDrLvO7WNw97F0LpS2):
            Comment on files.
        -   [Google Calendar](https://calendar.google.com/calendar/embed?src=c_07td7k4qjq0ssb4gdl6bmbnkik%40group.calendar.google.com):
            View event details.
    -   **Contributor** access:
        [join group](https://groups.google.com/a/carbon-lang.dev/g/contributors/about)
        -   [Google Docs](https://drive.google.com/drive/folders/1aC5JJ5EcI8B7cgVDrLvO7WNw97F0LpS2):
            Add, edit, and comment on files.
        -   [Google Calendar](https://calendar.google.com/calendar/embed?src=c_07td7k4qjq0ssb4gdl6bmbnkik%40group.calendar.google.com):
            View and edit event details.
    -   After you apply to join, please let us know on
        [#access-requests](https://discord.com/channels/655572317891461132/1006221387574292540);
        we don't get notifications otherwise.
-   GitHub Label/project contributor access:
    [ask on #access-requests](https://discord.com/channels/655572317891461132/1006221387574292540)
    -   Don't forget to mention your GitHub username.
    -   Used by the proposal process.

If you simply want to chime in on GitHub or Discord, none of this is needed. If
you're interested in joining meetings, ask for commenter access. If you're
trying to write proposals, both types of contributor access will help.

### Contribution tools

Please see our [contribution tool](/docs/project/contribution_tools.md)
documentation for information on setting up a git client for Carbon development,
as well as helpful tooling that will ease the contribution process. For example,
[pre-commit](https://pre-commit.com) is used to simplify
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

#### How to say things

-   Treat others with respect.
-   Recognize that other points of view are valid, and most decisions are about
    choosing the best set of trade-offs or closest alignment with Carbon's goals
    since there usually isn't a single best answer. It is important to frame
    feedback and discussion about someone else's proposal under the assumption
    that they too have deep experience in the area but may have come to a
    different conclusion.
-   Be clear when something is an opinion by using "I" or "me":
    -   Not as helpful: "`foo` is objectively better word to use"
    -   More helpful: "I find `bar` confusing since it has this alternate
        meaning, I think `foo` is clearer."
-   It can be helpful to define your terms.

When trying to make a point, please employ these strategies to make your
argument _effective_ and _helpful_:

-   Focus should be on explaining the basis by which others can come to a
    conclusion. Generally this means connecting a potential solution to the use
    cases it helps with.
-   Be specific and concrete, using examples to demonstrate the benefits and
    disadvantages of the different options
    -   Minimally helpful: "I like &lt;X>", "Carbon should have feature &lt;Y>"
    -   More helpful: "I think Carbon should have feature &lt;Z>, it would mean
        &lt;this example> would be written &lt;like this> instead of &lt;like
        that>." (Assuming the reader will think the example is representative of
        an important class of use cases.)
    -   Very helpful: "If we go with approach &lt;X>, it helps with problem
        &lt;P> in &lt;this example> of use case &lt;A>. It doesn't help with use
        case &lt;B>, but that can be better solved by &lt;Y>, as can be seen in
        &lt;other example>."
    -   Very helpful: "Yes &lt;X> gives a nice answer in that case, but it has a
        problem / I don't see how it applies in &lt;this other situation>."
-   Explain the reasoning by which you can come to your conclusion
    -   Avoid [fallacies](https://en.wikipedia.org/wiki/List_of_fallacies) like
        [arguing from authority](https://en.wikipedia.org/wiki/Argument_from_authority)
    -   Don't expect people to read long works to understand your point

If someone questions or argues with your point, try to directly address the
points being made. Try not to step backwards or switch to a more general or more
meta level as that can seem like you're evading the question.

##### Make your point concisely

-   **Asking questions is OK**: Asking questions about a particular discussion
    is almost always OK. If you're worried the questions might be more about
    _background_ and might be long enough to get distracting, you can always ask
    them in some of our dedicated spaces like `#language-questions` on Discord.
-   **Favor new and relevant information**: When sharing ideas and opinions in
    community discussions, it is important to do so in a way that both _adds new
    information_ in some way/shape/form, makes sure that information is
    _relevant_ to the discussion, and avoids _repetition_. This means reviewing
    what you write before posting a larger response, and editing it down to just
    the points you want to make that have not already been made, and any new
    arguments supporting those points.
-   **Be inclusive by staying concise**: We need to be mindful that writing many
    paragraphs of extra text is going to exclude people. Some people can be
    excluded from consuming the conversation, either because a wall of text is
    too intimidating to read, or they don't have the time or bandwidth to wade
    through the extra text to find the new information being conveyed. It can
    also drown out other contributors
-   **Prefer upvote to repetition**: One person saying "I don't like this
    feature" is useful, and that same message with 100 upvotes is extremely
    useful, but 100 people writing separate messages saying "I don't like this
    feature" is not. Emoji reactions are available in both Discord and GitHub
    and should be used for this purpose in both.

It is also okay to do things like ask a question to get clarification about what
someone has said or to solicit opinions about various options. Just be
respectful, and don't drown out other discussion.

## Style

### Google Docs and Markdown

Changes to Carbon documentation follow the
[Google developer documentation style guide](https://developers.google.com/style).

Markdown files should additionally use [Prettier](https://prettier.io) for
formatting, which we automate with
[pre-commit](/docs/project/contribution_tools.md#main-tools).

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
[this template](https://docs.google.com/document/d/1tAwE0230PDxweVruHUVY6DSfSnrJF2LnLOWXQYqNJuI/template/preview?usp=sharing&resourcekey=0-zsrwCWP7ictbxhCuePk-fw).
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
[TensorFlow](https://github.com/tensorflow/tensorflow/blob/master/CONTRIBUTING.md)
and [Flutter](https://github.com/flutter/flutter/blob/master/CONTRIBUTING.md)
guidelines. Many thanks to these communities for their help in providing a
basis.
