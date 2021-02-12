# Documentation style guide

<!--
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

<!-- toc -->

## Table of contents

-   [Background](#background)
-   [Baseline](#baseline)
-   [Carbon-local guidance](#carbon-local-guidance)
    -   [Documenting "provisionality"](#documenting-provisionality)
        -   [FIXME](#fixme)
        -   [Question for reviewers](#question-for-reviewers)
        -   [TODO](#todo)
        -   [Open question](#open-question)
        -   [Placeholder](#placeholder)
        -   [Provisional](#provisional)
        -   [Experimental](#experimental)

<!-- tocstop -->

## Background

Documentation in the Carbon project should use a consistent and well documented
style guide. However, we are not in the business of innovating significantly in
the space of technical writing, and so we work primarily to reuse existing best
practices and guidelines.

## Baseline

The baseline style guidance is the
[Google developer documentation style guide](https://developers.google.com/style).

Markdown files should additionally use
[Prettier](/docs/project/contribution_tools.md#prettier) for formatting.

## Carbon-local guidance

Other style points to be aware of are:

-   Whereas the Google developer documentation style guide
    [says to use an em dash](https://developers.google.com/style/dashes)
    (`text—text`), we are using a double-hyphen with surrounding spaces
    (`text -- text`). We are doing this because we frequently read Markdown with
    fixed-width fonts where em dashes are not clearly visible.
-   Always say "Discourse Forum" and "Discord Chat" to avoid confusion between
    systems.
-   Prefer the term "developers" when talking about people who would write
    Carbon code. We expect the Carbon's community to include people who think of
    themselves using many titles, including software developers, software
    engineers, systems engineers, reliability engineers, data scientists,
    computer scientists, programmers, and coders. We're using "developers" to
    succinctly cover the variety of titles.

### Documenting "provisionality"

Although all aspects of Carbon's design can be changed through the
[evolution process](docs/project/evolution.md), some parts of Carbon's design
will be more provisional than others, particularly early in their development.
In order to communicate that clearly, we have established a set of shorthand
terms to concisely indicate that some of a document's content is provisional in
some way.

For ease of skimming, we recommend putting these terms in boldface (delimited by
`**` in markdown). These terms will normally be part of some text that explains
what parts are provisional and why. When that text is more than a single
sentence, we recommend formatting it as a separate block-quote paragraph (using
a leading `>` in markdown).

These shorthands should gradually be removed from Carbon's documentation as we
answer the underlying questions and build confidence in the design. However,
that won't happen for numbered proposals, because they are generally not
"living" documents. Consequently, when a shorthand in a numbered proposal
indicates a need for future work, as with "TODO", "open question", and
especially "experimental", consider including a link to a tracking issue, as in
[the "experimental" example](#experimental), so that future readers can see how
the issue was resolved.

The specific shorthands are described below. It's important to bear in mind that
they are inherently somewhat fuzzy and subjective, because they're intended to
convey social rather than technical information.

#### FIXME

"FIXME" indicates material that needs to be added or changed before the proposal
is accepted. It should never appear in the Carbon repository itself, except in
places like the previous sentence, where it is being mentioned rather than used.
For example:

```markdown
> **FIXME:** Add a better rationale here.
```

#### Question for reviewers

"Question for reviewers" indicates a question that needs to be resolved before
the proposal is accepted. In the Carbon repository, it should appear only in
numbered proposal documents, with the answer captured in the corresponding
[formal decision](/docs/project/consensus_decision_making.html#formal-decision-content)
document. Everywhere else, it should be removed during the code review process.

```markdown
> **Question for reviewers:** Which of these directions should we adopt?
```

#### TODO

"TODO" indicates material that needs to be added or changed at some time after
the proposal is accepted, for example in a followup proposal, without specifying
how it should be done. This normally should only appear in documentation for
features that are still under active development. For example:

```markdown
> **TODO:** Provide a full API specification of this type.
```

#### Open question

"Open question" indicates questions that need to be resolved at some time after
the proposal is submitted. It is equivalent to "**TODO:** figure out the answer
to this question". For example:

```markdown
> **Open question:** Can we support this at run time, or only at compile time?
```

#### Placeholder

"Placeholder" indicates something that would otherwise be a TODO, but we've
filled in an arbitrary answer in order to concretely discuss other issues. It
indicates that we have at least moderate confidence that some Carbon feature
will fill this role, but its design remains wide open. This will most often be
used for syntax, but sometimes a semantic choice can be a placeholder, if it's
clear to the reader why it doesn't matter to the substance of the discussion.
For example:

```markdown
In the following discussion, we use C++-like `[...](...){...}` syntax as a
**placeholder** for whatever syntax Carbon uses for lambdas.
```

#### Provisional

"Provisional" indicates material that the Carbon team has not evaluated in
detail, but has adopted (or will, if the proposal is accepted) as their best
guess at how that aspect of Carbon will work. For example:

```markdown
> The design direction described in this proposal is **provisional**, and will
> be fleshed out in more detail by subsequent proposals.
```

This is primarily intended to help us "bootstrap" the design process by quickly
sketching the overall shape of a group of language features, without committing
to any of their specifics. Any design choice marked "provisional" will need to
be revisited by a future proposal that evaluates the tradeoffs and alternatives
in more depth, and may reach a different decision. However, proposal authors are
encouraged to take provisional design choices as givens, if that helps unblock
their own work.

#### Experimental

"Experiment" or "experimental" indicates that the Carbon team has chosen to try
something (or will, if the proposal is adopted), but considers it relatively
high-risk. It invites future proposals to call attention to areas where the
choice is creating obstacles, or even to argue for overturning that choice if
the evidence warrants. To the extent possible, the text should explain what
risks we see, and what kind of evidence we're looking for. In user-facing
documents, if the experiment is seeking feedback from user experience, we
recommend including a link to a forum thread or tracking bug where that feedback
can be collected. For example:

```markdown
The decision to use prefix notation for all arithmetic is **experimental**, and
may need to be revisited if user experience (which is being collected
[here](https://forums.carbon-lang.dev/t/dummy-thread) shows that it's
unworkable.
```

"Experimental" may be especially useful in principles docs, because that gives
us a lightweight way of making explicit some of the background design ideas that
are currently guiding Carbon's development.
