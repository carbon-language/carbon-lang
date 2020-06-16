# Contributing

<!--
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

Thank you for your interest in contributing to Carbon! There are many ways to
contribute, and we appreciate all of them. If you have questions, please feel
free to ask on our Discourse Forums or Discord Chat.

Everyone contributing to Carbon is expected to:

- Read and follow the [Code of Conduct](CODE_OF_CONDUCT.md). We expect everyone
  in our community to be welcoming, helpful, and respectful.
- Ensure you have signed the
  [Contributor License Agreement (CLA)](https://cla.developers.google.com/). We
  need this to cover some legal bases.

We also encourage anyone interested in contributing to check out all the
information here in our contributing guide, especially the
[guidelines and philosophy for contributions](#guidelines-and-philosophy-for-contributions)

## Ways to contribute

### Help comment on proposals

If you're looking for a quick way to contribute, commenting on proposals is a
way to provide proposal authors with a breadth of feedback. The "Evolution >
Ideas" forum is where authors will go for early, high-level feedback. The
"Evolution > Proposal reviews" forum will have more mature proposals that are
nearing the decision process. For more about the difference, see the
[evolution process](docs/project/evolution.md).

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
them is always a great way to get involved. When we have the bug tracker(s) set
up for this, we'll update this section with ideas of how to find these and get
started.

## How to become a contributor to Carbon

### Contributor License Agreements (CLAs)

We'd love to accept your documentation, pull requests, and comments! Before we
can accept them, we need you to cover some legal bases.

Please fill out either the individual or corporate CLA.

- If you are an individual contributing to spec discussions or writing original
  source code and you're sure you own the intellectual property, then you'll
  need to sign an
  [individual CLA](https://code.google.com/legal/individual-cla-v1.0.html).
- If you work for a company that wants to allow you to contribute your work,
  then you'll need to sign a
  [corporate CLA](https://code.google.com/legal/corporate-cla-v1.0.html).

Follow either of the two links above to access the appropriate CLA and
instructions for how to sign and return it. Once we receive it, we'll be able to
accept your documents, comments and pull requests.

**_NOTE_**: Only original content from you and other people who have signed the
CLA can be accepted as Carbon contributions: this covers the GitHub repository,
GitHub issues, Google Docs, Discourse Forums, and Discord Chat.

#### Future CLA plans

At present, we are using Google's CLA. In the future, we expect the Carbon
ownership and IP to formally transfer from Google to a Carbon-specific
foundation or other neutral third-party. When that happens, the foundation will
take ownership of providing a CLA.

### Collaboration systems

We use a few systems for collaboration which contributors should be aware of.
Membership is currently invite-only.

Before using these systems, everyone must sign the CLA. They are all governed by
the Code of Conduct.

- [The GitHub carbon-language organization](https://github.com/orgs/carbon-language)
  is used for our repositories. **To join:**

1.  Ask [an admin](docs/project/groups.md#admins) to send an invite, providing
    your GitHub account.
2.  Check your email to accept the invite, or try the standard
    [accept link](https://github.com/orgs/carbon-language/invitation?via_email=1)
    if you don't see the email.

- [Discourse Forums](https://forums.carbon-lang.dev) are used for long-form
  discussions. **To join:**

  1.  Go to [the forums](https://forums.carbon-lang.dev) and register your
      GitHub account.
      - You will be able to choose which GitHub email you want the forums to
        send email to.
  2.  [An admin](docs/project/groups.md#admins) will need to approve your
      registration.

- [Discord Chat](https://discord.com/app) is used for short-form chats. **To
  join:**

  1.  Ask [an admin](docs/project/groups.md#admins) for an invite link.
      - Please do not re-share the invite links: they're our only way to
        restrict access.
  2.  You will be prompted with the Code of Conduct. After reading it, click the
      check mark reaction icon at the bottom.

- [A shared Google Drive](https://drive.google.com/corp/drive/folders/0ALTu5Y6kc39XUk9PVA)
  is used for all of our Google Docs, particularly proposal drafts. **To join:**
  1.  Ask [an admin](docs/project/groups.md#admins) to invite you, providing
      your Google account email.
  2.  The admin will add you to the
      [Google Group](https://groups.google.com/g/carbon-lang-contributors) used
      for access.

### Contribution guidelines and standards

All documents and pull requests must be consistent with the guidelines and
follow the Carbon documentation and coding styles.

#### Guidelines and philosophy for contributions

- For **both** documentation and code:

  - When the Carbon team accepts new documentation or features, to Carbon, by
    default they take on the maintenance burden. This means they'll weigh the
    benefit of each contribution must be weighed against the cost of maintaining
    it.
  - The appropriate [style](#style) is applied.
  - The [license](#license) is present in all contributions.

- For documentation:

  - All documentation is written for clarity and readability. Beyond fixing
    spelling and grammar, this also means content is worded to be accessible to
    a broad audience.
  - Substantive changes to Carbon follow the
    [evolution process](docs/project/evolution.md). Pull requests are only sent
    after the documentation changes have been accepted by the reviewing team.
  - Typos or other minor fixes that don't change the meaning of a document do
    not need formal review, and are often handled directly as a pull request.

- For code:

  - New features should have a documented design that has been approved through
    the [evolution process](docs/project/evolution.md). This includes
    modifications to pre-existing designs.
  - Bug fixes and mechanical improvements don't need this.
  - All new features include unit tests, as they help to (a) document and
    validate concrete usage of the feature and its edge cases, and (b) guard
    against future breaking changes to lower the maintenance cost.
  - Bug fixes also generally include unit tests, because the presence of bugs
    usually indicates insufficient test coverage.
  - Unit tests must pass with the changes.
  - If some tests fail for unrelated reasons, we wait until they're fixed. It
    helps to contribute a fix!
  - Code changes are made with API compatibility and evolvability in mind.
    Reviewers will comment on any API compatibility issues.
  - Keep in mind that code contribution guidelines are incomplete while we start
    work on Carbon, and may change later.

## pre-commit

We use [pre-commit](/docs/project/contribution_tools.md#pre-commit) for
automated checks, including formatting of documentation. Please see
[contribution tools](/docs/project/contribution_tools.md) for more information.

Please use pre-commit to simplify reviews.

## Style

### Google Docs and Markdown

Changes to Carbon documentation follow the
[Google developer documentation style guide](https://developers.google.com/style).

Markdown files should additionally use
[Prettier](/docs/project/contribution_tools.md#prettier) for formatting.

Other style points to be aware of are:

- Always say "Discourse Forum" and "Discord Chat" to avoid confusion between
  systems.

### Other files

If you're not sure what style to use, please ask on Discourse Forums.

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
[CONTRIBUTING.md](https://github.com/carbon-language/carbon-lang/raw/master/CONTRIBUTING.md)'s
raw content.

### Other files

Every file type uses a variation on the same license text ("Apache-2.0 WITH
LLVM-exception") with similar formatting. If you're not sure what text to use,
please ask on Discourse Forums.

## Acknowledgements

Carbon's Contributing guidelines are based on
[Tensorflow](https://github.com/tensorflow/tensorflow/blob/master/CONTRIBUTING.md)
and [Flutter](https://github.com/flutter/flutter/blob/master/CONTRIBUTING.md)
guidelines. Many thanks to these communities for their help in providing a
basis.
