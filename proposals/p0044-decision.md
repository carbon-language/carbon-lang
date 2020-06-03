# Decision for: Carbon: Proposal Tracking

<!--
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

Proposal accepted on 2020-06-02

Affirming:

- [Chandler Carruth](https://github.com/chandlerc)
- [Dmitri Gribenko](https://github.com/gribozavr)
- [Geoff Romer](https://github.com/geoffromer)
- [Josh Levenberg](https://github.com/josh11b)
- [Matt Austern](https://github.com/austern)
- [Richard Smith](https://github.com/zygoloid)

Abstaining:

- [Chris Palmer](https://github.com/noncombatant)
- [Titus Winters](https://github.com/tituswinters)

## Open questions

### Do we use a Google Docs-centric or GitHub Markdown-centric flow?

**Decision:** GitHub Markdown-centric flow

### Where should proposals be stored in GitHub?

**Decision:** Store proposals in the repository they affect

### Should we push comments to focus on GitHub?

**Decision:** Push high-level comments to GitHub, rather than focusing these
discussions in Discourse.

### Should there be a tracking issue?

**Decision:** Don't require tracking issues

### Should declined/deferred proposals be committed?

**Decision:** Do not commit declined/deferred proposals

## Rationale

During the decision process, several of the individual rationales were
influenced by the idea that one could view the proposal process in one of two
ways:

1.  A PR-centric model. The review team is trying to achieve consensus around a
    PR as a whole. The PR may (and often will) implement the proposed changes.
    The proposal as essentially a description of the changes.

2.  A proposal-centric model. The review team is trying to achieve consensus
    around a proposal. The PR may show a preview of the implementation, but it
    is purely informative.

If one favors a PR-centric model, this steers one away from committing proposals
that are not accepted, towards committing proposals to the affected repository,
etc. In general, the PR-centric model was favored.

### Rationale for using a GitHub markdown-centric flow

- The GitHub markdown-centric flow makes the on-ramp as smooth as possible for
  external contributors.
  - This positions the project to maximize the ease of engaging-with and gaining
    contributions from the wider industry.
- The final documents to be in markdown form, so it is best if contributors have
  the option to stay in markdown for the whole process. This is significantly
  less complex than something that converts between formats:

  - Less to learn
  - Fewer steps in the process
  - No outdated versions in the old format left behind.

- The technical flow seems on balance better than the Google Docs-based
  workflow. The proposal does a really good job explaining pros and cons. In
  summary, the Google Docs-centric workflow has a lot of cons that make it
  difficult to work with proposals over the long term.

### Rationale for not requiring tracking issues

There were several members who had no strong preference on this issue. The
concensus was that until there is a compelling reason to require tracking
issues, the process is more light-weight without them.

### Rationale for not committing proposals that are declined or deferred

- This approach seems simpler.
- When a proposal PR includes the changes put forth in the proposal (PR-centric
  model), the declined PR might need to be considerably changed--and might lose
  context--in order to be committed.
- The community will put a lot of work into developing, discussing, and making a
  decision on a proposal. There may be valuable insight in rejected proposals,
  so it makes sense to archive them. However, as noted, committing the PR will
  not always be possible with reasonable effort if not working in a
  proposal-centric model, as the proposal text may not stand on its own.
- While we may discover issues with this approach, it is better to try this way,
  see if any issues can be rectified, and propose changes as necessary.

### Rationale for committing a proposal to the repository it affects

- This keeps the proposal close to the repository, and therefore, the community,
  that it affects.
- It facilitates autonomy of (future) review teams responsible for a particular
  aspect of Carbon. (For example, a reference implementation team responsible
  for the carbon-toolchain repository).
- It simplifies the common case and makes it easier to find how each repository
  evolves over time.

### Rationale for pushing high-level comments to GitHub

While opinions were not as strong, reasons given for perferring comments in
GitHub:

- This flow will maximize the alignment with “normal” GitHub development flow.
  - This both improves pulling external/new people into the flow, and will
    reduce the number of flows they need to learn/remember/tool for.
- We will get ecosystem benefits as this flow continues to be optimized by
  GitHub and those using it.
