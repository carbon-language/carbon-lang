# Code review

<!--
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

## Table of contents

<!-- toc -->

- [High level goals of code review](#high-level-goals-of-code-review)
- [What requires review?](#what-requires-review)
- [Who should review?](#who-should-review)
- [GitHub pull request mechanics](#github-pull-request-mechanics)
- [Resolving an impasse or conflict](#resolving-an-impasse-or-conflict)
- [Escalation](#escalation)
- [Code author guide](#code-author-guide)
  - [Write good change descriptions](#write-good-change-descriptions)
    - [First line](#first-line)
    - [Body](#body)
  - [Make small changes](#make-small-changes)
  - [Responding to review comments](#responding-to-review-comments)
    - [Responding to questions or confusion](#responding-to-questions-or-confusion)
    - [Understand the feedback in the comments](#understand-the-feedback-in-the-comments)
- [Code reviewer guide](#code-reviewer-guide)
  - [How quickly should you respond to a review request?](#how-quickly-should-you-respond-to-a-review-request)
  - [What should be covered by a review?](#what-should-be-covered-by-a-review)
  - [Writing review comments](#writing-review-comments)
  - [Approving the change](#approving-the-change)

<!-- tocstop -->

## High level goals of code review

Code review serves several goals in the Carbon project. It directly improves the
following aspects of the code or change:

- Correctness
- Ease of understanding by someone other than the author
- Consistentcy with the codebase

It also provides important benefits to the project as a whole:

- Promotes team ownership.
- Spreads knowledge across the team.

A more detailed discussion of the goals of code review can be found in chapter 9
"Code Review" of the book
_[Software Engineering at Google](https://www.amazon.com/Software-Engineering-Google-Lessons-Programming/dp/1492082791)_.

## What requires review?

Every change to Carbon's repositories requires code review. Even formal
[evolution decisions](evolution.md) which have been approved should have their
specific changes to the repository reviewed.

The term "code review" in the Carbon project is about more than just "code". We
expect changes to any files to be reviwed, including documentation and any other
material stored in the repository.

Many changes to Carbon repositories may _only_ require code review. Typically,
these include bug fixes, and development or documentation improvements clearly
in line with accepted designs. It may in some rare cases extend to exploring
experimental or prototype directions whose design is under active consideration.

## Who should review?

Anyone should feel free to review Carbon changes. Even providing small or
partial review can be a good way to start contributing to Carbon. Contributors
with specific domain expertise or familiarity should also try to provide review
on changes touching relevant parts of the project.

Additionally, at least one _code owner_ of any file changed needs to review that
change. The code owners and what files they are responsible for are defined
using the
[`CODEOWNERS`](https://help.github.com/en/github/creating-cloning-and-archiving-repositories/about-code-owners#codeowners-syntax)
file in the root of the repository. Pull requests will automatically request
reviewers based on this file and enforce that these reviews take place.

While we do encourage people interested in contributing to Carbon by reviewing
changes to do so, we also suggest not overloading a single review. It can be
daunting for the author of a change to get feedback from a large number of
reviewers, and so we suggest keeping the number of reviewers reasonably small.

## GitHub pull request mechanics

Carbon uses GitHub pull requests for code review, and we recommend some
mechanical best practices to most effectively navigate them.

- When reviewing, don't just comment in the pull request conversation.
  - Go te the `Files Changed` tab
  - Make any in-file comments needed, but add them to a pending review rather
    than sending them directly.
  - Finish the review and add any top-level review comments there.
- If you are a code owner who will be providing approval for the change, then
  make sure to mark a review as requesting changes when you want the author to
  begin addressing your comment. Only use the "comment" review state if you are
  still in the process of reviewing and don't expect the author to begin working
  on further changes.
- Don't reply to in-file comment threads in the conversation view, or with
  direct single reply comments.
  - Add all replies to in-file comment threads using the `Files Changed` tab and
    by adding each reply to a new review, and posting them as a batch when done.
  - You can get to the appropriate `Files Changed` tab by clicking on the change
    listed in the conversation view with the incoming set of in-file comments.

## Resolving an impasse or conflict

At some point, a review may reach an impasse or a genuine conflict. While our
goal is always to resolve these by building consensus in review, it may not be
possible. Both the author and any reviewers should be careful to recognize when
this point arrives and address it directly. Continuing the review is unlikely to
be productive and has a high risk of becoming acrimonious or worse.

There are two techniques to use to resolve these situations that should be tried
early on:

1. Bring another person into the review to help address the specific issue.
   Typically they should at least be a code owner, and may usefully be a member
   of the [core team](groups.md#core-team).

2. Ask the specific question in a broader forum (Discord Chat, Discourse Forums,
   or during a meeting) in order to get a broad set of perspectives on a
   particular area or issue.

The goal of these steps isn't to override the author or the reviewer but to get
more perspectives and voices involved. Often this will clarify the issue and
tradeoff and provide a simple resolution that all parties are happy with.
However, in some cases, the underlying conflict isn't actually addressed. While
there is a desire to generally bias towards the direction of the code owners
during reviews, they should _not_ turn into a voting process. The reason for
proceeding in a specific direction should always be explained sufficiently that
all parties on the review are comfortable with that direction.

Fundamentally, both reviewers and the author need to agree on the direction to
move forward. If reaching that agreement proves impossible, the review should be
[escalated](#escalation).

Once the impasse or conflict is addressed, it is _essential_ to commit to that
direction. It can be especially difficult for the author to accept a direction
that they initially disagree with and make changes to their code as a result.
However, refusing to make the necessary changes or slowing down progress isn't
an acceptable way to operate in the Carbon project. An essential skill is the
ability to
[disagree and commit](https://en.wikipedia.org/wiki/Disagree_and_commit).

## Escalation

At the request of any member of the [core team](evolution.md#core-team) or to
resolve any fundamental impasse in a review, the change should move to the
formal [evolution process](evolution.md#evolution-process). Ultimately, the
Carbon project [governance](evolution.md#governance-structure) structure is
always available as an escalation path.

Before escalating an impasse or conflict in code review, try asking another
reviewer to help resolve the issue or bridge any communication gaps. Consider
scheduling a quick video chat to discuss and better understand each others'
concerns and position.

## Code author guide

The goal of an author should be to ensure their change improves the overall
code, repository, and/or project. Within the context of code review, the goal is
to get a reviewer to validate that the change succeeds at this goal. That
involves finding an effective reviewer given the particular nature of the
change, helping them understand the change fully, and addressing any feedback
they provide.

### Write good change descriptions

The change description in the pull request is the first thing your reviewers
will see. This sets the context for the entire review, and is very important.

#### First line

The first line of a commit, or the subject of the pull request, should be a
short summary of specifically what is being done by that change. It should be a
complete sentence, written as though it was an order. Try to keep it short,
focused, and to the point.

#### Body

The description body needs to cover several important aspects of the change to
provide context for the reviewer:

- The problem being solved by the change
- Why the approach taken is the best one
- Any issues, concerns, or shortcomings of the approach
- Any alternatives considered or attempted
- Relevant supporting data such as examples or benchmarks

Try to anticipate both what information the reviewer of your change will need to
have in order to be effective. Also consider what information someone else will
need a year in the future when doing archaeology on the codebase and they come
across your change without any context.

### Make small changes

Small changes have many benefits:

- Faster review
- More thorough review
- Easier to merge
- Easier to revert if needed

The ideal size of a change is as small as possible while it remains
self-contained. It should address _just one thing_. Often, this results in a
change only addressing _part_ of a feature rather than the whole thing at once.
This makes work more incremental, letting the reviewer understand it piece by
piece. It can also make it much easier to critically evaluate whether each part
of a feature is adequately tested by showing it in isolation.

That said, a change should not be so small that its implications cannot easily
be understood. It is fine to provide the reviewer context or a framework of a
series of changes so they understand the big picture, but that will only go so
far. It is still possible to shrink a change so much that it becomes nonsensical
in isolation.

Consider using a set of stacked changes when necessary to split apart larger
work into small changes that are easier to review.

> TODO: link to the stacked pull request documentation when available.

### Responding to review comments

Many comments have easy and simple responses. The easiest is **"Done"**. When
the comment is a concrete suggestion that makes sense and you implement it, you
can simply let the reviewer know their suggestion has been incorporated. If the
_way_ you implemented the suggestion might need clarification, add that as well.
This may be if it required some tweaks or if you've applied the suggestion in
more places.

When a suggestion from the reviewer is explicitly optional, you may also have a
simple response that you're not going to make the change. This is totally fine
-- if it weren't, the reviewer shouldn't have listed it as optional -- but it
may be helpful to explain your reasoning to the reviewer so they understand
better why the optional suggestion didn't make sense to you.

Sometimes optional comments center around slight differences or preferences
around the code. Consider that the reviewer may be a better proxy for future
readers than you are. If the suggestion is no better or no worse than your
original code, consider adopting it as it may make the code easier to read for
your reviewer at least. But if you feel the current choice is _better_ (even if
minorly), stand up for yourself and keep it. The reviewer can always push for a
change and justify it if needed.

#### Responding to questions or confusion

Some comments in code review will be questions or confusion as the reviewer
tries to understand the code in question or why a particular approach was used.
**Don't assume that questions are a request for a change.** Reviewers should be
explicit if they think a change is needed rather than merely asking questions.
You should assume a question or confusion is just that -- something which needs
to be clarified.

When addressing a question or confusion, you should focus your response on
_changing the code_ in question to provide the necessary comments or clarity.
The reviewer is unlikely to be the last person to have these questions or to
become confused. You should use this as signal for what to clarify. Once done,
the review response should typically focus on verifying that the clarifications
made in the code are sufficient for the reviewer.

#### Understand the feedback in the comments

At times, review comments may be confusing or frustrating for you. While this is
something we always want reviewers to minimize, it will still happen at some
times and to some degree. When this happens, it helps to remember that the goal
of the review is to ensure the change results in the project improving over
time.

If the review comment doesn't make sense, ask the reviewer to help you
understand the feedback better. If it isn't constructive or doesn't seem to
provide any meaningful path forward, ask the reviewer to provide this. Making
comments both clear and constructive are part of the reviewers'
responsibilities. And especially if the reviewer comments are unkind, ad-hominem
attacks, unwelcoming, angry, or otherwise violating our community's
[code of conduct](/CODE_OF_CONDUCT.md), this is **_unacceptable_** and should be
dealt with directly. You can always reach out to the
[conduct team](/CODE_OF_CONDUCT.md#conduct-team) for help addressing these
issues.

Once there is a clear and effectively communicated comment that you understand,
it may still feel wrong or like it is unnecessarily blocking your progress. It
is important to try to step back in this situation and, no matter how ceratin
you are, genuinely consider whether there is valuable feedback. You should be
asking yourself whether the reviewer might be correct, potentially in an
unexpected or surprising way. If you can't decide this definitively, you may
need to work to get a deeper understanding.

If you are confident that the reviewer's comment is incorrect, that is _OK_. The
reviewer is also only human and is certain to make mistakes and miss things. The
response needs to try to explain what it is that leads you to be confident in
your assessment. Lay out the information you have and how you are reasoning
about the issue to arrive at the conclusion. Try not to make assumptions about
what the reviewer knows or why they made the comment. Instead, focus on
surfacing explicitly your perspective on the issue.

These part of a review will often be a discussion and may need to iterate a few
times. That isn't intrinsically bad, but try to make sure that it doesn't result
in reiterating positions or repeating things. Make sure the discussion is
_progressing_ towards deeper understanding and recognize when you reach an
impasse or conflict and shift strategy to
[resolve that](#resolving-an-impasse-or-conflict). It is also useful to avoid
long delays between these iterations. Try hopping into a real-time chat or a
video chat where useful to avoid too mayn multi-hour (or potentially day) round
trips as that can delay reviews significantly.

## Code reviewer guide

The specific goal for a particular review should always be to ensure that the
overal health of the code, repository, and/or project improves over time. This
requires that contributions _make progress_ -- otherwise nothing can improve.
However, the review should ensure that quality of changes does not cause the
health of the project to decrease over time.

The primary responsibility for ensuring that code review remains constructive,
productive, and helpful resides in the _reviewer_. As a reviewer, you are in a
position of power and asked to critique the authors hard work. With this power
comes responsibliity for conducting the review well.

### How quickly should you respond to a review request?

Try to respond to code review requests as soon as you can without interrupting a
focused task. At the latest, the next day you are working on the project. Note
that the review isn't expected to necessarily be complete after a single review.
It is more valuable to give reasonably quick but partial feedback than to delay
feedback in order to complete it.

Large changes are especially important to give incremental feedback on in order
to do so in a timely fashion. One of the first things to consider with large
changes is whether it can be split apart into smaller changes that are easier to
review promptly.

However, this timeliness guidedance doesn't apply to higher-level review such as
the [evolution process](evolution.md). Evaluating those types of proposals will
often require a larger time investment and have their own timelines spelled out
in the process. Here, we are talking about simply reviewing changes themselves
orthogonally to any evolutionary discussion and evaluation.

### What should be covered by a review?

Things to consider and evaluate when reviewing changes:

- Is the code well designed?
- Is the resulting functionality good for the users of the code?
- Are any user interface or user experience changes good for the users?
- Is any parallel or concurrent programming done safely?
- Can the code be simplified? Is there unnecessary complexity?
- Are things being implemented that aren't yet needed and only _might_ be needed
  in the future?
- Does it have appropriate unit tests?
- Do any integration tests need to be extended or added?
- Do any fuzz tests need to be extended or added?
- Are any tests well designed to be both thorough but also maintainable over
  time?
- Are the names used in the code clear?
- Are all important or non-obvious aspects of the code well commented? Do the
  comments focus on _why_ instead of _what_?
- Is there appropriate high level documentation for the change?
- Does the change adhere to all relevant style guides?
- Is the change consistent with other parts of the project?

### Writing review comments

These are general guidelines for writing effective code review comments:

- **Be kind.** Detailed review, especially in an open source project, can be
  stressful and difficult for the author. As a reviewer, part of the job is to
  ensure the review experience ends up positive and constructive for the author.
- **Stay constructive.** Don't comment negatively about the change. Suggest
  specific ways to improve the change. If you need to explain why an improvement
  is necessary, focus on objective ways the improvement helps and avoid both
  subjective assessments and anchoring on the current state.
- **Explain why.** It is important for the author to understand not merely the
  mechanical suggested change but what motivates it and why it matters. This may
  help clear up misunderstandings, help the suggestion be understood and applied
  more effectively, and allow internalizing improvements for future
  contributions.

Keep in mind that the goal is to improve the overall health of the code,
repository, and/or project over time. Sometimes, there will be pushback on
review comments. Consider carefully if the author is correct -- they may be
closer to the technical issues than you are and may have important insight. Also
consider whether the suggestion is necessary to achieve the overall goal. If the
suggestion isn't critical to make the change an overall improvement, it may be
fine for it to move forward as-is.

### Approving the change

Be explicit and unambiguous at the end of your review. When approving a change,
say so. The acronym "LGTM" for "Looks Good To Me" is often used, but the key is
to be explicit and clear. If you don't feel like you're in a position to approve
the change and are simply helping out with review feedback, make that explicit
as well. For example, say that "my comments are addressed, but leaving the final
review to others" to clearly indicate that you're happy but are deferring the
decision to others. If you are a code owner and deferring to someone else, it is
essential to suggest specific other reviewers. Otherwise we risk all the code
owners assuming another is going to approve the change.

An important technique to make progress, especially with different working hours
and timezones, is to approve changes even with outstanding comments. For
example, if the comments you have are straight forward and have unambiguous
fixes or suggested edits, you should give an LGTM with those comments addressed.
The author can always come back to you if they have questions, and we can always
revert changes if the resolution for some reason diverges wildly from your
expectations.
