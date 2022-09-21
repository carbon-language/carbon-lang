# Code review

<!--
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

<!-- toc -->

## Table of contents

-   [High level goals of code review](#high-level-goals-of-code-review)
-   [What requires review?](#what-requires-review)
-   [Who should review?](#who-should-review)
-   [GitHub pull request mechanics](#github-pull-request-mechanics)
-   [Code author guide](#code-author-guide)
    -   [Write good change descriptions](#write-good-change-descriptions)
        -   [First line](#first-line)
        -   [Body](#body)
    -   [Make small changes](#make-small-changes)
    -   [Responding to review comments](#responding-to-review-comments)
        -   [Responding to questions or confusion](#responding-to-questions-or-confusion)
        -   [Understand the feedback in the comments](#understand-the-feedback-in-the-comments)
-   [Code reviewer guide](#code-reviewer-guide)
    -   [How quickly should you respond to a review request?](#how-quickly-should-you-respond-to-a-review-request)
    -   [What should be covered by a review?](#what-should-be-covered-by-a-review)
    -   [Writing review comments](#writing-review-comments)
    -   [Approving the change](#approving-the-change)
-   [Merging pull requests](#merging-pull-requests)
    -   [Merge commit descriptions](#merge-commit-descriptions)
-   [Resolving an impasse or conflict](#resolving-an-impasse-or-conflict)
-   [Escalation](#escalation)

<!-- tocstop -->

## High level goals of code review

Code review serves several goals in the Carbon project. It directly improves the
correctness, clarity, and consistency of contributions, including both code and
documentation. These improvements range from the high-level functionality down
through the design and implementation details. It also promotes team ownership
and spreads knowledge across the team.

More detailed discussions can be found in chapter 9 "Code Review" of the book
_[Software Engineering at Google](https://www.amazon.com/Software-Engineering-Google-Lessons-Programming/dp/1492082791)_
and chapter 21 "Collaborative Construction" in
_[Code Complete: A Practical Handbook of Software Construction](https://www.amazon.com/Code-Complete-Practical-Handbook-Construction/dp/0735619670/)_.
However, these details aren't essential to understanding code review and how it
works in the Carbon project. All of the important details are provided in the
project documentation.

## What requires review?

Every change to Carbon's repositories requires code review. Even formal
[evolution decisions](evolution.md) which have been approved should have their
specific changes to the repository reviewed.

Many changes to Carbon repositories may _only_ require code review. Typically,
these include bug fixes, and development or documentation improvements clearly
in line with accepted designs. It may in some rare cases extend to exploring
experimental or prototype directions whose design is under active consideration.

The term "code review" in the Carbon project is not only about "code". We expect
changes to any file to be reviewed, including documentation and any other
material stored in the repository.

## Who should review?

Everyone should feel free to review Carbon changes. Even providing small or
partial review can be a good way to start contributing to Carbon. Contributors
with specific domain expertise or familiarity should also try to provide review
on changes touching relevant parts of the project.

Additionally, at least one developer with commit access must review each change.
In Carbon, developers will focus on particular areas, loosely broken down as:

-   [Carbon leads](groups.md#carbon-leads): [proposals](/proposals/) and other
    important project documents, including the [Main README](/README.md),
    [Code of Conduct](/CODE_OF_CONDUCT.md), [license](/LICENSE), and
    [goals](goals.md).

-   [Implementation team](groups.md#implementation-team): general changes.

    -   We split out auto-assignment by explorer, toolchain, and other files
        (including documentation).

[Auto-assignment](/.github/workflows/assign_prs.yaml) will help find owners, but
won't always be perfect -- developers may take a PR they weren't auto-assigned
in order to help review go quickly. Contributors can also request multiple
reviewers, but it can be daunting to get feedback from a large number of
reviewers, so we suggest keeping the number of reviewers reasonably small.

Any reviews that explicitly request changes should be addressed, either with the
changes or an explanation of why not, before a pull request is merged. Further,
any owners who have requested changes should explicitly confirm they're happy
with the resolution before the change is merged.

When a team gives an affirm decision on an [evolution proposal](evolution.md),
each team member should explicitly note any of their comments on the pull
request that, while not blocking the _decision_, still need to be resolved as
part of code review prior to it being merged. These might, for example, be
trivial or minor wording tweaks or improvements. Otherwise, the decision is
assumed to mean the prior review comments from members of that team are
addressed; the author is free to merge once the pull request is approved,
possibly with a code review separate from the proposal's review.

## GitHub pull request mechanics

Carbon uses GitHub pull requests for code review, and we recommend some
mechanical best practices to most effectively navigate them.

-   Be aware that the main thread of pull request doesn't support threaded
    discussions or "resolving" a comment.
    -   If either of those would be useful, you'll probably want to comment on a
        file.
    -   You can quote comments in the main conversation thread in a reply by
        clicking the three-dot menu on the original comment and selecting "Quote
        reply".
-   If you will want to comment on files, don't comment in the pull request
    conversation.
    -   Always go to the `Files Changed` tab.
    -   Make any in-file comments needed, but add them to a pending review
        rather than sending them directly.
    -   Finish the review and add any top-level review comments there.
-   If you are an owner who will be providing approval for the change, then make
    sure to mark a review as requesting changes when you want the author to
    begin addressing your comment. Only use the "comment" review state if you
    are still in the process of reviewing and don't expect the author to begin
    working on further changes.
    -   If you are not an owner asked to approve, use the difference between a
        comment and requesting a change to help the author know whether to
        circle back with you before landing the pull request if the relevant
        owner(s) approve it.
-   Don't reply to in-file comment threads in the conversation view, or with
    direct single reply comments.
    -   Add all replies to in-file comment threads using the `Files Changed` tab
        and by adding each reply to a new review, and posting them as a batch
        when done.
    -   You can get to the appropriate `Files Changed` tab by clicking on the
        change listed in the conversation view with the incoming set of in-file
        comments.
    -   This flow ensures an explicit update in the overall pull request that
        can help both the author and other reviewers note that new replies have
        arrived.
-   Don't reply to an in-file comment and then mark it as resolved. No one will
    see your reply as the thread will be hidden immediately when marked as
    resolved.
    -   Generally, the person who started the comment thread should mark it as
        resolved when their comments are sufficiently addressed. If another
        reviewer is also on the thread and should also agree, just state that
        you're happy and the last reviewer can mark it resolved.
    -   Trivially resolved threads can just be marked as "resolved" without
        further update. Examples: a suggested change that has been successfully
        applied, or a thread where the relevant reviewers have clearly indicated
        they're happy.

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

The description body may need to explain several important aspects of the change
to provide context for the reviewer when it isn't obvious from the change
itself:

-   The problem being solved by the change.
-   Why the approach taken is the best one.
-   Any issues, concerns, or shortcomings of the approach.
-   Any alternatives considered or attempted.
-   Relevant supporting data such as examples or benchmarks.

Try to anticipate what information the reviewer of your change will need to have
in order to be effective. Also consider what information someone else will need
a year in the future when doing archaeology on the codebase and they come across
your change without any context.

### Make small changes

Small changes have many benefits:

-   Faster review.
-   More thorough review.
-   Easier to merge.
-   Easier to revert if needed.

The ideal size of a change is as small as possible while it remains
self-contained. It should address only _one thing_. Often, this results in a
change only addressing _part_ of a feature rather than the whole thing at once.
This makes work more incremental, letting the reviewer understand it piece by
piece. It can also make it much easier to critically evaluate whether each part
of a feature is adequately tested by showing it in isolation.

That said, a change should not be so small that its implications cannot easily
be understood. It is fine to provide the reviewer context or a framework of a
series of changes so they understand the big picture, but that will only go so
far. It is still possible to shrink a change so much that it becomes nonsensical
in isolation. For example, a change without appropriate tests is not
self-contained.

### Responding to review comments

Many comments have easy and simple responses. The easiest is **"Done"**. When
the comment is a concrete suggestion that makes sense and you implement it, you
can simply let the reviewer know their suggestion has been incorporated. If the
_way_ you implemented the suggestion might need clarification, add that as well.
For example, consider mentioning tweaks to the suggestion or when the suggestion
was applied in more places.

When a suggestion from the reviewer is explicitly optional, you may also have a
simple response that you're not going to make the change. This is totally fine
-- if it weren't, the reviewer shouldn't have listed it as optional -- but it
may be helpful to explain your reasoning to the reviewer so they understand
better why the optional suggestion didn't make sense to you.

Sometimes comments, even optional ones, center around slight differences or
preferences around the code. Consider that the reviewer may be a good proxy for
future readers. If the suggestion is essentially equivalent to your original
code, consider adopting it as it may make the code easier to read for others.
But if you feel the current choice is _better_, even if only slightly, stand up
for yourself and keep it. The reviewer can always push for a change and justify
it if needed.

For non-optional comments, this section provides several suggestions on how best
to make progress. If none of these work, you may need to
[resolve an impasse or conflict](#resolving-an-impasse-or-conflict).

In response to suggestions, update the files in the pull request in new commits.
Rebasing, squashing, or force-pushing commits can break GitHub's comment
associations, and it makes it harder to determine what's changed since the last
review. With regular pushes, GitHub can show individual deltas, giving
additional flexibility to the reviewer. We squash pull requests when we merge
them, so the end result is the same.

It is good to reply to every comment so that the reviewer knows you saw them.
Best practice is to send the reply to all of the comments at once, after the
files in the pull request have been updated. If there are a lot of replies, it
can be helpful to include a message saying whether the pull request is now ready
for another round of review, or press the "Re-request review" button to the
right of the reviewer's name:
![The re-request review button on GitHub](https://user-images.githubusercontent.com/711534/189789784-fcf18d1b-137b-48ba-959a-0d05cee36c2d.png)

#### Responding to questions or confusion

Some comments in code review will be questions or confusion as the reviewer
tries to understand the code in question or why a particular approach was used.
Don't assume that questions are a request for a change. Reviewers should be
explicit if they think a change is needed rather than merely asking questions.
You should assume a question or confusion is something which only needs to be
clarified.

However, when responding to a question or confusion, consider making changes to
improve clarity in addition to responding within the review, such as by adding
comments or changing code structure. The reviewer may not be the last person to
need more clarity, and you should use their comments as a signal for
improvement. Once done, the review response should typically focus on verifying
that the clarifications made in the code are sufficient for the reviewer.

#### Understand the feedback in the comments

At times, review comments may be confusing or frustrating for you. While this is
something we always want reviewers to minimize, it will still happen at some
times and to some degree. It helps to remember that the goal of the review is to
ensure the change results in the project improving over time.

If the review comment doesn't make sense, ask the reviewer to help you
understand the feedback better. If it isn't constructive or doesn't seem to
provide any meaningful path forward, ask the reviewer to provide this. Making
comments both clear and constructive are part of the reviewers'
responsibilities.

Once there is a clear and effectively communicated comment that you understand,
it may still feel wrong or like it is unnecessarily blocking your progress. It
is important to try to step back in this situation and, no matter how certain
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

These parts of a review will often be a discussion and may need to iterate a few
times. That isn't intrinsically bad, but try to make sure that it doesn't result
in reiterating positions or repeating things. Make sure the discussion is
_progressing_ towards deeper understanding and recognize when you reach an
impasse or conflict and shift strategy to
[resolve that](#resolving-an-impasse-or-conflict). It is also useful to avoid
long delays between these iterations. Consider discussing over Discord chat or
scheduling a quick video chat on the specific issue. This can avoid multi-hour
-- or multi-day -- round trips.

## Code reviewer guide

The specific goal for a particular review should always be to ensure that the
overall health of the code, repository, and/or project improves over time. This
requires that contributions _make progress_ -- otherwise, nothing can improve.
However, the review should ensure that quality of changes does not cause the
health of the project to decrease over time.

The primary responsibility for ensuring that code review remains constructive,
productive, and helpful resides in the _reviewer_. As a reviewer, you are in a
position of power and asked to critique the authors hard work. With this power
comes responsibility for conducting the review well.

### How quickly should you respond to a review request?

Try to respond to code review requests as soon as you can without interrupting a
focused task. At the latest, the next day you are working on the project. Note
that the review isn't expected to necessarily be complete after a single review.
It is more valuable to give reasonably quick but partial feedback than to delay
feedback in order to complete it. If leaving partial feedback, make it clear to
the author which parts are covered and which you haven't gotten to yet.

Large changes are especially important to give incremental feedback on in order
to do so in a timely fashion. One of the first things to consider with large
changes is whether it can be split apart into smaller changes that are easier to
review promptly.

This timeliness guidance doesn't apply to the higher-level
[evolution process](evolution.md) reviews. Evaluating those proposals will often
require a larger time investment and have their own timelines spelled out in the
process. Here, we are talking about simply reviewing changes themselves
orthogonally to any evolutionary discussion and evaluation.

### What should be covered by a review?

Things to consider and evaluate when reviewing changes:

-   Is the code well designed?
    -   Is the resulting functionality, including its interface, good for the
        users of the code?
    -   Does the resulting design facilitate long-term maintenance?
    -   Can the code be simplified? Is there unnecessary complexity?
    -   Are things being implemented that aren't yet needed and only _might_ be
        needed in the future?
-   Is the code free of bugs and well tested?
    -   Is memory safely managed?
    -   Is any parallel or concurrent programming done safely?
    -   Do unit tests cover relevant behaviors and edge cases?
    -   Do any integration tests need to be extended or added?
    -   Do any fuzz tests need to be extended or added?
    -   Are any tests well designed to be both thorough but also maintainable
        over time?
-   Is the code easy to read?
    -   Are the names used in the code clear?
    -   Are all important or non-obvious aspects of the code well commented? Do
        the comments focus on _why_ instead of _what_?
    -   Is there appropriate high level documentation for the change?
    -   Does the change adhere to all relevant style guides?
    -   Is the change consistent with other parts of the project?

### Writing review comments

These are general guidelines for writing effective code review comments:

-   **Be kind.** Detailed review, especially in an open source project, can be
    stressful and difficult for the author. As a reviewer, part of the job is to
    ensure the review experience ends up positive and constructive for the
    author.
-   **Stay constructive.** Focus your comments on suggesting specific ways to
    improve the change. If you need to explain why an improvement is necessary,
    focus on objective ways the improvement helps and avoid both subjective
    assessments and anchoring on problems with the current state.
-   **Explain why.** It is important for the author to understand not merely the
    mechanical suggested change but what motivates it and why it matters. This
    may help clear up misunderstandings, help the suggestion be understood and
    applied more effectively, and allow internalizing improvements for future
    contributions.
-   **Provide a path forward.** The author needs to understand what they will
    need to do to respond to your comments. For example, always provide
    alternatives when commenting that the current approach won't work.

Keep in mind that the goal is to improve the overall health of the code,
repository, and/or project over time. Sometimes, there will be pushback on
review comments. Consider carefully if the author is correct -- they may be
closer to the technical issues than you are and may have important insight. Also
consider whether the suggestion is necessary to achieve the overall goal. If the
suggestion isn't critical to make the change an overall improvement, it may be
fine for it to move forward as-is.

As with all communication in the Carbon project, it is critical that your
comments are not unkind, unwelcoming, angry, ad-hominem attacks, or otherwise
violating our community's [code of conduct](/CODE_OF_CONDUCT.md).

### Approving the change

Be explicit and unambiguous at the end of your review. Select "Approve" when
submitting the review to mark this in GitHub. You can always include a message,
often "LGTM" or "Looks Good To Me" is often used. If you don't feel like you're
in a position to approve the change and are simply helping out with review
feedback, make that explicit as well. You should set the review to a "Comment"
in GitHub, but also state this explicitly in the message since this is the
default and doesn't indicate that your feedback _is_ addressed. For example, say
that "my comments are addressed, but leaving the final review to others" to
clearly indicate that you're happy but are deferring the decision to others. If
you are an owner and deferring to someone else, it is essential to suggest
specific other reviewers. Otherwise, we risk all the owners assuming another is
going to approve the change.

An important technique to make progress, especially with different working hours
and timezones, is to approve changes even with outstanding comments. For
example, if the comments you have are straightforward and have unambiguous fixes
or suggested edits, you should give an LGTM with those comments addressed. The
author can always come back to you if they have questions, and we can always
revert changes if the resolution for some reason diverges wildly from your
expectations.

## Merging pull requests

Pull requests are ready to be merged when reviewers have indicated they're happy
(for example, "LGTM" or "Looks good to me") or have approved the pull request.
While all merges require at least one approval, a reviewer might approve before
others are finished reviewing; all reviewers should be given time to comment to
ensure there's a consensus.

Either the author or reviewer may merge and resolve conflicts. The author may
indicate they want to merge by informing the reviewer and adding the
`DO NOT MERGE` label. The reviewer is encouraged to coordinate with the author
about merge timing if there are concerns about breaks. In either case, the
developer doing the merge is expected to be available to help address
post-commit issues, whether through a fix-forward or a rollback.

### Merge commit descriptions

When squashing and merging, GitHub tries to generate a description, but it's
recommended to use the first comment on the pull request review for the squashed
commit description. Authors should keep it up-to-date so that reviewers can
merge when the change is ready. Reviewers shouldn't edit or rewrite this message
themselves, and instead ask the author make those changes (possibly with
suggestions) just like other parts of the code review. It's important that the
commit message is one the author is comfortable with when merged.

When suggested edits have been merged into a pull request, GitHub will append a
`Co-authored-by:` line to its default proposed commit message for each reviewer
who suggested edits that were applied. These lines should be retained and
appended to the message from the initial comment.

## Resolving an impasse or conflict

At some point, a review may reach an impasse or a genuine conflict. While our
goal is always to resolve these by building consensus in review, it may not be
possible. Both the author and any reviewers should be careful to recognize when
this point arrives and address it directly. Continuing the review is unlikely to
be productive and has a high risk of becoming acrimonious or worse.

There are two techniques to use to resolve these situations that should be tried
early on:

1. Bring another person into the review to help address the specific issue.
   Typically they should at least be an owner, and may usefully be a
   [Carbon lead](groups.md#carbon-leads).

2. Ask the specific question in a broader forum, such as Discord, in order to
   get a broad set of perspectives on a particular area or issue.

The goal of these steps isn't to override the author or the reviewer, but to get
more perspectives and voices involved. Often this will clarify the issue and its
trade-offs, and provide a simple resolution that all parties are happy with.
However, in some cases, the underlying conflict isn't actually addressed. While
there is a desire to generally bias towards the direction of the owners during
reviews, reviews should _not_ turn into a voting process. The reason for
proceeding in a specific direction should always be explained sufficiently that
all parties on the review are satisfied by the explanation and don't feel the
need to escalate.

Fundamentally, both reviewers and the author need to agree on the direction to
move forward. If reaching that agreement proves impossible, the review should be
[escalated](#escalation). If you feel like an escalation is needed in a review,
be explicit and clear in requesting it. There is nothing bad about going through
this process, but it should only occur when needed and so it helps to be very
clear.

Once the impasse or conflict is addressed, it is _essential_ to commit to that
direction. It can be especially difficult for the author to accept a direction
that they initially disagree with and make changes to their code as a result. An
essential skill is the ability to
[disagree and commit](https://en.wikipedia.org/wiki/Disagree_and_commit).

## Escalation

At the explicit request of any [Carbon lead](evolution.md#carbon-leads-1) or to
resolve any fundamental impasse in a review, the change should move to a formal
[proposal](evolution.md#proposals). Ultimately, the Carbon project
[governance](evolution.md#governance-structure) structure is always available as
an escalation path.

Before escalating an impasse or conflict in code review, try asking another
reviewer to help resolve the issue or bridge any communication gaps. Consider
scheduling a quick video chat to discuss and better understand each other’s
concerns and position.

Note that the formal evolution process is heavyweight and relatively slow. The
expectation is that this is rarely used and only to resolve serious and severe
disagreements. If this becomes a more common problem, lighter weight processes
may be needed to help ensure a reasonable rate of progress.
