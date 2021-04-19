# Extending `while` loops with exit-conditional statements

<!--
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

[Pull request](https://github.com/carbon-language/carbon-lang/pull/####)

<!-- toc -->

## Table of contents

-   [TODO: Initial proposal setup](#todo-initial-proposal-setup)
-   [Problem](#problem)
-   [Background](#background)
-   [Proposal](#proposal)
-   [Details](#details)
-   [Rationale based on Carbon's goals](#rationale-based-on-carbons-goals)
-   [Alternatives considered](#alternatives-considered)

<!-- tocstop -->

## TODO: Initial proposal setup

> TIP: Run `./scripts/new_proposal.py "TITLE"` to do new proposal setup.

1. Copy this template to `new.md`, and create a commit.
2. Create a GitHub pull request, to get a pull request number.
    - Add the `proposal` and `WIP` labels to the pull request.
3. Rename `new.md` to `/proposals/p####.md`, where `####` should be the pull
   request number.
4. Update the title of the proposal (the `TODO` on line 1).
5. Update the link to the pull request (the `####` on line 11).
6. Delete this section.

TODOs indicate where content should be updated for a proposal. See
[Carbon Governance and Evolution](/docs/project/evolution.md) for more details.

## Problem

The traditional iteration loops `while` and `for` suffer from a minor
short-coming: they discard the outcome of the continuation condition check. Some
languages (such as Python) solve this by providing an optional `else`
statement. Backporting that approach to C or C++ is hard, because the obvious
`while ... else` syntax already has a (different) meaning in those languages,
but in a new language such as Carbon we can provide this functionality from the
start.

Non-problem: We are not proposing to provide syntactic sugar for every
conceivable control flow. The proposal addresses only one key issue that is both
realistic (i.e. it comes up somewhat regularly in real code) and for which
workarounds are unappealing.

## Background

Many structured programming languages have an iteration control flow structure
like the `while` loop. In pseudo-code, `while (condition) { BODY; }` can be
thought of as the following sequence of jumps:

```
start:
  if (condition) {
    BODY;
    goto start;
  }
end:
```

So far, so simple. However, a loop can also exit via something like a `break`
statement inside the body, which acts like `goto end;`. Consequently, the `end:`
label can be reached in two different ways (by flowing into it and by `goto`),
which are known to the execution, but the distinction is not available to the
programmer.

Following Python's design, we propose an optional subsequent statement called
`else` that is the `else` arm of the notional `if (condition)` statement. That
is, `while (condition) { BODY; } else { ELSE-BODY; }` becomes:

```
start:
  if (condition) {
    BODY;
    goto start;
  } else {
    ELSE-BODY;
  }
end:
```

## Proposal

We propose to make the syntax `while (condition) statement else statement`
valid, with the semantics described above. Consequently, `for`-loop syntax
`for (...) statement else statement` should also become available.

## Details

The proposal should be implementable with a few simple changes to the language
grammar and semantics, and the implementation should be straight-forward (since
all the constituent pieces should already be available and this corresponds
essentially to a frontend-only construction).

## Rationale based on Carbon's goals

I consider this proposal to be motivated by Carbon's goal to show what a better
C++ could look like. This feature could have been an original part of C and C++,
but it was missed then; now that we have a chance, we might as well provide
it. It will be one of those details where Carbon is both familiar to C++ users,
and also a pleasant improvement over it.

## Alternatives considered

There has been a lot of discussion on this particular part of the language in
C++ for a long time. Given that the straight-up `while ... else` syntax is not
available, multiple proposals have gone much further and asked for further
optional blocks to be called from a `break`, and from a `continue`, and there
have also been multiple proposals for various kinds of "labelled breaks" where a
`break` can exit a different containing loop other than the immediate one.

I consider such an extended "control flow zoo" unnecessary: Their use cases are
increasingly narrow, and the workaround is often straight-forward and not overly
burdensome. In detail, consider that an optional landing block for, say,
`break`, could currently be written by moving the optional code to just before
the `break` statement. Only when there are multiple `breaks` would this be an
issue at all, and even then, it may be possible to refactor the code
reasonably. Assuming style guidance that loops should generally be short and
simple, I expect that only very little code would really benefit from such
extended optional control structures.

In other worlds, I consider "on break" and "on continue" blocks pure syntactic
sugar with very limited use and only minor potential for improvement, and it
would require significant syntactic novelty. By contrast, the simple `while
... else` block fits naturally into existing syntax and exposes information that
simply would not be available to the programmer otherwise.

TODO: add list of C++ proposal papers

TODO: add survey of other languages
