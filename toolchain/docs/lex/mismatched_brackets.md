# Mismatched bracket recovery

<!--
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

<!-- toc -->

## Table of contents

-   [Overview](#overview)
-   [Repairs](#repairs)
-   [Reducing the problem](#reducing-the-problem)
    -   [Token abstraction](#token-abstraction)
    -   [Trusting matched pairs](#trusting-matched-pairs)
    -   [Regions](#regions)
-   [Searching](#searching)
-   [The cost model](#the-cost-model)
-   [Cues](#cues)
-   [Ambiguity](#ambiguity)
-   [Reporting and applying repairs](#reporting-and-applying-repairs)
-   [Evaluation](#evaluation)
-   [Limitations](#limitations)

<!-- tocstop -->

## Overview

Everything downstream of lexing assumes the token stream is well bracketed: each
`(`, `[`, and `{` has a matching closer, and the two cross-reference each other.
When the source doesn't satisfy that, the lexer repairs the stream before
handing it on, inserting recovery tokens and turning brackets it can't place
into error tokens.

The interesting question is not _whether_ a bracket is unmatched — a stack
matcher answers that — but _where_ the missing one belongs. Recovery has to pick
an insertion point, and that choice matters twice over: it is what the
diagnostic tells the developer, and it decides how much of the rest of the file
is misinterpreted. A `}` missing from the middle of a class definition, closed
at end of file instead of where it belongs, turns every following declaration
into a member of that class.

So we treat recovery as a search: enumerate ways to make a damaged part of the
file well bracketed, price each by how plausible the mistake it implies is, and
take the cheapest. The prices form a cost model calibrated by measurement
against real Carbon code, and the evidence it reasons from is indentation,
structural facts about the language, and formatting conventions.

`FixMismatchedBrackets` is the whole of that: it takes the token sequence and
returns a list of `BracketCorrection`s, each pairing a diagnostic to report with
a fix to apply to the token stream. Nothing else in the lexer knows how the
decision was made.

## Repairs

A repair is built from four moves:

-   Insert a closing bracket directly before a token: the group ended here.
-   Insert an opening bracket directly before a token. A synthetic opener only
    becomes a suggestion once a real closer matches it, since that is the point
    at which we know which closer it explains; a repair that reaches the end of
    the region with a synthetic opener still unmatched is rejected rather than
    inventing an empty pair.
-   Close whatever is still open when the region or the file ends.
-   Replace a bracket with an error token: give up on it. The token stays in the
    stream but carries no bracket structure, so nothing downstream tries to pair
    it.

Giving up is what lets the rest of the model be assertive. It is always
available at a fixed price, so it acts as a ceiling: a repair less plausible
than admitting defeat is not chosen.

## Reducing the problem

### Token abstraction

Recovery does not see tokens. It sees a `MismatchedBracketToken` per token: a
projection onto the properties bracket structure could depend on. That is a
coarse `BracketTokenKind` (the six brackets, `;`, `,`, `.`, statement
introducers, _leaf_ tokens that form a complete primary expression on their own,
three flavors of operator, end of file, and everything else), the line and
indentation, and a few flags — whether a `{` looks like a struct literal,
whether a keyword demands a bracket after it, whether whitespace precedes the
token. Nothing else about the source is available, which keeps the search cheap
and the rules easy to reason about.

### Trusting matched pairs

An ordinary stack match runs first. Most of the pairs it finds are genuine, and
reconsidering them would be both slow and risky, so every pair that passes a
cleanliness test collapses into a single `Item` that the search steps over as a
unit. `Item` is the search's unit of input from here on: one per token, except
where a trusted pair spans several.

Cleanliness is deliberately conservative. A brace pair is trusted when its `}`
lines up with the indentation of the statement that owns the `{` — not with the
column of the `{`, which for a header wrapped across lines is somewhere else
entirely — and when the separators directly inside it suit the brace's flavor: a
`;` directly inside a struct literal, or a `,` directly inside a block, means
the pairing has captured too much. A paren or square pair is suspect when an
unmatched opener of the same kind appears earlier in the same statement, or an
unmatched closer of the same kind later, because either could be the real
partner. Finally, the interior must be clean itself, with nothing inside it
pairing to something outside.

### Regions

The resulting item sequence is cut at top-level declaration boundaries: a
statement introducer at column 1 whose predecessor ended a statement. Each
region is solved independently. That bounds how far one mistake can smear, and
it gives unclosed brackets a natural place to be closed — before the next
declaration, rather than at end of file.

Regions whose remaining loose brackets already balance are skipped without
searching at all. That is the large majority of them, since a region typically
holds pairs that merely failed the cleanliness test, and skipping them is worth
an order of magnitude in running time.

## Searching

`RegionSearch` solves one region. A search state — a `BeamNode` — is a position
in the item sequence plus the stack of brackets open at that point, together
with which closer, if any, was inserted at this position already.

The search runs in layers, one per item. Within a layer, insertion moves are
applied: these don't consume the item, so they stay in the layer and can chain,
which is how several groups get closed at one point. Then every surviving state
advances over the item into the next layer. Both phases prune to a fixed beam
width, keeping the cheapest states.

Two states in a layer whose stacks agree are interchangeable, so they merge. A
merged state retains _every_ cheapest way it was reached rather than one, which
is what makes ambiguity visible at the end.

The search is bounded in each direction: beam width, stack depth, items per
region, and paths enumerated. Exceeding a bound costs quality, never correctness
— the fallback for an oversized region is a naive greedy matcher, and the output
is well bracketed either way. Beam search offers no guarantee of finding the
cheapest repair, but an intended repair is a cheap one by construction, so it
survives pruning.

## The cost model

Costs are ordinal; only their ordering carries meaning. They live in three
tables of `BracketRule`s:

-   `CloserRules` prices inserting a closing bracket before the current token.
-   `OpenerRules` prices inserting an opening bracket before it.
-   `AdvanceRules` prices stepping over the token with the current group still
    open.

The first two are first-match tables, ordered from the most specific cue to the
least, and a row may decline outright, meaning the move is not worth considering
there. The third is additive: every matching row contributes and the penalties
sum. That table is the other half of the same judgment as the first: a penalty
for swallowing a token that has no business inside the current group is what
makes "close the group before it" win.

A lookup buckets on two categorical facts — the context category (which bracket
is innermost, or which one would be inserted, with block and struct braces
distinguished) and the token's kind — and a compile-time index maps each bucket
to the rows that could apply in it, so only a handful are ever tested. Rows
condition further on `Cue`s, as sets of properties that must all hold, must not
hold, must not all hold, or must hold in part.

The model is calibrated on a single principle: the intended repair must be
_strictly_ cheapest. A tie is not resolved but reported, as described below, so
a rule that gets the right answer only about half the time is worse than no rule
at all.

## Cues

-   **Indentation.** A line dedented to or past the indentation of the statement
    that opened a block ends that block. The statement's indentation is what
    counts, not the column of the `{`: a `}` lines up with its `if` even when
    the `{` sits at the end of a wrapped header. A first-on-line `else` is
    treated the same way, by walking back over complete blocks to the branch it
    continues.
-   **Facts about the language.** A `;` cannot appear inside `(` or `[` at all.
    A block `{` cannot be the content of a keyword's parentheses or of a struct
    literal. A binary operator cannot start a group. A leaf cannot directly
    follow a value-ending token, so finding that pair is evidence a bracket is
    missing between them. `if`, `while`, `for`, and `match` require a following
    `(`, and `forall` a following `[`.
-   **Cascades.** Closing a group at a point where an enclosing group is already
    being closed is cheap, because a run of missing closers is usually a single
    mistake.
-   **Formatting conventions.** Formatted Carbon writes no space before `,`,
    `)`, or `.`, and none before a call's `(`. A space in one of those positions
    means the code is unformatted or something was deleted from the gap, and
    either way is weak evidence that a bracket belongs there. Such cues can only
    fire on input that isn't formatted, which is what makes them safe to state
    liberally.
-   **Position in the file.** Closing at the end of a region or of the file has
    a fixed price, set above every precise cue, so that a real cue always wins
    and an unclosed group only runs to the region end when nothing local
    explains it.

## Ambiguity

Once the cheapest cost is known, every repair achieving it is enumerated, up to
a cap. A suggestion present in one cheapest repair but absent from another is
ambiguous: the model has no basis for preferring either placement. Rather than
guess, we discard the suggestion and replace the bracket with an error token.
The developer is told the bracket is unmatched and is not pointed anywhere
misleading, and the parser receives a stream with no invented structure.

This is the reason the tables are ordered and priced as deliberately as they
are. Two competing repairs at distinguishable prices produce one answer; the
same two at equal prices produce none.

## Reporting and applying repairs

Each correction produces an error at the bracket that has no partner, plus a
note at the position the missing bracket would occupy. That position lies
between two tokens rather than at one, so the note is emitted against a source
pointer: immediately past the end of the token the bracket would follow, then
adjusted for how the bracket is conventionally written. A `}` at end of line
moves past the newline and in to its opener's indentation, since it belongs on a
line of its own, and an opening bracket moves past any spaces, since it binds to
what follows.

Insertions are buffered in an `ErrorRecoveryBuffer` and applied in one pass that
renumbers the token stream, after which bracket cross-references are recomputed.
Insertions requested at the same anchor are ordered closers before openers, so
that the result is well nested.

## Evaluation

`mismatched_brackets_eval` measures recovery by corrupting known-good code and
asking whether recovery reconstructs it. It deletes brackets from the Carbon
files in this repository, re-lexes, and compares the suggestions against what it
deleted:

```shell
bazel run -c opt //toolchain/lex:mismatched_brackets_eval -- \
    --trials=4587 --d-values=1 --mode=gapless
```

There are four corruption modes, because how a bracket goes missing changes what
evidence survives:

-   **blank** replaces the bracket with a space, preserving byte offsets. Cheap
    to reason about, but the space it leaves is itself a cue, so this mode
    flatters the formatting rules.
-   **gapless** deletes the bracket and closes up the text, leaving no space
    behind and so no artifact. This is the mode to trust for formatted input.
-   **truncate** cuts the file at a random token, so that everything still open
    is closed at end of file.
-   **truncate-region** deletes from a declaration boundary through the close of
    a pair, modeling code typed into an existing class. Over-extending is
    expensive here: swallowing the following declaration is the failure recovery
    exists to prevent.

Scoring is by structure, not position. A suggestion is matched to the deletion
it restores through the first surviving token it precedes, so closing anywhere
in trailing whitespace, or anywhere within a run of identical brackets, counts
as the same repair, while closing somewhere that changes the structure —
re-pairing a surviving bracket, swallowing the next declaration — does not. Each
trial is then Correct, Partial, None, or Incorrect.

Those outcomes are not weighted equally. An Incorrect suggestion misparses the
file and points the developer at the wrong place, whereas a trial with no
suggestion falls back to an error token and costs only the missing note. The
model is therefore tuned to minimize Incorrect first and maximize Correct
second.

Every rule carries a name, which the eval aggregates into per-rule firing counts
and precision. That is how an over-firing rule is found, and it is also what
makes the cost column tunable mechanically, by coordinate descent against the
eval rather than by hand.

`--d-values` sets how many brackets a trial deletes, which is worth raising to
see how gracefully recovery degrades when mistakes overlap. With one deletion
per trial, as of this writing:

| Mode            | Correct | Incorrect |
| --------------- | ------- | --------- |
| blank           | 95.6%   | 0.0%      |
| gapless         | 82.3%   | 4.3%      |
| truncate        | 98.9%   | 0.9%      |
| truncate-region | 82.6%   | 16.5%     |

The `fail_mismatched_brackets` tests in [testdata](/toolchain/lex/testdata) are
a showcase rather than a corpus: each case is a shape recovery handles, with a
comment naming the cue that decides it.

## Limitations

-   Recovery has no grammar. Its cues are local and categorical, so a bracket
    missing from deep inside an expression, with nothing unusual around it, is
    reported but not placed.
-   Region splitting relies on a top-level introducer at column 1. A file that
    never returns to top level is a single region, in which a mistake can smear
    further.
-   Costs are calibrated against this repository's own code in this repository's
    formatting. Substantially different style may want retuning.
-   The largest remaining source of wrong suggestions is a deleted tail: when
    the text that vanished held both the closer and the tokens that would have
    said where it goes, an unclosed `(` or `[` has nothing local to work from
    and over-extends.
