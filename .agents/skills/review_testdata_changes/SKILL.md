---
name: Review testdata changes
description:
    Instructions for judging whether changes to file test output
    (`// CHECK:STDOUT:` and `// CHECK:STDERR:` lines) are correct, which
    changes are acceptable churn, and which are regressions in disguise.
---

# Review testdata changes

<!--
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

## Introduction

Most toolchain work moves file test output. `./toolchain/autoupdate_testdata.py`
rewrites the `// CHECK:STDOUT:` and `// CHECK:STDERR:` lines in
`toolchain/*/testdata/` to match current behavior, so after running it the tests
pass again whether or not the new behavior is right. Deciding that the new
output is the output you wanted is a separate, manual step, and it is the step
this skill covers.

Three skills divide the work:

-   [Toolchain tests](../toolchain_tests/SKILL.md): how to _author_ tests and
    generate their output.
-   This skill: how to _judge_ an output diff.
-   [Summarize testdata changes](../summarize_testdata_changes/SKILL.md): how to
    _report_ an output diff once you believe it is correct.

> [!IMPORTANT] Never hand-edit `// CHECK:STDOUT:` or `// CHECK:STDERR:` lines.
> Everything below is about changing the _code_ until the generated output is
> right, never about editing the output to match the code.

## The rule that generates all the others

**Every line of testdata churn must have a cause you can name.** Not "it is
similar to the other changes", not "the tests pass now" — an actual sentence
saying which code change produced it and why that is the intended result.

A diff you cannot narrate is a diff you have not reviewed. Changes you cannot
explain are where regressions hide, because a regression and an intended change
look exactly alike once the autoupdater has written them down.

> [!CAUTION] Autoupdating is destructive to your evidence. Once the autoupdater
> has run, the previous expectations are gone from the working copy. Read the
> diff after _every_ autoupdate run, and if something changed for a reason you
> cannot name, fix the code before autoupdating again. Recovering the old
> expectations later means reverting and re-running.

## STDERR and STDOUT are different kinds of evidence

Treat them separately; they have different standards of proof.

**STDERR is user-visible behavior.** These are the diagnostics a Carbon
programmer sees. A change here is a change to the language implementation as
users experience it, so each one needs an individual justification. For a
refactoring, the expected STDERR diff is empty.

**STDOUT is internal representation.** SemIR dumps, parse trees, LLVM IR. Users
never see it. Churn here is normal and often unavoidable, so the standard is not
"no change" but "no change I cannot account for".

For a change that is supposed to preserve behavior, write the list of accepted
STDERR changes down _before_ you autoupdate, and keep it current. Order matters
more than form: written first, the list is a prediction the diff can falsify;
written afterwards, it is a description of whatever happened, and describes a
regression exactly as well as an intended change.

The list does not have to be a deliverable. Scratch notes you never publish do
the job, because the work is in committing to the list, not in presenting it.
What a reader needs is not the list but its exceptions: diagnostics that changed
without being predicted, and predictions that did not occur. Both are findings.
The matches are not, and reporting them buries the two entries that matter.

## Judging STDOUT churn

Sort each STDOUT change into mechanical or structural.

### Mechanical churn

Expected, and cheap to accept in bulk once you have confirmed the pattern:

-   **Renaming.** An instruction, type, or scope prints under a new name.
-   **Positional name renumbering.** Names like `%x.loc18_46.3` embed a line,
    column, and disambiguating index. Two different events move them:
    -   Adding or removing an instruction at a location renumbers the rest, so a
        _single_ removed instruction can show up as many changed lines in the
        same block. Confirm the cascade is a cascade before accepting it as one.
    -   Adding or removing a _diagnostic_ moves the source lines themselves, so
        the line component changes everywhere below it in the file. See
        [Autoupdate to a fixed point](#autoupdate-to-a-fixed-point).
-   **Fingerprint-derived names.** Mangled names and some scope names are
    derived from a hash of their inputs. If you changed a hashed input, these
    move. Confirm that each such difference is _only_ the fingerprint, and not a
    fingerprint difference concealing a structural one.

> [!TIP] Mechanical churn is usually wide and shallow: the same substitution,
> repeated across many files. If a "mechanical" pattern needs a different
> explanation in each file, it is not mechanical.

### Structural churn

Each of these needs its own explanation:

-   Instructions appearing or disappearing.
-   Changed `[concrete = ...]`, `[symbolic = ...]`, or other constant-value
    annotations.
-   Changed types on existing instructions.
-   Changed control flow: new or removed blocks, changed branch targets.
-   Raw instruction ids renumbering in `--dump-raw-sem-ir` output when you did
    not intend to change the id layout.

## Fewer instructions is not automatically better

A diff that removes instructions looks like an optimization. Whether it is one
depends on what the changed function owes its caller, and two functions can
produce the same shrinking diff for opposite reasons:

-   A function that only needs a constant value, but built instructions on the
    way to it, was doing wasted work. Dropping them and using the constant
    directly is correct, and the shorter output reflects that.
-   A function whose caller needs a **non-canonical instruction** can be made
    shorter the same way, by using the constant instead of creating an
    instruction of its own — and that is wrong. The instruction carries location
    information, and symbolic constant substitution operates on it; the constant
    value alone loses both.

The instruction count is identical evidence in both cases, so it cannot be the
thing you judge. Decide what the function is required to produce; the count
follows from that.

The same reasoning runs in reverse: a diff that _adds_ instructions is not
automatically a regression.

## Judging STDERR churn

### A diagnostic's authority depends on the file's prefix

The `fail_` and `todo_` prefixes (see
[Toolchain tests](../toolchain_tests/SKILL.md)) say how much the recorded
diagnostics are worth:

-   **`fail_...`** — the test should and does produce errors. The recorded
    diagnostic **is the specification**. Changing it is a user-visible behavior
    change and needs justification on its own merits.
-   **`fail_todo_...`** — the test produces errors (or crashes) but shouldn't,
    or produces the wrong errors. The recorded diagnostic is explicitly **not**
    the specification; the file exists to record that today's behavior is wrong.
    Changing it replaces one wrong answer with another. That is acceptable when
    you can say why the new message follows from your change and why it is no
    further from the intended eventual behavior — which, for many such files, is
    no diagnostic at all.
-   **`todo_fail_...`** — the test should produce errors but does not. Gaining a
    diagnostic here may be _progress_, not a regression. Either way the file
    must be renamed, since the framework requires a `fail_` prefix on any file
    that errors: `fail_...` if it now produces the right error, `fail_todo_...`
    if the error is the wrong one. Both renames make the test pass, so record
    which case it is instead of letting the rename settle it.
-   **`todo_...`** — behavior is wrong but produces no errors, and shouldn't.
    Gaining a diagnostic here is a regression unless you can argue otherwise.

> [!IMPORTANT] This is a reason to look at the _filename_ before judging a
> diagnostic change, not a license to ignore `todo_` files. "It was already
> broken" does not excuse making it differently broken for no reason.

### Reclassifying tests

If your change moves a test between the states above, the prefix must move with
it: when the test is fixed, when it starts failing, and when it starts failing
differently. The correspondence between the `fail_` prefix and whether
compilation actually failed is enforced by the test framework, not by the
autoupdater, so it surfaces when you run `bazelisk test` and not when you
autoupdate. **Autoupdating is not a substitute for running the tests.**

Only that half of the name is checked. Nothing enforces `todo_`, so a file whose
behavior you have just fixed can keep its `todo_` prefix indefinitely and still
pass. A missing `fail_` stops the build; a stale `todo_` is silent, and is yours
to catch.

When a fix drops a file's prefix, also check that the file still belongs where
it is and that its comments do not still describe the old broken behavior.

### Vaguer diagnostics are a signal, not a verdict

When a diagnostic becomes less specific — a general "unsupported" message
replacing one that named the problem — that usually means a code path stopped
finding information it previously had. Sometimes that is correct: the
information was misleading, and the old message was confidently wrong.

Do not accept it silently and do not reject it reflexively. Say which direction
it moved and whether the new message is closer to or further from the eventual
intended behavior.

## Signals in the shape of the diff

### Zero churn is a result

If you removed something you believed was doing work and _no_ testdata moved,
that is not a missing test run — it is the proof that the thing was a no-op.
Say so explicitly; it is one of the strongest pieces of evidence a refactoring
can produce.

The converse is also informative. If you expected a path to churn and it didn't,
either your model of the code is wrong or that path is untested. Find out which,
and consider adding a test before continuing.

### Churn should be proportional to the change

-   **Wide churn from a narrow change** means your model of the code is wrong.
    Do not autoupdate over it. Find the structural mistake first.
-   **Narrow churn from a sweeping change** means the affected paths are
    probably untested.

Set a rough expectation for the size of the diff before you run the autoupdater,
and treat a large mismatch in either direction as a finding.

### Never make the diff smaller by weakening the test

Editing test _input_ (the Carbon source, not the CHECK lines) to make a diff
look better is a behavior change in disguise. Deleting a test that now produces
awkward output is worse. If a test's input has to change, that is a separate,
explicitly-justified change, not diff cleanup.

## What the autoupdater will not fix for you

-   **`NOAUTOUPDATE` files.** Their expectations are maintained by hand. They
    fail under `bazelisk test` rather than being silently rewritten.
-   **Hand-written C++ expectations**, for example golden output asserted in a
    `_test.cpp`. When several assertions in one of these break together, often
    only the first failure is reported, so fixing it can reveal another. Re-run
    until clean rather than assuming one fix was the whole repair.
-   **Golden files outside `testdata/`**, and documentation that quotes compiler
    output.

## Review loop

### Run the autoupdater

Autoupdate everything, then read the diff a directory at a time. Passing no
paths is the default and updates every file test in the toolchain:

```bash
./toolchain/autoupdate_testdata.py
```

Narrow the scope only while iterating on one subdirectory you know you are not
done with, where each round would otherwise regenerate output you have already
read:

```bash
./toolchain/autoupdate_testdata.py toolchain/check/testdata/SUBDIR/**/*
```

The globs are expanded by the shell; the script filters its arguments to
`.carbon` files under a `testdata/` directory.

> [!TIP] If intermediate states crash on `CARBON_CHECK` failures, pass
> `--non-fatal-checks` so you can see the full set of downstream damage in one
> run instead of one crash at a time.

> [!IMPORTANT] Return to the full scope before judging the diff. Whether churn
> is proportional to the change, and whether a change to one phase moved another
> phase's testdata, are only visible across everything.

### Autoupdate to a fixed point

One pass is not always enough, because the autoupdater's output is part of its
own input. `// CHECK:STDERR:` lines sit inline, immediately above the source
line they describe, and the compiler reads them as comments in the file.
Gaining or losing a diagnostic therefore moves every source line below it.

Within a single run, the compiler has already read the file as it was, so the
two kinds of output end up in different states:

-   **STDERR is correct after one pass.** These lines locate themselves
    relatively, as `[[@LINE+N]]`, and the autoupdater recomputes `N` as it
    places them.
-   **STDOUT is stale after one pass.** SemIR names like `%x.loc18_46.3` embed
    an absolute line and column with no filename attached, and the autoupdater
    only remaps `file.carbon:18`-style references. Nothing rewrites the `loc`,
    so it still describes where the instruction was _before_ the diagnostic
    lines moved it.

Running again compiles the shifted file and the names catch up. The diagnostic
set does not change this time, so nothing shifts again and a third run is a
no-op. Keep running the autoupdater until it stops changing files: one pass when
the diagnostics held still, two when they didn't.

> [!WARNING] The intermediate state is self-inconsistent, not just unfinished:
> its `loc` names describe a file layout that no longer exists. Keep reading the
> diff after every run — that rule does not change — but do not chase positional
> churn to a cause until the file has converged, and do not present the diff
> until then either.

The converging pass should be positional renumbering and nothing else. If it
moves an instruction, a type, or a constant value, then something other than a
`loc` name is sensitive to where lines fall in the file. Find out what before
accepting it.

The file tests do catch a file left unconverged, since each test re-runs the
autoupdate in memory and fails with
`Autoupdate would make changes to the file content` when the result differs. But
that arrives at `bazelisk test` time, after you have already read a diff that
was describing a file which had moved out from under it.

### Inspect the diff

Inspect the diagnostics first, since that is the acceptance criterion:

```bash
# STDERR-only view, with jj.
jj --no-pager diff --git 'glob:toolchain/*/testdata/**' \
  | grep -E '^[-+].*CHECK:STDERR'

# The same, with git.
git diff -- 'toolchain/*/testdata/*' \
  | grep -E '^[-+].*CHECK:STDERR'
```

For a structured view separating test input, STDERR, and STDOUT changes, use the
helper from the
[Summarize testdata changes](../summarize_testdata_changes/SKILL.md) skill:

```bash
jj --no-pager diff --git 'glob:toolchain/*/testdata/**' \
  | python3 .agents/skills/summarize_testdata_changes/scripts/parse_diff.py

git diff -- 'toolchain/*/testdata/*' \
  | python3 .agents/skills/summarize_testdata_changes/scripts/parse_diff.py
```

The argument after the tool name is not interchangeable: `glob:...` is a jj
fileset, `-- ...` is a git pathspec. `--git` and `--no-pager` are jj flags; git
already emits this format and skips the pager when piped.

### Run the tests

Then run the tests, which is what catches prefix mismatches and non-autoupdated
expectations:

```bash
bazelisk test //toolchain/...
```

See the [Bazel usage](../bazel/SKILL.md) skill.

## Checklist

Before presenting a testdata diff as finished:

-   [ ] The autoupdater was run until it made no further changes, so no `loc`
        name describes a stale line numbering.
-   [ ] The list of accepted STDERR changes was written before autoupdating, and
        every change either matches it or is reported as an exception.
-   [ ] Every STDOUT change is either an instance of a named mechanical pattern
        or has its own explanation.
-   [ ] Instructions that appeared or disappeared are justified by what the
        changed function owes its caller, not by the instruction count.
-   [ ] Files whose prefix no longer matches their behavior have been renamed.
-   [ ] No test input was changed, and no test was deleted, to make the diff
        smaller.
-   [ ] The size of the diff is proportional to the size of the change.
