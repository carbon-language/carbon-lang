# Toolchain and language versioning

<!--
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

[Pull request](https://github.com/carbon-language/carbon-lang/pull/4105)

<!-- toc -->

## Table of contents

-   [Overview](#overview)
-   [Major version increments](#major-version-increments)
    -   [Breaking changes](#breaking-changes)
    -   [Exclusions to what constitutes a breaking change](#exclusions-to-what-constitutes-a-breaking-change)
-   [Minor version increments](#minor-version-increments)
-   [Patch version increments](#patch-version-increments)
    -   [Examples:](#examples)
-   [Pre-release versions](#pre-release-versions)
    -   [Release qualification pre-releases: `MAJOR.MINOR.PATCH-rc.N`](#release-qualification-pre-releases-majorminorpatch-rcn)
    -   [Incremental development versions: `MAJOR.MINOR.PATCH-0.{nightly,dev}.N`](#incremental-development-versions-majorminorpatch-0nightlydevn)
        -   [Nightly pre-release versions](#nightly-pre-release-versions)
        -   [Development pre-release versions](#development-pre-release-versions)
-   [Relevant proposal](#relevant-proposal)

<!-- tocstop -->

## Overview

Carbon uses a single versioning scheme across both the language itself and
toolchain, including the standard library, compiler, linker, and all development
tools released by the main Carbon project.

The scheme conforms to and is closely based on Semantic Versioning
(https://semver.org/ -- version 2.0.0):

-   Carbon versions: `MAJOR.MINOR.PATCH`
-   Releases with backwards incompatible changes, including a deprecation that
    might trigger a build-breaking warning, increment `MAJOR` after reaching the
    `1.0` milestone, and increment `MINOR` before then.
-   Releases with only backwards compatible changes are not expected to happen.
-   Releases containing only bug fixes increment `PATCH`.
-   Pre-release suffixes:
    -   `MAJOR.MINOR.PATCH-rc.N`: The N-th potentially viable candidate for a
        release.
    -   `MAJOR.MINOR.PATCH-0.nightly.YYYY.MM.DD`: A nightly incremental
        development build on a particular day during development of that
        version.
    -   `MAJOR.MINOR.PATCH-0.dev`: An interactive, incremental development build
        on a particular day by some developer during development of that
        version.

See the sections below for the details of each aspect of these versions.

## Major version increments

Aligned with SemVer, the major version must increment for any _breaking change_
to the language or any part of the toolchain.

The first increment from 0 to 1 is expected to be based on achieving a desired
milestone of feature completeness and quality to declare the language to have
reached a stable version.

Subsequent increments are expected to be done with a time-based release
strategy. Features (or breaking changes) ready to ship will do so, and others
will wait for the next major release. The exact cadence used is future work and
should be determined based on discussions with Carbon's users in the ramp up to
and after reaching the 1.0 milestone.

However, just because we increment the major version for a major release and
_can_ make breaking changes doesn't mean we _will_ or _should_. Breaking
language changes are extraordinarily expensive for users due to the inherent
scale of churn they impose. It is tempting to try to make non-breaking releases
instead, but our experience with C++ language, compiler, and standard library
updates is that truly making no breaking changes is extremely difficult and
overly constraining. We expect most releases, especially in the early phases of
the language to involve _some_ amount of breaking change and will simply work to
make these as cheap to upgrade through and minimal as we can.

At some future point, it is possible that Carbon will become so stable that it
makes sense to consider using minor version increments for some releases. If and
when this happens, we should revisit our versioning and release policies to
establish a predictable and unsurprising structure.

### Breaking changes

Beyond traditional breaking API changes in either standard library APIs or tool
APIs, Carbon also includes breaking changes in the language or toolchain.
Language and toolchain breaking changes are any that cause correct, functioning,
and non-reflective code to become invalid, rejected, incorrect, or silently
change behavior.

### Exclusions to what constitutes a breaking change

Carbon excludes "reflective code" which is in some way detecting the Carbon
version, or the presence or absence of features and as a consequence can be
"broken" by detecting changes that are correctly designed to otherwise be
non-breaking. We don't want adding features to be considered a breaking change
and so exclude code that specifically detects such additions.

Carbon also excludes breaking changes to incorrect code unless it was accepted
and functioning in some useful, and typically widespread, way despite its bugs.

## Minor version increments

Currently, Carbon plans to primarily use the minor version increments with a 0
major version to track our progress towards our 1.0 milestone of a feature
complete initial language. As a consequence we have defined 0.1 and 0.2
milestones and may define more steps as needed.

Beyond this and in a post-1.0 language world, we expect most significant
features to also accompany some small and manageable breaking changes from
deprecations. We may choose to revisit this in the future, but our current plan
is not to make minor version releases post-1.0 and instead focus on our
commitment to making those updates both easy and scalable for language users.

## Patch version increments

The patch version will increment when the change is fundamentally a bug fix to a
previously released version. We expect the vast majority of these to be strictly
backwards compatible bug fixes.

Patch releases are expected to be driven by demand, and not necessarily present
if unnecessary. However, the exact schedule and process will be determined as
part of getting ready for the 1.0 milestone. Before that milestone we don't
commit to any particular process or cadence for patch releases as no release
before that point should be considered stable.

Note that we still consider restoring the _intended_ "public API" of a release
to be a bug fix. When these bug fixes theoretically break new code in a way that
would typically require a major version increment, they may be made with merely
a patch version increment when they are in fact restoring our intended behavior
for that release. However, we take the SemVer guarantees very seriously as these
fixes can still be disruptive and so we work to hold a high bar for them:

-   They must be in some way fixing a _regressions_ for users from a previous
    release, not merely a missing feature.
    -   Can even be a regression in the overall cohesion or reliability of the
        language or tools, which a bug in a new feature might erode.
    -   Key is that we motivate any patch fix through the lens of a regression
        fix and stabilization rather than fixing forward.
-   The scope of disruption from the fix is demonstrably small, due to some
    combination of:
    -   The short time duration of the release containing the regression.
    -   The narrow scope of code constructs potentially impacted.
-   The impact of the regression is large and cannot be easily worked around,
    for example:
    -   Undermining a core priority of the Carbon Language for a significant set
        of users.
    -   Making the engineering cost of adopting the release uneconomical for any
        significant body of users.

### Examples:

-   We add a new feature that includes a bug which creates unsoundness and
    allows incorrect code to be compiled that will crash or exhibit UB when
    executed.
    -   While this is a new feature and not a bug in an existing feature, it
        would be a _serious_ regression to the reliability of the language as a
        whole and the ability of users to reason about the correctness of
        programs.
    -   This would be a good candidate for a patch release to address unless it
        is found very late (months) after the release _and_ the only ways to
        address would have similarly bad effects on code written since the
        initial release.
    -   Even that level of user disruption could potentially be overcome if for
        example the bug led to security vulnerabilities.
-   We add a narrowly used new feature that includes a bug where some code
    patterns that _should_ work with the feature are rejected at compile time or
    crash reliably.
    -   Unlikely this is worth a patch release to make an invasive change given
        the narrow use case.
    -   A good candidate to introduce a warning or error on using the feature in
        the way that might lead to a crash, and possibly on using the feature at
        all.
-   We add a compiler opt-in feature behind a flag that doesn't work reliably.
    -   Good candidate to have the flag disabled or trigger a warning message.
    -   Not a good candidate to try to fix the feature forward.
-   We add a compiler opt-out feature that doesn't work reliably.
    -   What to do likely depends on the scope of users impacted. If _very_ few
        users impacted, possibly just document how to opt-out.
    -   If enough are impacted to be a nuisance and a regression in experience
        in general, likely worth attempting a patch release that narrowly fixes
        or mitigates the issue.

## Pre-release versions

Beyond the major, minor, and patch versions of an actual release, SemVer
provides a foundation for pre-release version management. However, it is a very
open-ended foundation with a wide range of possibilities and no particular
semantic model provided. Some of this is to support the wide variety of
different needs across different projects. It also reflects the fundamentally
less crisp and well defined criteria needed for pre-releases.

That said, Carbon should try to have a small and well articulated scheme of
pre-release versions to help communicate what these releases do and don't
constitute and how they should be interpreted. These are selected to provide a
framework that allows pre-release versions to order in a reasonably cohesive way
when compared using SemVer.

In descending order, Carbon pre-releases may use:

-   `MAJOR.MINOR.PATCH-rc.N`
-   `MAJOR.MINOR.PATCH-0.nightly.YYYY.MM.DD`
-   `MAJOR.MINOR.PATCH-0.dev`

We expand on each of these below to provide criteria and details.

### Release qualification pre-releases: `MAJOR.MINOR.PATCH-rc.N`

We create release candidate or "rc" pre-releases when we believe that version to
be complete and ready to release and want to collect feedback. There should not
be interesting or significant gaps, even known ones, from the intended release.
The expectation should always be that unless some feedback arrives to the
contrary, the release candidate could simply become the release.

For each pre-release category, we always suffix with a sequential count `N` of
the pre-releases at that version. We must start with a `.0` in order to have
subsequent iterations of the same version and category of pre-release to sort
after the first.

### Incremental development versions: `MAJOR.MINOR.PATCH-0.{nightly,dev}.N`

These are versions that are not in any way associated with the actual expected
release, but are periodically produced as an incremental tracking of development
progress. Because they are not expected to be part of qualifying a specific
release, they're not defined by any particular criteria of completeness or
readiness. In practice, these will occur at every stage between one release and
the next.

#### Nightly pre-release versions

These are automated incremental development versions built each night when the
automated testing passes. There is _no_ attempt to provide any higher quality
bar or refinement than whatever was in the tree at the time the automation runs,
and whatever automated tests are present pass.

It is important to emphasize that the primary use case of these pre-release
versions is not to evaluate a potential release but for the developers and
contributors to Carbon itself to track incremental development progress. That
development-oriented goal drives how they are built and what they do and don't
provide.

Mechanically, we prefix the `nightly` pre-release version component with a `0`
component to ensure these versions sort before any and all release qualification
pre-release versions. We also add a date-derived suffix to provide a rough
chronological ordering of nightly builds with otherwise identical versions.

#### Development pre-release versions

During development, interactive builds of Carbon need to be versioned in an
unambiguous way, and we do that with the `dev` pre-release version. Much like
nightly versions, these are only produced as artifacts of development activities
and are never part of any actual release process. These versions are further not
built expected to be automated or necessarily repeatable. They may contain
in-flight edits and all manner of other variations.

Mechanically, we prefix the `dev` pre-release version component with `0` in the
same way as `nightly` is prefixed to ensure effective ordering. We don't add any
additional information to keep the mechanics of development builds simple -- for
example, there is no easy way to extract the date of the build. The exact
timestamp of the build may be available but doesn't participate for simplicity
and minimizing cache impact.

## Relevant proposal

-   [Proposal p4105](/proposals/p4105.md)
