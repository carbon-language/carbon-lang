# Establish toolchain and language versioning

<!--
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

[Pull request](https://github.com/carbon-language/carbon-lang/pull/4105)

<!-- toc -->

## Table of contents

-   [Abstract](#abstract)
-   [Problem](#problem)
-   [Background](#background)
-   [Proposal](#proposal)
-   [Details](#details)
-   [Directional sketch for the future](#directional-sketch-for-the-future)
    -   [Language evolution and breaking changes](#language-evolution-and-breaking-changes)
    -   [Long-Term Stable (LTS) versions and standardization](#long-term-stable-lts-versions-and-standardization)
-   [Rationale](#rationale)
-   [Alternatives considered](#alternatives-considered)
    -   [Do nothing, or just talk about a minimal nightly version.](#do-nothing-or-just-talk-about-a-minimal-nightly-version)
    -   [Make no breaking changes past 1.0.](#make-no-breaking-changes-past-10)
    -   [Version different parts of the language separately.](#version-different-parts-of-the-language-separately)
    -   [Use a custom versioning scheme rather than SemVer.](#use-a-custom-versioning-scheme-rather-than-semver)
    -   [Include more pre-release variations](#include-more-pre-release-variations)

<!-- tocstop -->

## Abstract

Proposal for how Carbon version numbers work:

-   A single version across language, standard library, compiler, linker, etc.
-   Semantic Versioning (SemVer) based
-   Details of how SemVer criteria for major, minor, and patch should apply to
    Carbon
-   Details of how we will operate before 1.0 and how this connects to Carbon's
    milestones
-   Directional guidance for future work including post-1.0 versions, LTS
    versions, and standardization

## Problem

We need a versioning scheme for Carbon both for the language and the reference
toolchain implementing that language. This is important even before we reach any
specific milestone, as we want to define the schema and implement it _before_ it
becomes useful for marking specific milestones.

## Background

-   [Semantic Versioning](https://semver.org/)
-   [Rust Editions](https://doc.rust-lang.org/edition-guide/editions/)

## Proposal

First, Carbon should have a single versioning scheme across both the language
itself and toolchain, including the standard library, compiler, linker, and all
development tools released by the main Carbon project.

Second, the Carbon versioning scheme should conform to and be closely based on
Semantic Versioning (SemVer), which is the de-facto standard for versioning
schemes in software today. Beyond that, it needs to clarify how the standards
laid out in SemVer map into a programming language context as programming
languages and standard libraries have an extraordinarily broad and tightly
coupled "API" to their users -- all of the source code written in the language.
Carbon needs to provide extra clarity around what constitutes our "public API"
for SemVer purposes and the criteria for changes.

Third, SemVer provides a schema for pre-release versions, but is largely
open-ended on their semantics. Carbon should have a specific set of well defined
pre-release versions with clearly communicated purpose, nature, and meaning to
avoid confusion.

Fourth, language versioning is an especially important area for the long-term
evolution and so Carbon should have some directional guidance around the future
work expected in the versioning front. This should speak to specific use cases
and needs that may be left seemingly unaddressed otherwise.

Summarizing the proposed outcome of these together:

-   Carbon versions: `MAJOR.MINOR.PATCH`
-   `MAJOR` increments on backwards incompatible changes, including a
    deprecation that might trigger a build-breaking warning.
    -   Doesn't make it free to make such changes. Each change must pay for its
        adoption and churn cost. But when a change is needed and well justified,
        this signifies its introduction.
    -   Used to establish our milestones for the language.
    -   Some explicit carve-outs of things designated to not be in the "public
        API" of the language.
-   `MINOR` increments only expected during early development with a major
    version of `0`.
    -   If Carbon some day stabilizes sufficiently to motivate it, we may
        revisit this and begin to use the minor version number to signal
        backwards compatible releases.
-   `PATCH` increments represent bug fixes only.
    -   Goal is always fully backwards compatible.
    -   When fixing the bug makes code written against the release with the bug
        break, may be unavoidable as the intent was never the buggy behavior.
        But this is rare and we hold a very high bar for such bug fixes due to
        their disruptive nature.
-   Pre-release suffixes:
    -   `MAJOR.MINOR.PATCH-rc.N`: The N-th potentially viable candidate for a
        release.
    -   `MAJOR.MINOR.PATCH-0.nightly.YYYY.MM.DD`: A nightly incremental
        development build on a particular day during development of that
        version.
    -   `MAJOR.MINOR.PATCH-0.dev`: An interactive, incremental development build
        on a particular day by some developer during development of that
        version.

## Details

See the added [versioning document](/docs/project/versioning.md).

## Directional sketch for the future

The mechanics outlined above provide a good basis for the initial versions of
the language (up to 1.0) and any necessary mechanics and tooling around those
versions. However, beyond 1.0 we expect the needs of the language and project to
expand and more detailed versioning and evolution tools to become critical. We
lay out directional sketches here for where Carbon should go in the future to
address these needs, but these are just directional guidance and will need their
own carefully considered proposals when the time comes.

### Language evolution and breaking changes

We don't expect a simple version number to be sufficient long-term for the
evolution needs of the Carbon language. We should plan to at least map these
major versions into Rust-edition-like controls within source code itself to
allow incremental adoption across a codebase of fixes for breaking changes or
adoption of new language features with a single toolchain version. That is, some
code will want to compile using previous major version semantics even with the
new compiler.

The approach taken in Rust and proposed for C++ to address this are "editions"
that source code opts into in order to allow the compiler to support a mixture
of code in a codebase during incremental adoption. Carbon will need at least
something equivalent to this, and may want to explore a more fine-grained system
of opting into specific functionality sets similar to how pragma-based extension
usage or Circle works.

Regardless of the specifics, a key is that breaking changes are not forcibly
coupled in their roll-out to updates to the Carbon toolchain. Each step needs to
be incrementally tackled.

### Long-Term Stable (LTS) versions and standardization

SemVer alone isn't sufficient to address some user needs for language stability.
It is enough to _detect_ the issues when they arise, but Carbon should also plan
for how to _address_ these issues.

The suggested direction here is towards designated LTS versions based on a
particular level of completeness and quality and the user demand. These versions
will likely need even longer time horizons of support than Linux distro LTS
releases. The direction should be to embrace this and the potential for
multi-decade support windows to support users' needs. As the windows of LTSes
expand, their frequency should reduce to avoid supporting an unsustainable
diversity of versions.

Exactly how a version is designated as LTS is left to the future work here, but
it should not be expected to change the schema and structure of the versioning,
just the support policy applied to the specific release version in question.

Some users may even require standardization of a programming language to make it
usable in their environment. Carbon should again embrace this need and see the
standardization as an analogous process to promoting a normal release into an
LTS. Some relevant and effective LTS should be selected and taken through
whatever process is identified to create a standard reflecting that LTS version
of Carbon. Updates to the standard should in turn track as updates to a newer
LTS. The specifics of how to do this are left to the future work, and they may
change exactly how this works.

Note that the goal of this future direction isn't to constrain how Carbon can
arrive effectively at either an LTS release or a standard. Instead, the goal is
to make it clear that we _should_ be open and planning to achieve these in order
to meet the needs of candidate Carbon users.

## Rationale

-   [Language tools and ecosystem](https://github.com/carbon-language/carbon-lang/blob/trunk/docs/project/goals.md#language-tools-and-ecosystem)
    -   Carbon needs a coherent versioning scheme across the language itself as
        well as its ecosystem. Especially as the language is developing rapidly,
        being able to sync across all of these with a single, simple versioning
        scheme is especially important to have the tooling and ecosystem agree
        about features of the language.
-   [Software and language evolution](https://github.com/carbon-language/carbon-lang/blob/trunk/docs/project/goals.md#software-and-language-evolution)
    -   Users of Carbon need a clear model for understanding the language's
        evolution, tracking it, and responding to significant aspects of it.
    -   The versioning scheme needs to support a wide variety of versions built
        in the process of evolving the language without creating confusion.

## Alternatives considered

### Do nothing, or just talk about a minimal nightly version.

Advantages:

-   Carbon is at a very early stage and is a long way from needing release
    candidates for stable releases or incrementing its major version numbers.
-   No need to imagine distant future scenarios.

Disadvantages:

-   We already discuss version numbers in various places in the Carbon project,
    including our [milestones](/docs/project/milestones.md) and
    [roadmap](/docs/project/roadmap.md).
-   Establishing how the Carbon project will communicate its updates with
    version number changes in advance of those updates makes that communication
    more effective.
-   Lets us telegraph our intentions and how we are thinking about releasing and
    cadence to the community and get earlier feedback.

A key to adopting the more detailed versioning plan is that we can change any
and all of this if and when we need to. This does not lock Carbon into using
this exact versioning scheme. We will listen to any feedback from potential
users and can adapt our approach if needed.

### Make no breaking changes past 1.0.

Advantages:

-   Stable languages have the lowest churn costs for users, and are easier to
    learn at scale.

Disadvantages:

-   This would be in direct opposition to our language goal of supporting
    software and language evolution.
    -   This relieves pressure on adding things to the language by avoiding an
        unreasonably high bar. The pressure is still large due to the very real
        churn costs, but we avoid amplifying that further with an absolute
        restriction on fixing issues.
    -   We expect Carbon to be a reasonably complex language in order to succeed
        at its goals, ranging from C++ interop to incremental memory safety.
        This complexity inherently comes with an increased risk and importance
        of being able to improve and fix issues.

### Version different parts of the language separately.

Advantages:

-   There will be eventually be many differences between changes to the
    toolchain and changes to the language itself. We might be able to capitalize
    on those to have a better cadence or versioning scheme for these
    independently.

Disadvantages:

-   We would have to carefully define and maintain a compatibility matrix
    between the different components. Increasingly in modern languages and
    development, the compiler, language, standard library, and tooling are all
    deeply interdependent.
-   Experience rolling out major updates to Clang and GCC in large codebases and
    software ecosystems show even compiler-only or toolchain-only changes easily
    become as disruptive as smaller updates to the C++ language itself have been
    over the years. As a consequence, while it is tempting to hope for a sharp
    difference here we don't in practice anticipate one.

### Use a custom versioning scheme rather than SemVer.

Advantages:

-   We don't actually use all parts of SemVer, which results in awkward unused
    component of our version number.
-   SemVer doesn't actually provide an opinionated versioning scheme, merely a
    relaxed schema that many versioning schemes can fit into.

Disadvantages:

-   Fitting into SemVer ends up needing only very cosmetic changes to a scheme
    that is meaningful for Carbon. And we can easily specify the open-ended
    parts of the scheme.
-   Allows us to easily fit into scripts and people's understanding using SemVer
    as a baseline model.
-   Avoids confusion when people from outside the community first encounter
    Carbon versions as their most likely intuition about the meaning of various
    aspects of the number will be a reasonable starting point.

### Include more pre-release variations

Initially there was a discussion of potentially defining `alpha` and `beta`
pre-release versions along with release candidate versions, nightly, and
development versions. The specific idea was to document versioning we could use
for longer-lived releases we want to make when not yet ready to call it a
release candidate.

Advantages:

-   If we end up wanting longer-lived releases prior to arriving at a numbered
    milestone such as `0.1`, this would provide a ready-made solution.
-   Specifically, communicating the option for this early might help others
    avoid being confused by what exactly the state of such a pre-release would
    be.

Disadvantages:

-   There's no real indication we would ever want to make such a release. It
    seems easy to imagine the combination of nightly builds and pre-releases
    completely covering all of the use cases we end up with in practice.
-   Keeping the infrastructure and documentation for such improbable use cases
    is drag and friction that doesn't buy us enough to be worthwhile.
