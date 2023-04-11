# Milestones

<!--
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

<!-- toc -->

## Table of contents

-   [Overview](#overview)
-   [Milestone 0.1: a minimum viable product (MVP) for evaluation](#milestone-01-a-minimum-viable-product-mvp-for-evaluation)
    -   [Goals](#goals)
    -   [Features](#features)
        -   [Code organization and structuring](#code-organization-and-structuring)
        -   [Type system](#type-system)
        -   [Functions, statements, expressions, ...](#functions-statements-expressions-)
-   [Milestone 0.2: feature complete product for evaluation](#milestone-02-feature-complete-product-for-evaluation)
    -   [Features explicitly deferred until at least 0.2](#features-explicitly-deferred-until-at-least-02)
-   [Milestone 1.0: no longer an experiment, usable in production](#milestone-10-no-longer-an-experiment-usable-in-production)
    -   [Features explicitly deferred beyond 0.2](#features-explicitly-deferred-beyond-02)

<!-- tocstop -->

## Overview

As Carbon progresses, we want to have some common long-term milestones that we
orient our work around. The annual [roadmap](roadmap.md) provides a specific and
immediate set of priorities for the year, but we want successive years to point
in a coherent direction with meaningful end goals. Milestones should typically
be long-term, spanning more than a year, and have some functional motivation.

We also assign version numbers to our initial milestones to make them easy to
refer to and incorporate into various versioning schemes.

## Milestone 0.1: a minimum viable product (MVP) for evaluation

The first milestone is also the most concrete -- it is the MVP for C++ users and
developers to begin evaluating Carbon seriously. We want to keep this milestone
as minimal as we can while still enabling a sufficient initial round of
evaluation.

### Goals

From the perspective of outcomes, our goals for the 0.1 language are centered
around what we expect evaluations to be able to include:

-   Language design components are documented, cohesive, and understandable by
    evaluators without placeholders.
    -   The components and language features must include the foundational core
        of the language. These features must also be sufficient to translate
        existing C++ code (not using coroutines) into obvious and unsurprising
        Carbon code.
    -   Also in-scope are additional features that impact API design or need
        early feedback, but only if they are low cost to both the design and
        implementation.
    -   Example language components for this are: lexical structure,
        expressions, statements, conditions, loops, user-defined types,
        dependencies.
    -   Example library components are integer types, floating point types,
        strings, arrays, ranges, pointers, optionals, variants, heap allocation,
        etc.
    -   Where these build on top of other language or library designs, those are
        transitively in-scope.
-   Design for both Carbon use of C++ and C++ use of Carbon, including all major
    C++ language features except for coroutines, is documented, cohesive, and
    understandable by evaluators without placeholders.
-   Evaluators can build and run tests of most C++ interoperability, including
    both real-world C++ code that can be built with the latest release of Clang
    and test Carbon code.
    -   Gaps from the design need to be ones that don't undermine evaluation
        confidence.
-   Evaluators can effectively stress-test build speed and scaling with Carbon.
-   Evaluators can build some key benchmarks that include C++ interoperation in
    the critical path and get representative performance results.
    -   This can in turn be a still smaller subset of all aspects of C++
        interoperability based around what impacts interest benchmarks.

### Features

These are focused on the core _necessary_ features for us to reach a successful
0.1 language that can address our goals. Some of these features are required
directly by the above goals, others are required due to dependencies or
interactions with the directly required features. However, we don't try to cover
all of the features in full granularity here. There will be many minor
components that are necessary for these to hold together but are not directly
addressed. In general, unless something is explicitly described as partial or
having exceptions, everything covered by that entry should be expected in the
0.1 language.

Another important point is that this doesn't commit Carbon to any _particular_
design for any of these bullet points. Instead, it just means that the Carbon
design must have _something_ that addresses each of these bullet points. That
might be to add the named design to Carbon, but it equally might be a clear
statement that Carbon will _not_ include this design but use some other language
features to address its use cases and when rewriting C++ code using that feature
into Carbon.

#### Code organization and structuring

-   Packages
-   Libraries
-   Implementation files
-   Importing
-   Namespaces

#### Type system

-   User-defined types
    -   Single inheritance
        -   Virtual dispatch
    -   Operator overloading
    -   **Uncertain:** Mixins (depending on how much need there is to evaluate
        C++'s multiple inheritance use cases)
-   Generics
    -   Both generic functions and types
    -   Checked generics
        -   Definition-checked variadics
    -   Integrated templates
        -   "Template interfaces" and named predicates
-   Sum types

#### Functions, statements, expressions, ...

-   Functions
    -   Separate declaration and definition
    -   Function overloading
-   Control flow statements
    -   Conditions
    -   Loops
        -   Range-based loops
        -   Good equivalents for a range of existing C/C++ looping constructs
        -   (maybe? hard to justify) Labeled-break and complex loop support
    -   Match (someth
-   Interoperation with C++ threading and atomic primitives
-   Error handling

## Milestone 0.2: feature complete product for evaluation

The second milestone already concretely in mind is reaching a level of
feature-completeness. The completeness metric here is based around the features
necessary to credibly address the _existing_ needs of C++ users and developers
interested in moving to Carbon, and so will be heavily driven by either features
already in use in C++ or necessary features to make moving off of C++ a viable
tradeoff. Carbon will continue to evolve and grow features beyond this, but at
this point there shouldn't be any feature gaps that are problematic for the
initial target audience coming from C++, and we should be able to finalize the
Carbon _experiment_ with this feature set.

That said, this is about as concrete as we can get for a milestone that remains
years in the future. The full scope of requirements for this milestone will be
defined as we complete 0.1 and begin getting feedback on it. Currently, we just
call out specific features that we are actively deferring until 0.2 but without
being listed somewhere could cause confusion.

### Features explicitly deferred until at least 0.2

-   Memory safety
-   Coroutines, async, generators, etc.
-   Comprehensive story for handling effects, and how functions can be generic
    across effects
-   Carbon-native threading
-   Long tail of metaprogramming features
-   Inline assembly
-   SIMD
-   Some ability to define an API that communicates & shares data across
    language versions, build configurations, and other FFI or ABI boundaries.
-   Necessary parts of the standard library

## Milestone 1.0: no longer an experiment, usable in production

Even less concrete is the milestone that marks Carbon no longer being an
experiment, but if successful, a usable language. Currently this is
speculatively called 1.0 but even that is highly subject to change as we
approach. Again, we simply call out features here that we want to explicitly
defer and not cause confusion by omission above.

### Features explicitly deferred beyond 0.2

-   Complete language evolution story
-   Package management story
-   High quality developer experience, ranging from compiler error messages to
    development tooling
-   Everything we've learned we need as part of the evaluation of 0.1 and 0.2
