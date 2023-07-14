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
    -   [Language features](#language-features)
        -   [Code organization and structuring](#code-organization-and-structuring)
        -   [Type system](#type-system)
        -   [Functions, statements, expressions, ...](#functions-statements-expressions-)
        -   [Standard library components](#standard-library-components)
    -   [Project features](#project-features)
-   [Milestone 0.2: feature complete product for evaluation](#milestone-02-feature-complete-product-for-evaluation)
    -   [Features explicitly deferred until at least 0.2](#features-explicitly-deferred-until-at-least-02)
        -   [Why are coroutines and async in this milestone?](#why-are-coroutines-and-async-in-this-milestone)
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

-   Evaluators have a clear idea of the long-term evolution strategy of Carbon
    and how it addresses different use cases and requirements.
-   Language design components are documented, cohesive, and understandable by
    evaluators without placeholders.
    -   The components and language features must include the foundational core
        of the language. These features must also be sufficient to translate
        existing C++ code
        ([except coroutines](#why-are-coroutines-and-async-in-this-milestone))
        into obvious and unsurprising Carbon code.
    -   Also in-scope are additional features that impact API design or need
        early feedback, but only if they are low cost to both the design and
        implementation.
    -   Example language components include: lexical structure, expressions,
        statements, conditions, loops, user-defined types, and their
        dependencies.
    -   Example library components include: integer types, floating point types,
        strings, arrays, ranges, pointers, optionals, variants, heap allocation,
        and their dependencies.
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
        interoperability based around what impacts interesting benchmarks.

### Language features

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
    -   C++ interop: importing C++ types into Carbon, exporting Carbon types
        into C++
    -   Single inheritance
        -   Virtual dispatch
        -   C++ interop:
            -   Bi-directional inheritance between C++ and Carbon
            -   Type hierarchy roots in both C++ and Carbon
            -   Mappings of inheritance features: abstract, final, virtual
    -   Operator overloading
        -   C++ interop:
            -   Synthesizing Carbon overloads for imported C++ types
            -   Exporting Carbon overloads into C++
    -   Sum types (discriminated unions)
    -   Unions (un-discriminated)
        -   C++ interop: mapping to and from C++ unions.
-   Generics
    -   Both generic functions and types
    -   Checked generics
        -   Definition-checked variadics
    -   Integrated templates
        -   Including template-style structural conformance to nominal
            constraints, both modeling the members (like interfaces) and
            arbitrary predicates (like C++20 expression validity predicates)
    -   C++ interop:
        -   Importing C++ templates, instantiating on Carbon types
        -   Exporting Carbon templates, instantiating on C++ types
        -   Exporting Carbon checked generics (as templates), instantiating on
            C++ types
        -   Mapping C++20 concepts into named predicates, and named predicates
            into C++20 concepts

#### Functions, statements, expressions, ...

-   Functions
    -   Separate declaration and definition
    -   Function overloading
    -   C++ interop:
        -   Importing C++ functions and methods and calling them from Carbon
        -   Exporting Carbon functions and methods and calling them from C++
        -   Importing C++ overload sets into Carbon overload sets where the
            model (closed overloading) fits
        -   Importing C++ open-overload-sets-as-extension-points (`swap`, etc)
            into synthetic Carbon interfaces for common cases (likely based on
            heuristics)
-   Control flow statements
    -   Conditions
    -   Loops
        -   Range-based loops
        -   Good equivalents for a range of existing C/C++ looping constructs
    -   Matching
        -   Good equivalents for C/C++ uses of `switch`
        -   Working with sum-types, especially for C++ `std::variant` and
            `std::optional` interop
        -   Both positive (`if let` in Rust) and negative (`let else` in Rust)
            combined match control flow and variable declaration
-   C++ interop: support for C++'s threading and atomic primitives, memory
    model, and synchronization tools
-   Error handling
    -   Any dedicated error handling control flow constructs
    -   C++ interop:
        -   Mechanisms to configure how exception handling should or shouldn't
            be integrated into C++ interop sufficient to address both
            `-fno-except` C++ dialects and standard C++ dialects
        -   Calling C++ functions which throw exceptions from Carbon and
            automatically using Carbon's error handling
        -   Export Carbon error handling using some reasonably ergonomic mapping
            into C++ -- `std::expected`, something roughly compatible with
            `std::expected`, C++ exceptions, etc.

#### Standard library components

Note: we expect to _heavily_ leverage the C++ standard library by way of interop
for the vast majority of what is needed in Carbon initially. As a consequence,
this is a surprisingly more minimal area than the language features.

-   Language and syntax support library components
    -   Fundamental types (`bool`, `iN`, `fN`)
    -   Any parts of tuple or array types needed in the library
    -   Pointer types
    -   Interfaces powering language syntax (operators, conversions, etc.)
-   Types with important language support
    -   String and related types used with string literals
    -   Optional
    -   Slices
-   C++ interop:
    -   Transparent mapping between Carbon fundamental types and C++ equivalents
    -   Transparent mapping between Carbon and C++ _non-owning_ string-related
        types
    -   Transparent mapping between Carbon and C++ _non-owning_ contiguous
        container types
        -   Includes starting from an owning container and forming the
            non-owning view and then transparently mapping that between
            languages.
    -   Transparent mapping between Carbon and C++ iteration abstractions

### Project features

There are a few important components of the overarching Carbon project that need
to be completed as part of 0.1 beyond _language_ features:

-   A functioning Carbon toolchain:
    -   Supports drop-in usage as a Clang C++ toolchain with the most common
        Make- and CMake-derived build systems.
    -   Implements most of the [features](#language-features) above, including
        C++ interop, and any remaining gaps don't undermine the ability to
        evaluate the remaining features or the confidence in the overall
        evaluation.
    -   Installs on Windows, macOS, and Linux, and builds working programs for
        those platforms.
-   Build system integration for CMake, and documentation for integrating with
    Make or similar build systems.
-   Basic documentation for evaluators from getting started to FAQs.

## Milestone 0.2: feature complete product for evaluation

The second milestone already concretely in mind is reaching a level of
feature-completeness. The completeness metric here is based around the features
necessary to credibly address the _existing_ needs of C++ users and developers
interested in moving to Carbon, and so will be heavily driven by either features
already in use in C++ or necessary features to make moving off of C++ a viable
tradeoff. Ultimately, we need this milestone to be sufficiently feature complete
that _users can complete their evaluation of Carbon_. The language will continue
to evolve and grow features beyond this, but at this point there shouldn't be
any feature gaps that are problematic for the initial target audience coming
from C++, and we should be able to finalize the Carbon _experiment_ with this
feature set.

That said, this is about as concrete as we can get for a milestone that remains
years in the future. The full scope of requirements for this milestone will be
defined as we complete 0.1 and begin getting feedback on it. Currently, we just
call out specific features that we are actively deferring until at least 0.2 but
without being listed somewhere could cause confusion.

### Features explicitly deferred until at least 0.2

-   Memory safety
-   Coroutines, async, generators, etc.
-   Comprehensive story for handling effects, and how functions can be generic
    across effects
-   Carbon-native threading
-   Long tail of metaprogramming features
-   Mixins
-   Properties
-   Inline assembly
-   SIMD
-   Some ability to define an API that communicates & shares data across
    language versions, build configurations, and other FFI or ABI boundaries.
-   Necessary parts of the standard library

#### Why are coroutines and async in this milestone?

Specifically, why not address them earlier in 0.1? Or if they can be deferred
why not defer them further?

Coroutines and async programming are large and complex topics to introduce into
the language. From watching C++, Rust, Swift, Kotlin, and many other languages
working in this space, we have a strong belief that trying to add these to the
0.1 language would _significantly_ increase the amount of work and likely delay
when we reach that milestone. Also, given the recency of coroutines being added
to C++, we expect evaluators to be able to reason about their absence and still
accomplish most of the evaluation of Carbon without issue.

However, we also expect that as coroutines start to be widely adopted in C++,
they will become an essential feature of the language that would be extremely
difficult to give up when moving to Carbon. So we expect coroutines to be a
necessary feature for us to effectively decide that the Carbon experiment is a
success and begin planning large-scale adoption and migration.

## Milestone 1.0: no longer an experiment, usable in production

Even less concrete is the milestone that marks Carbon no longer being an
experiment, but if successful, a usable language. Currently this is
speculatively called 1.0 but even that is highly subject to change as we
approach. Again, we simply call out features here that we _do_ expect to have in
a 1.0 milestone, but want to explicitly defer beyond the 0.2 milestone.

### Features explicitly deferred beyond 0.2

-   Robust language evolution strategy and plan, specifically addressing:
    -   The need to make on-going changes to the language to address feedback
        and a changing landscape.
    -   The cost on users of churn and change over time and how to manage that
        cost.
    -   A mechanism to address users who need true, durable stability over long
        time horizons.
-   Package management strategy and plan, and any early groundwork needed.
-   Developer experience is high enough quality to enable initial production
    users.
    -   Includes compiler error messages and basic developer tooling.
-   Everything we've learned we need as part of the evaluation of 0.1 and 0.2
