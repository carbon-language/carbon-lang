# Project FAQ

<!--
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

<!-- toc -->

## Table of contents

-   [What is Carbon?](#what-is-carbon)
-   [What is Carbon's status?](#what-is-carbons-status)
    -   [How soon can we use Carbon?](#how-soon-can-we-use-carbon)
    -   [Why make Carbon public while it's still an experiment?](#why-make-carbon-public-while-its-still-an-experiment)
    -   [How complete is Carbon's design?](#how-complete-is-carbons-design)
    -   [How many people are involved in Carbon?](#how-many-people-are-involved-in-carbon)
    -   [Is there a demo?](#is-there-a-demo)
-   [Why build Carbon?](#why-build-carbon)
    -   [Why is performance critical?](#why-is-performance-critical)
    -   [What level of C++ interoperability is expected?](#what-level-of-c-interoperability-is-expected)
    -   [What would migrating C++ code to Carbon look like?](#what-would-migrating-c-code-to-carbon-look-like)
-   [What alternatives did you consider? Why did they not work?](#what-alternatives-did-you-consider-why-did-they-not-work)
    -   [Why not improve C++?](#why-not-improve-c)
    -   [Why not fork C++?](#why-not-fork-c)
    -   [Why not Rust?](#why-not-rust)
    -   [Why not a garbage collected language, like Java, Kotlin, or Go?](#why-not-a-garbage-collected-language-like-java-kotlin-or-go)
-   [How will Carbon work?](#how-will-carbon-work)
    -   [What compiler infrastructure is Carbon using?](#what-compiler-infrastructure-is-carbon-using)
    -   [How will Carbon's bidirectional C++ interoperability work?](#how-will-carbons-bidirectional-c-interoperability-work)
    -   [How do Carbon generics differ from templates?](#how-do-carbon-generics-differ-from-templates)
    -   [What is Carbon's memory model?](#what-is-carbons-memory-model)
    -   [How will Carbon achieve memory safety?](#how-will-carbon-achieve-memory-safety)
-   [How will the Carbon _project_ work?](#how-will-the-carbon-_project_-work)
    -   [Where does development occur?](#where-does-development-occur)
    -   [How does Carbon make decisions?](#how-does-carbon-make-decisions)
    -   [What happens when a decision was wrong?](#what-happens-when-a-decision-was-wrong)
    -   [What license does Carbon use?](#what-license-does-carbon-use)
    -   [Why make Carbon open source?](#why-make-carbon-open-source)
    -   [Why does Carbon have a CLA?](#why-does-carbon-have-a-cla)
    -   [Who pays for Carbon's infrastructure?](#who-pays-for-carbons-infrastructure)

<!-- tocstop -->

## What is Carbon?

The [Carbon Language](/README.md) is an experimental successor to C++. It is an
effort to explore a possible future direction for the C++ language given the
[difficulties improving C++](difficulties_improving_cpp.md).

## What is Carbon's status?

Carbon is still an experiment. There remain significant open questions that we
need to answer before the project can consider becoming a production effort. For
now, we're focused on exploring this direction and gaining information to begin
answering these questions.

### How soon can we use Carbon?

Carbon is still years away — even if the experiment succeeds, it's unlikely that
it will be ready for serious or production use in the next few years. Everything
here is part of a long-term investigation.

### Why make Carbon public while it's still an experiment?

One of the critical questions we need to answer as part of this experiment is
whether the direction we're exploring with Carbon has both broad and significant
interest for the industry at large. We feel like this is best answered by
developing the language openly, publicly, and with broad participation.

### How complete is Carbon's design?

We've resolved several of the most challenging language design technical
decisions we anticipated based on experience with C++ and its constraints,
particularly around generics and inheritance. Beyond those two areas, we have
initial designs for class types, inheritance, operator overloading, syntactic
and lexical structure, and modular code organization. We are aiming to complete
the initial 0.1 language design around the end of 2022 although there are a
large number of variables in that timeline. See our [roadmap](roadmap.md) for
details.

References:

-   [Carbon design overview](/docs/design/README.md)
-   [How do Carbon generics differ from templates?](#how-do-carbon-generics-differ-from-templates)
-   [Roadmap](roadmap.md)

### How many people are involved in Carbon?

Prior to going public, Carbon has had a couple dozen people involved.
[GitHub Insights](https://github.com/carbon-language/carbon-lang/pulse/monthly)
provides activity metrics.

### Is there a demo?

Yes! A prototype interpreter demo `explorer` can be used to execute simple
examples. For example:

```
$ bazel run //explorer -- ./explorer/testdata/basic_syntax/print.carbon
```

Example source files can be found under
[/explorer/testdata](/explorer/testdata).

We're also working on making Carbon available on
[https://compiler-explorer.com](https://compiler-explorer.com/).

## Why build Carbon?

See the [project README](#why-build-carbon) for an overview of the motivation
for Carbon. This section dives into specific questions in that space.

### Why is performance critical?

Performance is critical for many users today. A few reasons are:

-   **Cost savings**: Organizations with large-scale compute needs
    [care about software performance](https://www.microsoft.com/en-us/research/publication/theres-plenty-of-room-at-the-top/)
    because it reduces hardware needs.
-   **Reliable latency**: Environments with specific latency needs or
    [concerns with bounding tail latency](https://research.google/pubs/pub40801/)
    need to be able to control and improve their latency.
-   **Resource constraints**: Many systems have constrained CPU or memory
    resources that require precise control over resource usage and performance.

### What level of C++ interoperability is expected?

Carbon code will be able to call C++, and the other way around, without
overhead. You will be able to migrate a single library to Carbon within a C++
application, or write new Carbon on top of their existing C++ investment.

While Carbon's interoperability may not cover every last case, most C++ style
guides (such as the C++ Core Guidelines or Google C++ Style Guide) steer
developers away from complex C++ code that's more likely to cause issues, and we
expect the vast majority of code to interoperate well.

For example, considering a pure C++ application:

<a href="/docs/images/snippets.md#c">
<!--
Edit snippet in /docs/images/snippets.md and:
https://drive.google.com/corp/drive/folders/1CsbHo3vamrxmBwHkoyz1kU0sGFqAh688
-->
<img src="/docs/images/cpp_snippet.svg" width="600"
     alt="A snippet of C++ code. Follow the link to read it.">
</a>

It's possible to migrate a single function to Carbon:

<a href="/docs/images/snippets.md#mixed">
<!--
Edit snippet in /docs/images/snippets.md and:
https://drive.google.com/corp/drive/folders/1CsbHo3vamrxmBwHkoyz1kU0sGFqAh688
-->
<img src="/docs/images/mixed_snippet.svg" width="600"
     alt="A snippet of mixed Carbon and C++ code. Follow the link to read it.">
</a>

References:

-   [Interoperability philosophy and goals](/docs/design/interoperability/philosophy_and_goals.md)
-   [How will Carbon's bidirectional C++ interoperability work?](#how-will-carbons-bidirectional-c-interoperability-work)

### What would migrating C++ code to Carbon look like?

Migration support is a
[key long-term goal for Carbon](goals.md#interoperability-with-and-migration-from-existing-c-code).

If a migration occurs, we anticipate:

-   Migration tools that automatically translate C++ libraries to Carbon at the
    file or library level with minimal human assistance.
-   Bidirectional C++ interoperability that allows teams to migrate libraries in
    any order they choose without performance concerns or maintaining
    interoperability wrappers.
-   Test-driven verification that migrations are correct.

## What alternatives did you consider? Why did they not work?

### Why not improve C++?

A lot of effort has been invested into improving C++, but
[C++ is difficult to improve](difficulties_improving_cpp.md).

For example, although [P2137](https://wg21.link/p2137r0) was not accepted, it
formed the basis for [Carbon's goals](goals.md).

### Why not fork C++?

While we would like to see C++ improve, we don't think that forking C++ is the
right path to achieving that goal. A fork could create confusion about what code
works with standard C++. We believe a _successor_ programming language is a
better approach because it gives more freedom for Carbon's design while
retaining the existing C++ ecosystem investments.

### Why not Rust?

TODO: WIP (should add before merging)

### Why not a garbage collected language, like Java, Kotlin, or Go?

If you can use one of these languages, you absolutely should.

Garbage collection provides dramatically simpler memory management for
developers, but at the expense of performance. The performance cost can range
from direct runtime overhead to significant complexity and loss of _control_
over performance. This trade-off makes sense for many applications, and we
actively encourage using these languages in those cases. However, we need a
solution for C++ use-cases that require its full performance, low-level control,
and access to hardware.

## How will Carbon work?

### What compiler infrastructure is Carbon using?

Carbon is being built using LLVM, and is expected to have Clang dependencies for
[interoperability](#how-will-carbons-bidirectional-c-interoperability-work).

### How will Carbon's bidirectional C++ interoperability work?

The Carbon toolchain will compile both Carbon and C++ code together, in order to
make the interoperability
[seamless](#what-level-of-c-interoperability-is-expected).

For example, for `import Cpp library "<vector>"`, Carbon will:

-   Call into Clang to load the AST of the `vector` header file.
-   Analyze the AST for public APIs, which will be turned into names that can be
    accessed from Carbon; for example, `std::vector` is `Cpp.std.vector` in
    Carbon.
-   Use Clang to instantiate the `Cpp.std.vector` template when parameterized
    references occur in Carbon code.
    -   In other words, C++ templates will be instantiated using standard C++
        mechanisms, and the instantiated versions are called by Carbon code.

Some code, such as `#define` preprocessor macros, will not work as well. C++
allows arbitrary content in a `#define`, and that can be difficult to translate.
As a consequence, this is likely to be a limitation of interoperability and left
to migration.

### How do Carbon generics differ from templates?

Carbon's
[generic programming](https://en.wikipedia.org/wiki/Generic_programming) support
will handle both templates (matching C++) and checked generics (common in other
languages: Rust, Swift, Go, Kotlin, Java, and so on).

The key difference between the two is that template arguments can only finish
type-checking _during_ instantiation, whereas generics specify an interface with
which arguments can finish type-checking _without_ instantiation. This has a
couple important benefits:

-   Type-checking errors for generics happen earlier, making it easier for the
    compiler to produce helpful diagnostics.
-   Generic functions can generate less compiled output, allowing compilation
    with many uses to be faster.
    -   For comparison, template instantiations are a major factor for C++
        compilation latency.

Although Carbon will prefer generics over templates, templates are provided for
migration of C++ code.

References:

-   [Generics: Goals: Better compiler experience](/docs/design/generics/goals.md#better-compiler-experience)
-   [Generics: Terminology: Generic versus template parameters](/docs/design/generics/terminology.md#generic-versus-template-parameters)

### What is Carbon's memory model?

Carbon will match C++'s memory model closely in order to maintain zero-overhead
interoperability. There may be some changes made as part of supporting memory
safety, but performance and interoperability will constrain flexibility in this
space.

### How will Carbon achieve memory safety?

See [memory safety in the project README](/#memory-safety).

References:

-   [Lifetime annotations for C++](https://discourse.llvm.org/t/rfc-lifetime-annotations-for-c/61377)
-   [Carbon principle: Safety strategy](principles/safety_strategy.md)

## How will the Carbon _project_ work?

### Where does development occur?

Carbon is using GitHub for its repository and code reviews. Most non-review
discussion occurs on our Discord server (TODO: link).

If you're interested in contributing, you can find more information in our
[Contributing file](/CONTRIBUTING.md).

### How does Carbon make decisions?

Any interested developer may [propose and discuss changes](evolution.md) to
Carbon. The [Carbon leads](groups.md#carbon-leads) are responsible for reviewing
proposals and surrounding discussion, then making decisions based on the
discussion. As Carbon grows, we expect to add feature teams to distribute
responsibility.

The intent of this setup is that Carbon remains a community-driven project,
avoiding situations where any single organization controls Carbon's direction.

References:

-   [Contributing](/CONTRIBUTING.md)
-   [Evolution process](evolution.md)

### What happens when a decision was wrong?

Carbon's [evolution process](evolution.md) is iterative: when we make poor
decisions, we'll work to fix them. If we realize a mistake quickly, it may make
sense to just roll back the decision. Otherwise, a fix will need to follow the
normal evolution process, with a proposal explaining why the decision was wrong
and proposing a better path forward.

### What license does Carbon use?

Carbon is under the [Apache License v2.0 with LLVM Exceptions](/LICENSE). We
want Carbon to be available under a permissive open source license. As a
programming language with compiler and runtime library considerations, our
project has the same core needs as the LLVM project for its license and we build
on their work to address these by combining the
[Apache License](https://spdx.org/licenses/Apache-2.0.html) with the
[LLVM Exceptions](https://spdx.org/licenses/LLVM-exception.html).

### Why make Carbon open source?

We believe it is important for a programming language like Carbon, if it is
successful, to be developed by and for a broad community. We feel that the open
source model is the most effective and successful approach for doing this. We're
closely modeled on LLVM and other similar open source projects, and want to
follow their good examples. We've structured the project to be attractive for
industry players big and small to participate in, but also to be resilient and
independent long-term.

The open source model, particularly as followed by Apache and LLVM, also
provides a strong foundation for handling hard problems like intellectual
property and licensing with a broad and diverse group of contributors.

### Why does Carbon have a CLA?

Carbon [uses a CLA](/CONTRIBUTING.md#contributor-license-agreements-clas)
(Contributor License Agreement) in case we need to fix issues with the license
structure in the future, something which has proven to be important in other
projects.

Any changes to the license of Carbon would be made very carefully and subject to
the exact same decision making process as any other change to the overall
project direction.

Initially, Carbon is bootstrapping using Google's CLA. We are planning to create
an open source foundation and transfer all Carbon-related rights to it; our goal
is for the foundation setup to be similar to other open source projects, such as
LLVM or Kubernetes.

### Who pays for Carbon's infrastructure?

Carbon is currently bootstrapping infrastructure with the help of Google. As
soon as a foundation is ready to oversee infrastructure, such as
[continuous integration](https://en.wikipedia.org/wiki/Continuous_integration)
and the CLA, they'll be transferred and run by the community in an open way.
