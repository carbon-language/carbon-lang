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
    -   [Where should I ask questions about Carbon Language?](#where-should-i-ask-questions-about-carbon-language)
    -   [Why isn't there a Carbon Language logo?](#why-isnt-there-a-carbon-language-logo)
-   [Why build Carbon?](#why-build-carbon)
    -   [Why is performance critical?](#why-is-performance-critical)
    -   [What level of C++ interoperability is expected?](#what-level-of-c-interoperability-is-expected)
    -   [What would migrating C++ code to Carbon look like?](#what-would-migrating-c-code-to-carbon-look-like)
-   [What alternatives did you consider? Why did they not work?](#what-alternatives-did-you-consider-why-did-they-not-work)
    -   [Why not improve C++?](#why-not-improve-c)
    -   [Why not fork C++?](#why-not-fork-c)
    -   [Why not Rust?](#why-not-rust)
        -   [If you can use Rust, ignore Carbon](#if-you-can-use-rust-ignore-carbon)
        -   [Why is adopting Rust difficult for C++ codebases?](#why-is-adopting-rust-difficult-for-c-codebases)
    -   [Why not a garbage collected language, like Java, Kotlin, or Go?](#why-not-a-garbage-collected-language-like-java-kotlin-or-go)
-   [How were specific feature designs chosen?](#how-were-specific-feature-designs-chosen)
    -   [Why aren't `<` and `>` used as delimiters?](#why-arent--and--used-as-delimiters)
    -   [Why do variable declarations have to start with `var` or `let`?](#why-do-variable-declarations-have-to-start-with-var-or-let)
    -   [Why do variable declarations have to have types?](#why-do-variable-declarations-have-to-have-types)
-   [How will Carbon work?](#how-will-carbon-work)
    -   [What compiler infrastructure is Carbon using?](#what-compiler-infrastructure-is-carbon-using)
    -   [How will Carbon's bidirectional C++ interoperability work?](#how-will-carbons-bidirectional-c-interoperability-work)
    -   [How do Carbon generics differ from templates?](#how-do-carbon-generics-differ-from-templates)
    -   [What is Carbon's memory model?](#what-is-carbons-memory-model)
    -   [How will Carbon achieve memory safety?](#how-will-carbon-achieve-memory-safety)
-   [How will the Carbon _project_ work?](#how-will-the-carbon-project-work)
    -   [Where does development occur?](#where-does-development-occur)
    -   [How does Carbon make decisions?](#how-does-carbon-make-decisions)
    -   [What happens when a decision was wrong?](#what-happens-when-a-decision-was-wrong)
    -   [What license does Carbon use?](#what-license-does-carbon-use)
    -   [Why make Carbon open source?](#why-make-carbon-open-source)
    -   [Why does Carbon have a CLA?](#why-does-carbon-have-a-cla)
    -   [Who pays for Carbon's infrastructure?](#who-pays-for-carbons-infrastructure)
-   [How can I contribute to Carbon?](#how-can-i-contribute-to-carbon)
    -   [What are the prerequisites for contributing to Carbon Language's design and tools?](#what-are-the-prerequisites-for-contributing-to-carbon-languages-design-and-tools)
    -   [When do we revisit decisions or reopen discussions?](#when-do-we-revisit-decisions-or-reopen-discussions)
    -   [What can I do if I disagree with a design decision?](#what-can-i-do-if-i-disagree-with-a-design-decision)
    -   [How can I best say "I like X" or "I don't like X"?](#how-can-i-best-say-i-like-x-or-i-dont-like-x)

<!-- tocstop -->

## What is Carbon?

The [Carbon Language](/README.md) is an experimental successor to C++. It is an
effort to explore a possible future direction for the C++ language given the
[difficulties improving C++](difficulties_improving_cpp.md).

## What is Carbon's status?

[Carbon is still an experiment.](/README.md#project-status) There remain
significant open questions that we need to answer before the project can
consider becoming a production effort. For now, we're focused on exploring this
direction and gaining information to begin answering these questions.

-   [Project status](/README.md#project-status)
-   [Roadmap](roadmap.md)

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
$ bazel run //explorer -- ./explorer/testdata/print/format_only.carbon
```

Example source files can be found under
[/explorer/testdata](/explorer/testdata).

Carbon can also be explored interactively on
[https://carbon.compiler-explorer.com](https://carbon.compiler-explorer.com/).

### Where should I ask questions about Carbon Language?

Please ask questions and hold discussions either by using
[GitHub Discussions](https://github.com/carbon-language/carbon-lang/discussions)
or
[#language-questions on Discord](https://discord.com/channels/655572317891461132/998959756045713438).

GitHub Issues should be reserved for missing features, bugs, and anything else
that is fixable by way of a Pull Request.

### Why isn't there a Carbon Language logo?

Establishing a Carbon Language logo isn't a priority right now. Remember that
this project is an _experiment_, and so we think it's best to concentrate
efforts on ensuring that the language succeeds at its goals instead.

We have a few drafts in the works, but it requires a fair amount of work to get
right, and getting it wrong is costly, so we won't be adding one in the near
future. Don't suggest logos, because we need to be careful about how we create
one.

## Why build Carbon?

See the [project README](/README.md#why-build-carbon) for an overview of the
motivation for Carbon. This section dives into specific questions in that space.

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
application, or write new Carbon on top of your existing C++ investment.

While Carbon's interoperability may not cover every last case, most C++ style
guides (such as the C++ Core Guidelines or Google C++ Style Guide) steer
developers away from complex C++ code that's more likely to cause issues, and we
expect the vast majority of code to interoperate well.

For example, considering a pure C++ application:

<a href="/docs/images/snippets.md#c">
<!--
Edit snippet in /docs/images/snippets.md and:
https://drive.google.com/drive/folders/1QrBXiy_X74YsOueeC0IYlgyolWIhvusB
-->
<img src="/docs/images/cpp_snippet.svg" width="600"
     alt="A snippet of C++ code. Follow the link to read it.">
</a>

It's possible to migrate a single function to Carbon:

<a href="/docs/images/snippets.md#mixed">
<!--
Edit snippet in /docs/images/snippets.md and:
https://drive.google.com/drive/folders/1QrBXiy_X74YsOueeC0IYlgyolWIhvusB
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

#### If you can use Rust, ignore Carbon

If you want to use Rust, and it is technically and economically viable for your
project, you should use Rust. In fact, if you can use Rust or any other
established programming language, you should. Carbon is for organizations and
projects that heavily depend on C++; for example, projects that have a lot of
C++ code or use many third-party C++ libraries.

We believe that Rust is an excellent choice for writing software within the pure
Rust ecosystem. Software written in Rust has properties that neither C++ nor
Carbon have. When you need to call other languages from Rust, RPCs are a good
option. Rust is also good for using APIs implemented in a different language
in-process, when the cost of maintaining the FFI boundary is reasonable.

When the foreign language API is large, constantly changes, uses advanced C++
features, or
[makes architectural choices that are incompatible with safe Rust](#why-is-adopting-rust-difficult-for-c-codebases),
maintaining a C++/Rust FFI may not be economically viable today (but it is an
area of active research: [cxx](https://crates.io/crates/cxx),
[autocxx](https://crates.io/crates/autocxx),
[Crubit](https://github.com/google/crubit/blob/main/docs/design.md)).

The Carbon community is looking for a language that existing, large, monolithic
C++ codebases can incrementally adopt and have a prospect of migrating away from
C++ completely. We would be very happy if Rust could be this language. However,
we are not certain that:

-   Idiomatic, safe Rust can seamlessly integrate into an existing C++ codebase,
    similarly to how TypeScript code can be added to a large existing JavaScript
    codebase.
-   Developers can incrementally migrate existing C++ code to Rust, just like
    they can migrate JavaScript to TypeScript one file at a time, while keeping
    the project working.

See
[Carbon's goals](/docs/project/goals.md#interoperability-with-and-migration-from-existing-c-code)
for an in-depth discussion of Carbon's vision for C++/Carbon interop and
migration.

#### Why is adopting Rust difficult for C++ codebases?

Large existing C++ codebases almost certainly made architectural choices that
are incompatible with safe Rust. Specifically:

-   Seamless interop where existing, unmodified **C++ APIs are made callable
    from safe Rust** requires the C++ code to follow borrow checking rules at
    the API boundary.
    -   To reduce the amount of Rust-side compile-time checking that makes
        interop difficult, C++ APIs can be exposed to Rust with pointers instead
        of references. However, that forces users to write _unsafe_ Rust, which
        can be even more tricky to write than C++ because it has new kinds of UB
        compared to C++; for example,
        [stacked borrows violations](https://github.com/rust-lang/unsafe-code-guidelines/blob/master/wip/stacked-borrows.md).
-   Seamless interop where **safe Rust APIs are made callable from C++**
    requires C++ users to follow Rust borrow checking rules.
-   **Incremental migration of C++ to safe Rust** means that C++ code gets
    converted to Rust without major changes to the architecture, data
    structures, or APIs. However Rust imposes stricter rules than C++,
    disallowing some design choices that were valid in C++. Therefore, the
    original C++ code must follow Rust rules before attempting a conversion.
    -   Original C++ code must be structured in such a way that the resulting
        Rust code passes borrow checking. C++ APIs and data structures are not
        designed with this in mind.
    -   Migrating C++ to _unsafe_ Rust would still require the code to follow
        Rust's
        [reference exclusivity](https://doc.rust-lang.org/book/ch04-02-references-and-borrowing.html#the-rules-of-references)
        and stacked borrows rules.

### Why not a garbage collected language, like Java, Kotlin, or Go?

If you can use one of these languages, you absolutely should.

Garbage collection provides dramatically simpler memory management for
developers, but at the expense of performance. The performance cost can range
from direct runtime overhead to significant complexity and loss of _control_
over performance. This trade-off makes sense for many applications, and we
actively encourage using these languages in those cases. However, we need a
solution for C++ use-cases that require its full performance, low-level control,
and access to hardware.

## How were specific feature designs chosen?

Throughout the design, we include 'Alternatives considered' and 'References'
sections which can be used to research the decision process for a particular
design.

### Why aren't `<` and `>` used as delimiters?

[One of our goals for Carbon](/docs/project/goals.md#fast-and-scalable-development)
is that it should support parsing without contextual or semantic information,
and experience with C++ has shown that using `<` as both a binary operator and
an opening delimiter makes that goal difficult to achieve.

For example, in C++, the expression `a<b>(c)` could parse as either a function
call with a template argument `b` and an ordinary argument `c`, or as a chained
comparison `(a < b) > (c)`. In order to resolve the ambiguity, the compiler has
to perform name lookup on `a` to determine whether there's a function named `a`
in scope.

It's also worth noting that Carbon
[doesn't use _any_ kind of brackets](https://github.com/carbon-language/carbon-lang/blob/trunk/docs/design/README.md#checked-and-template-parameters)
to mark template or generic parameters, so if Carbon had angle brackets, they
would mean something different than they do in C++, which could cause confusion.
We do use square brackets to mark _deduced_ parameters, as in:

```
fn Sort[T:! Comparable](a: Vector(T)*)
```

But deduced parameters aren't the same thing as template parameters. In
particular, deduced parameters are never mentioned at the callsite, so those
square brackets are never part of the expression syntax.

See [Proposal #676: `:!` generic syntax](/proposals/p0676.md) for more
background on how and why we chose our current generics syntax.

### Why do variable declarations have to start with `var` or `let`?

In Carbon, a declaration of a single variable looks like this:

```
var the_answer: i32 = 42;
```

But this is just the most common case. The syntax between `var` and `=` can be
any [irrefutable pattern](/docs/design/README.md#patterns), not just a single
variable binding. For example:

```
var ((x: i32, _: i32), y: auto) = ((1, 2), (3, 4));
```

This code is valid, and initializes `x` to `1` and `y` to `(3, 4)`. In the
future, we will probably also support destructuring structs in a similar way,
and many other kinds of patterns are possible.

Now consider how that example would look if the `var` token were not required:

```
((x: i32, _: i32), y: auto) = ((1, 2), (3, 4));
```

With this example, the parser would need to look four tokens ahead to determine
that it's parsing a variable declaration rather than an expression. With more
deeply-nested patterns, it would have to look ahead farther. Avoiding this sort
of unbounded lookahead is an important part of our
[fast and scalable development](/docs/project/goals.md#fast-and-scalable-development)
goal.

### Why do variable declarations have to have types?

As discussed above, Carbon variable declarations are actually doing a form of
pattern matching. In a declaration like this:

```
var the_answer: i32 = 42;
```

`the_answer: i32` is an example of a _binding pattern_, which matches any value
of the appropriate type, and binds the given name to it. The `: i32` can't be
omitted, because `the_answer` on its own is an expression, and any Carbon
expression is also a valid pattern, which matches if the value being matched is
equal to the value of the expression. So `var the_answer = 42;` would try to
match `42` with the value of the expression `the_answer`, which requires a
variable named `the_answer` to already exist.

There are other ways of approaching pattern matching, but there are tradeoffs.
Pattern matching is still on a provisional design, and as of August 2022 it
hasn't been fully reviewed with alternatives considered. A future proposal for
pattern matching will need to weigh the tradeoffs in more detail, and may come
to a different decision.

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
couple of important benefits:

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

See [memory safety in the project README](/README.md#memory-safety).

References:

-   [Lifetime annotations for C++](https://discourse.llvm.org/t/rfc-lifetime-annotations-for-c/61377)
-   [Carbon principle: Safety strategy](principles/safety_strategy.md)

## How will the Carbon _project_ work?

### Where does development occur?

Carbon is using GitHub for its repository and code reviews. Most non-review
discussion occurs on our [Discord server](https://discord.gg/ZjVdShJDAs).

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
and the CLA, we plan to transfer them so they are run by the community.

## How can I contribute to Carbon?

There are many ways to contribute, and we appreciate all of them. Begin by
reading the project's [Contributing](/CONTRIBUTING.md) page to learn more about
how you can contribute.

### What are the prerequisites for contributing to Carbon Language's design and tools?

**[Carbon Language isn't ready for use](#how-soon-can-we-use-carbon). This
section is for people wishing to participate in designing and implementing the
language.**

Carbon is being designed to migrate C++ codebases and to look familiar to C++
programmers. As such, familiarity with C++ is very important. Carbon is also
trying to learn from other programming languages, so having broad experience
with other programming languages will be helpful to see tradeoffs in design
decisions.

The Carbon toolchain is being implemented in C++, and we also use Python and
Starlark. As we're building off of the LLVM project, familiarity with Clang and
other parts of LLVM will be advantageous, but not required.

Our [contribution tools](/docs/project/contribution_tools.md) page documents
specific tools we use when building.

### When do we revisit decisions or reopen discussions?

Once a decision is made through the [evolution process](evolution.md) the
community should treat it as _firmly_ decided. This doesn't mean that the
decision is _definitely_ right or set in stone, it just means we'd like the
community to focus time and energy on other issues in the name of progress.

Sometimes, it will be appropriate to revisit a decision; for example, when there
is new information introduced, a new community joins Carbon, or there is an
order of magnitude growth in the community. These cases are handled as new
proposals through the [evolution process](evolution.md).

For example, we have done this with digit separators: we missed important
_domain_ conventions and had overly restricted where separators are allowed,
[an issue was filed with the new information](https://github.com/carbon-language/carbon-lang/issues/1485),
and we're fixing the choice.

See also the related questions
[What happens when a decision was wrong?](#what-happens-when-a-decision-was-wrong),
[How does Carbon make decisions?](#how-does-carbon-make-decisions), and
[What can I do if I disagree with a design decision?](#what-can-i-do-if-i-disagree-with-a-design-decision).

### What can I do if I disagree with a design decision?

We invite you to give us constructive feedback. Some of Carbon's design
decisions are made with the _expectation_ of receiving community feedback. We
understand that many decisions won't be universally popular, but we'd still like
to understand the community's reaction to Carbon.

We encourage you to investigate why Carbon came to be the way it is. Designs
will include links to the proposals and important alternatives considered that
led to them, typically linked at the bottom. Read through and understand the
context and rationale behind the decisions you are concerned about. You may find
that your concerns were already thoroughly discussed. If not, you will be in a
better place to present your thoughts in a convincing way.

Changing decisions that have come out of the [evolution process](evolution.md)
involves a formal process. See
[When do we revisit decisions or reopen discussions?](#when-do-we-revisit-decisions-or-reopen-discussions).
For these issues in particular, please be aware that other community members may
choose to not actively engage in detailed discussions, especially if the
discussion seems to be revisiting points made in the past.

If after reading this answer you are not sure how to proceed please feel free to
ask (see
[Where should I ask questions?](#where-should-i-ask-questions-about-carbon-language)).

### How can I best say "I like X" or "I don't like X"?

Both Discord and GitHub Discussions allow you to give an emoji "reaction" to
individual posts. If you'd like to amplify what has already been said, please
use these instead of posting messages that re-state substantially the same
thing. These make conversations easier to follow and understand general
sentiment in discussions involving many people.
