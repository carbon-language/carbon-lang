# Carbon: Goals

<!--
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

## Overview

Carbon is an experiment to explore a possible, distant future for the C++
programming language designed around a specific set of goals, priorities, and
use cases.

A programming language is a tool, and different tools are good for different
purposes. We think there is great value in priorities that differentiate Carbon
from the plethora of programming languages, rather than following the crowd and
regressing to the mean. Stating Carbon’s priorities clearly and explicitly helps
the entire community effectively evaluate and use the language.

Carbon's language goals have historically been best addressed by C++, and there
are large ecosystems and codebases written using C++ to these ends. We intend
Carbon to work with these existing C++ library ecosystems and codebases instead
of starting from scratch. Further, Carbon's success depends on convincing C++
developers to adopt Carbon, so the migration story is another key concern.
High-performance, bidirectional interoperability will benefit both of these
goals.

## Project goals

### Community and culture

Carbon has an overarching goal of promoting a healthy and vibrant community with
an inclusive, welcoming, and pragmatic culture. While this may not directly
affect Carbon's design, it affects how Carbon's design occurs. We cannot build a
good language without a good community. As the saying goes,
["culture eats strategy for breakfast"](https://techcrunch.com/2014/04/12/culture-eats-strategy-for-breakfast/).
Carbon's community, including both maintainers and users, needs to last for
years and be capable of scaling up. It needs to support people working on Carbon
across a wide range of companies as their full time job, but also people
contributing in small fractions of their time, or as students, teachers, or as a
hobby. There are several key ingredients to achieving this.

**The community and project needs a code of conduct.** We want Carbon's
community to be welcoming and respectful, with a deep commitment to
psychological safety. We need consistent expectations for how every community
member should behave, regardless of their position in the community. These
expectations around conduct and behavior need to be clearly articulated both to
set expectations for people joining, and to help remind and anchor us on
consistent standards. It is also important that we hold ourselves accountable to
these expectations and have real and meaningful mechanisms to moderate the
community. When behavior steps outside of our expectations, we need tools,
process, and policy for how we will recognize and correct it.

**An open, inclusive process for Carbon changes.** The community needs to be
able to effectively engage in the direction and evolution of the project and
language while the process remains efficient and effective. That means we need
an open, inclusive process where everyone feels comfortable participating.
Community members should understand how and why decisions are made, and have the
ability to both influence them before they occur and give feedback afterward. We
want to use this process to also ensure we stick to our language priorities and
have clear rationale for all of our technical designs and decisions.

**Being inclusive is different from including everyone.** We want to avoid
excluding or marginalizing members of the community. However, we expect to
inevitably make choices that benefit some Carbon community members more than
others. We will provide justification for these decisions, but achieving
Carbon's goals -- including that of a healthy community -- will be the guiding
rule.

### Language tools and ecosystem

Programming languages do not succeed in a vacuum. The Carbon project cannot
merely _design_ a language in order to succeed, it must tackle the full
ecosystem of tooling that makes programmers effective using the language. This
includes both necessary tools, such as a compiler and standard library, as well
as a broad range of other tools that enable programmers to be more effective,
efficient, or productive. There are several key examples of key tools or
ecosystem components that we have as concrete goals for the Carbon project.

**Reference implementation:** A working implementation is a key component of the
language tools that the Carbon project will provide. This helps the language
have a strong and consistent experience for users and a clear onboarding
process. It also enables us to carefully consider implementation considerations
throughout the design of the language. However, we do _not_ want this to be seen
as a replacement for a formal specification at any point.

**Formal specification:** Fully specifying the language enables other
implementations and allows us to clearly document the expected behavior of the
reference implementation. We also want to include machine readable forms of this
for many parts of the language, such as a grammar, to help enable widespread
tooling to reuse a consistent basis and help us check invariants and properties
of the language.

**Adoption tooling:** We want to provide a compelling suite of tools
out-of-the-box in order to encourage adoption of Carbon at scale where it can
augment existing C++ codebases. For example, we expect a code translator will be
important. We expect this to largely build on top of automation and tooling.

**Upgrade migration tooling:** As Carbon evolves over time, we expect to provide
tooling to help automate and scale migrating existing Carbon code to the new
version. The goal is to enable more rapid evolution of the language without the
churn tax and version skew becoming unsustainable.

**Developer tooling:** We need programmers to be productive reading and writing
Carbon code. We expect to provide a broad suite of development oriented tools
ranging from refactoring tools to [LSP](https://langserver.org/) implementations
and editor integrations.

**Library ecosystem:** We expect to also provide infrastructure to enable
package management and other ecosystem needs. The goal is to support what the
ecosystem needs, regardless of the exact form this ends up taking.

## Language goals and priorities

We believe Carbon must support:

1.  Performance-critical software
2.  Both software and language evolution
3.  Code that is easy to read, understand, and write
4.  Practical safety guarantees and testing mechanisms
5.  Fast and scalable development
6.  Current hardware architectures, OS platforms, and environments as they
    evolve
7.  Interoperability with and migration from existing C++ code

The first six of these represent our fundamental goals for the software we want
to implement in Carbon. However, we cannot simply replace all of the existing
C++ software and programmers. Carbon must be reachable from where we are, which
motivates the seventh goal around interoperability and migration.

These are expected to be largely independent goals. We cannot give up any one of
them, regardless of priority, without significant harm to our use cases. This
means we expect to design the language largely in ways that enable all of these
goals. However, the priority ordering provides guidance on how to manage the
conflicts that do arise between goals. When we can tailor the design of language
features to balance between different goals, we will strive to align that
balance with this prioritization. When we are unable to find a compromise that
enables both goals and we have a fundamental conflict or binary choice, we will
weigh the factors going into that choice based on how they interact with this
ranking.

### Goals in detail

Each goal is broad, and has several facets to consider when making decisions.
Below, we discuss all of these goals in more detail to give a deeper
understanding of both the nature and motivation of these goals.

#### Performance-critical software

All software consumes resources: time, memory, compute, power, binary size, and
so on. In many cases, raw resource usage is not the biggest concern. Instead,
algorithmic efficiency or business logic dominates these concerns. However,
there exists software where the rate of resource consumption—its performance—is
critical to its successful operation. Another way to think about when
performance is critical: would a performance regression be considered a bug for
users? Would it even be noticed?

Our goal is to support software where its performance with respect to some set
of resource constraints is critical to its successful operation. This
overarching goal can be decomposed into a few specific aspects.

**Provide the programmer control over every aspect of performance.** When faced
with some performance problem, the programmer should always have tools within
Carbon to address it. This does not mean that the programmer is necessarily
concerned with ultimate performance at every moment, but in the most constrained
scenarios they must be able to "open up the hood" without switching to another
language.

**Code should perform predictably.** The reader and writer of code should be
able to easily understand its expected performance, given sufficient background
knowledge of the environment in which it will run. This need not be precise, but
instead can use heuristics and guidelines to avoid surprise. The key priority is
that performance, whether good or bad, is unsurprising to users of the Carbon
language. Even pleasant surprises, when too frequent, can become a problem due
to establishing brittle baseline performance that cannot be reliably sustained.

**Leave no room for a lower level language.** Whether to gain control over
performance problems or to gain access to hardware facilities, programmers
should not need to leave the rules and structure of Carbon.

#### Both software and language evolution

Titus Winters writes in "Non-Atomic Refactoring and Software Sustainability":

> What is the difference between programming and software engineering? These are
> nebulous concepts and thus there are many possible answers, but my favorite
> definition is this: Software engineering is programming integrated over time.
> All of the hard parts of engineering come from dealing with time:
> compatibility over time, dealing with changes to underlying infrastructure and
> dependencies, and working with legacy code or data. Fundamentally, it is a
> different task to produce a programming solution to a problem (that solves the
> current [instance] of the problem) vs. an engineering solution (that solves
> current instances, future instances that we can predict, and - through
> flexibility - allows updates to solve future instances we may not be able to
> predict).

From this definition of "software engineering" vs. "programming" we suggest that
Carbon should prioritize being more of a "software engineering" language, and
less of a "programming" language. We specifically are interested in dealing with
the time-oriented aspects of software built in this language. We need to be
prepared for substantive changes in priority over the next decade, on par with
the changes experienced in the 2010s: 10x scaling of engineering organizations,
mobile, cloud, diversification of platforms and architectures, and so on.

**Support maintaining and evolving software written in Carbon for decades.** The
life expectancy of some software will be long and the software will not be
static or unchanging in that time. Mistakes will be made and need to be
corrected. New functionality will be introduced and old functionality retired
and removed. The design of Carbon must support and ease every step of this
process. This ranges from emphasizing testing and continuous integration to
tooling and the ability to make non-atomic changes. It also includes constraints
on the design of Carbon itself: we should avoid, or at least minimize, language
features that encourage unchangeable constructs. For example, any feature with a
contract that cannot be strengthened or weakened without breaking the expected
usage patterns is inherently hostile to refactoring. Analogously, features or
conventions that require simultaneously updating all users of an API when
extending it are inherently hostile towards long-term maintenance of software.

**Support maintaining and evolving the language itself for decades.** We will
not get the design of most language features correct on our first, second, or
73rd try. As a consequence, there must be a built-in plan and ability to move
Carbon forward at a reasonable pace and with a reasonable cost. Simultaneously,
an evolving language must not leave software behind to languish, but bring
software forward. This requirement should not imply compatibility, but instead
some migratability, likely tool-assisted.

**Be mindful of legacy.** Globally, there may be as many as 50 billion lines of
C++ code. Any evolution of Carbon that fails to account for human
investment/training and legacy code, representing significant capital, is doomed
from the start. Note that our priority is restricted to legacy source code; we
do not prioritize full support of legacy object code. While that still leaves
many options open, such as dedicated and potentially slower features, it does
limit the degree to which legacy use cases beyond source code should shape the
Carbon design.

#### Code that is easy to read, understand, and write

While this is perhaps the least unique among programming languages of the goals
we list here, we feel it is important to state it, explain all of what we mean
by this, and fit it into our prioritization scheme.

Software, especially at scale and over time, already imposes a burden on
engineers due to its complexity. Carbon should strive for simplicity to reduce
the complexity burden on reading, understanding, and writing code. The behavior
of code should be easily understood, especially by those unfamiliar with the
software system. Consider engineers attempting to diagnose a serious outage
under time pressure—every second spent trying to understand the _language_ is
one not spent understanding the _problem_.

While the source code of our software may be read far more often by machines,
humans are the most expensive readers and writers of software. As a consequence,
we need to optimize for human reading, understanding, and writing of software,
in that order.

**Excellent ergonomics.** Human capabilities and limitations in the domains of
perception, memory, reasoning, and decision-making affect interactions between
humans and systems. Ergonomic language design takes human factors into account
to increase productivity and comfort, reduces errors and fatigue, making Carbon
more suitable for humans to use. We can also say that ergonomic designs are
accessible to humans. "Readability" is a related, but a more focused concept,
connected to only the process of reading code. "Ergonomics" covers all
activities where humans interact with Carbon: reading, writing, designing,
discussing, reviewing, and refactoring code, as well as learning and teaching
Carbon. A few examples:

- Carbon should not use symbols that are difficult to type, see, or
  differentiate from similar symbols in commonly used contexts.
- Syntax should be easily parsed and scanned by any human in any development
  environment, not just a machine or a human aided by semantic hints from an
  IDE.
- Code with similar behavior should use similar syntax, and code with different
  behavior should use different syntax. Behavior in this context should include
  both the functionality and performance of the code. This is part of conceptual
  integrity.
- Explicitness must be balanced against conciseness, as verbosity and ceremony
  add cognitive overhead for the reader, while explicitness reduces the amount
  of outside context the reader must have or assume.
- Common yet complex tasks, such as parallel code, should be well-supported in
  ways that are easy to reason about.
- Ordinary tasks should not require extraordinary care, because humans cannot
  consistently avoid making mistakes for an extended amount of time.

**Support tooling at every layer of the developer experience, including IDEs.**
The design and implementation of Carbon should facilitate both the ease of
producing such tools and their effectiveness. Syntax and textual structures that
are difficult to recognize and mechanically change without losing meaning should
be avoided.

**Support software outside of the primary use cases well.** There are
surprisingly high costs for engineers to switch languages. Even when the primary
goal is to support performance-critical software, other kinds of software should
not be penalized unnecessarily.

> "The right tool for the job is often the tool you are already using -- adding
> new tools has a higher cost than many people
> appreciate."—[John Carmack](https://twitter.com/id_aa_carmack/status/989951283900514304)

**Focus on enabling better code patterns rather than restricting bad ones.**
Adding restrictions to otherwise general facilities can have a
disproportionately negative impact in the possibly rare cases when they get in
the way. Instead, Carbon should focus on enabling better patterns, encouraging
their use, and creating incentives to ensure people prefer them. The "bad"
pattern may be critical for some rare user or some future use case. Put
differently, we will not always be able to prevent engineers from writing bad or
unnecessarily complex code, and that is okay. We should instead focus on helping
reduce the rate that this occurs accidentally, and enabling tooling and
diagnostics that warn about dangerous or surprising patterns. The language
should stay out of the business of legislating bad code patterns except where it
affects detecting logic or performance errors.

**The behavior and semantics of code should be clearly and simply specified
whenever possible.** Leaving behavior undefined in some cases for invalid,
buggy, or non-portable code may be necessary but comes at a very high cost and
should be avoided. Every case where behavior is left undefined should be clearly
spelled out with a strong rationale for this tradeoff. The code patterns without
defined behavior should be teachable and understandable by engineers. Finally,
there must be mechanisms available to detect undefined behavior, at best
statically, and at worst dynamically with high probability and at minimal cost.

**Adhere to the principle of least surprise.** Defaults should match typical
usage patterns. Implicit features should be unsurprising and expected, while
explicit syntax should inform the reader about any behavior which might
otherwise be surprising. The core concepts of implicit vs. explicit syntax are
well articulated in
[the Rust community](https://blog.rust-lang.org/2017/03/02/lang-ergonomics.html#implicit-vs-explicit),
despite some specific examples and conclusions not necessarily adhering to this
principle.

**Design features to be simple to implement.** Syntax, structure, and language
features should be chosen while keeping the complexity of the implementation
manageable. This reduces bugs, and will in most cases make the features easier
to understand.

#### Practical safety guarantees and testing mechanisms

Our goal is to add as much language-level safety and security to Carbon as
possible when balanced against the pragmatic need for software performance,
programmer ergonomics, continued support of existing/legacy Carbon code, and
both migration from and interoperation with existing C and C++ code. This
results in a hybrid strategy where we prove as much safety as we can, within
these constraints, at compile time, and combine this with dynamic runtime
checking and a strong testing methodology ranging from unit tests through
integration and system tests all the way to coverage-directed fuzz testing. We
have specific criteria that are important for this strategy to be successful:

**Make unsafe or risky aspects of an operation, interface, or type explicit and
syntactically visible.** This will allow the software to use the precise
flexibility needed and to minimize its exposure, while still aiding the reader.
It can also help the reader more by indicating the specific nature of risk faced
by a given construct. More simply, safe things shouldn't look like unsafe things
and unsafe things should be easily recognized when reading code.

**Common patterns of unsafe or risky code must support static checking.**
Waiting until a dynamic check is too late to prevent the most common errors. A
canonical example here are
[thread-safety annotations](https://clang.llvm.org/docs/ThreadSafetyAnalysis.html)
for basic mutex lock management to allow static checking. This handles the
common patterns, and we use dynamic checks, such as TSan and deadlock detection,
to handle edge cases.

**All unsafe or risky operations and interfaces must support some dynamic
checking.** Users need some way to test and verify that their code using any
such interface is in fact correct. Uncheckable unsafety removes any ability for
the user to gain confidence. This means we need to design features with unsafe
or risky aspects with dynamic checking in mind. A concrete example of this can
be seen in facilities that allow indexing into an array: such facilities should
be designed to have the bounds of the array available to implement bounds
checking when desirable.

#### Fast and scalable development

Engineers interact with many tools when working in a language that need
different levels of processing. IDEs and editor tools often use minimal parsing
to give rapid feedback. Engineers will also iterate repeatedly on any compile
error. Building, testing, and debugging complete the "edit, test, debug" cycle
that is the critical path of software development iteration. Each step needs to
be fast and scalable. Raw speed is essential for small projects and local
development. Scalability is necessary to address the large software systems we
currently use.

**Syntax should parse with bounded, small look-ahead.** Using syntax that
requires unbounded look-ahead or fully general backtracking adds significant
complexity to parsing and makes it harder to provide high quality error
messages. The result is both slower iteration and more iterations, a
multiplicative negative impact on productivity. Humans aren't immune either and
can be confused by constructs that appear to mean one thing but actually mean
another. Instead, we should design for syntax that is fast to parse, with easy
and reliable error messages.

**No semantic or contextual information used when parsing.** The more context,
and especially the more _semantic_ context, required for merely parsing code,
the fewer options available to improve the performance of tools and compilation.
Cross-file context has an especially damaging effect on the potential
distributed build graph options. Without these options, we will again be unable
to provide fast programmer iteration as the codebase scales up.

**Support separate compilation, including parallel and distributed strategies.**
We cannot assume coarse-grained compilation without blocking fundamental
scalability options for build systems of large software.

#### Current hardware architectures, OS platforms, and environments as they evolve

Carbon must have strong support for all of the major, modern platforms, the
hardware architectures they run on, and the environments in which their software
runs. A non-exhaustive list of platforms we believe should be prioritized:

- Linux
- Android
- Windows
- macOS, iOS, watchOS, tvOS
- Fuchsia
- WebAssembly
- OS/kernel
- Bare-metal

Similarly, we should prioritize support for 64-bit little-endian hardware,
including:

- x86-64
- AArch64, also known as ARM 64-bit
- PPC64LE, also known as ISA, 64-bit, Little Endian
- RV64I, also known as RISC-V 64-bit

We believe Carbon should strive to support some GPUs, other restricted
computational hardware and environments, and embedded environments, although
likely not all historical platforms of this form. While this should absolutely
include future and emerging hardware and platforms, those shouldn't
disproportionately shape the fundamental library and language design—they remain
relatively new and narrow in user base at least initially.

We do not need to prioritize support for historical platforms. To use a hockey
metaphor, Carbon should not skate to where the puck is, much less where the puck
was twenty years ago. We have existing systems to support those platforms where
necessary. Instead, Carbon should be forward-leaning in its platform support. To
give a non-exhaustive list of example, we should not prioritize support for:

- Byte sizes other than 8-bits, or non-power-of-two word sizes.
- Source code encodings other than UTF-8.
- Big- or mixed-endian, at least for computation; accessing encoded data remains
  useful.
- Non-2's-complement integer formats.
- Non-IEEE 754 floating point format as default floating point types.
- Source code in file systems that don’t support file extensions or nested
  directories.

#### Interoperability with and migration from existing C++ code

Active and widely used languages that might satisfy the use cases and goals of
Carbon have one important challenge: they have not been designed specifically to
enable migration from and interoperability with _today's_ C++. They don't build
on top of C++'s _existing_ ecosystem. There are only a few significant examples
of programming languages that center around incremental migration of large
existing codebases. They are specifically designed to not require complete
rewrites, new programming models, or building an entire new stack/ecosystem.
However, there is no comparable option for C++ today:

- JavaScript → TypeScript
- Java → Kotlin
- C++ → **???**

Carbon explores what it would look like to fill this gap and align it with the
above priorities. This requires addressing both interoperability and migration.
These are deeply interconnected, but each also brings specific requirements for
Carbon to be successful.

We must be able to move existing _large_ C++ ecosystems—some with hundreds of
millions of lines of code and tens of thousands of active developers—onto
Carbon. Any migration of this scale will take years, will need to be
incremental, and may have libraries -- particularly third-party -- that remain
in C and C++ for decades longer. As part of this, C++ developers must be
successful in switching to being Carbon developers.

High _performance_ interoperability will be vital for performance-critical
libraries to migrate to Carbon, avoiding atomic migrations including callers: it
must be possible to rewrite a C++ library as Carbon without simultaneously
rewriting all of the libraries it depends on or all of the libraries that depend
on it. All migrations must be reasonably efficient and scalable, and so must be
amenable to tooling and widely applicable. However, while a given piece code
only needs to be migrated once, we expect interoperability to be invoked
continuously to support migrated code and will thus remain important for most
users.

We believe the following at least are necessary:

**Support bi-directional interoperability with existing C++ code.** We need
Carbon code to be able to call into C and C++ libraries with both reasonable API
clarity and high performance. We will also need some ability to implement C++
interfaces with business logic in Carbon, although this direction can tolerate
slightly more constraints both in supported features and performance overhead.
In all cases, the particular performance overhead imposed by moving between C++
and Carbon will need to be easily exposed and understood by engineers.

**Familiar to experienced C++ developers with a gentle learning curve.** We need
a feasible plan for retraining a C++ workforce to become proficient in Carbon.
If long and significant study is required to be minimally proficient, meaning
able to read, superficially understand, and do limited debugging or
modifications, then the inertia of C++ will inevitably win. Further, we need a
gentle and easily traversed learning curve to basic productivity in order for
the transition to not become a chore or otherwise unsustainable for teams and
individuals.

**Expressivity comparable to C++.** If an algorithm or data structure or system
architecture can naturally be written in C++, it should also be possible to
write it naturally in Carbon.

**Possible to mechanically source-to-source migrate large segments of
large-scale idiomatic C++ code bases with high fidelity.** We will prioritize
having very low, under 2%, human interaction to achieve high fidelity migration
results. It does not require all C++ code to be migratable in this fashion, and
the resulting Carbon may be non-idiomatic. We can add reasonable constraints
here if those constraints are already well established best practices for C++
development, including design patterns, testing coverage, or usage of
sanitizers. Over many years, as Carbon evolves and codebases have had time to
migrate, the results of the tooling may also drift further from idiomatic Carbon
and have less desirable results.

### Non-goals

There are common or expected goals of many programming languages that we
explicitly call out as non-goals for Carbon. That doesn't make these things bad
in any way, but reflects the fact that they do not provide meaningful value to
us and come with serious costs and/or risks.

#### Stable language and library ABI

We would prefer to provide better, dedicated mechanisms to decompose software
subsystems in ways that scale over time rather than providing a stable ABI
across the Carbon language and libraries. Our experience is that providing broad
ABI-level stability for high-level constructs is a significant and permanent
burden on their design. It becomes an impediment to evolution, which is one of
our stated goals.

This doesn't preclude having low-level language features or tools to create
specific and curated stable ABIs, or even serializable protocols. Using any such
facilities will also cause developers to explicitly state where they are relying
on ABI and isolating it in source from code which does not need that stability.
However, these facilities would only expose a restricted set of language
features to avoid coupling the high-level language to particular stabilized
interfaces. There is a wide range of such facilities that should be explored,
from serialization-based systems like
[protobufs](https://developers.google.com/protocol-buffers) or
[pickling in Python](https://docs.python.org/3/library/pickle.html), to
[COM](https://docs.microsoft.com/en-us/windows/win32/com/com-objects-and-interfaces)
or Swift's ["resilience"](https://swift.org/blog/library-evolution/) model. The
specific approach should be designed specifically around the goals outlined
above in order to fit the Carbon language.

#### Backwards or forwards compatibility

Our goals are focused on _migration_ from one version of Carbon to the next
rather than _compatibility_ between them. This is rooted in our experience with
evolving software over time more generally and a
[live-at-head model](https://abseil.io/blog/20171004-cppcon-plenary). Any
transition, whether based on backward compatibility or a migration plan, will
require some manual intervention despite our best efforts, due to
[Hyrum's Law](http://www.hyrumslaw.com), and so we should acknowledge that
upgrades require active migrations.

#### Legacy compiled libraries without source code or ability to rebuild

We consider it a non-goal to support legacy code for which the source code is no
longer available, though we do sympathize with such use cases and would like to
see tooling mentioned above allow easier bridging between ABIs in these cases.
Similarly, plugin ABIs aren’t our particular concern, yet we’re interested in
seeing tooling which can help bridge between programs and plugins which use
different ABIs.

#### Support for existing compilation and linking models

While it is essential to have interoperability with C++, we are willing to
change the compilation and linking model of C++ itself to enable this if
necessary. Compilation models and linking models should be designed to suit the
needs of Carbon and its use cases, tools, and environments, not what happens to
have been implemented thus far in compilers and linkers.

A concrete example of this non-goal: it means platforms that cannot update their
compiler and linker when updating the Carbon language are not supported.

#### Idiomatic migration of non-modern, non-idiomatic C++ code

While large-scale, tool-assisted migration of C++ code to Carbon is an explicit
goal, handling all C++ code with this is expressly not a goal. There is likely a
great deal of C++ code that works merely by chance or has serious flaws that
prevent us from understanding the programmer's intent. While we may be able to
provide a minimally "correct" migration to very unfriendly code, mechanically
reproducing exact C++ semantics even if bizarre, even this is not guaranteed and
improving on it is not a goal. Our migration concerns are around code that is
making every effort to adhere to reasonable C++ best practices. Not relying on
undefined behavior, reasonable test coverage that passes under sanitizers, and
other basic code health may be necessary to effectively migrate.

### Principles

Some language goals will have widely-applicable, high-impact, and sometimes
non-obvious corollaries. We collect concrete language design _principles_ as a
way to document and clarify these. Principles clarify, but do not supersede,
goals and priorities. Principles should be used as a tool in making decisions.

A key difference between a principle and the design of a language feature is
that a principle should inform multiple designs, whereas a feature's design is
typically more focused on achieving a specific goal or set of goals.

We expect the list of principles to grow over time.

## Prioritization beyond goals

The features, tools, and other efforts of Carbon should be prioritized based on
a clearly articulated rationale. This may be based on this document's
overarching goals and priorities, or if those don't offer enough clarity, we
will fall back on an engineering rationale such as a required implementation
order or a cost-benefit analysis.

**Cost-benefit will drive many choices.** We expect the impact on the project
and language as a whole to influence both the cost, including complexity, and
benefit, including helping users. Benefit increases over time, which means
providing incremental solutions earlier will typically increase total benefit.
It is also reasonable for the engineering basis of a decision to factor in both
effort already invested, and effort ready to commit to the feature. This should
not overwhelm any fundamental cost-benefit analysis. However, given two equally
impactful features, we should focus on the solution that is moving the fastest.

**Domain-motivated libraries and features are an example.** For these, the cost
function will typically be the effort required to specify and implement the
feature. Benefit will stem from the number of users and how much utility the
feature provides. We don't expect to have concrete numbers for these, but we
expect prioritization decisions between features to be expressed using this
framework.

## Acknowledgements

Carbon's goals are heavily based on
["Goals and priorities for C++"](https://wg21.link/p2137) Many thanks to the
authors and contributors for helping us formulate our goals and priorities.
