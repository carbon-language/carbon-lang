# Carbon Language: <br/> An experimental successor to C++

<!--
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

<p align="center">
  <a href="#why-build-carbon">Why?</a> |
  <a href="#language-goals">Goals</a> |
  <a href="#project-status">Status</a> |
  <a href="#getting-started">Getting started</a> |
  <a href="#join-us">Join us</a>
</p>

**See our [announcement video](https://youtu.be/omrY53kbVoA) from
[CppNorth](https://cppnorth.ca/).** Note that Carbon is
[not ready for use](#project-status).

<a href="docs/images/snippets.md#quicksort">
<!--
Edit snippet in docs/images/snippets.md and:
https://drive.google.com/drive/folders/1QrBXiy_X74YsOueeC0IYlgyolWIhvusB
-->
<img src="docs/images/quicksort_snippet.svg" align="right" width="575"
     alt="Quicksort code in Carbon. Follow the link to read more.">
</a>

<!--
Don't let the text wrap too narrowly to the left of the above image.
The `div` reduces the vertical height. The `picture` prevents autolinking.
-->
<div><picture><img src="docs/images/bumper.png" alt=""></picture></div>

**Fast and works with C++**

-   Performance matching C++ using LLVM, with low-level access to bits and
    addresses
-   Interoperate with your existing C++ code, from inheritance to templates
-   Fast and scalable builds that work with your existing C++ build systems

**Modern and evolving**

-   Solid language foundations that are easy to learn, especially if you have
    used C++
-   Easy, tool-based upgrades between Carbon versions
-   Safer fundamentals, and an incremental path towards a memory-safe subset

**Welcoming open-source community**

-   Clear goals and priorities with robust governance
-   Community that works to be welcoming, inclusive, and friendly
-   Batteries-included approach: compiler, libraries, docs, tools, package
    manager, and more

## Why build Carbon?

C++ remains the dominant programming language for performance-critical software,
with massive and growing codebases and investments. However, it is struggling to
improve and meet developers' needs, as outlined above, in no small part due to
accumulating decades of technical debt. Incrementally improving C++ is
[extremely difficult](/docs/project/difficulties_improving_cpp.md), both due to
the technical debt itself and challenges with its evolution process. The best
way to address these problems is to avoid inheriting the legacy of C or C++
directly, and instead start with solid language foundations like
[modern generics system](#generics), modular code organization, and consistent,
simple syntax.

Existing modern languages already provide an excellent developer experience: Go,
Swift, Kotlin, Rust, and many more. **Developers that _can_ use one of these
existing languages _should_.** Unfortunately, the designs of these languages
present significant barriers to adoption and migration from C++. These barriers
range from changes in the idiomatic design of software to performance overhead.

Carbon is fundamentally **a successor language approach**, rather than an
attempt to incrementally evolve C++. It is designed around interoperability with
C++ as well as large-scale adoption and migration for existing C++ codebases and
developers. A successor language for C++ requires:

-   **Performance matching C++**, an essential property for our developers.
-   **Seamless, bidirectional interoperability with C++**, such that a library
    anywhere in an existing C++ stack can adopt Carbon without porting the rest.
-   **A gentle learning curve** with reasonable familiarity for C++ developers.
-   **Comparable expressivity** and support for existing software's design and
    architecture.
-   **Scalable migration**, with some level of source-to-source translation for
    idiomatic C++ code.

With this approach, we can build on top of C++'s existing ecosystem, and bring
along existing investments, codebases, and developer populations. There are a
few languages that have followed this model for other ecosystems, and Carbon
aims to fill an analogous role for C++:

-   JavaScript → TypeScript
-   Java → Kotlin
-   C++ → **_Carbon_**

## Language Goals

We are designing Carbon to support:

-   Performance-critical software
-   Software and language evolution
-   Code that is easy to read, understand, and write
-   Practical safety and testing mechanisms
-   Fast and scalable development
-   Modern OS platforms, hardware architectures, and environments
-   Interoperability with and migration from existing C++ code

While many languages share subsets of these goals, what distinguishes Carbon is
their combination.

We also have explicit _non-goals_ for Carbon, notably including:

-   A stable
    [application binary interface](https://en.wikipedia.org/wiki/Application_binary_interface)
    (ABI) for the entire language and library
-   Perfect backwards or forwards compatibility

Our detailed [goals](/docs/project/goals.md) document fleshes out these ideas
and provides a deeper view into our goals for the Carbon project and language.

## Project status

Carbon Language is currently an experimental project. We are hard at work on a
toolchain implementation with compiler and linker. You can try out the current
state at [compiler-explorer.com](http://carbon.compiler-explorer.com/).

We want to better understand whether we can build a language that meets our
successor language criteria, and whether the resulting language can gather a
critical mass of interest within the larger C++ industry and community.

Currently, we have fleshed out several core aspects of both Carbon the project
and the language:

-   The strategy of the Carbon Language and project.
-   An open-source project structure, governance model, and evolution process.
-   Critical and foundational aspects of the language design informed by our
    experience with C++ and the most difficult challenges we anticipate. This
    includes designs for:
    -   Generics
    -   Class types
    -   Inheritance
    -   Operator overloading
    -   Lexical and syntactic structure
    -   Code organization and modular structure
-   A prototype interpreter demo that can both run isolated examples and gives a
    detailed analysis of the specific semantic model and abstract machine of
    Carbon. We call this the [Carbon Explorer](/explorer/).
-   An under-development [compiler and toolchain](/toolchain/) that will compile
    Carbon (and eventually C++ code as well) into standard executable code. This
    is where most of our current implementation efforts are directed.

If you're interested in contributing, we're currently focused on developing the
Carbon toolchain until it can
[support Carbon ↔ C++ interop](/docs/project/roadmap.md#access-most-non-template-c-apis-in-carbon).
Beyond that, we plan to continue developing the design and toolchain until we
can ship the
[0.1 language](/docs/project/milestones.md#milestone-01-a-minimum-viable-product-mvp-for-evaluation)
and support evaluating Carbon in more detail.

You can see our [full roadmap](/docs/project/roadmap.md) for more details.

## Carbon and C++

If you're already a C++ developer, Carbon should have a gentle learning curve.
It is built out of a consistent set of language constructs that should feel
familiar and be easy to read and understand.

C++ code like this:

<a href="docs/images/snippets.md#c">
<!--
Edit snippet in docs/images/snippets.md and:
https://drive.google.com/drive/folders/1QrBXiy_X74YsOueeC0IYlgyolWIhvusB
-->
<img src="docs/images/cpp_snippet.svg" width="600"
     alt="A snippet of C++ code. Follow the link to read it.">
</a>

corresponds to this Carbon code:

<a href="docs/images/snippets.md#carbon">
<!--
Edit snippet in docs/images/snippets.md and:
https://drive.google.com/drive/folders/1QrBXiy_X74YsOueeC0IYlgyolWIhvusB
-->
<img src="docs/images/carbon_snippet.svg" width="600"
     alt="A snippet of converted Carbon code. Follow the link to read it.">
</a>

You can call Carbon from C++ without overhead and the other way around. This
means you migrate a single C++ library to Carbon within an application, or write
new Carbon on top of your existing C++ investment. For example:

<a href="docs/images/snippets.md#mixed">
<!--
Edit snippet in docs/images/snippets.md and:
https://drive.google.com/drive/folders/1QrBXiy_X74YsOueeC0IYlgyolWIhvusB
-->
<img src="docs/images/mixed_snippet.svg" width="600"
     alt="A snippet of mixed Carbon and C++ code. Follow the link to read it.">
</a>

Read more about
[C++ interop in Carbon](/docs/design/interoperability/philosophy_and_goals.md).

Beyond interoperability between Carbon and C++, we're also planning to support
migration tools that will mechanically translate idiomatic C++ code into Carbon
code to help you switch an existing C++ codebase to Carbon.

## Generics

Carbon provides a
**[modern generics system](/docs/design/generics/overview.md#what-are-generics)**
with checked definitions, while still **supporting opt-in
[templates](/docs/design/templates.md) for seamless C++ interop**. Checked
generics provide several advantages compared to C++ templates:

-   **Generic definitions are fully type-checked**, removing the need to
    instantiate to check for errors and giving greater confidence in code.
    -   Avoids the compile-time cost of re-checking the definition for every
        instantiation.
    -   When using a definition-checked generic, usage error messages are
        clearer, directly showing which requirements are not met.
-   **Enables automatic, opt-in type erasure and dynamic dispatch** without a
    separate implementation. This can reduce the binary size and enables
    constructs like heterogeneous containers.
-   **Strong, checked interfaces** mean fewer accidental dependencies on
    implementation details and a clearer contract for consumers.

Without sacrificing these advantages, **Carbon generics support
specialization**, ensuring it can fully address performance-critical use cases
of C++ templates. For more details about Carbon's generics, see their
[design](/docs/design/generics).

In addition to easy and powerful interop with C++, Carbon templates can be
constrained and incrementally migrated to checked generics at a fine granularity
and with a smooth evolutionary path.

## Memory safety

Safety, and especially
[memory safety](https://en.wikipedia.org/wiki/Memory_safety), remains a key
challenge for C++ and something a successor language needs to address. Our
initial priority and focus is on immediately addressing important, low-hanging
fruit in the safety space:

-   Tracking uninitialized states better, increased enforcement of
    initialization, and systematically providing hardening against
    initialization bugs when desired.
-   Designing fundamental APIs and idioms to support dynamic bounds checks in
    debug and hardened builds.
-   Having a default debug build mode that is both cheaper and more
    comprehensive than existing C++ build modes even when combined with
    [Address Sanitizer](https://github.com/google/sanitizers/wiki/AddressSanitizer).

Once we can migrate code into Carbon, we will have a simplified language with
room in the design space to add any necessary annotations or features, and
infrastructure like [generics](#generics) to support safer design patterns.
Longer term, we will build on this to introduce **a safe Carbon subset**. This
will be a large and complex undertaking, and won't be in the 0.1 design.
Meanwhile, we are closely watching and learning from efforts to add memory safe
semantics onto C++ such as Rust-inspired
[lifetime annotations](https://discourse.llvm.org/t/rfc-lifetime-annotations-for-c/61377).

## Getting started

To try out Carbon immediately in your browser, you can use the toolchain at:
[carbon.compiler-explorer.com](http://carbon.compiler-explorer.com/).

We are developing a traditional toolchain for Carbon that can compile and link
programs. However, Carbon is still an early, experimental project, and so we
only have very experimental nightly releases of the Carbon toolchain available
to download, and only on limited platforms. If you are using a recent Ubuntu
Linux or similar (Debian, WSL, etc.), you can try these out by going to our
[releases](https://github.com/carbon-language/carbon-lang/releases) page and
download the latest nightly toolchain tar file:
`carbon_toolchain-0.0.0-0.nightly.YYYY.MM.DD.tar.gz`. Then you can try it out:

```shell
# A variable with the nightly version from yesterday:
VERSION="$(date -d yesterday +0.0.0-0.nightly.%Y.%m.%d)"

# Get the release
wget https://github.com/carbon-language/carbon-lang/releases/download/v${VERSION}/carbon_toolchain-${VERSION}.tar.gz

# Unpack the toolchain:
tar -xvf carbon_toolchain-${VERSION}.tar.gz

# Create a simple Carbon source file:
echo "import Core library \"io\"; fn Run() { Core.Print(42); }" > forty_two.carbon

# Compile to an object file:
./carbon_toolchain-${VERSION}/bin/carbon compile \
  --output=forty_two.o forty_two.carbon

# Install minimal system libraries used for linking. Note that installing `gcc`
# or `g++` for compiling C/C++ code with GCC will also be sufficient, these are
# just the specific system libraries Carbon linking still uses.
sudo apt install libgcc-11-dev

# Link to an executable:
./carbon_toolchain-${VERSION}/bin/carbon link \
  --output=forty_two forty_two.o

# Run it:
./forty_two
```

As a reminder, the toolchain is still very early and many things don't yet work.
Please hold off on filing lots of bugs: we know many parts of this don't work
yet or may not work on all systems. We expect to have releases that are much
more robust and reliable that you can try out when we reach our
[0.1 milestone](/docs/project/milestones.md#milestone-01-a-minimum-viable-product-mvp-for-evaluation).

If you want to build Carbon's toolchain yourself or are thinking about
contributing fixes or improvements to Carbon, you'll need to install our
[build dependencies](/docs/project/contribution_tools.md#setup-commands) (Clang,
LLD, libc++) and check out the Carbon repository. For example, on Debian or
Ubuntu:

```shell
# Update apt.
sudo apt update

# Install tools.
sudo apt install \
  clang \
  libc++-dev \
  libc++abi-dev \
  lld

# Download Carbon's code.
$ git clone https://github.com/carbon-language/carbon-lang
$ cd carbon-lang
```

Then you can try out our toolchain which has a very early-stage compiler for
Carbon:

```shell
# Build and run the toolchain's help to get documentation on the command line.
$ ./scripts/run_bazelisk.py run //toolchain -- help
```

For complete instructions, including installing dependencies on various
different platforms, see our
[contribution tools documentation](/docs/project/contribution_tools.md).

Learn more about the Carbon project:

-   [Project goals](/docs/project/goals.md)
-   [Language design overview](/docs/design)
-   [Carbon Explorer](/explorer)
-   [Carbon Toolchain](/toolchain)
-   [FAQ](/docs/project/faq.md)

## Conference talks

Carbon focused talks from the community:

### 2024

-   Generic implementation strategies in Carbon and Clang, LLVM Developers'
    Meeting ([video](https://youtu.be/j0BL52NdjAU),
    [slides](https://chandlerc.blog/slides/2024-llvm-generic-implementation/#/))
-   The Carbon Language: Road to 0.1, NDC {TechTown}
    ([video](https://youtu.be/bBvLmDJrzvI),
    [slides](https://chandlerc.blog/slides/2024-ndc-techtown-carbon-road-to-0-dot-1))
-   How designing Carbon with C++ interop taught me about C++ variadics and
    overloads, CppNorth ([video](https://youtu.be/8SGMy9ENGz8),
    [slides](https://chandlerc.blog/slides/2024-cppnorth-design-stories))
-   Generic Arity: Definition-Checked Variadics in Carbon, C++Now
    ([video](https://youtu.be/Y_px536l_80),
    [slides](https://docs.google.com/presentation/d/10aM1mFMN6Cd5ZkE4OfeiZtSnkVNbo33N-V0et21umww/edit))
-   Carbon: An experiment in different tradeoffs, panel session, EuroLLVM
    ([video](https://youtu.be/Za_KWj5RMR8),
    [slides](https://llvm.org/devmtg/2024-04/slides/LightningTalks/Smith-Carbons-high-level-semanticIR.pdf))
    -   [Alex Bradbury's notes](https://muxup.com/2024q2/notes-from-the-carbon-panel-session-at-eurollvm)
-   Carbon's high-level semantic IR lightning talk, EuroLLVM
    ([video](https://youtu.be/vIWT4RhUcyw))

### 2023

-   Carbon’s Successor Strategy: From C++ interop to memory safety, C++Now
    ([video](https://youtu.be/1ZTJ9omXOQ0),
    [slides](https://chandlerc.blog/slides/2023-cppnow-carbon-strategy/index.html#/))
-   Definition-Checked Generics, C++Now
    -   Part 1 ([video](https://youtu.be/FKC8WACSMP0),
        [slides](https://chandlerc.blog/slides/2023-cppnow-generics-1/#/))
    -   Part 2 ([video](https://youtu.be/VxQ3PwxiSzk),
        [slides](https://chandlerc.blog/slides/2023-cppnow-generics-2/#/))
-   Modernizing Compiler Design for Carbon’s Toolchain, C++Now
    ([video](https://youtu.be/ZI198eFghJk),
    [slides](https://chandlerc.blog/slides/2023-cppnow-compiler/index.html#/))

### 2022

-   Carbon Language: Syntax and trade-offs, Core C++
    ([video](https://youtu.be/9Y2ivB8VaIs),
    [slides](https://docs.google.com/presentation/d/1znvL12xCuEfcsP6tpPdrQPnh-UoPFOLnC_RVXZteYaM/edit))
-   Carbon Language: An experimental successor to C++, CppNorth
    ([video](https://youtu.be/omrY53kbVoA),
    [slides](https://chandlerc.blog/slides/2022-07-19-cppnorth-keynote/#/))

## Join us

We'd love to have folks join us and contribute to the project. Carbon is
committed to a welcoming and inclusive environment where everyone can
contribute.

-   Most of Carbon's design discussions occur on
    [Discord](https://discord.gg/ZjVdShJDAs).
-   To watch for major release announcements, subscribe to our
    [Carbon release post on GitHub](https://github.com/carbon-language/carbon-lang/discussions/1020)
    and [star carbon-lang](https://github.com/carbon-language/carbon-lang).
-   See our [code of conduct](CODE_OF_CONDUCT.md) and
    [contributing guidelines](CONTRIBUTING.md) for information about the Carbon
    development community.

### Contributing

You can also directly:

-   [Contribute to the language design](CONTRIBUTING.md#contributing-to-the-language-design):
    feedback on design, new design proposal
-   [Contribute to the language implementation](CONTRIBUTING.md#contributing-to-the-language-implementation)
    -   [Carbon Toolchain](/toolchain/), and project infrastructure

You can **check out some
["good first issues"](https://github.com/carbon-language/carbon-lang/labels/good%20first%20issue)**,
or join the `#contributing-help` channel on
[Discord](https://discord.gg/ZjVdShJDAs). See our full
[`CONTRIBUTING`](CONTRIBUTING.md) documentation for more details.
