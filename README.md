# Carbon

<!--
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

## **The Carbon Language project is an experiment exploring a future direction for the C++ programming language.**

<p align="center">
  <a href="#carbon-goals">Carbon goals</a> |
  <a href="#carbon-and-c">Carbon and C++</a> |
  <a href="#take-a-look">Take a look</a> |
  <a href="#join-us">Join us</a>
</p>

<img src="docs/images/quicksort_snippet.png" align="right" width="534">

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
    manager, and more.

<hr>

## Carbon goals

We believe Carbon must support:

1. Performance-critical software
2. Software and language evolution
3. Code that is easy to read, understand, and write
4. Practical safety and testing mechanisms
5. Fast and scalable development
6. Modern OS platforms, hardware architectures, and environments
7. Interoperability with and migration from existing C++ code

Many languages share these goals, and they can often be addressed independently
in a language's design. For the Carbon project, they are prioritized in that
order to help make clear what tradeoffs we intend to make.

Read the [language overview](docs/design/) for more on the language design
itself, and the [goals](docs/project/goals.md) for more on these values.

## Carbon and C++

If you're already a C++ developer, Carbon should have a short learning curve. It
is built out of a consistent set of language constructs that should feel
familiar. C++ code like this:

<img src="docs/images/cpp_example.png" width="760">

can be mechanically transformed to Carbon, like so:

<img src="docs/images/carbon_example.png" width="760">

without loss of performance or readability. Yet, translating C++ to Carbon isn't
necessary; you can call Carbon from C++ without overhead and the other way
around. You can port your library to Carbon, or write new Carbon on top of your
existing C++ investment. Carbon won't add a sea of dependencies or slow down
your performance-critical code. For example:

<img src="docs/images/mixed_example.png" width="760">

In terms of safety, any language that can seamlessly call C++ will not be
perfectly safe in every dimension. However, Carbon's design encourages you to
use safe constructs where possible, and on average you should see

Ultimately, C++ carries a significant historical legacy, including around ABI
stability, that constrains its evolution. Carbon is an attempt to set a new
direction for C++ developers that allows for fast development, flexibility, and
delight without sacrificing performance, interoperability, and familiarity.

Read more about
[C++ interop in Carbon](docs/design/interoperability/philosophy_and_goals.md).

## Take a look

Learn more about Carbon's design:

-   [Project goals](docs/project/goals.md)
-   [Language overview](docs/design/)
-   [Executable semantics](executable_semantics/)

## Join us

Carbon is committed to a welcoming and inclusive environment where everyone can
contribute.

-   To join the design discussion, join our
    [our Github forum](https://github.com/carbon-language/carbon-lang/discussions).
-   See our [code of conduct](CODE_OF_CONDUCT.md) and
    [contributing guidelines](CONTRIBUTING.md) for information about the Carbon
    development community.
