# Carbon language

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

<a href="docs/images/snippets.md#quicksort">
<!--
Edit snippet in docs/images/snippets.md and:
https://drive.google.com/corp/drive/folders/1CsbHo3vamrxmBwHkoyz1kU0sGFqAh688
-->
<img src="docs/images/quicksort_snippet.svg" align="right" width="575"
     alt="Quicksort code in Carbon. Follow the link to read more.">
</a>

<!--
Don't let the text wrap too narrowly to the left of the above image.
The `div` reduces the vertical height.
GitHub will autolink `img`, but won't produce a link when `href="#"`.
-->
<div><a href="#"><img src="docs/images/bumper.png"></a></div>

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

## Carbon goals

We believe Carbon must support:

1. Performance-critical software
2. Software and language evolution
3. Code that is easy to read, understand, and write
4. Practical safety and testing mechanisms
5. Fast and scalable development
6. Modern OS platforms, hardware architectures, and environments
7. Interoperability with and migration from existing C++ code

While many languages share subsets of these goals, what distinguishes Carbon is
their combination. For the Carbon project, they are prioritized in the above
order to help make clear what tradeoffs we intend to make. However, each and
every goal remains critically important: **Carbon _must_ have excellent C++
interoperability and migration to be successful.**

Read the [language overview](docs/design/) for more on the language design
itself, and the [goals](docs/project/goals.md) for more on these values.

## Carbon and C++

If you're already a C++ developer, Carbon should have a short learning curve. It
is built out of a consistent set of language constructs that should feel
familiar. C++ code like this:

<a href="docs/images/snippets.md#c">
<!--
Edit snippet in docs/images/snippets.md and:
https://drive.google.com/corp/drive/folders/1CsbHo3vamrxmBwHkoyz1kU0sGFqAh688
-->
<img src="docs/images/cpp_snippet.svg" width="600"
     alt="A snippet of C++ code. Follow the link to read it.">
</a>

can be mechanically transformed to Carbon, like so:

<a href="docs/images/snippets.md#carbon">
<!--
Edit snippet in docs/images/snippets.md and:
https://drive.google.com/corp/drive/folders/1CsbHo3vamrxmBwHkoyz1kU0sGFqAh688
-->
<img src="docs/images/carbon_snippet.svg" width="600"
     alt="A snippet of converted Carbon code. Follow the link to read it.">
</a>

without loss of performance or readability. Yet, translating C++ to Carbon isn't
necessary; you can call Carbon from C++ without overhead and the other way
around. You can port your library to Carbon, or write new Carbon on top of your
existing C++ investment. Carbon won't add a sea of dependencies or slow down
your performance-critical code. For example:

<a href="docs/images/snippets.md#mixed">
<!--
Edit snippet in docs/images/snippets.md and:
https://drive.google.com/corp/drive/folders/1CsbHo3vamrxmBwHkoyz1kU0sGFqAh688
-->
<img src="docs/images/mixed_snippet.svg" width="600"
     alt="A snippet of mixed Carbon and C++ code. Follow the link to read it.">
</a>

In terms of safety, any language that can seamlessly call C++ will not be
perfectly safe in every dimension. However, Carbon's design encourages you to
use safe constructs where possible.

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

-   To watch for major release announcements, subscribe to
    [our Carbon release post on GitHub](https://github.com/carbon-language/carbon-lang/discussions/1020)
    and [star carbon-lang](https://github.com/carbon-language/carbon-lang).
-   To join the design discussion, join our
    [our Github forum](https://github.com/carbon-language/carbon-lang/discussions).
-   See our [code of conduct](CODE_OF_CONDUCT.md) and
    [contributing guidelines](CONTRIBUTING.md) for information about the Carbon
    development community.
-   We discuss Carbon on Discord; a public link will be forthcoming.
