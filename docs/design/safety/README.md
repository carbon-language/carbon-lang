# Safety

<!--
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

<!-- toc -->

## Table of contents

-   [Overview](#overview)
-   [Safe and unsafe code](#safe-and-unsafe-code)
-   [Safety modes](#safety-modes)
-   [Memory safety model](#memory-safety-model)
    -   [Data races versus unsynchronized temporal safety](#data-races-versus-unsynchronized-temporal-safety)
-   [Safe library ecosystem](#safe-library-ecosystem)
-   [Build modes](#build-modes)
-   [Constraints](#constraints)
    -   [Probabilistic techniques likely cannot stop attacks](#probabilistic-techniques-likely-cannot-stop-attacks)
-   [References](#references)

<!-- tocstop -->

## Overview

One of Carbon's core goals is [practical safety]. This is referring to _[code
safety]_
as opposed to the larger space of [systems safety]. The largest aspect of code safety
at the language level is [memory safety], but this also applies to other aspects
of code safety such as avoiding undefined behavior in other forms.

[practical safety]:
    /docs/project/goals.md#practical-safety-and-testing-mechanisms
[code safety]:
    /docs/design/safety/terminology.md#code-software-or-program-safety
[systems safety]: /docs/design/safety/terminology.md#safety
[memory safety]: /docs/design/safety/terminology.md#memory-safety

However, Carbon also has a goal of fine grained, smooth interop-with and
migration-from existing C++ code, and C++ is a fundamentally unsafe language. It
has pervasive sources of undefined behavior and minimal memory safety
guarantees. Our safety strategy has to address how C++ code fits into it, and
provide an incremental path from where the code is at today towards increasing
levels of safety.

Ultimately, Carbon will both provide a [memory-safe language], _and_ provide a language
that is a target for mechanical migration from C++ and optimizes even further for
interop with unsafe C++ with minimal friction.

[memory-safe language]: /docs/design/safety/terminology.md#memory-safe-language

## Safe and unsafe code

Carbon will have both _safe_ and _unsafe_ code. Safe code provides limits on the
potential behavior of the program even in the face of bugs in order to prevent
[safety bugs] from becoming [vulnerabilities]. Unsafe code is any code or operation
which lacks limits or guarantees on behavior, and as a consequence may have undefined
behavior or be a safety bug.

[safety bugs]: /docs/design/safety/terminology.md#safety-bugs
[vulnerabilities]:
    /docs/design/safety/terminology.md#vulnerability-or-security-vulnerability

All things being equal, safe code constructs are preferable to unsafe
constructs, but many unsafe constructs are necessary. Where unsafe constructs
are needed, Carbon follows three principles to incorporate them into the
language:

-   The unsafe capabilities provided should be semantically narrow: only the
    minimal unsafe operation to achieve the desired result.
-   The unsafe code should be syntactically narrow and separable from
    surrounding safe code.
-   There should be a reasonable way of including the keyword `unsafe` in
    whatever syntax is used so that this keyword can be used as a common
    annotation in audits and review.

The result is that we don't model large regions or sections of Carbon code as
"unsafe" or have a parallel "unsafe" variant language. We instead focus on the
narrow and specific unsafe operations and constructs.

Note that when we're talking about the narrow semantics of an unsafe capability,
these are the semantics of the specific unsafe operation. For example, an unsafe
type conversion shouldn't also allow unsafe out-of-lifetime access. This is
separate from the _soundness_ implications of an unsafe operation, which may not
be as easily narrowed.

> **Future work**: More fully expand on the soundness principles and model for
> safe Carbon code. This is an interesting and important area of the design that
> isn't currently fleshed out in detail.

## Safety modes

The _safety mode_ of Carbon governs the extent to which unsafe code must include
the local `unsafe` keyword in its syntax to delineate it from safe code.

_Strict Carbon_ is the mode in which all unsafe code is marked with the `unsafe`
keyword. This mode is designed to satisfy the requirements of a [memory
safe language].

_Permissive Carbon_ is a mode optimized for interop with C++ and automated
migration from C++. In this mode, some unsafe code does not require an `unsafe`
keyword: specific aspects of C++ interop or pervasive patterns that occur when
migrating from C++. However, not _all_ unsafe code will omit the keyword, the
permissive mode is designed to be minimal in the unsafe code allowed.

Modes can be configured on an individual file as part of the package
declaration, or an a function body as part of the function definition. More
options such as regions of declarations or regions of statements can be explored
in the future based on demand in practice when working with mixed-strictness
code. More fine grained than statements is not expected given that the same core
expressivity is available at that finer granularity through explicitly marking
`unsafe` operations.

> **Future work**: Define the exact syntax for package declaration and function
> definition control of strictness. The syntax for enabling the permissive mode
> must include the `unsafe` keyword as it is standing in place of more granular
> use of the `unsafe` keywords and needs to still be discovered when auditing
> for safety.

> **Future work**: Define how to mark `import`s of permissive libraries in
> various contexts, balancing ergonomic burden against the potential for
> surprising amounts af unsafety in dependencies.

## Memory safety model

Carbon will use a hybrid of different techniques to achieve memory safety in its
safe code, largely broken down by the categories of memory safety:

-   [Type safety]: compile-time enforcement, the same as other statically typed languages
    with generic type systems.
-   [Initialization safety]: hybrid of run-time and compile-time enforcement.
-   [Spatial safety]: run-time enforcement.
-   [Temporal safety]: compile-time enforcement through its type system.
-   [Data-race safety]: compile-time enforcement through its type system.

[type safety]: /docs/design/safety/terminology.md#type-safety
[initialization safety]:
    /docs/design/safety/terminology.md#initialization-safety
[spatial safety]: /docs/design/safety/terminology.md#spatial-safety
[temporal safety]: /docs/design/safety/terminology.md#temporal-safety
[data-race safety]: /docs/design/safety/terminology.md#data-race-safety

**At this high level, this means Carbon's memory safety model will largely match
Rust's.** The only minor deviation at this level from Rust is the use of
run-time enforcement for initialization, where Carbon will more heavily leverage
run-time techniques such as automatic initialization and dynamic "optional"
semantics to improve the ergonomics in Carbon.

However, Carbon and Rust will have substantial differences in the _details_ of
how they approach both temporal and data-race safety.

### Data races versus unsynchronized temporal safety

Carbon has the option of distinguishing between two similar but importantly
different classes of bugs: data races and unsynchronized temporal safety
violations. Specifically, there is no evidence from security teams that there is
any significant volume of vulnerabilities that involve a data race bug but don't
also involve a temporal memory safety violation. For example, despite both Go
and non-strict-concurrency Swift only providing temporal safety, the rate of
memory safety vulnerabilities in software written in both matches the expected
low rate for memory-safe languages. As a consequence, Carbon has some
flexibility while still being a [memory-safe language] according to our definition:

-   Carbon might choose to _not_ prevent data race bugs that are not
    _themselves_ also temporal safety bugs, even though the data race may lead
    to corruption and cause the program to later violate various other forms of
    memory safety. So far, evidence has not shown this to be as significant and
    prevalent source of _vulnerabilities_ as other forms of memory safety bugs.
-   However, Carbon _must_ detect and prevent cases where a lack of
    synchronization directly allows temporal safety bugs, such as use after
    free.

Despite having this flexibility, preventing data race bugs remains _highly
valuable_ for correctness, debugging, and achieving [fearless concurrency]. If Carbon
can, it should work to prevent data races as well.

[fearless concurrency]: https://doc.rust-lang.org/book/ch16-00-concurrency.html

## Safe library ecosystem

Carbon will need access to memory-safe library ecosystems, both for
general-purpose, multi-platform functionality and for platform-specific
functionality. Currently, the industry is currently investing a massive amount
of effort to build out a sufficient ecosystem of general, multi-platform
software using Rust, and it is critical that Carbon does not impede, slow down,
or require duplicating that ecosystem. Similarly, if any other cross-platform
library ecosystems emerge in any viable memory-safe languages given our
performance constraints, we should work to reuse them and avoid duplication.

**Carbon's strategy for safe and generally reusable cross-platform libraries is
to leverage Rust libraries through interop.** This is a major motivating reason
for seamless and safe Rust interop. The Carbon project will work to avoid
creating duplication between the growing Rust library ecosystem and any future
Carbon library ecosystem. Carbon's ecosystem will be focused instead on
libraries and functionality that would either be missing or only available in
C++.

Platform-specific functionality is typically developed in that platform's native
language, whether that is Swift for Apple platforms or Kotlin for Android.
Again, the goal of Carbon should be to avoid duplicating functionality, and
instead to prioritize high quality interop with the existing platform libraries
in the relevant languages on those platforms.

## Build modes

Where Carbon's safety mode governs the language rules applied to unsafe code,
Carbon's build modes will change the _behavior_ of unsafe code.

There are two primary build modes:

-   **Release**: the primary build mode for programs in production, focuses on
    giving the best possible performance with a practical baseline of safety.
-   **Debug**: the primary build mode during development, focuses on
    high-quality developer experience.

The release build will include a baseline of hardening necessary to uphold the
run-time enforcement components of our
[memory safety model](#memory-safety-model) above. This means, for example, that
bounds checking is enabled in the release build. There is [evidence] that the
cost of these hardening steps is low. Following the specific guidance of our top
priority for [performance _control_], Carbon will provide ways to write unsafe code
that disables the run-time enforcement, enabling the control of any overhead incurred.

[evidence]: https://chandlerc.blog/posts/2024/11/story-time-bounds-checking/
[performance control]: /docs/project/goals.md#performance-critical-software

The debug build will change the actual behavior of both safe and unsafe code
with detectable bugs or detectable undefined or erroneous behavior to have
[fail-stop] behavior and provide detailed diagnostics to enable better
debugging. This mode will at least provide similar bug [detection] capabilities
to [AddressSanitizer] and [MemorySanitizer].

[fail-stop]: /docs/design/safety/terminology.md#fail-stop
[detection]: /docs/design/safety/terminology.md#detecting
[AddressSanitizer]: https://clang.llvm.org/docs/AddressSanitizer.html
[MemorySanitizer]: https://clang.llvm.org/docs/MemorySanitizer.html

Carbon will also have additional build modes to provide specific, narrow
capabilities that cannot be reasonably combined into either of the above build
modes. Each of these is expected to be an extension of either the debug or
release build for that specific purpose. For example:

-   Debug + [ThreadSanitizer]: detection of [data-races][data-race safety].
-   Release + specialized [hardening]: some users can afford significant
    run-time overhead in order to use additional hardening. Carbon will have
    several specialized build modes in this space. Hardening techniques in this
    space include [Control-Flow Integrity (CFI)][cfi] and hardware-accelerated
    address sanitizing ([HWASAN]).

[ThreadSanitizer]: https://clang.llvm.org/docs/ThreadSanitizer.html
[hardening]: /docs/design/safety/terminology.md#hardening
[cfi]: https://clang.llvm.org/docs/ControlFlowIntegrity.html
[hwasan]:
    https://clang.llvm.org/docs/HardwareAssistedAddressSanitizerDesign.html

Carbon will provide a way to disable the default hardening in release builds,
but not in a supported way. Its use is expected to be strictly for benchmarking
purposes.

## Constraints

### Probabilistic techniques likely cannot stop attacks

It's expected that non-cryptographic probabilistic techniques that can be
applied at the language level are attackable through a variety of techniques:

-   The attacker might be able to attack repeatedly until it gets through.
-   The attacker may be able to determine when the attack would be detected and
    only run the attack when it would not be.
-   The attacker might be able control the test condition to make detection much
    less likely or avoid detection completely. For example, if detection is
    based on the last 4 bits of a memory address, an attacker may be able to
    generate memory allocations, viewing the address and only attacking when
    there's a collision.

Hardware vulnerabilities may make these attacks easier than they might otherwise
appear. Future hardware vulnerabilities are difficult to predict. In general, we
do not expect non-cryptographic probabilistic techniques to be an effective
approach to achieving safety.

While _cryptographic_ probabilistic techniques can, and typically do, work
carefully to not be subject to these weaknesses, they face a very different
challenge. The overhead of a cryptographically secure hash is generally
prohibitive for use in language level constructs. Further, some of the defenses
against hardware vulnerabilities and improvements further exacerbate these
overheads. However, when these can be applied usefully such as with [PKeys],
they are robust.

[PKeys]: https://docs.kernel.org/core-api/protection-keys.html

## References

-   Proposal
    [#196: Language-level safety strategy](https://github.com/carbon-language/carbon-lang/pull/196).
-   Proposal
    [#5914: Updating Carbon's safety strategy](https://github.com/carbon-language/carbon-lang/pull/5914).
