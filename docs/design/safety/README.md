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
-   [Safe library ecosystem](#safe-library-ecosystem)
-   [Build modes](#build-modes)
-   [Constraints](#constraints)
    -   [Probabilistic techniques likely cannot stop attacks](#probabilistic-techniques-likely-cannot-stop-attacks)
-   [Carbon's memory safety model](#carbons-memory-safety-model)
    -   [Compile-time enforcement](#compile-time-enforcement)
    -   [Type safety](#type-safety)
    -   [Initialization safety](#initialization-safety)
    -   [Spatial safety](#spatial-safety)
    -   [Temporal safety](#temporal-safety)
    -   [Data-race safety](#data-race-safety)
    -   [Protecting immutable borrows from mutation](#protecting-immutable-borrows-from-mutation)
-   [Alternatives considered](#alternatives-considered)
-   [References](#references)

<!-- tocstop -->

## Overview

One of Carbon's core goals is [practical safety]. This is referring to _[code
safety]_ as opposed to the larger space of [systems safety]. The largest aspect
of code safety at the language level is [memory safety], but this also applies
to other aspects of code safety such as avoiding undefined behavior in other
forms.

[practical safety]:
    /docs/project/goals.md#practical-safety-and-testing-mechanisms
[code safety]: terminology.md#code-software-or-program-safety
[systems safety]: terminology.md#safety
[memory safety]: terminology.md#memory-safety

However, Carbon also has a goal of fine grained, smooth interop-with and
migration-from existing C++ code, and C++ is a fundamentally unsafe language. It
has pervasive sources of undefined behavior and minimal memory safety
guarantees. Our safety strategy has to address how C++ code fits into it, and
provide an incremental path from where the code is at today towards increasing
levels of safety.

Ultimately, Carbon will both provide a [memory-safe language], _and_ provide a
language that is a target for mechanical migration from C++ and optimizes even
further for interop with unsafe C++ with minimal friction.

[memory-safe language]: terminology.md#memory-safe-language

## Safe and unsafe code

Carbon will have both _safe_ and _unsafe_ code. Safe code provides limits on the
potential behavior of the program even in the face of [bugs] in order to
[prevent][preventing vulnerabilities] [safety bugs] from becoming
[vulnerabilities]. Unsafe code is any code or operation which lacks limits or
guarantees on behavior, and as a consequence may have undefined behavior or be a
safety bug.

[bugs]: terminology.md#bug-or-defect
[preventing vulnerabilities]: terminology.md#preventing-vulnerabilities
[safety bugs]: terminology.md#safety-bugs
[vulnerabilities]: terminology.md#vulnerability-or-security-vulnerability

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
keyword. This mode is designed to satisfy the requirements of a
[memory-safe language].

_Permissive Carbon_ is a mode optimized for interop with C++ and automated
migration from C++. In this mode, some unsafe code does not require an `unsafe`
keyword: specific aspects of C++ interop or pervasive patterns that occur when
migrating from C++. However, not _all_ unsafe code will omit the keyword, the
permissive mode is designed to be minimal in the unsafe code allowed.

Modes can be configured on an individual file as part of the package
declaration, or on a function body as part of the function definition. More
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
> surprising amounts of unsafety in dependencies.

## Safe library ecosystem

Carbon will need access to memory-safe library ecosystems, both for
general-purpose, multi-platform functionality and for platform-specific
functionality. Currently, the industry is investing a massive amount
of effort to build out a sufficient ecosystem of general, multi-platform
software using Rust, and it is critical that Carbon does not impede, slow down,
or require duplicating that ecosystem. Similarly, if any other cross-platform
library ecosystems emerge in any viable
[memory-safe languages][memory-safe language] given our performance constraints,
we should work to reuse them and avoid duplication.

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
    giving the best possible performance with a practical baseline of
    [safety][code safety].
-   **Debug**: the primary build mode during development, focuses on
    high-quality developer experience.

The release build will include a baseline of [hardening] necessary to uphold the
run-time enforcement components of our
[memory safety model](#carbons-memory-safety-model) below. This means, for
example, that bounds checking is enabled in the release build. There is
[evidence] that the cost of these hardening steps is low. Following the specific
guidance of our top priority for [performance control], Carbon will provide ways
to write unsafe code that disables the run-time enforcement, enabling the
control of any overhead incurred.

[hardening]: terminology.md#hardening
[evidence]: https://chandlerc.blog/posts/2024/11/story-time-bounds-checking/
[performance control]: /docs/project/goals.md#performance-critical-software

The debug build will change the actual behavior of both safe and unsafe code
with detectable [bugs] or detectable undefined or erroneous behavior to have
[fail-stop] behavior and provide detailed diagnostics to enable better
debugging. This mode will at least provide similar bug [detection] capabilities
to [AddressSanitizer] and [MemorySanitizer].

[fail-stop]: terminology.md#fail-stop
[detection]: terminology.md#detecting
[AddressSanitizer]: https://clang.llvm.org/docs/AddressSanitizer.html
[MemorySanitizer]: https://clang.llvm.org/docs/MemorySanitizer.html

Carbon will also have additional build modes to provide specific, narrow
capabilities that cannot be reasonably combined into either of the above build
modes. Each of these is expected to be an extension of either the debug or
release build for that specific purpose. For example:

-   Debug + [ThreadSanitizer]: [detection] of [data-races][data-race safety].
-   Release + specialized [hardening]: some users can afford significant
    run-time overhead in order to use additional hardening. Carbon will have
    several specialized build modes in this space. Hardening techniques in this
    space include [Control-Flow Integrity (CFI)][cfi] and hardware-accelerated
    address sanitizing ([HWASAN]).

[ThreadSanitizer]: https://clang.llvm.org/docs/ThreadSanitizer.html
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
-   The attacker may be able to determine when the attack would be
    [detected][detection] and only run the attack when it would not be.
-   The attacker might be able to control the test condition to make detection
    much less likely or avoid detection completely. For example, if detection is
    based on the last 4 bits of a memory address, an attacker may be able to
    generate memory allocations, viewing the address and only attacking when
    there's a collision.

Hardware [vulnerabilities] may make these attacks easier than they might
otherwise appear. Future hardware vulnerabilities are difficult to predict. In
general, we do not expect non-cryptographic probabilistic techniques to be an
effective approach to achieving [safety][code safety].

While _cryptographic_ probabilistic techniques can, and typically do, work
carefully to not be subject to these weaknesses, they face a very different
challenge. The overhead of a cryptographically secure hash is generally
prohibitive for use in language-level constructs. Further, some of the
[defenses][vulnerability defense] against hardware vulnerabilities
further exacerbate these overheads. However, when these can be
applied usefully such as with [PKeys], they are robust.

[vulnerability defense]: terminology.md#vulnerability-defense
[PKeys]: https://docs.kernel.org/core-api/protection-keys.html

## Carbon's memory safety model

Carbon will use a hybrid of different techniques to achieve [memory safety] in
its safe code, largely broken down by the categories of memory safety:

-   [Type safety]: compile-time enforcement, the same as other statically typed
    languages with generic type systems.
-   [Initialization safety]: hybrid of run-time and compile-time enforcement.
-   [Spatial safety]: run-time enforcement.
-   [Temporal safety]: compile-time enforcement through its type system.
-   [Data-race safety]: compile-time enforcement through its type system.

[type safety]: terminology.md#type-safety
[initialization safety]: terminology.md#initialization-safety
[spatial safety]: terminology.md#spatial-safety
[temporal safety]: terminology.md#temporal-safety
[data-race safety]: terminology.md#data-race-safety

**At this high level, this means Carbon's memory safety model will largely match
Rust's.** The only minor deviation at this level from Rust is the use of
run-time enforcement for initialization, where Carbon will more heavily leverage
run-time techniques such as automatic initialization and dynamic "optional"
semantics to improve the ergonomics in Carbon.

However, Carbon and Rust will have substantial differences in the _details_ of
how they approach both temporal and data-race safety.

### Compile-time enforcement

Carbon uses a combination of language mechanisms to enforce memory safety, each
of which will have its own document that goes into greater detail:

-   Places and place sets: give compile-time names to runtime object locations,
    and track what is known about those locations at compile-time.
-   Alias tracking: is how the signatures of functions and type definitions
    allow the compiler to determine which pointers and references could point to
    which places and place sets.
-   Safety effects: are annotations on a function's return to mark that callers
    must update their safety tracking state.
-   Ownership: allows an owner to be responsible for data that can still have
    independent fate (unlike fields), such as the items in a container. Carbon
    supports both single and multiple (reference-counted) ownership.
-   Type and place capabilities: restrict what can be done with an object based
    on its type or place, and prevent uses of objects that have capabilities
    that would allow them to violate safety invariants.
-   Flow-sensitivity: allows omitting some safety annotations within a function
    body, to be more concise and to allow them to be different at different
    points in the function as determined by the compiler's analysis to have
    greater precision.
-   Erasure of safety information before lowering: allows separate compilation
    and interop with unsafe code including C++.

### Type safety

To be memory safe we need compile-time enforcement that a memory location is
only accessed with its correct runtime type. Carbon's type safety is not fully
static since it has runtime type tests and conversions. Examples include
base-to-derived dynamic downcasts, determining whether an optional value is
present, and executing different code depending on which alternative is present
in a sum type. To be safe, the language will enforce, through the language
specification and compiler, that alternatives of a sum or optional type are only
accessed after a successful runtime test that the runtime state matches, through
(future) language constructs like `if let`, `guard let`, and `match`. In
addition, switching which alternative a sum type or optional holds must
invalidate any existing pointers into contained data.

### Initialization safety

To prevent reading uninitialized or moved-from memory without requiring every
variable to be initialized at declaration, Carbon defines an _unformed state_
that types can opt into for objects before initialization or after a move:

-   Such types allow specific operations, such as assignment and destruction,
    without knowing the runtime initialization state of an object.
-   Types whose unformed representation is distinct from valid values can also
    provide a mechanism to query the runtime initialization state of an object.

A binding's type indicates whether it can be in an unformed state, restricting
potentially unformed objects to operations valid in that state:

-   Local variables have implicit flow-sensitive state tracked by the compiler,
    marking whether each variable is initialized, uninitialized, or maybe
    unformed.
-   Objects of types with no unformed state must always be initialized, though
    they may be default initialized for types that support that.
-   A function's parameter types indicate which parameters may be unformed;
    other parameters must be initialized by the caller.
-   Fields of a formed object must never be observed in an unformed state unless
    declared with a maybe-unformed wrapper type (similar to Rust's
    `MaybeUninit`), or transitorily in the body of a function whose
    flow-sensitive tracking knows the field is unformed and restores it before
    exposing the object.

Safety effects on function signatures mark functions that definitely initialize
an object or that may leave an object in an unformed state (such as by moving
from it).

### Spatial safety

Spatial safety prevents out-of-bounds memory accesses and is enforced at runtime
in Carbon's standard library. Carbon [pointers] refer to a single object and do
not support indexing or pointer arithmetic; performing raw pointer arithmetic or
unchecked indexing is an `unsafe` operation. Safe access to contiguous elements
uses standard library container and slice types, which perform runtime bounds
checking by default even in release builds.

[pointers]: /docs/design/values.md#pointers

### Temporal safety

The goal of temporal safety is to prevent use-after-free. Every use-after-free
has four components:

```carbon
fn Run() {
  // 1. an allocation,
  var x: buf(i32) = (1, 20, 300);
  // 2. a pointer capturing an address in the allocation,
  var p: i32* = &x[0];
  // 3. a free or reallocation that invalidates that pointer, and
  x.PushBack(4000);
  // 4. a use of that now-dangling pointer, which Carbon
  //    will diagnose.
  Core.Print(*p);
}
```

Carbon will detect and diagnose these at compile time using these safety
ingredients:

-   The allocation will have an owner.
-   Owners have a place set that the allocations live in.
-   Pointers have a place set that says which places the pointer can point to.
-   Local pointers may omit that place set and have it determined by the
    compiler with flow sensitivity.
-   Operations that can free or reallocate the allocation require write access
    to the owner and are marked with an `invalidate` effect.
-   An `invalidate` effect includes a place set specifying what is invalidated
    by the operation, in this case the data owned by the `buf` object.
-   Alias tracking allows the compiler to know which pointers could point to the
    invalidated places, so they may be invalidated.
-   Any use of an invalid pointer is diagnosed.

### Data-race safety

Carbon's data-race safety builds on the model of [Clang's thread-safety
annotations] (`-Wthread-safety`) to make transitioning from annotated C++ code
easier, enforcing data-race freedom at compile time without attempting to
prevent deadlocks:

-   Each mutex has a place set of `guarded` variables, and function signatures
    use constraints to require that a guarding lock is held when called.
-   Rather than marking thread safety on types, Carbon tracks a `shared`
    attribute on places (flow-sensitively within a function and explicitly on
    `shared` reference and pointer parameters) when an object may be accessed
    across threads. Accessing a `shared` place is restricted to operations
    defined to work on `shared` places: unguarded data is deeply read-only,
    types with safe interior mutability (such as atomics and mutexes) opt into
    `shared` operations, and `guarded` variables can be read or mutated while
    their guarding lock is held.
-   Safety effects on function signatures update which places are `shared`
    across threads and which locks have been acquired or released.

[Clang's thread-safety annotations]:
    https://clang.llvm.org/docs/ThreadSafetyAnalysis.html

### Protecting immutable borrows from mutation

Carbon [values] (used by default for `let` bindings and function parameters) are
guaranteed not to change. Rather than being forced to copy an existing object to
ensure its value does not change, Carbon allows representing a value as a
`const` reference backed by an _immutable borrow_ of that object. While a place
is immutably borrowed for a local or parameter, Carbon tracks the borrow
flow-sensitively and prevents mutation through any reference or pointer whose
place set may overlap it.

Because an immutable borrow stands in for copying a value, it enforces
_copy-depth immutability_: for a copyable type, the scope of an immutable borrow
is all the data that would be copied by a copy operation. External unowned
pointers and shared-ownership pointers are copied shallowly and therefore
borrowed shallowly, whereas singly owned data is copied deeply and therefore
borrowed deeply.

[values]: /docs/design/values.md#value-expressions

## Alternatives considered

-   [Ante-inspired Rust-like model with non-exclusive mutable borrows](/proposals/p008118-memory-safety-intro.md#ante-inspired-rust-like-model-with-non-exclusive-mutable-borrows)
-   [Safety-effects model with flow-sensitive checking](/proposals/p008118-memory-safety-intro.md#safety-effects-model-with-flow-sensitive-checking)

## References

-   Proposal
    [#196: Language-level safety strategy](https://github.com/carbon-language/carbon-lang/pull/196)
-   Proposal
    [#257: Initialization of memory and variables](https://github.com/carbon-language/carbon-lang/pull/257)
-   Proposal
    [#5914: Updating Carbon's safety strategy](https://github.com/carbon-language/carbon-lang/pull/5914)
-   (Pending) proposal
    [#7640: Reworking unformed state](https://github.com/carbon-language/carbon-lang/pull/7640)
-   Proposal
    [#8118: Memory safety intro](https://github.com/carbon-language/carbon-lang/pull/8118)
