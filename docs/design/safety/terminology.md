# Safety: Terminology

<!--
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

<!-- toc -->

## Table of contents

-   [Core Terminology](#core-terminology)
    -   [_Hazard_](#hazard)
    -   [_Bug_ or _defect_](#bug-or-defect)
        -   [_Active bug_](#active-bug)
        -   [_Latent bug_](#latent-bug)
    -   [_Safety_](#safety)
    -   [_Code_, _software_, or _program safety_](#code-software-or-program-safety)
    -   [_Safety bugs_](#safety-bugs)
    -   [_Initial bug_](#initial-bug)
    -   [_Fail-stop_](#fail-stop)
-   [Vulnerability Terminology](#vulnerability-terminology)
    -   [_Vulnerability_ or _security vulnerability_](#vulnerability-or-security-vulnerability)
    -   [_Vulnerability defense_](#vulnerability-defense)
        -   [_Detecting_](#detecting)
        -   [_Mitigating_](#mitigating)
        -   [_Preventing_ vulnerabilities](#preventing-vulnerabilities)
        -   [_Ensuring_ correctness](#ensuring-correctness)
        -   [_Hardening_](#hardening)
-   [Memory Safety Specifics](#memory-safety-specifics)
    -   [_Memory safety_](#memory-safety)
        -   [_Temporal safety_](#temporal-safety)
        -   [_Spatial safety_](#spatial-safety)
        -   [_Type safety_](#type-safety)
        -   [_Initialization safety_](#initialization-safety)
        -   [_Data-race safety_](#data-race-safety)
    -   [_Memory safety bug_](#memory-safety-bug)
    -   [_Memory-safe platform_ or _environment_](#memory-safe-platform-or-environment)
    -   [_Memory-safe language_](#memory-safe-language)

<!-- tocstop -->

## Core Terminology

### _Hazard_

Unsafe coding construct that may lead to a bug or vulnerability. For example,
indexing an array with a user-supplied and unvalidated index is a hazard.

### _Bug_ or _defect_

Reachable program behavior contrary to the author's intent.

#### _Active bug_

Buggy behavior that is actively occurring for users of the program.

#### _Latent bug_

Buggy behavior that does not currently occur for users, but is reachable.
Behaviors that are reachable, and so _can_ happen, but don't happen today in
practice _are always still bugs_!

### _Safety_

Absent a qualifier or narrow context, refers to _[system safety]_, and _[safety
engineering]_.
Always a property of a system or product as a whole, including human factors,
etc.

[system safety]: https://en.wikipedia.org/wiki/System_safety
[safety engineering]: https://en.wikipedia.org/wiki/Safety_engineering

### _Code_, _software_, or _program safety_

Invariants or limits on program behavior in the face of bugs.

-   Very narrow and specific meaning.
-   Often necessary but not sufficient for system safety.

This is a specific subset of [safety](#safety) concerns, and the ones we are
most often focused on with programming language and library design.

### _Safety bugs_

Bugs where some aspect of program behavior has insufficient (often none)
invariants or limits.

For example, _undefined behavior_ definitionally has no invariant or limit, and
reaching it is always a safety bug.

### _Initial bug_

The first behavior contrary to the author's intent, distinct from subsequent
deviations.

### _Fail-stop_

The behavior of immediately terminating the program, minimizing any further
business logic. This is in contrast to any form of "correct" program
termination, continuing execution, or unwinding.

## Vulnerability Terminology

### _Vulnerability_ or _security vulnerability_

A bug that creates the possibility for a malicious actor to subvert a program's
intended behavior in a way that violates a security policy (for example,
confidentiality, integrity, availability). Vulnerabilities are often exploitable
manifestations of underlying bugs.

### _Vulnerability defense_

The set of strategies and techniques employed to reduce the risks posed by
vulnerabilities arising from bugs. These strategies operate at different levels
and have varying degrees of effectiveness.

#### _Detecting_

While still leaving the code vulnerable, a defense that attempts to recognize
and potentially track when a specific bug has occurred dynamically. Requires
_some_ invariant or limit, but very minimal.

#### _Mitigating_

Making a vulnerability significantly more expensive, difficult, or improbable to
be exploited.

#### _Preventing_ vulnerabilities

Making it impossible for a bug to be exploited as a vulnerability without
resolving the underlying bug -- the program still doesn't behave as intended, it
just cannot be exploited. Often this is done by defining behavior to
[fail-stop](#fail-stop).

#### _Ensuring_ correctness

Ensures that if the program compiles successfully, it behaves as intended. This
typically prevents a bug being written and compiled into a program in the first
place. For example, statically typed languages typically ensure that the types
used in the program are correct.

#### _Hardening_

Combinations of mitigation, prevention, and ensured correctness to reduce the
practical risk of vulnerabilities due to bugs.

## Memory Safety Specifics

### _Memory safety_

Having well-defined and predictable behavior regarding memory access, even in
the face of bugs. Memory safety encompasses several key aspects:

#### _Temporal safety_

Memory accesses occur only within the valid lifetime of the intended memory
object.

#### _Spatial safety_

Memory accesses remain within the intended bounds of memory regions.

#### _Type safety_

Memory is accessed and interpreted according to its intended type, preventing
type confusion.

#### _Initialization safety_

Memory is properly initialized before being read, avoiding the use of
uninitialized data.

#### _Data-race safety_

Memory writes are synchronized with reads or writes on other threads.

### _Memory safety bug_

A safety bug that violates memory safety.

### _Memory-safe platform_ or _environment_

A computing platform or execution environment that provides mechanisms to
prevent memory safety bugs in programs running on it from becoming
vulnerabilities. This is a _systems_ path to achieving memory safety by
providing the well-defined and predictable behavior by way of the execution
environment. For example, a strongly sandboxed WebAssembly runtime environment
can allow a program that is itself unsafe to be executed safely

### _Memory-safe language_

A programming language with sufficient defenses against memory safety bugs for
them to not be a significant source of security vulnerabilities. This requires
_preventing_ vulnerabilities or _ensuring_ correctness; mitigation is not
sufficient to provide an adequate level of memory safety.

We identify several key requirements for a language to be memory-safe:

-   The default mode or subset of the language must provide guaranteed spatial,
    temporal, type, and initialization memory safety.
-   Any unsafe subset must only be needed and only be used in rare, exceptional
    cases. Any use of the unsafe subset must also be well delineated and
    auditable.
-   Currently, security evidence doesn't _require_ providing guaranteed
    data-race safety for
    [data-race bugs that are not _also_ temporal memory safety bugs](/docs/design/safety#data-races-versus-unsynchronized-temporal-safety).
    However, the temporal memory safety guarantee must still hold.
