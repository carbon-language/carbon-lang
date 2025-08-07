# Safety: Terminology

<!--
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

<!-- toc -->

## Table of contents

-   [Core Terminology](#core-terminology)
-   [Vulnerability Terminology](#vulnerability-terminology)
-   [Memory Safety Specifics](#memory-safety-specifics)

<!-- tocstop -->

## Core Terminology

-   <a name="hazard">**Hazard**</a>: Unsafe coding construct that may lead to a
    bug or vulnerability.
-   <a name="bug">**Bug (or Defect)**</a>: reachable program behavior contrary
    to the author's intent
    -   <a name="active-bug">**Patent** or **active bug**</a>: buggy behavior
        that is actively occurring for users of the program
    -   <a name="latent-bug">**Latent bug**</a>: buggy behavior that does not
        currently occur for users, but can occur
    -   Behaviors that can happen but don't today in practice _are always still
        bugs_!
-   <a name="safety">**Safety**</a>: absent a qualifier or narrow context,
    refers to **system safety**, and **safety engineering**.
    -   Always a property of a _system_ or _product_ as a whole, including human
        factors, etc.
-   <a name="code-safety">**Code**, **software**, or **program safety**</a>:
    invariants or limits on program behavior in the face of bugs.
    -   Very narrow and specific meaning. Often necessary but not sufficient for
        system safety.
-   <a name="safety-bugs">**Safety bugs**</a>: bugs where some aspect of program
    behavior has insufficient (often none) invariants or limits
    -   For example, **undefined behavior** definitionally has no invariant or
        limit, and is always a safety bug.
-   <a name="initial-bug">**Initial bug**</a>: the first behavior contrary to
    the author's intent, distinct from subsequent deviations.
-   <a name="fail-stop">**Fail-stop** behavior</a>: the behavior of immediately
    terminating the program, minimizing any further business logic. This is in
    contrast to any form of "correct" program termination, continuing execution,
    or unwinding.

## Vulnerability Terminology

-   <a name="vuln">**Vulnerability** or **security vulnerability**</a>: A subset
    of bugs that creates the possibility for a malicious actor to subvert a
    program's intended behavior in a way that violates a security policy (for
    example, confidentiality, integrity, availability). Vulnerabilities are
    often exploitable manifestations of underlying bugs.
-   <a name="defense">**Vulnerability defense**</a>: The set of strategies and
    techniques employed to reduce the risks posed by vulnerabilities arising
    from bugs. These strategies operate at different levels and have varying
    degrees of effectiveness.
    -   <a name="detecting">**Detecting**</a>: while still vulnerable, detecting
        or tracking the exploit of a bug. Requires _some_ invariant or limit,
        but very minimal.
    -   <a name="mitigating">**Mitigating**</a>: making a vulnerability
        significantly more expensive, difficult, or improbable to be exploited.
    -   <a name="preventing">**Preventing** vulnerabilities</a>: while still a
        bug, making it impossible to be a vulnerability. Often this is done by
        defining behavior to [fail-stop](#fail-stop).
    -   <a name="ensuring">**Ensuring** correctness</a>: no longer a bug, much
        less a vulnerability.
    -   <a name="hardening">**Hardening**</a>: combinations of mitigation,
        prevention, and ensured correctness to reduce practical risk of
        vulnerabilities due to bugs.

## Memory Safety Specifics

-   <a name="memory-safety">**Memory safety**</a>: Having well-defined and
    predictable behavior regarding memory access, even in the face of bugs.
    Memory safety encompass several key aspects:
    -   <a name="temporal-safety">**Temporal safety**</a>: Memory accesses occur
        only within the valid lifetime of the intended memory object.
    -   <a name="spatial-safety">**Spatial safety**</a>: Memory accesses remain
        within the intended bounds of memory regions.
    -   <a name="type-safety">**Type safety**</a>: Memory is accessed and
        interpreted according to its intended type, preventing type confusion.
    -   <a name="init-safety">**Initialization safety**</a>: Memory is properly
        initialized before being read, avoiding the use of uninitialized data.
    -   <a name="race-safety">**Data-race safety**</a>: Memory writes are
        synchronized with reads or writes on other threads.
-   <a name="memory-safety-bug">**Memory safety bug**</a>: a safety bug that
    violates memory safety.
-   <a name="memory-safety-env">**Memory safe platform** or **environment**</a>:
    A computing platform or execution environment that provides mechanisms to
    prevent memory safety bugs in programs running on it from becoming
    vulnerabilities. This is a _systems_ path to achieving memory safety by
    providing the well-defined and predictable behavior by way of the execution
    environment.
-   <a name="memory-safe-language">**Memory safe language**</a>: A programming
    language with sufficient defenses against memory safety bugs for them to not
    be a significant source of security vulnerabilities. This requires
    _preventing_ vulnerabilities or _ensuring_ correctness; mitigation is not
    sufficient to provide an adequate level of memory safety. We identify
    several key requirements for a language to be memory safe:
    -   The default mode or subset of the language must provide guaranteed
        spatial, temporal, type, and initialization memory safety.
    -   Any unsafe subset must only be needed and only be used in rare,
        exceptional cases. Any use of the unsafe subset must also be well
        delineated and auditable.
    -   Currently, security evidence doesn't _require_ providing guaranteed
        data-race safety for
        [data-race bugs that are not _also_ temporal memory safety bugs](/docs/design/safety/README.md#data-races-vs-unsynchronized-temporal-safety).
        However, the temporal memory safety guarantee must still hold.
