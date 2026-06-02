# Support octal literals

<!--
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

[Pull request](https://github.com/carbon-language/carbon-lang/pull/6910)

<!-- toc -->

## Table of contents

-   [Abstract](#abstract)
-   [Problem](#problem)
-   [Background](#background)
-   [Proposal](#proposal)
-   [Future work](#future-work)
    -   [File permissions API](#file-permissions-api)
-   [Rationale](#rationale)
-   [Alternatives considered](#alternatives-considered)
    -   [No octal literals](#no-octal-literals)

<!-- tocstop -->

## Abstract

Support octal literals, mainly for migrating Unix file permissions. Reflects
leads decision
[#6821](https://github.com/carbon-language/carbon-lang/issues/6821).

## Problem

Carbon currently does not support octal numeric literals, because they're very
rare, as previously decided in proposal
[#143: Numeric literals](https://github.com/carbon-language/carbon-lang/pull/143).
However, as part of interoperability with POSIX file system calls such as
[`umask`](https://pubs.opengroup.org/onlinepubs/9699919799/functions/umask.html),
we want an easy way to express octal file permissions.

## Background

Leads discussed support of octal literals in issue
[#6821: Support octal literals](https://github.com/carbon-language/carbon-lang/issues/6821).

Unix file permissions written in octal are familiar to both programmers and
non-programmers who have experience administering Unix-like machines. For
example:

-   `chmod OCTAL-MODE FILE...`
-   [POSIX `mode_t` values](https://pubs.opengroup.org/onlinepubs/9699919799/basedefs/sys_stat.h.html),
    including the argument to
    [`umask`](https://pubs.opengroup.org/onlinepubs/9699919799/functions/umask.html).

However, they are are not likely to be readable to those unfamiliar with them,
nor are they very common in code. We also expect these to be the primary use of
octal numeric literals in Carbon.

Given these issues, proposal #143
[rejected octal literals](/proposals/p000143-numeric-literals.md#octal-literals).
Now, we're testing interoperability of POSIX file system calls, and we are
considering octal literals as a potential solution.

Proposal #143 [discussed the Carbon-style `0o` versus C++-style `0` prefix for
octal literals. The same logic still applies, so we will not address it here.

## Proposal

Introduce support for octal literals using the `0o` prefix (for example,
`0o755`), followed by one or more octal digits (`0-7`). This provides a very
simple lexical space for octal numbers, mapping clearly from C++ and building
consistently off the existing `0x...` syntax for hexadecimal and `0b...` syntax
for binary.

## Future work

### File permissions API

We may still provide a file permissions API in Carbon, for example as part of a
`Core` file system API. This proposal takes no stance on what that API should
look like. The only decision being made right now is that supporting octal
literals is worthwhile for interoperability and migration.

## Rationale

This proposal effectively advances Carbon's goals by focusing on:

-   [Interoperability with and migration from existing C++ code](/docs/project/goals.md#interoperability-with-and-migration-from-existing-c-code):
    Providing a direct counterpart for C++ octal literals simplifies the
    migration of Unix file system code without needing to wait for a better file
    permission API.
-   [Code that is easy to read, understand, and write](/docs/project/goals.md#code-that-is-easy-to-read-understand-and-write):
    The explicit `0o` prefix avoids the frequent confusion caused by C++'s
    leading `0` syntax, while maintaining consistency with hex and binary
    prefixes.
-   [Software and language evolution](/docs/project/goals.md#software-and-language-evolution):
    The `0o` octal literal syntax is consistent with other literals. We don't
    expect it to hinder future language features.

## Alternatives considered

### No octal literals

Proposal #143 rejected octal literals. The main argument was that they are
rarely used. However, the cost of octal literal syntax is low, and the benefit
for C++ interoperability and migration is enough that we should add them.
