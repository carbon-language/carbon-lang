# Principle: Errors are values

<!--
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

[Pull request](https://github.com/carbon-language/carbon-lang/pull/301)

<!-- toc -->

## Table of contents

-   [Problem](#problem)
-   [Background](#background)
-   [Proposal](#proposal)
-   [Alternatives considered](#alternatives-considered)
    -   [Decouple errors from return values](#decouple-errors-from-return-values)
    -   [Implicit error propagation](#implicit-error-propagation)

<!-- tocstop -->

## Problem

Carbon needs a coherent, consistent approach to handling and propagating errors.

## Background

The handling and propagation of errors is a pervasive concern in almost every
nontrivial codebase, and every language provides some form of support for it.
Whether that takes the form of direct language support for errors, or emerges as
a special case of more general-purpose language features, it always has a
powerful effect on the performance, safety, and ergonomics of the language.
Consequently, Carbon's ability to meet its goals will be strongly influenced by
how it supports errors.

## Proposal

I propose establishing as a design principle that Carbon's error handling will
be based on return values, and specifically return values of sum types, rather
than exceptions or other non-return side channels.

## Alternatives considered

### Decouple errors from return values

Rather than representing errors using function return values, we could convey
errors by way of a separate "channel", as in Swift. This would require a
separate syntax for indicating whether a function can raise errors, and unless
we want to follow Swift in making errors dynamically typed, that syntax would
also need to indicate the type of those errors. We would additionally need a
separate syntax for stopping error propagation and resuming normal control flow,
such as Swift's `do`/`catch`, because the error "channel" is invisible to
ordinary code.

This approach may have some performance advantages, because the compiler always
statically knows whether a given object represents an error or a successful
return value, although it's not clear how significant that advantage will be in
Carbon. It would also make it easier to explore different implementation
strategies, such as table-driven stack unwinding, that generate very different
code for propagating errors than for handling ordinary return values.

This approach will make it harder for Carbon code to interoperate with C++ code
that uses error returns, and harder to migrate such code to Carbon. It might be
an easier migration/interoperation target for C++ code that uses exceptions, but
that's less certain, because this approach still differs from C++ exceptions in
fairly important ways, such as the fact that typing is static and propagation is
explicit. By the same token, this approach is somewhat less likely to feel
familiar to C++ programmers.

I recommend against this approach for the present, because the need for
something like `do`/`catch` makes it more complex to design, and because the
potential advantages, particularly with respect to performance, are speculative.
It seems better to start with the simpler approach, and let subsequent design
changes be driven by concrete experience with whatever problems it has.

### Implicit error propagation

As an extension of the previous option, we could treat errors as a separate
channel from return values, and allow them to propagate across stack frames
implicitly, without an explicit `?` or `try` at the callsite. This is basically
how C++ exceptions work, so this would ease migration and interoperation with
C++ code that uses them. However, this entails a readability tradeoff:
error-propagating code can be distracting boilerplate, but it can also be a
vital signal about what the code actually does, depending on the needs of the
reader.

I recommend against this approach for the present. Instead, we should
investigate ways to make explicit error propagation as syntactically lightweight
as possible, in order to address the second kind of reader use case, while
minimizing the burden on the first.
