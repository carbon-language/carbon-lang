# Principle: Flow checking

<!--
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

<!-- toc -->

## Table of contents

-   [Background](#background)
-   [Principle](#principle)
-   [Applications of these principles](#applications-of-these-principles)

<!-- tocstop -->

## Background

Making rigorous memory safety guarantees without runtime overhead can be
achieved using static checking. How such static checking should relate to
type-checking is a fundamental design question.

## Principle

Carbon supports (at least) two forms of static checking: type checking and _flow
checking_. They have complementary purposes:

-   the type system checks invariants of objects: an variable with a given type
    keeps the same type throughout.

-   flow checking tracks changes to statically known _state_, at every program
    point. It thus supports "strong updates", a variable changing its "flow
    type" between program points.

Flow checking only happens for programs that pass type-checking. It thus comes
conceptually "later" and can make full use of type information.

This constraint is about data dependency, not about forcing sequential phases.
In terms of implementation, a Carbon compiler may well interleave type and flow
checking of program fragments, as long as flow checking has access to all
necessary information from type checking.

This principle does not preclude decorating types with constraints on flow
state. It only informs where the checking of flow-related properties happens,
namely as part of flow checking.

In order to gain expressiveness, decorating types with constraints on flow state
is possible. Type checking ignores such flow state decorations.

## Applications of these principles

Flow checking enables the tracking of unformed state and having knowledge about
reference access being valid at some program points and invalid at others.

Functions where argument and/or return types are decorated with flow information
can be called to perform effects on flow state. This can be used to transform an
object in unformed state into one that is fully formed.

Flow checking also clarifies how functions can be decorated with effect
annotations, which can then be used to transform flow state at callsite
directly. This can be used to track invalidation of references.
