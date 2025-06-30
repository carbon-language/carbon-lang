# Principle: Progressive disclosure

<!--
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

<!-- toc -->

## Table of contents

-   [Background](#background)
-   [Principle](#principle)
-   [Applications of this principle](#applications-of-this-principle)

<!-- tocstop -->

## Background

[Progressive disclosure](https://en.wikipedia.org/wiki/Progressive_disclosure)
is a UX design pattern which defers revealing information and advanced/
special-case controls to the user until they are relevant to the current task,
in order to make the system easier to learn and use. Although the term
originates in GUI design, it can be applied to language design as well. In
particular, it's one of
[Swift's core design principles](https://www.youtube.com/watch?v=CRtyWqwLM3M&t=369s).

## Principle

In Carbon, we will prefer designs that support progressive disclosure, in the
sense defined
[here](https://www.douggregor.net/posts/swift-for-cxx-practitioners-extensions/#:~:text=By%20default%2C%20any%20code,understanding%20of%20the%20language):

> ... the idea that one can ignore certain aspects of the language when starting
> out, and then learn about them only at the time when you need them, without
> invalidating any of your prior understanding of the language.

## Applications of this principle

Types such as `i32` are designed to follow progressive disclosure: they are
actually class types defined in the prelude library that support arithmetic
operations by implementing certain customization interfaces. However,
programmers can learn to use them by thinking of them as primitive types like
their C counterparts, without needing to understand the arithmetic operator
interfaces, interface implementation in general, libraries, the prelude, or even
class types. Subsequently learning those concepts doesn't invalidate that
original mental model, but rather shows how that model is built out of more
fundamental pieces.
