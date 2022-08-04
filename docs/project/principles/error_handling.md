# Principle: Errors are values

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

Most nontrivial programs contain functions that can _fail_, meaning that even if
all their preconditions are met, they may not be able to perform their primary
behavior. For example, a function that reads data from a remote server may fail
if the server is unreachable, and a function that parses a string to return an
integer may fail if the input string is not a properly-formatted integer.

In many cases, the function author wants these failures to be _recoverable_,
meaning that a direct or transitive caller can respond to the failure in some
way that enables the program to continue running.

## Principle

A Carbon function that needs to report recoverable failures should return a sum
type whose alternatives represent the success case and failure cases, such as
`Optional(T)`, `Result(T, Error)`, or `bool`. The function's successful return
value, and any metadata about the failure, should be embedded in the
alternatives of the sum type, rather than reported by way of output parameters
or other side channels. Carbon's design will prioritize making this form of
error handling efficient and ergonomic.

## Applications of these principles

Carbon errors, unlike exceptions in C++ and similar languages, will not be
propagated implicitly. Instead, Carbon will very likely need to provide some
explicit but syntactically lightweight means of propagating errors, such as
Rust's `?` operator, so that error-propagation boilerplate doesn't make it hard
for readers to follow the logic of the success path.

Carbon will not have a special syntax for specifying what kind of errors a
function can emit, such as `noexcept` or
[dynamic exception specifications](https://en.cppreference.com/w/cpp/language/except_spec)
in C++, or `throws` in Java, because that information will be embedded in the
function's return type. Similarly, Carbon errors will be statically typed,
because Carbon return values are statically typed.
