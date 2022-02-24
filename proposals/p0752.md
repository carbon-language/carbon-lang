# `api` file default-`public`

<!--
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

[Pull request](https://github.com/carbon-language/carbon-lang/pull/752)

<!-- toc -->

## Table of contents

-   [Problem](#problem)
-   [Background](#background)
-   [Proposal](#proposal)
-   [Rationale based on Carbon's goals](#rationale-based-on-carbons-goals)
-   [Alternatives considered](#alternatives-considered)
    -   [Default `api` to private](#default-api-to-private)
    -   [Default `impl` to public](#default-impl-to-public)
    -   [Make keywords either optional or required in separate definitions](#make-keywords-either-optional-or-required-in-separate-definitions)

<!-- tocstop -->

## Problem

Question for leads
[#665: private vs public _syntax_ strategy, as well as other visibility tools like external/api/etc.](https://github.com/carbon-language/carbon-lang/issues/665)
decided that methods on classes should default to public. Should `api` echo the
similar strategy?

## Background

-   In C++, `struct` members default public, while `class` members default
    `public`.
-   In proposal
    [#107: Code and name organization](https://github.com/carbon-language/carbon-lang/pull/107),
    an `api` keyword was used to indicate public APIs within an `api` file.
-   In [#665](https://github.com/carbon-language/carbon-lang/issues/665), it was
    decided that Carbon class members should default `public`.
    -   This issue was reopened to discuss alternatives in this proposal.

## Proposal

APIs in the `api` file should default public, without need for an additional
`api` keyword. `private` may be specified to designate APIs that are internal to
the library, and only visible to `impl` files.

Nothing is necessary within `impl` files, and APIs there will be private unless
forward declared in the `api` file.

## Rationale based on Carbon's goals

-   [Code that is easy to read, understand, and write](/docs/project/goals.md#code-that-is-easy-to-read-understand-and-write):
    It will be easier for developers to understand code if APIs have
    semantically similar behavior when comparing the visibility of class methods
    to other code, and the library to other packages.

## Alternatives considered

### Default `api` to private

Default private is what was implied by `api`, and was the previous state.

Advantages:

-   Decreases the likelihood that developers will accidentally expose APIs,
    because it's an explicit choice.
-   Can move functions between `api` and `impl` without visibility changing.

Disadvantages:

-   The `api` file's primary purpose is to expose APIs, and so it may be more
    natural for developers to assume things there are public.
-   Inconsistent with "default public" behavior on classes.

We are switching to default public in `api` files for consistency with class
behaviors.

### Default `impl` to public

Noting that we default `api` to public, we could similarly default `impl` to
public.

Advantages:

-   Can move functions between `api` and `impl` without visibility changing.

Disadvantages:

-   Everything in an `impl` file must be private unless it's a separate
    definition of an `api` declaration. As a consequence, everything declared in
    the `impl` file would need to be explicitly `private`.

In order to avoid the toil of explicitly declaring everything in the `impl` as
`private`, `impl` will be `private` by default. As a consequence of being the
default behavior, no `private` should be specified, just as `public` is not
allowed in `api` files.

Note the visibility behavior can be described as making declarations the most
visible possible for its context, which in `api` files is `public`, and in
`impl` is `private`.

### Make keywords either optional or required in separate definitions

When a prior declaration exists, keywords are _disallowed_ in separate
definitions. We could instead allow keywords, making them either optional or
required. This would allow developers to determine visibility when reading a
definition.

The downside of this is that optional keywords can be confusing. For example:

-   `api` file:

    ```
    class Foo {
      private fn Bar();
      private fn Wiz();
    };
    ```

-   `impl` file:

    ```
    fn Foo.Bar() { ...impl... }
    private fn Foo.Wiz() { ...impl... }
    fn Baz() { ...impl... }
    ```

In an "optional" setup, the above is valid code. However, the lack of a
`private` keyword on `Foo.Bar` may lead a developer to conclude that it's public
without checking the `api` file (particularly because `Foo.Wiz` is explicitly
private), when it's actually private. This is an accident that could also occur
on refactoring; for example, removing the keyword on the `impl` version of
`Foo.Wiz` would be valid but does not make it public.

A response may be to make keywords required to match, so that reading the `impl`
file would have a compiled guarantee of correctness, avoiding confusion.
However, consider a similar example:

-   `api` file:

    ```
    class Foo {
      fn Bar();
      private fn Wiz();
    };
    ```

-   `impl` file:

    ```
    fn Foo.Bar() { ...impl... }
    private fn Foo.Wiz() { ...impl... }
    fn Baz() { ...impl... }
    ```

In this example, `Foo.Bar` is public, and that may lead developers to conclude
that `Baz` is public. This could be corrected by requiring `private` on `Baz`,
but we are hesitant to do that per
[Default `impl` to public](#default-impl-to-public).

There is still some risk of confusion if the forward declaration and separate
definition are both in the `api` file. For example:

```
private fn PrintLeaves(Node node);

fn PrintNode(Node node) {
  Print(node.value);
  PrintLeaves(node);
);

fn PrintLeaves(Node node) {
  for (Node leaf : node.leaves) {
    PrintNode(leaf);
  }
}
```

In this, a reader may read the `PrintLeaves` definition and incorrectly conclude
that it is implicitly `public` because (a) it has no keywords and (b) it is in
the `api` file. This will be addressed as part of
[Open question: Calling functions defined later in the same file #472](https://github.com/carbon-language/carbon-lang/issues/472#issuecomment-915407683).

Overall, the decision to _disallow_ keywords on separate definitions means that
`impl` files shouldn't have any visibility keywords at the file scope (they will
on classes), which is considered a writability improvement while keeping the
`api` as the single source of truth for `public` entities, addressing
readability.
