# Remove `bound`

<!--
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

[Pull request](https://github.com/carbon-language/carbon-lang/pull/8110)

<!-- toc -->

## Table of contents

-   [Abstract](#abstract)
-   [Problem](#problem)
-   [Background](#background)
-   [Proposal](#proposal)
-   [Details](#details)
    -   [Standard library interfaces](#standard-library-interfaces)
-   [Rationale](#rationale)
-   [Alternatives considered](#alternatives-considered)

<!-- tocstop -->

## Abstract

Remove the `bound` parameter modifier introduced in
[proposal #5434](https://github.com/carbon-language/carbon-lang/pull/5434), as
it is superseded by the upcoming memory safety design.

## Problem

[Proposal #5434](p005434-ref-parameters-arguments-returns-and-val-returns.md#bound-parameters)
introduced the `bound` parameter modifier to mark parameters whose storage may
be referenced by a function's return value, modeled after Clang's
[`[[clang::lifetimebound]]` attribute](https://clang.llvm.org/docs/AttributeReference.html#lifetimebound).
This served two purposes:

1.  Allowing the compiler to diagnose some use-after-free bugs when a function's
    return is used after the lifetime of a `bound` argument ends.
2.  Distinguishing `ref` parameters that may be referenced by the return value
    (`captures(ret: address, provenance)`) from those that cannot
    (`captures(none)`), so that non-`bound` `ref` parameters could use a
    "move-in-move-out" calling convention.

However, `bound` will be superseded by the upcoming memory safety design, which
replaces this ad-hoc annotation with a more comprehensive lifetime model.
Retaining `bound` in the meantime is a burden for other design work, such as
pending
[proposal #5389](https://github.com/carbon-language/carbon-lang/pull/5389).

## Background

[Proposal #5434: "`ref` parameters, arguments, returns and `val` returns"](/proposals/p005434-ref-parameters-arguments-returns-and-val-returns.md)
introduced `ref` bindings, `ref` and `val` returns, and the `bound`
parameter modifier. Under proposal #5434:

-   Any parameter whose storage could be referenced by the function's return
    had to be marked `bound`.
-   Function declarations in interfaces could mark parameters `bound`, and
    `impl` declarations could omit `bound` from parameters marked `bound` in
    the interface, but could not add `bound` to parameters not marked
    `bound` in the interface.
-   The address of a `ref` binding was specified as
    [LLVM's `noalias`](https://llvm.org/docs/LangRef.html#function-attributes)
    and either
    [`captures(none)`](https://llvm.org/docs/LangRef.html#function-attributes)
    (if not marked `bound`) or
    [`captures(ret: address, provenance)`](https://llvm.org/docs/LangRef.html#function-attributes)
    (if marked `bound`).

## Proposal

We remove the `bound` parameter modifier from Carbon:

-   Parameters are no longer marked `bound` when a function's return may
    reference their storage.
-   The interface and `impl` matching rules for `bound` parameters are removed.
-   `bound` is removed from standard library interfaces such as `Deref`,
    `IndexWith`, `IndirectIndexWith`, `BindToRef`, and `BindToValue`.

For now:

-   The address of a `ref` parameter will no lower with aliasing or capturing
    information.
-   Support for the "move-in-move-out" calling convention on `ref` parameters is
    removed.

These will be restored in a new form with the new memory safety design.

## Details

It remains erroneous behavior to return something that references a local object
whose lifetime ends when the function returns, including local variables,
temporary objects, and `var` parameters.

### Standard library interfaces

Standard library interfaces that previously used `bound` on parameters whose
storage may be referenced by the return value now omit `bound`:

-   `Deref`:

    ```carbon
    interface Deref {
      let Result: type;
      fn Op(ref self) -> ref Result;
    }

    final impl forall [T: type] T* as Deref {
      where .Result = T;
      fn Op(self) -> ref T = "builtin.deref";
    }
    ```

-   [Indexing interfaces](/docs/design/expressions/indexing.md) `IndexWith` and
    `IndirectIndexWith`:

    ```carbon
    interface IndexWith(SubscriptType: type) {
      let ElementType: type;
      fn At(self, subscript: SubscriptType) -> val ElementType;
      fn Ref(ref self, subscript: SubscriptType) -> ref ElementType;
    }

    interface IndirectIndexWith(SubscriptType: type) {
      require Self impls IndexWith(SubscriptType);
      fn Ref(self, subscript: SubscriptType) -> ref ElementType;
    }
    ```

-   [Member binding interfaces](/docs/design/expressions/member_access.md#member-binding-interfaces)
    `BindToValue` and `BindToRef`:

    ```carbon
    interface BindToValue(T: type) {
      extend require impls Bind(T);
      fn Op(self, x: T) -> Result;
    }

    interface BindToRef(T: type) {
      extend require impls Bind(T);
      fn Op(self, ref p: T) -> ref Result;
    }
    ```

## Rationale

-   [Software and language evolution](/docs/project/goals.md#software-and-language-evolution)
    -   There is no need to slow down language evolution at this early stage to
        keep compatibility with previous design proposals since `bound` was
        never implemented in the toolchain.
-   [Practical safety and testing mechanisms](/docs/project/goals.md#practical-safety-and-testing-mechanisms)
    -   Lifetime relationships between parameters and return values will be
        handled cohesively by the memory safety design rather than a separate
        `[[clang::lifetimebound]]`-style marker.

## Alternatives considered

Alternatives to `bound` were considered as part of the upcoming memory safety
design, and are out of scope for this proposal.
