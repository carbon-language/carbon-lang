# Unqualified names

<!--
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

<!-- toc -->

## Table of contents

-   [Overview](#overview)
-   [Alternatives considered](#alternatives-considered)
-   [References](#references)

<!-- tocstop -->

## Overview

An _unqualified name_ is a [word](../lexical_conventions/words.md) other than a
keyword that is not preceded by a `.`. An unqualified name is looked up in all
enclosing scopes, and refers to the entity found by
[unqualified name lookup](../name_lookup.md). If the lookup finds more than one
entity in the enclosing scopes, the program is invalid due to ambiguity.

```
// #1
class MyClass { fn Function(); }

namespace SomeNamespace;

// #2
class SomeNamespace.MyClass {
  fn Function();

  // #3
  class MyClass {
    fn Function();
  }
}

// OK, `MyClass` finds only #1.
fn MakeMyClass() -> MyClass;

// OK, qualified name finds only #2.
fn MakeClassInNamespace() -> SomeNamespace.MyClass;

// Ambiguous. Enclosing scopes are SomeNamespace and package scope;
// #1 found in package scope, and #2 found in namespace scope.
fn SomeNamespace.MakeMyClass() -> MyClass;

// OK, qualified name.
fn SomeNamespace.MyClass.MyClass.Function() {
  // Ambiguous, could be #1, #2, or #3.
  var a: MyClass;

  // Ambiguous, first `MyClass` could be #1, #2, or #3, even though only
  // #2 has a member named `MyClass`.
  var b: MyClass.MyClass;

  // OK
  var c: SomeNamespace.MyClass.MyClass;

  // OK
  var d: Self;
}
```

## Alternatives considered

-   [FIXME](/docs/proposals/p0845.md#FIXME)

## References

-   Proposal
    [#845: unqualified names](https://github.com/carbon-language/carbon-lang/pull/845).
