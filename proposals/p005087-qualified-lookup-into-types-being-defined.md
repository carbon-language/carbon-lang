# Qualified lookup into types being defined

<!--
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

[Pull request](https://github.com/carbon-language/carbon-lang/pull/5087)

<!-- toc -->

## Table of contents

-   [Abstract](#abstract)
-   [Problem](#problem)
-   [Background](#background)
-   [Proposal](#proposal)
-   [Details](#details)
    -   [Uses requiring a complete type](#uses-requiring-a-complete-type)
    -   [Lookups into an incomplete generic](#lookups-into-an-incomplete-generic)
-   [Rationale](#rationale)
-   [Alternatives considered](#alternatives-considered)
    -   [Require completeness for qualified name lookup](#require-completeness-for-qualified-name-lookup)
    -   [Do not require a definition for name lookup](#do-not-require-a-definition-for-name-lookup)

<!-- tocstop -->

## Abstract

Allow qualified name lookup into classes and interfaces as soon as we reach the
`{` of the definition, rather than disallowing such lookups until we reach the
`}`.

## Problem

We allow unqualified lookups within a type definition to find names that were
already declared, but not qualified lookups:

```carbon
class A {
  class B {}
  // ✅ OK, `B` names `A.B`.
  fn F() -> B;
  // ❌ Error (before this proposal): `A` is not complete.
  fn G() -> A.B;
}

interface I {
  let T:! type;
  // ✅ OK, `T` names `Self.T`.
  fn F() -> T;
  // ❌ Error (before this proposal): type `I` of `Self` is not complete.
  fn G() -> Self.T;
}
```

This is inconsistent and prevents useful code:

```carbon
interface Container {
  let ValueType:! type
  // ❌ Error (before this proposal): type `Container` of `.Self`
  // is not complete in implicit access to `.Self.ValueType`.
  // Implicit access to `Self.ValueType` is OK though.
  let SliceType:! Container where .ValueType == ValueType;
}
```

An additional inconsistency is that namespaces already support qualified lookup
before the full list of names is known -- indeed, the full list of names in a
namespace is never known.

## Background

Proposal
[#3763](https://github.com/carbon-language/carbon-lang/blob/trunk/proposals/p3763.md)
introduces a name poisoning rule:

> In a declarative scope, it is an error if a name is first looked up and not
> found, and later introduced.

This is achieved by _poisoning_ a name in a scope when we perform a failed
lookup for that name in that scope, and diagnosing if the name is later declared
in a scope where it is poisoned.

With that rule, there is no risk in allowing name lookups into a scope to
succeed even before the scope is complete. If the name lookup's meaning would be
changed by a later declaration, an error is issued.

The motivation for this proposal and the proposed rule change were discussed in
open discussion on
[2025-02-21](https://docs.google.com/document/d/1Iut5f2TQBrtBNIduF4vJYOKfw7MbS8xH_J01_Q4e6Rk/edit?resourcekey=0-mc_vh5UzrzXfU4kO-3tOjA&tab=t.0#heading=h.ix77am1xk6po),
[2025-02-27a](https://docs.google.com/document/d/1Iut5f2TQBrtBNIduF4vJYOKfw7MbS8xH_J01_Q4e6Rk/edit?resourcekey=0-mc_vh5UzrzXfU4kO-3tOjA&tab=t.0#heading=h.ank09kkr0tnn),
and
[2025-03-03b](https://docs.google.com/document/d/1Iut5f2TQBrtBNIduF4vJYOKfw7MbS8xH_J01_Q4e6Rk/edit?resourcekey=0-mc_vh5UzrzXfU4kO-3tOjA&tab=t.0#heading=h.19yyjdek3asm).

## Proposal

Allow name lookups into a type once it is defined, even before it is complete.

## Details

We adopt the following terminology:

-   An entity is _defined_ at the point where we reach the `{` of its
    definition. _Exception:_ A namespace is defined by its first declaration.
-   An entity is _complete_ at the point where we reach the `}` of its
    definition.

Instead of qualified name lookup into the scope of a type requiring the type to
be complete, we now only require it to be defined. Therefore, qualified lookup
within the braces of the type definition are now permitted.

Such a name lookup only finds names that were declared prior to the lookup, in
line with the
[information accumulation principle](/docs/project/principles/information_accumulation.md).
If the name is first declared after the point at which it is looked up, the
later declaration of the name is rejected by to the poisoning rule [described earlier](#background].

```carbon
base class A {
  class Inner {}
}

class B {
  extend base: A;
  // ✅ OK for now, `Inner` names `A.Inner` because
  // no results were found directly in scope of `B`.
  var i: B.Inner;
  // ❌ Error: name `Inner` is poisoned due to prior lookup in this scope.
  class Inner {}
}
```

### Uses requiring a complete type

While this proposal permits qualified name lookup into types that are
incomplete, many uses of the names found by such lookups will still require
completeness. For example:

-   Instance binding for a class instance member requires a complete type.

    ```carbon
    class X[T:! type](v: T) {}

    class A {
      var n: i32;
      fn F() -> A;
      // ❌ Error: `A` is not complete.
      var m: X(fn (A a) => F().n);
    }
    ```

-   Some uses of interfaces require the interface to be complete, although
    determining which uses require this is outside the scope of this proposal.

### Lookups into an incomplete generic

It is possible to perform qualified name lookup into a generic type before the
type is complete.

```carbon
class X[T:! type](x: T) {}

class A(T:! type) {
  fn F() -> A(T);
  // OK, argument to `X` is the `F` function in `A(i32)`.
  var v: X(A(i32).F);
}
```

This may require extra work to handle in the toolchain. In particular, we
currently cannot form a specific for the class definition until the class is
complete. We have at least two viable implementation strategies to handle this:

-   Ensure all declarations that can be nested within a type have their own
    corresponding generic. This is already the case for most such declarations,
    with field declarations and alias declarations being notable exceptions.
-   Support evaluating an incomplete eval block for a generic, and finish
    evaluating each incomplete eval block at the end of the generic definition.

## Rationale

-   [Code that is easy to read, understand, and write](/docs/project/goals.md#code-that-is-easy-to-read-understand-and-write)
    -   This change removes an ergonomic hurdle and an inconsistency in the
        language rules.
-   [Interoperability with and migration from existing C++ code](/docs/project/goals.md#interoperability-with-and-migration-from-existing-c-code)
    -   This rule is more closely aligned with the C++ rule for lookup of class
        members within the class definition, which allows members to be named
        before the class is complete.

## Alternatives considered

### Require completeness for qualified name lookup

The status quo ante is to disallow qualified access into types until they are
complete. The rationale for rejecting that alternative is described in this
proposal.

### Do not require a definition for name lookup

We could permit qualified and unqualified name lookup into types that are merely
declared and not yet defined. Such a lookup would find nothing. This allowance
could be used in a case where a type's scope can be extended before the type is
defined, which is itself currently not permitted. For example:

```carbon
interface I;
interface J {
  let T:! type;
}

interface K {
  extend I;
  extend J;
  let Unqual:! T;
  let Qual:! K.T;
}
```

We could choose to permit this, and make the lookups for the name `T` in the
scope of `I` find nothing and poison the name `T` in `I`, so that the name can
be resolved immediately to `J.T`. If `I` were to later introduce a name `T`,
that would result in an error.

However, this seems like it may be a step too far, and isn't justified by the
motivations for this change. From an implementation standpoint, it would also
require tracking a list of poisoned names in a type that doesn't even have a
scope yet, although that is likely straightforward to handle.
