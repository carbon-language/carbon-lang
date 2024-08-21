# Declaring entities

<!--
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

<!-- toc -->

## Table of contents

-   [Overview](#overview)
-   [Matching redeclarations of an entity](#matching-redeclarations-of-an-entity)
    -   [Details](#details)
-   [`extern` and `extern library`](#extern-and-extern-library)
    -   [Valid scopes for `extern`](#valid-scopes-for-extern)
    -   [Effect on indirect imports](#effect-on-indirect-imports)
-   [Alternatives considered](#alternatives-considered)
-   [References](#references)

<!-- tocstop -->

## Overview

Entities may have up to three declarations:

-   An optional, owning forward declaration.
    -   For example, `class MyClass;`.
    -   This must come before the definition. The API file is considered to be
        before the implementation file.
-   A required, owning definition.
    -   For example, `class MyClass { ... }`.
    -   The definition might be the _only_ declaration.
-   An optional, non-owning `extern library "<owning_library>"` declaration.
    -   For example, `extern library "OtherLibrary" class MyClass;`.
    -   It must be in a separate library from the definition.
    -   The owning library's API file must import the `extern` declaration, and
        must also contain a declaration.
    -   The owning library's declarations must have the `extern` modifier
        (without `library`).
        -   For example, `extern class MyClass;`.

For example, a library can have a forward declaration of an entity in the API
file, and use the implementation file for the entity's definition. Putting the
definition in an implementation file this way can reduce the dependencies for
API file evaluation, improving compile time. This is commonly done with
functions. For example:

```
library "MyLibrary";

fn DoSomething();
```

```
impl library "MyLibrary";

fn DoSomething() {
  ...
}
```

## Matching redeclarations of an entity

In order to determine whether two redeclarations refer to the same entity, we
apply the rules:

-   Two named declarations _declare the same entity_ if they have the same scope
    and the same name. This includes imported declarations.
-   When two named declarations declare the same entity, the second is said to
    be a _redeclaration_.
-   Two owned declarations _differ_ if they don't syntactically match.
    -   Otherwise, if one is a non-owned `extern library` declaration,
        declarations differ if they don't match semantically.
-   The program is invalid if it contains two declarations of the same entity
    that differ.

```carbon
class A {
  // This function will be redeclared in order to provide a definition.
  fn F(n: i32);
}

// ✅ Valid: The declaration matches syntactically.
fn A.F(n: i32) {}

// ❌ Invalid: The parameter name differs.
fn A.F(m: i32) {}

// ❌ Invalid: The parameter type differs syntactically.
fn A.F(n: (i32)) {}
```

### Details

TODO: Figure out what details to pull from
[#3762](https://github.com/carbon-language/carbon-lang/pull/3762) and
[#3763](https://github.com/carbon-language/carbon-lang/pull/3763).

## `extern` and `extern library`

There are two forms of the `extern` modifier:

-   On an owning declaration, `extern` limits access to the definition.
    -   The entity must be directly imported in order to use of the definition.
    -   An `extern library` declaration is optional.
-   On a non-owning declaration, `extern library` allows references to an entity
    without depending on the owning library.
    -   The library name indicates where the entity is defined.
    -   This can be used to improve build performance, such as by splitting out
        a declaration in order to reduce a library's dependencies.

For example, a use of both might look like:

```
library "owner";

// This `import` is required due to the `extern library`, but we also make use
// of `MyClassFactory` below. This is a circular use of `MyClass` that we
// couldn't split between libraries without `extern`.
import library "factory";

extern class MyClass {
  fn Make() -> MyClass* {
    return MyClassFactory();
  }

  var val: i32 = 0;
}
```

```
library "factory";

// Declares `MyClass` so that `MyClassFactory` can return it.
extern library "owner" class MyClass;

fn MyClassFactory(val: i32) -> MyClass*;
```

```
impl library "factory";

// Imports the definition of `MyClass`.
import library "owner";

extern fn MyClassFactory(val: i32) -> MyClass* {
  var c: MyClass* = new MyClass();
  c->val = val;
  return c;
}
```

### Valid scopes for `extern`

The `extern` modifier is only valid on namespace-scoped entities, including in
the file scope. In other words, `class C { extern fn F(); }` is invalid.

### Effect on indirect imports

Indirect imports won't see the definition of an `extern` entity. We expect this
to primarily affect return types of functions. If an incomplete type is
encountered this way, it can be resolved by directly importing the definition.
For example:

```
library "type";

// Because this is `extern`, the definition must be directly imported.
extern class MyType { var x: i32 }
```

```
library "make_type";

import library "type";

// Here we have a function which returns the type.
fn MakeMyType() -> MyType*;
```

```
library "invalid_use";

import library "make_type";

fn InvalidUse() -> i32 {
  // ❌ Invalid: `MyType` is incomplete because it's `extern` and not directly
  // imported. `x` cannot be accessed.
  return MakeMyType()->x;
}
```

```
library "valid_use";

import library "make_type";

// ✅ Valid: By directly importing the definition, we can now access `x`.
import library "type";

fn ValidUse() -> i32 {
  return MakeMyType()->x;
}
```

## Alternatives considered

-   [Other modifier keyword merging approaches](/proposals/p3762.md#other-modifier-keyword-merging-approaches)
-   [No `extern` keyword](/proposals/p3762.md#no-extern-keyword)
-   [Looser restrictions on declarations](/proposals/p3762.md#looser-restrictions-on-declarations)
-   [`extern` naming](/proposals/p3762.md#extern-naming)
-   [Default `extern` to private](/proposals/p3762.md#default-extern-to-private)
-   [Opaque types](/proposals/p3762.md#opaque-types)
-   [Require a library provide its own `extern` declarations](/proposals/p3762.md#require-a-library-provide-its-own-extern-declarations)
-   [Allow cross-package `extern` declarations](/proposals/p3762.md#allow-cross-package-extern-declarations)
-   [Use a partially or fully semantic rule](/proposals/p3763.md#use-a-partially-or-fully-semantic-rule)
-   [Use package-wide name poisoning](/proposals/p3763.md#use-package-wide-name-poisoning)
-   [Allow shadowing in implementation file after use in API file](/proposals/p3763.md#allow-shadowing-in-implementation-file-after-use-in-api-file)
-   [Allow multiple non-owning declarations, remove the import requirement, or both](/proposals/p3980.md#allow-multiple-non-owning-declarations-remove-the-import-requirement-or-both)
-   [Total number of allowed declarations (owning and non-owning)](/proposals/p3980.md#total-number-of-allowed-declarations-owning-and-non-owning)
    -   [Do not restrict the number of forward declarations](/proposals/p3980.md#do-not-restrict-the-number-of-forward-declarations)
    -   [Allow up to two declarations total](/proposals/p3980.md#allow-up-to-two-declarations-total)
    -   [Allow up to four declarations total](/proposals/p3980.md#allow-up-to-four-declarations-total)
-   [Don't require a modifier on the owning declarations](/proposals/p3980.md#dont-require-a-modifier-on-the-owning-declarations)
-   [Only require `extern` on the first owning declaration](/proposals/p3980.md#only-require-extern-on-the-first-owning-declaration)
-   [Separate require-direct-import from non-owning declarations](/proposals/p3980.md#separate-require-direct-import-from-non-owning-declarations)
-   [Other `extern` syntaxes](/proposals/p3980.md#other-extern-syntaxes)
-   [Have types with `extern` members re-export them](/proposals/p3980.md#have-types-with-extern-members-re-export-them)
-   [Require syntactic matching for `extern library` declarations](/proposals/p3980.md#require-syntactic-matching-for-extern-library-declarations)

## References

-   Proposal
    [#3762: Merging forward declarations](https://github.com/carbon-language/carbon-lang/pull/3762)
-   Proposal
    [#3763: Matching redeclarations](https://github.com/carbon-language/carbon-lang/pull/3763)
-   Proposal
    [#3980: Singular `extern` declarations](https://github.com/carbon-language/carbon-lang/pull/3980)
