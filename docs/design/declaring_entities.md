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
        -   [Modifier keywords](#modifier-keywords)
        -   [Syntactic matching and scopes](#syntactic-matching-and-scopes)
-   [`extern` and `extern library`](#extern-and-extern-library)
    -   [Valid scopes for `extern`](#valid-scopes-for-extern)
    -   [Effect on indirect imports](#effect-on-indirect-imports)
        -   [Indirect imports of non-`extern` types](#indirect-imports-of-non-extern-types)
    -   [Using imported declarations](#using-imported-declarations)
    -   [Validation for non-owning `extern library` declarations](#validation-for-non-owning-extern-library-declarations)
-   [Alternatives considered](#alternatives-considered)
-   [References](#references)

<!-- tocstop -->

## Overview

Entities may have up to four declarations:

-   An optional, owning forward declaration.
    -   For example, `class MyClass;`.
    -   This must come before the definition. The API file is considered to be
        before the implementation file.
-   A required, owning definition.
    -   For example, `class MyClass { ... }`.
    -   The definition might be the _only_ declaration.
-   An optional, owning declaration in a `match_first` block
    -   This only applies to `impl` declarations.
    -   This must be in the same file as the first owning declaration.
-   An optional, non-owning `extern library "<owning_library>"` declaration
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

An `impl` may appear in at most one `match_first` block, and only once within
it. A declaration in a `match_first` block may be the forward declaration or the
definition. It may also be a fourth declaration that is neither, and which is
only allowed to exist within such a block.

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

#### Modifier keywords

Rules for modifier keywords are adopted from
[proposal #3762](/proposals/p003762-merging-forward-declarations.md#modifier-keywords)
and
[proposal #3980](/proposals/p003980-singular-extern-declarations.md#proposal).
As a rule of thumb, modifier keywords are required when, if prior optional
declarations were removed, the lack of the modifier keyword would change
behavior.

-   `extend` in `extend impl` is only on the declaration in the class body
    (whether that is a forward declaration or definition).
-   Class and interface modifiers other than `extend` (`abstract`, `base`,
    `final`) exist only on the definition, not on the forward declaration.
-   Function modifiers (`override`, `virtual`, `default`, `abstract`, `final`)
    must match between forward declaration and definition (though `abstract`
    functions won't have definitions).
-   If any owning declaration has the `extern` modifier, all owning declarations
    must have it (see
    [proposal #3980](/proposals/p003980-singular-extern-declarations.md#owning-extern-declarations)).
-   Access modifiers (`private` and `protected`) must match across all
    declarations and definitions, including between an `extern library
    "<owning_library>"` declaration and the owning `extern` declaration.

#### Syntactic matching and scopes

-   Two owned declarations _syntactically match_ if the sequence of tokens in
    the declaration following the introducer keyword and the optional scope, up
    to the semicolon or open brace, is identical, except for `unused` modifiers
    on parameters (see
    [proposal #3763](/proposals/p003763-matching-redeclarations.md#proposal)
    and
    [proposal #3980](/proposals/p003980-singular-extern-declarations.md#declarations)).
-   For a qualified declaration (see
    [proposal #3763](/proposals/p003763-matching-redeclarations.md#scope-differences)):
    -   Take the portion of the declaration from the introducer up to the end of
        the scope.
    -   Replace the introducer keyword with the introducer keyword of the scope.
    -   Replace the trailing `.` with a `;`.
    -   The result must be a valid declaration of the scope, ignoring
        restrictions on how often the scope can be redeclared.
-   To redeclare an `impl` after the end of the `class` scope it was declared
    in, that scope may be re-entered as part of the `impl` redeclaration, in the
    same way, except with parentheses around the name of the `impl`, as in
    `impl X.(as Y) { ... }` (see
    [proposal #5366](/proposals/p005366-the-name-of-an-impl-in-class-scope.md#proposal)).
-   For `let` and `var` declarations with a single name binding
    (`let Scope.A: Type = Value;`), the end of the declaration is at the `=` or
    `;` rather than at the `}` or `;`. An arbitrary pattern that is not a single
    binding (`let (A: Type1, B: Type2) = Value;`) does not permit
    redeclarations (see
    [proposal #3763](/proposals/p003763-matching-redeclarations.md#let-and-var-declarations)).

## `extern` and `extern library`

There are two forms of the `extern` modifier:

-   On an owning declaration, `extern` limits access to the definition.
    -   The entity must be directly imported in order to use the definition;
        otherwise it is incomplete (see
        [proposal #3980](/proposals/p003980-singular-extern-declarations.md#impact-on-indirect-imports)).
    -   An `extern library` declaration is optional.
-   On a non-owning declaration, `extern library` allows references to an entity
    without depending on the owning library.
    -   The library name indicates where the entity is defined.
    -   This can be used to improve build performance, such as by splitting out
        a declaration in order to reduce a library's dependencies.

The non-owned `extern library` declarations will only use semantic matching for
redeclarations, not syntactic matching (see
[proposal #3980](/proposals/p003980-singular-extern-declarations.md#no-syntactic-matching-for-extern-library-declarations)).

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

#### Indirect imports of non-`extern` types

As adopted in
[proposal #3980](/proposals/p003980-singular-extern-declarations.md#indirect-imports-of-non-extern-types),
non-`extern` entities are complete if their definition is imported, even if that
import is indirect, as in:

```
library "a";

class C { fn F(); }
```

```
library "b";
import library "a";

fn G() -> C;
```

```
library "c";
import library "b";

// Valid: `C` is complete here, even though it's not in name lookup.
G().F();
```

### Using imported declarations

As adopted in
[proposal #3980](/proposals/p003980-singular-extern-declarations.md#using-imported-declarations),
since `extern library "a" class C;` must be imported by the owning library, we
allow uses of the imported name prior to its declaration within the same file.
This means the following works:

```
library "extern";

extern library "use_extern" class MyType;
```

```
library "use_extern";
import library "extern";

// Uses the `extern library` declaration.
fn Foo(val: MyType*);

extern class MyType {
  fn Bar[ref self: Self]() { Foo(&self); }
}
```

### Validation for non-owning `extern library` declarations

As adopted in
[proposal #3980](/proposals/p003980-singular-extern-declarations.md#validation-for-non-owning-extern-library-declarations),
we offer some validation that the library in `extern library` is correct. When
the owning library is incorrect, it's very likely to be detected in two cases:

-   A compile-time error when the owning library imports the non-owning library,
    when the owning declaration is evaluated.
-   A link-time error as a fallback.

Other cases, such as when both libraries are independently imported, may or may
not be caught, dependent upon the cost of validation.

## Alternatives considered

-   [Other modifier keyword merging approaches](/proposals/p003762-merging-forward-declarations.md#other-modifier-keyword-merging-approaches)
-   [No `extern` keyword](/proposals/p003762-merging-forward-declarations.md#no-extern-keyword)
-   [Looser restrictions on declarations](/proposals/p003762-merging-forward-declarations.md#looser-restrictions-on-declarations)
-   [`extern` naming](/proposals/p003762-merging-forward-declarations.md#extern-naming)
-   [Default `extern` to private](/proposals/p003762-merging-forward-declarations.md#default-extern-to-private)
-   [Opaque types](/proposals/p003762-merging-forward-declarations.md#opaque-types)
-   [Require a library provide its own `extern` declarations](/proposals/p003762-merging-forward-declarations.md#require-a-library-provide-its-own-extern-declarations)
-   [Allow cross-package `extern` declarations](/proposals/p003762-merging-forward-declarations.md#allow-cross-package-extern-declarations)
-   [Use a partially or fully semantic rule](/proposals/p003763-matching-redeclarations.md#use-a-partially-or-fully-semantic-rule)
-   [Use package-wide name poisoning](/proposals/p003763-matching-redeclarations.md#use-package-wide-name-poisoning)
-   [Allow shadowing in implementation file after use in API file](/proposals/p003763-matching-redeclarations.md#allow-shadowing-in-implementation-file-after-use-in-api-file)
-   [Allow multiple non-owning declarations, remove the import requirement, or both](/proposals/p003980-singular-extern-declarations.md#allow-multiple-non-owning-declarations-remove-the-import-requirement-or-both)
-   [Total number of allowed declarations (owning and non-owning)](/proposals/p003980-singular-extern-declarations.md#total-number-of-allowed-declarations-owning-and-non-owning)
    -   [Do not restrict the number of forward declarations](/proposals/p003980-singular-extern-declarations.md#do-not-restrict-the-number-of-forward-declarations)
    -   [Allow up to two declarations total](/proposals/p003980-singular-extern-declarations.md#allow-up-to-two-declarations-total)
    -   [Allow up to four declarations total](/proposals/p003980-singular-extern-declarations.md#allow-up-to-four-declarations-total)
-   [Don't require a modifier on the owning declarations](/proposals/p003980-singular-extern-declarations.md#dont-require-a-modifier-on-the-owning-declarations)
-   [Only require `extern` on the first owning declaration](/proposals/p003980-singular-extern-declarations.md#only-require-extern-on-the-first-owning-declaration)
-   [Separate require-direct-import from non-owning declarations](/proposals/p003980-singular-extern-declarations.md#separate-require-direct-import-from-non-owning-declarations)
-   [Other `extern` syntaxes](/proposals/p003980-singular-extern-declarations.md#other-extern-syntaxes)
-   [Have types with `extern` members re-export them](/proposals/p003980-singular-extern-declarations.md#have-types-with-extern-members-re-export-them)
-   [Require syntactic matching for `extern library` declarations](/proposals/p003980-singular-extern-declarations.md#require-syntactic-matching-for-extern-library-declarations)
-   [Use semantic match for the scope](/proposals/p005366-the-name-of-an-impl-in-class-scope.md#use-semantic-match-for-the-scope)

## References

-   Proposal
    [#3762: Merging forward declarations](https://github.com/carbon-language/carbon-lang/pull/3762)
-   Proposal
    [#3763: Matching redeclarations](https://github.com/carbon-language/carbon-lang/pull/3763)
-   Proposal
    [#3980: Singular `extern` declarations](https://github.com/carbon-language/carbon-lang/pull/3980)
-   Proposal
    [#5337: Interface extension and `final impl` update](https://github.com/carbon-language/carbon-lang/pull/5337)
-   Proposal
    [#5366: The name of an `impl` in `class` scope](https://github.com/carbon-language/carbon-lang/pull/5366)
-   Proposal
    [#7493: Disallow impl in match_first twice](https://github.com/carbon-language/carbon-lang/pull/7493)
