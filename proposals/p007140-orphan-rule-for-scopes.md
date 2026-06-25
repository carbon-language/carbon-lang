# Orphan rule for scopes

<!--
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

[Pull request](https://github.com/carbon-language/carbon-lang/pull/7140)

<!-- toc -->

## Table of contents

-   [Abstract](#abstract)
-   [Problem](#problem)
-   [Background](#background)
-   [Proposal](#proposal)
-   [Details](#details)
    -   [What the orphan rule disallows](#what-the-orphan-rule-disallows)
    -   [Re-entering a nested scope in an `impl` declaration](#re-entering-a-nested-scope-in-an-impl-declaration)
    -   [Interaction with evaluation](#interaction-with-evaluation)
-   [Rationale](#rationale)
-   [Alternatives considered](#alternatives-considered)
    -   [A syntactic check, instead of applying the rule after evaluation](#a-syntactic-check-instead-of-applying-the-rule-after-evaluation)
    -   [Disallowing the anchor name to be in a nested scope](#disallowing-the-anchor-name-to-be-in-a-nested-scope)
    -   [Anchoring to a definition](#anchoring-to-a-definition)

<!-- tocstop -->

## Abstract

Extend the orphan rule to require that at least one name in the type structure
of an `impl` declaration is introduced in, or by, the same scope as the `impl`
declaration, or in a scope nested within the scope of the `impl` declaration.

## Problem

The current orphan rule states:

> Some name from the type structure of an `impl` declaration must be defined in
> the same library as the `impl`, that is some name must be _local_.
>
> ...
>
> We further require anything looking up this `impl` to import the _definitions_
> of all of those names. Seeing a forward declaration of these names is
> insufficient, since you can presumably see forward declarations without seeing
> an `impl` with the definition.

The goal of the orphan rule is:

> Every attempt to use an `impl` will see the exact same `impl` definition,
> making the interpretation and semantics of code consistent no matter its
> context, in accordance with the
> [low context-sensitivity principle](/docs/project/principles/low_context_sensitivity.md).

However with its current wording, it is possible to violate this ambition by
placing a type definition (an owning declaration) in a library api file, then
placing an `impl` decl referring to it in the same library's impl file. This
satisfies the rule that the `impl` is in the same library as the definition of
the type, and that users performing `impl` lookup can import the definition of
the type. But the library impl file sees the `impl` decl in the impl file, while
other libraries do not, creating an inconsistent point of view for which `impl`
will be used.

```carbon
// API file
library "problem";

class C {}
impl forall [T:! type] T as Z where .A = () {}
fn F(T:! Z) -> T.(Z.A);
```

```carbon
// Impl file
impl library "problem";

impl C as Z where .A = {} {}
fn F(T:! Z) -> C.(Z.A) { ... }
```

In this example the return type of `F(C)` will be `{}` in the `problem` library
impl file but will be `()` in other libraries.

Second, this rule allows an `impl` declaration in a generic scope that can be
referred to outside of that generic scope. This exposes the generic bindings
through the `impl` without providing a way for the user to specify their values
by making a specific of the enclosing generic.

```carbon
interface Z { Z1:! type }
class C;
fn F(T:! type) {
  impl C as Z where .Z1 = T;
}
fn G() -> C.(Z.Z1);
```

In this example the value `C.(Z.Z1)` is a generic binding in `F`, but no
specific for the generic `F` is constructed, so the value of `T` cannot be
known. This makes the `impl` declaration available, but unusable.

## Background

-   [Out-of-line `impl`](https://github.com/carbon-language/carbon-lang/blob/2445ad9703c6f481ec371224803922e2fa4e6167/docs/design/generics/details.md#out-of-line-impl)
-   [Orphan rule](https://github.com/carbon-language/carbon-lang/blob/2445ad9703c6f481ec371224803922e2fa4e6167/docs/design/generics/details.md#orphan-rule)
-   [Declaring entities](https://github.com/carbon-language/carbon-lang/blob/4d1a61de29e4a12f2c74298edf499029ad7bc12c/docs/design/declaring_entities.md)
    for the concept of an owning declaration.
-   [`impl` redeclarations](https://github.com/carbon-language/carbon-lang/blob/2445ad9703c6f481ec371224803922e2fa4e6167/proposals/p3763.md#redeclarations)
    for declaring an `impl` into a nested scope.
-   [Scope differences](https://github.com/carbon-language/carbon-lang/blob/2445ad9703c6f481ec371224803922e2fa4e6167/proposals/p3763.md#scope-differences)
    for comparing nested scopes in re-declarations.

## Proposal

We propose to change the orphan rule to:

> Some name from the type structure of an `impl` declaration must be an _anchor
> name_, which is a name that names an entity whose first
> [owning declaration](/docs/design/declaring_entities.md) is in the same file
> as each owning declaration of the `impl`, and either:
>
> -   the scope of the owning declaration directly contains the `impl`
>     declaration, or
> -   the owning declaration is within the scope containing the `impl`
>     declaration, including nested scopes.

## Details

This proposal largely subsumes the existing orphan rule. Libraries can not share
a file, so if an `impl` declaration named something with a definition in the
same file, it would also name something with a definition in the same library.

However the rule is changed to require the anchor name to be an owning
declaration, instead of a definition. That means we allow one thing the previous
rule did not:

-   A declaration of a `class C` in a library api file.
-   An `impl` declaration naming `C` as its anchor name in the library api file.
-   The definition of `C` in the library impl file.

This does not violate the intent of the orphan rule, as all users of `class C`
will have a consistent view of the `impl` declarations that apply to it. This
follows from the fact that all users of `class C` will see its owning
declaration.

In exchange, the new rule disallows placing an `impl` declaration in a library
impl file that only refers to names whose first owning declaration is in the
library api file, since this creates coherence issues.

The new rule further restricts other `impl` declarations so that they can not
only refer to names unrelated to the scope containing the `impl` declaration.

All examples below assume the "z" library is available as follows, and imported:

```carbon
library "z";
interface Z {}
```

There are three cases allowed by the new rule:

1.  An anchor name is introduced by the scope containing the `impl` declaration.

```carbon
fn F() {
  class C {
    impl Self as Z;
  }
}
```

Here the `Self` is resolved to `C`, and `C` is being introduced by the scope
containing the `impl` declaration. If `C` is generic, any use of the impl will
require naming a specific `C`.

2.  A name is introduced in the same scope as the `impl` declaration.

```carbon
fn F() {
  class C;
  impl C as Z;
}
```

Here the `impl` declaration is in the scope of `fn F`, and the first owning
declaration of `C` is within the same scope. All users of the `impl` declaration
will have to be inside `F` since it uses the name `C` which is introduced inside
the scope of `F`. Thus if `F` is generic, all users of the `impl` will share a
consistent view of any generic bindings used by the `impl` declaration.

3.  A name is introduced in a scope nested within the scope containing the `impl`
    declaration.

```carbon
fn F() {
  class C {
    class D {}
  }
  impl C.D as Z;
}
```

Here the `impl` declaration is in the same scope as the first owning declaration
of `C`. The first owning declaration of `D` is within the nested scope of `C`.
Any user of the `impl` declaration will need to see the first owning declaration
of both `C` and `D`, which means it will have to be inside `F`. Thus if `F` is
generic, all users of the `impl` will share a consistent view of any generic
bindings used by the `impl` declaration.

### What the orphan rule disallows

This rule forbids the following:

1.  An `impl` declaration where all names are declared outside the scope
    containing the `impl`.

```carbon
class A {}

class B {
  // ERROR: Neither `A` nor `Z` is defined by or has its owning declaration
  // within the scope `B`.
  impl A as Z {};
}

class C {
  class D {
    // ERROR: Neither `C` nor `Z` is defined by or has its owning declaration
    // within the scope `D`.
    impl C as Z {}
  }
}

fn F() {
  class E {}
  if (true) {
    // ERROR: Neither `E` nor `Z` is defined by or has its owning declaration
    // within this `if` block-scope.
    impl E as Z {}
  }
}

fn G() {
  // ERROR: Neither `A` nor `Z` is defined by or has its owning declaration
  // within the scope `G`.
  impl A as Z {}
}
```

2.  An `impl` declaration where all names have their owning declaration in a
    different file than the `impl`.

```carbon
// Library api file.
library "example";
class A;
class B {}
interface Z {}
```

```carbon
// Library impl file.
impl library "example";

// The definition is in this file, but the owning declaration comes first
// and is in the api file.
class A {}

// ERROR: Neither `A` nor `Z` have their owning declarations in this file.
impl A as Z {}

// ERROR: Neither `B` nor `Z` have their owning declarations in this file.
impl B as Z {}
```

### Re-entering a nested scope in an `impl` declaration

It is possible to re-enter a nested scope by writing a qualified path for the
entire `Type as Interface` expression, such as `impl C.(D as Z)`. This functions
like writing `impl D as Z` within the nested scope `C`, or in other words, by
performing name lookups from the scope of `C`.

By re-entering the nested scope `C`, it becomes the scope containing the `impl`
declaration when applying the orphan rule.

For example, this is equivalent to writing `impl D as Z` inside the class `C`,
which is allowed by the proposed orphan rule.

```carbon
class C {
  class D {}
}
impl C.(D as Z);
```

Whereas it is not allowed to write `impl C as Z` inside the scope of `D`, so it
is also not allowed to write `impl C.D.(C as Z)`.

```carbon
class C {
  class D {}
}
// ERROR: Neither `C` nor `Z` is defined by or has its owning declaration
// within the scope `C.D`.
impl C.D.(C as Z);
```

### Interaction with evaluation

The orphan rule is applied to the `impl` declaration after evaluation. This
means aliases and compile-time functions are evaluated and resolved before the
rule is verified.

The following is rejected, since neither `C` and `Z` is defined by or has its
owning declaration within the scope containing the `impl` declaration.

```carbon
library "a";

musteval fn F() -> auto {
  class C {}
  return C;
}
```

```carbon
library "b";
import library "a";

musteval fn G() -> auto {
  return F();
}

// ERROR: Neither `C` nor `Z` is defined by or has its owning declaration
// within the current scope.
impl G() as Z;
```

But if the class `C` is declared in a function that is within the same scope as
the `impl` declaration, then it is accepted.

```carbon
musteval fn G() -> auto {
  class C {}
  return C;
}

impl G() as Z;
```

## Rationale

By resolving a coherence issue, where different libraries had a different view
of what `impl` to use for a given type, we support the
[low context-sensitivity principle](/docs/project/principles/low_context_sensitivity.md).

By ensuring generic bindings inherited by an `impl` declaration always have a
value that can be made more specific, we support the goal of
[Code that is easy to read, understand, and write](/docs/project/goals.md#code-that-is-easy-to-read-understand-and-write).
It enables providing a clear error when when an invalid `impl` is written that
describes the reason why, instead of an indirect error related to the generic
bindings later.

## Alternatives considered

### A syntactic check, instead of applying the rule after evaluation

This avoids the need to look "through" a function to figure out where a given
entity was defined. But creates other questions instead. It would prevent the
use of evaluation in an `impl` declaration, which would prevent things like
aliases as well, and would generally be fighting against Carbon's eager
evaluation model.

### Disallowing the anchor name to be in a nested scope

The original formulation of this rule required the anchor name to be from the
same scope as the `impl` declaration. This is more restrictive than required to
meet the goals of the rule. Accessing the nested scope of a name in the same
scope as the `impl` declaration still requires anchoring the `impl` declaration
to a name in its containing scope, which is used to qualify the path to the
nested name.

Without allowing nested scopes, the following would be disallowed, without a
good reason:

```carbon
fn F() {
  class C { class D {} }
  // D is declared in a nested scope.
  impl C.D as Z {}

  fn G() -> auto {
    class C {}
    return C;
  }
  // G() resolves to C, which is declared in an nested scope.
  impl G() as Z {}
}
```

### Anchoring to a definition

The original formation of this rule required the definition of the anchor name
to be within the same scope as the `impl` declaration. This matched the wording
of the previous orphan rule. However we noted the coherence issue where a name
is forward declarad in an library api file, but declared in a library impl file.
In this scenario, the only valid place to write the `impl` declaration using
that name as its anchor name is in the api file, with the owning declaration.
Any other choice allows multiple views of which `impl` to apply to a type or
interface involving the anchor name.
