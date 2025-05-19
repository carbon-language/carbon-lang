# The name of an `impl` in `class` scope

<!--
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

[Pull request](https://github.com/carbon-language/carbon-lang/pull/5366)

<!-- toc -->

## Table of contents

-   [Abstract](#abstract)
-   [Problem](#problem)
-   [Background](#background)
-   [Proposal](#proposal)
-   [Details](#details)
    -   [Optional `Self` before `as`](#optional-self-before-as)
-   [Rationale](#rationale)
-   [Alternatives considered](#alternatives-considered)
    -   [Use semantic match for the scope](#use-semantic-match-for-the-scope)

<!-- tocstop -->

## Abstract

```
class C {
  impl as I;
}
```

is redeclared

```
impl C.(as I)
```

for purposes of `match_first`/`impl_priority` blocks and definitions.

## Problem

An `impl` declaration can be declared in `class` scope:

```
class C {
  alias T = bool;
  impl as As(T);
}
```

or in file scope:

```
class C {
  alias T = bool;
}
alias T = i32;
impl C as As(T);
```

These `impl` declarations need to be named so they may be redeclared in a
definition or
[`match_first`/`impl_priority` block](/docs/design/generics/details.md#prioritization-rule).
Under the current rules introduced in
[proposal #1084](https://github.com/carbon-language/carbon-lang/pull/1084) and
modified in [#3763](https://github.com/carbon-language/carbon-lang/pull/3763),
an `impl` redeclaration must match syntactically, and that only works if the
redeclaration enters the same scope as the original declaration.

This problem is demonstrated in the example above. We need some indication
whether to lookup `T` in the file or the class scope, otherwise these both would
be redeclared as `impl C as As(T)`.

```carbon
class C(T:! type) {
  class E {}
  impl E as I(C(T), E);
}

// No ability to do syntactic match
// C(T) does not match C(T:! type)
// C(T).E does not match E
impl forall [T:! type] C(T).E as I(C(T), C(T).E);
```

## Background

The need for forward declarations of entities comes from the
[information accumulation principle](/docs/project/principles/information_accumulation.md).

[Leads issue #1132](https://github.com/carbon-language/carbon-lang/issues/1132)
defined the initial rules for matching forward declarations to their
definitions. Those rules were partially incorporated into the design by
[proposal #1084, "Generics details 9: forward declarations"](/proposals/p1084.md).

A replacement approach was
[discussed on 2024-03-11](https://docs.google.com/document/d/1s3mMCupmuSpWOFJGnvjoElcBIe2aoaysTIdyczvKX84/edit?resourcekey=0-G095Wc3sR6pW1hLJbGgE0g&tab=t.0#heading=h.p69b78lovqb7)
and mentioned in [proposal #3763](/proposals/p3763.md#redeclarations). This is
the approach of syntactic matching and re-entering the same scope. The syntax
adopted by this proposal was first suggested in those.

[Proposal #3762: Merging forward declarations](https://github.com/carbon-language/carbon-lang/pull/3762)
and
[proposal #3980: Singular `extern` declarations](https://github.com/carbon-language/carbon-lang/pull/3980)
refined the rules for forward declarations, including the rules for `extern`
declarations.

[Leads issue #5251: `impl` declarations in a generic class context](https://github.com/carbon-language/carbon-lang/issues/5251)
is (pending resolution) saying that an `impl` declaration in class scope must
use `Self` in a deducible position.

## Proposal

An `impl` declaration is associated with the scope it is first declared in, and
can only be redeclared in that scope, matching all other declarations. Consider
how a function is redeclared outside the scope of a class in which it was
originally defined:

```
class Z(T:! type) {
  // Forward declaration of a function.
  fn F();
}

// Definition of the function that was forward declared.
fn Z(T:! type).F() { ... }
```

To redeclare an `impl` after the end of the scope it was declared in, that scope
may be re-entered as part of the `impl` redeclaration, in the same way, except
with parentheses around the name of the `impl`, as in:

```carbon
class X {
  // Forward declaration that `X impls Y`:
  impl as Y;
}

// Definition of the `impl` that `X impls Y`
// that was forward declared in `X`:
impl X.(as Y) { ... }
```

More generally, in a `class` scope

```carbon
class __X__ {
  impl __Y__;
}
```

is redeclared `impl __X__.(__Y__)` outside of that `class` scope. Here `__X__`
is whatever sequence of tokens appears in that position in the `class`
declaration, and `__Y__` is the sequence of tokens in the `impl` declaration.
These declarations are matched syntactically, and anything in `__Y__` is
interpreted as if it appeared in the scope of `__X__` like it was first
declared.

## Details

Here are some examples of this in practice:

```carbon
class F {}

class A {
  impl as As(i32);
  impl Self as As(bool);
  impl A as As(f64);

  impl F as As(A);

  class G {}
  impl G as As(A);
}

impl A.(as As(i32)) { ... }
impl A.(Self as As(bool)) { ... }
impl A.(A as As(f64)) { ... }
impl A.(F as As(A)) { ... }
impl A.(G as As(A)) { ... }
```

Parameterized classes:

```carbon
class B(T:! type) {
  impl B(i32) as AddWith(A(T));
}

impl B(T:! type).(B(i32) as AddWith(A(T))) { ... }
```

Parameterized impl:

```carbon
class C {
  impl forall [T:! type] as I(T);
}

impl C.(forall [T:! type] as I(T));
```

Putting the `forall` inside the parens both simplifies the syntactic match and
means that any mentions of names in that clause are in scope. For example:

```
class D(T:! type) {
  class E {}
  impl forall [U:! J(E)] as I(U);
}

impl D(T:! type).(forall [U:! J(E)] as I(U));
```

Notice how the `E` in the constraint on `U` is found in the `D(T:! type)` scope.

Nested classes:

```
class C1 {
  class C2 {
    class C3 {
      impl as I;
      class C4 {}
    }
  }
}

// Defining impl that was forward declared within the `C3` definition:
impl C1.C2.C3.(as I) { ... }

// Defining a new impl:
impl C1.C2.C3.C4 as I { ... }
```

Notice that we don't know which form will be used until we see:

-   the open paren (`(`) after a `.`,
-   a parameter pattern,
-   a non-parameter argument, or
-   the `as`.

### Optional `Self` before `as`

The normalization to add `Self` before `as` when that type is omitted before
performing syntactic match is preserved from proposals
[#1084](https://github.com/carbon-language/carbon-lang/pull/1084) and
[#3763](https://github.com/carbon-language/carbon-lang/pull/3763). Note that the
`Self` is inserted in the parentheses when the `as` appears there. So:

```
class A {
  // First impl declaration is equivalent
  // to `impl Self as As(i32);`
  impl as As(i32);

  // Second impl declaration.
  impl Self as As(bool);
}

// Redeclaration of the first impl declaration.
impl A.(Self as As(i32)) { ... }

// Since this is equivalent to
// `impl A.(Self as As(bool)) { ... }`, is a
// valid redeclaration of the second impl
// declaration.
impl A.(as As(bool)) { ... }
```

## Rationale

The need for this proposal comes from supporting forward declarations for the
[information accumulation principle](/docs/project/principles/information_accumulation.md).
The specifics of this proposal were chosen comply with these
[Carbon goals](/docs/project/goals.md):

-   Unambiguous rules that are simple to implement benefit
    [language tools and ecosystem](/docs/project/goals.md#language-tools-and-ecosystem).
-   [Code that is easy to read, understand, and write](/docs/project/goals.md#code-that-is-easy-to-read-understand-and-write)
    benefits by having rules that are simple to state and validate. The
    syntactic match rule means redeclarations are close to a copy of the
    original declaration, which makes authoring straightforward and automatable.

## Alternatives considered

### Use semantic match for the scope

[Leads issue #5367: `impl` in `class` redeclaration syntax with parameterization](https://github.com/carbon-language/carbon-lang/issues/5367)
considered an alternative where

```carbon
// Alternative
impl forall [T:! type] B(T).(as I) { ... }
```

instead of:

```carbon
// This proposal
impl B(T:! type).(as I) { ... }
```

It avoided putting a `forall` clause inside the parentheses, but that both
greatly limited the syntactic matching that we could do, and meant examples
like:

```carbon
// This proposal
impl D(T:! type).(forall [U:! J(E)] as I(U));
```

had to instead be written with more qualfiers:

```carbon
// Alternative
impl forall [T:! type, U:! J(D(T).E)] D(T).(as I(U));
```

It is not just helpful for the compiler: being able to make fewer and more
mechanical changes after copy-pasting the `impl` declaration to make the
redeclaration makes authoring the code easier, and simplifies tooling to
automate the process.
