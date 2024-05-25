# Merging forward declarations

<!--
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

[Pull request](https://github.com/carbon-language/carbon-lang/pull/3762)

<!-- toc -->

## Table of contents

-   [Abstract](#abstract)
-   [Problem](#problem)
-   [Background](#background)
    -   [Prior discussion](#prior-discussion)
    -   [Forward declarations in current design](#forward-declarations-in-current-design)
    -   [ODR (One definition rule)](#odr-one-definition-rule)
    -   [Requiring matching declarations to merge](#requiring-matching-declarations-to-merge)
-   [Proposal](#proposal)
-   [Details](#details)
    -   [`extern` keyword](#extern-keyword)
    -   [When declarations are allowed](#when-declarations-are-allowed)
        -   [No forward declarations after declarations](#no-forward-declarations-after-declarations)
        -   [Files must either use an imported declaration or declare their own](#files-must-either-use-an-imported-declaration-or-declare-their-own)
        -   [Libraries cannot both define an entity and declare it `extern`](#libraries-cannot-both-define-an-entity-and-declare-it-extern)
        -   [`impl` files with a forward declaration must contain the definition](#impl-files-with-a-forward-declaration-must-contain-the-definition)
        -   [Type scopes may contain both a forward declaration and definition](#type-scopes-may-contain-both-a-forward-declaration-and-definition)
    -   [Using `extern` declarations in `extern impl`](#using-extern-declarations-in-extern-impl)
    -   [`impl` lookup involving `extern` types](#impl-lookup-involving-extern-types)
    -   [Modifier keywords](#modifier-keywords)
-   [Rationale](#rationale)
-   [Alternatives considered](#alternatives-considered)
    -   [Other modifier keyword merging approaches](#other-modifier-keyword-merging-approaches)
    -   [No `extern` keyword](#no-extern-keyword)
    -   [Looser restrictions on declarations](#looser-restrictions-on-declarations)
    -   [`extern` naming](#extern-naming)
    -   [Default `extern` to private](#default-extern-to-private)
    -   [Opaque types](#opaque-types)
    -   [Require a library provide its own `extern` declarations](#require-a-library-provide-its-own-extern-declarations)
    -   [Allow cross-package `extern` declarations](#allow-cross-package-extern-declarations)
-   [Appendix](#appendix)
    -   [Migration of C++ forward declarations](#migration-of-c-forward-declarations)

<!-- tocstop -->

## Abstract

-   Add the `extern` keyword for forward declarations in libraries that don't
    provide the definition.
-   Treat repeated forward declarations as redundant.
    -   Allow them when they prevent a dependence on an imported name.
-   Clarify rules for when modifier keywords should be on forward declarations
    and definitions.

## Problem

A forward declaration can be merged with a definition when they match. However,
there is ambiguity about behavior:

-   Whether a forward declaration can be repeated, including after a definition.
-   Whether a keyword is required to make a forward declaration separate from
    the implementing library.
-   Whether modifier keywords need to match between forward declarations and
    definitions.

## Background

### Prior discussion

-   [Proposal #1084: Generics details 9: forward declarations](https://github.com/carbon-language/carbon-lang/pull/1084)
    covered specifics for `impl` and `interface`.
-   [Issue #1132: How do we match forward declarations with their definitions?](https://github.com/carbon-language/carbon-lang/issues/1132)
-   [Issue #3384: are abstract / base specifiers permitted or required on class forward declarations?](https://github.com/carbon-language/carbon-lang/issues/3384)
    asks one question about which keywords exist on declarations versus
    definitions. There has also been discussion about what happens for member
    functions and other situations. At present there is no decision on #3384.
-   The
    [2023-11-21 toolchain meeting](https://docs.google.com/document/d/1s3mMCupmuSpWOFJGnvjoElcBIe2aoaysTIdyczvKX84/edit?resourcekey=0-G095Wc3sR6pW1hLJbGgE0g&tab=t.0#heading=h.y377x4v44h2c)
    had some early discussion regarding forward declarations and the need for a
    syntactic difference, which evolved to `extern` here.
-   [On Discord's #syntax on 2024-01-29](https://discord.com/channels/655572317891461132/709488742942900284/1201693766449102878),
    there was discussion about the question "What is the intended interaction
    between declaration modifiers and qualified declaration names?"
-   [On Discord's #syntax on 2024-02-23](https://discord.com/channels/655572317891461132/709488742942900284/1210694116955000932),
    there was discussion that started about how `alias` would resolve forward
    declaration conflicts. This continued to forward declarations in general,
    and has discussion about `extern` use.
    -   This discussion led to a substantial fraction of this proposal.

### Forward declarations in current design

Example rules for forward declarations in the current design:

-   [High-level](https://github.com/carbon-language/carbon-lang/blob/trunk/docs/design/README.md#declarations-definitions-and-scopes)
-   [Classes](https://github.com/carbon-language/carbon-lang/blob/trunk/docs/design/classes.md#forward-declaration)
-   [Functions](https://github.com/carbon-language/carbon-lang/blob/trunk/docs/design/functions.md#function-declarations)
-   Generics:
    -   [`impl`](https://github.com/carbon-language/carbon-lang/blob/trunk/docs/design/generics/details.md#forward-impl-declaration)
    -   [`interface`](https://github.com/carbon-language/carbon-lang/blob/trunk/docs/design/generics/details.md#declaring-interfaces-and-named-constraints)
-   [Matching and agreeing](https://github.com/carbon-language/carbon-lang/blob/trunk/docs/design/generics/details.md#matching-and-agreeing)

### ODR (One definition rule)

[C++'s ODR](https://en.cppreference.com/w/cpp/language/definition) requires each
entity to have only one definition. C++ has trouble detecting issues at compile
time, and linking has trouble catching linker issues too. ODR violation
detection is an active problem
(https://maskray.me/blog/2022-11-13-odr-violation-detection).

This similarly applies to Carbon. In Carbon, only one library can define an
entity; other libraries can make forward declarations of it, but those don't
constitute a definition. Part of the goal in declaration syntax (`extern` in
particular) is to be able to better diagnose ODR violations based on
declarations.

### Requiring matching declarations to merge

When two declarations have the same name, either within the same file or through
imports, they need to "match", or will be diagnosed as a conflict. This is laid
out at under
[Matching and agreeing](https://github.com/carbon-language/carbon-lang/blob/trunk/docs/design/generics/details.md#matching-and-agreeing)
in generics. This proposal does not significantly affect these type and name
restrictions.

For example:

-   `fn F(); fn F() {}` could merge because the function signatures match,
    forward declaring a function is valid, and only one definition is provided.
-   `fn F(); fn F(x: i32);` won't merge because they differ in parameters,
    resulting in a diagnostic.

## Proposal

1.  Add `extern` declarations.
    1.  `extern` is used to mark forward declarations of entities defined in a
        different library.
        -   The defining library must declare the entity as public in an `api`
            file.
        -   It is invalid for the defining library to declare as `extern`.
        -   `extern` must only be used for declarations in libraries that don't
            contain the definition.
    2.  In modifiers, `extern` comes immediately after access control keywords,
        and before other keywords.
    3.  Declaring an entity as `extern` anywhere in a file means it must be used
        as `extern` _throughout_ that file.
        -   The entity cannot be used _at all_ prior to the `extern`
            declaration, even if there is an imported declaration.
        -   This allows refactoring to add and remove declarations without
            impacting files that have an `extern`.
    4.  If a library declares an entity as non-`extern` in either the `api` or
        `impl`, it is invalid to declare the same entity as `extern` elsewhere
        within the library.
2.  An entity may only be forward declared once in a given file. A forward
    declaration is only allowed before a definition.
3.  Modifier keywords will be handled on a case-by-case basis for merging
    declarations.

## Details

### `extern` keyword

An `extern` modifier keyword is added. It modifies a forward declaration to
indicate that the entity is not defined by the declaring library. A library
using `extern` to declare an entity cannot define that entity. Only the library
which defines an entity can omit `extern`, and omitting `extern` means it must
provide a definition.

An `extern` declaration forms an entity which has no definition or storage. For
example:

-   `extern fn` forms a function that can be called.
-   `extern class` forms an incomplete type.
-   `extern interface` forms an undefined interface.
-   `extern var` or `extern let` will bind the name without allocating storage.
    Initializers are disallowed.

The `extern` keyword is invalid on declarations that have _only_ a declaration
syntax and lack storage, such as `alias` or `namespace`. It is only valid on
namespace-scoped names; it is invalid on type-scoped names
(`class Foo { extern fn Member(); }`).

In declaration modifiers, `extern` comes immediately after access control
keywords, and before other keywords. For example,
`private extern <other modifiers> class B;`. At present, when `extern` is on a
declaration, only access modifiers are valid (see
[Modifier keywords](#modifier-keywords)).

### When declarations are allowed

When considering whether a declaration is allowed, we apply the rules:

1. A declaration should always add new information.
    - No declarations after a definition.
2. Only one library can declare an entity without `extern`.
3. Support moving declarations between already-imported `api` files without
   affecting compilation of client libraries.

#### No forward declarations after declarations

In a file, a forward declaration must never follow a forward declaration or
definition for the same entity.

For example:

```carbon
class A { ... }
// Invalid: Disallowed after the definition.
class A;

class B;
// Invalid: Disallowed due to repetition.
class B;

class C;
// Valid: Allowed because the definition is added.
class C { ... }
```

#### Files must either use an imported declaration or declare their own

In a file, if a declaration or definition of an entity is imported, the file
must choose between either using that version or declaring its own. It cannot do
both.

For example:

```carbon
package Foo library "a" api;

class C { ... }
```

```carbon
package Foo library "b" api;
import library "a";

extern class C;

// Valid: Uses the incomplete type of the extern declaration.
fn Foo(c: C*);
```

```carbon
package Foo library "c" api;
import library "a";

// Valid: Uses the complete type of the imported definition.
fn Foo(c: C);
```

```carbon
package Foo library "d" api;
import library "a";

fn Foo(c: C);

// Invalid: `Foo` used the imported `C`, so an `extern` declaration of `C` is now
// invalid.
extern class C;
```

#### Libraries cannot both define an entity and declare it `extern`

In a library, if the `impl` defines an entity, the `api` must not use `extern`
when declaring it.

For example:

```carbon
package Wiz library "a" api;

extern class C;
```

```carbon
package Wiz library "a" impl;

// Invalid: The `api` file declared `C` as `extern`.
class C { ... }
```

In a library, the `api` might make an `extern` declaration that the `impl`
imports and uses the definition of. This is consistent because the `impl` file
is not declaring the entity.

#### `impl` files with a forward declaration must contain the definition

In an `impl` file, if the `impl` file forward declares an entity, it must also
provide the definition. In libraries with multiple `impl` files, this means that
using an entity in one `impl` file when it's defined in a different `impl` file
requires a (possibly `private`) forward declaration in the `api` file. An
`extern` declaration cannot be used for this purpose because the library defines
the entity.

This allows Carbon to provide a compile-time diagnostic if an entity declared in
the `impl` is not defined locally. Note that an entity declared in the `api` may
still not get a compile-time diagnostic unless the compiler is told it's seeing
_all_ available `impl` files.

For example:

```carbon
package Bar library "a" api;

class C;

class D { ... }
```

```carbon
package Bar library "a" impl;

// Invalid: Missing a definition in the `impl` file, but if one were added, then
// this would be valid.
class C;

// Invalid: The `api` defines `D`. As a consequence, there is no way to make
// this forward declaration valid.
class D;
```

This doesn't prevent `impl` from providing a forward declaration. It might when
it also provides the definition, which can be useful to unravel dependency
cycles:

```carbon
package Bar library "a" api;

class D;
```

```
package Bar library "a" impl;

class D;

class E {
  fn F[self: Self](d: D*) { ... }
}

class D {
  fn G[self: Self](e: E) { ... }
}
```

Here, the `impl` could not use the imported forward declaration in the `api`
because of the rule
[Files must either use an imported declaration or declare their own](#files-must-either-use-an-imported-declaration-or-declare-their-own).
Without `class D;` present, the definition of `F` would be invalid.

#### Type scopes may contain both a forward declaration and definition

The combination of a forward declaration and a definition is allowed in type
scopes.

For example:

```carbon
class C {
  class D;

  fn F() -> D;

  class D {
    fn G() -> Self { return C.F(); }

    var x: i32;
  }

  fn F() -> D { return {.x = 42}; }
}
```

This is necessary because type bodies are not automatically moved out-of-line,
unlike function bodies.

### Using `extern` declarations in `extern impl`

It is invalid for a non-`extern` `impl` declaration to use an `extern` type in
its type structure.

Consider two libraries, one defining `A` and declaring `B` as `extern`, and the
other defining `B` and declaring `A` as `extern`. Neither should be able to
define an `impl` involving both `A` and `B`, otherwise both could.

### `impl` lookup involving `extern` types

If `impl` lookup involving `extern` types finds a non-`final` parameterized
`impl`, the result is that the lookup succeeds, but none of the values of the
associated entities of interface are known. This is because there may be another
more specialized `impl` that applies that is not visible (as can also happen
with constrained generics).

For example:

```
library "a" api;
extern class C;
extern class D(T:! type);
extern impl forall [T:! type] D(T) as I where .Result = i32;
```

In the above, `D(C)` impls `I`, but with unknown `.Result`, since it might not
be `i32`.

### Modifier keywords

When considering various modifiers on a forward declaration versus definition:

-   `extern` is only valid on a forward declaration. Rules are detailed above.
-   `extend` in `extend impl` is only on the declaration in the class body
    (whether that is a forward declaration or definition), as described at
    [Forward `impl` declaration](https://github.com/carbon-language/carbon-lang/blob/trunk/docs/design/generics/details.md#forward-impl-declaration).
-   Other class, impl, and interface modifiers (`abstract`, `base`, `final`)
    exist only on the definition, not on the forward declaration.
-   Function modifiers (`impl`, `virtual`, `default`, `abstract`, `final`) must
    match between forward declaration and definition.
    -   This only affects type-scoped names because they are invalid on
        namespace names.
    -   `abstract` won't have a definition so is moot here.
-   Access modifiers (`private` and `protected`) must match.
    -   As an exception, an `extern` name may be `private` when the actual name
        is public.
        -   This allows a library to forward declare another library's type
            without allowing clients to depend on its forward declaration.
        -   On merging, the more public declaration will take precedence, hiding
            `private extern` declarations.
    -   This affects both type-scoped names and namespace names.

## Rationale

-   [Software and language evolution](/docs/project/goals.md#software-and-language-evolution)
    -   This proposal supports moving classes between libraries without
        affecting the compilation of clients.
        -   Allowing a redundant `extern` declaration when a non-`extern`
            declaration is imported allows _adding_ the class to an
            already-imported library where `extern` declarations were previously
            present.
        -   Requiring the use of a local `extern` declaration prevents
            accidental uses that might hinder _removing_ the class from an
            imported library.
    -   Requiring keywords on `extern` declarations only when they affect
        calling conventions means that, in most cases, keywords can be added and
        removed from declarations in the defining library without breaking
        `extern` declarations in other libraries.
-   [Code that is easy to read, understand, and write](/docs/project/goals.md#code-that-is-easy-to-read-understand-and-write)
    -   Explicitly flagging `extern` will assist readers in understanding when a
        library is working with a type in order to avoid dependency loops, with
        minimal impact on writing code.
    -   Preventing redundant forward declaration removes a potential avenue for
        confusion by making the meaning of entities clearer.
    -   Requiring keywords be repeated for type-scoped functions is intended to
        improve readability.
-   [Practical safety and testing mechanisms](/docs/project/goals.md#practical-safety-and-testing-mechanisms)
    -   Requiring `extern` declarations be clearly marked should improve our
        ability to diagnose ODR violations. This will help developers by
        improving detection of a subtle correctness issue.
-   [Fast and scalable development](/docs/project/goals.md#fast-and-scalable-development)
    -   `extern` declarations are considered essential to supporting separate
        compilation of libraries, which in turn supports scaling compilation.
-   [Interoperability with and migration from existing C++ code](/docs/project/goals.md#interoperability-with-and-migration-from-existing-c-code)
    -   `extern` is chosen for consistency with C++, and carries a similar --
        albeit slightly different -- meaning.
-   [Principle: Prefer providing only one way to do a given thing](https://github.com/carbon-language/carbon-lang/blob/trunk/docs/project/principles/one_way.md)
    -   Setting _requirements_ for whether a keyword belongs on a forward
        declaration or with the definition, instead of making places _optional_,
        supports developers making conclusions based on which keywords they see
        -- either by presence or absence.

## Alternatives considered

### Other modifier keyword merging approaches

There has been intermittent discussion about which modifiers to allow or require
on forward declarations versus definitions. There are advantages and
disadvantages about redundancy and being able to copy-paste declarations. There
might be strict requirements for some modifiers to be present in order to
correctly use a forward declaration.

This proposal suggests a partial decision here, at least for a reasonable
starting point that we can implement towards. This will likely evolve in future
proposals, particularly as more keywords are added. However, this still offers a
baseline.

The trade-offs we consider are:

-   Consistency in when a modifier keyword is expected (if applicable) is
    valuable.
    -   Adding a keyword to an entity with a separate definition may require
        adding the keyword to the forward declaration, the separate definition,
        or both. It is disallowed where not required to be added.
        -   For example, `base` is added to `class C { ... }`, and disallowed on
            `class C;`.
    -   Although we could make keywords optional where it would not affect
        semantics, we prefer for the presence or absence of a keyword to carry a
        clear meaning.
        -   For example, if `base` were optional to allow `base class C;`, then
            an adjacent `class D;` lends itself to being incorrectly interpreted
            as meaning "`D` is not a base class" when it actually means "`D` may
            or may not be a base class".
-   Access control has a certain necessity for consistency, so that a consumer
    of a forward declaration would still be allowed to use the definition if
    refactored.
    -   We could require similar consistency on `extern`, and should have more
        nuanced rules if the access control rules go beyond public and
        `private`. For example, a package-private type shouldn't be allowed to
        be made public through an `extern` declaration; but package-private
        isn't actually part of the language right now.
-   For a type, we are choosing to have minimal modifiers on the declaration.
    -   The modifiers we disallow (`abstract`, `base`, `final`, and `default`)
        have no effect on uses because the forward declared type is incomplete.
        -   Requiring them would end up leaking an implementation detail and
            create toil.
        -   Requiring them would be somewhat inconsistent with C++.
    -   `extern` is a special-case where its presence is intrinsic to the
        keyword's semantics.
-   For type-scoped members, we are choosing to duplicate modifiers between the
    declaration and definition.

    -   For example:

        ```
        class A {
          private class B;
        }
        // `private` is required here.
        private class A.B { ... }
        ```

    -   There is more emphasis on being able to copy-paste a function
        declaration. This in particular may help developers more than classes
        because all the parameters must also be copied. Things such as `static`
        in C++ being declaration-only is also something we view as a source of
        friction rather than a benefit.
    -   Modifiers such as `virtual` affect the calling convention of functions,
        and as a consequence _must_ be on the first declaration.
    -   A downside of this approach is that it means the class name is inserted
        in the middle of the out-of-scope definition, rather than near the
        front.

The most likely alternative would be to disallow most modifiers on out-of-line
definitions after a forward declaration, for type member functions in specific.
This would be because, unlike other situations, a member must have a forward
declaration if there is an out-of-line definition. We would probably want to
drop these collectively in order to maximize copy-paste ability (ideally,
everything before `fn` is dropped). However, it would shift understandability of
the definition in a way that may be harmful. For now this proposal suggests
adopting the more verbose approach and seeing how it goes.

Another alternative is that we could allow flexibility to choose which modifier
keywords are provided where. For example, keywords on a function definition must
be some subset of the keywords on the declaration; keywords on a type
declaration must be some subset of the keywords on the definition. This would
allow authors to choose when they expect keywords will be most relevant.
However, it could also serve to detract from readability: two functions in the
same file might be declared similarly, but have different keywords on the
definition, implying a difference in behavior that would not exist. With
consideration for the
[Principle: Prefer providing only one way to do a given thing](https://github.com/carbon-language/carbon-lang/blob/trunk/docs/project/principles/one_way.md),
this proposal takes the more prescriptive approach, rather than offering
flexibility.

Note a common aspect between types and functions in the proposed model is that
modifiers on the definition are typically a superset of modifiers on the
declaration (`extern` as an exception). While looking at the declaration always
gives an incomplete view, looking at the definition can give a complete view.

### No `extern` keyword

If we had no `extern` keyword, then a declaration wouldn't give any hint of
whether a library contains the definition. Given two forward declarations in
different `api` files, either both or neither could have a definition in their
`impl` file. Sometimes this would be detected during linking, particularly if
both are linked together. However, providing an `extern` keyword gives a hint
about the intended behavior, allowing us to evaluate more cases for warnings. It
also gives a hint to the reader, about whether an entity is expected to be
declared later in the library (even if not in the same file).

For example:

```carbon
package Foo library "a" api;

class C;
```

```carbon
package Foo library "a" impl;

class C {}
```

```carbon
package Foo library "b" api;
import library "a";

class C {};
```

Without the `extern` keyword, this code should be expected to compile. Ideally
it would be caught during linking that there are two definitions of `class C`,
but that relies on some tricks to catch issues. When `extern` is added, then the
processing of library "b" results in a conflict between the `class C;` forward
declared in the api of library "a".

Note this probably does not fundamentally alter the amount that can be
diagnosed, but will mainly allow some diagnostics to occur during compilation
that otherwise would either be linker diagnostics or missed.

The `extern` keyword is being added mainly for diagnostics and readability.

### Looser restrictions on declarations

We are being restrictive with declarations and when they're allowed. Primarily,
we want to avoid confusion with code such as:

```carbon
class C { ... }

// This declaration has no effect.
class C;
```

But, we could also allow code such as:

```carbon
package Foo library "a" api;

class C { ... }
```

```carbon
package Foo library "b" api;
import library "a";

fn F(c: C) { ... }

extern class C;

fn G(c: C*) { ... }
```

Here, `F` requires the imported definition of `C`. But, is `G` seeing an
incomplete type from the `extern`, or is the `extern` redundant and `G` sees the
imported definition of `C`? Would a library importing library "b" see `C` as an
incomplete type, or a complete type?

In order to eliminate potential understandability issues with the choices we may
make, we are choosing the more restrictive approaches which disallow both of
these. In the first case, the redeclaration after a definition in the same file
is simply disallowed. In the second case, library "b" cannot declare `C` as
`extern` after using the imported definition. Restrictions such as these should
make code clearer by helping developers catch redundant, and possibly incorrect,
code.

### `extern` naming

Beyond `extern`, we also considered `external`. `extern` implies external
linkage in C++. We're choosing `extern` mainly for the small consistency with
C++.

### Default `extern` to private

The `extern` keyword could have an access control implication equivalent to
`private`. Then `extern` would need explicit work to export the symbol. The
`export` keyword was proposed for this purpose, with the idea that
`export import` syntax might also be provided to re-export all symbols of an
imported library.

1.  `extern` and `export` semantic consistency

    This would mean that `extern` declarations have different access control
    than other declarations, which is a different visibility model to
    understand, and may also not be intuitive from the "external" meaning. The
    use of `export` matches C++'s `export` keyword, which may also imply to
    developers that other semantics match C++, such as `export` being necessary
    for all declared names.

2.  Interaction with additional access control features

    We'll probably also want more granular access control than just public and
    private for API names. Adding package-private access modifier seems useful
    (for example, Java provides this as `package`): in C++, this is sometimes
    achieved through an "internal" or "details" namespace. If Carbon only
    supports library-private symbols, that still addresses some of these
    use-cases, but will sometimes require private implementation details to
    exist in a single `api` file in order to get language-supported visibility
    restrictions. In some cases this will result in an unwieldy amount of code
    for a single file.

    For example of how package-private visibility might be used, gtest has
    [an internal header directory](https://github.com/google/googletest/blob/main/googletest/include/gtest/internal)
    that contains thousands of lines of code. If this needed to migrate to `api`
    files in Carbon, it would be ideal if users could not access the names by
    importing an internal library.

    For example of how this would interact:

    | Default visibility             | Public (proposed)         | Private (alternative)            |
    | ------------------------------ | ------------------------- | -------------------------------- |
    | Public                         | `extern class C;`         | `export extern class C;`         |
    | Library-private                | `private extern class C;` | `extern class C;`                |
    | Package-private (hypothetical) | `package extern class C;` | `export package extern class C;` |

3.  Risks of a public extern

    When making an `extern fn`, it is callable. This creates a risk of a
    function being declared as `extern` for internal use, but accidentally
    allowing dependencies on the function. This risk could be mitigated by
    requiring access control of an `extern` to be either equal to or more
    restrictive than the original symbol (which might be hard to validate, but
    could be validated when both symbols are seen together).

    When making an `extern class`, it's an incomplete type. It cannot be
    instantiated, but pointers and references may be declared. This is only
    really useful if there are functions which take a pointer as a parameter,
    but an instance of the pointer could only be created by either unsafe casts
    or if there's a function that returns the pointer type.

    Unsafe casts carry an inherent risk. In the case of a returned pointer, that
    type could be captured by `auto`. Having `extern class` default to private
    does not prevent the type's use.

Considering the trade-offs involved combination of these three points, this
proposal suggests using the regular access control semantic. That choice may be
reconsidered later, based on how the semantics work out and any issues that
arise, particularly around access control.

### Opaque types

Omitting the `extern` modifier means a definition is required in a library.
There may be a use-case for opaque types which are "owned" by a library and have
no definition. If so, there are possible solutions such as a modifier keyword to
indicate an opaque type. For now, an empty definition in the `impl` file of a
library should have a similar effect: `api` users would not be able to provide a
definition.

For example:

```carbon
package Foo library "a" api;

// An opaque type which can be imported by other libraries.
class C;
```

```carbon
package Foo library "a" impl;

// An empty definition. This could be in its own file, or at the end after logic,
// to prevent misuse.
class C {}
```

This proposal requires a definition to exist. That choice may be reconsidered
later, based on use-cases.

### Require a library provide its own `extern` declarations

As proposed, any library can provide `extern` declarations for other libraries
in the same package. It was proposed that this should be restricted so that a
library would need to make `extern` declarations available, either through a
separate `extern` file (similar to `api` and `impl`) or through additional
markup in the `api` file which could be used to automatically generate an
`extern`-only subset of the `api` (still requiring entities which _should_ be
`extern` to be explicitly marked).

Advantages:

-   Centralizes ownership of `extern` declarations.
    -   We are already planning to require a package to provide `extern`
        declarations. This goes a step further, requiring the `extern`
        declaration be provided by the same library that's defining the entity,
        providing a clear, central ownership.
-   Simplifies refactoring.
    -   The current plan of record is to require that `extern` declarations
        agree with the library's declaration of the definition. This includes
        small details such as parameter names. A consequence of this is that
        changing these details in one declaration requires changing them in
        _all_ declarations, atomically. There's a desire to limit the scope of
        atomic refactorings; for example, similar package-scope atomic
        refactoring requirements lead us to _allow_ certain redundant forward
        declarations elsewhere in this proposal.
-   Reduces complexity for migrating C++ forward declarations.
    -   As described in
        [Migration of C++ forward declarations](#migration-of-c-forward-declarations),
        we expect migrating forward declarations to be difficult. Several of the
        steps needed already can produce results similar to this alternative.
        Under this alternative, we would make the handling of an entity defined
        in a different library identical to as if it were in a different
        package: a reduction of one case.

Disadvantages:

-   Makes it difficult for libraries which want to use `extern` declarations to
    use minimal imports for a declaration.
    -   If a library provides multiple `extern` declarations, the `extern`
        file's imports would be a superset of the imports for those
        declarations. If a client library only wants one of those `extern`
        declarations, it would still get the full set of dependencies; under the
        proposed syntax, only the single declaration's dependencies are
        required. This allows for fewer dependencies.
-   Does not support use-cases that may occur in C++.
    -   It's possible that an extern declaration may depend on a defined entity,
        where that entity is being defined in the same file. For example,
        `class C { ... }; extern fn F(c: C);`. Under the proposed syntax, this
        is supported; under the alternative syntax, another solution would need
        to be found. There are potential ways to fix this through refactoring,
        including moving one entity to a different library, changing the
        `extern` declaration to use only `extern` declarations, or using generic
        programming to create an indirection. However, each of these is a
        refactoring that may hinder adoption.

At present, the disadvantages are considered to outweigh the advantages. It's
possible that this may be revisited later if we get more code and libraries
using these tools and recognize some patterns that we can better or more
directly support. However, we should be hesitant to provide a second syntax for
`extern` if it's not adding substantial value, under
[Principle: Prefer providing only one way to do a given thing](https://github.com/carbon-language/carbon-lang/blob/trunk/docs/project/principles/one_way.md).

### Allow cross-package `extern` declarations

We could choose a syntax for `extern` declarations that allows cross-package
`extern` declarations. These are effectively supported in C++, where there are
no package boundaries. Dropping support will create a migration barrier.

However, there is a strong desire to restrict the use of cross-package
declarations in order to reduce the difficult and complex refactoring costs that
result from cross-package declarations: requiring that a particular name not
change its declaration category (a `class` must remain a `class`, preventing
`alias`) and that parameters (either function or generic) must remain the same,
not even allowing implicit conversions.

The package boundaries serve an important purpose in balancing costs for
refactoring.

## Appendix

### Migration of C++ forward declarations

This is not proposing a particular approach to migration. However, for
consideration of the proposal, it can be helpful to consider how migration will
work.

It's expected under this approach that migration of a forward declaration will
require identifying the Carbon library that defines the entity. Then:

-   The forward declaration will need to be adjusted based on library and
    package boundaries:
    -   If the forward declaration is disallowed in Carbon, it may need to be
        removed.
    -   If the forward declaration is in the same library as the defining
        library, then no `extern` is required.
    -   If the forward declaration is in a different library but the same
        package, then `extern` is added.
    -   If the forward declaration is in a different package, the forward
        declaration must be removed. To replace it, there are a couple options
        which would need to be chosen by heuristic:
        1.  Add a dependency on the actual definition. This might be infeasible
            when the defining library has many complex dependencies.
        2.  Add a library to the other package that provides the necessary
            `extern` declaration. This might be infeasible when the package is
            not owned by the package being migrated.
    -   If the forward declared code is in C++, we need to retain a forward
        declaration in C++. A couple examples of how we might achieve that are:
        -   Provide Carbon syntax for an in-file forward declaration of C++ code
            (for example, `extern cpp <declaration>`).
        -   Create a small C++ header providing the forward declaration, and
            depend on it.
-   Fix meaningful differences in parameter names, for example by updating the
    forward declaration's parameter names to match the definition.
-   Fix meaningful differences in modifier keywords, for example by adding a
    function forward declaration's modifiers to the definition.
