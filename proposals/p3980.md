# Singular `extern` declarations

<!--
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

[Pull request](https://github.com/carbon-language/carbon-lang/pull/3980)

<!-- toc -->

## Table of contents

-   [Abstract](#abstract)
-   [Problem](#problem)
-   [Background](#background)
-   [Proposal](#proposal)
    -   [Declarations](#declarations)
    -   [Owning `extern` declarations](#owning-extern-declarations)
-   [Details](#details)
    -   [Type coherency](#type-coherency)
        -   [Impact on indirect imports](#impact-on-indirect-imports)
        -   [Indirect imports of non-`extern` types](#indirect-imports-of-non-extern-types)
    -   [Using imported declarations](#using-imported-declarations)
    -   [`private extern`](#private-extern)
    -   [Validation for non-owning `extern library` declarations](#validation-for-non-owning-extern-library-declarations)
    -   [No syntactic matching for `extern library` declarations](#no-syntactic-matching-for-extern-library-declarations)
    -   [Versus proposal #3762](#versus-proposal-3762)
-   [Rationale](#rationale)
-   [Future work](#future-work)
    -   [`extern` and template interactions](#extern-and-template-interactions)
-   [Alternatives considered](#alternatives-considered)
    -   [Allow multiple non-owning declarations, remove the import requirement, or both](#allow-multiple-non-owning-declarations-remove-the-import-requirement-or-both)
    -   [Total number of allowed declarations (owning and non-owning)](#total-number-of-allowed-declarations-owning-and-non-owning)
        -   [Do not restrict the number of forward declarations](#do-not-restrict-the-number-of-forward-declarations)
        -   [Allow up to two declarations total](#allow-up-to-two-declarations-total)
        -   [Allow up to four declarations total](#allow-up-to-four-declarations-total)
    -   [Don't require a modifier on the owning declarations](#dont-require-a-modifier-on-the-owning-declarations)
    -   [Only require `extern` on the first owning declaration](#only-require-extern-on-the-first-owning-declaration)
    -   [Separate require-direct-import from non-owning declarations](#separate-require-direct-import-from-non-owning-declarations)
    -   [Other `extern` syntaxes](#other-extern-syntaxes)
    -   [Have types with `extern` members re-export them](#have-types-with-extern-members-re-export-them)
    -   [Require syntactic matching for `extern library` declarations](#require-syntactic-matching-for-extern-library-declarations)

<!-- tocstop -->

## Abstract

An entity may be declared `extern` (such as `extern class Foo;`); this means
that its type is only complete if the definition is directly imported. It also
allows for a single declaration in a different library, which must be marked as
`extern library "<owning_library>"` (such as `extern library "Bar" class Foo;`).

Also, establish a different rule of thumb for when modifier keywords are
required: modifier keywords are required when, if prior optional declarations
were removed, the lack of the modifier keyword would change behavior.

## Problem

In the `extern` model from
[#3762: Merging forward declarations](https://github.com/carbon-language/carbon-lang/pull/3762),
multiple `extern` declarations are allowed.
[#3763: Matching redeclarations](https://github.com/carbon-language/carbon-lang/pull/3763)
further evolved the `extern` keyword.

The prior `extern` model assumed that the `extern` and non-`extern` declarations
of a class formed two different types, which could be merged.
[As discussed on #packages-and-libraries](https://discord.com/channels/655572317891461132/1217182321933815820/1230990636073881693),
this runs into an issue with code such as:

```
library "a";
class C {}
```

```
library "b";
extern class C;
extern fn F() -> C*;
```

```
library "c";
import library "a";
extern fn F() -> C*;
```

Here, the return types of `F` differ.

This proposal aims to address the differing return types by unifying the type of
`C` regardless of whether it's `extern`. This could be done under multiple
different approaches, and this proposal aims for one which enables efficient
implementation strategies.

## Background

Proposals:

-   [#3762: Merging forward declarations](https://github.com/carbon-language/carbon-lang/pull/3762)
-   [#3763: Matching redeclarations](https://github.com/carbon-language/carbon-lang/pull/3763)

Discussions:

-   [#packages-and-libraries: `extern` type coherency](https://discord.com/channels/655572317891461132/1217182321933815820/1230990636073881693)
-   [#packages-and-libraries: When to allow/disallow redeclarations](https://discord.com/channels/655572317891461132/1217182321933815820/1236016051632865421)
-   [Open discussion 2024-05-09: Number of allowed redeclarations](https://docs.google.com/document/d/1s3mMCupmuSpWOFJGnvjoElcBIe2aoaysTIdyczvKX84/edit?resourcekey=0-G095Wc3sR6pW1hLJbGgE0g&tab=t.0#heading=h.bu7djkos4xo)
-   [Issue #3986: Alternative naming for `has_extern` keyword](https://github.com/carbon-language/carbon-lang/issues/3986)
-   [Issue #4025: Handling of indirect access of `extern` types](https://github.com/carbon-language/carbon-lang/issues/4025)
-   [#typesystem: Will `&` have an extension point?](https://discord.com/channels/655572317891461132/708431657849585705/1258150877714452581)

## Proposal

### Declarations

A given entity may have up to three declarations:

-   An optional, non-owning `extern library "<owning_library>"` declaration
    -   It must be in a separate library from the definition.
    -   The owning library's API file must import the `extern` declaration, and
        must also contain a declaration.
-   An optional, owning forward declaration
    -   This must come before the definition. The API file is considered to be
        before the implementation file.
-   A required, owning definition

The consequential changes to the [problem example](#problem) are then:

```
library "a";

// This proposal makes the import required.
import library "b";

// This proposal makes `extern` required here.
extern class C {}
```

```
library "b";
// This proposal makes `library "a"` required here.
extern library "a" class C;
extern fn F() -> C*;
```

```
library "c";
import library "a";
extern fn F() -> C*;
```

### Owning `extern` declarations

On an owning `extern` declaration, such as `extern class C {}`, there are two
key effects:

1.  The declaration must be explicitly imported in order to be complete.
    -   An "explicit import" means some import path exists where the name is
        available to name lookup, including `export import` and `export <name>`.
2.  A non-owning `extern library "<owning_library">` declaration is allowed, but
    not required.

If _either_ owning declaration has the `extern` modifier, _both_ must have it.

## Details

### Type coherency

In the context of the example that is the [problem](#problem), `C` will produce
the same type regardless of whether `C` is the owning or non-owning declaration.
This means that both function signatures have identical types.

We do this by only producing a complete type if the owning definition of `C` is
imported by name: either directly through `import library "a"`, or indirectly
through a chain of `export import library "a"` and `export C;`. Otherwise, an
incomplete type is used.

This does mean that adding `extern` to an owning declaration changes the import
semantic. As a consequence, it is a potentially breaking change for API
consumers that didn't explicitly import the time.

In the presence of `extern library "a" class C;`, the required
`import library "b"` means that all owning `extern class C` declarations are
able to see the `extern library "a" class C` declaration as a name collision,
which is merged. This allows the compiler to easily apply the same type to all
declarations. That in turn will be used to ensure libraries which import both
understand the type equality.

#### Impact on indirect imports

An entity marked as `extern` is only complete when the definition is explicitly
imported. In the following, examples of indirect, non-explicit uses are given
inside `library "o"`.

```
library "m";

extern class C { fn Member(); }
```

```
library "n";
import library "m";

fn F() -> C;
var c: C = {};
var pc: C* = &c;
```

```
library "o";
import library "n";

// Invalid: The return type of `C` is incomplete, making the function signature
// invalid.
fn G() { F(); }

// Invalid: Accessing members requires `C` to be complete.
fn UseC() { c.Member(); }

// Valid: Taking the address of `C` doesn't require it to be complete. This is
// possible because `&` doesn't have an extension point.
var indirect_pc: auto = &c;

// Invalid: Copying `C` requires the complete type.
var copy_c: auto = c;

// Valid: Pointer-to-pointer copies are okay.
var copy_pc: auto = pc;
```

#### Indirect imports of non-`extern` types

The above rules explicitly do not apply for non-`extern` types, as decided in
[Issue #4025](https://github.com/carbon-language/carbon-lang/issues/4025). In
other words:

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

Since `extern library "a" class C;` must be imported by the owning library, we
now allow uses of the imported name prior to its declaration within the same
file. This is a divergence from
[#3762](https://github.com/carbon-language/carbon-lang/pull/3762). It means the
following now works:

```
library "extern";

extern library "use_extern" class MyType;
```

```
library "use_extern";
import library "extern"

// Uses the `extern library` declaration.
fn Foo(val: MyType*);

extern class MyType {
  fn Bar[addr self: Self*]() { Foo(self); }
}
```

### `private extern`

Previously, in
[#3762](https://github.com/carbon-language/carbon-lang/pull/3762), a non-owning
`private extern` was valid to declare something as extern without exposing the
name. In this proposal, that would be a non-owning
`private extern library "<owning_library>"` for an owning public `extern`
declaration. However, rather than supporting this version of the syntax, it will
instead be invalid because the name would never be visible to the owning
library. Instead, visibility must match between an
`extern library "<owning_library>"` declaration and the owning `extern`
declaration.

Note, because an owning `extern` declaration can be used independently of
`extern library "<owning_library>"`, an owning `private extern` declaration is
valid in an API file. It has no special behaviors about it, and is merged as
normal.

### Validation for non-owning `extern library` declarations

We should offer some validation that the library in `extern library` is correct.
When the owning library is incorrect, it's very likely to be detected in two
cases:

-   A compile-time error when the owning library imports the non-owning library,
    when the owning declaration is evaluated.
-   A link-time error as a fallback.

Other cases, such as when both libraries are independently imported, may or may
not be caught, dependent upon the cost of validation.

### No syntactic matching for `extern library` declarations

The non-owned `extern library` declarations will only use semantic matching for
redeclarations, not syntactic matching. Details of syntactic matching laid out
in [#3763](https://github.com/carbon-language/carbon-lang/pull/3763) will only
apply to owned declarations in the same library, which may include owned
`extern` declarations.

### Versus proposal #3762

Versus proposal
[#3762](https://github.com/carbon-language/carbon-lang/pull/3762), the `extern`
feature is essentially rewritten. No part of `extern` should be assumed to still
apply.

## Rationale

-   [Software and language evolution](/docs/project/goals.md#software-and-language-evolution)
    -   Unifying the type of `extern` entities addresses a type coherency issue.
    -   The `extern` behavior of requiring an explicit import is intended to
        assist library authors in carefully managing the dependencies on their
        API.
-   [Fast and scalable development](/docs/project/goals.md#fast-and-scalable-development)
    -   Requiring the non-owning `extern library` declaration be imported by the
        owning library should improve compiler performance.

This proposal makes a trade-off with
[Interoperability with and migration from existing C++ code](/docs/project/goals.md#interoperability-with-and-migration-from-existing-c-code).
The restriction of a unique `extern` declaration is expected to require
additional work in migration, because C++ `extern` declarations will need to be
consolidated. This is currently counter-balanced by the trade-offs involved,
although it may result in a reevaluation of that aspect of this proposal.

## Future work

### `extern` and template interactions

We've only loosely discussed template interactions with `extern`. Right now,
what we expect is that when a template declaration uses an `extern` type, the
_instantiation_ still occurs in the calling file. Thus, the `extern` type's name
would need to be imported in both the file declaring the template, and the file
calling the template.

When the template is in the same package as the `extern` type, it could
re-export it. However, we don't support re-exporting names cross-package, and
something like `let template ExternType:! auto = OwningPackage.ExternType;`
would not actually forward the _completeness_ of `ExternType`.

This is expected to be inconvenient, but it may be okay if `extern` sees limited
use. It may also be that the template model ends up different from expected.

## Alternatives considered

### Allow multiple non-owning declarations, remove the import requirement, or both

We limit to one non-owning `extern library` declaration. Continuing to allow
multiple `extern library` declarations (the previous state) is feasible.
Similarly, we could not require the owning `extern` declaration to import the
non-owning `extern library` declaration; this could be done with or without
multiple non-owning `extern library` declarations. For this set of alternatives,
the issues which would arise are similar.

In the compiler, we want to be able to determine that two types are equal
through a unique identifier, such as a 32-bit integer. When one declaration sees
another directly, as through an import, we identify the redeclaration by name,
and reuse the unique identifier. This deduplication can occur once per
declaration. Indirect imports can continue to use the unique identifier.

We could instead support unifying declarations that did not see each other.
However, this would require canonicalizing all types by name instead of by
unique identifier. For example, consider:

```
package Other library "type";
extern class MyType {
  fn Print();
};
```

```
package Other library "use_type";
import library "type";
fn Make() -> MyType*;
```

```
package Other library "extern";
extern library "type" class MyType;
```

```
package Other library "use_extern";
import library "extern";
fn Print(val: MyType*);
```

```
library "merge";
import Other library "use_type";
import Other library "use_extern";
Other.Print(Other.Make());
```

Here, the "merge" library doesn't see either declaration of `MyType` directly.
However, `Print(Make())` requires that both declarations of `MyType` be
determined as equivalent. This particular indirect use also means that the names
will not have been added to name lookup, so there is no reason for the two
declarations to be associated by name.

In order to do merge these declarations, we would need to identify that fully
qualified names and other structural details are equivalent when the type is
used (including non-explicit uses, such as interface lookup). We could achieve
this, for example, by having a name lookup table for in-use types, managed per
library. Each library would also need to validate that declarations were
semantically equivalent, versus the current approach validating as part of the
redeclaration. The cost of a per-library approach is expected to have a
significant impact on the amount of work done as part of semantic analysis.

We may end up wanting to do similar work in order to improve diagnostics for
invalid cases where the non-owning `extern library` is not correctly declared
and imported. However, additional work building good diagnostics for
already-identified invalid code is less of a concern than additional work on
fully valid code.

In order to maintain a high-performance compiler, we are taking a restrictive
approach that makes it simpler to associate type information.

### Total number of allowed declarations (owning and non-owning)

A few options were considered regarding the number of allowed declarations.

We limit to two owning declarations: the optional forward declaration, and
required definition. The need to provide interface implementations (for example,
`impl MyType as Add`) is considered to constrain this choice.

In this category, alternatives considered were:

-   Do not restrict the number of declarations
-   Allow up to two declarations total
-   Allow up to four declarations total

Details for why each alternative was declined are below.

#### Do not restrict the number of forward declarations

We could not restrict the number of forward declarations, allowing an arbitrary
amount -- possibly also after the definition. This would be consistent with C++.

One thing to consider here is modifier keyword behavior. If we require modifier
keywords to match across all declarations, that could become a maintenance
burden for developers. If we don't, it makes the meaning of a given forward
declaration more ambiguous.

This option is declined due to the lack of clear benefit.

#### Allow up to two declarations total

Under this option, we would only allow one forward declaration, treating the
non-owning `extern library` declaration as a forward declaration. This would
mean two declarations overall, instead of three.

For this, the main concern was interactions between file placement of the
definition, and file placement of interface implementations. Interface
implementations must generally be in API files in order to be seen by other
libraries.

For example:

```
library "i";
interface I {}
```

```
library "e";
import library "i";
extern library "o" class C;
extern library "o" impl C as I;
```

```
library "o";
import library "e";
extern class C { }
extern impl C as I;
```

```
impl library "o";
extern impl C as I { }
```

If the definition is required to be in the API file in order to allow the
interface implementations in the API file, the API file would need to import
libraries required to construct the definition. That could create issues for
separation of build dependencies, and could also make it more difficult to
unravel some dependency cycles between libraries.

If the definition was allowed to be in the implementation file even when there
were interface implementations in the API file, the ambiguity of seeing a
non-owning `extern library` declaration and being unsure of whether this was the
owning library could have negative consequences for evaluation of interface
constraints.

The purpose of allowing a forward declaration when there is a non-owning
`extern` declaration is to make it clear for interface implementations that they
exist in the owning library, while processing the API file.

#### Allow up to four declarations total

The four declarations would be:

1.  Non-owning `extern library` declaration
2.  Forward declaration in API file
3.  Forward declaration in implementation file
4.  Definition

The number of forward declarations allowed is consistent with the current state
from [#3762](https://github.com/carbon-language/carbon-lang/pull/3762).

This would allow for clarity when defining in the implementation file, to also
be able to put a forward declaration above -- even when the forward declaration
is pulled from the API file.

If we're allowing declarations from another file (including the non-owning
`extern library` declaration) to be used before an entity is declared in the
same file, the motivating factor for allowing a repeat forward declaration in an
implementation file is removed. Previously, that was required for an entity to
be referenced prior to its definition.

In discussion of this option, it was considered unclear why we would allow two
forward declarations, but not allow even more. The more popular choice seemed to
be not restricting, which was also declined.

### Don't require a modifier on the owning declarations

Instead of requiring an `extern` modifier on owning declarations, we could infer
from the presence of a non-owning `extern library` declaration.

We had declined allowing a definition to control whether `extern library` was
allowed in discussion of
[#3762](https://github.com/carbon-language/carbon-lang/pull/3762), although this
is not directly mentioned in the proposal. At the time, it was dropped because
the owning library didn't need to include `extern` declarations, and so having
the definition opt-in to allowing `extern` was viewed as low benefit. However,
now that the owning library must import the `extern` declaration, there is a
tighter association and so we reevaluated.

The `extern` modifier offers a benefit for being able to verify the association
between non-owning and owning declarations, and offers additional parity in
modifiers. It also makes it easy for a tool to know if it's missing a
declaration.

### Only require `extern` on the first owning declaration

At present, we require `extern` on _all_ owning declarations. We could instead
only require `extern` on the first owning declaration and, if there's a separate
forward declaration and definition, infer it for the definition. For example:

```
// `extern` on the forward declaration.
extern class C;
// Infer `extern` for the definition.
class C {}
```

The decision to require `extern` on all owning declarations is based on wanting
the forward declaration to be optional. A rule of thumb was discussed wherein if
a forward declaration could be removed without breaking the definition (as
defined by it being in the same lexical scope), keywords should be duplicated to
the definition. This is not proposed as a rule because it's not clear whether
we'll generally follow it, but it's why this particular choice is taken.

### Separate require-direct-import from non-owning declarations

At present, an `extern` modifier on an owning declaration serves two purposes:

1.  Indicates that a non-owning `extern library` declaration _can_ exist.
2.  Indicates the declaration must be directly imported in order to be complete.

This means that:

-   The presence of `extern` on an owning declaration cannot be used to
    determine whether a non-owning declaration exists.
    -   Because the location of a non-owning declaration isn't explicit in the
        owning code, this may lead to a developer failing to find the non-owning
        declaration and misunderstanding that as the non-existence of a
        non-owning declaration.
-   Libraries which happen to be imported by the owning declaration may freely
    add or remove non-owning `extern library` declarations without modifying the
    owning library.

We could give distinct syntax to the two purposes, so that they could be managed
separately. The preference at present is to use a single syntax for both
purposes, rather than emphasizing control or correspondence.

### Other `extern` syntaxes

[Issue #3986](https://github.com/carbon-language/carbon-lang/issues/3986)
discussed other syntaxes for `extern` + `extern library`. These were mainly
`has_extern`/`is_extern`/`externed` + `extern`.

Breaking down `extern`, there are two features which could have been provided
separately:

1.  Declaring an entity has a forward declaration in a separate library.
    -   Also, declaring that forward declaration in a separate library:
        `extern library "<owning_library>"`.
2.  Declaring an entity must be imported directly.

Although (1) must depend on (2), a different design could provide (2) without
making (1) possible, for example with different keywords to differentiate
between intended usage (`has_extern class C;` meaning (1) and (2), `must_import`
meaning (2) only). However, the `extern` keyword approach means developers have
all or nothing.

Considering that, the trade-offs are viewed as:

-   The primary motivation is to provide feature (1).
-   Leads wanted a syntax on the owning declaration that states something
    positive about the owning declaration itself, rather than expressing that
    other declarations exist, which suggests that the syntax on the owning
    declaration should provide feature (2).
-   Leads consider it valuable, though secondary, to support (2) separate from
    (1), and find it acceptable to make (1) optional to achieve this (in other
    words, making the `extern library "<owning_library>"` declaration optional).
    -   It's okay that that `extern library "<owning_library>"` can be added and
        removed from imported libraries without modifying the owning library.
    -   If a developer considers it important to disambiguate the intended use
        of a declaration `extern class C;` and whether there should be a
        declaration in a separate library, they can add comments.
-   `extern` seemed like an acceptable name for this approach, and alternative
    names seemed significantly less good.
-   Using `extern` for both features still only creates one new keyword, versus
    multi-keyword approaches.
-   Adding the owning library with `extern library "<owning_library>"` will
    hopefully improve diagnostics and human understandability of the code.
    -   It is _very_ verbose, but this verbosity goes on the forward declaration
        in the non-owning library. When it's read, which will hopefully be less
        often than the actual declaration, it will provide the reader directions
        to find the actual declaration.
    -   If in practice we find the verbosity becomes a significant issue, we can
        revisit syntaxes to address that specifically. For example, if we have
        significant repetiton, we might consider a grouping structure such as
        `extern library "..." { <many forward declarations> }`.

### Have types with `extern` members re-export them

We expect there will be types that have `extern` members; these types are only
truly complete if their members are complete.

We discussed having such types automatically re-export the `extern` members,
possibly requiring the types to also be `extern` in order to be allowed to have
`extern` members. For example:

```
library "a"
extern library "b" class A;
```

```
library "b"
import library "a"
extern class A {}
// B re-exports A so that it's complete on use.
class B { var a: A; }
```

```
library "c"
import library "b"
// Importing this function declaration gets B, which again, re-exports A so that
// it's complete on use.
fn F() -> B { ... }
```

```
library "d"
// This import loads the incomplete name for A.
import library "a"
// This import loads F, which loads B, which loads the definition of A.
import library "c"

// Because of the import behaviors, this is valid.
var a: A;
```

We consider this action-at-a-distance. Type coherency means the `A` member of
`B` is the same as the `A` in name lookup; we could make them behave slightly
differently, but then we get into provenance tracking of type information.
Several various forms of this have been discussed as part of the `extern`
design, and it's something we've decided to avoid.

Although it's more inconvenient, we will require `A` to be deliberately imported
in order for `B` to be complete.

### Require syntactic matching for `extern library` declarations

We will not require syntactic matching for `extern library` declarations, but we
could.

When a redeclaration is in the same library, we've designed name lookup in a way
such that syntactic matching is effectively a superset of semantic matching.
However, that relies on poisoning entries in name lookup, with later
redeclarations seeing identical name lookup data. Because different libraries
have different name lookup data, syntactic matching _not_ a superset of semantic
matching cross-library. We address this schism by only requiring semantic
matching.

Semantic matching will include parameter names. The difference is primarily in
whether different ways of producing the same type information are considered
invalid or not.

For example:

```
library "a";

class A {}
namespace NS;
extern library "c" fn NS.F() -> A;
```

```
library "b";

namespace NS;
class A {}
```

```
library "c"; import library "a" import library "b"

extern fn NS.F() -> NS.A {}
```

Semantically, `NS.F` in libraries "a" and "c" are identical. Syntactically, they
differ because of `NS.A` in "c". Writing `A` in "c" is invalid because it would
use `NS.A` from "b". But in "a", there is nothing to make the declaration
invalid: it would only be invalid after completing cross-library compilation.

However, we could also have code such as:

```
library "d";

class D {}
namespace NS;
extern library "e" fn NS.G() -> D;
```

```
library "e";

namespace NS;
alias NS.D = D;
extern fn NS.G() -> D {}
```

Here, the semantics and syntax match, but this would be invalid in a normal
redeclaration due to the different name lookup result for `D`.

This additionally gets into a different statement made in
[#3763](https://github.com/carbon-language/carbon-lang/pull/3763) to justify
synactic matching: "The intention is that whenever the syntax matches, the
semantics must also match." Due to the differences in name lookup, syntax
matching does not mean semantics must match; instead of `alias NS.D = D;`, that
could have been `alias NS.D = i32;` and the syntax would have still matched.
This only works in a library because "...we persist syntactic information from
the API file to implementation files." We cannot persist syntactic information
cross-library, across imports.

Due to the differences in the guarantees that syntactic matching provides for
owned declarations versus non-owned declarations, we will not enforced syntactic
matching on the non-owned `extern library` declarations.
