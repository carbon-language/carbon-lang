# Forward `impl` declaration of an incomplete interface

<!--
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

[Pull request](https://github.com/carbon-language/carbon-lang/pull/5168)

<!-- toc -->

## Table of contents

-   [Abstract](#abstract)
-   [Problem](#problem)
-   [Background](#background)
    -   [Related work and past discussion](#related-work-and-past-discussion)
    -   [Terminology](#terminology)
    -   [Other considerations](#other-considerations)
-   [Proposal](#proposal)
    -   [Prior to this proposal](#prior-to-this-proposal)
    -   [Proposed rules](#proposed-rules)
-   [Details](#details)
    -   [An `impl` implements a single interface](#an-impl-implements-a-single-interface)
    -   [Associated constants may be assigned in the body of an `impl` definition](#associated-constants-may-be-assigned-in-the-body-of-an-impl-definition)
    -   [An `impl` may be forward declared without the interface being complete](#an-impl-may-be-forward-declared-without-the-interface-being-complete)
    -   [Interface requirements](#interface-requirements)
    -   [Examples](#examples)
-   [Future work](#future-work)
    -   [Addressing the subsumption use case previously addressed by `extend` in interfaces](#addressing-the-subsumption-use-case-previously-addressed-by-extend-in-interfaces)
    -   [Opt-in to using the interface's default definition of an associated function](#opt-in-to-using-the-interfaces-default-definition-of-an-associated-function)
-   [Rationale](#rationale)
-   [Alternatives considered](#alternatives-considered)
    -   [Allow implementing multiple interfaces with a single `impl` declaration](#allow-implementing-multiple-interfaces-with-a-single-impl-declaration)
    -   [An `impl` of an interface also implements the interfaces it extends](#an-impl-of-an-interface-also-implements-the-interfaces-it-extends)
    -   [Use the contents of definitions if available](#use-the-contents-of-definitions-if-available)
    -   [Delayed checking of incomplete types](#delayed-checking-of-incomplete-types)
    -   [Allow `impl` declarations with rewrites of defined but not complete interfaces](#allow-impl-declarations-with-rewrites-of-defined-but-not-complete-interfaces)
    -   [Can omit function declarations from `impl` body](#can-omit-function-declarations-from-impl-body)
    -   [Different introducer for assigning associated constants in an `impl` definition](#different-introducer-for-assigning-associated-constants-in-an-impl-definition)

<!-- tocstop -->

## Abstract

Revise rules for what is required and provided by declarations and definitions
of interfaces and impls. In particular:

-   allow `impl` declarations of incomplete interfaces, and
-   shift from a "use the information from the type definition if it happens to
    be complete" model to a "only use the information from the definition in
    contexts where it is required to be defined or complete" model.

Resolves questions-for-leads issues
[#4566](https://github.com/carbon-language/carbon-lang/issues/4566),
[#4672](https://github.com/carbon-language/carbon-lang/issues/4672),
[#4579](https://github.com/carbon-language/carbon-lang/issues/4579).

## Problem

There are several kinds of entities in Carbon, and most of them support separate
forward declaration from definition. You might declare an entity before defining
it for a few reasons:

-   Non-generic runtime functions may be called using only a declaration in an
    api file, as long as they are defined in the corresponding impl file. This
    hides implementation details, and allows build work to be parallelized by
    way of separate compilation.
-   By
    [the information accumulation principle](https://docs.carbon-lang.dev/docs/project/principles/information_accumulation.html),
    we don't want code to rely on information written later in the file. For
    example, names can't be used before the declaration that introduces that
    name. This requires the developer to order their code in a way that
    satisfies those constraints.

For example, functions and types (classes, interfaces, named constraints, and so
on) can reference the names of any kind of type in their declaration and
definition.

In some cases, two types will reference each other, requiring forward
declarations:

A
[more complex example with interfaces](/docs/design/generics/details.md#example-of-declaring-interfaces-with-cyclic-references)
can be found in the generics design, that requires introducing named constraint
forward declarations to represent constraints that can't be expressed yet.

We don't want to create a situation where Carbon is introducing a lot of
[accidental complexity](https://en.wikipedia.org/wiki/No_Silver_Bullet) into the
programming process by creating a declaration ordering puzzle that is difficult
or impossible to solve. We would like ordering rules that are straightforward to
satisfy, and simple to remember.

There are a few different options for ordering requirements, all of which have
downsides:

-   Light requirements are easy to satisfy, but don't provide as much
    information to use. For example, if you don't require that an interface is
    defined at a point, then you can't use the interface requirements from its
    definition.
-   Strong requirements are difficult to satisfy, and in the worst case can
    create dependency cycles that can't be satisfied at all.
-   We could have light requirements, but then use additional information when
    it is available. This creates complexity. There is complexity in the
    implementation since there are many cases to consider, and if something goes
    wrong it could be for many different possible reasons. It increases the
    danger of introducing coherence concerns, where reordering declarations and
    definitions could change the meaning of the code. This gives some
    (situational) flexibility to the developer to solve ordering problems.

The design prior to this proposal often uses this last option, creating
implementation complexity. We would like to find an alternate approach that
gives similar flexibility without the downsides.

## Background

### Related work and past discussion

-   [Proposal #722](https://github.com/carbon-language/carbon-lang/pull/722):
    Nominal classes and methods
-   [Proposal #818](https://github.com/carbon-language/carbon-lang/pull/818):
    Constraints for generics (generics details 3)
    -   Introduced implied constraints.
-   [PR #1026](https://github.com/carbon-language/carbon-lang/pull/1026):
    Clarify class declaration syntax
-   [Proposal #1084](https://github.com/carbon-language/carbon-lang/pull/1084):
    Generics details 9: forward declarations
    -   See
        [the section in the generic details design document](https://github.com/carbon-language/carbon-lang/blob/trunk/docs/design/generics/details.md#forward-declarations-and-cyclic-references)
-   [Proposal #2760](https://github.com/carbon-language/carbon-lang/pull/2760):
    Consistent class and interface syntax
-   [Proposal #3762](https://github.com/carbon-language/carbon-lang/pull/3762):
    Merging forward declarations
-   [Proposal #3763](https://github.com/carbon-language/carbon-lang/pull/3763):
    Matching redeclarations
    -   Updated the rules of `impl` declarations and matching
-   [Proposal #3980](https://github.com/carbon-language/carbon-lang/pull/3980):
    Singular extern declarations
-   [PR #4230](https://github.com/carbon-language/carbon-lang/pull/4230): Add
    documentation for entity declaration design work
-   Pending
    [leads question #4566](https://github.com/carbon-language/carbon-lang/issues/4566):
    Implementing multiple interfaces with a single impl definition
    -   The currently proposed resolution includes `impl`ing an interface does
        not `impl` the interfaces it `extend`s, reducing the need to see the
        `interface` definition in `impl` declarations.
-   Pending
    [leads question #4579](https://github.com/carbon-language/carbon-lang/issues/4579):
    When are interface requirements enforced for an `impl`?
    -   Considers what is required and provided by `impl` forward declarations
        and definitions. This proposal offers a resolution to these questions.
-   Pending
    [leads question #4672](https://github.com/carbon-language/carbon-lang/issues/4672):
    Declaration and definition of impls and their associated functions
    -   The currently proposed resolution includes moving assignment of
        associated constants into the `impl` definition in the normal case,
        reducing the need to see the `interface` definition in `impl`
        declarations.
-   [Proposal #5087](https://github.com/carbon-language/carbon-lang/pull/5087):
    Qualified lookup into types being defined
    -   Allows member access inside a type's definition, in addition to
        afterwards. Introduces the _defined_ terminology.

### Terminology

-   A _declaration_ specifies information about an entity. Declarations are
    either _forward declarations_ (which generally end with a semicolon `;`) or
    _definitions_ (which generally end with a body enclosed in curly braces
    `{`...`}`).
-   We say an entity is _declared_ by the first declaration textually in the
    source. If that declaration is a definition, we say the entity is declared
    at the point the body begins (at the open curly `{`).
-   We say an entity is _defined_ once the body of its definition is started (at
    the open curly `{`). We allow name lookup into an entity once it is defined.
    See
    [proposal #5087: Qualified lookup into types being defined](https://github.com/carbon-language/carbon-lang/pull/5087).
-   We say an entity is _complete_ at the end of its definition (at the close
    curly `}`).
-   We say a facet type is _identified_ if all the interfaces it references are
    declared and all of its named constraints are complete. An identified facet
    type is associated with a known set of interfaces.
-   An _implied constraint_ is a condition that must hold if a type is used in a
    given type expression, which we enforce in some way other than checking that
    the condition holds at that point. For example:

    ```
    interface Hash { ... }
    class HashSet(T:! Hash) { ... }

    // `U` must satisfy the constraints on the `T` parameter to `HashSet`.
    // In this case, we enforce it using an implied constraint, so we don't
    // require `U impls Hash` beforehand (say by being declared `U:! Hash`).
    fn Contains[U:! type](needle: U, haystack: HashSet(U)) -> bool;
    // In this case, `Contains` is equivalent to this declaration that
    // doesn't use implied constraints:
    fn Contains[U:! type where .Self impls Hash]
               (needle: U, haystack: HashSet(U)) -> bool;

    // In this example, the implied constraint is on something more
    // complicated than a single symbolic parameter.
    fn F[V:! type, W:! type](x: HashSet(Pair(V, W)));
    // is equivalent to:
    fn F[V:! type, W:! type where Pair(V, .Self) impls Hash]
        (x: HashSet(Pair(V, W)));
    ```

    -   We generally require that there always be a way to explicitly declare
        constraints so implied constraints are not needed.
    -   Implied constraints allow declarations to be more concise, particularly
        we hope that common requirements can be omitted since they are implied.
    -   Implied constraints are also a tool we can choose to use to say that a
        requirement on a parameter can be enforced at a later point. In this
        example,

        ```
        interface B {}
        interface A;
        class C(T:! B);
        fn F(U:! A, x: C(U));

        interface A {
        require Self impls B;
        }
        ```

        the symbolic type parameter `U` has a constraint that it impls `A`, and
        we can add an implied constraint that it also impls `B` because it is
        used as an argument to `C`. This constraint might be redundant with a
        requirement from the definition of `A`, which may or may not have been
        available at the point where `F` is declared. Callers have to provide
        arguments that satisfy the additional requirement if it turns out that
        `A` alone is not sufficient.

### Other considerations

Different kinds of entities have different requirements (ignoring `extern`
declarations):

-   A compile-time function must be defined before a call to it is evaluated.
-   A generic function must be defined before the end of each file in which a
    call to it is type-checked.
-   Non-generic runtime functions may be defined separately. They may be forward
    declared in an api file and defined in a corresponding impl file.
-   Types must be complete before they are used in a function definition. This
    includes the parameter types, the return type, the argument types of
    functions that are called in the body, and the types of local variables.
-   Interfaces and named constraints must be defined in the same file they are
    declared.
-   We want to allow some uses of classes that are declared in an api file and
    defined in a corresponding impl file -- in particular, we treat the "pointer
    to `C`" type `C*` to be complete even if `C` is not.

Our main tool for making ordering easier is forward declarations. Declarations
necessarily come after all the declarations of entities named in its parameter
and return types. So declarations need to be ordered in some
[topological sorted order](https://en.wikipedia.org/wiki/Topological_sorting)
that respects those dependencies. This is unavoidable, though, as long as we are
not allowing forward references, following
[the information accumulation principle](https://docs.carbon-lang.dev/docs/project/principles/information_accumulation.html).

Definitions will have all the dependencies of a declaration, plus perhaps some
more. As a result we will want to put them later in the file. This is possible
if a forward declaration is good enough for other declarations to use that name
in all the ways that they need to.

## Proposal

The goal is to allow forward declarations in more situations, and allow them to
satisfy requirements in more situations. In particular, we make these changes:

-   We accept the proposed resolution of
    [leads question #4566](https://github.com/carbon-language/carbon-lang/issues/4566):
    -   An `impl` of an interface `I` does not `impl` any interface `I` extends.
    -   A single `impl` can be for only a single interface.
-   We accept the proposed resolution of
    [leads question #4672](https://github.com/carbon-language/carbon-lang/issues/4672):
    -   Assignment of associated constants is moved into the `impl` definition
        in the normal case, though may still be present in a `where` clause at
        the end of the declaration.
    -   Associated constants are assigned using `where X = value;` declarations,
        with semantics matching rewrite constraints in `where` clauses.
    -   No part of the `impl` declaration will be excluded from syntactic match.
    -   Every `impl` must be defined in the same file (not just the same
        library) as its declaration.
        -   However, the definitions of its member functions may be separate in
            the impl file, out of line, as provided in
            [proposal #3763](/proposals/p3763.md#out-of-line-definitions-of-associated-functions).
-   An `impl` may be forward declared without the interface being defined.
    -   This is enabled by not needing access to the definition to see other
        interfaces required or extended or to see the associated constants that
        need to be given values.
    -   Instead we require the interface to be _identified_. This means that if
        the facet type to the right of the `as` is given by a named constraint,
        that named constraint needs to be complete so we can see which interface
        it corresponds to. (It is an error if it doesn't correspond to a single
        interface, by the resolution of
        [#4566](https://github.com/carbon-language/carbon-lang/issues/4566).)
    -   However, an `impl` declaration of a facet type with `.A =`... rewrite
        constraints (for example in a `where` clause), does still require the
        interface to be complete.
-   We answer questions from
    [leads issue #4579](https://github.com/carbon-language/carbon-lang/issues/4579):
    -   Since `impl` forward declarations do not require the interface to be
        defined, any requirements that other interfaces be defined from the
        interface definition are ignored.
    -   An `impl` definition of an interface `I` requires first establishing
        that the type implements any interfaces required by `I`, but that can be
        satisfied by an `impl` declaration.
-   We shift from a "use the information from the type definition if it happens
    to be complete" model to a "only use the information from the definition in
    contexts where it is required to be defined or complete" model
    -   However, an interface `I` containing a `require Self impls J` or
        `extend J` declaration adds a fact that "types `T` satisfying
        `T impls I` also satisfy `T impls J`" to the environment.

These changes are intended to allow developers to write their api files in an
order that roughly follows:

-   namespace declarations in any order;
-   other declarations in any topological dependency order;
-   type (interface, constraint, class) definitions in any order;
-   impl definitions in any order;
-   function definitions, including both those that are needed in the api file
    plus those the developer wishes to include, in any order.

This would be sufficient except we have a few additional constraints, which
remain:

-   Performing member access into a type requires the type to be defined, not
    just declared.
-   Members of an effectively final impl may only be accessed after their value
    has been established.
-   A function that is called at compile time must be defined before that call
    is executed by the compiler.

### Prior to this proposal

|                                                                                     | prereq                                                                                 | provides                                                                               | complete by [EOF](https://en.wikipedia.org/wiki/End-of-file) |
| :---------------------------------------------------------------------------------- | :------------------------------------------------------------------------------------- | :------------------------------------------------------------------------------------- | :----------------------------------------------------------- |
| `impl C as Y;`                                                                      | **`Y` complete**                                                                       | `C impls Y`                                                                            |                                                              |
| `impl C as Y where .A =` ...                                                        | `Y` complete                                                                           | `C impls Y`                                                                            |                                                              |
| `impl C as Y {` ... `}`                                                             | `Y` complete                                                                           | `C impls Y`                                                                            | `C`                                                          |
| `interface I;`                                                                      |                                                                                        | `I` declared                                                                           | `I`                                                          |
| `interface Y {` <br> ` require Self impls Z;` <br> `}`                              | **`Z` declared**                                                                       | **`Y` complete**                                                                       | `Z`                                                          |
| `interface Y {` <br> ` require Self impls Z;` <br> `}` <br> `impl C as Y {` ... `}` | **open question [\#4579](https://github.com/carbon-language/carbon-lang/issues/4579)** | **open question [\#4579](https://github.com/carbon-language/carbon-lang/issues/4579)** |                                                              |
| `fn F[T:! I](x: T);`                                                                | `I` declared                                                                           |                                                                                        | `F`, `I`                                                     |
| `fn F[T:! I](x: T) {` ... `}`                                                       | `I` complete                                                                           | `F` complete                                                                           |                                                              |
| `interface I;` <br> `class C;` <br> `class D(T:! I);` <br> `fn F(x: D(C));`         | `C impls I`                                                                            |                                                                                        | `I`, `C`?, `D`                                               |
| `interface I;` <br> `class D(T:! I);` <br> `fn F[U:! type](x: D(U));`               |                                                                                        | `U impls I` <br> (implied constraint)                                                  | `I`, `D`, `F`                                                |

### Proposed rules

|                                                                                     | prereq             | provides                                                                               | complete by [EOF](https://en.wikipedia.org/wiki/End-of-file) |
| :---------------------------------------------------------------------------------- | :----------------- | :------------------------------------------------------------------------------------- | :----------------------------------------------------------- |
| `impl C as Y;`                                                                      | **`Y` identified** | `C impls Y`                                                                            | **`Y`, `impl C as Y`**                                       |
| `impl C as Y where .A =` ...                                                        | `Y` complete       | `C impls Y`                                                                            | **`impl C as Y` ...**                                        |
| `impl C as Y {` ... `}`                                                             | `Y` complete       | `C impls Y`                                                                            | `C`                                                          |
| `interface I;`                                                                      |                    | `I` declared                                                                           | `I`                                                          |
| `interface Y {` <br> ` require Self impls Z;` <br> `}`                              | **`Z` identified** | **for any symbolic type `T`, <br> `T impls Y` implies `T impls Z`; <br> `Y` complete** | `Z`                                                          |
| `interface Y {` <br> ` require Self impls Z;` <br> `}` <br> `impl C as Y {` ... `}` | **`C impls Z`**    | **`C impls Y`**                                                                        |                                                              |
| `fn F[T:! I](x: T);`                                                                | `I` declared       |                                                                                        | `F`, `I`                                                     |
| `fn F[T:! I](x: T) {` ... `}`                                                       | `I` complete       | `F` complete                                                                           |                                                              |
| `interface I;` <br> `class C;` <br> `class D(T:! I);` <br> `fn F(x: D(C));`         | `C impls I`        |                                                                                        | `I`, `C`?, `D`                                               |
| `interface I;` <br> `class D(T:! I);` <br> `fn F[U:! type](x: D(U));`               |                    | `U impls I` <br> (implied constraint)                                                  | `I`, `D`, `F`                                                |

> Whether classes need to be complete by the end of the file is not the subject
> of this proposal.

## Details

### An `impl` implements a single interface

We resolve
[leads question #4566](https://github.com/carbon-language/carbon-lang/issues/4566)
with two rules:

-   Implementing an interface does not implement any of the interfaces it
    extends.
-   An impl declaration should implement a single interface.

We allow facet type expressions to the right of `impl`...`as`, as long as that
facet type corresponds to a single interface (ignoring interfaces it extends,
which incidentally seems to
[match Rust supertraits](https://stackoverflow.com/questions/75847426/why-do-we-need-a-separate-impl-for-a-supertrait)).
This supports the use case of `Core.Add` actually being a named constraint
defined in terms of `Core.AddWith`. This requires that the facet type is
identified, at a minimum, when used in an `impl` declaration, so it can be
resolved into the single interface actually being implemented.

The "subsumption" use case of "when an interface X is implemented, Y will be
implemented without the user having to write a separate impl definition" will
instead be handled using
[blanket implementations](/docs/design/generics/details.md#blanket-impl-declarations).
There are a couple of variations:

-   One interface is a strict superset of the other. In this case it would be a
    lot less confusing/surprising if the interfaces use the same implementations
    of functions that overlap. This is accomplished using a
    [`final`](https://github.com/carbon-language/carbon-lang/blob/trunk/docs/design/generics/details.md#final-impl-declarations)
    blanket implementation. For example, we will define
    `final impl forall [T:! type, U:! ImplicitAs(T)] U as As(T)` and have
    `As(T).Convert` forward to `ImplicitAs(T).Convert`. This way types will
    either implement `ImplicitAs(T)` or `As(T)`, and will get an error if they
    try to implement both.

-   One interface can be implemented in terms of the other. For example, if
    `Ordered` implies an implementation of `IsEqual`, types might still want to
    provide an explicit definition of `IsEqual` when they can do so more
    efficiently. This would use a non-`final` blanket implementation.

Note that the orphan rule prevents this blanket impl from being written unless
the two interfaces in a subsumption relationship are in the same library.
Support for these use cases is
[future work](#addressing-the-subsumption-use-case-previously-addressed-by-extend-in-interfaces).
For now, `extend I;` in an interface definition continues to mean that `I` is
required and the names of `I` are included as aliases, matching the meaning in
named constraints, see "interface extension" in
[proposal #553](https://github.com/carbon-language/carbon-lang/pull/553).

### Associated constants may be assigned in the body of an `impl` definition

We resolve
[leads question #4672](https://github.com/carbon-language/carbon-lang/issues/4672)
with the following updated rules:

-   The `impl` definition must be in the same file as its owning declaration.
-   The first declaration establishes that the type implements the interface.
-   That declaration may optionally specify a subset of the associated constants
    in a `where` clause. The `where` clause syntax is the same as before this
    proposal, but the requirement that all associated constants be given their
    final values (or accept their defaults where available) is removed.
    -   This adds some flexibility to ordering, allowing a developer to say that
        a type implements an interface before being able to specify the values
        of its associated constants, while also allowing a subset to be
        specified if some later declaration needs to access their values.
-   Redeclarations of an `impl` no longer have any special treatment of `where`
    clauses.
    -   Each redeclaration is required to match syntactically, so all will have
        the same `where` clause, if any.
    -   The part between `impl` and the end of the declaration (marked by a `{`
        or `;`) is the name of the `impl`. This will be used in places where we
        want to name the `impl`, such as in an impl priority ("match first")
        block or out-of-line definitions of `impl` members.
-   Associated constant assignments should be encouraged to be placed in the
    `impl` definition rather than in a `where` clause in the declaration, unless
    needed in the declaration to resolve dependencies.
    -   This makes the name of the `impl` shorter and easier to state.
    -   The syntax for specifying the value of an associated constant in an
        `impl` definition is `where X = value;`, just like a rewrite clause in a
        `where` expression, except that `where` in this case is an introducer
        rather than a binary operator, and the names of the associated constants
        are in scope, so there is no need to prefix them with a period (`.`).
-   The non-function associated constants are fixed at the closing curly `}` of
    the `impl` definition. At that point, each associated constant that has not
    been assigned a value is given its default value. If it does not have a
    default, an error diagnostic is issued.
-   The associated function declarations are fixed by the end of the `impl`
    definition.
    -   Only associated functions with defaults may be omitted from the `impl`
        definition.
    -   Not mentioning an associated function with a default in the `impl`
        definition means the default version from the interface will be used.
    -   Associated functions declared in the `impl` definition must have a
        matching definition.
    -   If the declared signature of an associated function in the `impl`
        definition does not match the signature given in the `interface`
        definition, but are compatible, a stub function that does the conversion
        is generated to bridge between the two versions.
-   Out-of-line function bodies are allowed in the impl file, even when the
    `impl` definition is in the api file.
    -   This parallels how function definitions are not part of the definition
        of classes.
-   Missing function body definitions are diagnosed at link time.

### An `impl` may be forward declared without the interface being complete

We have removed the reasons to require that the interfaces being implemented are
defined for an `impl` forward declaration:

-   We no longer need to see which interfaces are required or extended from the
    definition of the interface, since the `impl` no longer implements those.
-   We no longer need to see which associated constants are members of the
    interface to see if a declaration specifies values for all of them.
-   If a `where` clause is used in the declaration, the interface must be
    defined in order to name the associated constants, but `where` clauses are
    discouraged.

However, if the facet type being implemented is a named constraint, we do need
that to be complete so we can resolve the interface it resolves to. (It is an
error if it doesn't correspond to a single interface, by the resolution of
[#4566](https://github.com/carbon-language/carbon-lang/issues/4566).)

This means that generally we can establish that a type implements an interface
right after they are both declared.

### Interface requirements

[Leads issue #4579](https://github.com/carbon-language/carbon-lang/issues/4579)
concerns itself with interfaces that require other interfaces to be implemented,
as in:

```
interface Z;

interface Y {
  require Self impls Z;
}
```

There are two sides to this: what do you have to establish about `Z` before an
`impl` declaration or definition that a type implements `Y`, and what does a
declaration or definition that a type implements `Y` establish about `Z`?

We adopt the following rules:

-   Since `impl` forward declarations do not require the interface to be
    defined, any requirements that other interfaces be defined from the
    interface definition are ignored.

    ```
    class C1;
    // ✅ Allowed
    impl C1 as Y;

    // `class C1` and `impl C1 as Y` must be defined in the same file.
    ```

-   An `impl` definition of an interface `I` requires first establishing that
    the type implements any interfaces required by `I`, but that can be
    satisfied by an `impl` declaration.

    ```
    class C2;
    // ❌ Invalid, must first establish `C2 impls Z`. Still invalid if
    // `impl C2 as Z` appears later in the file.
    impl C2 as Y {}

    class C3;
    // ✅ Allowed
    impl C3 as Z;
    impl C3 as Y {}

    class C4;
    // ✅ Allowed
    impl C4 as Z {}
    impl C4 as Y {}

    // The classes and `impl C3 as Z` must be defined in the same file.
    ```

This is is aligned with the shift from a "use the information from the type
definition if it happens to be complete" model to a "only use the information
from the definition in contexts where it is required to be defined or complete"
model that this proposal this is switching to. To recapture some of that context
sensitivity, we say that the definition of an interface with requirements, like
`Y` above, introduces the extra information that "symbolic types `T` that
satisfy `T impls Y` also satisfy `T impls J`."

For example:

```
interface Z;

interface Y {
  require Self impls Z;
}

class C(T:! Z);

class D(U:! Y) {
  // ✅ U impls Y so it also impls Z.
  //
  // Wouldn't use implied constraints here since `U` is
  // from a containing scope.
  fn F(x: C(U));
}
```

This rule only applies to symbolic types, since we want to only say that
concrete types implement an interface if we have a declared `impl` to witness
that fact. Symbolic types depend on the value of some generic parameters and we
accept that some accesses of interface members will result in symbolic values
that will only have known values once concrete argument values are supplied for
the generic parameters. For concrete types, the access is performed immediately,
and it is an error if we don't have an `impl` declaration we can point to with
the witness. For concrete types, there is no later point where an argument is
supplied to delay these checks. For example:

```
interface Z;

interface Y {
  require Self impls Z;
}

class C(T:! Z);

class D {}

// ✅ Allowed, since `impl` declarations are allowed for declared
// entities. Interface requirements are not considered.
impl D as Y;

// ❌ Error: D is not known to implement Z. The fact that Y
// requires Z is not used since D is a concrete type.
fn F(x: C(D));

// Too late to affect the previous declaration, by the information
// accumulation principle.
impl D as Z;
```

Similarly, it is also an error to access a member of an `impl` of a concrete
type that doesn't have a known value, even if it is given a value later in the
file. For example:

```
interface I {
  let A:! type;
  let B:! type;
}

fn F(T:! I) -> T.A;

class C {}
class D {}

impl C as I;

fn G1() {
  // ❌ Error: C impls I, but C.(I.A) is unknown.
  var x: auto = F(C);
  // ❌ Error: C.(I.A) is unknown.
  let y: C.(I.A) = 0;
  // ❌ Error: C.(I.B) is unknown.
  let b: C.(I.B) = false;
}

impl D as I where .A = i32;

fn H() {
  // ✅ Allowed: D.(I.A) = i32;
  var x: auto = F(D);
  var y: D.(I.A) = 0;
  // ❌ Error: D.(I.B) is unknown.
  let b: D.(I.B) = false;
}

impl C as I {
  where A = i32;
  where B = bool;
}

fn G2() {
  // ✅ Allowed: C.(I.A) = i32;
  var x: auto = F(C);
  let y: C.(I.A) = 0;
  // ✅ Allowed: C.(I.B) = bool;
  let b: C.(I.B) = false;
}
```

### Examples

This example shows using incomplete entities following the rules of this
proposal:

```
interface X;

// ✅ Allowed to use incomplete interfaces in function declarations.
fn F(U:! X);

class C;

// ✅ Allowed to use incomplete types and interfaces in impl declarations.
impl C as X;

interface Y;

interface X {
  // ✅ Allowed to use an incomplete interface.
  require Self impls Y;
}

// Classes must be defined before being used in a function definition.
class C { ... }

fn G() {
  // ✅ Allowed since C is complete and we have a declaration `impl C as X;`
  F(C);
}

// The above declarations require that `interface Y`, `fn F` (since it is
// generic), and `impl C as X` are defined in the same file.
interface Y { ... }
fn F(U:! X) { ... }
// Required for `impl C as X` definition.
impl C as Y;
impl C as X { ... }
impl C as Y { }
```

This example demonstrates using associated constants and interface requirements:

```
interface X2 {
  let A:! type;
  let B:! type;
}
interface Y2 {
  require Self impls X2;
  // ✅ Allowed since `X2` is required and we can access
  // its `A` member since `X2` is complete.
  fn F() -> X2.A;
}

class C2 {}

impl C2 as X2 where .A = i32;

// ✅ Allowed since `C2` and `Y2` are complete, and
// `C2 impls X2` so `C2` satisfies the requirement in `Y2`.
impl C2 as Y2 {
  // ✅ Allowed since the value of `C2.(X2.A)` is known to
  // be `i32`, even though `impl C2 as X2` is not complete.
  fn F() -> i32;
}

// There needs to be a definition of `C2 as X2 where .A = i32`
// in the same file. The `where` clause needs to be repeated
// verbatim, since redeclaration requires a syntactic match.
impl C2 as X2 where .A = i32 {
  // Remaining members of `X2` that do not have default
  // values need to be assigned.
  where B = i32;
}
```

## Future work

### Addressing the subsumption use case previously addressed by `extend` in interfaces

We do expect to have collections of interfaces that have a stronger "extending"
relationship than is provided by the current `extend` declaration in interfaces.
For example, a type implementing `ImplicitAs` should not have to have a separate
declaration that it also implements `As`. Addressing this use case is out of
scope of this proposal, though, and will be addressed in a future proposal. This
future proposal may include:

-   A feature to copy the members of an interface into another to make the
    subsumption use case easier to write and involve less duplication.

-   A feature to define an implementation of an interface in terms of another by
    forwarding, for similar reasons.

-   A `final` version of the `match_first`/`impl_priority` feature to resolve
    conflicts when multiple interfaces want to subsume a common interface. We
    likely want a feature like this for function overloading as well.

-   Some way of handling an `impl` that could overlap with a `final impl`, but
    doesn't in practice.

-   Possible support for implementing multiple interfaces with a single impl
    definition, as the result of using an `&` operator or named constraint to
    the right of `as`, as in
    [the considered alternative](#allow-implementing-multiple-interfaces-with-a-single-impl-declaration).

For now, we leave `extend` as meaning "requires plus include aliases of the
names," matching the behavior in named constraints. But this will be
reconsidered once we have support for this other use case.

### Opt-in to using the interface's default definition of an associated function

We've considered that we may want to allow an `impl` to opt into using the
default definition of a function from the interface by writing `= default;`
instead of an inline body in curly braces `{`...`}`. We will see if that is a
desirable construct to add with experience. This idea was suggested in
[issue #4672](https://github.com/carbon-language/carbon-lang/issues/4672#issuecomment-2539845620).

## Rationale

The specific solution was chosen to align with
[the information accumulation principle](https://docs.carbon-lang.dev/docs/project/principles/information_accumulation.html).
In particular, allowing `impl` declarations for incomplete interfaces gives
additional flexibility to developers to satisfy those constraints.

By reducing the different behavior based on whether a previous declaration was a
definition, this proposal reduces complexity in the toolchain and tools that
operate on Carbon source code. This benefits the
[Language tools and ecosystem](/docs/project/goals.md#language-tools-and-ecosystem)
and
[Fast and scalable development](/docs/project/goals.md#fast-and-scalable-development)
[Carbon goals](/docs/project/goals.md).

This is intended to also help humans have a simpler mental model of the
compiler, to help the
[Code that is easy to read, understand, and write](/docs/project/goals.md#code-that-is-easy-to-read-understand-and-write)
goal.

## Alternatives considered

The trade offs and alternatives were discussed in
[this document](https://docs.google.com/document/d/1NNhBU5tywGkeneLmTQ4DwHyCp5vYlwTiR_wmnjn9wnM/edit)
and in open discussion meetings on these dates:

-   [2025-02-07](https://docs.google.com/document/d/1Iut5f2TQBrtBNIduF4vJYOKfw7MbS8xH_J01_Q4e6Rk/edit?resourcekey=0-mc_vh5UzrzXfU4kO-3tOjA&tab=t.0#heading=h.6ziehwl15x6a)
-   [2025-03-04](https://docs.google.com/document/d/1Iut5f2TQBrtBNIduF4vJYOKfw7MbS8xH_J01_Q4e6Rk/edit?resourcekey=0-mc_vh5UzrzXfU4kO-3tOjA&tab=t.0#heading=h.22fvs78tr2ej)
-   [2025-03-05](https://docs.google.com/document/d/1Iut5f2TQBrtBNIduF4vJYOKfw7MbS8xH_J01_Q4e6Rk/edit?resourcekey=0-mc_vh5UzrzXfU4kO-3tOjA&tab=t.0#heading=h.x0brmxoi93n4)
-   [2025-03-06](https://docs.google.com/document/d/1Iut5f2TQBrtBNIduF4vJYOKfw7MbS8xH_J01_Q4e6Rk/edit?resourcekey=0-mc_vh5UzrzXfU4kO-3tOjA&tab=t.0#heading=h.lsqxlvgwdvoh)
-   [2025-03-13](https://docs.google.com/document/d/1Iut5f2TQBrtBNIduF4vJYOKfw7MbS8xH_J01_Q4e6Rk/edit?resourcekey=0-mc_vh5UzrzXfU4kO-3tOjA&tab=t.0#heading=h.lsufbqunqul0)
-   [2025-03-18](https://docs.google.com/document/d/1Iut5f2TQBrtBNIduF4vJYOKfw7MbS8xH_J01_Q4e6Rk/edit?resourcekey=0-mc_vh5UzrzXfU4kO-3tOjA&tab=t.0#heading=h.v0stu27nansb)
-   [2025-03-20](https://docs.google.com/document/d/1Iut5f2TQBrtBNIduF4vJYOKfw7MbS8xH_J01_Q4e6Rk/edit?resourcekey=0-mc_vh5UzrzXfU4kO-3tOjA&tab=t.0#heading=h.jagz2i1dynjx)

### Allow implementing multiple interfaces with a single `impl` declaration

This was considered in
[leads question #4566](https://github.com/carbon-language/carbon-lang/issues/4566).
There are definite use cases for this feature, particularly arising from
evolution. For example, you might want to split an interface into two new
interfaces, and have a named constraint with the original name extend both so
that existing code continues to work the same. With this proposal, those changes
will be harder and have more steps.

Two specific approaches to implementing multiple interfaces were eliminated from
consideration in that issue:

-   The "all or nothing" approach where the `impl` definition is used for all of
    the interfaces, or none of them are. This would create too much uncertainty
    about whether an `impl` is applicable, particularly since constraints in
    generic code are not sensitive to whether something is specialized.
-   The "Constrained impls" approach where an `impl` of multiple interfaces is
    treated as a collection of `impl`s of the individual interfaces with the
    additional constraint that no specialization changes the values of any
    non-function associated constants of any of the interfaces. Those
    constraints though are ultimately circular and not well defined.

The remaining "independent impls" approach seemed possible. In this approach an
impl of multiple interfaces is treated as a collection of impls of the
individual interfaces. In particular, the definition of a member of one
interface can assume that the other interfaces are implemented, but not that the
associated types (or other non-function associated constants) have expected
values. This would introduce some complexities and a number of questions would
need to be answered around how a single `impl` definition would be split into
definitions of the individual interfaces, how dependencies between those pieces
would be resolved, and how these restrictions would be exposed to the user in
diagnostics.

### An `impl` of an interface also implements the interfaces it extends

This was the design before this proposal, but in
[leads question #4566](https://github.com/carbon-language/carbon-lang/issues/4566)
we found a number of problems with that approach:

-   A parameterized interface extending a non-parameterized interface, or an
    interface with fewer parameters, leads to multiple implementations of the
    extended interface.

-   There are multiple possible semantics you might want, and having a single
    `impl` does not provide the affordances for choosing between those options,
    where one `impl` per interface would. For example, in
    `impl forall [T:! type] C(T) as I & J where .(I.x) = i32 and .(J.y) = .(I.x)`,
    if there is a specialization of `C(T)` for `I`, will `J.y` have the value
    `i32` or the `I.x` from the specialization? In practice, the semantics of
    rewrites mean that `.(I.x)` is replaced with `i32` at an early stage in the
    compiler (to support things like `.(J.y) = .(I.x).D`), and so only the first
    option is consistent. This is a particular concern for the "Independent
    impls" option above. If this `impl` is split into two, then the different
    possible meanings have different spellings:

    -   `impl forall [T:! type] C(T) as J where .(J.y) = i32` means `J.y` will
        be `i32` independent of any specialization of `C(T)` for `I`

    -   `impl forall [T:! type where C(T) impls I] C(T) as J where .(J.y) = .(I.x)`
        means `J.y` matches `I.x` even if `C(T)` is specialized

    -   `impl forall [T:! type where C(T) impls (I where .x = i32)] C(T) as J where .(J.y) = .(I.x)`
        means this impl won't be used unless `I.x` is `i32`. Note this last form
        approximates the "Constrained impls" approach above, but with an
        explicit ordering to determine the semantics, and the existing language
        rules preventing the code from declaring cycles that would make it
        ambiguous.

-   If an interface `J` extends `I` but they are defined in distinct libraries,
    there is no guarantee that an implementation of `J` belongs in the same
    library as an implementation of `I` for the same type due to the orphan
    rule.

-   The current documented rule for which interfaces an impl implements is those
    whose members are defined in the interface definition. This rule is
    ambiguous for empty interfaces or interfaces where all the associated
    functions have defaults. It also requires
    [a lot of context](https://github.com/carbon-language/carbon-lang/blob/trunk/docs/project/principles/low_context_sensitivity.md)
    to answer that question (this was intentional, to allow options for
    refactoring an interface without having to update implementations of it). In
    addition to being the source of readability concerns, this muddies the
    meaning of `impl` declarations, and make the compiler implementation much
    trickier (the compiler can't say what an impl declaration provides at the
    point where it is written, making it hard to give that declaration a clear
    type in the SemIR).

-   Ideas we considered to determine the interfaces implemented from only the
    impl declaration ran into problems. Without being able to control which
    interfaces an impl is defining, then it isn't clear how to handle
    implementing two interfaces that have common interface they both extend
    unless you implement them both in a single impl definition (which may not
    even be possible due to the orphan rule). Another idea we had in this space
    was a way to say "this interface minus one of the interfaces it extends"
    (maybe `J \ I`?).

We considered some restrictions on `extend` to address some of these concerns:

-   Perhaps implementing an interface only implemented the interfaces it extends
    that have to be defined in this library by the orphan rule.

-   Perhaps an interface may only extend an interface defined in the same
    library.

-   Perhaps an interface may only be extended by a single interface. This
    precluded motivating use cases for `extend`, though, like the subtyping
    relationships between different kinds of graphs used in
    [the Boost Graph library](https://docs.google.com/document/d/15Brjv8NO_96jseSesqer5HbghqSTJICJ_fTaZOH0Mg4/edit?resourcekey=0-CYSbd6-xF8vYHv9m1rolEQ&tab=t.0)).

Changing the meaning of `extend` is left for a future proposal, when we address
[the subsumption use case previously addressed by `extend` in interfaces](#addressing-the-subsumption-use-case-previously-addressed-by-extend-in-interfaces).

### Use the contents of definitions if available

In the course of implementing the existing design, uses of "TryToCompleteType"
function were found to be prone to leading to coherence issues. If being
incomplete does not lead to an error, we need to establish that the results of
the definition appearing before and after that test are the same. When there
were multiple types involved, this led to an explosion of combinations to test.
The new model has less conditional logic, and as a result less complexity.

### Delayed checking of incomplete types

We could delay checking uses of incomplete types until some later point, either
when the type is complete, a use that requires that type to be complete, or the
end of the file where we know the most about it. This is an option, and we might
adopt it if we found that it was too hard in practice to satisfy the constraints
adopted by this proposal. However it would significantly increase the complexity
of the toolchain implementation, which would lead to a corresponding increase in
the difficulty of understanding how the code will be interpreted.

### Allow `impl` declarations with rewrites of defined but not complete interfaces

We at first did not have a rule saying that `I` needed to be complete in
`impl C as I where .X = ...`. The rationale was that accessing members of `I`
only needed `I` to be defined, not complete. However, an `impl` declaration
inside the definition of the interface being implemented could ultimately never
be defined, even if we allowed the declaration. This is because the `impl`
definition would have to be in a different scope than its declaration, as can be
seen in this example that uses a lambda function to get a scope that can have an
`impl` declaration inside an `interface` definition:

```
interface I {
  let U:! type;
  default let T:! type = (fn() -> type {
    class C { }

    // ✅ Allowed since `I` is declared, with the exception that
    // this requires a definition before the end of the file.
    impl C as I;

    // ✅ Would be allowed since `I` is defined, including its
    // member `U`. Again this requires a definition in this file.
    impl C as I where .U = C;

    // ❌ Neither of the above impls may be defined:
    // - Can't be defined here since `I` is not complete.
    // - Can't define after `I` is complete, since redeclarations
    //   must match syntactically and have no hope of naming this
    //   `C` that way.
    // - In general, the impl definition would have to re-enter the
    //   same scope.
    impl C as I where .U = C {}

    return C;
  })();
}
```

This simplifies the toolchain implementation of this feature, since it means we
can create an `impl` witness for declarations that use rewrite constraints with
the full knowledge of the interface's definition.

### Can omit function declarations from `impl` body

We considered saying that an `impl` definition does not need to include
declarations that are unchanged from the interface definition. However, this
raised a number of questions and problems:

-   The `interface` definition is in a different scope from the `impl`, and
    potentially a different file. This, combined with the evolution problems of
    depending on the specific token sequence used in the `interface` definition,
    suggest that we would need a different matching rule that was semantic
    instead of syntactic.
-   Having a different declaration matching rule was anticipated as creating a
    bunch of difficulties once function overloading is added to Carbon,
    particularly overloaded functions in interfaces.
-   Without a declaration in the `impl` body, there were a number of problems
    resolving how to handle a function in the `impl` that was compatible with
    but a different signature from the interface, and when that different
    signature would be used.
-   These issues seemed to magnify if the interface had a default definition for
    the function.

### Different introducer for assigning associated constants in an `impl` definition

Other options we considered were `provides`, `alias`, and whatever we eventually
choose for [#5028](https://github.com/carbon-language/carbon-lang/issues/5028).
We ultimately liked matching how you would assign associated constants in the
declaration using the `where` operator and a rewrite constraint, particularly
since we wanted to use the same semantics.
