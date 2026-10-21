# Associated constants

<!--
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

<!-- toc -->

## Table of contents

-   [Overview](#overview)
-   [Declaration checking](#declaration-checking)
-   [Specifying rewrite constraints](#specifying-rewrite-constraints)
-   [Definition of associated constant values](#definition-of-associated-constant-values)
-   [Use of associated constants](#use-of-associated-constants)
    -   [Simple member access](#simple-member-access)
    -   [Compound member access](#compound-member-access)
    -   [Forming the constant value](#forming-the-constant-value)

<!-- tocstop -->

## Overview

_Note:_ This document only describes non-function associated constants.

An associated constant is declared within an interface scope with the syntax:

```carbon
[MODIFIERS] let NAME: TYPE [= INITIALIZER] ;
```

Associated constants introduce a slot in the witness table for an interface that
contains a value of type `TYPE`.

Associated constants are declared within the interface-with-self generic, which
is parameterized by the `Self` type of the interface, and is nested within the
interface generic, which is parameterized by any other generic parameters of the
interface. Symbolic instructions within the declaration of an associated
constant, such as those used to compute its type, are part of the
interface-with-self generic.

Associated constant entities are held in the `associated_constants` value store
as objects of type `AssociatedConstant`. Each declaration of an associated
constant is modeled by an `AssociatedConstantDecl` instruction. Each such
instruction is then wrapped in an `AssociatedEntity` instruction which
represents the slot within an interface witness where the constant's value can
be found.

## Declaration checking

Because associated constants share the syntax of `let` declarations, a lot of
the checking logic is also shared. This logic is in
[handle_let_and_var.cpp](/toolchain/check/handle_let_and_var.cpp). The parser
produces distinct parse nodes for a `let` declaration in an interface scope, and
associated constant declaration handling proceeds as follows:

1.  ```carbon
    let NAME: TYPE [= INITIALIZER] ;
    ^
    ```

    The handler for `AssociatedConstantIntroducer` is called at the start of
    the declaration. This:

    -   Pushes an instruction block to hold instructions within the declaration
        of the constant.
    -   Performs the same setup as for other `let` declarations, such as
        starting a full pattern and an expression region for the type.

2.  ```carbon
    let NAME: TYPE [= INITIALIZER] ;
        ~~~~^~~~~~
    ```

    Process the name and type. This is done by the handler for
    `AssociatedConstantNameAndType` in
    [handle_binding_pattern.cpp](/toolchain/check/handle_binding_pattern.cpp),
    which creates an `AssociatedConstantDecl` and corresponding
    `AssociatedConstant` entity. This instruction is then produced as the
    pattern for the declaration.

3.  ```carbon
    let NAME: TYPE ;
                   ^
    let NAME: TYPE = INITIALIZER ;
                   ^
    ```

    When we reach the end of the pattern, either because we reached the `=` or
    because we reached the `;` and there was no initializer, the full pattern
    is ended and `EndAssociatedConstantDeclRegion` is called. This:

    -   Builds an `AssociatedEntity` object, reserving a slot in the
        interface's witness table for the constant.
    -   Adds the associated constant to name lookup.

4.  ```carbon
    let NAME: TYPE = INITIALIZER ;
                   ^
    ```

    If there is an initializer, the handler for `AssociatedConstantInitializer`
    starts processing it, in the same way as for other `let` declarations.

5.  ```carbon
    let NAME: TYPE [= INITIALIZER] ;
                                   ^
    ```

    At the end of the declaration, the handler for `AssociatedConstantDecl`
    finalizes the declaration. This:

    -   If the pattern is an error, marks the interface-with-self scope as
        having an error, and discards the instruction block.
    -   Otherwise:
        -   If there is an initializer, converts it to the type of the
            constant, and stores the result as the `default_value_id` of the
            `AssociatedConstant`.
        -   Pops the instruction block created in step 1 and attaches it to
            the `AssociatedConstantDecl`.
        -   Adds the `AssociatedConstantDecl` to the enclosing instruction
            block.

## Specifying rewrite constraints

TODO: Fill this out. In particular, note that we do not convert the rewrite to
the type of the associated constant as part of forming a `where` expression if
the constant's type is symbolic, and instead defer that until the facet type is
resolved.

## Definition of associated constant values

Associated constant values are stored into witness tables as part of impl
processing in [impl.cpp](/toolchain/check/impl.cpp).

TODO: Fill this out once the new model is implemented.

## Use of associated constants

The work to handle uses of associated constants starts in
[member_access.cpp](/toolchain/check/member_access.cpp).

When an `AssociatedEntity` is the member in a member access, impl lookup is
performed to find the corresponding impl witness. The self type in impl lookup
depends on how the member name was found.

### Simple member access

In `LookupMemberNameInScope`, if lookup for `y` in `x.y` finds an associated
constant from interface `I`, then a witness is determined as follows:

-   If the lookup scope is the type `T` of `x`, then `PerformImplLookup` is
    called to perform impl lookup for `T as I`.
-   If the lookup scope is `x` itself, then:
    -   If `x` is a namespace or a facet type other than `type`, impl lookup is
        not performed, and the result is simply `y`. This happens for cases
        such as `Interface.AssocConst`.
    -   Otherwise, `x` must be a type, and `PerformImplLookup` is called to
        perform impl lookup for `x as I`.

### Compound member access

In `PerformCompoundMemberAccess` for `x.(y)`, if `y` is an associated constant
from interface `I`, then `GetAssociatedValueImpl` converts `x` itself to a facet
value of type `I`, and performs impl lookup for `x as I` to find the witness
containing the constant value. The same logic is used by `GetAssociatedValue`,
which finds the value of an associated entity for a given type or facet without
going through member access syntax.

_Note:_ This differs from the handling of associated functions that are
instance methods, for which impl lookup is performed for `T as I`, where `T`
is the type of `x`.

### Forming the constant value

Once the witness is determined, a specific for the interface-with-self generic
is formed by `MakeSpecificWithInnerSelf` from the specific for the interface and
the self facet value. For simple member access, this happens in
`PerformImplLookup`, which then calls `AccessMemberOfImplWitness`. For compound
member access, this happens directly in `GetAssociatedValueImpl`.

`GetTypeForSpecificAssociatedEntity` is then used to form the type of the
constant by substituting the interface-with-self specific into the type of the
associated constant. Then, an `ImplWitnessAccess` instruction is created to
extract the relevant slot from the witness. Constant evaluation of this
instruction reads the associated constant from the witness table.

If the access appears within a `where` expression that has a rewrite constraint
for the same associated constant, the `ImplWitnessAccess` is wrapped in an
`ImplWitnessAccessSubstituted` instruction that also records the rewritten
value.
