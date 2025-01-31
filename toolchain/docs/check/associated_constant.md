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
[MODIFIERS] let NAME:! TYPE [= INITIALIZER] ;
```

Associated constants introduce a slot in the witness table for an interface that
contains a value of type `TYPE`.

Associated constants are always generic entities, because they're always
parameterized at least by the `Self` type of the interface, as well as any other
enclosing generic parameters. Note that the interface itself is _not_
parameterized by its `Self`.

Associated constant entities are held in the `associated_constants` value store
as objects of type `AssociatedConstant`. Each declaration of an associated
constant is modeled by an `AssociatedConstantDecl` instruction. Each such
instruction is then wrapped in an `AssociatedEntity` instruction which
represents the slot within an interface witness where the constant's value can
be found.

## Declaration checking

Because associated constants share the syntax of `let` declarations, a lot of
the checking logic is also shared. This logic is in
[handle_let_and_var.cpp](/toolchain/check/handle_let_and_var.cpp). Associated
constant declaration handling proceeds as follows:

1.  ```carbon
    let NAME:! TYPE [= INITIALIZER] ;
    ^
    ```

    `StartAssociatedConstant` is called at the start of an interface-scope `let`
    declaration. This:

    -   Starts a generic declaration region.
    -   Pushes an instruction block to hold instructions within the declaration
        of the constant. These form the body of the generic.

2.  ```carbon
    let NAME:! TYPE [= INITIALIZER] ;
        ~~~~^~~~~~~
    ```

    Process the symbolic binding pattern. This is done in
    [handle_binding_pattern.cpp](/toolchain/check/handle_binding_pattern.cpp),
    which detects that we are at interface scope, and creates an
    `AssociatedConstantDecl` and corresponding `AssociatedConstant` entity. This
    binding is then produced as the instruction associated with the binding
    pattern.

    _Note:_ This is somewhat unusual: usually, a pattern instruction would be
    associated with a pattern parse node.

3.  ```carbon
    let NAME:! TYPE ;
                    ^
    let NAME:! TYPE = INITIALIZER ;
                    ^
    ```

    When we reach the end of the pattern in an interface-scope `let` binding,
    either because we reached the `=` or because we reached the `;` and there
    was no initializer, `EndAssociatedConstantDeclRegion` is called. This:

    -   Ends the generic declaration region.
    -   Builds an `AssociatedEntity` object, reserving a slot in the interface's
        witness table for the constant.
    -   Adds the associated constant to name lookup.

    _Note:_ The pattern might not be valid for an associated constant. In this
    case, we won't have built an `AssociatedConstantDecl` in the previous step.
    When this happens, we instead just discard the generic declaration region
    and continue. The invalid pattern will be diagnosed later.

4.  ```carbon
    let NAME:! TYPE = INITIALIZER ;
                    ^
    ```

    If there is an initializer, we start the generic definition region.

5.  ```carbon
    let NAME:! TYPE [= INITIALIZER] ;
                                    ^
    ```

    At the end of the declaration, `FinishAssociatedConstant` is called to
    finalize the declaration. This:

    -   Diagnoses if the pattern handling didn't create an
        `AssociatedConstantDecl`.
    -   Finishes handling the initializer, if it's present:
        -   Converts the initializer to the type of the constant.
        -   Ends the generic definition region.
    -   Pops the inst block created by `StartAssociatedConstant` and attaches it
        to the `AssociatedConstantDecl`.
    -   Adds the `AssociatedConstantDecl` to the enclosing inst block.

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

-   If the lookup scope is the type `T` of `x`, then:
    -   If `T` is a non-type facet, the witness for that facet is used. TODO:
        That facet might not contain a witness for `I`. In that case we will
        need to perform impl lookup for `T as I` instead.
    -   Otherwise, impl lookup for `T as I` is performed to find the witness.
-   If the lookup scope is `x` itself, then:
    -   If `x` is a facet type or a namespace, impl lookup is not performed, and
        the result is simply `y`. This happens for cases such as
        `Interface.AssocConst`.
    -   Otherwise, `x` must be a type other than a facet type, and impl lookup
        for `x as I` is performed to find the witness.

### Compound member access

In `PerformCompoundMemberAccess` for `x.(y)`, if `y` is an associated constant
then impl lookup is performed for `T as I`, where `T` is the type of `x` and `I`
is the interface in which `y` is declared to find the witness containing the
constant value.

### Forming the constant value

Once the witness is determined, `AccessMemberOfImplWitness` is called to find
the value of the associated constant in the witness. In the case where an impl
lookup is needed, `PerformImplLookup` calls `AccessMemberOfImplWitness`,
otherwise it's called directly.

`AccessMemberOfImplWitness` uses `GetTypeForSpecificAssociatedEntity` to form
the type of the constant. This substitutes both the generic arguments (if any)
for the interface and the `Self` type into the type of the associated constant.
Then, an `ImplWitnessAccess` instruction is created to extract the relevant slot
from the witness. Constant evaluation of this instruction reads the associated
constant from the witness table.
