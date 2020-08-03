# Unions

<!--
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

## Table of contents

<!-- toc -->

-   [Overview](#overview)

<!-- tocstop -->

## Overview

Fields of a struct can be grouped into _unions_:

```
struct Number {
  union {
    var Int: int_value;
    var Double: double_value;
  }

  // 0 if int_value is active, 1 if double_value is active.
  var Int1: discriminator;
}
```

All members of a union share the same storage, and at most one member of a given
union can be active at a time. It is an error to access a member that is not
active, even if it has the same type as the active member. There is no intrinsic
way to determine which member of the union is active; user code is responsible
for tracking that information, as with the `discriminator` field in the above
example.

**TODO:** Carbon should have a separate facility for more user-friendly
discriminated unions. It will also need a separate facility for "type punning",
i.e. reinterpreting the representation of one type to the representation of a
different type. When those are designed, cite them above.

Unlike C++ unions, Carbon unions never have names, are not objects, and do not
have types; they are a means of controlling the layout of fields in a struct.
Consequently, it is not possible to form a pointer to a union. You can form a
pointer to a member field of a union, even if it is not active, but you cannot
cast it to a pointer to the type of a different member.

The active member can only be changed by destroying the current active member
(if any), and then constructing the new active member, using the `destroy` and
`create` keywords:

```
fn SetDoubleValue(Ptr(Number): n, Double: value) {
  if (n->discriminator == 0) {
    destroy n->int_value;
    create: n->double_value = value;
    n->discriminator = 1;
  } else {
    n->double_value = value;
  }
}
```

`create` and `destroy` correspond to placement-`new` and pseudo-destructor calls
in C++, but they can only be applied to union members; the lifetimes of ordinary
struct fields and variables are always tied to their scope. `destroy` can be
thought of as a unary operator, but a `create` statement has the syntax and
semantics of a variable declaration, with `create` taking the place of `var`,
and the field expression taking the place of a variable name.

**Concern**: The `create` syntax may not make it sufficiently clear that the `=`
represents initialization, not assignment.

**TODO:** The spelling of `create` and `destroy` are chosen for consistency with
`operator create` and `operator destroy`, the spelling of constructor and
destructor declarations in the currently-pending structs proposal. They should
be updated as necessary to stay consistent.

A union member can be either a field or a _group_ of fields:

```
struct SsoString {
  bitfield(1) var Bool: is_small;

  union {
    group small {
      bitfield(7) var Int7: size;
      var FixedArray(Char, 22): buffer;
    }
    group large {
      bitfield(63) var Int63: size;
      var Int64: capacity;
      var UniquePtr(Array(Char)): buffer;
    }
  }
}
```

**TODO:** The treatment of bitfields, arrays, and owning pointers in this
example is speculative, and should be updated to reflect the eventual design of
those features.

Field groups are sets of fields that can be made active or inactive as a unit.
They are initialized from anonymous structs whose fields have the same types,
names, and order, and their names are part of the names of their fields:

```
var SsoString: str = {.is_small = True,
                      .small = {.size = 0, .buffer = uninitialized}
                     };
Assert(str.small.size == 0);
destroy str.small;
create: str.large = {.size = 0, .capacity = 100, .buffer = MakeBuffer(100)};
Assert(str.large.capacity == 100);
```

However, field groups are not objects, and do not have types; like unions, field
groups are a way of controlling the layout of the fields of a struct. For
example, if `large` and `small` were structs, the bitfields would not save any
space, and `is_small` would have to be followed by 63 bytes of padding in order
to ensure proper alignment of `large`.

FIXME: Interop and safety stories.

FIXME: rationales for above decisions (including general discussion of low
level)
