# Unions

<!--
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

## Table of contents

<!-- toc -->

-   [Overview](#overview)
-   [Safety](#safety)
-   [C++ interoperability and migration](#c-interoperability-and-migration)

<!-- tocstop -->

FIXME: add a proposal doc with rationales for these decisions (including general
discussion of the fact that this is a low-level facility)

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
and the field expression taking the place of the variable name.

**FIXME**: Does the `create` syntax make it sufficiently clear that the `=`
represents initialization, not assignment? Can we do better?

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

Field groups are sets of fields (and/or unions) that can be made active or
inactive as a unit. They are initialized from anonymous structs whose fields
have the same types, names, and order, and their names are part of the names of
their fields:

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

The layout of a union is determined as follows: we express the layout of a field
in terms of its starting and ending offsets, which are measured in bits in order
to handle bitfields. The starting and ending offsets of each field in a field
group are the same as if the fields were all direct members of the enclosing
struct. A union member that is not a field group is laid out as if it were a
field group containing a single field. The ending offset of a union is the
maximum ending offset of any field in any of its field groups, rounded up to the
next whole byte.

**TODO:** This presupposes that struct fields are laid out sequentially; if that
is not the case, this algorithm will need to be revised accordingly.

The C/C++ memory model treats any maximal contiguous sequence of bitfields as a
single memory location, but C/C++ do not allow bitfields to be split across the
beginning of a union. Introducing that ability in Carbon implies that the size
of a memory location can change, which the memory model doesn't seem to
countenance. In order to fix that inconsistency, we model the creation or
destruction of an active union member as also destroying any immediately
preceding bitfields, and then recreating them with the same contents. This
implies that you cannot access the preceding bitfields concurrently with
creating or destroying the active union member.

A union cannot be nested directly within a union, and a field group cannot be
nested directly within a struct or a field group. Such constructions would be
logically valid, but are forbidden because they are useless or nearly useless.

Unions and field groups cannot contain constructors, destructors, or methods.

## Safety

The rules described above are designed to permit unambiguous and logically
straightforward dynamic checking: it is always an error to access or destroy a
union member that is not active, and the operations that can change the active
member are always explicit and unambiguous. It should be quite straightforward
for a sanitizer to check direct accesses to union members, by tracking the
active member in shadow memory, and verifying that the member being accessed is
active.

**FIXME:** How do we sanitize accesses via pointers? Naively, it seems like it
would require a pointers to a union member to carry additional metadata to
indicate which member it points to. Pointers to subobjects create further
challenges: that metadata would need to be propagated into subobject pointers,
and nested unions would require multiple layers of metadata.

## C++ interoperability and migration

Carbon unions are more restrictive than C++ unions in several respects:

-   C++ unions can be types, and can have names.
-   C++ permits the active member to be implicitly changed by assigning to an
    inactive member.
-   C++ permits accessing fields of an inactive member if they are part of a
    "common initial sequence" of fields that's shared with the active member.
-   In practice, C++ permits accessing inactive members even in ways that
    violate the "common initial sequence" rule, with the semantics that the
    object representation of the active member is reinterpreted as a
    representation of the inactive member. This is formally undefined behavior,
    but broadly supported and fairly common in practice.

Conversely, C++ is more restrictive in one respect: the members of a C++ union
must be objects, and the union's alignment must conform to the alignment
requirements of its member objects. For example, there doesn't seem to be any
way of expressing this Carbon struct in terms of C++ unions while preserving
both its structure and its layout:

```
struct S {
  union {
    group g1 {
      Int32: a;
      Int16: b;
      Int8: c;
    }
    group g2 {}
  }
  union {
    group g3 {
      Int8: d;
      Int32: e;
    }
    group g4 {}
  }
}
```

If that structure were naively translated to C++, `g3` would be required to to
have 4-byte alignment, which would force the addition of a byte of padding
between the unions. That would in turn force the addition of 3 bytes of padding
between `d` and `e`, in order to properly align `e`.

We can preserve the layout by expanding all groups so that they contain all
prior members of the struct, and hence all start at offset 0:

```c++
struct S {
  union {
    struct /* g3 */ {
      union {
        struct {
          int32_t a;
          int16_t b;
          int8_t c;
        } g1;
        struct {} g2;
      };
      int8_t d;
      int32_t e;
    } g3;
    struct /* g4 */ {
      union {
        struct {
          int32_t a;
          int16_t b;
          int8_t c;
        } g1;
        struct {} g2;
      };
    } g4;
  };
};
```

However, this doesn't preserve the naming structure of the fields: `s.g1.a` in
Carbon would become either `s.g3.g1.a` or `s.g4.g1.a` in C++ (the two are
equivalent because they are part of a common initial sequence), and every
additional union in the struct compounds this problem. Rather than expose this
complexity to users, the members that form the actual data layout will be made
private, and the original members will be exposed via methods that return
references, so that `s.g1.a` becomes `s.g1().a()`.

**TODO:** With this scheme, all data members up to and including the last union
in a struct must be exposed via methods. The structs design will need to
determine whether any subsequent members are exposed in C++ as data members,
methods or both.

Note that this mapping doesn't quite preserve concurrency semantics: Carbon code
can safely access the first union while concurrently changing the active member
of the second union, but in C++ the corresponding operation would be undefined
behavior.

**FIXME:** Does this matter in practice, and can we do anything to avoid the
undefined behavior?

**TODO:** The design of the Carbon memory model will need to address this
inconsistency. While we generally intend to adopt the C/C++ memory model, it's
unclear exactly what that means in cases like this one, where Carbon code
creates situations that are inexpressible in C/C++. Note that it's not entirely
clear whether the undefined behavior in question is a data race per se, because
it's not clear whether invoking a trivial constructor or destructor actually
modifies the memory locations containing the object.

**TODO:** It looks very difficult to support exposing C++ unions to Carbon in an
automated way, unless we are willing to allow type-punning via ordinary pointer
reads, and allow assignment through a pointer to implicitly destroy and create
objects. It may be possible to support partial or full automation in cases where
the union is sufficiently encapsulated, but this will require further research
about what encapsulation patterns are common.
