# Unions

<!--
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

## Table of contents

<!-- toc -->

-   [Overview](#overview)
-   [Union members](#union-members)
-   [Changing the live member](#changing-the-live-member)
-   [Field groups](#field-groups)
-   [Layout](#layout)
-   [Safety](#safety)
-   [C++ interoperability and migration](#c-interoperability-and-migration)

<!-- tocstop -->

## Overview

Fields of a struct can be grouped into _unions_. For example:

```
struct Number {
  union {
    var Int64: int_value;
    var Float64: float_value;
  }

  // 0 if no live member, 1 if int_value is live, 2 if double_value is live.
  var Int2: discriminator;
}
```

A union consists of zero or more _members_, at most one of which can be live at
a time. A member can be either a field of any type, or a
[field group](#field-groups). All members of a union share the same storage, so
`Number` as a whole is 9 bytes, not 17. It is out of contract to access a member
that is not live, even if it has the same type as the live member. There is no
intrinsic way to determine which member of the union is live, and in fact that
information will probably not be tracked in production builds. Instead, user
code is responsible for doing whatever bookkeeping is necessary to ensure that
only the live member is accessed. This typically takes the form of an associated
discriminator field, as in `Number`, but this is not required. For example, some
user code may be able to satisfy that requirement statically, with no run-time
tracking at all.

> **TODO:** If Carbon supports auto-generating struct operations such as
> copying, assignment, or destruction, we will need to specify that it does not
> support structs that contain unions, since the generated code won't know which
> field is live.

> **TODO:** Carbon should also permit users to easily define "algebraic data
> types", which are type-safe union-like constructs with compiler-provided
> discriminators. It will also need a separate facility for reinterpreting the
> representation of one type to the representation of a different type (also
> known as "type punning"). When those are designed, cite them above.

## Union members

Unlike C++ unions, Carbon unions never have names, are not objects, and do not
have types; they are a means of controlling the layout of fields in a struct.
Consequently, it is not possible to form a pointer to a union. You can form a
pointer to a member field of a union, but you cannot cast it to a pointer to the
type of a different member.

Union fields are referenced and initialized exactly like ordinary members of the
enclosing struct. However, union fields can be omitted from the initializer,
unlike ordinary fields. Obviously the initializer cannot initialize more than
one member of a union, but it can initialize either one member or none. If no
member of the union is initialized, no member of the union is initially live:

```
// Neither n1.int_value nor n1.float_value are live
Number: n1 = ( .discriminator = 0 );

// n2.int_value is live and holds 0
Number: n2 = ( .int_value = 0, .discriminator = 1 );

// n3.int_value is live, but not initialized
Number: n3 = ( .int_value = uninit, .discriminator = 1 );

// Error: cannot initialize multiple members of a single union
Number: n4 = ( .int_value = 0, .float_value = uninit, .discriminator = 1 );
```

Unions follow similar rules in pattern matching: a pattern can mention either
zero or one member of a given union. If no member is mentioned, the union is not
accessed during pattern matching, and has no effect on whether the pattern
matches. If the pattern mentions a union member, the corresponding subpattern is
matched against that member, which means that user code must ensure that pattern
matching does not reach that point unless that member is live. Note that this
applies even if the subpattern is a wildcard that is guaranteed to match.

> **TODO:** Ensure the above remains consistent with the overall design for
> struct initialization and pattern matching, and ensure pattern matching gives
> enough control over evaluation order to make it possible to safely mention
> union members in patterns.

## Changing the live member

The live member can only be changed by destroying the current live member (if
any), and then constructing the new live member, using the `destroy` and
`create` keywords:

```
fn SetFloatValue(Ptr(Number): n, Float64: value) {
  if (n->discriminator == 0) {
    destroy n->int_value;
    create: n->float_value = value;
    n->discriminator = 1;
  } else {
    n->float_value = value;
  }
}
```

`create` and `destroy` can only be applied to union members (unlike their C++
counterparts, placement `new` and pseudo-destructor calls); the lifetimes of
ordinary struct fields and variables are always tied to their scope. It is out
of contract to apply `create` to a member of a union that already has a live
member, or apply `destroy` to a member that is not live. `destroy` can be
thought of as a unary operator, but a `create` statement has the syntax and
semantics of a variable declaration, with `create` taking the place of `var` and
the field type, and the field expression taking the place of the variable name.
`destroy` permanently invalidates any pointers to the destroyed object; they do
not become valid again if that member is re-created.

> **FIXME**: Does the `create` syntax make it sufficiently clear that the `=`
> represents initialization, not assignment? Is it OK that the `create` syntax
> omits the type? Can we do better?

> **TODO:** The spelling of `create` and `destroy` are chosen for consistency
> with `operator create` and `operator destroy`, which are how constructor and
> destructor declarations are spelled in the currently-pending structs proposal.
> They should be updated as necessary to stay consistent.

## Field groups

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

> **TODO:** The treatment of bitfields, arrays, and owning pointers in this
> example is speculative, and should be updated to reflect the eventual design
> of those features.

Field groups are sets of fields (and/or unions) that can be created and
destroyed as a unit. They are initialized from anonymous structs whose fields
have the same types, names, and order. The name of a field group is part of the
name of each field in the group:

```
var SsoString: str = (.is_small = True,
                      .small = (.size = 0, .buffer = uninitialized)
                     );
Assert(str.small.size == 0);
destroy str.small;
create: str.large = (.size = 0, .capacity = 100, .buffer = MakeBuffer(100));
Assert(str.large.capacity == 100);
```

However, field groups are not objects, and do not have types; like unions, field
groups are a way of controlling the layout of the fields of a struct. For
example, if `large` and `small` were structs, the bitfields would not save any
space, and `is_small` would have to be followed by 63 bits of padding in order
to ensure proper alignment of `large`.

## Layout

We express the layout of a field in terms of its starting and ending offsets,
which are measured in bits in order to handle bitfields. The starting and ending
offsets of each field in a field group are the same as if the fields were all
direct members of the enclosing struct. A union member that is not a field group
is laid out as if it were a field group containing a single field. The ending
offset of a union is the maximum ending offset of any field in any of its field
groups, rounded up to the next whole byte. Note that this means that bitfields
after the union are not "packed" together with any bitfields at the end of the
union.

> **TODO:** This presupposes that struct fields are laid out sequentially; if
> that is not the case, this algorithm will need to be revised accordingly.

The C/C++ memory model (which we expect Carbon to adopt) treats any maximal
contiguous sequence of bitfields as a single memory location, but C/C++ do not
allow bitfields to be split across the beginning of a union. Introducing that
ability in Carbon implies that the size of a memory location can change, which
the memory model doesn't seem to countenance. In order to fix that
inconsistency, we model the creation or destruction of union member as also
destroying any immediately preceding bitfields, and then recreating them with
the same contents. This implies that you cannot access the preceding bitfields
concurrently with creating or destroying a union member.

A union cannot be nested directly within a union, and a field group cannot be
nested directly within a struct or a field group. Unions and field groups cannot
contain constructors, destructors, or methods.

## Safety

The safety rules for Carbon unions are easily summarized: it is always out of
contract to access or destroy a union member that is not live, and the
operations that can change the live member are always explicit and unambiguous.
It should be quite straightforward for a sanitizer to check direct accesses to
union members, by tracking the live member in shadow memory, and verifying that
the member being accessed is live.

However, union members can also be accessed through pointers (including pointers
to nested subobjects), and such pointers are indistinguishable from pointers to
any other object. Reliably sanitizing accesses through such pointers would
require dynamically tracking which union member (if any) each pointer points to,
propagating that information to subobject accesses, and instrumenting every
pointer access in the program to determine whether it is accessing a union, and
if so whether the currently live member matches the one the pointer value was
created with. This would definitely be too costly for a hardened production
build mode, and might even be too costly (relative to its benefit) to be useful
as a sanitizer.

> **TODO:** Figure out whether and how to sanitize invalid union accesses,
> probably in the context of an overall design for temporal memory safety.

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
private, and the original members will be exposed through methods that return
references, so that `s.g1.a` becomes `s.g1().a()`.

> **TODO:** With this scheme, all data members up to and including the last
> union in a struct must be exposed through methods. The structs design will
> need to determine whether any subsequent members are exposed in C++ as data
> members, methods or both.

Note that this mapping doesn't quite preserve concurrency semantics: Carbon code
can safely access the first union while concurrently creating or destroying a
member of the second union, but in C++ the corresponding operation would be
undefined behavior.

> **FIXME:** Does this matter in practice, and can we do anything to avoid the
> undefined behavior?

> **TODO:** The design of the Carbon memory model will need to address this
> inconsistency. While we generally intend to adopt the C/C++ memory model, it's
> unclear exactly what that means in cases like this one, where Carbon code
> creates situations that are inexpressible in C/C++. Note that it's not
> entirely clear whether the undefined behavior in question is a data race per
> se, because it's not clear whether invoking a trivial constructor or
> destructor actually modifies the memory locations containing the object.

> **TODO:** It looks very difficult to support exposing C++ unions to Carbon in
> an automated way, unless we are willing to allow type-punning through ordinary
> pointer reads, and allow assignment through a pointer to implicitly destroy
> and create objects. It may be possible to support partial or full automation
> in cases where the union is sufficiently encapsulated, but this will require
> further research about what encapsulation patterns are common.
