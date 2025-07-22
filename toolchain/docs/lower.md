# Lower

<!--
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

<!-- toc -->

## Table of contents

-   [Overview](#overview)
-   [Generic lowering](#generic-lowering)
-   [Cross-file lowering](#cross-file-lowering)
-   [Specific deduplication and fingerprinting](#specific-deduplication-and-fingerprinting)
-   [Mangling](#mangling)
    -   [Examples](#examples)

<!-- tocstop -->

## Overview

Lowering takes the SemIR and produces LLVM IR. At present, this is done in a
single pass, although it's possible we may need to do a second pass so that we
can first generate type information for function arguments.

The lowering context is split into three layers:

-   The `Context` object holds state for an overall lowering process that
    produces a single LLVM module.
-   The `FileContext` object holds state for lowering from a particular
    `SemIR::File`, and holds a pointer to its enclosing `Context`. Multiple
    files may be involved in a single lowering process when lowering a generic,
    where the definition of the generic and the specific may be owned by
    distinct files. This setup would also allow us to lower an entire library
    into a single LLVM module if we chose to do so.
-   The `FunctionContext` object holds state for lowering a particular function,
    including an `IRBuilder` and mappings from the local `InstId`s to their
    lowered `llvm::Value*`s and from the local `InstBlockId`s to their lowered
    `llvm::BasicBlock*`s.

Lowering is done per `SemIR::InstBlock`. This minimizes changes to the
`IRBuilder` insertion point, something that is both expensive and potentially
fragile.

## Generic lowering

In order to support lowering generic functions, the `FunctionContext` tracks
both the `FunctionId` of the function being lowered and a corresponding
`SpecificId`. Whenever `FunctionContext` or a `HandleInst` function inspects a
property of an instruction that can vary between specifics -- in particular, the
type or constant value of an instruction -- that value is looked up in the
current specific, and the corresponding type or value is used instead.

`FunctionContext::GetTypeOfInst` and `FunctionContext::GetTypeIdOfInst` do this
mapping for the type of an instruction, and should be used instead of directly
looking at the `type_id` field of a typed instruction throughout function
lowering. Similarly, `FunctionContext::GetValue` does this mapping when looking
up the constant value of an instruction.

## Cross-file lowering

`FunctionContext` lowering may draw information used to lower the function from
two different files:

-   The file in which the function was defined.
-   For a generic function, the file in which the specific was formed.

Each of these files has its own `FileContext`, which tracks its corresponding
`SemIR::File`, as well as mappings from its constant values to
`llvm::Constant*`s and mappings from its functions to `llvm::Function*`s, and so
on.

When querying the type of an instruction using
`FunctionContext::GetTypeIdOfInst`, the resulting type may be owned by either of
these files. The type is represented as a `TypeInFile`, which is a pair of the
owning `SemIR::File*` and the `SemIR::TypeId` within that file. Care must be
taken to only pass the `TypeId` in a `TypeInFile` to code that expects a
`TypeId` within the corresponding `SemIR::File*`. To reduce the risk of errors,
code within `FunctionContext` and `HandleInst` functions should not directly
interact with `TypeId`s, and should instead always use `TypeInFile`.

Similarly, other type properties have `FunctionContext` wrappers that track the
file that owns the `TypeId`s:

-   `FunctionContext::GetValueRepr` returns a `ValueReprInFile` which is a pair
    of a `SemIR::File*` and a `SemIR::ValueRepr`.
-   `FunctionContext::GetReturnTypeInfo` returns a `ReturnTypeInfoInFile` which
    is a pair of a `SemIR::File*` and a `SemIR::ReturnTypeInfo`.

These pairs are kept wrapped in the `*InFile` structs wherever possible, in
order to minimize the chance of an ID being used with the wrong file.

## Specific deduplication and fingerprinting

Specifics for the same generic are deduplicated by detecting whether we
generated the same LLVM IR for all the portions of the specific that depend on
generic arguments. This is accomplished in part by computing a fingerprint for
each specific. The fingerprint contains:

-   For each symbolic constant value used while lowering, the lowered LLVM value
    in the specific.
-   For each symbolic type used while lowering, the lowered LLVM type in the
    specific.
-   For each called function, information about the specific callee. TODO:
    Describe how we handle deduplicating strongly-connected components of the
    call graph.
-   For each other property of the specific that lowering depends on, the value
    of that property.

These fingerprinted values are tracked by the `FunctionContext` accessors that
obtain the information from SemIR:

-   `FunctionContext::GetType` adds the `llvm::Type*` produced for a symbolic
    type to the fingerprint.
-   `FunctionContext::GetValue` adds the `llvm::Value*` produced for a symbolic
    constant to the fingerprint.
-   `FunctionContext::GetValueRepr` adds the kind of the value representation,
    but not the value representation type, to the fingerprint.
-   `FunctionContext::GetInitRepr` adds the kind of the initializing
    representation to the fingerprint.
-   `FunctionContext::GetReturnTypeInfo` adds the kind of the return
    representation, but not the type, to the fingerprint.

For `GetValueRepr` and `GetReturnTypeInfo`, the corresponding type is
represented as a `TypeInFile`. The convention in use is that `TypeInFile` values
represent types that have not yet been added to the fingerprint for the
specific, and the mapping from `TypeInFile` to `llvm::Type*` is the point where
the type is added to the fingerprint, but other data such as the enumeration
values stored on `ReturnTypeInfoInFile` have already been added to the
fingerprint.

Additional information queried from SemIR by `FunctionContext` or a `HandleInst`
function should follow the same pattern, adding a getter on `FunctionContext`
that adds the information to the fingerprint, and returns a `*InFile` wrapper
struct if the result contains any `TypeId`s.

## Mangling

Part of lowering is choosing deterministically unique identifiers for each
lowered entity to use in platform object files. Any feature of an entity (such
as parent namespaces or overloaded function parameters) that would create a
distinct entity must be included in some way in the generated identifier.

The current rudimentary name mangling scheme is as follows:

-   As a special case, `Main.Run` is emitted as `main`.

Otherwise the resulting name consists of:

1.  `_C`
2.  The unqualified function name (function name mangling is the only thing
    implemented at the moment).
3.  If the function is a thunk, `:thunk` to distinguish it from the function it
    invokes.
4.  `.`
5.  If the function being mangled is a member of:
    -   an `impl`, then add:
        1.  The implementing type, per the scope mangling.
        2.  `:`
        3.  The interface type, per the scope mangling.
    -   a type or namespace, then add:
        1.  The scope, per the scope mangling.

The scope mangling scheme is as follows:

1.  The unqualified name of the type or namespace.
2.  If the type or namespace is within another type or namespace:
    1.  `.`
    2.  The enclosing scope, per the scope mangling.
3.  `.`
4.  The package name.

### Examples

```carbon
package P1;
interface Interface {
  fn Op[self: Self]();
}
```

```carbon
namespace NameSpace;
class NameSpace.Implementation {
  // Mangled as:
  // `_COp.Implementation.NameSpace.Main:Interface.P1`
  impl as P1.Interface {
    fn Op[self: Self]() {
    }
  }
}
// Mangled as `main`.
fn Run() {
  var v: NameSpace.Implementation;
  v.(P1.Interface.Op)();
}
```
