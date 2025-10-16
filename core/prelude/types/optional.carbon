// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

package Core library "prelude/types/optional";

import library "prelude/copy";
import library "prelude/destroy";
import library "prelude/operators/as";
import library "prelude/types/bool";
import library "prelude/types/maybe_unformed";

// TODO: Decide how to expose this in the public API.
private interface OptionalStorage {
  let Type:! type;
  fn None() -> Type;
  fn Some[self: Self]() -> Type;
  fn Has(value: Type) -> bool;
  fn Get(value: Type) -> Self;
}

// TODO: We don't have an approved design for an `Optional` type yet, but it's
// used by the design for `Iterate`. The API here is a placeholder.
class Optional(T:! OptionalStorage) {
  fn None() -> Self {
    return T.None() as Self;
  }
  fn Some(value: T) -> Self {
    return value.Some() as Self;
  }
  fn HasValue[self: Self]() -> bool {
    return T.Has(self as T.Type);
  }
  fn Get[self: Self]() -> T {
    return T.Get(self as T.Type);
  }

  impl T as ImplicitAs(Optional(T)) {
    fn Convert[self: T]() -> Optional(T) {
      return Some(self);
    }
  }

  // TODO: Use `private adapt` or `unsafe adapt` once available.
  adapt T.Type;
}

// By default, an `Optional(T)` is stored as a pair of a `bool` and a
// `MaybeUnformed(T)`, with the `MaybeUnformed(T)` left uninitialized if the
// `bool` is `false`.
//
// TODO: Revisit this once we have choice types implemented in the toolchain.
private class DefaultOptionalStorage(T:! Copy) {
  var value: MaybeUnformed(T);
  var has_value: bool;
}

impl forall [T:! Copy] T as OptionalStorage
    where .Type = DefaultOptionalStorage(T) {
  fn None() -> DefaultOptionalStorage(T) {
    returned var me: DefaultOptionalStorage(T);
    me.has_value = false;
    return var;
  }
  fn Some[self: Self]() -> DefaultOptionalStorage(T) {
    returned var me: DefaultOptionalStorage(T);
    // TODO: Should be:
    //   me.value = self as MaybeUnformed(T);
    me.value unsafe as T = self;
    me.has_value = true;
    return var;
  }
  fn Has(value: DefaultOptionalStorage(T)) -> bool {
    return value.has_value;
  }
  fn Get(value: DefaultOptionalStorage(T)) -> T {
    return value.value unsafe as T;
  }
}

// For pointers, we use a null pointer value as the "None" value. This allows
// `Optional(T*)` to be ABI-compatible with a C++ nullable pointer.
final impl forall [T:! type] T* as OptionalStorage
    where .Type = MaybeUnformed(T*) {
  fn None() -> MaybeUnformed(T*) = "pointer.make_null";
  fn Some[self: Self]() -> MaybeUnformed(T*) {
    returned var result: MaybeUnformed(T*);
    result unsafe as T* = self;
    return var;
  }
  fn Has(value: MaybeUnformed(T*)) -> bool = "pointer.is_null";
  fn Get(value: MaybeUnformed(T*)) -> T* {
    return value unsafe as T*;
  }
}
