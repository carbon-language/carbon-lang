// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

package Core library "prelude/types/optional";

import library "prelude/copy";
import library "prelude/default";
import library "prelude/destroy";
import library "prelude/operators/as";
import library "prelude/operators/bitwise";
import library "prelude/types/bool";
import library "prelude/types/maybe_unformed";

// TODO: Decide how to expose this in the public API.
private interface OptionalStorage {
  let Type:! type;
  fn None() -> Type;
  fn Some(self) -> Type;
  fn Has(value: Type) -> bool;
  fn Get(value: Type) -> Self;
  // Copying can also be performed by Get(value).Some(), but Get returns by
  // copy, so that performs two copies.
  fn Copy(value: Type) -> Type;
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
  fn HasValue(self) -> bool {
    return T.Has(self as T.Type);
  }
  fn Get(self) -> T {
    return T.Get(self as T.Type);
  }

  // TODO: Use `private adapt` or `unsafe adapt` once available.
  adapt T.Type;
}

impl forall [T:! OptionalStorage & UnformedInit] Optional(T) as UnformedInit {}

impl forall [T:! OptionalStorage] Optional(T) as Copy {
  fn Op(self) -> Self {
    return T.Copy(self as T.Type) as Optional(T);
  }
}

// Support for converting a `T` to an `Optional(U)` if `T` converts to `U`.
// Once we have match_first, this can be rewritten more simply as:
//
// match_first {
//   impl forall [T:! OptionalStorage]
//       T as ImplicitAs(Optional(T)) {
//     fn Convert(self: T) -> Optional(T) {
//       return Optional(T).Some(self);
//     }
//   }
//
//   impl forall [T:! Destroy & OptionalStorage, U:! ImplicitAs(T)]
//       U as ImplicitAs(Optional(T)) {
//     fn Convert(self: U) -> Optional(T) {
//       return Optional(T).Some(self.Convert());
//     }
//   }
// }

private interface OptionalAs(T:! OptionalStorage) {
  fn Convert(self) -> Optional(T);
}

final impl forall [T:! OptionalStorage]
    T as OptionalAs(T) {
  fn Convert(self) -> Optional(T) {
    return Optional(T).Some(self);
  }
}

impl forall [T:! Destroy & OptionalStorage, U:! ImplicitAs(T)]
    U as OptionalAs(T) {
  fn Convert(self) -> Optional(T) {
    return Optional(T).Some(self.Convert());
  }
}

impl forall [T:! Destroy & OptionalStorage,
             U:! Destroy & OptionalStorage & ImplicitAs(T)]
    Optional(U) as OptionalAs(T) {
  fn Convert(self) -> Optional(T) {
    if (self.HasValue()) {
      return Optional(T).Some(self.Get().Convert());
    }
    return Optional(T).None();
  }
}

impl forall [T:! OptionalStorage, U:! OptionalAs(T)]
    U as ImplicitAs(Optional(T)) {
  fn Convert(self) -> Optional(T) {
    return self.Convert();
  }
}

// By default, an `Optional(T)` is stored as a pair of a `bool` and a
// `MaybeUnformed(T)`, with the `MaybeUnformed(T)` left uninitialized if the
// `bool` is `false`.
//
// TODO: Revisit this once we have choice types implemented in the toolchain.
private class DefaultOptionalStorage(T:! Copy & Destroy) {
  var value: MaybeUnformed(T);
  var has_value: bool;
}

private fn MakeUninitializedOptionalStorage(T:! Copy & Destroy)
    -> DefaultOptionalStorage(T) = "make_uninitialized";

impl forall [T:! Copy & Destroy] T as OptionalStorage
    where .Type = DefaultOptionalStorage(T) {
  fn None() -> DefaultOptionalStorage(T) {
    returned var me: DefaultOptionalStorage(T) =
        MakeUninitializedOptionalStorage(T);
    me.has_value = false;
    return var;
  }
  fn Some(self) -> DefaultOptionalStorage(T) {
    // TODO: This whole function should be just
    //   return {.value = self, .has_value = true};
    // but that requires that `T` implements `ImplicitAs(MaybeUnformed(T))`.
    returned var me: DefaultOptionalStorage(T) =
        MakeUninitializedOptionalStorage(T);
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
  fn Copy(value: DefaultOptionalStorage(T)) -> DefaultOptionalStorage(T) {
    if (Has(value)) {
      return (value.value unsafe as T).(Some)();
    } else {
      return None();
    }
  }
}

private fn PointerIsNull[T:! type](value: MaybeUnformed(T*)) -> bool = "pointer.is_null";

private fn MakeUninitializedOptionalPointer(T:! type)
    -> MaybeUnformed(T*) = "make_uninitialized";

// For pointers, we use a null pointer value as the "None" value. This allows
// `Optional(T*)` to be ABI-compatible with a C++ nullable pointer.
final impl forall [T:! type] T* as OptionalStorage
    where .Type = MaybeUnformed(T*) {
  fn None() -> MaybeUnformed(T*) = "pointer.make_null";
  fn Some(self) -> MaybeUnformed(T*) {
    returned var result: MaybeUnformed(T*) =
        MakeUninitializedOptionalPointer(T);
    result unsafe as T* = self;
    return var;
  }
  fn Has(value: MaybeUnformed(T*)) -> bool {
    return not PointerIsNull(value);
  }
  fn Get(value: MaybeUnformed(T*)) -> T* {
    return value unsafe as T*;
  }
  fn Copy(value: MaybeUnformed(T*)) -> MaybeUnformed(T*) = "primitive_copy";
}
