// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

package Core library "prelude/types/cpp/nullptr";

import library "prelude/copy";
import library "prelude/operators/as";
import library "prelude/types/optional";

private fn MakeMaybeUnformed(t: type) -> type = "maybe_unformed.make_type";

class CppNullptrT {
  // nullptr_t has the same size and alignment as a pointer, but the
  // corresponding pointer is always unformed.
  // TODO: Give this type a custom empty value representation.
  adapt MakeMaybeUnformed(()*);

  fn Make() -> Self {
    returned var s: Self;
    return var;
  }

  impl as Copy {
    fn Op[self: Self]() -> Self {
      return Make();
    }
  }

  impl forall [T:! type] as ImplicitAs(Optional(T*)) {
    fn Convert[self: Self]() -> Optional(T*) {
      return Optional(T*).None();
    }
  }
}
