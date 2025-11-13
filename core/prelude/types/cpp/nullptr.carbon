// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

package Core library "prelude/types/cpp/nullptr";

import library "prelude/copy";
import library "prelude/destroy";
import library "prelude/operators/as";
import library "prelude/types/optional";
import library "prelude/types/maybe_unformed";

namespace CppCompat;

// C++ `std::nullptr_t` as a Carbon type.
//
// The C++ type `decltype(nullptr)`, also known by the library-provided alias
// `std::nullptr_t`, is a fundamental type, not a class type, so it needs a
// custom mapping as part of C++ interoperability.  We map it to this class
// type. After a suitable import, this class can be named as
// `Cpp.std.nullptr_t`. This is also the type of the constant `Cpp.nullptr`.
//
// This type supports implicit conversion to any optional pointer type, and
// produces the `None` value of that type.
class CppCompat.NullptrT {
  // nullptr_t has the same size and alignment as a pointer, but the
  // corresponding pointer is always unformed.
  // TODO: Give this type a custom empty value representation.
  adapt MaybeUnformed(()*);

  fn Make() -> Self {
    returned var s: Self;
    return var;
  }

  impl as Copy {
    fn Op[self: Self]() -> Self {
      return Make();
    }
  }

  // TODO: impl as EqWith(Self)

  impl forall [T:! type] as ImplicitAs(Optional(T*)) {
    fn Convert[self: Self]() -> Optional(T*) {
      return Optional(T*).None();
    }
  }
}
