// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

package Core library "prelude/types/cpp/nullptr";

import library "prelude/copy";
import library "prelude/default";
import library "prelude/destroy";
import library "prelude/operators/as";
import library "prelude/types/cpp/void";
import library "prelude/types/maybe_unformed";
import library "prelude/types/optional";

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
  adapt MaybeUnformed(VoidBase*);

  // TODO: This should be just
  //   fn Make() -> Self = "make_uninitialized";
  // but we don't yet delay processing builtin function definitions until the
  // end of the enclosing class.
  fn Make() -> Self {
    fn MakeImpl() -> Self = "make_uninitialized";
    return MakeImpl();
  }

  impl as Copy {
    fn Op(unused self) -> Self {
      return Make();
    }
  }

  // TODO: impl as EqWith(Self)

  impl forall [T:! type] as ImplicitAs(Optional(T*)) {
    fn Convert(unused self) -> Optional(T*) {
      return Optional(T*).None();
    }
  }
}

impl CppCompat.NullptrT as UnformedInit {}
