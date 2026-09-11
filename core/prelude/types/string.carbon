// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

package Core library "prelude/types/string";

import library "prelude/copy";
import library "prelude/default";
import library "prelude/destroy";
import library "prelude/iterate";
import library "prelude/range";
import library "prelude/types/char";
import library "prelude/types/uint";
import library "prelude/operators/index";
import library "prelude/types/int";
import library "prelude/operators/as";
import library "prelude/operators/comparison";
import library "prelude/operators/arithmetic";

class String {
  fn Size(self) -> i64 { return self.size; }

  impl as Copy {
    fn Op(self) -> Self { return {.ptr = self.ptr, .size = self.size}; }
  }
  // TODO: This should be an array iterator.
  private var ptr: Char*;
  // TODO: This should be a word-sized integer.
  private var size: i64;
}

impl String as UnformedInit {}

impl forall [T: ImplicitAs(i64)] String as IndexWith(T) where .ElementType = Char {
  fn At(self, subscript: T) -> Char = "string.at";
}

impl String as EqWith(String) {
  fn Equal(self: String, other: String) -> bool {
    if (self.Size() != other.Size()) {
      return false;
    }
    for (i: i64 in IntRange(64).Make(0, self.Size())) {
      if (self[i] != other[i]) {
        return false;
      }
    }
    return true;
  }
  fn NotEqual(self: String, other: String) -> bool {
    return not Equal(self, other);
  }
}
