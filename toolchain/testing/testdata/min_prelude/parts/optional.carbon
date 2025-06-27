// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/parts/bool.carbon

// --- min_prelude/parts/optional.carbon

package Core library "prelude/parts/optional";

import library "prelude/parts/bool";

class Optional(T:! type) {
  fn None() -> Self {
    returned var me: Self;
    me.has_value = false;
    return var;
  }

  fn Some(value: T) -> Self {
    return {.has_value = true, .value = value};
  }

  fn HasValue[self: Self]() -> bool { return self.has_value; }
  fn Get[self: Self]() -> T { return self.value; }

  var has_value: bool;
  var value: T;
}
