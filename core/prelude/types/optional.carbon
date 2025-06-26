// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

package Core library "prelude/types/optional";

import library "prelude/types/bool";

// For now, an `Optional(T)` is stored as a pair of a `bool` and a `T`, with
// the `T` left uninitialized if the `bool` is `false`. This isn't a viable
// approach in the longer term, but is the best we can do for now.
//
// TODO: Revisit this once we have choice types implemented in the toolchain.
//
// TODO: We don't have an approved design for an `Optional` type yet, but it's
// used by the design for `Iterate`. The API here is a placeholder.
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

  private var has_value: bool;
  private var value: T;
}
