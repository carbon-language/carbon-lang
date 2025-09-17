// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

package Core library "prelude/types/string";

import library "prelude/copy";
import library "prelude/destroy";
import library "prelude/types/char";
import library "prelude/types/uint";

class String {
  fn Size[self: Self]() -> u64 { return self.size; }

  impl as Copy {
    fn Op[self: Self]() -> Self { return {.ptr = self.ptr, .size = self.size}; }
  }

  // TODO: This should be an array iterator.
  private var ptr: Char*;
  // TODO: This should be a word-sized integer.
  private var size: u64;
}
