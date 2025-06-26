// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// TODO: This library is not part of the design. Either write a matching
// proposal or remove this.

package Core library "range";

import library "prelude/iterate";
import library "prelude/operators/arithmetic";
import library "prelude/operators/as";
import library "prelude/operators/comparison";
import library "prelude/types/int";
import library "prelude/types/int_literal";
import library "prelude/types/optional";

class IntRange(N:! IntLiteral()) {
  fn Make(start: Int(N), end: Int(N)) -> Self {
    return {.start = start, .end = end};
  }

  impl as Iterate where .CursorType = Int(N) and .ElementType = Int(N) {
    fn NewCursor[self: Self]() -> Int(N) { return self.start; }
    fn Next[self: Self](cursor: Int(N)*) -> Optional(Int(N)) {
      var value: Int(N) = *cursor;
      if (value < self.end) {
        ++*cursor;
        return Optional(Int(N)).Some(value);
      } else {
        return Optional(Int(N)).None();
      }
    }
  }

  private var start: Int(N);
  private var end: Int(N);
}

fn Range(end: i32) -> IntRange(32) {
  return IntRange(32).Make(0, end);
}

fn InclusiveRange(start: i32, end: i32) -> IntRange(32) {
  return IntRange(32).Make(start, end + 1);
}
