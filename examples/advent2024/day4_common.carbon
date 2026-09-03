// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// https://adventofcode.com/2024/day/4

library "day4_common";

import library "io_utils";

class Wordsearch {
  impl as Core.UnformedInit {}

  fn Read() -> Wordsearch {
    returned var s: Wordsearch;
    for (y: i32 in Core.Range(140)) {
      for (x: i32 in Core.Range(140)) {
        // TODO: Assert on failure.
        s.grid[x][y] = ReadChar() as i32;
      }
      // TODO: Assert on failure.
      SkipNewline();
    }
    return var;
  }

  fn At(self, x: i32, y: i32) -> i32 {
    return if x < 0 or x >= 140 or y < 0 or y >= 140 then -1 else self.grid[x][y];
  }

  // TODO: Deduce N.
  fn Check(self, generic N: i32, xmas: array(i32, N), x: i32, y: i32, dx: i32, dy: i32) -> bool {
    for (i: i32 in Core.Range(N)) {
      if (self.At(x + i * dx, y + i * dy) != xmas[i]) {
        return false;
      }
    }
    return true;
  }

  var grid: array(array(i32, 140), 140);
}
