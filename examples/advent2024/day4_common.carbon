// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// https://adventofcode.com/2024/day/4

library "day4_common";

import library "io_utils";

class Wordsearch {
  fn Read() -> Wordsearch {
    returned var s: Wordsearch;
    // TODO: Use for loops once they're implemented.
    var y: i32 = 0;
    while (y < 140) {
      var x: i32 = 0;
      while (x < 140) {
        // TODO: Assert on failure.
        s.grid[x][y] = ReadChar();
        ++x;
      }
      // TODO: Assert on failure.
      SkipNewline();
      ++y;
    }
    return var;
  }

  fn At[self: Self](x: i32, y: i32) -> i32 {
    return if x < 0 or x >= 140 or y < 0 or y >= 140 then -1 else self.grid[x][y];
  }

  // TODO: Make this generic in the length of the search query.
  fn Check4[self: Self](xmas: array(i32, 4), x: i32, y: i32, dx: i32, dy: i32) -> bool {
    var i: i32 = 0;
    while (i < 4) {
      if (self.At(x + i * dx, y + i * dy) != xmas[i]) {
        return false;
      }
      ++i;
    }
    return true;
  }

  fn Check3[self: Self](mas: array(i32, 3), x: i32, y: i32, dx: i32, dy: i32) -> bool {
    var i: i32 = 0;
    while (i < 3) {
      if (self.At(x + i * dx, y + i * dy) != mas[i]) {
        return false;
      }
      ++i;
    }
    return true;
  }

  var grid: array(array(i32, 140), 140);
}
