// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// https://adventofcode.com/2024/day/8

library "day8_common";

import Core library "io";
import library "io_utils";

class Grid {
  fn Read() -> Grid {
    returned var me: Grid;
    var y: i32 = 0;
    while (y < 50) {
      var x: i32 = 0;
      while (x < 50) {
        me.data[x][y] = ReadChar();
        ++x;
      }
      SkipNewline();
      ++y;
    }
    return var;
  }

  var data: array(array(i32, 50), 50);
}
