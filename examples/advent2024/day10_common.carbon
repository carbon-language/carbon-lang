// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// https://adventofcode.com/2024/day/10

library "day10_common";

import Core library "io";
import Core library "range";
import library "io_utils";

class Terrain {
  fn Read() -> Terrain {
    returned var me: Terrain;
    for (y: i32 in Core.Range(43)) {
      for (x: i32 in Core.Range(43)) {
        me.height[x][y] = ReadChar() - 0x30;
      }
      SkipNewline();
    }
    return var;
  }

  var height: array(array(i32, 43), 43);
}
