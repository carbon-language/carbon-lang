// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// https://adventofcode.com/2024/day/4

import Core library "io";

import library "day4_common";
import library "io_utils";

fn Run() {
  var search: Wordsearch = Wordsearch.Read();
  var mas: array(i32, 3) = (0x4D, 0x41, 0x53);
  var found: i32 = 0;

  // TODO: Use for loops once they're implemented.
  var y: i32 = 1;
  while (y < 139) {
    var x: i32 = 1;
    while (x < 139) {
      if ((search.Check3(mas, x - 1, y - 1, 1, 1) or
           search.Check3(mas, x + 1, y + 1, -1, -1)) and
          (search.Check3(mas, x - 1, y + 1, 1, -1) or
           search.Check3(mas, x + 1, y - 1, -1, 1))) {
        ++found;
      }
      ++x;
    }
    ++y;
  }
  Core.Print(found);
}
