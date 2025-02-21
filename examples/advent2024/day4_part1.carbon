// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// https://adventofcode.com/2024/day/4

import Core library "io";

import library "day4_common";
import library "io_utils";

fn Run() {
  var search: Wordsearch = Wordsearch.Read();
  var xmas: array(i32, 4) = (0x58, 0x4D, 0x41, 0x53);
  var found: i32 = 0;

  // TODO: Use for loops once they're implemented.
  var y: i32 = 0;
  while (y < 140) {
    var x: i32 = 0;
    while (x < 140) {
      var dy: i32 = -1;
      while (dy <= 1) {
        var dx: i32 = -1;
        while (dx <= 1) {
          if (search.Check4(xmas, x, y, dx, dy)) {
            ++found;
          }
          ++dx;
        }
        ++dy;
      }
      ++x;
    }
    ++y;
  }
  Core.Print(found);
}
