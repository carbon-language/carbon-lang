// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// https://adventofcode.com/2024/day/4

import Core library "io";
import Core library "range";

import library "day4_common";
import library "io_utils";

fn Run() {
  var search: Wordsearch = Wordsearch.Read();
  var xmas: array(i32, 4) = (0x58, 0x4D, 0x41, 0x53);
  var found: i32 = 0;

  for (y: i32 in Core.Range(140)) {
    for (x: i32 in Core.Range(140)) {
      for (dy: i32 in Core.InclusiveRange(-1, 1)) {
        for (dx: i32 in Core.InclusiveRange(-1, 1)) {
          if (search.Check4(xmas, x, y, dx, dy)) {
            ++found;
          }
        }
      }
    }
  }
  Core.Print(found);
}
