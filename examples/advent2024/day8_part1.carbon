// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// https://adventofcode.com/2024/day/8

import Core library "io";
import Core library "range";

import library "day8_common";
import library "io_utils";

fn IsAntinode(grid: Grid, ox: i32, oy: i32) -> bool {
  for (ay: i32 in Core.Range(50)) {
    let by: i32 = ay * 2 - oy;
    if (by >= 0 and by < 50) {
      for (ax: i32 in Core.Range(50)) {
        let bx: i32 = ax * 2 - ox;
        if (bx >= 0 and bx < 50 and (ax != bx or ay != by)) {
          if (grid.data[ax][ay] != 0x2E and
              grid.data[ax][ay] == grid.data[bx][by]) {
            return true;
          }
        }
      }
    }
  }
  return false;
}

fn CountAntinodes(grid: Grid) -> i32 {
  var count: i32 = 0;
  for (y: i32 in Core.Range(50)) {
    for (x: i32 in Core.Range(50)) {
      if (IsAntinode(grid, x, y)) {
        ++count;
      }
    }
  }
  return count;
}

fn Run() {
  Core.Print(CountAntinodes(Grid.Read()));
}
