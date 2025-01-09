// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// https://adventofcode.com/2024/day/8

import Core library "io";

import library "day8_common";
import library "io_utils";

fn IsAntinode(grid: Grid, ox: i32, oy: i32) -> bool {
  var ay: i32 = 0;
  while (ay < 50) {
    let by: i32 = ay * 2 - oy;
    if (by >= 0 and by < 50) {
      var ax: i32 = 0;
      while (ax < 50) {
        let bx: i32 = ax * 2 - ox;
        if (bx >= 0 and bx < 50 and (ax != bx or ay != by)) {
          if (grid.data[ax][ay] != 0x2E and
              grid.data[ax][ay] == grid.data[bx][by]) {
            return true;
          }
        }
        ++ax;
      }
    }
    ++ay;
  }
  return false;
}

fn CountAntinodes(grid: Grid) -> i32 {
  var count: i32 = 0;
  var y: i32 = 0;
  while (y < 50) {
    var x: i32 = 0;
    while (x < 50) {
      if (IsAntinode(grid, x, y)) {
        ++count;
      }
      ++x;
    }
    ++y;
  }
  return count;
}

fn Run() {
  Core.Print(CountAntinodes(Grid.Read()));
}
