// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// https://adventofcode.com/2024/day/8

import Core library "io";

import library "day8_common";
import library "io_utils";

fn MarkAndCount(marks: array(array(bool, 50), 50)*, x: i32, y: i32) -> i32 {
  if (not (*marks)[x][y]) {
    (*marks)[x][y] = true;
    return 1;
  }
  return 0;
}

fn MarkAndCountAntinodesFor(grid: Grid, marks: array(array(bool, 50), 50)*, ax: i32, ay: i32) -> i32 {
  var count: i32 = 0;
  var by: i32 = 0;
  while (by < 50) {
    var bx: i32 = 0;
    while (bx < 50) {
      let dx: i32 = bx - ax;
      let dy: i32 = by - ay;
      if (grid.data[bx][by] != grid.data[ax][ay] or (dx == 0 and dy == 0)) {
        ++bx;
        continue;
      }

      var x: i32 = bx;
      var y: i32 = by;
      while (x >= 0 and x < 50 and y >= 0 and y < 50) {
        count += MarkAndCount(marks, x, y);
        x += dx;
        y += dy;
      }
      ++bx;
    }
    ++by;
  }
  return count;
}

fn MarkAndCountAntinodes(grid: Grid, marks: array(array(bool, 50), 50)*) -> i32 {
  var count: i32 = 0;
  var y: i32 = 0;
  while (y < 50) {
    var x: i32 = 0;
    while (x < 50) {
      if (grid.data[x][y] != 0x2E) {
        count += MarkAndCountAntinodesFor(grid, marks, x, y);
      }
      ++x;
    }
    ++y;
  }
  return count;
}

fn Run() {
  var marks: array(array(bool, 50), 50);
  var y: i32 = 0;
  while (y < 50) {
    var x: i32 = 0;
    while (x < 50) {
      marks[x][y] = false;
      ++x;
    }
    ++y;
  }

  Core.Print(MarkAndCountAntinodes(Grid.Read(), &marks));
}
