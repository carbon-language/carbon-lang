// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// https://adventofcode.com/2024/day/15

library "day15_common";

import Core library "io";
import library "io_utils";

fn Wall() -> i32 { return 0x23; }
fn Box() -> i32 { return 0x4F; }
fn Empty() -> i32 { return 0x2E; }
fn Robot() -> i32 { return 0x40; }

class Grid {
  fn Read() -> Grid {
    returned var me: Grid;
    var y: i32 = 0;
    while (y < 50) {
      var x: i32 = 0;
      while (x < 50) {
        me.data[x][y] = ReadChar();
        if (me.data[x][y] == Robot()) {
          me.robot = (x, y);
        }
        ++x;
      }
      SkipNewline();
      ++y;
    }
    return var;
  }

  fn Move[addr self: Self*](d: (i32, i32)) {
    var distance: i32 = 1;
    while (self->data[self->robot.0 + distance * d.0][self->robot.1 + distance * d.1] == Box()) {
      ++distance;
    }
    if (self->data[self->robot.0 + distance * d.0][self->robot.1 + distance * d.1] == Empty()) {
      self->data[self->robot.0][self->robot.1] = Empty();
      self->data[self->robot.0 + distance * d.0][self->robot.1 + distance * d.1] = Box();
      self->data[self->robot.0 + d.0][self->robot.1 + d.1] = Robot();
      self->robot.0 += d.0;
      self->robot.1 += d.1;
    }
  }

  fn Print[self: Self]() {
    var y: i32 = 0;
    while (y < 50) {
      var x: i32 = 0;
      while (x < 50) {
        Core.PrintChar(self.data[x][y]);
        ++x;
      }
      Core.PrintChar(0xA);
      ++y;
    }
    Core.PrintChar(0xA);
  }

  fn Signature[self: Self]() -> i32 {
    var s: i32 = 0;
    var y: i32 = 0;
    while (y < 50) {
      var x: i32 = 0;
      while (x < 50) {
        if (self.data[x][y] == Box()) {
          s += y * 100 + x;
        }
        ++x;
      }
      ++y;
    }
    return s;
  }

  var data: [[i32; 50]; 50];
  var robot: (i32, i32);
}
