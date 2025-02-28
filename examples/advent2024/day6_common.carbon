// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// https://adventofcode.com/2024/day/6

library "day6_common";

import Core library "io";
import library "io_utils";

// TODO: Use a choice type.
fn Empty() -> i8 { return 0; }
fn Visited() -> i8 { return 1; }
fn Wall() -> i8 { return 2; }

class Maze {
  fn Read() -> Maze {
    returned var me: Maze;

    var y: i32 = 0;
    while (y < 130) {
      var x: i32 = 0;
      while (x < 130) {
        var cell: i32 = Core.ReadChar();
        if (cell == 0x23) {
          me.data[x][y] = Wall();
        } else if (cell == 0x5E) {
          // TODO: Handle other starting directions?
          me.data[x][y] = Visited();
          me.loc = (x, y);
          me.dir = (0, -1);
        } else {
          me.data[x][y] = Empty();
        }
        ++x;
      }
      SkipNewline();
      ++y;
    }
    return var;
  }

  fn Copy[self: Self]() -> Maze {
    returned var copy: Maze;
    var y: i32 = 0;
    while (y < 130) {
      var x: i32 = 0;
      while (x < 130) {
        copy.data[x][y] = self.data[x][y];
        ++x;
      }
      ++y;
    }
    copy.loc = self.loc;
    copy.dir = self.dir;
    return var;
  }

  fn Step[addr self: Self*]() -> bool {
    let x: i32 = self->loc.0 + self->dir.0;
    let y: i32 = self->loc.1 + self->dir.1;
    if (x < 0 or x >= 130 or y < 0 or y >= 130) {
      return false;
    }
    if (self->data[x][y] == Wall()) {
      // TODO: Should this work?
      // self->dir = (-self->dir.1, self->dir.0);
      let d: (i32, i32) = self->dir;
      self->dir = (-d.1, d.0);
    } else {
      self->loc = (x, y);
      self->data[x][y] = Visited();
    }
    return true;
  }

  fn AddObstacle[addr self: Self*]() -> bool {
    let x: i32 = self->loc.0 + self->dir.0;
    let y: i32 = self->loc.1 + self->dir.1;
    if (x < 0 or x >= 130 or y < 0 or y >= 130) {
      return false;
    }
    if (self->data[x][y] != Empty()) {
      return false;
    }
    self->data[x][y] = Wall();
    return true;
  }

  fn CountVisited[self: Self]() -> i32 {
    var total: i32 = 0;
    var y: i32 = 0;
    while (y < 130) {
      var x: i32 = 0;
      while (x < 130) {
        if (self.data[x][y] == Visited()) {
          ++total;
        }
        ++x;
      }
      ++y;
    }
    return total;
  }

  var data: array(array(i8, 130), 130);
  var loc: (i32, i32);
  var dir: (i32, i32);
}
