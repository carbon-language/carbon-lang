// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// https://adventofcode.com/2024/day/15

library "day15_common";

import Core library "io";
import library "io_utils";

// TODO: Use a choice type.
class Square {
  adapt char;

  impl as Core.Copy {
    fn Op[self: Self]() -> Self {
      return (self as char).(Core.Copy.Op)() as Self;
    }
  }
  impl as Core.EqWith(Square) {
    fn Equal[self: Self](other: Self) -> bool {
      // TODO: Use a `char` comparison when it's supported.
      return (self as u8) == (other as u8);
    }
    fn NotEqual[self: Self](other: Self) -> bool {
      // TODO: Use a `char` comparison when it's supported.
      return (self as u8) != (other as u8);
    }
  }

  fn Make(c: CharOrEOF) -> Square {
    // TODO: Use `match`.
    if (c == '#') { return ('#' as char) as Square; }
    if (c == 'O') { return ('O' as char) as Square; }
    if (c == '.') { return ('.' as char) as Square; }
    if (c == '@') { return ('@' as char) as Square; }
    return ('!' as char) as Square;
  }
}

let Wall: Square = ('#' as char) as Square;
let Box: Square = ('O' as char) as Square;
let Empty: Square = ('.' as char) as Square;
let Robot: Square = ('@' as char) as Square;

class Grid {
  fn Read() -> Grid {
    returned var me: Grid;
    var y: i32 = 0;
    while (y < 50) {
      var x: i32 = 0;
      while (x < 50) {
        me.data[x][y] = Square.Make(ReadChar());
        if (me.data[x][y] == Robot) {
          me.robot = (x, y);
        }
        ++x;
      }
      SkipNewline();
      ++y;
    }
    return var;
  }

  fn Move[ref self: Self](d: (i32, i32)) {
    var distance: i32 = 1;
    while (self.data[self.robot.0 + distance * d.0][self.robot.1 + distance * d.1] == Box) {
      ++distance;
    }
    if (self.data[self.robot.0 + distance * d.0][self.robot.1 + distance * d.1] == Empty) {
      self.data[self.robot.0][self.robot.1] = Empty;
      self.data[self.robot.0 + distance * d.0][self.robot.1 + distance * d.1] = Box;
      self.data[self.robot.0 + d.0][self.robot.1 + d.1] = Robot;
      self.robot.0 += d.0;
      self.robot.1 += d.1;
    }
  }

  fn Print[self: Self]() {
    var y: i32 = 0;
    while (y < 50) {
      var x: i32 = 0;
      while (x < 50) {
        Core.PrintChar(self.data[x][y] as char);
        ++x;
      }
      Core.PrintChar('\n');
      ++y;
    }
    Core.PrintChar('\n');
  }

  fn Signature[self: Self]() -> i32 {
    var s: i32 = 0;
    var y: i32 = 0;
    while (y < 50) {
      var x: i32 = 0;
      while (x < 50) {
        if (self.data[x][y] == Box) {
          s += y * 100 + x;
        }
        ++x;
      }
      ++y;
    }
    return s;
  }

  var data: array(array(Square, 50), 50);
  var robot: (i32, i32);
}

fn ReadAndExecuteActions(ref g: Grid) {
  while (true) {
    var m: CharOrEOF = ReadChar();
    // To watch the progress, uncomment the following:
    // g.Print();
    // Core.PrintChar(((m as i32) as u8) as char);
    // Core.PrintChar('\n');
    // TODO: Use `match` once it's available.
    if (m == '^') {
      g.Move((0, -1));
    } else if (m == '<') {
      g.Move((-1, 0));
    } else if (m == '>') {
      g.Move((1, 0));
    } else if (m == 'v') {
      g.Move((0, 1));
    } else if (m == CharOrEOF.EOF()) {
      return;
    }
  }
}
