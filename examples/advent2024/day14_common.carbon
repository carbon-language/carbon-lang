// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// https://adventofcode.com/2024/day/14

library "day14_common";

import Core library "io";
import library "io_utils";

fn ReadXY(xy: (i32, i32)*) -> bool {
  return ReadInt(&xy->0) and ConsumeChar(0x2C) and ReadInt(&xy->1);
}

fn Mod(a: i32, d: i32) -> i32 {
  let rem: i32 = a % d;
  return if rem < 0 then rem + d else rem;
}

class Robot {
  fn Read() -> Robot {
    returned var me: Robot;
    // "p="
    SkipNChars(2);
    ReadXY(&me.p);
    // " v="
    SkipNChars(3);
    ReadXY(&me.v);
    SkipNewline();
    return var;
  }

  fn Pos[self: Self](t: i32) -> (i32, i32) {
    return (Mod((self.p.0 + self.v.0 * Mod(t, 101)), 101),
            Mod((self.p.1 + self.v.1 * Mod(t, 103)), 103));
  }

  var p: (i32, i32);
  var v: (i32, i32);
}
