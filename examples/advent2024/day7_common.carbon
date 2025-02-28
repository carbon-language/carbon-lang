// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// https://adventofcode.com/2024/day/7

library "day7_common";

import Core library "io";
import library "io_utils";

fn Concat(a_val: i64, b_val: i64) -> i64 {
  var a: i64 = a_val;
  var b: i64 = b_val;
  if (b == 0) {
    return a * 10;
  }
  while (b != 0) {
    a = a * 10;
    b = b / 10;
  }
  return a + b_val;
}

class Equation {
  fn Read() -> Equation {
    returned var me: Equation;
    me.num_operands = 0;
    ReadInt64(&me.result);
    ConsumeChar(0x3A);
    while (ConsumeChar(0x20)) {
      ReadInt64(&me.operands[me.num_operands]);
      ++me.num_operands;
    }
    while (SkipNewline()) {}
    return var;
  }

  fn SolveFrom[self: Self](start: i32, value: i64, concat: bool) -> bool {
    if (value > self.result) {
      return false;
    }
    if (start == self.num_operands) {
      return value == self.result;
    }
    return self.SolveFrom(start + 1, value + self.operands[start], concat) or
           self.SolveFrom(start + 1, value * self.operands[start], concat) or
           (concat and self.SolveFrom(start + 1, Concat(value, self.operands[start]), true));
  }

  fn Solve[self: Self](concat: bool) -> bool {
    return self.SolveFrom(1, self.operands[0], concat);
  }

  var operands: array(i64, 16);
  var num_operands: i32;
  var result: i64;
}
