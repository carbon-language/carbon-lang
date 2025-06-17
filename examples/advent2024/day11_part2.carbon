// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// https://adventofcode.com/2024/day/11

import Core library "io";

import library "day11_common";
import library "io_utils";

class Digits {
  fn Make() -> Digits {
    returned var me: Digits;
    var digit: i32 = 0;
    while (digit < 10) {
      var depth: i32 = 0;
      while (depth < 75) {
        me.count[digit][depth] = 0;
        ++depth;
      }
      ++digit;
    }
    return var;
  }

  fn Print[self: Self](max_depth: i32) {
    var digit: i32 = 0;
    while (digit < 10) {
      Core.PrintChar(digit + 0x30);
      Core.PrintChar(0x3A);
      var depth: i32 = 0;
      while (depth <= max_depth) {
        Core.PrintChar(0x20);
        PrintInt64NoNewline(self.count[digit][depth]);
        ++depth;
      }
      Core.PrintChar(0x0A);
      ++digit;
    }
    Core.PrintChar(0x0A);
  }

  var count: array(array(i64, 75), 10);
}

fn ReduceToDigits(n: i64, depth: i32, multiplicity: i64, digits: Digits*) -> i64 {
  if (n == -1) { return 0; }
  if (depth == 0) { return multiplicity; }
  if (n < 10) {
    let count: i64* = &digits->count[n as i32][depth - 1];
    *count += multiplicity;
    return 0;
  }
  let next: (i64, i64) = Next(n);
  return ReduceToDigits(next.0, depth - 1, multiplicity, digits) +
         ReduceToDigits(next.1, depth - 1, multiplicity, digits);
}

fn Run() {
  let max_depth: i32 = 75;
  var total: i64 = 0;
  var digits: Digits = Digits.Make();

  var n: i64;
  while (ReadInt64(&n)) {
    total += ReduceToDigits(n, max_depth, 1, &digits);
    PrintInt64(total);
    digits.Print(max_depth - 1);
    SkipSpaces();
  }

  var depth: i32 = max_depth - 1;
  while (depth >= 0) {
    PrintInt64(total);
    digits.Print(depth);
    var digit: i64 = 0;
    while (digit < 10) {
      let m: i64 = digits.count[digit as i32][depth];
      if (m > 0) {
        let next: (i64, i64) = Next(digit);
        total += ReduceToDigits(next.0, depth, m, &digits) +
                 ReduceToDigits(next.1, depth, m, &digits);
      }
      ++digit;
    }
    --depth;
  }

  PrintInt64(total);
}
