// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// https://adventofcode.com/2024/day/11

import Core library "io";
import Core library "range";

import library "day11_common";
import library "io_utils";

class Digits {
  fn Make() -> Digits {
    returned var me: Digits;
    for (digit: i32 in Core.Range(10)) {
      for (depth: i32 in Core.Range(75)) {
        me.count[digit][depth] = 0;
      }
    }
    return var;
  }

  fn Print[self: Self](max_depth: i32) {
    for (digit: i32 in Core.Range(10)) {
      Core.PrintChar(((digit + 0x30) as u8) as char);
      Core.PrintChar(':');
      for (depth: i32 in Core.Range(max_depth)) {
        Core.PrintChar(' ');
        PrintInt64NoNewline(self.count[digit][depth]);
      }
      Core.PrintChar('\n');
    }
    Core.PrintChar('\n');
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
    // PrintInt64(total);
    // digits.Print(max_depth - 1);
    SkipSpaces();
  }

  var depth: i32 = max_depth - 1;
  while (depth >= 0) {
    // PrintInt64(total);
    // digits.Print(depth);
    for (digit: i32 in Core.Range(10)) {
      let m: i64 = digits.count[digit][depth];
      if (m > 0) {
        let next: (i64, i64) = Next(digit as i64);
        total += ReduceToDigits(next.0, depth, m, &digits) +
                 ReduceToDigits(next.1, depth, m, &digits);
      }
    }
    --depth;
  }

  PrintInt64(total);
}
