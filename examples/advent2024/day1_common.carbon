// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// https://adventofcode.com/2024/day/1

library "day1_common";

import library "io_utils";

// Read a sequence of lines each containing a pair of numbers into two arrays.
// Returns the number of lines read.
fn ReadInputs(ap: array(i32, 1000)*, bp: array(i32, 1000)*) -> i32 {
  var n: i32 = 0;
  var a: i32;
  var b: i32;
  while (n < 1000 and ReadInt(&a) and SkipSpaces() and ReadInt(&b) and SkipNewline()) {
    (*ap)[n] = a;
    (*bp)[n] = b;
    ++n;
  }
  return n;
}
