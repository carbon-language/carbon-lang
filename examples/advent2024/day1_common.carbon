// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// https://adventofcode.com/2024/day/1

library "day1_common";

import Core library "range";

import library "io_utils";

// Read a sequence of lines each containing a pair of numbers into two arrays.
// Returns the number of lines read.
fn ReadInputs[N:! Core.IntLiteral](ref a: array(i32, N), ref b: array(i32, N)) -> i32 {
  for (i: i32 in Core.Range(N)) {
    var av: i32;
    var bv: i32;
    if (ReadInt(ref av) and SkipSpaces() and ReadInt(ref bv) and SkipNewline()) {
      a[i] = av;
      b[i] = bv;
    } else {
      return i;
    }
  }
  return N;
}
