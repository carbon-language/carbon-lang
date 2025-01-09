// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// https://adventofcode.com/2024/day/11

import Core library "io";

import library "day11_common";
import library "io_utils";

fn Run() {
  var n: i64;
  var total: i32 = 0;
  while (ReadInt64(&n)) {
    total = total + Count(n, 25);
    SkipSpaces();
  }
  Core.Print(total);
}
