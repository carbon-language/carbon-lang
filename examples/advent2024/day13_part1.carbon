// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// https://adventofcode.com/2024/day/13

import Core library "io";

import library "day13_common";
import library "io_utils";

fn Run() {
  var total_cost: i64 = 0;
  while (true) {
    var m: Machine = Machine.Read();
    total_cost = total_cost + CostIfPossible(m);
    if (not SkipNewline()) {
      break;
    }
  }
  PrintInt64(total_cost);
}
