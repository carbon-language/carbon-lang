// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// https://adventofcode.com/2024/day/7

import Core library "io";

import library "day7_common";
import library "io_utils";

fn Run() {
  var total: i64 = 0;
  while (PeekChar() != Core.EOF()) {
    var eq: Equation = Equation.Read();
    if (eq.Solve(true)) {
      // TODO: total += eq.result;
      total = total + eq.result;
    }
  }
  PrintInt64(total);
}
