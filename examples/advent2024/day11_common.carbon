// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// https://adventofcode.com/2024/day/11

library "day11_common";

import Core library "io";
import library "io_utils";

fn Next(n: i64) -> (i64, i64) {
  if (n == 0) { return (1, -1); }
  var pow10: i64 = 10;
  var pow100: i64 = 1;
  while (n / pow100 >= 100) {
    pow100 = pow100 * 100;
    pow10 = pow10 * 10;
  }
  if (n / pow100 >= 10) {
    return (n / pow10, n % pow10);
  }
  return (n * 2024, -1);
}

fn Count(n: i64, depth: i32) -> i32 {
  if (n == -1) { return 0; }
  if (depth == 0) { return 1; }
  let next: (i64, i64) = Next(n);
  return Count(next.0, depth - 1) + Count(next.1, depth - 1);
}
