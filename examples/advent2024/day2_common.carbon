// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// https://adventofcode.com/2024/day/2

library "day2_common";

fn Abs(n: i32) -> i32 {
  return if n < 0 then -n else n;
}

fn IsSafeDelta(from: i32, to: i32) -> bool {
  return from != to and Abs(from - to) <= 3;
}
