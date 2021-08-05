// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

struct Point {
  var x: i32;
  var y: i32;
}

fn main() -> i32 {
  var p1: auto = Point(.x = 1, .y = 2);
  var p2: auto = p1;
  p2.x = 3;
  return p1.x - 1;
}
