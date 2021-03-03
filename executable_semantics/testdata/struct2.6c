// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

struct Point {
  var Int: x;
  var Int: y;
}

fn main() -> Int {
  var auto: p1 = Point(.x = 1, .y = 2);
  var auto: p2 = p1;
  p2.x = 3;
  return p1.x - 1;
}
