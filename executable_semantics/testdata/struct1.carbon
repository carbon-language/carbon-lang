// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

struct Point {
  var x: Int;
  var y: Int;
}

fn main() -> Int {
  var p: auto = Point(.x = 1, .y = 2);
  return p.y - p.x - 1;
}
