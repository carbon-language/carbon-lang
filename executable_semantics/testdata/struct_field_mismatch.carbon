// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

struct Point {
  var x: Int;
  var y: Int;
}

fn main() -> Int {
  return Point(.x = 1, .z = 2).x - 1;
}
