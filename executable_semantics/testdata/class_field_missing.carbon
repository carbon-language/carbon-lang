// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

package ExecutableSemanticsTest api;

class Point {
  var x: i32;
  var y: i32;
}

fn main() -> i32 {
  return Point(.x = 1).x - 1;
}
