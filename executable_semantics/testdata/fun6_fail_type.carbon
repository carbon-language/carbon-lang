// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

fn f(x: i32, y: i32) -> i32 { return x + y; }

fn main() -> i32 {
  var xy: (i32, i32) = (1, 2);
  // should fail to type-check
  return f(xy);
}
