// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Test that assignment performs a copy and does not create an alias.

fn main() -> i32 {
  var x: i32 = 0;
  var y: i32 = x;
  x = 1;
  return y;
}
