// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Test overshadowing of global variable.

var x: i32 = 1;

fn identity(x: i32) {
  return x;
}

fn main() -> i32 {
  return identity(0);
}
