// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Test a global variable depending on another global.

var x: Int = 0;

var y: Int = x;

fn main() -> Int {
  return y;
}
