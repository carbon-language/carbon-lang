// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

package ExecutableSemanticsTest api;

// Test a global variable depending on a function.

fn f() -> i32 {
  return 0;
}

var y: i32 = f();

fn main() -> i32 {
  return y;
}
