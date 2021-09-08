// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

package ExecutableSemanticsTest api;

// Test that mutations to a global variable in one function is visible
// in another function.

var flag: i32 = 1;

fn flipFlag() {
  flag = 0;
}

fn main() -> i32 {
  flipFlag();
  return flag;
}
