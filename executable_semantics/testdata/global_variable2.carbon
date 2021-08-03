// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Test that mutations to a global variable in one function is visible
// in another function.

var flag: Int = 1;

fn flipFlag() -> () {
  flag = 0;
}

fn main() -> Int {
  flipFlag();
  return flag;
}
