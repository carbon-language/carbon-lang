// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Test overshadowing of global variable.

var Int x = 1;

fn identity(Int x) {
  return x;
}

fn main() -> Int {
  return identity(0);
}
