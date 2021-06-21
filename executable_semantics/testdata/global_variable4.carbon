// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Test mutation of a global variable.

var zero: Int = 1;

fn main() -> Int {
  zero = 0;
  return zero;
}
