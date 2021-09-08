// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

package ExecutableSemanticsTest api;

// Test mutation of a global variable.

var zero: i32 = 1;

fn main() -> i32 {
  zero = 0;
  return zero;
}
