// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Test that a global variable may not depend on a later global.
// Error expected.

var Int x = y;

var Int y = 0;

fn main() -> Int {
  return x;
}
