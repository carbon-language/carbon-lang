// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Test that assignment performs a copy and does not create an alias.

fn main() -> Int {
  var Int x = -1;
  {
     var Int y = 0;
     x = y;
     // y dies here
  }
  return x;
}
