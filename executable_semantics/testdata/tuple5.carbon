// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Test the that field order matters for tuples.

fn main() -> Int {
  var t: (.x = Int, .y = Int) = (.y = 2, .x = 3);
  return 0;
}
