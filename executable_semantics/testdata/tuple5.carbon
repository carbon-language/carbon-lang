// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Test the that field order matters for tuples.

fn main() -> i32 {
  var t: (.x = i32, .y = i32) = (.y = 2, .x = 3);
  return 0;
}
