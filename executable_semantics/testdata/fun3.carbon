// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Test multiple arguments
fn f(x: Int, y: Int) -> Int {
  return x + y;
}

fn main() -> Int {
  return f(2,3) - 5;
}
