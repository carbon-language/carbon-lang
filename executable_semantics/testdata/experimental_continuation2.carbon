// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Test creating and running a continuation.

fn main() -> Int {
  var x: Int = 0;
  __continuation k {
    x = x + 1;
  }
  __run k;
  return x;
}
