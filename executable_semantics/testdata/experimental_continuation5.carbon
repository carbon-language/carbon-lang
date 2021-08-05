// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Test access to block-scoped variables upon resuming a continuation.

fn main() -> i32 {
  var y: i32 = 0;
  __continuation k {
    var x: i32 = 0;
    x = x + 1;
    __await;
    x = x + 2;
    y = x;
  }
  __run k;
  __run k;
  return y;
}
