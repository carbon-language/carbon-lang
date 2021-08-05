// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Test pausing a continuation with `__await` and restarting it with
// `__run`.

fn main() -> i32 {
  var x: i32 = 0;
  __continuation k {
    x = x + 1;
    __await;
    x = x + 2;
  }
  __run k;
  __run k;
  return x;
}
