// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// RUN: %{explorer-run}
// RUN: %{explorer-run-trace}
// CHECK:STDOUT: result: 0

package ExplorerTest api;

// Test multiple arguments
fn f(x: i32, y: i32) -> i32 {
  return x + y;
}

fn Main() -> i32 {
  return f(2, 3) - 5;
}
