// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// RUN: %{explorer-run}
// RUN: %{explorer-run-trace}
// CHECK:STDOUT: result: 1

package ExplorerTest api;

// Test creating and running a continuation.

fn Main() -> i32 {
  var x: i32 = 0;
  __continuation k {
    x = x + 1;
  }
  __run k;
  return x;
}
