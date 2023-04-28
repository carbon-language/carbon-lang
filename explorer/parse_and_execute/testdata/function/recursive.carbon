// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// RUN: %{explorer-run}
// RUN: %{explorer-run-trace}
// CHECK:STDOUT: result: 0

package ExplorerTest api;

fn f(x: i32) -> i32 {
  if (x == 0) {
    return x;
  } else {
    return f(x - 1);
  }
}

fn Main() -> i32 {
  return f(2);
}
