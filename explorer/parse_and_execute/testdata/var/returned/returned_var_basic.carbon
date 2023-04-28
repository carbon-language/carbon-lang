// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// RUN: %{explorer-run}
// RUN: %{explorer-run-trace}
// CHECK:STDOUT: result: 3

package ExplorerTest api;

fn AddInt(a: i32, b: i32) -> i32 {
  returned var ret: i32 = a + b;
  return var;
}

fn Main() -> i32 {
  return AddInt(1, 2);
}
