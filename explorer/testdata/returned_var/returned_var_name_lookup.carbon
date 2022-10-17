// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// RUN: %{explorer-run}
// RUN: %{explorer-run-trace}
// CHECK:STDOUT: result: 1

package ExplorerTest api;

fn Main() -> i32 {
  returned var x: i32 = 1;
  if (true) {
    var x: i32 = 2;
    return var;
  }
  x = 3;
  return var;
}
