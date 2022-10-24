// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// RUN: %{explorer-run}
// RUN: %{explorer-run-trace}
// CHECK:STDOUT: result: 0

package ExplorerTest api;

fn Main() -> i32 {
  var t2: {.x: i32, .y: i32} = {.x = 2, .y = 5};
  t2.y = 3;
  // 3 - 2 - 1
  return t2.y - t2.x - 1;
}
