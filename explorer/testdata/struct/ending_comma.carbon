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
  var t1: {.x: i32,} = {.x = 5,};
  var t2: {.x: i32, .y: i32} = {.x = 2, .y = 3,};
  return t1.x - t2.x - t2.y;
}
