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
  var p1: auto = {.x = 1, .y = 2};
  var p2: auto = p1;
  p2.x = 3;
  return p1.x - 1;
}
