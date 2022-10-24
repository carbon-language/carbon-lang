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
  var t1: (i32,) = (5,);
  var t2: (i32, i32) = (2, 3,);
  return t1[0] - t2[0] - t2[1];
}
