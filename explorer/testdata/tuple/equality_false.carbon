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
  var t1: (i32, i32) = (5, 2);
  var t2: (i32, i32) = (5, 4);
  if (t1 == t2) {
    return 1;
  } else {
    return 0;
  }
}
