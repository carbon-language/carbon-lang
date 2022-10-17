// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// RUN: %{explorer-run}
// RUN: %{explorer-run-trace}
// CHECK:STDOUT: result: 1

package ExplorerTest api;

fn IntDiff(a: i32, b: i32) -> i32 {
  if (a == b) {
    returned var ret: i32 = a - b;
    return var;
  } else {
    returned var ret2: i32 = b - a;
    return var;
  }
}

fn Main() -> i32 {
  return IntDiff(1, 2);
}
