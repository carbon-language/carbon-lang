// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// RUN: %{explorer-run}
// RUN: %{explorer-run-trace}
// CHECK:STDOUT: HALLO WELT 0
// CHECK:STDOUT: HALLO WELT 1
// CHECK:STDOUT: HALLO WELT 2
// CHECK:STDOUT: HALLO WELT 3
// CHECK:STDOUT: result: 4

package ExplorerTest api;

fn Main() -> i32 {
  var ar: [i32; 4] = (0, 1, 2, 3);
  var count: i32 = 0;
  for (x: auto in ar) {
    Print("HALLO WELT {0}", x);
    count = count + 1;
  }
  return count;
}
