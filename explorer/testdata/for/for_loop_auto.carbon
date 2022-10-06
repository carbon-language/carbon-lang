// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// RUN: %{explorer} %s 2>&1 | %{FileCheck-strict} %s
// RUN: %{explorer-trace} %s 2>&1 | %{FileCheck-allow-unmatched} %s
// AUTOUPDATE: %{explorer} %s
// CHECK:HALLO WELT 0
// CHECK:HALLO WELT 1
// CHECK:HALLO WELT 2
// CHECK:HALLO WELT 3
// CHECK:result: 4

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
