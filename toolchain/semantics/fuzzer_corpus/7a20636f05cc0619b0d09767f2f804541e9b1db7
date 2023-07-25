// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// CHECK:STDOUT: result: 0

package ExplorerTest api;

fn Main() -> i32 {
  var (x: auto, y: [i32;]) = (42, (0, 1));
  var index: i32 = 1;
  y[index] = 0;
  return y[0] + y[1];
}
